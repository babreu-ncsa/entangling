from qiskit import QuantumCircuit
import numpy as np
from qiskit import Aer
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.circuit import Parameter
import qiskit.circuit.library as qcl
from qiskit.quantum_info.operators import Operator

def create_graph(n_nodes, topology):
    """
    Creates a graph using networkx: https://networkx.org/

    Arguments
        - n_nodes (int): number of nodes in the graph
        - topology (srt): graph class
    Returns
        - graph: the graph object
    """
    topologies = ["linear", "star", "circular", "complete"]
    
    assert topology in topologies, "Topology not implemented. Must be linear, star, circular, or complete."
    assert n_nodes > 0, "Number of nodes must be larger than zero."
    
    G = nx.Graph()

    if topology == topologies[0]:
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            G.add_edge(i, i+1)
        return G
    elif topology == topologies[1]:
        G.add_node(0)
        for i in range(n_nodes - 1):
            G.add_edge(0, i+1)
        return G
    elif topology == topologies[2]:
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            G.add_edge(i, (i+1) % n_nodes)
        return G
    elif topology == topologies[3]:
        G = nx.complete_graph(n_nodes)
        return G
    


def create_circuit_from_graph_and_unitary(graph, operator):
    """
    Generates a circuit with control qubits and 2 copies of a Uniformly Weighted Graph State [ref]. 
    Applies *operator* to qubits connected by a *graph* edge.

    Arguments
        - graph (networkx.Graph): the graph that defines how qubits are connected
        - operator (qiskit.quantum_info.operators.Operator): 2-qubit operator

    Returns
        - qc (qiskit.QuantumCircuit): the quantum circuit
    """
    assert nx.is_empty(graph) == False, "Graph must have edges."
    assert operator.num_qubits == 2, "Only two-qubit operators are accepted."

    n_bits = graph.number_of_nodes()
    n_qubits = 3*n_bits
    qc = QuantumCircuit(n_qubits, n_bits)
    
    # Apply Hadamard gates to all qubits
    for i in range(n_qubits):
        qc.h(i)

    # Apply unitary gates based on graph edges
    for edge in graph.edges:
        qubits = [edge[0] + n_bits, edge[1] + n_bits]
        qc.unitary(operator, qubits)

        qubits = [edge[0]+2*n_bits, edge[1]+2*n_bits]
        qc.unitary(operator, qubits)

    # apply cswaps
    qc.barrier()
    for i in range(n_bits):
        qc.cswap(i, i+n_bits, i+2*n_bits)

    # swap test
    qc.barrier()
    for i in range(n_bits):
        qc.h(i)
        qc.measure(i, i)
        
    return qc


def simulate_circuit(circuit, n_shots):
    """
    Simulates the quantum circuit using Qiskit AerSimulator.

    Arguments
        - circuit (qiskit.QuantumCircuit): the circuit to simulate
        - n_shots (int): number of times to run the simulation

    Returns
        - result (qiskit.result): simulation results
    
    """
    simulator = Aer.get_backend('aer_simulator')
    job = simulator.run(circuit, shots=n_shots)
    
    return job.result()



def calculate_entanglement_from_simulation_result(result, s):
    """
    Calculates concentratable entanglement [ref] from a simulation result.

    Arguments
        - result (qiskit.result): simulation result
        - s (list(int)): list of integers with qubit indexes to be traced out

    Returns
        - ce (float): concentratable entanglement
    """
    def generate_bitstrings(bit_count):
        binary_strings = []
        def genbin(n, bs=''):
            if len(bs) == n:
                binary_strings.append(bs)
            else:
                genbin(n, bs + '0')
                genbin(n, bs + '1')
        genbin(bit_count)
        return binary_strings

    def filter_bitstrings(bitstrings, positions):
        filtered_bitstrings = []
        max_position = max(positions)
        for bitstring in bitstrings:
            if len(bitstring) > max_position and all(bitstring[pos] == '0' for pos in positions):
                filtered_bitstrings.append(bitstring)
        return filtered_bitstrings

    ce = 0.0
    n_qubits = len(list(result.get_counts().keys())[0])
    bitstrings = generate_bitstrings(n_qubits)
    fstrings = filter_bitstrings(bitstrings, s)
    for string in fstrings:
        try:
            ce += float(result.get_counts()[string])
        except:
            pass
    ce = 1 - ce / sum(result.get_counts().values())
    return ce



class ExperimentConfig():
    """
    *Gate* needs to be a parametrized gate with one parameter..
    """
    def __init__(
        self,
        name: str,
        n_qubits: int,
        graph_type: str,
        gate,
        parameter_bounds: list,
        n_parameters: int,
        traced_qubits: list,
        n_shots: int
    ):
        self.name = name
        self.num_qubits = n_qubits
        self.graph_type = graph_type
        self.gate = gate
        self.parameter_bounds = parameter_bounds
        self.num_parameters = n_parameters
        self.traced_qubits = traced_qubits
        self.num_shots = n_shots
        self.run_count = 0

    def details(self):
        print("\n\nExperiment configuration:", self.name)
        print("Number of qubits:", self.num_qubits)
        print("Graph type:", self.graph_type)
        print("Gate:", self.gate)
        print("Parameter bounds:", self.parameter_bounds)
        print("Number of parameters within bounds:", self.num_parameters)
        print("Qubit indexes to be traced out:", self.traced_qubits)
        print("Number of simulation shots:", self.num_shots)
        print("Number of times this configuration was used: ", self.run_count)

    def update_count(self):
        self.run_count += 1



class Experiment():
    def __init__(self, config):
        self.configuration = config
        self.result = None

    def run(self):
        conf = self.configuration
        graph = create_graph(conf.num_qubits, conf.graph_type)
        params = np.linspace(conf.parameter_bounds[0], conf.parameter_bounds[1], conf.num_parameters)
        self.parameters = params

        results = []
        for p in params:
            gate = conf.gate
            gate.params = [p]
            op = Operator(gate)
            qc = create_circuit_from_graph_and_unitary(graph, op)
            sim_result = simulate_circuit(qc, conf.num_shots)
            ce = calculate_entanglement_from_simulation_result(sim_result, conf.traced_qubits)
            results.append(ce)
            
        self.result = np.array(results)
        self.configuration.update_count()

    def plot_result(self):
        assert self.result is not None, "Experiment was never run."
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(self.parameters, self.result)
        plt.xlabel("Parameter values");
        plt.ylabel("Concentratable entanglement");
        plt.title("Qiskit gate name: " + self.configuration.gate.name);
        plt.show()
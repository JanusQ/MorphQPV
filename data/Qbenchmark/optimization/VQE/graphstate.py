from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GraphState


def create_circuit(n_qubits: int, degree: int = 2) -> QuantumCircuit:
    """Returns a quantum circuit implementing a graph state.

    Keyword arguments:
    n_qubits -- number of qubits of the returned quantum circuit
    degree -- number of edges per node
    """

    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q, name="graphstate")

    g = nx.random_regular_graph(degree, n_qubits)
    a = nx.convert_matrix.to_numpy_array(g)
    qc.compose(GraphState(a), inplace=True)
    qc.measure_all()

    return qc

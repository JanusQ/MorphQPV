from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


def create_circuit(n_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Fourier Transform algorithm.

    Keyword arguments:
    n_qubits -- number of qubits of the returned quantum circuit
    """

    q = QuantumRegister(n_qubits, "q")
    c = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(q, c, name="qft")
    qc.compose(QFT(n_qubits=n_qubits), inplace=True)
    qc.measure_all()

    return qc

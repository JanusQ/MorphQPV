from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


def create_circuit(n_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Fourier Transform algorithm using entangled qubits.

    Keyword arguments:
    n_qubits -- number of qubits of the returned quantum circuit
    """

    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q)
    qc.h(q[-1])
    for i in range(1, n_qubits):
        qc.cx(q[n_qubits - i], q[n_qubits - i - 1])

    qc.compose(QFT(n_qubits=n_qubits), inplace=True)

    qc.measure_all()
    qc.name = "qftentangled"

    return qc

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

class GHZ(QuantumCircuit):
    """GHZ state preparation circuit."""
    def __init__(self,n_qubits) -> None:
        self.n_qubits = n_qubits
        super().__init__(n_qubits)
        self.h(self.qubits[-1])
        for i in range(1, n_qubits):
            self.cx(self.qubits[n_qubits - i], self.qubits[n_qubits - i - 1])
        self.name = "ghz"
    def gen_circuit(self):
        return self
    

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

class GHZ(QuantumCircuit):
    """GHZ state preparation circuit."""
    def __init__(self,num_qubits) -> None:
        self.num_qubits = num_qubits
        super().__init__(num_qubits)
        self.h(self.qubits[-1])
        for i in range(1, num_qubits):
            self.cx(self.qubits[num_qubits - i], self.qubits[num_qubits - i - 1])
        self.name = "ghz"
    def gen_circuit(self):
        return self
    

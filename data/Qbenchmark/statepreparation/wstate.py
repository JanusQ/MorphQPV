from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

class WState(QuantumCircuit):
    """W state preparation circuit."""
    def __init__(self,num_qubits) -> None:
        self.num_qubits = num_qubits
        super().__init__(num_qubits)
        self.x(self.qubits[-1])
        for m in range(1, num_qubits):
            self.f_gate(num_qubits - m, num_qubits - m - 1, num_qubits, m)
        for k in reversed(range(1, num_qubits)):
            self.cx(k - 1, k)
        self.name = "wstate"

    def f_gate(self, i: int, j: int, n: int, k: int) -> None:
        theta = np.arccos(np.sqrt(1 / (n - k + 1)))
        self.ry(-theta, self.qubits[j])
        self.cz(self.qubits[i], self.qubits[j])
        self.ry(theta, self.qubits[j])

    def gen_circuit(self):
        return self
    
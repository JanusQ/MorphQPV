from qiskit.quantum_info import random_clifford
class Clifford:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    def gen_circuit(self):
        gates = random_clifford(self.n_qubits).to_circuit()
        return gates
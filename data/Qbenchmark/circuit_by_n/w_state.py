
from qiskit import QuantumCircuit
import math
class WState:
    def __init__(self,width:int) -> None:
        self.n_qubits = width
        self.qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        n = self.n_qubits
        self.qc.x(n - 1)
        for i in range(0, n - 1):
            self.F_gate( n - 1 - i, n - 2 - i, n, i + 1)

        for i in range(0, n - 1):
            self.qc.cx(n - 2 - i, n - 1 - i)


    def F_gate(self,i, j, n, k):
        theta = math.acos(math.sqrt(1. / (n - k + 1)))
        self.qc.ry(-theta, j)
        self.qc.cz(i, j)
        self.qc.ry(theta, j)

    def gen_circuit(self):
        """
        Create a circuit implementing the w_state algorithm

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq with nq ClassicalRegister and
            no measurements
        """
        return self.qc

    

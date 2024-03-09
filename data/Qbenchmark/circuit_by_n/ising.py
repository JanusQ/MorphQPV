
from qiskit import QuantumCircuit
import random

class ising:
    def __init__(self,width:int,seed = 555) -> None:
        random.seed(seed)
        self.n_qubits = width
        self.Bx = 2.0
        self.total_T = 15.0
        self.M = 1
        self.Bz = [random.uniform(-2.0, 2.0) for i in range(0, self.n_qubits)]
        self.J = [random.uniform(-2.0, 2.0) for i in range(0, self.n_qubits)]
        self.qc = QuantumCircuit( self.n_qubits)
        self.initialize()
        for i in range(1, self.M + 1, self.n_qubits):
            self.red_hamiltonian(i)
            self.black_hamiltonian(i)
            self.Bz_hamiltonian(i)

    def CZ(self, q1, q2, phi):
        self.qc.rz(0.5 * phi, q2)
        self.qc.cx(q1, q2)
        self.qc.rz(-0.5 * phi, q2)
        self.qc.cx(q1, q2)
    def ZcrossZ(self, q1, q2, phi):
        self.qc.rz(phi, q1)
        self.qc.rz(-phi, q2)
        self.CZ(q1, q2, -2.0 * phi)


    def initialize(self):
        for i in range(0, self.n_qubits):
            self.qc.h(i)


    def red_hamiltonian(self, m):
        for i in range(0, self.n_qubits - 1, 2):
            phi = self.J[i] * (2.0 * m - 1) / self.M
            self.ZcrossZ( i, i + 1, phi)


    def black_hamiltonian(self, m):
        for i in range(1, self.n_qubits - 1, 2):
            phi = self.J[i] * (2.0 * m - 1) / self.M
            self.ZcrossZ( i, i + 1, phi)


    def Bz_hamiltonian(self, m):
        for i in range(0, self.n_qubits):
            theta1 = (1.0 - (2.0 * m - 1.0) / self.M) * -2.0 * self.Bx * self.total_T / self.M
            theta2 = (1.0 - (2.0 * m - 1.0) / self.M) * -2.0 * self.Bz[i] * self.total_T / self.M
            self.qc.h(i)
            self.qc.rz(theta1, i)
            self.qc.h(i)
            self.qc.rz(theta2, i)

    def gen_circuit(self):
        return self.qc

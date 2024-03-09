from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import random
import math
class Parrot:
# instance attributes
    def __init__(self, width,seed= 555):
        self.n_qubits = width
        random.seed(seed)
        self.qr = QuantumRegister(self.n_qubits)
        self.qc = QuantumCircuit(self.qr)
    def init(self):
        for i in range(1, len(self.qr) // 2 + 1):
            ro = random.uniform(-2.0, 2.0) * math.pi
            self.qc.rx(ro + random.random(), self.qr[i])
            self.qc.rx(ro + random.random(), self.qr[i + len(self.qr) // 2])
    def swapTest(self):
        self.qc.h(self.qr[0])
        n_swap = len(self.qr) // 2
        for i in range(n_swap):
            self.qc.cswap(self.qr[0], self.qr[i + 1], self.qr[i + n_swap + 1])
        self.qc.h(self.qr[0])

    def gen_circuit(self):
        self.init()
        self.swapTest()
        return self.qc
        
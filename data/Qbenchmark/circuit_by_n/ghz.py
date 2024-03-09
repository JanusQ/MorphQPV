from typing import Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister

class GHZ:
    def __init__(self,width):
        self.width =  width
        self.qc = QuantumCircuit(self.width)
        self.qc.h(0)
        for i in range(self.width - 1):
            self.qc.cx(i, i + 1)
    
    def gen_circuit(self):
        return self.qc
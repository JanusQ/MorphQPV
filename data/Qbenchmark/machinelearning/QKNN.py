
from qiskit.circuit import QuantumCircuit,ClassicalRegister,QuantumRegister
import numpy as np

class QKNN:
    def __init__(self,width):
        self.qubit_count = width
        self.quantum_register =QuantumRegister(self.qubit_count)
        self.classical_register =ClassicalRegister(1)
        self.circuit =QuantumCircuit(self.quantum_register)
        self.data_load()
        self.swap_test()
        # self.circuit.measure(self.quantum_register[0],self.classical_register[0])


    def data_load(self):
        for qubit in list(self.quantum_register)[1:]:
            self.circuit.ry(np.random.rand()*np.pi,qubit)

    def swap_test(self):
        self.circuit.h(0)
        qubit_list = list(self.quantum_register)[1:]
        q1 = qubit_list[:len(qubit_list)//2]
        q2 = qubit_list[len(qubit_list)//2:]
        for q_1,q_2 in zip(q1,q2):
            self.circuit.cswap(self.quantum_register[0],q_1,q_2)
        self.circuit.h(0)
    def gen_circuit(self):
        return self.circuit
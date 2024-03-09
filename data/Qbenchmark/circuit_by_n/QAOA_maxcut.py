import random
import numpy as np
from qiskit import QuantumCircuit

class QAOA_maxcut:
    def __init__(self,width) -> None:
        self.n = width
        None
    def initialize_qaoa(self,V, E):
        self.qc= QuantumCircuit(len(V), len(V))

        self.qc.h(range(len(V)))
        

    def apply_cost_hamiltonian(self, V, E, gamma):
        for k, l, weight in E:
            self.qc.cp(-2*gamma*weight, k, l)
            self.qc.p(gamma*weight, k)
            self.qc.p(gamma*weight, l)
        


    def apply_mixing_hamiltonian(self, V, E, beta):
        self.qc.rx(2*beta, range(len(V)))
        
        return self.qc


    def construct_full_qaoa(self,p, gammas, betas, V, E):
        self.initialize_qaoa(V, E)
        for i in range(p):
            self.apply_cost_hamiltonian(V, E, gammas[i])
            self.apply_mixing_hamiltonian(V, E, betas[i])

    def gen_circuit(self):
        E = []
        for _ in range(random.randint(5, 20)):
            sample = random.sample(range(0, self.n), 2)
            E.append((sample[0], sample[1], random.random()))
        self.construct_full_qaoa(1, [.4], [.8], range(self.n), E)
        return self.qc


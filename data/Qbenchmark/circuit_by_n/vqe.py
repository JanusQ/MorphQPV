import random
from qiskit.circuit.library import EfficientSU2
from qiskit.compiler import transpile
import numpy as np
import qiskit


class VQE:
    def __init__(self, width):
        self.PAULI_MATRICES = {'Z': np.array([[1, 0], [0, -1]]),
                               'X': np.array([[0, 1], [1, 0]]),
                               'Y': np.array([[0, 0 - 1j], [0 + 1j, 0]]),
                               'I': np.array([[1, 0], [0, 1]])}
        self.k = width
        self.circuit = None
        self.hamiltonian = np.zeros((2 ** self.k, 2 ** self.k))
        self.generate_random_hamiltonian_matrix()
        self.generate_trainable_circuit()

    def generate_random_hamiltonian_matrix(self):
        weights = np.random.randint(10, size=10)
        for weight in weights:
            new_matrix = 1
            for i in range(self.k):
                new_matrix = np.kron(new_matrix, self.PAULI_MATRICES[self.z_or_i()])
            self.hamiltonian += new_matrix * weight * 0.5

    def z_or_i(self):
        p = 0.5
        if random.random() > p:
            return "Z"
        else:
            return "I"

    def generate_trainable_circuit(self):
        self.circuit = EfficientSU2(n_qubits=self.k, entanglement='linear')
        self.circuit = transpile(self.circuit, basis_gates=['cx', 'rz', 'sx', 'id', 'x'])
        n_param = self.circuit.n_parameters
        # print(n_param)
        self.circuit = self.circuit.assign_parameters(np.random.rand(n_param) * np.pi)
    def gen_circuit(self):
        """
        Create a circuit implementing the vqe algorithm

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq with nq ClassicalRegister and
            no measurements
        """
        return self.circuit

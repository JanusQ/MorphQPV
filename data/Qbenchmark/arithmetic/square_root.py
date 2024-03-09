from qiskit import QuantumCircuit
from math import pi
import math

class SquareRoot(QuantumCircuit):
    def __init__(self,n_qubits) -> None:
        super().__init__(n_qubits)
        self.initialize()
    
    def initialize(self):
        n = self.num_qubits // 3
        N = 2 ** n
        
        nstep = math.floor((pi / 4) * math.sqrt(N))

        for i in range(0, n):
            self.h(i)
        for _ in range(1, nstep + 1):
            self.Sqr(n)
            self.EQxMark(0, n)
            self.Sqr(n)
            self.diffuse(n)
        self.Sqr(n)
        self.EQxMark(1,n)
        return self
    def gen_circuit(self):
        return self
    def _toffoli(self, x, y, z):
        """
        Implement the toffoli gate using 1 and 2 qubit gates
        """
        self.h(z)
        self.cx(y, z)
        self.tdg(z)
        self.cx(x, z)
        self.t(z)
        self.cx(y, z)
        self.t(y)
        self.tdg(z)
        self.cx(x, z)
        self.cx(x, y)
        self.t(z)
        self.h(z)
        self.t(x)
        self.tdg(y)
        self.cx(x, y)
            
    def diffuse(self, n):
        for j in range(0, n):
            self.h(j)
        for j in range(0, n):
            self.x(j)
        for j in range(0, n - 1):
            self.reset(2 * n + 1 + j)
        self._toffoli(self,1, 0, 2 * n + 1)
        for j in range(1, n - 1):
            self._toffoli(self,2 * n + 1 + j - 1, j + 1, 2 * n + 1 + j)
        self.z(2 * n + 1 + n - 2)
        for j in range(n - 2, 0, -1):
            self._toffoli(self,2 * n + 1 + j - 1, j + 1, 2 * n + 1 + j)
        self._toffoli(self,1, 0, 2 * n + 1)
        for j in range(0, n):
            self.x(j)
        for j in range(0, n):
            self.h(j)


    def EQxMark(self,tF, n):
        for j in range(0, n):
            if j != 1:
                self.x(n + j)
        for j in range(0, n - 1):
            self.reset(2 * n + 1 + j)
        self._toffoli(self,n + 1, n, 2 * n + 1)
        for j in range(1, n - 1):
            self._toffoli(self,2 * n + 1 + j - 1, n + j + 1, 2 * n + 1 + j)
        if tF != 0:
            self.cx(2 * n + 1 + n - 2, 2 * n)
        else:
            self.z(2 * n + 1 + n - 2)
        for j in range(n - 2, 0, -1):
            self._toffoli(self,2 * n + 1 + j - 1, n + j + 1, 2 * n + 1 + j)
        self._toffoli(self,n + 1, n, 2 * n + 1)
        for j in range(0, n):
            if j != 1:
                self.x(n + j)


    def Sqr(self, n):
        for i in range(0, ((n - 1) // 2) + 1):
            k = i * 2
            self.cx(i, n + k)
        for i in range(((n + 1) // 2), n):
            k = 2 * i - n
            self.cx(i, n + k)
            self.cx(i, n + k + 1)
        
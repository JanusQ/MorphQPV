from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import sys
import random
import math
def _toffoli(qc, x, y, z):
        """
        Implement the toffoli gate using 1 and 2 qubit gates
        """
        qc.h(z)
        qc.cx(y, z)
        qc.tdg(z)
        qc.cx(x, z)
        qc.t(z)
        qc.cx(y, z)
        qc.t(y)
        qc.tdg(z)
        qc.cx(x, z)
        qc.cx(x, y)
        qc.t(z)
        qc.h(z)
        qc.t(x)
        qc.tdg(y)
        qc.cx(x, y)

def carry(qc, c0, a, b, c1):
    _toffoli(qc,a, b, c1)
    qc.cx(a, b)
    _toffoli(qc,c0, b, c1)


def uncarry(qc, c0, a, b, c1):
    _toffoli(qc,c0, b, c1)
    qc.cx(a, b)
    _toffoli(qc,a, b, c1)


def carry_sum(qc, c0, a, b):
    qc.cx(a, b)
    qc.cx(c0, b)


def adder(qc, qubits):
    n = int(len(qubits) / 3)
    c = qubits[0::3]
    a = qubits[1::3]
    b = qubits[2::3]
    for i in range(0, n - 1):
        carry(qc, c[i], a[i], b[i], c[i + 1])
    carry_sum(qc, c[n - 1], a[n - 1], b[n - 1])
    for i in range(n - 2, -1, -1):
        uncarry(qc, c[i], a[i], b[i], c[i + 1])
        carry_sum(qc, c[i], a[i], b[i])


def multiplier(qc, qubits):
    n = int(len(qubits) / 5)
    a = qubits[1:n * 3:3]
    y = qubits[n * 3:n * 4]
    x = qubits[n * 4:]

    for i, x_i in enumerate(x):
        for a_qubit, y_qubit in zip(a[i:], y[:n - i]):
            _toffoli(qc,x_i, y_qubit, a_qubit)
        adder(qc, qubits[:3 * n])
        for a_qubit, y_qubit in zip(a[i:], y[:n - i]):
            _toffoli(qc,x_i, y_qubit, a_qubit)


def init_bits(qc, x_bin, *qubits):
    for x, qubit in zip(x_bin, list(qubits)[::-1]):
        if x == '1':
            qc.x(qubit)

class Multiplier:
    def __init__(self,width:int) -> None:
        self.width = 5 * (width//5)
        self.n = width//5
    def gen_circuit(self):
        n_qubits = 5 * self.n
        random.seed(555)
        qr = QuantumRegister(n_qubits)
        qc = QuantumCircuit(qr)

        maxv = math.floor(math.sqrt(2 ** (self.n)))
        p = random.randint(1, maxv)
        q = random.randint(1, maxv)

        y_bin = '{:08b}'.format(p)[-self.n:]
        x_bin = '{:08b}'.format(q)[-self.n:]

        b = qr[2:self.n * 3:3]
        y = qr[self.n * 3:self.n * 4]
        x = qr[self.n * 4:]

        init_bits(qc, x_bin, *x)
        init_bits(qc, y_bin, *y)
        multiplier(qc, qr)
        return qc


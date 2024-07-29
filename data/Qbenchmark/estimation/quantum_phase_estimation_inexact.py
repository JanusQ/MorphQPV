from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


def create_circuit(n_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Phase Estimation algorithm for a phase which cannot be
    exactly estimated.

    Keyword arguments:
    n_qubits -- number of qubits of the returned quantum circuit
    """

    n_qubits = n_qubits - 1  # because of ancilla qubit
    q = QuantumRegister(n_qubits, "q")
    psi = QuantumRegister(1, "psi")
    c = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(q, psi, c, name="qpeinexact")

    # get random n+1-bit string as target phase
    random.seed(10)
    theta = 0
    while theta == 0 or (theta & 1) == 0:
        theta = random.getrandbits(n_qubits + 1)
    lam = Fraction(0, 1)
    # print("theta : ", theta, "correspond to", theta / (1 << (n+1)), "bin: ")
    for i in range(n_qubits + 1):
        if theta & (1 << (n_qubits - i)):
            lam += Fraction(1, (1 << i))

    qc.x(psi)
    qc.h(q)

    for i in range(n_qubits):
        angle = (lam * (1 << i)) % 2
        if angle > 1:
            angle -= 2
        if angle != 0:
            qc.cp(angle * np.pi, psi, q[i])

    qc.compose(
        QFT(n_qubits=n_qubits, inverse=True),
        inplace=True,
        qubits=list(range(n_qubits)),
    )
    qc.barrier()
    qc.measure(q, c)

    return qc

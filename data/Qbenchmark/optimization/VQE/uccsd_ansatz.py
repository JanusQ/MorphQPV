"""
Teague Tomesh - 3/25/2019

Implementation of the UCCSD ansatz for use in the VQE algorithm.

Based on the description given in Whitfield et al.
(https://arxiv.org/abs/1001.3855?context=physics.chem-ph)

Adapted from a Scaffold implementation by Pranav Gokhale]
(https://github.com/epiqc/ScaffCC) 

NOTE:
Qiskit orders their circuits increasing from top -> bottom
  0 ---
  1 ---
  2 ---

Both Whitfield et al. and Barkoutsos et al. order increasing from bottom -> top
  p 3 ---
  q 2 ---
  r 1 ---
  s 0 ---

Not a problem. Qubit index is what matters. Set reverse_bits = True when 
drawing Qiskit circuit.

"""

from qiskit import QuantumCircuit, QuantumRegister
import sys
import math
import numpy as np


class UCCSD(QuantumCircuit):
    """
    Class to implement the UCCSD ansatz as described in Whitfield et al.
    (https://arxiv.org/abs/1001.3855?context=physics.chem-ph)

    A UCCSD circuit can be generated for a given instance of the UCCSD class
    by calling the gen_circuit() method.

    Attributes
    ----------
    width : int
        number of qubits
    parameters : str
        choice of parameters [random, seeded]
    seed : int
        a number to seed the number generator with
    barriers : bool
        should barriers be included in the generated circuit
    regname : str
        optional string to name the quantum and classical registers. This
        allows for the easy concatenation of multiple QuantumCircuits.
    qr : QuantumRegister
        Qiskit QuantumRegister holding all of the quantum bits
    circ : QuantumCircuit
        Qiskit QuantumCircuit that represents the uccsd circuit
    """

    def __init__(
        self, width, parameters="random", seed=None, barriers=False, regname=None
    ):

        # number of qubits
        self.n_qubits = width

        # set flags for circuit generation
        self.parameters = parameters
        self.seed = seed
        self.barriers = barriers
        super().__init__(width,name=regname)

    def M_d(self, i, p, q, r, s, dagger=False):
        """
        See Double Excitation Operator circuit in Table A1
        of Whitfield et al 2010

        Y in Table A1 of Whitfield et al 2010 really means Rx(-pi/2)
        """

        if dagger:
            angle = math.pi / 2
        else:
            angle = -math.pi / 2

        qr = self.qregs[0]

        if i == 1:
            self.h(qr[p])
            self.h(qr[q])
            self.h(qr[r])
            self.h(qr[s])
        elif i == 2:
            self.rx(angle, qr[p])
            self.rx(angle, qr[q])
            self.rx(angle, qr[r])
            self.rx(angle, qr[s])
        elif i == 3:
            self.h(qr[p])
            self.rx(angle, qr[q])
            self.h(qr[r])
            self.rx(angle, qr[s])
        elif i == 4:
            self.rx(angle, qr[p])
            self.h(qr[q])
            self.rx(angle, qr[r])
            self.h(qr[s])
        elif i == 5:
            self.rx(angle, qr[p])
            self.rx(angle, qr[q])
            self.h(qr[r])
            self.h(qr[s])
        elif i == 6:
            self.h(qr[p])
            self.h(qr[q])
            self.rx(angle, qr[r])
            self.rx(angle, qr[s])
        elif i == 7:
            self.rx(angle, qr[p])
            self.h(qr[q])
            self.h(qr[r])
            self.rx(angle, qr[s])
        elif i == 8:
            self.h(qr[p])
            self.rx(angle, qr[q])
            self.rx(angle, qr[r])
            self.h(qr[s])

    def CNOTLadder(self, controlStartIndex, controlStopIndex):
        """
        Applies a ladder of CNOTs, as in the dashed-CNOT notation at bottom of
        Table A1 of Whitfield et al 2010

        Qubit indices increase from bottom to top
        """

        qr = self.qregs[0]

        if controlStopIndex > controlStartIndex:
            delta = 1
            index = controlStartIndex + 1
            controlStopIndex += 1
        else:
            delta = -1
            index = controlStartIndex

        while index is not controlStopIndex:
            self.cx(qr[index], qr[index - 1])
            index += delta

    def DoubleExcitationOperator(self, theta, p, q, r, s):
        """
        Prerequisite: p > q > r > s
        """

        qr = self.qregs[0]

        for i in range(1, 9):

            if self.barriers:
                self.barrier(qr)

            self.M_d(i, p, q, r, s, dagger=False)

            if self.barriers:
                self.barrier(qr)

            self.CNOTLadder(p, q)
            self.cx(qr[q], qr[r])
            self.CNOTLadder(r, s)

            self.rz(theta, qr[s])  # Rz(reg[s], Theta_p_q_r_s[p][q][r][s]);

            self.CNOTLadder(s, r)
            self.cx(qr[q], qr[r])
            self.CNOTLadder(q, p)

            if self.barriers:
                self.barrier(qr)

            self.M_d(i, p, q, r, s, dagger=True)

    def SingleExcitationOperator(self, theta, p, q):
        """
        Prerequisite: p > q
        See Single Excitation Operator circuit in Table A1 of
        Whitfield et al 2010
        """

        qr = self.qregs[0]

        if self.barriers:
            self.barrier(qr)

        self.h(qr[p])
        self.h(qr[q])

        self.CNOTLadder(p, q)
        self.rz(theta, qr[q])  # Rz(reg[q], Theta_p_q[p][q]);
        self.CNOTLadder(q, p)

        if self.barriers:
            self.barrier(qr)

        self.h(qr[p])
        self.h(qr[q])

        self.rx(-math.pi / 2, qr[p])
        self.rx(-math.pi / 2, qr[q])
        self.CNOTLadder(p, q)
        self.rz(theta, qr[q])  # Rz(reg[q], Theta_p_q[p][q]);
        self.CNOTLadder(q, p)

        if self.barriers:
            self.barrier(qr)

        self.rx(-math.pi / 2, qr[p])
        self.rx(-math.pi / 2, qr[q])

    def gen_circuit(self):
        """
        Create a circuit implementing the UCCSD ansatz

        Given the number of qubits and parameters, construct the
        ansatz as given in Whitfield et al.

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq with no ClassicalRegister and
            no measurements
        """
        # Ensure the # of parameters matches what is neede by the current value of
        # Nq, then set a counter, p_i=0, when the circuit is first initialized, and
        # every call to the single or double operator will take param[p_i] as its
        # parameter and then increment the value of p_i

        n_dbl = (
            self.nq**4 - 6 * self.nq**3 + 11 * self.nq**2 - 6 * self.nq
        ) / 24
        n_sgl = (self.nq**2 - self.nq) / 2
        numparam = int(n_dbl + n_sgl)

        if self.parameters == "random":

            param = np.random.uniform(-np.pi, np.pi, numparam)

        elif self.parameters == "seeded":
            if self.seed is None:
                raise Exception("A valid seed must be provided")
            else:
                np.random.seed(self.seed)

            param = np.random.uniform(-np.pi, np.pi, numparam)

        else:
            raise Exception("Unknown parameter option")

        p_i = 0

        # enumerate all Nq > p > q > r > s >= 0 and apply Double Excitation Operator
        for p in range(self.nq):
            for q in range(p):
                for r in range(q):
                    for s in range(r):
                        # print(p,q,r,s)
                        # For the 4 qubit case this function is called a single time
                        self.DoubleExcitationOperator(param[p_i], p, q, r, s)
                        p_i += 1

        # enumerate all Nq > p > q >= 0 and apply Single Excitation Operator
        for p in range(self.nq):
            for q in range(p):
                # print(p,q)
                self.SingleExcitationOperator(param[p_i], p, q)
                p_i += 1

        return self

from __future__ import annotations

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms import Grover as qiskitGrover
from qiskit.circuit.library import GroverOperator

class Grover:
    """Grover's algorithm."""

    def __init__(self,n_qubits) -> None:
        self.n_qubits = n_qubits

    def gen_circuit(self):
        return create_circuit(self.n_qubits)

def create_circuit(n_qubits: int, ancillary_mode: str = "noancilla") -> QuantumCircuit:
    """Returns a quantum circuit implementing Grover's algorithm.

    Keyword arguments:
    n_qubits -- number of qubits of the returned quantum circuit
    ancillary_mode -- defining the decomposition scheme
    """

    n_qubits = n_qubits - 1  # -1 because of the flag qubit
    q = QuantumRegister(n_qubits, "q")
    flag = AncillaRegister(1, "flag")

    state_preparation = QuantumCircuit(q, flag)
    state_preparation.h(q)
    state_preparation.x(flag)

    oracle = QuantumCircuit(q, flag)
    oracle.mcp(np.pi, q, flag)

    operator = GroverOperator(oracle, mcx_mode=ancillary_mode)
    iterations = qiskitGrover.optimal_n_iterations(1, n_qubits)

    n_qubits = operator.n_qubits - 1  # -1 because last qubit is "flag" qubit and already taken care of

    # n_qubits may differ now depending on the mcx_mode
    q2 = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q2, flag, name="grover")
    qc.compose(state_preparation, inplace=True)

    qc.compose(operator.power(iterations), inplace=True)
    qc.measure_all()
    qc.name = qc.name + "-" + ancillary_mode

    return qc

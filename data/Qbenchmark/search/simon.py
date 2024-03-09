from qiskit import QuantumCircuit


class Simon:
    def __init__(self, bitstring: str):
        self.bitstring = bitstring
        self.circuit = gen_circuit(bitstring)

    def gen_circuit(self):
        return self.circuit
    
def simon_oracle(b):
    """returns a Simon oracle for bitstring b"""
    b = b[::-1] # reverse b for easy iteration
    n = len(b)
    qc = QuantumCircuit(n*2)
    # Do copy; |x>|0> -> |x>|x>
    for q in range(n):
        qc.cx(q, q+n)
    if '1' not in b:
        return qc  # 1:1 mapping, so just exit
    i = b.find('1') # index of first non-zero bit in b
    # Do |x> -> |s.x> on condition that q_i is 1
    for q in range(n):
        if b[q] == '1':
            qc.cx(i, (q)+n)
    return qc


def gen_circuit(bitstring: str) -> QuantumCircuit:

    n = len(bitstring)
    simon_circuit = QuantumCircuit(n * 2)

    # Apply Hadamard gates before querying the oracle
    simon_circuit.h(range(n))

    simon_circuit.compose(simon_oracle(bitstring))

    # Apply Hadamard gates to the input register
    simon_circuit.h(range(n))

    return simon_circuit

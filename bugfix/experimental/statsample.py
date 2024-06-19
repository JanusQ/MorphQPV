from qiskit.quantum_info import random_clifford,Statevector
from itertools import product
from morphQPV.execute_engine.excute import ExcuteEngine,qiskit_circuit_to_layer_cirucit
def sampling_head(n_qubits,method,basenum):
    if method == 'basis':
        for state in product(['0','1'],repeat=n_qubits):
            yield [{'name':'x','qubits':[i]} for i in range(n_qubits) if state[i] == '1']
    elif method == 'clifford':
        for _ in range(basenum):
            clifford = random_clifford(n_qubits)
            yield qiskit_circuit_to_layer_cirucit(clifford.to_circuit())
    
def get_statistical_sample(process,output_qubits=None):
    return ExcuteEngine.excute_on_qiskit(process,type ='prob',shots=10000,output_qubits=output_qubits)

def pauli_gates(qubits):
    paulis = ['I','X','Y','Z']
    num_qubits = len(qubits)
    pauli_basis_gates = []
    for pauli in product(paulis, repeat=num_qubits):
        gate = []
        for i in range(num_qubits):
            if pauli[i] == 'I':
                continue
            gate.append({
                "name": pauli[i],
                "qubits": qubits[i],
            })
        pauli_basis_gates.append(gate)
    return pauli_basis_gates




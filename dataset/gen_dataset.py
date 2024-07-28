from bugfix.clliford.utills import custom_random_circuit
from bugfix.paramizefix.qiskit import  generate_bugged_circuit 
from qiskit import transpile
import pickle

num_qubits = 20
qc_correct, qc_error = [],[]
for i in range(50):
    
    correct_circuit = custom_random_circuit(num_qubits,5,gate_set = ['h','cx','cz','x','y','z','rx','ry','rz'])
    # correct_circuit = random_circuit(num_qubits, depth=5, max_operands=2)
    bugged_circuit = generate_bugged_circuit(correct_circuit.copy(), error_rate=0.3)
    qc_correct.append(correct_circuit)
    qc_error.append(bugged_circuit)

qcs = qc_correct + qc_error

qcs_trans = transpile(qcs, basis_gates=['h','u3','cx'], optimization_level=1)

with open(f'dataset/{num_qubits}/qcs.pkl','wb') as f:
    pickle.dump(qcs_trans,f)


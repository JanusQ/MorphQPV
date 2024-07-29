from random import randint
from tqdm import tqdm
import jax
import optax
from jax import numpy as jnp
import pennylane as qml
from bugfix.circuit import Circuit, circuit_to_pennylane
from bugfix.clliford.clliford_gate_variables import CllifordCorrecter
from bugfix.clliford.utills import CllifordProgram, custom_random_circuit, generate_inout_stabilizer_tables
from bugfix.paramizefix.pennylane_siwei import generate_input_states as generate_input_states_pennylane, GateParameterOptimizer
from bugfix.paramizefix.qiskit import apply_circuit, generate_bugged_circuit, generate_input_states, optimize_parameters, replace_param_gates_with_clifford

'''试试直接梯度下降'''

n_qubits = 3
correct_circuit = custom_random_circuit(
    n_qubits, 5, gate_set=['h', 'cx', 'cz', 'rx', 'ry', 'rz']) # 'x', 'y', 'z', 

bugged_circuit = generate_bugged_circuit(
    correct_circuit.copy(), error_rate=0.3)
print("correct_circuit:")
print(correct_circuit)
print("Bugged Circuit:")
print(bugged_circuit)

correct_circuit = Circuit(correct_circuit)

optimizer = GateParameterOptimizer.from_circuit(bugged_circuit)
optimizer.optimize_target_unitary(correct_circuit.matrix(), n_epochs=1000)

input_states = generate_input_states_pennylane(n_qubits, n_states=10)
target_output_states = []
for input_state in input_states:
    target_output_states.append(correct_circuit.run(input_state))
optimizer.optimize_target_output_states(input_states, target_output_states, n_epochs=1000, n_batch = 20)




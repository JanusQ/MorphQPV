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


n_qubits = 3
correct_circuit = custom_random_circuit(
    n_qubits, 5, gate_set=['h', 'cx', 'cz', 'rx', 'ry', 'rz']) # 'x', 'y', 'z', 


bugged_circuit = generate_bugged_circuit(
    correct_circuit.copy(), error_rate=0.3)
print("correct_circuit:")
print(correct_circuit)
print("Bugged Circuit:")
print(bugged_circuit)

# TODO: 改成找最近的
correct_clliford = replace_param_gates_with_clifford(correct_circuit)  
bugged_clifford = replace_param_gates_with_clifford(bugged_circuit)
print("correct_clliford:")
print(correct_clliford)
print("bugged_clifford:")
print(bugged_clifford)

program = CllifordProgram.from_circuit(bugged_clifford, n_qubits=n_qubits)
correcter = CllifordCorrecter(program.n_qubits, len(program))
inputs, outputs = [], []
for _ in range(1):
    input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(
        program.n_qubits, program)
    inputs.append(input_stabilizer_table)
    outputs.append(output_stabilizer_table)
    correcter.add_iout(input_stabilizer_table, output_stabilizer_table)
find_program = correcter.solve(100)

# 找不到就用旧的
if find_program is None:
    find_program = program

for input_stabilizer_table, output_stabilizer_table in zip(inputs, outputs):
    predict_out = find_program.output_stablizers(input_stabilizer_table)
    assert predict_out.is_eq(output_stabilizer_table)

print('find program\n', find_program.to_circuit())

input_states = generate_input_states_pennylane(n_qubits, n_states=10)

# dev = qml.device("default.qubit", wires=n_qubits)
# cir = Circuit(correct_circuit)
# @jax.jit
# @qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
# def run(input_state):
#     qml.QubitStateVector(input_state, wires=range(n_qubits))
#     circuit_to_pennylane(cir)
#     return qml.probs()

target_output_states = []
correct_circuit = Circuit(correct_circuit)
for input_state in input_states:
    target_output_states.append(correct_circuit.run(input_state))
    
optimizer = GateParameterOptimizer.from_clifford_program(find_program)
optimizer.optimize_target_output_states(input_states, target_output_states, n_epochs=2000, n_batch = 20)



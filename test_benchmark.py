import jax
import optax
from jax import numpy as jnp
import pennylane as qml
from bugfix.clliford.clliford_gate_variables import CllifordCorrecter
from bugfix.clliford.utills import CllifordProgram, custom_random_circuit, generate_inout_stabilizer_tables
from bugfix.paramizefix.pennylane import CircuitTape, update_step
from bugfix.paramizefix.qiskit import apply_circuit, generate_bugged_circuit, generate_input_states, optimize_parameters, replace_param_gates_with_clifford


n_qubits = 3
correct_circuit = custom_random_circuit(n_qubits,20,gate_set = ['h','cx','cz','x','y','z','rx','ry','rz'])
bugged_circuit = generate_bugged_circuit(correct_circuit.copy(), error_rate=0.3)
print("correct_circuit:")
print(correct_circuit)
print("Bugged Circuit:")
print(bugged_circuit)

correct_clliford = replace_param_gates_with_clifford(correct_circuit)
bugged_clifford = replace_param_gates_with_clifford(bugged_circuit)
print("correct_clliford:")
print(correct_clliford)
print("bugged_clifford:")
print(bugged_clifford)

program = CllifordProgram.from_circuit(bugged_clifford)
correcter = CllifordCorrecter(program.n_qubits,len(program))
inputs, outputs = [],[]
for _ in range(1):
    input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(program.n_qubits,program)
    inputs.append(input_stabilizer_table)
    outputs.append(output_stabilizer_table)
    correcter.add_iout(input_stabilizer_table,output_stabilizer_table)
find_program = correcter.solve()

if find_program is None:
    find_program = program
for input_stabilizer_table, output_stabilizer_table in zip(inputs,outputs):
    predict_out = find_program.output_stablizers(input_stabilizer_table)
    assert predict_out.is_eq(output_stabilizer_table)
print('find program\n',find_program.to_circuit())


circuit_tape = CircuitTape.from_clifford_program(find_program)
n_wires = find_program.n_qubits
dev = qml.device("default.qubit", wires=n_wires)


loss_history = []
from tqdm import tqdm
from random import randint
n_params = circuit_tape.get_n_params()
opt = optax.adam(learning_rate=0.1)
params = jax.random.uniform(jax.random.PRNGKey(randint(0, 200)), (n_params,), minval = 0, maxval= jnp.pi * 2)
# params = jnp.array(circuit_tape.gate_params())

print("n_params:", n_params)
opt_state = opt.init(params)

with tqdm(total=50) as pbar:
    for i in range(50):
        params, opt_state, loss_val = update_step(opt, params, opt_state, n_wires)
        pbar.update(1)
        loss_val = round(float(loss_val), 7)
        pbar.set_description(f"Loss: {loss_val}")
        loss_history.append(loss_val)
        
print(params)
print(loss_history)
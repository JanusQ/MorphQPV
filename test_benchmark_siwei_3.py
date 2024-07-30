from random import randint
from qiskit import transpile
from tqdm import tqdm
import jax
import optax
from jax import numpy as jnp
import pennylane as qml
from bugfix.circuit import Circuit, circuit_to_pennylane
# from bugfix.clliford.clliford_gate_variables import CllifordCorrecter
from bugfix.clliford.clliford_soft import CllifordCorrecter
from bugfix.clliford.utills import CllifordProgram, custom_random_circuit, generate_inout_stabilizer_tables
from bugfix.paramizefix.pennylane_siwei import generate_input_states as generate_input_states_pennylane, GateParameterOptimizer
from bugfix.paramizefix.qiskit import apply_circuit, generate_bugged_circuit, generate_input_states, optimize_parameters, replace_param_gates_with_clifford

'''
    加入迭代
'''

n_qubits = 8
basic_gates = ['h', 'cx', 'rx', 'ry', 'rz']  # 'x', 'y', 'z', 'cz', 
correct_circuit = custom_random_circuit(
    n_qubits, 20, gate_set=basic_gates)


bugged_circuit = generate_bugged_circuit(
    correct_circuit.copy(), 3)

print("correct_circuit:")
print(correct_circuit)
print("Bugged Circuit:")
print(bugged_circuit)

while True:
    # TODO: 先梯度下降
    
    # 先找 patches
    correct_clliford = replace_param_gates_with_clifford(correct_circuit) 
    # bugged_clifford = replace_param_gates_with_clifford(correct_circuit) 
    bugged_clifford = replace_param_gates_with_clifford(bugged_circuit)
    print("\n\n\n\n\ncorrect_clliford:")
    print(correct_clliford)
    print("bugged_clifford:")
    print(bugged_clifford)

    program = CllifordProgram.from_circuit(bugged_clifford, n_qubits=n_qubits)  # TODO: n_layers
    correcter = CllifordCorrecter(program.n_qubits, len(program), time_out_eff = 1000)
    inputs, outputs = [], []
    for _ in tqdm(range(min(2**n_qubits//2, 20))):
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(
            program.n_qubits, program)
        inputs.append(input_stabilizer_table)
        outputs.append(output_stabilizer_table)
        correcter.add_iout(input_stabilizer_table, output_stabilizer_table)
    find_program = correcter.solve()  # 60

    # 找不到就用旧的
    if find_program is None:
        find_program = program

    # for input_stabilizer_table, output_stabilizer_table in zip(inputs, outputs):
    #     predict_out = find_program.output_stablizers(input_stabilizer_table)
    #     assert predict_out.is_eq(output_stabilizer_table)

    correct_circuit = Circuit(correct_circuit)

    optimizer = GateParameterOptimizer.from_clifford_program(find_program)
    repaired_circuit, dist = optimizer.optimize_target_unitary(correct_circuit.matrix(), n_epochs=1000)


    # input_states = generate_input_states_pennylane(n_qubits, n_states=2**n_qubits//2)
    # target_output_states = []
    # for input_state in input_states:
    #     target_output_states.append(correct_circuit.run(input_state))
    # repaired_circuit, dist = optimizer.optimize_target_output_states(input_states, target_output_states, n_epochs=1000, n_batch = 20)

    
    if dist < 0.1:
        break

    correct_circuit = correct_circuit.to_qiskit()
    bugged_circuit = transpile(repaired_circuit.to_qiskit(), basis_gates=basic_gates, optimization_level=3)


import os
from dataset2.baselline import load_dataset

from random import randint
from qiskit import transpile
from tqdm import tqdm
import jax
import optax
from jax import numpy as jnp
import pennylane as qml
from bugfix.circuit import Circuit, circuit_to_pennylane
# from bugfix.clliford.clliford_gate_variables import CllifordCorrecter
from bugfix.clliford.cllifor_gate_parr import CllifordCorrecter
from bugfix.clliford.utills import CllifordProgram, custom_random_circuit, generate_inout_stabilizer_tables
from bugfix.paramizefix.pennylane_siwei import generate_input_states as generate_input_states_pennylane, GateParameterOptimizer
from bugfix.paramizefix.qiskit import apply_circuit, generate_bugged_circuit, generate_input_states, optimize_parameters, replace_param_gates_with_clifford
import json
from time import perf_counter
import ray
import time

def repair(correct_circuit, bugged_circuit,n_qubits, n_errors,id):
    basic_gates = ['cx','u']  # 'x', 'y', 'z', 'cz', 
    
    # correct_circuit = custom_random_circuit(
    #     n_qubits, 20, gate_set=basic_gates)
    # bugged_circuit = generate_bugged_circuit(
    #     correct_circuit.copy(), 3)
    
    # print("correct_circuit:")
    # print(correct_circuit)
    # print("Bugged Circuit:")
    # print(bugged_circuit)
    metrics = {}
    repair_start = perf_counter()
    for i in range(5):
        print(f"repairing {i+1}th time")
        # TODO: 先梯度下降
        
        # 先找 patches
        correct_clliford = replace_param_gates_with_clifford(correct_circuit) 
        bugged_clifford = replace_param_gates_with_clifford(bugged_circuit) 
        # bugged_clifford = replace_param_gates_with_clifford(bugged_circuit)
        # print("\n\n\n\n\ncorrect_clliford:")
        # print(correct_clliford)
        # print("bugged_clifford:")
        # print(bugged_clifford)

        program = CllifordProgram.from_circuit(bugged_clifford, n_qubits=n_qubits)  # TODO: n_layers
        correct_program = CllifordProgram.from_circuit(correct_clliford, n_qubits=n_qubits)  # TODO: n_layers
        correcter = CllifordCorrecter(program.n_qubits, 5, time_out_eff = 100)
        inputs, outputs = [], []
        for _ in tqdm(range(min(2**n_qubits//2, 20))):
            input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(
                program.n_qubits, correct_program)
            middle_stabilizer_table = program.output_stablizers(input_stabilizer_table)
            inputs.append(middle_stabilizer_table)
            outputs.append(output_stabilizer_table)
        correcter.add_iout(inputs, outputs)
        start = perf_counter()
        solve_program = correcter.solve()  # 60
        end = perf_counter()
        metrics[f'clliford_time_{i}'] = end - start
        new_program = program.copy()
        # 找不到就用旧的
        if solve_program:
            new_program.extend(solve_program)
        find_program = new_program

        correct_circuit = Circuit(correct_circuit)
        optimizer = GateParameterOptimizer.from_clifford_program(find_program)
        start = perf_counter()
        repaired_circuit, dist = optimizer.optimize_target_unitary(correct_circuit.matrix(), n_epochs=1000)
        end = perf_counter()
        metrics[f'param_time_{i}'] = end - start

        # input_states = generate_input_states_pennylane(n_qubits, n_states=2**n_qubits//2)
        # target_output_states = []
        # for input_state in input_states:
        #     target_output_states.append(correct_circuit.run(input_state))
        # repaired_circuit, dist = optimizer.optimize_target_output_states(input_states, target_output_states, n_epochs=1000, n_batch = 20)

        
        if dist < 0.1:
            break
        
        correct_circuit = correct_circuit.to_qiskit()
        bugged_circuit = transpile(repaired_circuit.to_qiskit(), basis_gates=basic_gates, optimization_level=3)

    # print("repaired circuit:")
    # print(repaired_circuit)
    # print("distance:", dist)
    metrics['distance'] = float(dist)
    metrics['repaired_circuit'] = repaired_circuit.to_json()
    metrics['total_time'] = perf_counter() - repair_start
    metrics['repaired_times'] = i+1
    if dist < 0.1:
        
        metrics['state'] = 'success'
        
        ## save the repaired circuit as qasm
        if not os.path.exists(f'data/repaired_circuits/{n_qubits}_errors_{n_errors}/'):
            os.makedirs(f'data/repaired_circuits/{n_qubits}_errors_{n_errors}/')
        repaired_circuit.to_qasm(f'data/repaired_circuits/{n_qubits}_errors_{n_errors}/repaired_circuit{id}.qasm')
    else:
        metrics['state'] = 'failed'
        
    ## save the metrics to json
    if not os.path.exists(f'data/metrics/{n_qubits}_errors_{n_errors}/'):
        os.makedirs(f'data/metrics/{n_qubits}_errors_{n_errors}/')
    with open(f'data/metrics/{n_qubits}_errors_{n_errors}/metrics{id}.json', 'w') as f:
        json.dump(metrics, f)

import traceback
@ray.remote
def repair_remote(*args, **kwargs):
    try:
        # raise Exception('test')
        return repair(*args, **kwargs)
    except Exception as e:
        with open('error.log', 'a') as f:
            traceback.print_exc(file=f)
        return None
if __name__ == '__main__':
    # for n_qubits in [5,10]:
    #     print('n_qubits:', n_qubits)
    #     for n_errors in range(2,11,2):
    #         print('n_errors:', n_errors)
    #         # n_qubits = 5
    #         # n_errors = 2
    #         qcs = load_dataset(n_qubits, n_errors)

    #         assert len(qcs) == 100
    #         qc_corrects, qc_bugs = qcs[:50], qcs[50:]
    #         metrics = ray.get([repair_remote.remote(qc_correct, qc_bug,n_qubits, n_errors,id=i) for i, (qc_correct, qc_bug) in enumerate(zip(qc_corrects, qc_bugs))])
    #         # for qc_correct, qc_bug in zip(qc_corrects, qc_bugs):
    #         #     repair(qc_correct, qc_bug,id= i)
    #         #     i += 1

    futures = []
    for n_qubits in [10]:
        for n_errors in [8, 10]:
            qcs = load_dataset(n_qubits, n_errors)
            assert len(qcs) == 100
            qc_corrects, qc_bugs = qcs[:50], qcs[50:]
            futures +=[repair_remote.remote(qc_correct, qc_bug,n_qubits, n_errors,id=i) for i, (qc_correct, qc_bug) in enumerate(zip(qc_corrects, qc_bugs))]
            time.sleep(1200)
            
    ray.get(futures)
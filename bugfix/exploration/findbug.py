from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import numpy as np


def replace_with_param_gates(qc):
    new_qc = QuantumCircuit(qc.num_qubits)
    params = 0
    initial_params = []
    for gate in qc.data:
        op,g_qargs = gate.operation,gate.qubits
        g_name = op.name
        g_params = Parameter(f'θ_{params}')
        if g_name == 'x':
            new_qc.rx(g_params, g_qargs[0])
            initial_params.append(np.pi/2)
        elif g_name == 'y':
            new_qc.ry(g_params, g_qargs[0])
            initial_params.append(np.pi/2)
        elif g_name == 'z':
            new_qc.rz(g_params, g_qargs[0])
            initial_params.append(np.pi/2)
        elif g_name == 'rx':
            new_qc.rx(g_params, g_qargs[0])
            initial_params.append(op.params[0])
        elif g_name == 'ry':
            new_qc.ry(g_params, g_qargs[0])
            initial_params.append(op.params[0])
        elif g_name == 'rz':
            new_qc.rz(g_params, g_qargs[0])
            initial_params.append(op.params[0])
        elif g_name == 'cx':
            new_qc.crx(g_params,g_qargs[0], g_qargs[1])
            initial_params.append(np.pi/2)
        elif g_name == 'cy':
            new_qc.cry(g_params, g_qargs[0], g_qargs[1])
            initial_params.append(np.pi/2)
        elif g_name == 'cz':
            new_qc.crz(g_params, g_qargs[0], g_qargs[1])
            initial_params.append(np.pi/2)
        elif g_name == 'crx':
            new_qc.crx(g_params, g_qargs[0], g_qargs[1])
            initial_params.append(op.params[0])
        elif g_name == 'cry':
            new_qc.cry(g_params, g_qargs[0], g_qargs[1])
            initial_params.append(op.params[0])
        elif g_name == 'crz':
            new_qc.crz(g_params, g_qargs[0], g_qargs[1])
            initial_params.append(op.params[0])
        else:
            new_qc.append(gate)
        params += 1
    return new_qc, initial_params

def simulate_circuit(qc, param_values,input_state=None):
    param_dict = {param: value for param, value in zip(qc.parameters, param_values)}
    bound_qc = qc.bind_parameters(param_dict)
    backend = Aer.get_backend('statevector_simulator')
    if input_state is not None:
        initial_state = execute(input_state, backend).result().get_statevector(input_state)
        job = execute(bound_qc, backend, initial_statevector=initial_state)
    else:
        job = execute(bound_qc, backend)
    result = job.result()
    statevector = result.get_statevector(bound_qc)
    return statevector

def fidelity(state1, state2):
    return np.abs(np.dot(np.conj(state1), state2))**2

def objective_function(params, correct_state, wrong_qc,input_states):
    fidelity_sum = 0
    for input_state in input_states:
        correct_state = simulate_circuit(correct_qc, params, input_state)
        wrong_state = simulate_circuit(wrong_qc, params, input_state)
        fidelity_sum += fidelity(correct_state, wrong_state)
    return 1 - fidelity_sum / len(input_states)

def gradient(objective_fn, params, correct_state, wrong_qc,input_states, i):
    shift = np.pi / 100
    params_shifted = params.copy()
    params_shifted[i] += shift
    forward = objective_fn(params_shifted, correct_state, wrong_qc,input_states)
    
    params_shifted[i] -=  shift
    backward = objective_fn(params_shifted, correct_state, wrong_qc,input_states)
    
    return (forward - backward) / 2

def find_bug_location(correct_qc, wrong_qc,input_states):
    # Replace gates with parameterized gates
    # param_correct_qc, num_params = replace_with_param_gates(correct_qc)
    param_wrong_qc, initial_params = replace_with_param_gates(wrong_qc)
    num_params = len(initial_params)
    # Simulate the correct circuit
    correct_state = simulate_circuit(correct_qc, [0] * num_params)
    # Calculate gradients for each parameter

    gradients = []
    for i in range(len(initial_params)):
        grad = gradient(objective_function, initial_params, correct_state, param_wrong_qc,input_states, i)
        gradients.append(grad)
    print(gradients)
    
    # Identify the gate with the maximum gradient
    max_gradient_index = np.argmax(np.abs(gradients))
    return max_gradient_index, gradients[max_gradient_index]

# Generate multiple input states
input_states = [QuantumCircuit(2) for _ in range(4)]
input_states[0].h(0)
input_states[0].h(1)
input_states[1].rx(0.4,1)
input_states[1].ry(0.2,0)
input_states[2].h(1)
input_states[2].h(0)
input_states[2].cx(0, 1)
input_states[3].ry(0.4,0)
input_states[3].rx(0.2,1)
input_states[3].h(0)

# Example usage
correct_qc = QuantumCircuit(2)
correct_qc.h(0)
correct_qc.y(1)
correct_qc.cx(0, 1)
correct_qc.rz(0.7, 0)
correct_qc.cx(1, 0)
correct_qc.ry(0.7, 1)

wrong_qc = QuantumCircuit(2) # Introduce a bug here
wrong_qc.h(0)
wrong_qc.y(1)
wrong_qc.cx(0,1)
wrong_qc.rz(0.5, 0)
wrong_qc.cx(1, 0)
wrong_qc.ry(0.6, 1)

bug_location, gradient_value = find_bug_location(correct_qc, wrong_qc,input_states)
print(f"The bug is likely located at gate index: {bug_location}, with a gradient of: {gradient_value}")

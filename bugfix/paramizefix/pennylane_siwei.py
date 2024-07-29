import pennylane as qml

import jax
from jax import numpy as jnp
from jax import grad, jit
import numpy as np
import optax
from jax.config import config
from qiskit import QuantumCircuit
from sklearn.utils import shuffle
from bugfix.circuit import Circuit, circuit_to_pennylane
from tqdm import tqdm
from random import randint

from bugfix.optimizer import OptimizingHistory

config.update("jax_enable_x64", True)


def generate_input_states(n_qubits, n_states=8):
    states = []
    from qiskit.quantum_info import random_clifford

    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        matrix = cllifordgate.to_matrix()
        init_state = jnp.zeros(2**n_qubits)
        init_state = init_state.at[0].set(1)
        state = jnp.dot(matrix, init_state)
        states.append(state)
    return states


class GateParameterOptimizer:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit

        n_qubits = circuit.n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @jax.jit
        @qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
        def run(input_state, params):
            qml.QubitStateVector(input_state, wires=range(n_qubits))
            circuit_to_pennylane(circuit, params)
            return qml.probs()

        self.run = run

    @staticmethod
    def from_clifford_program(clifford_program: list[tuple[str, list]]):
        """convert the clifford program to circuit_data"""
        n_qubits = clifford_program.n_qubits

        circuit = []
        for inst, qubits in clifford_program:
            if isinstance(qubits, int):
                circuit.append([{
                    'name': 'u3',
                    'qubits': [qubits],
                    'params': [0, 0, 0],
                }])
            elif inst.lower() in ("cnot", 'cx'):
                circuit.append([{
                    'name': 'cx',
                    'qubits': qubits,
                    'params': [],
                }])
            else:
                raise ValueError("Invalid gate type: {}".format(inst.name))

        circuit = Circuit(circuit, n_qubits)
        return GateParameterOptimizer(circuit)

    @staticmethod
    def from_circuit(circuit: Circuit):
        if isinstance(circuit, QuantumCircuit):
            circuit = Circuit(circuit)

        new_circuit = []
        for layer in circuit:
            new_layer = []
            for gate in layer:
                qubits = gate['qubits']
                if len(qubits) == 1:
                    new_layer.append({
                        'name': 'u3',
                        'qubits': qubits,
                        'params': [0, 0, 0],
                    })
                elif gate['name'] in ('cx', 'cnot'):
                    new_layer.append({
                        'name': 'cx',
                        'qubits': qubits,
                        'params': [],
                    })
                elif gate['name'] in ('cz', ):
                    new_layer.append({
                        'name': 'cz',
                        'qubits': qubits,
                        'params': [],
                    })
                else:
                    raise ValueError("Invalid gate type: {}".format(gate))
            new_circuit.append(new_layer)

        circuit = Circuit(new_circuit, circuit.n_qubits)
        print('optimized circuit')
        print(circuit)
        return GateParameterOptimizer(circuit)

    def get_n_params(self):
        """gate paramize size"""
        n_params = 0

        for gate in self.circuit.gates:
            if gate["name"] in ('rx', 'x', 'ry', 'y', 'rz', 'z'):
                n_params += 1
            elif gate["name"] in ('u3',):
                n_params += 3

        return n_params

    def gate_params(self):
        """parameterize the gates"""

        params = []
        for gate in self.circuit.gates:
            if gate["name"] in ("rx", "ry", "rz", "u3"):
                params += gate["params"]
            if gate["name"] in ('x', 'y', 'z'):
                params.append(jnp.pi)

        return params

    def optimize_target_output_states(self, input_states, target_output_states, n_epochs=50, n_batch=10, lr=0.1):
        input_states = jnp.array(input_states)
        target_output_states = jnp.array(target_output_states)

        n_params = self.get_n_params()
        print("n_params:", n_params)

        opt = optax.adam(learning_rate=lr)
        params = jax.random.uniform(jax.random.PRNGKey(
            randint(0, 200)), (n_params,), minval=0, maxval=jnp.pi * 2)
        # params = jnp.array(circuit_tape.gate_params())
        opt_state = opt.init(params)

        def loss_fn(params, input_state, target_output_state):
            output_state = self.run(input_state, params)
            return jnp.mean(jnp.abs(output_state - target_output_state))

        # @jax.jit
        def loss_batch(params, input_states, target_output_states):
            return jnp.sum(jax.vmap(loss_fn, in_axes=(None, 0, 0))(params, input_states, target_output_states))
        
            # loss = 0.0
            # for input_state, target_output_state in zip(input_states, target_output_states):
            #     output_state = self.run(input_state, params)
            #     loss += jnp.mean(jnp.abs(output_state - target_output_state))
            # return loss

        opt_history = OptimizingHistory(
            params, lr, 0.001, 500, n_epochs, 0.01, False)

        with tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):
                input_states, target_output_states = shuffle(
                    input_states, target_output_states)

                loss_val, grads = jax.value_and_grad(loss_batch)(
                    params, input_states[:n_batch], target_output_states[:n_batch])
                updates, opt_state = opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                opt_history.update(loss_val, params)

                pbar.update(1)
                loss_val = round(float(loss_val), 7)
                pbar.set_description(f"Loss: {loss_val}")

                if opt_history.should_break:
                    break

        return assign_params(opt_history.best_params, self.circuit)

    def optimize_target_unitary(self, target_unitary: jnp.ndarray, n_epochs=50, lr=0.1):
        n_params = self.get_n_params()
        print("n_params:", n_params)

        opt = optax.adam(learning_rate=lr)
        params = jax.random.uniform(jax.random.PRNGKey(
            randint(0, 200)), (n_params,), minval=0, maxval=jnp.pi * 2)
        # params = jnp.array(circuit_tape.gate_params())
        opt_state = opt.init(params)

        # @jax.jit
        def loss_fn(params):
            return matrix_distance_squared(self.circuit.matrix(params), target_unitary)

        opt_history = OptimizingHistory(
            params, lr, 0.001, 400, n_epochs, 0.01, False)

        with tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):
                loss_val, grads = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                opt_history.update(loss_val, params)

                pbar.update(1)
                loss_val = round(float(loss_val), 7)
                pbar.set_description(f"Loss: {loss_val}")

                if opt_history.should_break:
                    break

        return assign_params(opt_history.best_params, self.circuit), opt_history.min_loss

        # return loss_history, opt_history.best_params


def assign_params(params, circuit: Circuit) -> Circuit:
    circuit = circuit.copy()
    count = 0
    for gates in circuit:
        for gate in gates:
            for index, _ in enumerate(gate['params']):
                gate['params'] = np.array(gate['params'])
                gate['params'][index] = params[count]
                count += 1
    return circuit


@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])

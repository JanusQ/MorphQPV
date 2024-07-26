import pennylane as qml

import jax
from jax import numpy as jnp
from jax import grad, jit
import numpy as np
import optax

n_wires = 3


class CircuitTape:
    def __init__(self, *args, **kwargs):
        self.circuit_data = []
        return

    def update(self):
        return

    """ the following methods are used to add gates to the circuit_data """

    def x(self, qubits: list):
        self.circuit_data.append({"name": "x", "qubits": qubits})
        return

    def y(self, qubits: list):
        self.circuit_data.append({"name": "y", "qubits": qubits})
        return

    def z(self, qubits: list):
        self.circuit_data.append({"name": "z", "qubits": qubits})
        return

    def h(self, qubits: list):
        self.circuit_data.append({"name": "h", "qubits": qubits})
        return

    def rx(self, qubits: list, params: list):
        """add a rx gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angles for each qubit, unit: rad
        """
        self.circuit_data.append({"name": "rx", "qubits": qubits, "params": params})
        return

    def ry(self, qubits: list, params: list):
        """add a ry gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angles for each qubit, unit: rad
        """
        self.circuit_data.append({"name": "ry", "qubits": qubits, "params": params})
        return

    def rz(self, qubits: list, params: list):
        """add a rz gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angle
        """
        self.circuit_data.append({"name": "rz", "qubits": qubits, "params": params})
        return

    def s(self, qubits: list):
        """add a s gate to the circuit_data
        Args:
            qubits (list): the qubits to apply the gate
        """
        self.circuit_data.append({"name": "s", "qubits": qubits})
        return

    def cnot(self, qubits: list):
        """add a cnot gate to the circuit_data

        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({"name": "cx", "qubits": qubits})
        return

    def cx(self, qubits: list):
        """add a cnot gate to the circuit_data


        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({"name": "cx", "qubits": qubits})
        return

    def cz(self, qubits: list):
        """add a cz gate to the circuit_data

        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({"name": "cz", "qubits": qubits})
        return

    def swap(self, qubits: list):
        """add a swap gate to the circuit_data


        Args:
            qubits (list): [qubit1, qubit2]
        """
        self.circuit_data.append({"name": "swap", "qubits": qubits})
        return

    def t(self, qubits: list):
        self.circuit_data.append({"name": "t", "qubits": qubits})
        return

    def tdg(self, qubits: list):
        self.circuit_data.append({"name": "tdg", "qubits": qubits})
        return

    def u1(self, qubits: list, params: list):
        """add a u1 gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the phase angles for each qubit, unit: rad
        """
        self.circuit_data.append({"name": "u1", "qubits": qubits, "params": params})
        return

    def u2(self, qubits: list, params: list):
        """add a u2 gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the [phi, lambda] angles for each qubit, unit: rad
        """
        self.circuit_data.append({"name": "u2", "qubits": qubits, "params": params})
        return

    def u3(self, qubits, params):
        self.circuit_data.append({"name": "u3", "qubits": qubits, "params": params})
        return

    def unitary(self, qubits, params):
        self.circuit_data.append(
            {"name": "unitary", "qubits": qubits, "params": params}
        )
        return

    def mcx(self, qubits):
        self.circuit_data.append({"name": "mcx", "qubits": qubits})
        return

    def mcp(self, qubits):
        self.circuit_data.append({"name": "mcp", "qubits": qubits})
        return

    def to_qiskit_circuit(self):
        """convert the circuit_data to qiskit circuit"""
        return

    def from_qiskit_circuit(self, qiskit_circuit):
        """convert the qiskit circuit to circuit_data"""
        return

    def to_layer_circuit(self, circuit_data):
        """convert the circuit_data to layer circuit"""
        layer_circuit = [[]]
        for gate in circuit_data:
            is_new_layer = False
            for qubit in gate["qubits"]:
                if qubit in [gate["qubits"] for gate in layer_circuit[-1]]:
                    is_new_layer = True
                    break
            if is_new_layer:
                layer_circuit.append([gate])
            else:
                layer_circuit[-1].append(gate)
        return layer_circuit

    def replace_param_gates_with_clifford(self, circuit_data):
        """replace the parameterized gates with clifford gates"""
        clifford_circuit = []
        for gate in circuit_data:
            if gate["name"] == "rx":
                clifford_circuit.append({"name": "x", "qubits": gate["qubits"]})
            elif gate["name"] == "ry":
                clifford_circuit.append({"name": "y", "qubits": gate["qubits"]})
            elif gate["name"] == "rz":
                clifford_circuit.append({"name": "z", "qubits": gate["qubits"]})
            else:
                clifford_circuit.append(gate)
        return clifford_circuit

    def gate_paramize(self, params):
        """parameterize the gates"""
        num_params = 0
        for gate in self.circuit_data:
            if gate["name"] == "rx" or gate["name"] == "x":
                # print('rx')
                qml.RX(params[num_params], wires=gate["qubits"][0])
                num_params += 1
            elif gate["name"] == "ry" or gate["name"] == "y":
                # print('ry')
                qml.RY(params[num_params], wires=gate["qubits"][0])
                num_params += 1

            elif gate["name"] == "rz" or gate["name"] == "z":
                # print('rz')

                qml.RZ(params[num_params], wires=gate["qubits"][0])
                num_params += 1

            else:
                self.to_pennylane_gate(gate)

        return num_params

    def gate_params(self):
        """parameterize the gates"""

        params = []
        for gate in self.circuit_data:
            if gate["name"] in ("rx", "ry", "rz", "u3"):
                params += gate["params"]
            if gate["name"] in ('x', 'y', 'z'):
                params.append(jnp.pi)

        return params

    def to_pennylane_circuit(self):
        for gate in self.circuit_data:
            self.to_pennylane_gate(gate)

    def to_qiskit_circuit(self):
        """
        Purpose:  convert the circuit_data to qiskit circuit
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_wires)
        for gate in self.circuit_data:
            if gate["name"] == "x":
                qc.x(gate["qubits"][0])

            elif gate["name"] == "y":
                qc.y(gate["qubits"][0])

            elif gate["name"] == "z":
                qc.z(gate["qubits"][0])

            elif gate["name"] == "h":
                qc.h(gate["qubits"][0])

            elif gate["name"] == "rx":
                qc.rx(gate["params"][0], gate["qubits"][0])

            elif gate["name"] == "ry":
                qc.ry(gate["params"][0], gate["qubits"][0])

            elif gate["name"] == "rz":
                qc.rz(gate["params"][0], gate["qubits"][0])

            elif gate["name"] == "cx":
                qc.cx(gate["qubits"][0], gate["qubits"][1])

            elif gate["name"] == "cz":
                qc.cz(gate["qubits"][0], gate["qubits"][1])

    # end def
    def to_pennylane_gate(self, gate):
        """convert the circuit_data to pennylane circuit"""
        if gate["name"] == "x":
            qml.PauliX(wires=gate["qubits"][0])

        elif gate["name"] == "y":
            qml.PauliY(wires=gate["qubits"][0])
        elif gate["name"] == "z":
            qml.PauliZ(wires=gate["qubits"][0])

        elif gate["name"] == "h":
            qml.Hadamard(wires=gate["qubits"][0])

        elif gate["name"] == "rx":
            qml.RX(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "ry":
            qml.RY(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "rz":
            qml.RZ(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "cx":
            qml.CNOT(wires=gate["qubits"])

        elif gate["name"] == "cz":
            qml.CZ(wires=gate["qubits"])

        elif gate["name"] == "swap":
            qml.SWAP(wires=gate["qubits"])

        elif gate["name"] == "t":
            qml.T(wires=gate["qubits"][0])

        elif gate["name"] == "tdg":
            qml.T(wires=gate["qubits"][0]).inv()

        elif gate["name"] == "u1":
            qml.U1(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "u2":
            qml.U2(gate["params"][0], gate["params"][1], wires=gate["qubits"][0])

        elif gate["name"] == "u3":
            qml.U3(
                gate["params"][0],
                gate["params"][1],
                gate["params"][2],
                wires=gate["qubits"][0],
            )

        elif gate["name"] == "unitary":
            qml.QubitUnitary(gate["params"][0], wires=gate["qubits"])

        elif gate["name"] == "mcx":
            qml.MultiControlledX(
                gate["qubits"], control_values="111", wires=range(n_wires)
            )

        else:
            raise ValueError("Gate not recognized")


circuit_tape = CircuitTape()
circuit_tape.h([0])
circuit_tape.h([1])
circuit_tape.cx([0, 1])
circuit_tape.x([0])
circuit_tape.y([1])
circuit_tape.z([2])
circuit_tape.rx([0], [0.5])
circuit_tape.cx([1, 0])

dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev)
def target_circuit(input_state):
    """Quantum circuit ansatz"""

    qml.QubitStateVector(input_state, wires=range(n_wires))
    circuit_tape.to_pennylane_circuit()
    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    # return qml.state()
    return qml.probs()


@qml.qnode(dev)
def parameterized_circuit(input_state, params):
    """Quantum circuit ansatz"""

    qml.QubitStateVector(input_state, wires=range(n_wires))
    circuit_tape.gate_paramize(params)
    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    # return qml.state()
    return qml.probs()


from random import randint

num_params = circuit_tape.gate_paramize(jnp.ones(9))
opt = optax.adam(learning_rate=0.1)



def generate_input_states(num_qubits, num_states=8):
    states = []
    from qiskit.quantum_info import random_clifford

    for _ in range(num_states):
        cllifordgate = random_clifford(num_qubits)
        matrix = cllifordgate.to_matrix()
        init_state = jnp.zeros(2**num_qubits)
        init_state = init_state.at[0].set(1)
        state = jnp.dot(matrix, init_state)
        states.append(state)
    return states


    

target_output_states = []
input_states = generate_input_states(n_wires)
for input_state in input_states:
    target_output_states.append(target_circuit(input_state))

    
# for input_state, target_state in zip(input_states, target_output_states):
#     parm_output_state = parameterized_circuit(input_state, params)
#     output_state = target_circuit(input_state)
    
#     print(np.round(parm_output_state, 2))
#     print(np.round(output_state, 2))
#     # print(np.round(output_state - parm_output_state, 2))
#     print(jnp.mean(jnp.abs(parm_output_state - output_state)))
#     print()

@jax.jit
def loss_fn(params):
    loss = 0

    for input_state, target_state in zip(input_states, target_output_states):
        output_state = parameterized_circuit(input_state, params)
        loss += jnp.mean(jnp.abs(output_state - target_state))
    return loss


def update_step(
    opt,
    params,
    opt_state,
):
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


loss_history = []
from tqdm import tqdm

# params = jnp.zeros(num_params)
# params = jax.random.normal(jax.random.PRNGKey(randint(0, 200)), (num_params,))
params = jax.random.uniform(jax.random.PRNGKey(randint(0, 200)), (num_params,), minval = 0, maxval= jnp.pi * 2)
# params = jnp.array(circuit_tape.gate_params())
print("num_params:", num_params)
opt_state = opt.init(params)

with tqdm(total=1000) as pbar:
    for i in range(1000):
        params, opt_state, loss_val = update_step(opt, params, opt_state)
        pbar.update(1)
        loss_val = round(float(loss_val), 7)
        pbar.set_description(f"Loss: {loss_val}")
        loss_history.append(loss_val)
        
print(params)
print(loss_history)

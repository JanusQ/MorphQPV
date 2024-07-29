import pennylane as qml

import jax
from jax import numpy as jnp
from jax import grad, jit
import numpy as np
import optax
from jax.config import config
from qiskit import QuantumCircuit
from sklearn.utils import shuffle
from bugfix.circuit import Circuit
from tqdm import tqdm
from random import randint
from bugfix.paramizefix.pennylane import CircuitTape, generate_input_states as generate_input_states_pennylane

config.update("jax_enable_x64", True)

n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

@jax.jit
@qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
def run(input_state):
    qml.QubitStateVector(input_state, wires=range(n_qubits))
    qml.X(wires=0)
    qml.RX(np.pi/3, wires = 1)
    for i in range(100): 
        qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.probs()

input_states = generate_input_states_pennylane(n_qubits, n_states=10)

for input_state in tqdm(input_states):
    run(input_state)


    

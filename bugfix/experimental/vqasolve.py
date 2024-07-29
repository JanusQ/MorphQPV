## This file use a VQA algorithm to find the linear combination of the unitaries that is closest to the target unitary.

import jax.numpy as jnp
from jax import value_and_grad, jit, grad
from tqdm import tqdm
from jax import random
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from jax.example_libraries.optimizers import adam
from itertools import product
from pennylane.pauli import PauliSentence,PauliWord
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor

a = 0.25
b = 0.75

# matrix to be decomposed
A = np.array(
    [[a,  0, 0,  b],
     [0, -a, b,  0],
     [0,  b, a,  0],
     [b,  0, 0, -a]]
)

LCU = qml.pauli_decompose(A)

print(f"LCU decomposition:\n {LCU}")
print(f"Coefficients:\n {LCU.coeffs}")
print(f"Unitaries:\n {LCU.ops}")
def tensor_product(*matrices):
    """ TENSOR PRODUCT
    Args:
        matrices: matrices
    Returns:
        tensor_product: tensor product of matrices
    """
    if len(matrices)==1:
        return matrices[-1]
    else:
        return matrices[-1]@ tensor_product(*matrices[:-1])

## define the linear combination of the unitaries circuit
def generate_pauli_unitaries(n_qubits):
    
    paulis = [qml.PauliX, qml.PauliY,qml.PauliZ,qml.Identity]
    for i,pauliset in enumerate(product(paulis, repeat=n_qubits)):
        gate = tensor_product([pauli(4+i) for i,pauli in enumerate(pauliset)])
        yield gate

dev2 = qml.device("default.qubit", wires=6)
@qml.qnode(dev2)
def lcu_circuit(params):  # block_encode
    # PREP
    alphas= (jnp.sqrt(params) / jnp.linalg.norm(jnp.sqrt(params)))
    qml.StatePrep(alphas, wires=[0,1,2,3])

    # SEL
    qml.Select(unitaries, control=[0,1,2,3])

    # PREP_dagger
    qml.adjoint(qml.StatePrep(alphas, wires=[0,1,2,3]))
    return qml.probs(wires=[4,5])

## define the target unitary
## 2 qubit unitary
target_unitary = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

n_qubits = 2
unitaries = list(generate_pauli_unitaries(n_qubits))


def target_unitary_circuit():
    qml.QubitUnitary(target_unitary, wires=[0, 1])
    return qml.probs(wires=[0, 1])

target_probs = target_unitary_circuit()

## define the cost function
def distance(params):
    output_probs = lcu_circuit(params)
    return np.sum(np.abs(output_probs - target_probs))

## define the optimizer


## define the main function
def main():
    params = np.random.uniform(0, 1, (16,))
    
    opt_init, opt_update, get_params = adam(step_size=0.01)
    min_cost = 1e10
    last_cost = min_cost
    opt_state = opt_init(params)
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            params = get_params(opt_state)
            cost, grads = value_and_grad(distance)(params)
            opt_state=  opt_update(i, grads, opt_state)
            if cost < min_cost:
                min_cost = cost
                bast_params = params
                pbar.set_description(f'sgd optimizing ')
                #设置进度条右边显示的信息
                pbar.set_postfix(loss=cost, min_loss=min_cost)
            if jnp.abs(min_cost-last_cost) < 1e-5:
                min_iter+=1
            else:
                min_iter = 0
            # 当连续50次迭代损失函数变化小于1e-5时，认为收敛
            if min_iter > 50:
                pbar.set_description(f'sgd optimizing converge')
                pbar.set_postfix(loss=cost, min_loss=min_cost)
                break
            last_cost = min_cost
    print(f"Final cost: {distance(params)}")
    print(f"Optimized params: {params}")

if __name__ == "__main__":
    main()


    
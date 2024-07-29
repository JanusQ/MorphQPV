## Programming Quantum Computers
##   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
##   O'Reilly Media
##
## More samples like this can be found at http://oreilly-qc.github.io
from pennylane import FlipSign
from qiskit import QuantumCircuit, QuantumRegister
import math
import numpy as np
class PhaseKickback(QuantumCircuit):
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
    
    def flip_sign(self, wires, arr_bin):
        r"""Apply a phase kickback to a state
        Args:
            first_qubit (int): first qubit to apply the phase kickback
            wires (array[int]): wires that the operator acts on
            arr_bin (array[int]): binary array vector representing the state to flip the sign
        Raises:
            ValueError: "Wires length and flipping state length does not match, they must be equal length "
        """
        if len(wires)-1 != len(arr_bin):
            raise ValueError(
                "Wires length and flipping state length does not match, they must be equal length "
            )
        if arr_bin[-1] == 0:
            self.x(wires[-1])
        self.mcp(math.pi, wires[:-1],wires[-1],control_value=arr_bin[:-1])
        if arr_bin[-1] == 0:
            self.x(wires[-1])
        
    def get_random_key(self, n_qubits):
        return [np.random.randint(2) for _ in range(n_qubits)]
    def gen_circuit(self):
        self.h(0)
        self.flip_sign(self.qubits, self.get_random_key(self.n_qubits-1))
        self.h(0)
        return self

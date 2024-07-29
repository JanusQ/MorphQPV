
from itertools import product
from morphQPV import MorphQC
from copy import deepcopy
import numpy as np
def tensor_product(*matrices):
    """ TENSOR PRODUCT
    Args:
        matrices: matrices
    Returns:
        tensor_product: tensor product of matrices
    """
    tensor_product = matrices[0]
    for matrix in matrices[1:]:
        tensor_product = np.kron(tensor_product,matrix)
    return tensor_product
def get_unitary(n_qubits,parms):
    """ GET UNITARY
    Args:
        n_qubits: number of qubits
    Returns:
        unitary: unitary matrix
    """
    pauliX = np.array([[0, 1], [1, 0]])
    pauliY = np.array([[0, -1j], [1j, 0]])
    pauliZ = np.array([[1, 0], [0, -1]])
    paulis = [np.eye(2)] + [pauliX, pauliY, pauliZ]
    unitary = np.eye(2**n_qubits)
    for i,pauli in enumerate(product(paulis, repeat=n_qubits)):
        gate = tensor_product(*pauli)
        unitary = unitary + parms[i] * gate
    return unitary

    
def pauli_gates(qubits):
    """ GENERATE PAULI BASIS GATES
    Args:
        n_qubits: number of qubits
    Returns:
        pauli_basis_gates: list of pauli basis gates
    
    """
    pauli_basis_gates = []
    n_qubits = len(qubits)
    paulis = ['I', 'x', 'y', 'z']
    for pauli in product(paulis, repeat=n_qubits):
        gate = []
        for i in range(n_qubits):
            if pauli[i] == 'I':
                continue
            gate.append({
                "name": pauli[i],
                "qubits": qubits[i],
            })
        pauli_basis_gates.append(gate)
    return pauli_basis_gates
def to_layered_circuit(circuit_data):
    """ TO LAYERED CIRCUIT
    Args:
        circuit_data: circuit data
    Returns:
        layered_circuit: layered circuit
    """
    layered_circuit = []
    layer_qubits = []
    layer = []
    for gate in circuit_data:
        qubit=gate['qubits']
        if isinstance(qubit,int):
            qubit = [qubit]
        hasin = any([q in layer_qubits for q in qubit])
        if not hasin:
            layer_qubits+=qubit
            layer.append(gate)
        else:
            layered_circuit.append(layer)
            layer = [gate]
            layer_qubits = qubit
            
    return layered_circuit

def get_qubits(circuit_data):
    """ GET QUBITS
    Args:
        circuit_data: circuit data
    Returns:
        qubits: qubits
    """
    qubits = []
    for gate in circuit_data: 
        if isinstance(gate['qubits'],int):
            qubits.append(gate['qubits'])
        else:
            qubits+=gate['qubits']
    return list(set(qubits))
class MorphQC2(MorphQC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tracein_idx = None
        self.traceout_idx = None
    def draw(self):
        """ draw the quantum circuit with line and gate
        for example:
        0: --x--h--y--x--
        1: --y--h--y--x--
        2: --x--h--y--x--
        3: --y--h--y--x--
        """
        wires = ''
        self.qubits = get_qubits(self.circuit_data)
        for i,qubit in enumerate(self.qubits):
            wires+=f'{i}: '
            for gate in self.circuit_data:
                if isinstance(gate['qubits'],int):
                    gate['qubits'] = [gate['qubits']]
                if qubit in gate['qubits']:
                    wires+=f'--{gate["name"]}'
                else:
                    wires+=f'---'
            wires+='\n'
        return wires
    def __repr__(self) -> str:
        return self.draw()

    
    def add_unitary_tracein(self,*args):
        """ ADD UNITARY TRACEIN
        Args:
            Qubits: qubits to trace in
        Returns:    
            None
        """
        self.circuit_data.append({'name': 'tracein', 'qubits': args})
        self.tracein_idx = len(self.circuit_data)-1
        return
    def trace_gates(self):
        return self.circuit_data[self.tracein_idx+1:self.traceout_idx]
    def full_circuit(self):
        """ FULL CIRCUIT
        Returns:
            full_circuit: full circuit
        """
        ## drop the tracein and traceout gates
        
        circuit = deepcopy(self.circuit_data)
        for gate in reversed(circuit):
            if gate['name'] == 'tracein':
                circuit.remove(gate)
            elif gate['name'] == 'traceout':
                circuit.remove(gate)
        return to_layered_circuit(circuit)
    def add_unitary_traceout(self,*args):
        """ ADD UNITARY TRACEOUT
        Args:
            Qubits: qubits to trace out 
        Returns:
            None
        """
        self.circuit_data.append({'name': 'traceout', 'qubits': args})
        self.traceout_idx = len(self.circuit_data)-1
        assert self.circuit_data[self.tracein_idx]['qubits'] == self.circuit_data[self.traceout_idx]['qubits']
        return
    
    def replace_trace_gate(self,new_gates):
        """ REPLACE TRACE GATE
        Args:
            new_gates: new gates to be added
        Returns:
            None
        """
        for i in reversed(range(self.tracein_idx,self.traceout_idx+1)):
            self.circuit_data.remove(self.circuit_data[i])
        self.traceout_idx = self.tracein_idx
        for new_gate in new_gates:
            self.circuit_data.insert(self.tracein_idx,new_gate)
            self.traceout_idx+=1
        return 
        
    
    def replace_gate(self,gateindex,new_gates):
        """ REPLACE GATE
        Args:
            gateindex: index of the gate to be replaced
            qubits: qubits of the gate to be replaced
            new_gates: new gates to be added
        Returns:
            None
        """
        self.circuit_data.remove(self.circuit_data[gateindex])
        for new_gate in new_gates:
            self.circuit_data.insert(gateindex,new_gate)
            gateindex+=1
        return
    





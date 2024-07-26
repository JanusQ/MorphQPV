import numpy as np
from z3 import *
from typing import List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
class StabilizerTable:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.X = np.zeros((n_qubits, n_qubits), dtype=bool)
        self.Z = np.zeros((n_qubits, n_qubits), dtype=bool)
        self.P = np.zeros(n_qubits, dtype=bool)
        # Initialize the stabilizer table with Z operators
        for i in range(n_qubits):
            self.Z[i][i] = True  # Z on each qubit
    
    def copy(self):
        new = StabilizerTable(self.n)
        new.X = self.X.copy()
        new.Z = self.Z.copy()
        new.P = self.P.copy()
        return new
    
    def apply_hadamard(self, qubit):
        # Swap X and Z for the given qubit
        self.P ^= (self.X[:,qubit] & self.Z[:,qubit])
        self.X[:, qubit], self.Z[:, qubit] = self.Z[:, qubit].copy(), self.X[:, qubit].copy()
        # P \xor X[qubit]*Z[qubit]

    def apply_phase(self, qubit):
        # S gate (phase gate) on a qubit
        self.P ^= (self.X[:,qubit] & self.Z[:,qubit])
        self.Z[:, qubit] ^= self.X[:, qubit]
    def apply_y(self, qubit):
        # Y gate on a qubit
        self.P ^=  (self.X[:,qubit] ^ self.Z[:,qubit])
        
    def apply_cnot(self, control, target):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            self.P[i] ^= (self.X[i,control] & self.Z[i,target] & (self.X[i,target] ^ self.Z[i,control] ^ True))
        self.X[:, target] ^= self.X[:, control]
        self.Z[:, control] ^= self.Z[:, target]


    def apply_clifford(self, other):
        """
        Applies another stabilizer table (Clifford operator) to this one.
        """
        new_X = np.zeros_like(self.X)
        new_Z = np.zeros_like(self.Z)
        new_P = np.zeros_like(self.P)

        for i in range(self.n):
            x_part = np.zeros(self.n, dtype=bool)
            z_part = np.zeros(self.n, dtype=bool)
            phase = False
            for j in range(self.n):
                # Applying the effect of Z[j] and X[j] from the other stabilizer
                if self.X[i, j]:
                    x_part ^= other.Z[j, :]
                    z_part ^= other.X[j, :]
                    phase ^= other.P[j]
                if self.Z[i, j]:
                    x_part ^= other.X[j, :]
                    z_part ^= other.Z[j, :]
            new_X[i, :] = x_part
            new_Z[i, :] = z_part
            new_P[i] = phase
        self.X = new_X
        self.Z = new_Z
        self.P = new_P
    
    @property
    def table(self):
        """
        Generates the full stabilizer table as a 2N x (2N+1) numpy boolean array.
        """
        full_table = np.zeros((2 * self.n, 2 * self.n + 1), dtype=bool)

        # Fill the full table with X and Z parts and the phase P
        full_table[:self.n, :self.n] = self.X  # Fill X part
        full_table[:self.n, self.n:2 * self.n] = self.Z  # Fill Z part
        full_table[:self.n, 2 * self.n] = self.P  # Fill P part for the stabilizers

        # For destabilizers, the format is usually (I -Z), (X I)
        full_table[self.n:, :self.n] = self.Z  # X -> Z part
        full_table[self.n:, self.n:2 * self.n] = self.X  # Z -> X part
        full_table[self.n:, 2 * self.n] = self.P  # Phase information

        return full_table
    
    def is_eq(self, other):
        """
        Checks if two stabilizer tables are equal.
        """
        return (self.X == other.X).all() and (self.Z == other.Z).all() and (self.P == other.P).all()
    
    def to_string(self,label= 'S'):
        """
        Converts the stabilizer table to a human-readable form including both stabilizers and destabilizers.
        """
        stabilizer_strings = []
        destabilizer_strings = []

        # Generate stabilizers
        for i in range(self.n):
            s = ""
            s += " " + ("+" if not self.P[i] else "-")
            for j in range(self.n):
                if self.X[i, j] and self.Z[i, j]:
                    s += "Y"
                elif self.X[i, j]:
                    s += "X"
                elif self.Z[i, j]:
                    s += "Z"
                else:
                    s += "I"
            
            stabilizer_strings.append(s)

        # Generate destabilizers
        for i in range(self.n):
            d = ""
            d += " " + ("+" if not self.P[i] else "-")
            for j in range(self.n):
                if self.Z[i, j] and self.X[i, j]:
                    d += "Y"
                elif self.Z[i, j]:
                    d += "X"
                elif self.X[i, j]:
                    d += "Z"
                else:
                    d += "I"
            
            destabilizer_strings.append(d)

        stabilizers_section = "Stabilizers: " + " ".join(stabilizer_strings)
        destabilizers_section = "Destabilizers: " + " ".join(destabilizer_strings)
        if label == 'S':
            return " ".join(stabilizer_strings)
        elif label == 'D':
            return " ".join(destabilizer_strings)
        else:
            return "\n".join([stabilizers_section, destabilizers_section])


class CllifordProgram(list):
    def __init__(self, n_qubits:int):
        super().__init__()
        self.n_qubits = n_qubits
    def __setitem__(self, index, item):
        super().__setitem__(index, str(item))

    def insert(self, index, item):
        super().insert(index, str(item))
    def copy(self):
        new_program = CllifordProgram(self.n_qubits)
        for item in self:
            new_program.append(item)
        return new_program
    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            raise TypeError("Can only extend CllifordProgram with another type")
    def h(self, qubit):
        self.append(['H',qubit])
    def s(self, qubit):
        self.append(['S',qubit])
    def sdg(self, qubit):
        for _ in range(3):
            self.append(['S',qubit])
    def cz(self, control, target):
        self.append(['H',target])
        self.append(['CNOT',(control,target)])
        self.append(['H',target])
        
    def x(self, qubit):
        ## X = HZH = HSSH
        self.append(['H',qubit])
        self.append(['S',qubit])
        self.append(['S',qubit])
        self.append(['H',qubit])

    def y(self, qubit):
        ## Y = XZ
        self.append(['Y',qubit])

    def z(self, qubit):
        ## Z = HS = HSH
        for _ in range(2):
            self.append(['S',qubit])
    def cnot(self, control, target):
        self.append(['CNOT',(control,target)])

    def output_stablizers(self,input_stabilizer_table:StabilizerTable=None):
        """
        This function takes the input stabilizer table and outputs the stabilizers and destabilizers of the Clliford program.
        """
        if input_stabilizer_table is None:
            output_stabilizer_table = StabilizerTable(self.n_qubits)
        else:
            output_stabilizer_table = input_stabilizer_table.copy()
        for gate in self:
            if gate[0] == 'H':
                output_stabilizer_table.apply_hadamard(gate[1])
            elif gate[0] == 'S':
                output_stabilizer_table.apply_phase(gate[1])
            elif gate[0] == 'CNOT':
                output_stabilizer_table.apply_cnot(gate[1][0], gate[1][1])
            elif gate[0] == 'Y':
                output_stabilizer_table.apply_y(gate[1])
            else:
                raise ValueError("Invalid gate type: {}".format(gate[0]))
            # print(output_stabilizer_table.to_string())
        return output_stabilizer_table
    def to_circuit(self):
        """
        This function converts the Clliford program to a circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)
        for gate in self:
            if gate[0] == 'H':
                circuit.h(gate[1])
            elif gate[0] == 'S':
                circuit.s(gate[1])
            elif gate[0] == 'CNOT':
                circuit.cx(gate[1][0], gate[1][1])
            elif gate[0] == 'Y':
                circuit.y(gate[1])
            else:
                raise ValueError("Invalid gate type: {}".format(gate[0]))
        return circuit
    
    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit):
        """
        This function takes a circuit and converts it to a Clliford program.
        """
        program = cls(circuit.num_qubits)
        for inst, qargs, cargs in circuit.data:
            qubitindex = circuit.find_bit(qargs[0])[0]
            if inst.name == "h":
                program.h(qubitindex)
            elif inst.name.lower()  == "s":
                program.s(qubitindex)
            elif inst.name.lower() == "sdg" :
                program.sdg(qubitindex)
            elif inst.name.lower()  == "x":
                program.x(qubitindex)
            elif inst.name.lower()  == "y":
                program.y(qubitindex)
            elif inst.name.lower()  == "z":
                program.z(qubitindex)
            elif inst.name.lower()  == "cz":
                program.cz(qubitindex,circuit.find_bit(qargs[1])[0])
            elif inst.name.lower()  == "cx" or inst.name.lower()  == "cnot":
                program.cnot(qubitindex, circuit.find_bit(qargs[1])[0])
            elif inst.name.lower()  == "swap":
                program.append(['CNOT',(circuit.find_bit(qargs[0])[0],circuit.find_bit(qargs[1])[0])])
                program.append(['CNOT',(circuit.find_bit(qargs[1])[0],circuit.find_bit(qargs[0])[0])])
                program.append(['CNOT',(circuit.find_bit(qargs[0])[0],circuit.find_bit(qargs[1])[0])])
            else:
                raise ValueError("Invalid gate type: {}".format(inst.name))
        return program

def generate_input_stabilizer_table(n_qubits:int):
    """
    Generates a random Clliford program with the given number of qubits and depth.
    """
    
    cliff = random_clifford(n_qubits)
    circuit =cliff.to_circuit()
    program = CllifordProgram.from_circuit(circuit)
    out_stab = program.output_stablizers()
    return out_stab

def generate_inout_stabilizer_tables(n_qubits:int,program: CllifordProgram):
    """
    Generates the input and output stabilizer tables for the given Clliford program.
    """
    input_stabilizer_table = generate_input_stabilizer_table(n_qubits)
    # print('output stabilizer table producing')
    output_stabilizer_table = program.output_stablizers(input_stabilizer_table)
    return input_stabilizer_table, output_stabilizer_table

        
    








    
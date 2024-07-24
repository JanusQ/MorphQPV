import numpy as np
from z3 import *
from typing import List, Tuple
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, Aer, transpile, execute
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
    def apply_hadamard(self, qubit):
        # Swap X and Z for the given qubit
        self.P ^= (self.X[:,qubit] & self.Z[:,qubit])
        self.X[:, qubit], self.Z[:, qubit] = self.Z[:, qubit].copy(), self.X[:, qubit].copy()
        # P \xor X[qubit]*Z[qubit]

    def apply_phase(self, qubit):
        # S gate (phase gate) on a qubit
        self.P ^= (self.X[:,qubit] & self.Z[:,qubit])
        self.Z[:, qubit] ^= self.X[:, qubit]

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
    
    def to_string(self):
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
        return stabilizers_section + "\n" + destabilizers_section




class CllifordSolver:
    def __init__(self, n_qubits, d_max=2):
        self.n = n_qubits
        self.Xs = []
        self.Zs = []
        self.Ps = []
        self.d_max = d_max
        for d in range(d_max+1):
            X = [[Bool("X_{}_{}_{}".format(d, q, j)) for j in range(n_qubits)] for q in range(n_qubits)]
            Z = [[Bool("Z_{}_{}_{}".format(d, q, j)) for j in range(n_qubits)] for q in range(n_qubits)]
            P = [Bool("P_{}_{}".format(d, q)) for q in range(n_qubits)]
            self.Xs.append(X)
            self.Zs.append(Z)
            self.Ps.append(P)
        self.Svars, self.Hvars, self.CNOTvars = self.define_variables()
        self.constraints = []
        
    def apply_hadamard(self, qubit,Hvar,d):
        # Swap X and Z for the given qubit
        for i in range(self.n):
            self.constraints.append(Implies(Hvar, self.Ps[d+1][i] == self.Ps[d][i]^self.Xs[d][i][qubit]&self.Zs[d][i][qubit]))
            self.constraints.append(Implies(Hvar, self.Xs[d+1][i][qubit] == self.Zs[d][i][qubit]))
            self.constraints.append(Implies(Hvar, self.Zs[d+1][i][qubit]== self.Xs[d][i][qubit]))

    def apply_phase(self, qubit, PHASEvar,d):
        # S gate (phase gate) on a qubit
        for i in range(self.n):
            self.constraints.append(Implies(PHASEvar, self.Ps[d+1][i] == self.Ps[d][i]^(self.Zs[d][i][qubit]&self.Xs[d][i][qubit])))
            self.constraints.append(Implies(PHASEvar, self.Zs[d+1][i][qubit] ==  self.Zs[d][i][qubit]^self.Xs[d][i][qubit]))

    def apply_cnot(self, control, target, CNOTvar,d):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            self.constraints.append(Implies(CNOTvar, self.Ps[d+1][i] == self.Ps[d][i]^(self.Xs[d][i][control]&self.Zs[d][i][target]&(self.Xs[d][i][target]^self.Zs[d][i][control]^1))))
            self.constraints.append(Implies(CNOTvar, self.Xs[d+1][i][target] == self.Xs[d][i][target]^self.Xs[d][i][control]))
            self.constraints.append(Implies(CNOTvar, self.Xs[d+1][i][control] == self.Xs[d][i][control]))
            self.constraints.append(Implies(CNOTvar, self.Zs[d+1][i][control] == self.Zs[d][i][target]^self.Zs[d][i][control]))
            self.constraints.append(Implies(CNOTvar, self.Zs[d+1][i][target] == self.Zs[d][i][target]))

    def define_variables(self):
        """
        Defines the variables used in the Clliford solver.
        """
        Svars = []
        for q in range(self.n):
            Svars.append([Bool("S_{}_{}".format(q, d)) for d in range(self.d_max)])
        Hvars = []
        for q in range(self.n):
            Hvars.append([Bool("H_{}_{}".format(q, d)) for d in range(self.d_max)])

        CNOTvars = {}
        for q in range(self.n):
            CNOTvars[q] = {}
            for t in range(self.n):
                if q!= t:
                    CNOTvars[q][t] = [Bool("CNOT_{}_{}_{}".format(q, t, d)) for d in range(self.d_max)]

        return Svars, Hvars, CNOTvars
    def apply_gates(self):
        """
        Applies  gates 
        """
        for d in range(self.d_max):
            for q in range(self.n):
                self.apply_hadamard(q,self.Hvars[q][d])
                self.apply_phase(q,self.Svars[q][d])
                # self.constraints.append(Implies(self.Svars[q][d], self.stabilizer_table.apply_phase(q)))

        ## Apply CNOT gates
        for d in range(self.d_max):
            for q in range(self.n):
                for t in range(self.n):
                    if q!= t:
                        self.apply_cnot(q, t,self.CNOTvars[q][t][d])
    
    def unique_gate(self):
        """
        one depth and one qubit at a time
        """
        for d in range(self.d_max):
            for q in range(self.n):
                self.constraints.append(Sum([self.Hvars[q][d], self.Svars[q][d], *[self.CNOTvars[q][t][d] for t in range(self.n) if q!= t], *[self.CNOTvars[t][q][d] for t in range(self.n) if q!= t]]) == 1)
        
    def inverse_cancel(self):
        for q in range(self.n):
            for d in range(self.d_max-1):
                self.constraints.append(Implies(self.Hvars[q][d], Not(self.Hvars[q][d+1])))
                for t in range(self.n):
                    if q!= t:
                        self.constraints.append(Implies(self.CNOTvars[q][t][d], Not(self.CNOTvars[q][t][d+1])))
    
    def get_eq_expression(self,d,stabilizer_table:StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        expr = []
        for i in range(self.n):
            for j in range(self.n):
                expr.append(self.Xs[d][i][j] == bool(stabilizer_table.X[i][j]))
                expr.append(self.Zs[d][i][j] == bool(stabilizer_table.Z[i][j]))
            expr.append(self.Ps[d][i] == bool(stabilizer_table.P[i]))
        return And(*expr)
    def add_iout(self,input_stabilizer_table:StabilizerTable,output_stabilizer_table:StabilizerTable):
        """
        Adds the input and output stabilizer tables to the program.
        """
        input_expr= self.get_eq_expression(0,input_stabilizer_table)
        output_expr = self.get_eq_expression(self.d_max,output_stabilizer_table)
        self.constraints.append(Implies(input_expr, output_expr))
    
    
    


class CllifordCorrecter:
    def __init__(self, n_qubits,d_max:int = 4,BugProgram:List = None):
        self.solver = CllifordSolver(n_qubits,d_max)
        self.BugProgram = BugProgram
        self.n = n_qubits
        self.d_max = d_max
        self.Svars, self.Hvars, self.CNOTvars = self.solver.Svars, self.solver.Hvars, self.solver.CNOTvars
        pass
    
    def stabilizer_table_eq(self,stabilizer_table1:StabilizerTable,stabilizer_table2:StabilizerTable):
        """
        asserts that two stabilizer tables are equal.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.solver.constraints.append(stabilizer_table1.X[i][j]== int(stabilizer_table2.X[i][j]))
                self.solver.constraints.append(stabilizer_table1.Z[i][j]== int(stabilizer_table2.Z[i][j]))
            self.solver.constraints.append(stabilizer_table1.P[i]== int(stabilizer_table2.P[i]))

    def add_iout(self,input_stabilizer_table:StabilizerTable,output_stabilizer_table:StabilizerTable):
        """
        Adds the input and output stabilizer tables to the program.
        """
        self.solver.add_iout(input_stabilizer_table,output_stabilizer_table)
    
    def solve(self):
        """
        Solves the Clliford solver.
        """
        self.solver.unique_gate()
        self.solver.inverse_cancel()
        s = Solver()
        for c in self.solver.constraints:
            s.add(c)
        
        s.set("timeout", 10000)
        fix_program = CllifordProgram(self.n)
        print("Solving...")
        print(s.check())
        if s.check() == sat:
            m = s.model()
            for q in range(self.n):
                for d in range(self.d_max):
                    if m[self.Hvars[q][d]]:
                        fix_program.h(q)
                    if m[self.Svars[q][d]]:
                        fix_program.s(q)
                    for t in range(self.n):
                        if q!= t:
                            if m[self.CNOTvars[q][t][d]]:
                                fix_program.cnot(q,t)

            return fix_program
        else:
            return None
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
        self.append(['H',qubit])
        for _ in range(3):
            self.append(['S',qubit])
        self.append(['H',qubit])
        for _ in range(3):
            self.append(['S',qubit])

    def z(self, qubit):
        ## Z = HS = HSH
        for _ in range(2):
            self.append(['S',qubit])
    def cnot(self, control, target):
        self.append(['CNOT',(control,target)])

    def output_stablizers(self,input_StabilizerTable:StabilizerTable=None):
        """
        This function takes the input stabilizer table and outputs the stabilizers and destabilizers of the Clliford program.
        """
        output_StabilizerTable = StabilizerTable(self.n_qubits)
        for gate in self:
            if gate[0] == 'H':
                output_StabilizerTable.apply_hadamard(gate[1])
            elif gate[0] == 'S':
                output_StabilizerTable.apply_phase(gate[1])
            elif gate[0] == 'CNOT':
                output_StabilizerTable.apply_cnot(gate[1][0], gate[1][1])
            else:
                raise ValueError("Invalid gate type: {}".format(gate[0]))
        if input_StabilizerTable is None:
            return output_StabilizerTable
        else:
            input_StabilizerTable.apply_clifford(output_StabilizerTable)
            return input_StabilizerTable
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
            if inst.name == "h":
                program.h(qargs[0].index)
            elif inst.name == "s":
                program.s(qargs[0].index)
            elif inst.name == "sdg":
                program.sdg(qargs[0].index)
            elif inst.name == "x":
                program.x(qargs[0].index)
            elif inst.name == "y":
                program.y(qargs[0].index)
            elif inst.name == "z":
                program.z(qargs[0].index)
            elif inst.name == "cz":
                program.cz(qargs[0].index, qargs[1].index)
            elif inst.name == "cx" or inst.name == "cnot":
                program.cnot(qargs[0].index, qargs[1].index)
            else:
                raise ValueError("Invalid gate type: {}".format(inst.name))
        return program

def generate_input_stabilizer_table(n_qubits:int):
    """
    Generates a random Clliford program with the given number of qubits and depth.
    """
    
    circuit = random_clifford(n_qubits).to_circuit()
    program = CllifordProgram.from_circuit(circuit)
    return program.output_stablizers()

def generate_inout_stabilizer_tables(n_qubits:int,program: CllifordProgram):
    """
    Generates the input and output stabilizer tables for the given Clliford program.
    """
    input_stabilizer_table = generate_input_stabilizer_table(n_qubits)
    output_stabilizer_table = program.output_stablizers(input_stabilizer_table)
    return input_stabilizer_table, output_stabilizer_table

def check_circuit_equality(circuit1: QuantumCircuit, circuit2: QuantumCircuit):
    """
    Checks if two circuits are equal.
    """
    if circuit1.num_qubits != circuit2.num_qubits:
        return False
    ## check their unitary matrix
    backend = Aer.get_backend('statevector_simulator')
    qc = circuit1.compose(circuit2.inverse())
    job = execute(qc, backend)
    output_state = job.result().get_statevector(qc)
    print(output_state)
    zeros = np.zeros(2**n_qubits)
    zeros[0] = 1
    ## calculate the fidelity between the two circuits
    fidelity = np.abs(np.dot(output_state,zeros))**2
    print(f'Fidelity: {fidelity}')
    if 1-fidelity < 1e-3:
        return True
    else:
        return False
        
    


if __name__ == "__main__":
    
    
    
    
            
    # # check the stabilizer table is right

    # from qiskit.quantum_info import random_clifford,StabilizerState,Clifford, StabilizerTable as qiskit_StabilizerTable
    # qc = QuantumCircuit(2)
    # qc.h(0)
    # qc.s(0)
    # qc.s(0)
    # qc.cx(0,1)
    # qc.h(0)
    # qc.h(1)
    # qc.cx(1,0)
    
    # # qc.cx(1, 0)
    # # qc.reverse_bits()
    # cliff = Clifford(qc)

    #     # Print the Clifford
    # print(cliff)

    # # Print the Clifford stabilizer rows
    # print(cliff.to_labels(mode="S"))
    # stab = StabilizerState(qc)
    # print(stab)
    # stab_table = StabilizerTable(2)
    # stab_table.apply_hadamard(0)
    # stab_table.apply_phase(0)
    # stab_table.apply_phase(0)
    # stab_table.apply_cnot(0,1)
    # stab_table.apply_hadamard(0)
    # stab_table.apply_hadamard(1)
    # stab_table.apply_cnot(1,0)
    
    
    # # stab_table.apply_cnot(1,0)
    # print(stab_table.table)
    # print(stab_table.to_string())
    # # exit()
    # program = CllifordProgram.from_circuit(qc)
    # outstablizer = program.output_stablizers()
    # print(outstablizer.table)
    # print(outstablizer.to_string())
    # exit()
    n_qubits = 2
    d_max = 1
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    circuit.s(1)
    print(circuit)
    program = CllifordProgram.from_circuit(circuit)
    wrong_program = program.copy()
    wrong_program.pop()
    # wrong_program.pop()
    correcter = CllifordCorrecter(n_qubits,d_max,wrong_program)
    for _ in range(8):
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(n_qubits,program)
        correcter.add_iout(input_stabilizer_table, output_stabilizer_table)
    fix_program = correcter.solve()
    print(fix_program.to_circuit())
    wrong_program.extend(fix_program)
    # print(wrong_program)
    fix_circuit = wrong_program.to_circuit()
    print(fix_circuit)
    print(circuit)
    print(check_circuit_equality(circuit, fix_circuit))

    








    
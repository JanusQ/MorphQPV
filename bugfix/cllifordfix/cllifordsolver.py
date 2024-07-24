import numpy as np
from z3 import *
from typing import List, Tuple

def xor(x, y):
    """
    Computes the bitwise XOR of two boolean arrays.
    """
    return (1-x)*y + (1-y)*x

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
        


    def apply_cnot(self, control, target):
        # Apply CNOT gate from control to target
        self.X[:, target] ^= self.X[:, control]
        self.Z[:, control] ^= self.Z[:, target]
        self.P ^= (self.X[:,control] & self.Z[:,target] & np.logical_not(self.X[:,target] ^ self.Z[:,control]))

    def apply_phase(self, qubit):
        # S gate (phase gate) on a qubit
        self.Z[:, qubit] ^= self.X[:, qubit]
        self.P ^= self.X[:,qubit] & self.Z[:,qubit]

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
    


class StabilizerTableVar:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.X = []
        self.Z = []
        self.P = []
        for i in range(n_qubits):
            self.X.append([0 for j in range(n_qubits)])
            self.Z.append([0 for j in range(n_qubits)])
            self.P.append(0)
        for i in range(n_qubits):
            self.Z[i][i] = 1  # Z on each qubit

    @staticmethod
    def from_table(stabilizer_table: StabilizerTable):
        """
        Clones the stabilizer table.
        """
        new_table = StabilizerTableVar(stabilizer_table.n)
        for i in range(stabilizer_table.n):
            for j in range(stabilizer_table.n):
                new_table.X[i][j] = int(stabilizer_table.X[i][j])
                new_table.Z[i][j] = int(stabilizer_table.Z[i][j])
            new_table.P[i] = int(stabilizer_table.P[i])
        return new_table

        
    def apply_hadamard(self, qubit,Hvar):
        # Swap X and Z for the given qubit
        for i in range(self.n):
            temp = self.X[i][qubit] * self.Z[i][qubit] *Hvar 
            self.P[i]  =  xor(self.P[i], temp)
            self.X[i][qubit]= self.Z[i][qubit] * Hvar + self.X[i][qubit] * Not(Hvar)
            self.Z[i][qubit]= self.X[i][qubit] * Hvar + self.Z[i][qubit] * Not(Hvar)
            


    def apply_cnot(self, control, target, CNOTvar):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            self.X[i][target] =  xor(self.X[i][target], self.X[i][control] * CNOTvar)
            self.Z[i][control] =  xor(self.Z[i][control], self.Z[i][target] * CNOTvar)
            temp = (1- xor(self.X[i][control] , self.X[i][target]))*self.Z[i][target]*self.X[i][control]*CNOTvar
            self.P[i] = xor(self.P[i], temp)

    def apply_phase(self, qubit, PHASEvar):
        # S gate (phase gate) on a qubit
        for i in range(self.n):
            self.Z[i][qubit] =  xor(self.Z[i][qubit], self.X[i][qubit] * PHASEvar)
            self.P[i] =  xor(self.P[i], self.X[i][qubit] * self.Z[i][qubit] * PHASEvar)

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
    def __init__(self, n_qubits,d_max:int = 4):
        self.n = n_qubits
        self.stabilizer_table = StabilizerTableVar(n_qubits)
        self.d_max = d_max
        self.Svars, self.Hvars, self.CNOTvars = self.define_variables()
        self.constraints = []
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
                self.stabilizer_table.apply_hadamard(q,self.Hvars[q][d])
                self.stabilizer_table.apply_phase(q,self.Svars[q][d])
                # self.constraints.append(Implies(self.Svars[q][d], self.stabilizer_table.apply_phase(q)))

        ## Apply CNOT gates
        for d in range(self.d_max):
            for q in range(self.n):
                for t in range(self.n):
                    if q!= t:
                        self.stabilizer_table.apply_cnot(q, t,self.CNOTvars[q][t][d])
    
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


class CllifordCorrecter:
    def __init__(self, n_qubits,d_max:int = 4,BugProgram:List = None):
        self.solver = CllifordSolver(n_qubits,d_max)
        self.BugProgram = BugProgram
        self.n = n_qubits
        self.d_max = d_max
        self.Svars, self.Hvars, self.CNOTvars = self.solver.Svars, self.solver.Hvars, self.solver.CNOTvars
        pass
    def apply_program_on_input(self,input_stabilizer_table: StabilizerTable,program: List):
        """
        Applies the program on the input stabilizer table.
        """
        for gate in program:
            if gate[0] == "H":
                q = int(gate[1])
                input_stabilizer_table.apply_hadamard(q)
            elif gate[0] == "S":
                q = int(gate[1])
                input_stabilizer_table.apply_phase(q)
            elif gate[0] == "CNOT":
                c, t =  gate[1]
                c, t = int(c), int(t)
                input_stabilizer_table.apply_cnot(c, t)
            else:
                raise ValueError("Invalid gate: {}".format(gate))
        return input_stabilizer_table
    
    def stabilizer_table_eq(self,stabilizer_table1:StabilizerTableVar,stabilizer_table2:StabilizerTable):
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
        temp_stabilizer_table = self.apply_program_on_input(input_stabilizer_table,self.BugProgram)
        self.solver.stabilizer_table = StabilizerTableVar.from_table(temp_stabilizer_table)
        self.solver.apply_gates()

        self.stabilizer_table_eq(self.solver.stabilizer_table,output_stabilizer_table)
    
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

        # s.set("parallel.enable", True)
        # s.set("parallel.threads", 50)
        print("Solving...")
        print(s.check())
        if s.check() == sat:
            m = s.model()
            for q in range(self.n):
                for d in range(self.d_max):
                    if m[self.Hvars[q][d]].as_long() == 1:
                        self.stabilizer_table.apply_hadamard(q)
                    if m[self.Svars[q][d]].as_long() == 1:
                        self.stabilizer_table.apply_phase(q)
                    for t in range(self.n):
                        if q!= t:
                            if m[self.CNOTvars[q][t][d]].as_long() == 1:
                                self.stabilizer_table.apply_cnot(q, t)


class CllifordProgram(list):
    def __init__(self, n_qubits:int):
        super().__init__()
        self.n_qubits = n_qubits
    def __setitem__(self, index, item):
        super().__setitem__(index, str(item))

    def insert(self, index, item):
        super().insert(index, str(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(str(item) for item in other)
    def h(self, qubit):
        self.append(['H',qubit])
    def s(self, qubit):
        self.append(['S',qubit])
    def sdg(self, qubit):
        for _ in range(3):
            self.append(['S',qubit])
    
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
   

def generate_input_stabilizer_table(n_qubits:int,d_max:int = 10):
    """
    Generates a random Clliford program with the given number of qubits and depth.
    """
    program = CllifordProgram(n_qubits)
    for _ in range(d_max):
        gate_type = np.random.choice(["H", "S", "CNOT"])
        if gate_type == "H":
            qubit = np.random.randint(n_qubits)
            program.h(qubit)
        elif gate_type == "S":
            qubit = np.random.randint(n_qubits)
            program.s(qubit)
        elif gate_type == "CNOT":
            control = np.random.randint(n_qubits)
            target = np.random.randint(n_qubits)
            while control == target:
                target = np.random.randint(n_qubits)
            program.cnot(control, target)
    return program.output_stablizers()

def generate_inout_stabilizer_tables(n_qubits:int,program:List,d_max:int = 10):
    """
    Generates the input and output stabilizer tables for the given Clliford program.
    """
    input_stabilizer_table = generate_input_stabilizer_table(n_qubits,d_max)
    output_stabilizer_table = program.output_stablizers(input_stabilizer_table)
    return input_stabilizer_table, output_stabilizer_table


if __name__ == "__main__":
    n_qubits = 2
    d_max = 4
    program = CllifordProgram(n_qubits)
    program.h(0)
    program.h(1)
    # program.h(2)
    program.cnot(0,1)
    # program.cnot(1,2)
    # program.cnot(0,2)
    program.y(0)
    program.z(1)
    program.cnot(0,1)
    # program.cnot(1,2)
    # program.cnot(0,2)
    wrong_program = program.copy()
    wrong_program.pop()
    wrong_program.pop()
    correcter = CllifordCorrecter(n_qubits,d_max,wrong_program)
    for _ in range(1):
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(n_qubits,program,d_max)
        correcter.add_iout(input_stabilizer_table,output_stabilizer_table)
    correcter.solve()

    








    
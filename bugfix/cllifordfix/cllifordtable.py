import numpy as np
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
        self.X[:, qubit], self.Z[:, qubit] = self.Z[:, qubit].copy(), self.X[:, qubit].copy()
        # P \xor X[qubit]*Z[qubit]
        self.P ^= (self.X[:,qubit] & self.Z[:,qubit])

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
    
from qiskit import QuantumCircuit
import numpy as np


class HamiltonianSimulation:
    """Quantum benchmark focused on the ability to simulate 1D
    Transverse Field Ising Models (TFIM) of variable length.

    Device performance is based on how closely the experimentally obtained
    average magnetization (along the Z-axis) matches the noiseless value.
    Since the 1D TFIM is efficiently simulatable with classical algorithms,
    computing the noiseless average magnetization remains scalable over a large
    range of benchmark sizes.
    """

    def __init__(self, width: int, time_step: int = 1, total_time: int = 1) -> None:
        """Args:
        n_qubits: int
            Size of the TFIM chain, equivalent to the number of qubits.
        time_step: int
            Size of the timestep in attoseconds.
        total_time:
            Total simulation time of the TFIM chain in attoseconds.
        """
        self.n_qubits = width
        self.time_step = time_step
        self.total_time = total_time
        self.circuit =  QuantumCircuit(self.n_qubits)

    def gen_circuit(self) -> QuantumCircuit:
        """Generate a self.circuit to simulate the evolution of an n-qubit TFIM
        chain under the Hamiltonian:

        H(t) = - Jz * sum_{i=1}^{n-1}(sigma_{z}^{i} * sigma_{z}^{i+1})
               - e_ph * cos(w_ph * t) * sum_{i=1}^{n}(sigma_{x}^{i})

        where,
            w_ph: frequency of E" phonon in MoSe2.
            e_ph: strength of electron-phonon coupling.
        """
        hbar = 0.658212  # eV*fs
        jz = (
            hbar * np.pi / 4
        )  # eV, coupling coeff; Jz<0 is antiferromagnetic, Jz>0 is ferromagnetic
        freq = 0.0048  # 1/fs, frequency of MoSe2 phonon

        w_ph = 2 * np.pi * freq
        e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

       

        # Build up the circuit over total_time / time_step propagation steps
        for step in range(int(self.total_time / self.time_step)):
            # Simulate the Hamiltonian term-by-term
            t = (step + 0.5) * self.time_step

            # Single qubit terms
            psi = -2.0 * e_ph * np.cos(w_ph * t) * self.time_step / hbar
            for qubit in range(self.n_qubits):
                self.circuit.h(qubit)
                self.circuit.rz(psi, qubit)
                self.circuit.h(qubit)

            # Coupling terms
            psi2 = -2.0 * jz * self.time_step / hbar
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
                self.circuit.rz(psi2, i + 1)
                self.circuit.cx(i, i + 1)

        return self.circuit

from typing import  Optional, Union, Iterable, List
import numpy as np
from qiskit.quantum_info.random import random_clifford
from qiskit import QuantumCircuit
class StandardRB:
    """An experiment to characterize the error rate of a gate set on a device.

    # section: overview

    Randomized Benchmarking (RB) is an efficient and robust method
    for estimating the average error rate of a set of quantum gate operations.
    See `Qiskit Textbook
    <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`_
    for an explanation on the RB method.

    A standard RB experiment generates sequences of random Cliffords
    such that the unitary computed by the sequences is the identity.
    After running the sequences on a backend, it calculates the probabilities to get back to
    the ground state, fits an exponentially decaying curve, and estimates
    the Error Per Clifford (EPC), as described in Refs. [1, 2].

    .. note::
        In 0.5.0, the default value of ``optimization_level`` in ``transpile_options`` changed
        from ``0`` to ``1`` for RB experiments. That may result in shorter RB circuits
        hence slower decay curves than before.

    # section: analysis_ref
        :class:`RBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887
    """
    def __init__(
        self,
        n_qubits: int,
        lengths: Iterable[int]=None,
        n_samples: int = 3,
        seed: Optional[Union[int,Iterable]] = None,
        full_sampling: Optional[bool] = False
    ):
        """Initialize a standard randomized benchmarking experiment.

        Args:
            physical_qubits: List of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            n_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for all lengths.
                           If False for sample of lengths longer sequences are constructed
                           by appending additional samples to shorter sequences.
                           The default is False.

        Raises:
            Error: If any invalid argument is supplied.
        """
        if lengths is None:
            lengths = [10, 20, 50, 75, 100, 125, 150, 175, 200]
        # Verify parameters
        if any(length <= 0 for length in lengths):
            raise ValueError(
                f"The lengths list {lengths} should only contain " "positive elements."
            )
        if len(set(lengths)) != len(lengths):
            raise ValueError(
                f"The lengths list {lengths} should not contain " "duplicate elements."
            )
        if n_samples <= 0:
            raise ValueError(f"The number of samples {n_samples} should " "be positive.")
        self.n_qubits = n_qubits
        self.physical_qubits = list(range(n_qubits))
        self.seed = seed
        self.lengths = lengths
        self.full_sampling = full_sampling
        self.n_samples = n_samples

    def __sample_sequence(self, length: int, rng: Iterable) -> List:
        # Return circuit object instead of Clifford object for 3 or more qubits case for speed
        return [random_clifford(self.n_qubits, rng).to_circuit() for _ in range(length)]

    def _sample_sequences(self):
        """Sample RB sequences

        Returns:
            A list of RB sequences.
        """
        rng = np.random.default_rng(self.seed)
        sequences = []
        if self.full_sampling:
            for _ in range(self.n_samples):
                for length in self.lengths:
                    sequences.append(self.__sample_sequence(length, rng))
        else:
            for _ in range(self.n_samples):
                longest_seq = self.__sample_sequence(max(self.lengths), rng)
                for length in self.lengths:
                    sequences.append(longest_seq[:length])

        return sequences
    
    def sample_circuits(self,
        ) -> Iterable[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        for seq in self._sample_sequences():
            circ = QuantumCircuit(self.n_qubits)
            for elem in seq:
                circ.append(elem.to_instruction(), circ.qubits)
            # Compute inverse, compute only the difference from the previous shorter sequence
            for elem in seq[::-1]:
                circ.append(elem.to_instruction(), circ.qubits)
            circ.metadata = {
                "xval": self.n_samples*len(self.lengths),
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
            }
            yield circ
    
    def gen_circuit(self,length):
        seq = self.__sample_sequence(length, np.random.default_rng(self.seed))
        circ = QuantumCircuit(self.n_qubits)
        for elem in seq:
            circ.append(elem.to_instruction(), circ.qubits)
        for elem in seq[::-1]:
            circ.append(elem.to_instruction(), circ.qubits)
        circ.metadata = {
            "length": length,
            "group": "Clifford",
            "physical_qubits": self.physical_qubits,
        }
        return circ
        
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit.library import TwoLocal

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

class TwoLocalRandom(TwoLocal):
    """TwoLocal with random parameter values."""

    def __init__(self, n_qubits: int, reps: int = 3) -> None:
        super().__init__(n_qubits, reps=reps)
        np.random.seed(10)
        n_params = self.n_parameters
        self._params = 2 * np.pi * np.random.rand(n_params)

    def gen_circuit(self) -> QuantumCircuit:
        return self.bind_parameters(self._params)

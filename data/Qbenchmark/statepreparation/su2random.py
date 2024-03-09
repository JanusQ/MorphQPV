from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit.library import EfficientSU2

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

class EfficientSU2Random(EfficientSU2):
    """EfficientSU2 with random parameter values."""

    def __init__(self, num_qubits: int, reps: int = 3) -> None:
        super().__init__(num_qubits, reps=reps)
        np.random.seed(10)
        num_params = self.num_parameters
        self._params = 2 * np.pi * np.random.rand(num_params)

    def gen_circuit(self) -> QuantumCircuit:
        return self.bind_parameters(self._params)
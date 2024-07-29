# inspired from https://qiskit.org/ecosystem/machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

class QNN:
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits

    def gen_circuit(self) -> QuantumCircuit:
        """Returns a quantum circuit implementing a Quantum Neural Network (QNN) with a ZZ FeatureMap and a RealAmplitudes ansatz.

        Keyword arguments:
        n_qubits -- number of qubits of the returned quantum circuit
        """
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits)
        ansatz = RealAmplitudes(n_qubits=self.n_qubits, reps=1)

        qc = QuantumCircuit(self.n_qubits)
        feature_map = feature_map.bind_parameters([1 for _ in range(feature_map.n_parameters)])
        ansatz = ansatz.bind_parameters(np.random.rand(ansatz.n_parameters))
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qc.name = "qnn"
        qc.measure_all()
        return qc

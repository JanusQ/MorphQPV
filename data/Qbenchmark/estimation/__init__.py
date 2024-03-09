from .amplitude_estimation import Amplitudeestimation
class QuantumPhaseEstimation:
    def __init__(self,num_qubits) -> None:
        self.num_qubits= num_qubits
    def gen_circuit(self,type='exact'):
        if type == 'exact':
            from .quantum_phase_estimation_exact import create_circuit
            return create_circuit(self.num_qubits)
        elif type == 'inexact':
            from .quantum_phase_estimation_inexact import create_circuit    
            return create_circuit(self.num_qubits)
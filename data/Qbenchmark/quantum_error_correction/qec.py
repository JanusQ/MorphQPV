from qiskit.circuit import QuantumCircuit, QuantumRegister,ClassicalRegister
class QEC(QuantumCircuit):
    def __init__(self,m,k):
        qubits = QuantumRegister(m * k)
        ancilla_qubits = QuantumRegister((m-1) * k)
        cbits = ClassicalRegister((m-1) * k)
        circuit = QuantumCircuit(qubits, cbits,name=f"Qec_{m}_xyz_{k}")
        circuit.add_register(ancilla_qubits)
        super().__init__(*circuit.qregs, name=circuit.name)
        for i, j, k in zip(range(0, len(qubits), m), range(0, len(ancilla_qubits), m-1), range(0, len(cbits), m-1)):
            self.state_preparation( qubits[i:i + m], cbits[k:k + m-1])
            self.z_error_detection( qubits[i:i + m], ancilla_qubits[j:j + m-1], cbits[k:k + m-1])
            self.x_error_detection( qubits[i:i + m], ancilla_qubits[j:j + m-1], cbits[k:k + m-1])
    def state_preparation(self, qubits, cbits):
        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cnot(qubits[0], qubits[i])
    def z_error_detection(self, qubits, ancilla_qubits, cbits):
        for i in range(len(qubits) - 1):
            self.cnot(qubits[i], qubits[i + 1])
        for i in range(len(qubits) - 1):
            self.cnot(qubits[i], ancilla_qubits[i])
        for i in range(len(qubits) - 1):
            self.cnot(qubits[i + 1], ancilla_qubits[i])
        for i in range(len(qubits) - 1):
            self.cnot(qubits[i], qubits[i + 1])
        # for i in range(len(qubits) - 1):
        #     self.measure(ancilla_qubits[i], cbits[i])
    def x_error_detection(self, qubits, ancilla_qubits, cbits):
        for i in range(len(qubits) - 1):
            self.h(qubits[i])
            self.h(ancilla_qubits[i])
            self.cnot(qubits[i], qubits[i + 1])
            self.cnot(ancilla_qubits[i], qubits[i + 1])
            self.h(qubits[i])
            self.h(ancilla_qubits[i])
        # for i in range(len(qubits) - 1):
        #     self.measure(ancilla_qubits[i], cbits[i])
    def gen_circuit(self):
        return self
        
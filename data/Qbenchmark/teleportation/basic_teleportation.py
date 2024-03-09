## Programming Quantum Computers
##   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
##   O'Reilly Media
##
## More samples like this can be found at http://oreilly-qc.github.io
##
## A complete notebook of all Chapter 4 samples (including this one) can be found at
##  https://github.com/oreilly-qc/oreilly-qc.github.io/tree/master/samples/Qiskit

from typing import Dict, Optional, Sequence, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ, BasicAer
import math

from qiskit.circuit.bit import Bit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.register import Register
from typing import Dict, Optional, Sequence, Union
## Uncomment the next line to see diagrams when running in a notebook
#%matplotlib inline

class TeleportationCircuit(QuantumCircuit):
    def __init__(self, *regs: Union[Register,int,Sequence[Bit]], name: Union[str, None]= None, global_phase: ParameterValueType = 0, metadata:Union[Dict,None] = None):
        super().__init__(*regs, name=name, global_phase=global_phase, metadata=metadata)
        self._initialize(*regs)
        self._teleport(*regs)
        self.barrier()
        self._verify(*regs)
        self.barrier()
        self._correct(*regs)
        self.barrier()
        self.measure_all()
    def _initialize(self, alice: QuantumRegister, ep: QuantumRegister, bob: QuantumRegister):
        self.h(ep)
        self.cx(ep, bob)
    def _teleport(self, alice: QuantumRegister, ep: QuantumRegister, bob: QuantumRegister):
        self.cx(alice, ep)
        self.h(alice)
    def _verify(self, alice: QuantumRegister, ep: QuantumRegister, bob: QuantumRegister):
        self.h(bob)
        self.rz(math.radians(-45), bob)
        self.h(bob)
    def _correct(self, alice: QuantumRegister, ep: QuantumRegister, bob: QuantumRegister):
        self.cx(ep, bob)
        self.cz(alice, bob)
    def get_circuit(self):
        return self
    

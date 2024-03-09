import random
from .arithmetic import RCAdder, multiplier,QuantumCounting,Shor,SquareRoot
from .Dynamics import Dynamics,Qwalk
from .estimation import amplitude_estimation,QuantumPhaseEstimation
from .finance import portfolioqaoa,portfoliovqe,pricingcall,pricingput
from .machinelearning import QCNN,QNN,QKNN,QSVM,QuGAN
from .nature import GroundStateEstimation,Ising
from .optimization import TSP,QAOA,Routing
from .teleportation import TeleportationCircuit
from .Supremacy import Qgrid
from .qrams import gen_circuit as gen_qram_circuit
from .qrams import QROM
from .statepreparation import GHZ,WState,RealAmplitudeRandom,TwoLocalRandom,EfficientSU2Random,Clifford
from .quantum_error_correction.qec import QEC
from .search import BernsteinVazirani,DeutschJozsa,Grover,PhaseKickback,Simon
from .QFT import QFT
from .circuit_converter import qiskit_circuit_to_layer_cirucit
from .randomized_benchmarking.random_bench import StandardRB
from qiskit import transpile,QuantumCircuit
import numpy as np

def get_bitstr(n_qubits):
    b = ""
    for i in range(n_qubits):
        if random.random() > 0.5:
            b += '0'
        else:
            b += '1'
    return b
def get_data(name, qiskit_circuit,basis_gates=['u1', 'u3', 'cx', 'id','rz','x','h','cz','mcz','ctrl','cond_x','cond_z','initialize','unitary','channel','measure']):
    transpile_circuit = transpile(qiskit_circuit, basis_gates=basis_gates)
    transpile_circuit.name = name
    return transpile_circuit
def get_clliford_data(name, qiskit_circuit,basis_gates=['h', 's', 't','tdg' 'cnot','swap']):
    transpile_circuit = transpile(qiskit_circuit, basis_gates=basis_gates)
    transpile_circuit.name = name
    return transpile_circuit
def layer_circuit_generator(name, n_qubits,**kwargs):
    if name == "basis":
        return standard_state_prepare(n_qubits)
    return qiskit_circuit_to_layer_cirucit(circuit_generator(name, n_qubits,**kwargs))

def standard_state_prepare(n_qubits):
    gates = [[{'name':'x','qubits':list(np.random.choice(n_qubits,size=np.random.randint(0,n_qubits),replace=False))}]]
    return gates
def circuit_generator(name, n_qubits,**kwargs):
    if name.lower() == "bv":
        return get_data(f'BV_{n_qubits}', BernsteinVazirani(get_bitstr(n_qubits-1)).gen_circuit())
    if name.lower() == "qram":
        return get_data(f'QRAM_{n_qubits}', QROM(n_qubits, [random.random() for i in range(2**n_qubits)]).gen_circuit())
    if name.lower() == "clliford":
        return get_data(f'clliford_{n_qubits}', Clifford(n_qubits).gen_circuit())
    if name.lower() == "adder":
        return get_data(f'adder_{n_qubits}', RCAdder(n_qubits).gen_circuit())
    if name.lower() == "wstate":
        return get_data(f'wstate_{n_qubits}', WState(n_qubits).gen_circuit())
    if name.lower() == "qft":
        return get_data(f'qft_{n_qubits}', QFT(n_qubits,approximation_degree=0))
    if name.lower() == "qft":
        return get_data(f'qft_{n_qubits}', QFT(n_qubits,approximation_degree=0))
    if name.lower() == "qknn":
        return get_data(f'qknn_{n_qubits}', QKNN(n_qubits).gen_circuit())
    if name.lower() == "qsvm":
        return get_data(f'qsvm_{n_qubits}', QSVM(n_qubits).gen_circuit())
    if name.lower() == "vqc":
        return get_data(f'vqc_{n_qubits}', QNN(n_qubits).gen_circuit())
    if name.lower() == "qaoa":
        return get_data(f'qaoa_{n_qubits}', QAOA(n_qubits).gen_circuit())
    if name.lower() == "qnn":
        return get_data(f'qnn_{n_qubits}', QNN(n_qubits).gen_circuit())
    if name.lower() == "dynamics":
        return get_data(f'dynamics_{n_qubits}', Dynamics(n_qubits).gen_circuit())
    if name.lower() == "qwalk":
        return get_data(f'qwalk_{n_qubits}', Qwalk(n_qubits).gen_circuit())
    if name.lower() == "supremacy":
        return get_data(f'supremacy_{n_qubits}', Qgrid(n_qubits, 10, n_qubits*10).gen_circuit())
    if name.lower() == "teleportation":
        return get_data(f'teleportation_{n_qubits}', TeleportationCircuit(n_qubits).gen_circuit())
    if name.lower() == "multiplier":
        return get_data(f'multiplier_{n_qubits}', multiplier(n_qubits).gen_circuit())
    if name.lower() == "ghz":
        return get_data(f'ghz_{n_qubits}', GHZ(n_qubits).gen_circuit())
    if name.lower() == "simon":
        return get_data(f'simon_{n_qubits}', BernsteinVazirani(get_bitstr(n_qubits-1)).gen_circuit())
    if name.lower() == "square_root":
        return get_data(f'square_root_{n_qubits}', SquareRoot(n_qubits).gen_circuit())
    if name.lower() == "deutsch_jozsa" or name.lower() == "dj":
        return get_data(f'deutsch_jozsa_{n_qubits}', DeutschJozsa(get_bitstr(n_qubits-1)).gen_circuit())
    if name.lower() == "quantum_counting":
        return get_data(f'quantum_counting_{n_qubits}', QuantumCounting(n_qubits).gen_circuit())
    if name.lower() == "phase_kickback":
        return get_data(f'phase_kickback_{n_qubits}', PhaseKickback(n_qubits).gen_circuit())
    if name.lower() == "qugan":
        return get_data(f'qugan_{n_qubits}', QuGAN(n_qubits).gen_circuit())
    if name.lower() == "ising":
        return get_data(f'ising_{n_qubits}', Ising(n_qubits).gen_circuit())
    if name.lower() == "shor":
        return get_data(f'shor_{n_qubits}', QFT(n_qubits,approximation_degree=0))
    if name.lower() == "two_local_random":
        return get_data(f'two_local_random_{n_qubits}', TwoLocalRandom(n_qubits).gen_circuit())
    if name.lower() == "su2_random":
        return get_data(f'su2_random_{n_qubits}', EfficientSU2Random(n_qubits).gen_circuit())
    if name.lower() == "real_amplitude_random":
        return get_data(f'real_amplitude_random_{n_qubits}', RealAmplitudeRandom(n_qubits).gen_circuit())
    if name.lower() == "ground_state_estimation":
        return get_data(f'ground_state_estimation_{n_qubits}', GroundStateEstimation(n_qubits).gen_circuit())
    if name.lower() == "portfolio_vqe":
        return get_data(f'portfolio_vqe_{n_qubits}', portfoliovqe.create_circuit(n_qubits))
    if name.lower() == "portfolio_qaoa":
        return get_data(f'portfolio_qaoa_{n_qubits}', portfolioqaoa.create_circuit(n_qubits))
    if name.lower() == "pricing_call":
        return get_data(f'pricing_call_{n_qubits}', pricingcall.create_circuit(n_qubits))
    if name.lower() == "pricing_put":
        return get_data(f'pricing_put_{n_qubits}', pricingput.create_circuit(n_qubits))
    if name.lower() == "routing":
        return get_data(f'routing_{n_qubits}', Routing(n_qubits).gen_circuit())
    if name.lower() == "tsp":
        return get_data(f'tsp_{n_qubits}', TSP(n_qubits).gen_circuit())
    if name.lower() == "qpe":
        return get_data(f'qpe_{n_qubits}', QuantumPhaseEstimation(n_qubits).gen_circuit())
    if name.lower() == "amplitude_estimation":
        return get_data(f'amplitude_estimation_{n_qubits}', amplitude_estimation(n_qubits).gen_circuit())
    if name.lower() == "qcnn":
        return get_data(f'qcnn_{n_qubits}', QCNN(n_qubits).gen_circuit())
    if name.lower() == "grover":
        return get_data(f'grover_{n_qubits}', Grover(n_qubits).gen_circuit())
    if name.lower() == "simon":
        return get_data(f'simon_{n_qubits}', Simon(n_qubits).gen_circuit())
    if name.lower() == "xeb":
        if 'length' in kwargs:
            length = kwargs['length']
        else:
            length = 1
        return get_data(f'XEB_{n_qubits}_{length}', StandardRB(n_qubits).gen_circuit(length=length))
    if name.lower() == "rb":
        if 'length' in kwargs:
            length = kwargs['length']
        else:
            length = 1
        return get_data(f'rb_{n_qubits}_{length}', StandardRB(n_qubits).gen_circuit(length=length))
    if name.lower() == "qec":
        if 'm' in kwargs and 'k' in kwargs:
            m = kwargs['m']
            k = kwargs['k']
            assert (2*m-1)*k == n_qubits
        else:
            m = n_qubits//2+1
            k = 1
        return get_data(f'qec_{n_qubits}_{m}_{k}', QEC(m,k).gen_circuit())

def get_bitstr(n_qubits):
    b = ""
    for i in range(n_qubits):
        if random.random() > 0.5:
            b += '0'
        else:
            b += '1'
    return b

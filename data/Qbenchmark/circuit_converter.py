import pennylane as qml

from qiskit import QuantumCircuit
import copy

from qiskit.circuit import ClassicalRegister
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

def layer_circuit_to_qml_circuit(layer_cirucit):
    for layer in layer_cirucit:
        for gate in layer:
            gate_qubits = [q for q in gate['qubits']]
            if gate['name'] == 'u':
                theta, phi, lam = gate['params']  # TODO: 这个地方的参数位置需要检查一下
                qml.U3(theta, phi, lam, wires=gate_qubits)
            elif gate['name'] == 'h':
                qml.Hadamard(wires=gate_qubits)
            elif gate['name'] == 'x':
                qml.PauliX(wires=gate_qubits)
            elif gate['name'] == 'y':
                qml.PauliY(wires=gate_qubits)
            elif gate['name'] == 'z':
                qml.PauliZ(wires=gate_qubits)
            elif gate['name'] == 'rz':
                qml.RZ(gate['params'][0], wires=gate_qubits)
            elif gate['name'] == 'cx':
                qml.CNOT(wires=gate_qubits)
            elif gate['name'] == 'cz':
                qml.CZ(wires=gate_qubits)
            elif gate['name'] == 'cond_x':
                qml.ctrl(qml.PauliX, control=gate_qubits[0], wires=gate_qubits[1])
            elif gate['name'] == 'cond_z':
                qml.ctrl(qml.PauliZ, control=gate_qubits[0], wires=gate_qubits[1])
            elif gate['name'] == 'mcz':
                qml.MultiControlledZ(wires=gate_qubits[:-1], control_wires=gate_qubits[-1])
            elif gate['name'] == 'initialize':
                qml.QubitStateVector(gate['params'], wires=gate_qubits)
            elif gate['name'] == 'unitary':
                unitary = gate['params']
                qml.QubitUnitary(unitary, wires=gate_qubits)
            elif gate['name'] == 'wirecut':
                pass # 啥都不干
            elif gate['name'] == 'ctrl':
                operation = gate['params']
                operation = qml.QubitUnitary(operation, wires=gate['ctrled_qubits'])
                qml.ctrl(operation, control=gate['ctrl_qubits'])
            elif gate['name'] == 'measure':
                qml.sample(qml.PauliZ(wires=gate_qubits))
            elif gate['name'] == 'channel':
                channel = gate['params']
                qml.QubitChannel(channel, wires=gate_qubits)
            else:
                raise Exception('Unkown gate type', gate)
            
def layer_circuit_to_qiskit_circuit(layer_cirucit,N_qubit):
    qiskit_circuit = QuantumCircuit(N_qubit)
    for layer in layer_cirucit:
        for gate in layer:
            n_qubits = gate['qubits']
            if gate['name'] == 'u':
                theta, phi, lam = gate['params']  # TODO: 这个地方的参数位置需要检查一下
                qiskit_circuit.u(theta, phi, lam, qubit=n_qubits)
            elif gate['name'] == 'h':
                qiskit_circuit.h(qubit=n_qubits)
            elif gate['name'] == 'x':
                qiskit_circuit.x(qubit=n_qubits)
            elif gate['name'] == 'rz':
                qiskit_circuit.rz(gate['params'][0], qubit=n_qubits)
            elif gate['name'] == 'cond_x':
                classcial_register = ClassicalRegister(1)
                qiskit_circuit.add_register(classcial_register)
                qiskit_circuit.measure(n_qubits[0], classcial_register)
                qiskit_circuit.x(qubit=n_qubits[1]).c_if(classcial_register, 1)
            elif gate['name'] == 'cond_z':
                classcial_register = ClassicalRegister(1)
                qiskit_circuit.add_register(classcial_register)
                qiskit_circuit.measure(n_qubits[0], classcial_register)
                qiskit_circuit.z(qubit=n_qubits[1]).c_if(classcial_register, 1)
            elif gate['name'] == 'cx':
                qiskit_circuit.cx(control_qubit=n_qubits[0], target_qubit=n_qubits[1])
            elif gate['name'] == 'cz':
                qiskit_circuit.cz(control_qubit=n_qubits[0], target_qubit=n_qubits[1])
            elif gate['name'] == 'mcz':
                qiskit_circuit.mcp(gate['params'][0], control_qubits=n_qubits[:-1], target_qubit=n_qubits[-1])
            elif gate['name'] == 'ctrl':
                unitary = gate['params']
                operation_circuit = QuantumCircuit(len(gate['ctrled_qubits']))
                operation_circuit.unitary(unitary, qubits=list(range(len(gate['ctrled_qubits']))))
                custom_control = operation_circuit.to_gate().control(len(gate['ctrl_qubits']))
                qiskit_circuit.append(custom_control, gate['ctrl_qubits'] + gate['ctrled_qubits'])
            elif gate['name'] == 'initialize':
                qiskit_circuit.initialize(gate['params'],qubits= n_qubits)
            elif gate['name'] == 'unitary':
                unitary = gate['params']
                qiskit_circuit.unitary(unitary,qubits= n_qubits)
            elif gate['name'] == 'channel':
                channel = gate['params']
                qiskit_circuit.channel(channel, qubits= n_qubits)
            elif gate['name'] == 'measure':
                classcial_register = ClassicalRegister(len(n_qubits))
                qiskit_circuit.add_register(classcial_register)
                qiskit_circuit.measure(n_qubits, classcial_register)
            else:
                raise Exception('Unkown gate type', gate)
    return qiskit_circuit

def qiskit_circuit_to_layer_cirucit(qiskit_circuit: QuantumCircuit) -> list:
    all_qubits = qiskit_circuit.qubits
    layer2qiskit_instructions = get_layered_instructions(qiskit_circuit)
    layer_circuit = []
    for layer_index, layer_instructions in enumerate(layer2qiskit_instructions):
        layer_instructions = [qiskit_to_layer_gate(
            instruction,all_qubits) for instruction in layer_instructions]
        layer_circuit.append(layer_instructions)

    return layer_circuit


def format_circuit(layer_cirucit):
    new_circuit = []

    id = 0
    for layer in layer_cirucit:
        layer = copy.deepcopy(layer)

        new_layer = []
        for gate in layer:
            gate['id'] = id
            gate['layer'] = len(new_circuit)
            new_layer.append(gate)
            id += 1
        new_circuit.append(new_layer)

        if layer[0]['name'] == 'breakpoint':
            assert len(layer) == 1  # 如果存在一个assertion, 就不能存在别的操作
            # new_circuit.append([])  # breakpoint 占两层，下一层用unitary gate来制备

    return new_circuit


def get_layered_instructions(circuit):
    dagcircuit, instructions, nodes = my_circuit_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  # Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(
            node, DAGOpNode) and node.op.name not in ('barrier', 'measure')]
        if len(layer) != 0:
            layer2operations.append(layer)

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):  # 层号，该层操作
        layer_instructions = []
        for node in operations:  # 该层的一个操作
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer  # instruction在第几层
        layer2instructions.append(layer_instructions)

    return layer2instructions  #, instruction2layer, instructions, dagcircuit, nodes


def qiskit_to_layer_gate(instruction,all_qubits):
    name = instruction.operation.name
    parms = list(instruction.operation.params)
    return {
        'name': name,
        'qubits': [all_qubits.index(qubit) for qubit in instruction.qubits],
        'params': [ele if isinstance(ele, float) else float(ele) for ele in parms],
    }


def my_circuit_to_dag(circuit: QuantumCircuit):
    instructions = []
    dagnodes = []

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    # for instruction, qargs, cargs in circuit.data:
    for instruction in circuit.data:
        operation = instruction.operation

        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        if operation.name == 'barrier':
            continue
        instructions.append(instruction)  # TODO: 这个里面会不会有barrier
        dagnodes.append(dag_node)
        
        operation._index = len(dagnodes) - 1
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes

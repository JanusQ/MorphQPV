
if __name__ == "__main__":
    from clliford.utills import generate_inout_stabilizer_tables,CllifordProgram
    from clliford.clliford_gate_variables import CllifordCorrecter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford,random_clifford
    n_qubits = 3
    circuit = random_clifford(n_qubits).to_circuit()
    circuit.data = circuit.data[:10]
    ## 测试要逼近的电路
    program = CllifordProgram.from_circuit(circuit)
    print('test program\n',program.to_circuit())
    correcter = CllifordCorrecter(n_qubits,len(program))
    inputs, outputs = [],[]
    ## 生成输入和输出
    for _ in range(30):
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(n_qubits,program)
        inputs.append(input_stabilizer_table)
        correcter.add_iout(input_stabilizer_table,output_stabilizer_table)
    ## 求解得到的程序
    find_program = correcter.solve()
    print('find program\n',find_program.to_circuit())
    for input_stabilizer_table, output_stabilizer_table in zip(inputs,outputs):
        print('input')
        print(input_stabilizer_table.to_string())
        print('real output')
        solver = CllifordCorrecter(n_qubits,len(find_program))
        solver_out = solver.inference(input_stabilizer_table, find_program)
        assert solver_out.is_eq(output_stabilizer_table)
    








    
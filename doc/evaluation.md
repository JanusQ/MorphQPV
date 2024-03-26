## Evaluation of the paper experiments
The evaluation is conducted on a machine with an Intel Core i7-15700K CPU and 32GB RAM. The operating system is Windows 15. The experiments are implemented in Python 3.11. The experiments are conducted on the following quantum circuits:
- **Shor**: Shor's algorithm
- **QNN**: Quantum Neural Network
- **QFT**: Quantum Fourier Transform
- **XEB**: Cross-Entropy Benchmarking
- **QEC**: Quantum Error Correction
- **QL**: Quantum Lock

The source code for the experiments can be found in the [examples](../examples) folder. The following is the detailed evaluation for each experiment.

### Quantum Lock (Fig.7 in the paper)
User can run the following command after installing the required packages to reproduce the result in the paper:
```bash
conda activate morphenv
python examples/fig7-quantumlock_verify.py --qubits [MAX_QUBITS]
```
the parameter `--qubits` is the maximum number of qubits in the quantum lock circuit. The evaluation will be conducted on quantum lock circuits with qubits from 4 to `MAX_QUBITS`. The result will be saved to the default path [`examples/fig7-quantumlock_verify/`](../examples/fig7-quantumlock_verify/) directory.
When the number of qubits reaches 15 or more, it takes a longer time to find the minimal samples, so mounting the program to run in the background with `nohup` or other would be better.

### Comparison with other assertion methods(Table 4 in the paper)
User can run the following command after installing the required packages to reproduce the result in the paper:
```bash
python examples/table4-compare.py
```
The result will be saved to the default path [`examples/table4-compare/`](../examples/table4-compare/) directory. 

### evaluation of theorem 1 (Fig.11(a) in the paper)
User can run the following command after installing the required packages to reproduce the result in the paper:
```bash
python examples/fig11a-theorem1.py
```
The result will be saved to the default path [`./examples/fig11a-theorem1/`](../examples/fig11a-theorem1/) directory. This is also a long time task.


### evaluation of theorem 2 (Fig.11(b) in the paper)
User can run the following command after installing the required packages to reproduce the result in the paper:
```bash
python examples/fig11b-theorem2.py
```
The result will be saved to the default path [`examples/fig11b-theorem2/`](../examples/fig11b-theorem2/) directory.

### the confidence for different quantum circuits (Fig.15 in the paper)
Users can run the following command after installing the required packages to reproduce the result in the paper:
```bash
python examples/fig15-confidence.py
```
The result will be saved to the default path [`examples/fig15-confidence/`](../examples/fig15-confidence/) directory.

### the comparison of different optimization strategies (Fig.11 in the paper)
Users can run the following command after installing the required packages to reproduce the result in the paper:
This file needs to have a good connection with the internet to download the MNIST dataset, so please check your internet before running this command.
The result will be saved to the default path [`examples/fig11-opt_strategy/`](../examples/fig11-opt_strategy/) directory. 

### the ablation study of using Clifford gates and basis gates (Fig.15(a) in the paper)
Users can run the following command after installing the required packages to reproduce the result in the paper:
```bash
python examples/fig15a-ablation_study.py
```
The result will be saved to the default path [`examples/fig15a-ablation_study/`](../examples/fig15a-ablation_study/) directory.

### the runtime comparison of different optimization solvers (Fig.15(b) in the paper)
Users can run the following command after installing the required packages to reproduce the result in the paper:
```bash
python examples/fig15b-solvers_compare.py
```
The result will be saved to the default path [`examples/fig15b-solvers_compare/`](../examples/fig15b-solvers_compare/) directory.
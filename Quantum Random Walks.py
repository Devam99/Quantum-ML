import numpy as np
import time
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

#Parameters
n_position = 4                # 4 qubits -> 16 positions
n_coin = 1                    # 1 qubit -> 2 states (|0> and |1>)
n_steps = 8                   # Number of steps in the quantum walk
N_positions = 2 ** n_position # Total number of positions 16

coin_q = 0
pos_q = list(range(1, n_position + 1))
total_qubits = n_coin + n_position

print ("=" * 60)
print("Quantum Random Walks Simulation")
print ("=" * 60)
print(f"Number of position qubits: {n_position} (Total positions: {N_positions})")
print(f"Number of coin qubits: {n_coin}")
print(f"Total qubits in the circuit: {total_qubits}")
print(f"Number of steps in the quantum walk: {n_steps}")
print ("=" * 60)

# Create quantum walk circuit

start_build = time.perf_counter()

qc = QuantumCircuit(total_qubits, name="Quantum Walk")


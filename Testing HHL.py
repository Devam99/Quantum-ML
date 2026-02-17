import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, RYGate, QFTGate
from qiskit.quantum_info import Statevector


# Define the matrix A and vector b
# Matrix A must be Hermitian with known (or estimable) eigenvalues
# This A has eigenvalues λ=1 and λ=2 (chosen to map cleanly to QPE)
A = np.array([
    [1.5, 0.5],
    [0.5, 1.5]
])
b_vec = np.array([1.0, 0.0])

#classical solution for verification
x_classical = np.linalg.solve(A, b_vec)
x_classical_norm = x_classical / np.linalg.norm(x_classical)

evals, evecs = np.linalg.eigh(A)

print(f"Eigenvalues of A: {evals}")
print(f"Eigenvectors of A:\n{evecs}")
print(f"Classical solution: {x_classical}")
print(f"Classical solution (normalized): {x_classical_norm}")
print()


#HHL parameters
n_clock = 2  #clock qubits for QPE (2 qubits can represent 4 eigenvalues, which is sufficient for our 2x2 matrix)
t0 = 2 * np.pi / (2 ** n_clock)  # = 2π/4 = π/2, chosen to map eigenvalues to [0,1) range for QPE

C = 1.0  # Scaling factor for the controlled rotation, set to 1 for simplicity, must be <= min(eigenvalues) to avoid unphysical rotations

# Create the quantum circuit
#qubit 0 -> system register (encodes |b⟩, eventually |x⟩)
#qubits 1 -> clock LSB
#qubits 2 -> clock MSB
#qubits 3 -> ancilla (for eigenvalue inversion)

sys_q = 0
clk_q = [1, 2]
anc_q = 3

qc = QuantumCircuit(4, name="HHL")

# Step 1: Prepare |b⟩ in the system register
# b = [1, 0] corresponds to |0⟩ state, so we can skip this step as the system register is already initialized to |0⟩
#for b = [0,1] you would add qc.x(sys_q) to prepare |1⟩
#for b = [1/sqrt(2), 1/sqrt(2)] you would add qc.h(sys_q) to prepare (|0⟩ + |1⟩)/sqrt(2)

# Step 2: Quantum Phase Estimation (QPE)
#Hadamard gates to create superposition in clock register
qc.h(clk_q)

#Controlled unitary operations for QPE : clock[j] controls e^{iA·t0·2^j}
for j, c in enumerate(clk_q):
    U_power = expm(1j * A * t0 * (2 ** j))  # Compute the unitary for this power
    gate = UnitaryGate(U_power, label=f"U^{2**j}")
    qc.append(gate.control(1), [c, sys_q])  # Control on clock qubit, target on system qubit

# Step 3: Inverse Quantum Fourier Transform (QFT†) on clock register
iqft = QFTGate(n_clock).inverse()
qc.append(iqft, clk_q)

qc.barrier(label="QPE done")

# Verify the state after QPE (should be a superposition of eigenstates with phases corresponding to eigenvalues)
sv_after_qpe = Statevector.from_instruction(qc)
print("=== State after QPE ===")
print(sv_after_qpe.draw("latex_source" if False else "text"))
print()

# Step 4: Controlled rotation based on the clock register (eigenvalue inversion)

eigenvalue_clock_map = {
    1: '01',  # λ=1 corresponds to binary 01 in the clock register stored in clock as |01⟩
    2: '10',   # λ=2 corresponds to binary 10 in the clock register stored in clock as |10⟩
}

for lam, ctrl_state in eigenvalue_clock_map.items():
    theta = 2 * np.arcsin(C / lam)  # Rotation angle for the controlled rotation
    cry_gate = RYGate(theta).control(2, ctrl_state=ctrl_state)  # Control on clock register, target on ancilla
    qc.append(cry_gate, [clk_q[0], clk_q[1], anc_q]) # Control on clock qubits, target on ancilla)

qc.barrier(label="Controlled rotations done")

# Step 5: Inverse QPE to uncompute the clock register

qft = QFTGate(n_clock)
qc.append(qft, clk_q)

for j in reversed(range(n_clock)):
    U_dag = expm(-1j * A * t0 * (2 ** j))
    gate = UnitaryGate(U_dag, label=f"U†^{2**j}")
    qc.append(gate.control(1), [clk_q[j], sys_q])

qc.h(clk_q)

qc.barrier(label="Inverse QPE done")

sv_final = Statevector.from_instruction(qc)

state_dict = sv_final.to_dict()

print("=== Full Statevector (non-zero amplitudes) ===")
for label, amp in sorted(state_dict.items()):
    if abs(amp) > 1e-10:
        print(f"  |{label}⟩ : {amp:.6f}  (prob={abs(amp) ** 2:.6f})")
print()

# Extract postselected amplitude for ancilla = |1⟩ (indicating successful eigenvalue inversion)

amp_sys0 =state_dict.get('1000', 0)  # |1⟩_sys |00⟩_clk |0⟩_anc
amp_sys1 =state_dict.get('1001', 0)  # |1⟩_sys |00⟩_clk |1⟩_anc

print("=== Postselected Amplitudes (ancilla = |1⟩, clock = |00⟩) ===")
print(f"  amplitude for sys=|0⟩: {amp_sys0:.6f}")
print(f"  amplitude for sys=|1⟩: {amp_sys1:.6f}")


x_hhl = np.array([amp_sys0, amp_sys1])

p_success = np.linalg.norm(x_hhl) ** 2
print(f"\n  P(success) = {p_success:.6f}")

#Normalise to get HHL solution state
if p_success > 1e-10:
    x_hhl_norm = np.real(x_hhl / np.linalg.norm(x_hhl))
    print(f"\n  P(success) = {p_success:.6f}")
else:
    x_hhl_norm = np.real(x_hhl)
    print("  WARNING: post-selection probability is ~0")

#Compare with classical solution

print("\n" + "=" * 55)
print("COMPARISON")
print("=" * 55)
print(f"  Classical x (normalized):  {x_classical_norm}")
print(f"  HHL x (normalized):        {np.real(x_hhl_norm)}")

fidelity = abs(np.dot(np.conj(x_hhl_norm), x_classical_norm)) ** 2
print(f"\n  Fidelity: {fidelity:.6f}")
if fidelity > 0.99:
    print("HHL matches the classical solution!")
else:
    print("Mismatch — check parameters")


#circuit visualisation

print("\n === HHL Circuit ===")
print(qc.draw(output="text" , fold=120))

print(f"\n=== Circuit Stats ===")
print(f"  Total qubits: {qc.num_qubits}")
print(f"  Total gates:  {qc.size()}")
print(f"  Depth:        {qc.depth()}")

from qiskit.circuit.library import Measure

qc_meas = qc.copy()
qc_meas.add_register(__import__('qiskit', fromlist=['ClassicalRegister']).ClassicalRegister(2, 'result'))
qc_meas.measure(sys_q, 0) #measure system qubit to classical bit 0
qc_meas.measure(sys_q, 1) #measure system qubit to classical bit 1 (for postselection on ancilla=|1⟩, we would measure ancilla instead)

from qiskit import transpile, ClassicalRegister
from qiskit_aer import AerSimulator

qc_meas = qc.copy()
cr = ClassicalRegister(4, 'meas')
qc_meas.add_register(cr)
qc_meas.measure(sys_q, 0)     # bit 0 = system
qc_meas.measure(clk_q[0], 1)  # bit 1 = clock LSB
qc_meas.measure(clk_q[1], 2)  # bit 2 = clock MSB
qc_meas.measure(anc_q, 3)     # bit 3 = ancilla

backend = AerSimulator()
shots = 100000

# IMPORTANT: disable optimization to preserve qubit ordering
qc_transpiled = transpile(qc_meas, backend, optimization_level=0)
job = backend.run(qc_transpiled, shots=shots)
counts = job.result().get_counts()

print(f"\n=== Shot-based Simulation ({shots} shots) ===")
print(f"  Raw counts: {counts}")

# Post-select: ancilla=1 AND clock=00
# Bit string format: bit3 bit2 bit1 bit0 = anc clk_msb clk_lsb sys
post_selected = {}
for bitstr, count in counts.items():
    anc = bitstr[0]       # bit 3 (leftmost)
    clk_msb = bitstr[1]   # bit 2
    clk_lsb = bitstr[2]   # bit 1
    sys_bit = bitstr[3]    # bit 0 (rightmost)

    if anc == '1' and clk_msb == '0' and clk_lsb == '0':
        post_selected[sys_bit] = post_selected.get(sys_bit, 0) + count

total_post = sum(post_selected.values())
print(f"  Post-selected counts (anc=1, clk=00): {post_selected}")
print(f"  Post-selection rate: {total_post}/{shots} = {total_post/shots:.4f}")

if total_post > 0:
    n_sys0 = post_selected.get('0', 0)
    n_sys1 = post_selected.get('1', 0)

    prob_0 = n_sys0 / total_post
    prob_1 = n_sys1 / total_post

    # Reconstruct amplitudes (note: sign is lost in measurement)
    x_shots = np.array([np.sqrt(prob_0), -np.sqrt(prob_1)])  # sign from classical knowledge
    x_shots_norm = x_shots / np.linalg.norm(x_shots)

    print(f"\n  P(sys=0 | post-selected) = {prob_0:.4f}")
    print(f"  P(sys=1 | post-selected) = {prob_1:.4f}")
    print(f"  Shot-based x (normalized): {x_shots_norm}")
    print(f"  Classical x (normalized):  {x_classical_norm}")

    fidelity_shots = abs(np.dot(x_shots_norm, x_classical_norm)) ** 2
    print(f"  Shot fidelity: {fidelity_shots:.4f}")


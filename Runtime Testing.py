import time
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, RYGate, QFTGate
from qiskit.quantum_info import Statevector

# ============================================================
# THE MATRIX AND VECTOR
# ============================================================
A = np.array([
    [1.5, 0.5],
    [0.5, 1.5]
])
b_vec = np.array([1.0, 0.0])

# ============================================================
# BENCHMARK 1: CLASSICAL SOLVE
# ============================================================
n_classical_runs = 100000  # run many times to get stable timing

start = time.perf_counter()
for _ in range(n_classical_runs):
    x_classical = np.linalg.solve(A, b_vec)
end = time.perf_counter()

t_classical_total = end - start
t_classical_single = t_classical_total / n_classical_runs

print("=" * 60)
print("CLASSICAL SOLVE")
print("=" * 60)
print(f"  Runs:          {n_classical_runs:,}")
print(f"  Total time:    {t_classical_total:.4f} s")
print(f"  Per solve:     {t_classical_single * 1e6:.2f} μs")
print(f"  Solution:      {x_classical}")
print()

# ============================================================
# BENCHMARK 2: HHL — CIRCUIT CONSTRUCTION
# ============================================================
start_build = time.perf_counter()

n_clock = 2
t0 = 2 * np.pi / (2 ** n_clock)
C = 1.0

sys_q = 0
clk_q = [1, 2]
anc_q = 3

qc = QuantumCircuit(4, name='HHL')

# QPE
qc.h(clk_q)
for j, c in enumerate(clk_q):
    U_power = expm(1j * A * t0 * (2 ** j))
    gate = UnitaryGate(U_power, label=f'U^{2 ** j}')
    qc.append(gate.control(1), [c, sys_q])

iqft = QFTGate(n_clock).inverse()
qc.append(iqft, clk_q)

# Controlled rotations
eigenvalue_clock_map = {1: '01', 2: '10'}
for lam, ctrl_state in eigenvalue_clock_map.items():
    theta = 2 * np.arcsin(C / lam)
    cry_gate = RYGate(theta).control(2, ctrl_state=ctrl_state)
    qc.append(cry_gate, [clk_q[0], clk_q[1], anc_q])

# Inverse QPE
qft = QFTGate(n_clock)
qc.append(qft, clk_q)
for j in reversed(range(n_clock)):
    U_dag = expm(-1j * A * t0 * (2 ** j))
    gate = UnitaryGate(U_dag, label=f'U†^{2 ** j}')
    qc.append(gate.control(1), [clk_q[j], sys_q])
qc.h(clk_q)

end_build = time.perf_counter()
t_build = end_build - start_build

# ============================================================
# BENCHMARK 3: HHL — STATEVECTOR SIMULATION
# ============================================================
start_sim = time.perf_counter()
sv_final = Statevector.from_instruction(qc)
end_sim = time.perf_counter()
t_simulate = end_sim - start_sim

# ============================================================
# BENCHMARK 4: HHL — POST-PROCESSING
# ============================================================
start_post = time.perf_counter()
state_dict = sv_final.to_dict()
amp_sys0 = state_dict.get('1000', 0)
amp_sys1 = state_dict.get('1001', 0)
x_hhl = np.array([amp_sys0, amp_sys1])
x_hhl_norm = np.real(x_hhl / np.linalg.norm(x_hhl))
end_post = time.perf_counter()
t_post = end_post - start_post

t_hhl_total = t_build + t_simulate + t_post

# ============================================================
# BENCHMARK 5: HHL — SHOT-BASED SIMULATION
# ============================================================
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator

qc_meas = qc.copy()
cr = ClassicalRegister(4, 'meas')
qc_meas.add_register(cr)
qc_meas.measure(sys_q, 0)
qc_meas.measure(clk_q[0], 1)
qc_meas.measure(clk_q[1], 2)
qc_meas.measure(anc_q, 3)

backend = AerSimulator()

start_shots = time.perf_counter()
qc_transpiled = transpile(qc_meas, backend, optimization_level=0)
job = backend.run(qc_transpiled, shots=100000)
counts = job.result().get_counts()
end_shots = time.perf_counter()
t_shots = end_shots - start_shots

# ============================================================
# RESULTS
# ============================================================
print("=" * 60)
print("HHL QUANTUM SIMULATION")
print("=" * 60)
print(f"  Circuit build time:      {t_build * 1000:.2f} ms")
print(f"  Statevector simulation:  {t_simulate * 1000:.2f} ms")
print(f"  Post-processing:         {t_post * 1000:.4f} ms")
print(f"  ─────────────────────────────────────")
print(f"  Total (statevector):     {t_hhl_total * 1000:.2f} ms")
print(f"  Total (shot-based):      {t_shots * 1000:.2f} ms")
print(f"  Solution:                {x_hhl_norm}")
print()

print("=" * 60)
print("COMPARISON")
print("=" * 60)

ratio_sv = t_hhl_total / t_classical_single
ratio_shots = t_shots / t_classical_single

print(f"  Classical per solve:     {t_classical_single * 1e6:.2f} μs")
print(f"  HHL (statevector):       {t_hhl_total * 1e6:.0f} μs")
print(f"  HHL (shot-based):        {t_shots * 1e6:.0f} μs")
print()
print(f"  ┌──────────────────────────────────────────────┐")
print(f"  │ Classical is {ratio_sv:,.0f}x faster (vs statevector)  │")
print(f"  │ Classical is {ratio_shots:,.0f}x faster (vs shot-based) │")
print(f"  └──────────────────────────────────────────────┘")
print()

# ============================================================
# BREAKDOWN VISUALIZATION
# ============================================================
print("=" * 60)
print("WHERE HHL SPENDS ITS TIME")
print("=" * 60)

total = t_build + t_simulate + t_post
print(f"  Matrix exponential (expm):  {t_build/total*100:5.1f}%  ← {t_build*1000:.1f} ms")
print(f"  Statevector simulation:      {t_simulate/total*100:5.1f}%  ← {t_simulate*1000:.1f} ms")
print(f"  Post-processing:             {t_post/total*100:5.1f}%  ← {t_post*1000:.4f} ms")
print()
print(f"  The matrix exponential (scipy.linalg.expm) dominate")
print(f"  because they ARE classical computation — the quantum")
print(f"  simulation is simulating quantum gates using classical math.")
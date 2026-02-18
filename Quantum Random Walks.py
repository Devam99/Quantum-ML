import numpy as np
import time
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, state_fidelity
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

#Initial State:
#Position  = 0 -> in binary with offset, |0000> (4 qubits for position) (value 8)
#Encode position as unsigned integer, then subtract 8 later
#Position 8 in binary is 1000, -> set qubit pos_q[3] to 1 (MSB)

qc.x(pos_q[3])  # Set position to 8 (|1000>)

#coin starts in |0> state, so no need to apply any gate

qc.barrier(label = "Initial State")

for step in range(n_steps):

    # ── STEP A: COIN FLIP ──
    qc.h(coin_q)

    # ── STEP B: CONTROLLED INCREMENT (coin=|1⟩ → move right) ──
    # Flip MSB if coin=1 AND bit2=1 AND bit1=1 AND bit0=1
    qc.mcx([coin_q, pos_q[0], pos_q[1], pos_q[2]], pos_q[3])
    # Flip bit2 if coin=1 AND bit1=1 AND bit0=1
    qc.mcx([coin_q, pos_q[0], pos_q[1]], pos_q[2])
    # Flip bit1 if coin=1 AND bit0=1
    qc.ccx(coin_q, pos_q[0], pos_q[1])
    # Flip bit0 if coin=1
    qc.cx(coin_q, pos_q[0])

    # ── STEP C: CONTROLLED DECREMENT (coin=|0⟩ → move left) ──
    qc.x(coin_q)    # flip coin so we control on original |0⟩

    qc.x(pos_q)     # flip all position bits (two's complement trick)

    qc.mcx([coin_q, pos_q[0], pos_q[1], pos_q[2]], pos_q[3])
    qc.mcx([coin_q, pos_q[0], pos_q[1]], pos_q[2])
    qc.ccx(coin_q, pos_q[0], pos_q[1])
    qc.cx(coin_q, pos_q[0])

    qc.x(pos_q)     # flip back
    qc.x(coin_q)    # flip coin back

    if step < n_steps - 1:
        qc.barrier(label=f'Step {step + 1}')

qc.barrier(label=f'Step {n_steps} (final)')

t_build = time.perf_counter() - start_build

start_sim = time.perf_counter()
sv = Statevector.from_instruction(qc)
t_sim = time.perf_counter() - start_sim

probs = {}
state_dict = sv.to_dict()

for label, amp in state_dict.items():
    if abs(amp) <1e-10:
        pos_bits= label[:-1]
        pos_int = int(pos_bits, 2)

        pos_signed = pos_int - (N_positions // 2)

        if pos_signed not in probs:
            probs[pos_signed] =0.0
        probs[pos_signed] += abs(amp) ** 2

quantum_positions = sorted(probs.keys())
quantum_probs = [probs[p] for p in quantum_positions]

print("=" * 60)
print("Quantum Walk Simulation Results")
print("=" * 60)
print(f"Time taken to build the circuit: {t_build * 1000:.2f} ms")
print(f"Simulation Time: {t_sim * 1000:.2f} ms")
print()
print(f" {'Position':>10} | {'Probability':>12} {'Bar':>30}")
print(f"  {'─' * 10} {'─' * 12} {'─' * 30}")
for pos, prob in zip(quantum_positions, quantum_probs):
    bar = "█" * int(prob * 100)
    print(f"  {pos:>10} {prob:>12.6f}  {bar}")

qc.meas = qc.copy()
cr = ClassicalRegister(n_position, 'pos')
qc.meas.add_register(cr)
for i in range(n_position):
    qc.meas.measure(pos_q[i], i)

backend = AerSimulator()
shots = 100000

start_shots = time.perf_counter()
qc_transpiled = transpile(qc.meas, backend, optimization_level=0)
job = backend.run(qc_transpiled, shots=shots)
counts = job.result().get_counts()
t_shots = time.perf_counter() - start_shots

shots_probs = {}
for bitstr, count in counts.items():
    pos_int = int(bitstr, 2)
    pos_signed = pos_int - (N_positions // 2)
    shots_probs[pos_signed] = count / shots


print()
print("=" * 60)
print(f"Quantum Walk Results (Shot-based, {shots} shots)")
print("=" * 60)
print(f"Time taken for shot-based simulation: {t_shots * 1000:.2f} ms")
print()

start_classical = time.perf_counter()
n_classical_walkers = 1000000

steps = np.random.choice([-1, 1], size=(n_classical_walkers, n_steps))
final_positions = np.sum(steps, axis=1)

classical_probs = {}
for pos in range(-(n_steps), n_steps + 1):
    count = np.sum(final_positions == pos)
    if count > 0:
        classical_probs[pos] = count / n_classical_walkers

t_classical = time.perf_counter() - start_classical


print()
print("=" * 60)
print("CLASSICAL vs QUANTUM RANDOM WALK")
print("=" * 60)

all_positions = sorted(set(list(classical_probs.keys()) + list(probs.keys())))

all_positions = [p for p in all_positions if -n_steps <= p <= n_steps]

print(f"  {'Pos':>4} │ {'Classical':>10} │ {'Quantum':>10} │ {'Classical':>25} │ {'Quantum':>25}")
print(f"  {'─' * 4}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 25}─┼─{'─' * 25}")

for pos in all_positions:
    cp = classical_probs.get(pos, 0)
    qp = probs.get(pos, 0)
    c_bar = "█" * int(cp * 80)
    q_bar = "▓" * int(qp * 80)
    print(f"  {pos:>4} │ {cp:>10.4f} │ {qp:>10.4f} │ {c_bar:<25} │ {q_bar:<25}")


print()
print("=" * 60)
print("STATISTICS")
print("=" * 60)

q_mean = sum(pos * prob for pos, prob in probs.items())
q_var = sum(pos ** 2 * prob for pos, prob in probs.items()) - q_mean ** 2
q_std = np.sqrt(abs(q_var))

c_mean = np.mean(final_positions)
c_std = np.std(final_positions)

c_std_theory = np.sqrt(n_steps)
q_std_theory = n_steps * 1 / np.sqrt(2)

print(f"  {'Metric':<30} {'Classical':>12} {'Quantum':>12}")
print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
print(f"  {'Mean position':<30} {c_mean:>12.4f} {q_mean:>12.4f}")
print(f"  {'Std deviation (measured)':<30} {c_std:>12.4f} {q_std:>12.4f}")
print(f"  {'Std deviation (theory)':<30} {c_std_theory:>12.4f} {q_std_theory:>12.4f}")
print(f"  {'Spread ratio (quantum/class)':<30} {'':>12} {q_std / c_std:>12.4f}x")
print()
print(f"  Classical spread:  σ ~ √t = √{n_steps} = {c_std_theory:.2f}")
print(f"  Quantum spread:    σ ~ t/√2 = {n_steps}/√2 = {q_std_theory:.2f}")
print(f"  Quantum spreads {q_std_theory / c_std_theory:.1f}x faster!")

print()
print("=" * 60)
print("TIMING")
print("=" * 60)
print(f"  Classical ({n_classical_walkers:,} walkers): {t_classical * 1000:.2f} ms")
print(f"  Quantum (statevector):              {(t_build + t_sim) * 1000:.2f} ms")
print(f"  Quantum (shot-based):               {t_shots * 1000:.2f} ms")

print()
print("=" * 60)
print("CIRCUIT STATS")
print("=" * 60)
print(f"  Total qubits:  {qc.num_qubits}")
print(f"  Total gates:   {qc.size()}")
print(f"  Circuit depth:  {qc.depth()}")
print(f"  Gates per step: {qc.size() // n_steps}")

print()
print("=" * 60)
print("CIRCUIT DIAGRAM (first 2 steps)")
print("=" * 60)

qc_small = QuantumCircuit(total_qubits, name='QWalk (2 steps)')
qc_small.x(pos_q[3])
qc_small.barrier(label='Init')
for step in range(2):
    qc_small.h(coin_q)
    qc_small.mcx([coin_q, pos_q[0], pos_q[1], pos_q[2]], pos_q[3])
    qc_small.mcx([coin_q, pos_q[0], pos_q[1]], pos_q[2])
    qc_small.ccx(coin_q, pos_q[0], pos_q[1])
    qc_small.cx(coin_q, pos_q[0])
    qc_small.x(coin_q)
    qc_small.x(pos_q)
    qc_small.mcx([coin_q, pos_q[0], pos_q[1], pos_q[2]], pos_q[3])
    qc_small.mcx([coin_q, pos_q[0], pos_q[1]], pos_q[2])
    qc_small.ccx(coin_q, pos_q[0], pos_q[1])
    qc_small.cx(coin_q, pos_q[0])
    qc_small.x(pos_q)
    qc_small.x(coin_q)
    qc_small.barrier(label=f'Step {step + 1}')

print(qc_small.draw(output='text', fold=120))

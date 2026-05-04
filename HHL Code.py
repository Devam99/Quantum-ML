import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import expm
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, RYGate, QFTGate, HamiltonianGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def validate_inputs(A, b):
    """Check that A is Hermitian, square, invertible, dimensions are
    power of 2, and b has matching dimension."""
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError(f"Dimension N={N} must be a power of 2.")
    if b.shape[0] != N:
        raise ValueError(f"b has dimension {b.shape[0]}, expected {N}.")
    if np.abs(np.linalg.det(A)) < 1e-10:
        raise ValueError("A must be invertible.")


def get_system_info(A, b):
    """Compute eigenvalues, classical solution, and HHL parameters."""
    N = A.shape[0]
    n_system = int(np.log2(N))

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    kappa = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

    x_classical = np.linalg.solve(A, b)
    x_classical_norm = x_classical / np.linalg.norm(x_classical)

    return {
        'N': N,
        'n_system': n_system,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'kappa': kappa,
        'x_classical': x_classical,
        'x_classical_norm': x_classical_norm,
    }


def choose_parameters(eigenvalues, n_clock):
    """Choose t0 and C for the HHL circuit.
    t0 is chosen so that eigenvalue phases fit in the clock register.
    C is set to lambda_min to maximise success probability."""
    lambda_min = np.min(np.abs(eigenvalues))
    lambda_max = np.max(np.abs(eigenvalues))

    # t0 chosen so that lambda_max * t0 * 2^n_clock / (2*pi) < 2^n_clock
    # i.e. all eigenvalues map to phases in (0, 1)
    t0 = 2 * np.pi / (2**n_clock)

    C = lambda_min

    return t0, C


def build_hhl_circuit(A, b, n_clock, t0, C, eigenvalues, n_system,
                      use_trotter=False, trotter_steps=4,
                      full_rotation=False):
    """Build the full HHL circuit.

    Parameters:
        full_rotation: if True, program a rotation for EVERY clock
                       register state k=1,...,2^n_clock - 1, not just
                       the states corresponding to known eigenvalues.
                       Required for correct behavior with non-integer
                       eigenvalues. Expensive: O(2^n_clock) rotations.
    """
    N = 2 ** n_system
    n_total = n_clock + n_system + 1

    sys_q = list(range(n_system))
    clk_q = list(range(n_system, n_system + n_clock))
    anc_q = n_system + n_clock

    qc = QuantumCircuit(n_total, name='HHL')

    # Step 1: Prepare |b>
    b_norm = b / np.linalg.norm(b)
    qc.initialize(b_norm, sys_q)

    # Step 2: QPE
    qc.h(clk_q)

    for j, c in enumerate(clk_q):
        if use_trotter:
            trotter_qc = build_trotter_circuit(
                A, t0 * (2 ** j), n_system, trotter_steps
            )
            controlled_gate = trotter_qc.to_gate().control(1)
        else:
            U_power = expm(1j * A * t0 * (2 ** j))
            controlled_gate = UnitaryGate(
                U_power, label=f'U^{2 ** j}'
            ).control(1)

        qc.append(controlled_gate, [c] + sys_q)

    iqft = QFTGate(n_clock).inverse()
    qc.append(iqft, clk_q)

    qc.barrier(label='QPE done')

    # Step 3: Controlled rotations
    if full_rotation:
        # Rotation for EVERY clock state k = 1, ..., 2^n_clock - 1
        for k in range(1, 2 ** n_clock):
            lam_k = k * 2 * np.pi / (t0 * (2 ** n_clock))
            ratio = C / lam_k
            if np.abs(ratio) > 1:
                ratio = np.sign(ratio)
            theta = 2 * np.arcsin(ratio)

            # Skip negligibly small rotations
            if np.abs(theta) < 1e-10:
                continue

            ctrl_state = format(k, f'0{n_clock}b')
            cry_gate = RYGate(theta).control(
                n_clock, ctrl_state=ctrl_state
            )
            qc.append(cry_gate, clk_q + [anc_q])
    else:
        # Original: only known eigenvalues
        eigenvalue_clock_map = build_eigenvalue_clock_map(
            eigenvalues, t0, n_clock
        )
        for lam, info_dict in eigenvalue_clock_map.items():
            ctrl_state = info_dict['ctrl_state']
            lam_rounded = info_dict['lam_rounded']

            ratio = C / lam_rounded
            if np.abs(ratio) > 1:
                ratio = np.sign(ratio)
            theta = 2 * np.arcsin(ratio)
            cry_gate = RYGate(theta).control(
                n_clock, ctrl_state=ctrl_state
            )
            qc.append(cry_gate, clk_q + [anc_q])

    qc.barrier(label='Rotations done')

    # Step 4: Inverse QPE
    qft = QFTGate(n_clock)
    qc.append(qft, clk_q)

    for j in reversed(range(n_clock)):
        if use_trotter:
            trotter_qc = build_trotter_circuit(
                A, -t0 * (2 ** j), n_system, trotter_steps
            )
            controlled_gate = trotter_qc.to_gate().control(1)
        else:
            U_dag = expm(-1j * A * t0 * (2 ** j))
            controlled_gate = UnitaryGate(
                U_dag, label=f'U†^{2 ** j}'
            ).control(1)

        qc.append(controlled_gate, [clk_q[j]] + sys_q)

    qc.h(clk_q)

    qc.barrier(label='Inverse QPE done')

    return qc, sys_q, clk_q, anc_q


def build_eigenvalue_clock_map(eigenvalues, t0, n_clock):
    """Map each eigenvalue to its binary control string and
    the rounded eigenvalue that the clock register actually stores."""
    clock_map = {}

    for lam in eigenvalues:
        k = lam * t0 * (2 ** n_clock) / (2 * np.pi)
        k_rounded = int(np.round(k)) % (2 ** n_clock)

        if k_rounded == 0:
            continue

        ctrl_state = format(k_rounded, f'0{n_clock}b')

        # Recover the eigenvalue the clock register actually represents
        lam_rounded = k_rounded * 2 * np.pi / (t0 * (2 ** n_clock))

        if lam not in clock_map:
            clock_map[lam] = {
                'ctrl_state': ctrl_state,
                'k_rounded': k_rounded,
                'lam_rounded': lam_rounded,
            }

    return clock_map

def extract_solution_statevector(qc, sys_q, clk_q, anc_q, n_system, n_clock):
    """Extract the HHL solution using statevector simulation.
    Post-selects on ancilla=|1> and clock=|00...0>."""
    sv = Statevector.from_instruction(qc)
    state_dict = sv.to_dict()

    N = 2**n_system
    n_total = n_clock + n_system + 1
    clock_zeros = '0' * n_clock

    solution_amplitudes = np.zeros(N, dtype=complex)

    for label, amp in state_dict.items():
        if abs(amp) < 1e-12:
            continue

        anc_bit = label[0]
        clk_bits = label[1:1 + n_clock]
        sys_bits = label[1 + n_clock:]

        if anc_bit == '1' and clk_bits == clock_zeros:
            sys_idx = int(sys_bits[::-1], 2)
            solution_amplitudes[sys_idx] = amp

    p_success = np.linalg.norm(solution_amplitudes)**2

    if p_success < 1e-12:
        print("WARNING: post-selection probability is ~0")
        return solution_amplitudes, p_success

    x_hhl_norm = solution_amplitudes / np.linalg.norm(solution_amplitudes)

    return x_hhl_norm, p_success


def extract_solution_shots(qc, sys_q, clk_q, anc_q, n_system, n_clock,
                           shots=100000):
    """Extract the HHL solution using shot-based simulation.
    Post-selects on ancilla=1 and clock=00...0."""
    # Add classical register and measurements
    qc_meas = qc.copy()
    n_total = n_clock + n_system + 1
    cr = ClassicalRegister(n_total, 'meas')
    qc_meas.add_register(cr)

    # Measure all qubits
    for i in range(n_total):
        qc_meas.measure(i, i)

    backend = AerSimulator()
    qc_transpiled = transpile(qc_meas, backend, optimization_level=0)
    job = backend.run(qc_transpiled, shots=shots)
    counts = job.result().get_counts()

    # Post-select: ancilla=1, clock=00...0
    N = 2**n_system
    post_selected = {}
    clock_zeros = '0' * n_clock

    for bitstr, count in counts.items():
        # Qiskit bitstring: highest index qubit on the left
        anc_bit = bitstr[0]
        clk_bits = bitstr[1:1 + n_clock]
        sys_bits = bitstr[1 + n_clock:]

        if anc_bit == '1' and clk_bits == clock_zeros:
            post_selected[sys_bits] = post_selected.get(sys_bits, 0) + count

    total_post = sum(post_selected.values())
    p_success = total_post / shots

    if total_post == 0:
        print("WARNING: no post-selected counts")
        return np.zeros(N), {}, p_success

    # Build probability vector
    probs = np.zeros(N)
    for sys_bits, count in post_selected.items():
        sys_idx = int(sys_bits[::-1], 2)
        probs[sys_idx] = count / total_post

    return probs, post_selected, p_success


def compare_solutions(x_classical_norm, x_hhl_norm, probs_shots=None):
    """Compare HHL solution with classical solution."""
    N = len(x_classical_norm)

    print("\n" + "=" * 60)
    print("COMPARISON: Statevector Simulation")
    print("=" * 60)

    # Handle global phase: align signs by multiplying by phase factor
    x_hhl_real = align_phase(x_hhl_norm, x_classical_norm)

    print(f"\n  Classical x (normalised):  {x_classical_norm}")
    print(f"  HHL x (normalised):        {x_hhl_real}")

    print(f"\n  Element-wise comparison:")
    for i in range(N):
        print(f"    x[{i}]: classical = {x_classical_norm[i]:+.6f}, "
              f"HHL = {x_hhl_real[i]:+.6f}")

    fidelity = np.abs(np.dot(np.conj(x_hhl_norm), x_classical_norm))**2
    print(f"\n  Fidelity: {fidelity:.6f}")

    if fidelity > 0.95:
        print("  ✓ HHL matches the classical solution!")
    elif fidelity > 0.80:
        print("  ~ Approximate match — phase estimation errors present")
    else:
        print("  ✗ Mismatch — check parameters")

    if probs_shots is not None:
        print("\n" + "=" * 60)
        print("COMPARISON: Shot-based Simulation")
        print("=" * 60)

        # Reconstruct amplitudes from probabilities
        # Note: sign information is lost in measurement
        x_shots = np.sqrt(probs_shots)

        # Try to assign signs from classical solution
        for i in range(N):
            if x_classical_norm[i] < 0:
                x_shots[i] = -x_shots[i]

        x_shots_norm = x_shots / np.linalg.norm(x_shots) if np.linalg.norm(x_shots) > 1e-12 else x_shots

        print(f"\n  Classical x (normalised):  {x_classical_norm}")
        print(f"  Shots x (normalised):      {x_shots_norm}")

        fidelity_shots = np.abs(np.dot(x_shots_norm, x_classical_norm))**2
        print(f"\n  Fidelity: {fidelity_shots:.6f}")


def align_phase(x_hhl, x_classical):
    """Remove global phase from HHL solution to align with classical solution.
    Finds the phase factor e^{i*phi} that best aligns the two vectors."""
    # Find the component with largest magnitude in classical solution
    idx = np.argmax(np.abs(x_classical))

    if np.abs(x_hhl[idx]) < 1e-12:
        return np.real(x_hhl)

    # Compute phase difference at that component
    phase = x_classical[idx] / x_hhl[idx]
    phase = phase / np.abs(phase)

    return np.real(x_hhl * phase)


def decompose_hermitian(A):
    """Decompose a Hermitian matrix A into a sum of tensor products
    of Pauli matrices. Returns a list of (coefficient, pauli_string) pairs."""
    from qiskit.quantum_info import SparsePauliOp
    n = int(np.log2(A.shape[0]))
    op = SparsePauliOp.from_operator(A)
    return op


def build_trotter_circuit(A, t, n_system, trotter_steps=1):
    """Build a first-order Trotter approximation to e^{iAt}.
    Decomposes A into Pauli terms and exponentiates each separately."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter
    from qiskit.quantum_info import SparsePauliOp

    op = SparsePauliOp.from_operator(A)

    evolution_gate = PauliEvolutionGate(
        op,
        time=t,
        synthesis=LieTrotter(reps=trotter_steps)
    )

    qc = QuantumCircuit(n_system)
    qc.append(evolution_gate, range(n_system))

    return qc

def run_hhl(A, b, n_clock=None, use_trotter=False, trotter_steps=4):
    """Main function to run HHL and compare with classical solution."""

    validate_inputs(A, b)

    info = get_system_info(A, b)
    N = info['N']
    n_system = info['n_system']
    eigenvalues = info['eigenvalues']
    kappa = info['kappa']
    x_classical_norm = info['x_classical_norm']

    if n_clock is None:
        n_clock = max(2, n_system + 2)

    # Auto-enable Trotter for larger systems
    if n_system >= 3 and not use_trotter:
        print(f"  Note: consider use_trotter=True for {N}x{N} systems")

    t0, C = choose_parameters(eigenvalues, n_clock)

    print("=" * 60)
    print("HHL Algorithm Implementation")
    print("=" * 60)
    print(f"\n  System size:      {N} x {N}")
    print(f"  System qubits:    {n_system}")
    print(f"  Clock qubits:     {n_clock}")
    print(f"  Total qubits:     {n_system + n_clock + 1}")
    print(f"  Method:           {'Trotter' if use_trotter else 'Dense expm'}")
    if use_trotter:
        print(f"  Trotter steps:    {trotter_steps}")
    print(f"\n  Eigenvalues:      {eigenvalues}")
    print(f"  Condition number: {kappa:.4f}")
    print(f"  t0:               {t0:.6f}")
    print(f"  C:                {C:.6f}")
    print(f"\n  Classical solution (normalised): {x_classical_norm}")

    eigenvalue_clock_map = build_eigenvalue_clock_map(eigenvalues, t0, n_clock)
    print(f"\n  Eigenvalue -> Clock register mapping:")
    for lam, info_dict in eigenvalue_clock_map.items():
        print(f"    λ = {lam:.4f} -> k = {info_dict['k_rounded']} -> "
              f"λ_rounded = {info_dict['lam_rounded']:.4f} -> "
              f"ctrl = {info_dict['ctrl_state']}")

    print(f"\n  Building HHL circuit...")
    qc, sys_q, clk_q, anc_q = build_hhl_circuit(
        A, b, n_clock, t0, C, eigenvalues, n_system,
        use_trotter=use_trotter, trotter_steps=trotter_steps
    )
    print(f"  Circuit depth: {qc.depth()}")
    print(f"  Gate count:    {qc.size()}")

    print(f"\n  Running statevector simulation...")
    x_hhl_norm, p_success = extract_solution_statevector(
        qc, sys_q, clk_q, anc_q, n_system, n_clock
    )
    print(f"  Post-selection probability: {p_success:.6f}")

    shots = 100000
    print(f"\n  Running shot-based simulation ({shots} shots)...")
    probs_shots, post_counts, p_success_shots = extract_solution_shots(
        qc, sys_q, clk_q, anc_q, n_system, n_clock, shots=shots
    )
    print(f"  Post-selection rate: {p_success_shots:.4f}")
    print(f"  Post-selected counts: {post_counts}")

    compare_solutions(x_classical_norm, x_hhl_norm, probs_shots)

    print(f"\n{'=' * 60}")
    print("CIRCUIT")
    print("=" * 60)
    print(qc.draw(output='text', fold=120))

    return {
        'circuit': qc,
        'x_classical': x_classical_norm,
        'x_hhl': x_hhl_norm,
        'fidelity': np.abs(np.dot(np.conj(x_hhl_norm), x_classical_norm))**2,
        'p_success': p_success,
        'eigenvalues': eigenvalues,
        'kappa': kappa,
        'info': info,
    }

def export_circuit_diagrams(A, b, n_clock, label='', use_trotter=False):
    """Export circuit diagrams for the dissertation."""
    info = get_system_info(A, b)
    eigenvalues = info['eigenvalues']
    n_system = info['n_system']
    t0, C = choose_parameters(eigenvalues, n_clock)

    # Full HHL circuit
    qc, sys_q, clk_q, anc_q = build_hhl_circuit(
        A, b, n_clock, t0, C, eigenvalues, n_system,
        use_trotter=use_trotter
    )

    fig = qc.draw(output='mpl', fold=30, style={'fontsize': 10})
    fig.savefig(f'hhl_circuit_{label}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hhl_circuit_{label}.png")

    # Circuit stats
    stats = {
        'depth': qc.depth(),
        'gate_count': qc.size(),
        'n_system': n_system,
        'n_clock': n_clock,
        'total_qubits': n_system + n_clock + 1,
    }

    return stats

def generate_examples(include_32=False):
    """Generate test examples for Chapter 6."""
    examples = {}

    # 2x2: eigenvalues 1, 2 (kappa = 2)
    examples['2x2'] = {
        'A': np.array([[1.5, 0.5],
                        [0.5, 1.5]]),
        'b': np.array([1.0, 0.0]),
        'n_clock': 2,
    }

    # 4x4
    Q4, _ = np.linalg.qr(np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1]
    ], dtype=float))
    examples['4x4'] = {
        'A': Q4 @ np.diag([1.0, 2.0, 3.0, 4.0]) @ Q4.T,
        'b': np.array([1.0, 1.0, 0.0, 0.0]),
        'n_clock': 4,
    }

    # 8x8
    np.random.seed(42)
    Q8, _ = np.linalg.qr(np.random.randn(8, 8))
    examples['8x8'] = {
        'A': Q8 @ np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) @ Q8.T,
        'b': np.zeros(8),
        'n_clock': 6,
    }
    examples['8x8']['b'][0] = 1.0
    examples['8x8']['b'][7] = 1.0
    examples['8x8']['b'] /= np.linalg.norm(examples['8x8']['b'])

    # 16x16
    np.random.seed(43)
    Q16, _ = np.linalg.qr(np.random.randn(16, 16))
    eigs16 = np.arange(1, 17, dtype=float)
    examples['16x16'] = {
        'A': Q16 @ np.diag(eigs16) @ Q16.T,
        'b': np.zeros(16),
        'n_clock': 8,
    }
    examples['16x16']['b'][0] = 1.0
    examples['16x16']['b'][15] = 1.0
    examples['16x16']['b'] /= np.linalg.norm(examples['16x16']['b'])

    # 32x32 — optional, slow
    if include_32:
        np.random.seed(44)
        Q32, _ = np.linalg.qr(np.random.randn(32, 32))
        eigs32 = np.arange(1, 33, dtype=float)
        examples['32x32'] = {
            'A': Q32 @ np.diag(eigs32) @ Q32.T,
            'b': np.zeros(32),
            'n_clock': 14,
        }
        examples['32x32']['b'][0] = 1.0
        examples['32x32']['b'][31] = 1.0
        examples['32x32']['b'] /= np.linalg.norm(examples['32x32']['b'])

    return examples


def run_all_examples():
    """Run all examples and collect results."""
    examples = generate_examples()
    results = {}

    for name, ex in examples.items():
        print(f"\n\n{'#' * 60}")
        print(f"  EXAMPLE: {name}")
        print(f"{'#' * 60}\n")
        results[name] = run_hhl(ex['A'], ex['b'], n_clock=ex['n_clock'])

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'=' * 80}")
    print(f"{'System':<10} {'Qubits':<10} {'Kappa':<10} {'Fidelity':<12} "
          f"{'P(success)':<12} {'Depth':<10}")
    print(f"{'-' * 80}")

    for name, r in results.items():
        n_total = r['info']['n_system'] + r['circuit'].num_qubits - r['info']['n_system']
        print(f"{name:<10} {r['circuit'].num_qubits:<10} "
              f"{r['kappa']:<10.2f} {r['fidelity']:<12.6f} "
              f"{r['p_success']:<12.6f} {r['circuit'].depth():<10}")

    return results


def run_classical_comparison(examples=None):
    """Compare HHL runtime and accuracy with classical solvers."""
    if examples is None:
        examples = generate_examples()

    print(f"\n{'=' * 90}")
    print(f"{'CLASSICAL vs HHL COMPARISON':^90}")
    print(f"{'=' * 90}")
    print(f"{'System':<10} {'N':<6} {'Kappa':<8} {'Classical (s)':<15} "
          f"{'HHL Build (s)':<15} {'HHL Sim (s)':<15} {'Fidelity':<12}")
    print(f"{'-' * 90}")

    results = {}

    for name, ex in examples.items():
        A = ex['A']
        b = ex['b']
        n_clock = ex['n_clock']
        N = A.shape[0]

        info = get_system_info(A, b)
        eigenvalues = info['eigenvalues']
        kappa = info['kappa']
        x_classical_norm = info['x_classical_norm']

        # Time classical solution
        t_start = time.time()
        for _ in range(100):  # average over 100 runs
            x_cl = np.linalg.solve(A, b)
        t_classical = (time.time() - t_start) / 100

        # Time HHL circuit build
        t0, C = choose_parameters(eigenvalues, n_clock)
        t_start = time.time()
        qc, sys_q, clk_q, anc_q = build_hhl_circuit(
            A, b, n_clock, t0, C, eigenvalues, info['n_system']
        )
        t_build = time.time() - t_start

        # Time HHL simulation
        t_start = time.time()
        x_hhl_norm, p_success = extract_solution_statevector(
            qc, sys_q, clk_q, anc_q, info['n_system'], n_clock
        )
        t_sim = time.time() - t_start

        fidelity = np.abs(np.dot(np.conj(x_hhl_norm), x_classical_norm))**2

        print(f"{name:<10} {N:<6} {kappa:<8.2f} {t_classical:<15.6f} "
              f"{t_build:<15.4f} {t_sim:<15.4f} {fidelity:<12.6f}")

        results[name] = {
            'N': N,
            'kappa': kappa,
            't_classical': t_classical,
            't_build': t_build,
            't_sim': t_sim,
            'fidelity': fidelity,
            'p_success': p_success,
            'depth': qc.depth(),
        }

    # Print repetition-adjusted comparison
    print(f"\n{'=' * 90}")
    print(f"{'REPETITION-ADJUSTED COMPARISON':^90}")
    print(f"{'=' * 90}")
    print(f"{'System':<10} {'P(success)':<12} {'Expected reps':<15} "
          f"{'HHL total (s)':<15} {'Classical (s)':<15} {'Ratio':<10}")
    print(f"{'-' * 90}")

    for name, r in results.items():
        reps = 1.0 / r['p_success'] if r['p_success'] > 1e-12 else float('inf')
        hhl_total = reps * r['t_sim']
        ratio = hhl_total / r['t_classical'] if r['t_classical'] > 1e-12 else float('inf')
        print(f"{name:<10} {r['p_success']:<12.6f} {reps:<15.1f} "
              f"{hhl_total:<15.4f} {r['t_classical']:<15.6f} {ratio:<10.1f}")

    return results

def run_shot_comparison(examples=None, shots=100000):
    """Generate shot-based results table for Chapter 6."""
    if examples is None:
        examples = generate_examples()

    # Only run manageable sizes
    skip = ['32x32']

    print(f"\n{'=' * 85}")
    print(f"{'SHOT-BASED vs STATEVECTOR COMPARISON':^85}")
    print(f"{'=' * 85}")
    print(f"{'System':<10} {'SV Fidelity':<14} {'Shot Fidelity':<14} "
          f"{'p(SV)':<10} {'p(shots)':<10} {'Post shots':<12}")
    print(f"{'-' * 85}")

    results = {}

    for name, ex in examples.items():
        if name in skip:
            continue

        A = ex['A']
        b = ex['b']
        nc = ex['n_clock']

        info = get_system_info(A, b)
        eigenvalues = info['eigenvalues']
        n_system = info['n_system']
        x_classical_norm = info['x_classical_norm']

        t0, C = choose_parameters(eigenvalues, nc)
        qc, sys_q, clk_q, anc_q = build_hhl_circuit(
            A, b, nc, t0, C, eigenvalues, n_system
        )

        # Statevector
        x_sv, p_sv = extract_solution_statevector(
            qc, sys_q, clk_q, anc_q, n_system, nc
        )
        fid_sv = np.abs(np.dot(np.conj(x_sv), x_classical_norm))**2

        # Shots
        probs, counts, p_shots = extract_solution_shots(
            qc, sys_q, clk_q, anc_q, n_system, nc, shots=shots
        )
        total_post = sum(counts.values()) if counts else 0

        # Reconstruct amplitudes from probabilities with sign from classical
        x_shots = np.sqrt(probs)
        for i in range(len(x_classical_norm)):
            if x_classical_norm[i] < 0:
                x_shots[i] = -x_shots[i]
        norm = np.linalg.norm(x_shots)
        if norm > 1e-12:
            x_shots = x_shots / norm
        fid_shots = np.abs(np.dot(x_shots, x_classical_norm))**2

        print(f"{name:<10} {fid_sv:<14.6f} {fid_shots:<14.6f} "
              f"{p_sv:<10.6f} {p_shots:<10.4f} {total_post:<12}")

        results[name] = {
            'fid_sv': fid_sv,
            'fid_shots': fid_shots,
            'p_sv': p_sv,
            'p_shots': p_shots,
            'total_post': total_post,
        }

    return results


def experiment_1_precision(n_clock_range=None, label='4x4_nonint'):
    """Experiment 1: Fidelity vs clock qubits with full rotation."""

    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(4, 4))
    eigs = np.array([0.5, 2.0, 3.5, 5.5])
    A = Q @ np.diag(eigs) @ Q.T

    b = np.array([1.0, 0.0, 0.0, 1.0])
    b = b / np.linalg.norm(b)

    if n_clock_range is None:
        n_clock_range = [3, 4, 5, 6, 7, 8]

    info = get_system_info(A, b)
    eigenvalues = info['eigenvalues']
    kappa = info['kappa']
    n_system = info['n_system']
    x_classical_norm = info['x_classical_norm']

    t0 = 1.0
    C = np.min(np.abs(eigenvalues))

    print(f"\n{'=' * 75}")
    print(f"{'EXPERIMENT 1: Phase Estimation Precision vs Fidelity':^75}")
    print(f"{'=' * 75}")
    print(f"  System: 4x4, kappa = {kappa:.2f}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  t0 = {t0} (fixed)")
    print(f"  Using full_rotation=True")
    print(f"\n{'nc':<6} {'Resolution':<12} {'Fidelity':<12} "
          f"{'p_success':<12} {'Depth':<8} {'Rotations':<10}")
    print(f"{'-' * 75}")

    results = []
    for nc in n_clock_range:
        qc, sys_q, clk_q, anc_q = build_hhl_circuit(
            A, b, nc, t0, C, eigenvalues, n_system,
            full_rotation=True
        )
        x_hhl, p_success = extract_solution_statevector(
            qc, sys_q, clk_q, anc_q, n_system, nc
        )
        fidelity = np.abs(
            np.dot(np.conj(x_hhl), x_classical_norm)
        ) ** 2
        resolution = 2 * np.pi / (2 ** nc)
        depth = qc.depth()
        n_rotations = 2 ** nc - 1

        results.append({
            'n_clock': nc,
            'resolution': resolution,
            'fidelity': fidelity,
            'p_success': p_success,
            'depth': depth,
            'n_rotations': n_rotations,
        })
        print(f"{nc:<6} {resolution:<12.4f} {fidelity:<12.6f} "
              f"{p_success:<12.6f} {depth:<8} {n_rotations:<10}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ncs = [r['n_clock'] for r in results]
    fids = [r['fidelity'] for r in results]
    ress = [r['resolution'] for r in results]

    color1 = 'tab:blue'
    color2 = 'tab:red'

    ax1.plot(ncs, fids, 'o-', color=color1, linewidth=2,
             markersize=8, label='Measured fidelity')
    ax1.set_xlabel('Clock qubits $n_c$', fontsize=12)
    ax1.set_ylabel('Fidelity', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1.05])
    ax1.set_xticks(ncs)

    ax2 = ax1.twinx()
    ax2.plot(ncs, ress, 's--', color=color2, linewidth=2,
             markersize=8, label=r'Resolution $2\pi/2^{n_c}$')
    ax2.set_ylabel('Eigenvalue resolution', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
               fontsize=11)

    plt.title(f'Experiment 1: Precision vs Fidelity '
              rf'($\kappa = {kappa:.1f}$)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figures/exp1_precision_{label}.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: figures/exp1_precision_{label}.png")

    return results

def experiment_2_kappa(n_system=2, kappas=None, label='kappa_scaling'):
    """Experiment 2: Success probability and fidelity vs condition number
    at fixed system size."""
    if kappas is None:
        kappas = [2, 4, 8, 16, 32, 64]

    N = 2 ** n_system
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(N, N))

    b = np.ones(N)
    b = b / np.linalg.norm(b)

    print(f"\n{'=' * 80}")
    print(f"{'EXPERIMENT 2: Condition Number Scaling':^80}")
    print(f"{'=' * 80}")
    print(f"  System size: {N}x{N}")
    print(f"  Kappas tested: {kappas}")
    print(f"\n{'kappa':<8} {'nc':<6} {'Fidelity':<12} {'p_success':<12} "
          f"{'1/kappa^2':<12} {'Depth':<8}")
    print(f"{'-' * 80}")

    results = []
    for kappa in kappas:
        # Eigenvalues from 1 to kappa (integer endpoints)
        eigs = np.array([1.0, float(kappa)])
        if N > 2:
            eigs = np.linspace(1, kappa, N)
        A = Q @ np.diag(eigs) @ Q.T

        # Choose nc so that integer eigenvalues are exact
        nc = max(3, int(np.ceil(np.log2(kappa))) + 1)

        info = get_system_info(A, b)
        eigenvalues = info['eigenvalues']
        x_classical_norm = info['x_classical_norm']

        t0, C = choose_parameters(eigenvalues, nc)

        qc, sys_q, clk_q, anc_q = build_hhl_circuit(
            A, b, nc, t0, C, eigenvalues, n_system
        )
        x_hhl, p_success = extract_solution_statevector(
            qc, sys_q, clk_q, anc_q, n_system, nc
        )
        fidelity = np.abs(
            np.dot(np.conj(x_hhl), x_classical_norm)
        ) ** 2
        bound = 1.0 / kappa ** 2
        depth = qc.depth()

        results.append({
            'kappa': kappa,
            'n_clock': nc,
            'fidelity': fidelity,
            'p_success': p_success,
            'p_lower_bound': bound,
            'depth': depth,
            'eigenvalues': eigs,
        })
        print(f"{kappa:<8} {nc:<6} {fidelity:<12.6f} {p_success:<12.6f} "
              f"{bound:<12.6f} {depth:<8}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ks = [r['kappa'] for r in results]
    ps = [r['p_success'] for r in results]
    bounds = [r['p_lower_bound'] for r in results]
    fids = [r['fidelity'] for r in results]

    # Left plot: success probability (log-log)
    ax1.loglog(ks, ps, 'bo-', linewidth=2, markersize=8,
               label='Measured $p_1$')
    ax1.loglog(ks, bounds, 'r--', linewidth=2, markersize=8,
               label=r'$1/\kappa^2$ bound')
    ax1.set_xlabel(r'Condition number $\kappa$', fontsize=12)
    ax1.set_ylabel('Success probability', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_title('Success Probability vs $\\kappa$', fontsize=13)
    ax1.grid(True, which='both', alpha=0.3)

    # Right plot: fidelity
    ax2.semilogx(ks, fids, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel(r'Condition number $\kappa$', fontsize=12)
    ax2.set_ylabel('Fidelity', fontsize=12)
    ax2.set_ylim([0, 1.05])
    ax2.set_title('Fidelity vs $\\kappa$', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'figures/exp2_{label}.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: figures/exp2_{label}.png")

    return results

def experiment_3_noise(error_rates=None, shots=200000, label='noise'):
    """Experiment 3: Effect of gate noise on HHL performance.
    Uses the 2x2 system (shallowest circuit, perfect noiseless fidelity)
    with depolarizing noise at varying rates."""

    if error_rates is None:
        error_rates = [0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    # 2x2 system: perfect fidelity when noiseless
    A = np.array([[1.5, 0.5],
                   [0.5, 1.5]])
    b = np.array([1.0, 0.0])

    info = get_system_info(A, b)
    eigenvalues = info['eigenvalues']
    n_system = info['n_system']
    x_classical_norm = info['x_classical_norm']
    nc = 2

    t0, C = choose_parameters(eigenvalues, nc)
    qc, sys_q, clk_q, anc_q = build_hhl_circuit(
        A, b, nc, t0, C, eigenvalues, n_system
    )
    n_total = n_system + nc + 1

    # Add measurements
    qc_meas = qc.copy()
    cr = ClassicalRegister(n_total, 'meas')
    qc_meas.add_register(cr)
    for i in range(n_total):
        qc_meas.measure(i, i)

    # Transpile once to basis gates
    backend = AerSimulator()
    qc_transpiled = transpile(qc_meas, backend, optimization_level=0)
    basis_depth = qc_transpiled.depth()
    basis_count = qc_transpiled.size()

    print(f"\n{'=' * 80}")
    print(f"{'EXPERIMENT 3: Noise Modelling':^80}")
    print(f"{'=' * 80}")
    print(f"  System: 2x2, kappa = {info['kappa']:.2f}")
    print(f"  Clock qubits: {nc}")
    print(f"  Transpiled circuit: depth={basis_depth}, gates={basis_count}")
    print(f"  Shots per run: {shots}")
    print(f"\n{'p_err':<10} {'Fidelity':<12} {'p_success':<12} "
          f"{'Post shots':<14} {'Est. circuit err':<16}")
    print(f"{'-' * 80}")

    results = []
    clock_zeros = '0' * nc

    for p_err in error_rates:
        if p_err == 0:
            noise_model = None
        else:
            noise_model = NoiseModel()
            error_1q = depolarizing_error(p_err, 1)
            error_2q = depolarizing_error(10 * p_err, 2)
            noise_model.add_all_qubit_quantum_error(
                error_1q, ['sx', 'x', 'rz', 'h', 'ry']
            )
            noise_model.add_all_qubit_quantum_error(
                error_2q, ['cx']
            )

        if noise_model is not None:
            noisy_backend = AerSimulator(noise_model=noise_model)
        else:
            noisy_backend = AerSimulator()

        qc_t = transpile(qc_meas, noisy_backend, optimization_level=0)
        job = noisy_backend.run(qc_t, shots=shots)
        counts = job.result().get_counts()

        # Post-select
        N = 2 ** n_system
        post_selected = {}
        total_post = 0

        for bitstr, count in counts.items():
            anc_bit = bitstr[0]
            clk_bits = bitstr[1:1 + nc]
            sys_bits = bitstr[1 + nc:]

            if anc_bit == '1' and clk_bits == clock_zeros:
                post_selected[sys_bits] = (
                    post_selected.get(sys_bits, 0) + count
                )
                total_post += count

        p_success = total_post / shots

        probs = np.zeros(N)
        if total_post > 0:
            for sys_bits, count in post_selected.items():
                sys_idx = int(sys_bits[::-1], 2)
                probs[sys_idx] = count / total_post

        # Reconstruct amplitudes with sign from classical
        x_shots = np.sqrt(probs)
        for i in range(N):
            if x_classical_norm[i] < 0:
                x_shots[i] = -x_shots[i]
        norm = np.linalg.norm(x_shots)
        if norm > 1e-12:
            x_shots = x_shots / norm
        fidelity = np.abs(np.dot(x_shots, x_classical_norm)) ** 2

        # Estimate total circuit error
        est_err = 1.0 - (1.0 - p_err) ** basis_count if p_err > 0 else 0.0

        results.append({
            'error_rate': p_err,
            'fidelity': fidelity,
            'p_success': p_success,
            'total_post': total_post,
            'est_circuit_err': est_err,
        })
        print(f"{p_err:<10.1e} {fidelity:<12.6f} {p_success:<12.6f} "
              f"{total_post:<14} {est_err:<16.4f}")

    # Noiseless statevector fidelity for reference
    x_sv, p_sv = extract_solution_statevector(
        qc, sys_q, clk_q, anc_q, n_system, nc
    )
    fid_sv = np.abs(np.dot(np.conj(x_sv), x_classical_norm)) ** 2

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ers = [r['error_rate'] for r in results]
    fids = [r['fidelity'] for r in results]
    ps = [r['p_success'] for r in results]

    # Use small offset for log scale on the p_err=0 point
    ers_plot = [max(e, 5e-5) for e in ers]

    ax1.semilogx(ers_plot, fids, 'bo-', linewidth=2, markersize=8,
                 label='Measured fidelity')
    ax1.axhline(y=fid_sv, color='r', linestyle='--', linewidth=2,
                label=f'Noiseless (statevector) = {fid_sv:.3f}')
    ax1.set_xlabel('Depolarizing error rate $p$', fontsize=12)
    ax1.set_ylabel('Fidelity', fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=10)
    ax1.set_title('Fidelity vs Gate Error Rate', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(ers_plot, ps, 'go-', linewidth=2, markersize=8,
                 label='Measured $p_1$')
    ax2.axhline(y=p_sv, color='r', linestyle='--', linewidth=2,
                label=f'Noiseless = {p_sv:.3f}')
    ax2.set_xlabel('Depolarizing error rate $p$', fontsize=12)
    ax2.set_ylabel('Post-selection probability', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_title('Success Probability vs Gate Error Rate', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'figures/exp3_{label}.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"\n  Noiseless statevector fidelity: {fid_sv:.6f}")
    print(f"  Plot saved: figures/exp3_{label}.png")

    return results


if __name__ == "__main__":

    # Example 1: 2x2 system (same as your working code)
    # print("\n>>> Example 1: 2x2 system (eigenvalues 1 and 2)\n")
    # A1 = np.array([
    #     [1.5, 0.5],
    #     [0.5, 1.5]
    # ])
    # b1 = np.array([1.0, 0.0])
    # result1 = run_hhl(A1, b1, n_clock=2)
    #
    # # Example 2: 2x2 system with different b
    # print("\n\n>>> Example 2: 2x2 system with b = [1, 1]/sqrt(2)\n")
    # A2 = np.array([
    #     [1.5, 0.5],
    #     [0.5, 1.5]
    # ])
    # b2 = np.array([1.0, 1.0])
    # result2 = run_hhl(A2, b2, n_clock=2)
    #
    # # Example 3: 2x2 system with higher condition number
    # print("\n\n>>> Example 3: 2x2 system with higher condition number\n")
    # A3 = np.array([
    #     [2, 1],
    #     [1, 3]
    # ])
    # b3 = np.array([1.0, 0.0])
    # result3 = run_hhl(A3, b3, n_clock=4)
    #
    # # Example 4: 4x4 system
    # print("\n\n>>> Example 4: 4x4 system\n")
    # A4 = np.array([
    #     [4, 1, 0, 0],
    #     [1, 3, 1, 0],
    #     [0, 1, 2, 1],
    #     [0, 0, 1, 5]
    # ])
    # b4 = np.array([1.0, 0.0, 0.0, 1.0])
    # result4 = run_hhl(A4, b4, n_clock=6)

    #Example 5: User input
    # print("\n\n>>> Example 5: Custom input\n")
    # try:
    #     n = int(input("Enter dimension N (must be power of 2): "))
    #     print(f"Enter {n}x{n} Hermitian matrix A row by row:")
    #     A_custom = np.zeros((n, n), dtype=float)
    #     for i in range(n):
    #         row = input(f"  Row {i}: ").split()
    #         A_custom[i] = [float(x) for x in row]
    #
    #     print(f"Enter vector b ({n} space-separated values):")
    #     b_custom = np.array([float(x) for x in input("  b: ").split()])
    #
    #     nc = int(input("Enter number of clock qubits: "))
    #     result_custom = run_hhl(A_custom, b_custom, n_clock=nc)
    #
    # except (ValueError, KeyboardInterrupt, EOFError):
    #     print("\nSkipping custom input.")

    # Example: 24 qubit system (256 x 256 matrix)
    # print("\n\n>>> Large system: 256x256 (24 qubits)\n")
    # n = 256
    # n_system = 8
    # n_clock = 15
    #
    # # Build a random sparse Hermitian matrix with controlled condition number
    # np.random.seed(42)  # for reproducibility
    #
    # # Generate random eigenvalues between 1 and 10 (condition number = 10)
    # eigenvalues_custom = np.random.uniform(1.0, 10.0, size=n)
    #
    # # Build random orthogonal matrix for eigenvectors
    # Q, _ = np.linalg.qr(np.random.randn(n, n))
    #
    # # Construct A = Q * diag(eigenvalues) * Q^T
    # A_large = Q @ np.diag(eigenvalues_custom) @ Q.T
    #
    # # Random b vector
    # b_large = np.random.randn(n)
    #
    # result_large = run_hhl(A_large, b_large, n_clock=n_clock)
    #
    #
    # Practical large example: 32 x 32 (18 qubits)
    # print("\n\n>>> Large system: 32x32 (18 qubits)\n")
    # n = 32
    # np.random.seed(42)
    # eigenvalues_custom = np.random.uniform(1.0, 10.0, size=n)
    # Q, _ = np.linalg.qr(np.random.randn(n, n))
    # A_med = Q @ np.diag(eigenvalues_custom) @ Q.T
    # b_med = np.random.randn(n)
    # result_med = run_hhl(A_med, b_med, n_clock=12, use_trotter=False)

    # Generate diagrams for dissertation

    # 2x2 circuit diagram
    # A1 = np.array([[1.5, 0.5], [0.5, 1.5]])
    # b1 = np.array([1.0, 0.0])
    # stats1 = export_circuit_diagrams(A1, b1, n_clock=2, label='2x2')
    #
    # # 4x4 circuit diagram
    # A4 = np.array([[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,5]], dtype=float)
    # b4 = np.array([1.0, 0.0, 0.0, 1.0])
    # stats4 = export_circuit_diagrams(A4, b4, n_clock=6, label='4x4')
    #
    # print(f"\n2x2 stats: {stats1}")
    # print(f"4x4 stats: {stats4}")

    results = run_all_examples()
    comparison = run_classical_comparison()
    shot_results = run_shot_comparison()
    exp1 = experiment_1_precision()
    exp2 = experiment_2_kappa()
    exp3 = experiment_3_noise()




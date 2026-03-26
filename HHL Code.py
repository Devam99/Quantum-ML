import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, RYGate, QFTGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

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


def build_hhl_circuit(A, b, n_clock, t0, C, eigenvalues, n_system):
    """Build the full HHL circuit for arbitrary A and b."""
    N = 2 ** n_system
    n_total = n_clock + n_system + 1

    # Qubit layout:
    # [0, ..., n_system-1]          -> system register
    # [n_system, ..., n_system+n_clock-1] -> clock register
    # [n_system+n_clock]            -> ancilla
    sys_q = list(range(n_system))
    clk_q = list(range(n_system, n_system + n_clock))
    anc_q = n_system + n_clock

    qc = QuantumCircuit(n_total, name='HHL')

    # ---- Step 1: Prepare |b> in the system register ----
    b_norm = b / np.linalg.norm(b)
    qc.initialize(b_norm, sys_q)

    # ---- Step 2: QPE ----
    # Hadamard on all clock qubits
    qc.h(clk_q)

    # Controlled-U^{2^j} for each clock qubit
    for j, c in enumerate(clk_q):
        U_power = expm(1j * A * t0 * (2 ** j))
        gate = UnitaryGate(U_power, label=f'U^{2 ** j}')
        qc.append(gate.control(1), [c] + sys_q)

    # Inverse QFT on clock register
    iqft = QFTGate(n_clock).inverse()
    qc.append(iqft, clk_q)

    qc.barrier(label='QPE done')

    # ---- Step 3: Controlled rotations ----
    eigenvalue_clock_map = build_eigenvalue_clock_map(eigenvalues, t0, n_clock)

    for lam, ctrl_state in eigenvalue_clock_map.items():
        ratio = C / lam
        if np.abs(ratio) > 1:
            ratio = np.sign(ratio)
        theta = 2 * np.arcsin(ratio)
        cry_gate = RYGate(theta).control(n_clock, ctrl_state=ctrl_state)
        qc.append(cry_gate, clk_q + [anc_q])

    qc.barrier(label='Rotations done')

    # ---- Step 4: Inverse QPE ----
    qft = QFTGate(n_clock)
    qc.append(qft, clk_q)

    for j in reversed(range(n_clock)):
        U_dag = expm(-1j * A * t0 * (2 ** j))
        gate = UnitaryGate(U_dag, label=f'U†^{2 ** j}')
        qc.append(gate.control(1), [clk_q[j]] + sys_q)

    qc.h(clk_q)

    qc.barrier(label='Inverse QPE done')

    return qc, sys_q, clk_q, anc_q


def build_eigenvalue_clock_map(eigenvalues, t0, n_clock):
    """Map each eigenvalue to its binary control string for the clock register.

    After QPE, eigenvalue lambda_j is stored as integer
    k_j = lambda_j * t0 * 2^n_clock / (2*pi) in the clock register.

    The control string must match Qiskit's qubit ordering for the
    multi-controlled gate."""
    clock_map = {}

    for lam in eigenvalues:
        k = lam * t0 * (2 ** n_clock) / (2 * np.pi)
        k_rounded = int(np.round(k)) % (2 ** n_clock)

        if k_rounded == 0:
            continue

        # Binary string with n_clock bits
        # Qiskit ctrl_state expects LSB first ordering
        ctrl_state = format(k_rounded, f'0{n_clock}b')

        if lam not in clock_map:
            clock_map[lam] = ctrl_state

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

def run_hhl(A, b, n_clock=None):
    """Main function to run HHL and compare with classical solution."""

    # Validate inputs
    validate_inputs(A, b)

    # Get system info
    info = get_system_info(A, b)
    N = info['N']
    n_system = info['n_system']
    eigenvalues = info['eigenvalues']
    kappa = info['kappa']
    x_classical_norm = info['x_classical_norm']

    # Choose n_clock automatically if not specified
    if n_clock is None:
        # Use enough clock qubits to represent eigenvalues
        # At minimum 2, scale up with system size
        n_clock = max(2, n_system + 2)

    # Choose HHL parameters
    t0, C = choose_parameters(eigenvalues, n_clock)

    # Print system info
    print("=" * 60)
    print("HHL Algorithm Implementation")
    print("=" * 60)
    print(f"\n  System size:      {N} x {N}")
    print(f"  System qubits:    {n_system}")
    print(f"  Clock qubits:     {n_clock}")
    print(f"  Total qubits:     {n_system + n_clock + 1}")
    print(f"\n  Eigenvalues:      {eigenvalues}")
    print(f"  Condition number: {kappa:.4f}")
    print(f"  t0:               {t0:.6f}")
    print(f"  C:                {C:.6f}")
    print(f"\n  Classical solution (normalised): {x_classical_norm}")

    # Check eigenvalue-to-clock mapping
    eigenvalue_clock_map = build_eigenvalue_clock_map(eigenvalues, t0, n_clock)
    print(f"\n  Eigenvalue -> Clock register mapping:")
    for lam, ctrl_state in eigenvalue_clock_map.items():
        k = lam * t0 * (2**n_clock) / (2 * np.pi)
        print(f"    λ = {lam:.4f} -> k = {k:.2f} -> "
              f"rounded = {int(np.round(k))} -> ctrl = {ctrl_state}")

    # Build circuit
    print(f"\n  Building HHL circuit...")
    qc, sys_q, clk_q, anc_q = build_hhl_circuit(
        A, b, n_clock, t0, C, eigenvalues, n_system
    )
    print(f"  Circuit depth: {qc.depth()}")
    print(f"  Gate count:    {qc.size()}")

    # Statevector simulation
    print(f"\n  Running statevector simulation...")
    x_hhl_norm, p_success = extract_solution_statevector(
        qc, sys_q, clk_q, anc_q, n_system, n_clock
    )
    print(f"  Post-selection probability: {p_success:.6f}")

    # Shot-based simulation
    shots = 100000
    print(f"\n  Running shot-based simulation ({shots} shots)...")
    probs_shots, post_counts, p_success_shots = extract_solution_shots(
        qc, sys_q, clk_q, anc_q, n_system, n_clock, shots=shots
    )
    print(f"  Post-selection rate: {p_success_shots:.4f}")
    print(f"  Post-selected counts: {post_counts}")

    # Compare
    compare_solutions(x_classical_norm, x_hhl_norm, probs_shots)

    # Print circuit
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

    # Example 5: User input
    # print("\n\n>>> Example 5: Custom input\n")
    # try:
    #     #n = int(input("Enter dimension N (must be power of 2): "))
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


    # Practical large example: 32 x 32 (18 qubits)
    # print("\n\n>>> Large system: 32x32 (18 qubits)\n")
    # n = 32
    # np.random.seed(42)
    # eigenvalues_custom = np.random.uniform(1.0, 10.0, size=n)
    # Q, _ = np.linalg.qr(np.random.randn(n, n))
    # A_med = Q @ np.diag(eigenvalues_custom) @ Q.T
    # b_med = np.random.randn(n)
    # result_med = run_hhl(A_med, b_med, n_clock=12)
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from scipy.linalg import expm


def validate_inputs(A, b):
    """Check that A is Hermitian, square, dimensions are power of 2,
    and b has matching dimension."""
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")
    if np.log2(N) != int(np.log2(N)):
        raise ValueError(f"Dimension N={N} must be a power of 2.")
    if b.shape[0] != N:
        raise ValueError(f"b has dimension {b.shape[0]}, expected {N}.")
    if np.linalg.det(A) == 0:
        raise ValueError("A must be invertible.")


def normalise(v):
    """Return the normalised version of a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalise the zero vector.")
    return v / norm


def classical_solve(A, b):
    """Solve Ax = b classically and return the normalised solution."""
    x = np.linalg.solve(A, b)
    return normalise(x)


def get_eigenvalues(A):
    """Return eigenvalues and eigenvectors of A."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvalues, eigenvectors


def choose_t0(eigenvalues, n_clock):
    """Choose the time parameter t0 so that eigenvalues map to phases
    representable in n_clock bits."""
    lambda_max = np.max(np.abs(eigenvalues))
    t0 = 2 * np.pi / (2**n_clock)
    # Scale so that lambda_max * t0 < 2*pi
    t0 = 2 * np.pi / (lambda_max * 1.1)  # small buffer
    return t0


def build_hamiltonian_simulation(A, t, n_system):
    """Build the unitary e^{iAt} as a gate."""
    U = expm(1j * A * t)
    qc = QuantumCircuit(n_system, name=f'e^(iAt)')
    qc.unitary(U, range(n_system))
    return qc.to_gate()


def build_controlled_hamiltonian(A, t, power, n_system):
    """Build controlled-U^{2^power} = controlled-e^{iA * 2^power * t}."""
    U = expm(1j * A * t * (2**power))
    qc = QuantumCircuit(n_system, name=f'c-e^(iA*2^{power}*t)')
    qc.unitary(U, range(n_system))
    return qc.to_gate().control(1)


def build_qpe(A, t0, n_clock, n_system):
    """Build the quantum phase estimation circuit."""
    qc = QuantumCircuit(n_clock + n_system, name='QPE')

    # Step 1: Hadamard on all clock qubits
    for i in range(n_clock):
        qc.h(i)

    # Step 2: Controlled-U^{2^k} operations
    for k in range(n_clock):
        cu = build_controlled_hamiltonian(A, t0, k, n_system)
        control_qubit = k
        target_qubits = list(range(n_clock, n_clock + n_system))
        qc.append(cu, [control_qubit] + target_qubits)

    # Step 3: Inverse QFT on clock register
    qc.append(build_inverse_qft(n_clock), range(n_clock))

    return qc.to_gate()


def build_inverse_qft(n):
    """Build the inverse QFT on n qubits."""
    qc = QuantumCircuit(n, name='QFT†')

    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    for target in range(n):
        for control in range(target):
            angle = -np.pi / (2**(target - control))
            qc.cp(angle, control, target)
        qc.h(target)

    return qc.to_gate()


def build_qft(n):
    """Build the QFT on n qubits (used for inverse QPE)."""
    qc = QuantumCircuit(n, name='QFT')

    for target in range(n - 1, -1, -1):
        qc.h(target)
        for control in range(target - 1, -1, -1):
            angle = np.pi / (2**(target - control))
            qc.cp(angle, control, target)

    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    return qc.to_gate()


def build_controlled_rotation(n_clock):
    """Build the controlled rotation that maps
    |lambda_j>|0> -> |lambda_j>(C/lambda_j|1> + sqrt(1-C^2/lambda_j^2)|0>).
    Implemented as controlled-Ry rotations based on clock register bits."""
    qc = QuantumCircuit(n_clock + 1, name='C-Rot')

    for k in range(n_clock):
        # The k-th clock qubit represents the value 2^{-(k+1)} in the
        # binary fraction. The rotation angle is chosen so that
        # sin(theta/2) = C / lambda, approximated bit by bit.
        theta = 2 * np.arcsin(1 / (2**(n_clock - k)))
        qc.cry(theta, k, n_clock)

    return qc.to_gate()


def build_inverse_qpe(A, t0, n_clock, n_system):
    """Build the inverse QPE circuit."""
    qc = QuantumCircuit(n_clock + n_system, name='QPE†')

    # Inverse of QPE: QFT, then inverse controlled unitaries, then Hadamards
    qc.append(build_qft(n_clock), range(n_clock))

    for k in range(n_clock - 1, -1, -1):
        cu = build_controlled_hamiltonian(A, -t0, k, n_system)
        control_qubit = k
        target_qubits = list(range(n_clock, n_clock + n_system))
        qc.append(cu, [control_qubit] + target_qubits)

    for i in range(n_clock):
        qc.h(i)

    return qc.to_gate()


def build_hhl_circuit(A, b, n_clock):
    """Build the full HHL circuit for a given A and b."""
    N = A.shape[0]
    n_system = int(np.log2(N))
    t0 = choose_t0(get_eigenvalues(A)[0], n_clock)

    # Registers
    clock = QuantumRegister(n_clock, name='clock')
    system = QuantumRegister(n_system, name='system')
    ancilla = QuantumRegister(1, name='ancilla')
    c_ancilla = ClassicalRegister(1, name='c_ancilla')

    qc = QuantumCircuit(clock, system, ancilla, c_ancilla)

    # Prepare |b> in the system register
    b_normalised = normalise(b)
    qc.initialize(b_normalised, system[:])

    # Step 1: QPE
    qpe_qubits = list(range(n_clock)) + list(range(n_clock, n_clock + n_system))
    qc.append(build_qpe(A, t0, n_clock, n_system), qpe_qubits)

    # Step 2: Controlled rotation
    rot_qubits = list(range(n_clock)) + [n_clock + n_system]
    qc.append(build_controlled_rotation(n_clock), rot_qubits)

    # Step 3: Inverse QPE
    qc.append(build_inverse_qpe(A, t0, n_clock, n_system), qpe_qubits)

    # Step 4: Measure ancilla
    qc.measure(ancilla, c_ancilla)

    return qc, t0


def extract_solution(qc, n_clock, n_system):
    """Run the circuit on the statevector simulator and extract the
    solution state conditioned on ancilla = |1>."""
    # Remove measurement for statevector simulation
    qc_copy = qc.remove_final_measurements(inplace=False)

    simulator = AerSimulator(method='statevector')
    qc_copy.save_statevector()
    result = simulator.run(qc_copy).result()
    statevector = result.get_statevector()

    n_total = n_clock + n_system + 1
    amplitudes = np.array(statevector)

    # Extract amplitudes where ancilla qubit = |1>
    # The ancilla is the last qubit in our register ordering
    N_system = 2**n_system
    solution_amplitudes = np.zeros(N_system, dtype=complex)

    for idx in range(len(amplitudes)):
        # Convert index to binary string
        binary = format(idx, f'0{n_total}b')

        # Check if ancilla (last qubit) is 1 and clock register is all 0
        ancilla_bit = binary[-1]
        clock_bits = binary[:n_clock]

        if ancilla_bit == '1' and all(c == '0' for c in clock_bits):
            # Extract system register bits
            system_bits = binary[n_clock:n_clock + n_system]
            system_idx = int(system_bits, 2)
            solution_amplitudes[system_idx] = amplitudes[idx]

    # Normalise
    norm = np.linalg.norm(solution_amplitudes)
    if norm < 1e-12:
        print("Warning: post-selection probability is near zero.")
        return solution_amplitudes

    return solution_amplitudes / norm


def run_hhl(A, b, n_clock=4):
    """Main function to run HHL and compare with classical solution."""
    # Validate
    validate_inputs(A, b)

    N = A.shape[0]
    n_system = int(np.log2(N))

    print("=" * 60)
    print("HHL Algorithm Implementation")
    print("=" * 60)
    print(f"\nSystem size: {N} x {N}")
    print(f"System qubits: {n_system}")
    print(f"Clock qubits: {n_clock}")
    print(f"Total qubits: {n_system + n_clock + 1}")

    # Eigenvalue information
    eigenvalues, _ = get_eigenvalues(A)
    kappa = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
    print(f"\nEigenvalues of A: {eigenvalues}")
    print(f"Condition number: {kappa:.4f}")

    # Classical solution
    x_classical = classical_solve(A, b)
    print(f"\nClassical solution (normalised): {x_classical}")

    # Build and run HHL
    print(f"\nBuilding HHL circuit...")
    qc, t0 = build_hhl_circuit(A, b, n_clock)
    print(f"Circuit depth: {qc.depth()}")
    print(f"Gate count: {qc.count_ops()}")

    print(f"\nRunning HHL on simulator...")
    x_hhl = extract_solution(qc, n_clock, n_system)
    print(f"HHL solution (normalised): {x_hhl}")

    # Comparison
    fidelity = np.abs(np.dot(x_classical.conj(), x_hhl))**2
    print(f"\nFidelity |<x_classical|x_hhl>|^2: {fidelity:.6f}")
    print(f"Element-wise comparison:")
    for i in range(N):
        print(f"  x[{i}]: classical = {x_classical[i]:.6f}, "
              f"HHL = {x_hhl[i]:.6f}")
    print("=" * 60)

    return x_classical, x_hhl, fidelity, qc


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # Example 1: 2x2 system
    print("\n>>> Example 1: 2x2 system\n")
    A1 = np.array([[2, 1],
                    [1, 3]], dtype=float)
    b1 = np.array([1, 0], dtype=float)
    run_hhl(A1, b1, n_clock=4)

    # Example 2: 4x4 system
    print("\n>>> Example 2: 4x4 system\n")
    A2 = np.array([[4, 1, 0, 0],
                    [1, 3, 1, 0],
                    [0, 1, 2, 1],
                    [0, 0, 1, 5]], dtype=float)
    b2 = np.array([1, 0, 0, 1], dtype=float)
    run_hhl(A2, b2, n_clock=6)

    # Example 3: User input
    print("\n>>> Example 3: Custom input\n")
    print("Enter a Hermitian matrix A and vector b.")
    print("(Dimensions must be a power of 2)\n")

    try:
        n = int(input("Enter dimension N: "))
        print(f"Enter {n}x{n} matrix A row by row "
              f"(space-separated values):")
        A_custom = np.zeros((n, n), dtype=complex)
        for i in range(n):
            row = input(f"  Row {i}: ").split()
            A_custom[i] = [complex(x) for x in row]

        print(f"Enter vector b ({n} space-separated values):")
        b_custom = np.array([complex(x) for x in input("  b: ").split()])

        run_hhl(A_custom, b_custom, n_clock=6)

    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nSkipping custom input: {e}")
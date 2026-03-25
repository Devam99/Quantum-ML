import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

def validate_inputs(A, b):
    """Check if A is Hermitian, square, dimensions of power 2, and b has a matching dimension."""
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")
    if np.log2(N) != int(np.log2(N)):
        raise ValueError(f"Dimension N={N} must be a power of 2.")
    if b.shape[0] != N:
        raise ValueError(f"b has dimension {b.shape[0]}, expected {N}.")
    if np.linalg.det(A) == 0:
        raise ValueError("A must be invertible.")

def normalise(v):
    """Return normalised vector v."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("v must be positive.")
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
    """Choose the time parameter t0 so that eigenvalues map to phases representable in the n_clock bits."""
    lambda_max = np.max(np.abs(eigenvalues))
    t0 = 2 * np.pi / (2**n_clock)  #scale so that lambda_max * to t0 < 2*pi
    t0 = 2 * np.pi / (lambda_max * 1.1) #small buffer to ensure all eigenvalues fit within the range of representable phases
    return t0

def build_controlled_hamiltonian(A, t, power, n_system):
    """Build controlled-U^{2^power} = controlled-e^{iA * 2^power * t}."""
    U = expm(1j * A * t * (2**power))
    qc = QuantumCircuit(n_system, name=f'c-e^(iA*2^{power}*t)')
    qc.unitary(U, range(n_system))
    return qc.to_gate().control(1)

def build_qpe(A, t0, n_clock, n_system):
    """Build the Quantum Phase Estimation circuit."""
    qc.QuantumCircuit(n_system + n_system, name='QPE')

    #Step 1: Hadamard on all clock qubits
    for i in range(n_clock):
        qc.h(i)

    #Step 2: Controlled-U^{2^k} operations
    for k in range(n_clock):
        cu = build_controlled_hamiltonian(A, t0, k, n_system)
        control_qubit = k
        target_qubits = list(range(n_clock, n_clock + n_system))
        qc.append(cu, [control_qubit] + target_qubits)

    #Step 3: Inverse QFT on clock register
    qc.append(build_inverse_qft(n_clock), range(n_clock))

    retrun qc.to_gate()

def build_inverse_qft(n):
    """Build the inverse QFT on n qubits."""
    qc = QuantumCircuit(n, name='QFT†')

    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    for target in range(n):
        for control in range(target):
            angle = -np.pi / (2 ** (target - control))
            qc.cp(angle, control, target)
        qc.h(target)
    return qc.to_gate()

def build_qft(n):
    """Build the QFT on n qubits."""
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
    """Build the inverse QPE on n qubits."""
    qc = QuantumCircuit(n_clock + n_system, name='QPE†')

    #Inverse of QPE: QFT, then inverse controlled unitaries, then Hadamard's
    qc.append(build_qft(n_clock), range(n_clock))

    for k in range(n_clock - 1, -1, -1):
        cu = build_controlled_hamiltonian(A, t0, k, n_system)
        control_qubit = k
        target_qubits = list(range(n_clock, n_clock + n_system))
        qc.append(cu, [control_qubit] + target_qubits)


    for i in range(n_clock):
        qc.h(i)

    return qc.to_gate()
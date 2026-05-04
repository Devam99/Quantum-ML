# HHL Quantum Linear Systems Algorithm

Implementation of the Harrow–Hassidim–Lloyd (HHL) algorithm for solving 
linear systems Ax = b using IBM's Qiskit framework. Written as part of a 
Third Year BSc Mathematics dissertation at Queen Mary, University of London.

## Overview

The HHL algorithm solves Hermitian linear systems in time O(κ polylog(N)), 
offering an exponential speedup over classical methods in the system 
dimension N under assumptions of sparsity and bounded condition number κ. 
This implementation tests the algorithm's behaviour on small-scale systems 
using classical simulation, and compares results against numpy's classical 
solver.

## Main File

**`HHL Code.py`** — the complete implementation, containing:

- Circuit construction (state preparation, QPE, controlled rotation, 
  inverse QPE, post-selection)
- Statevector and shot-based simulation via Qiskit Aer
- Three structured experiments from the dissertation:
  - **Experiment 1** (`experiment_1_precision`): Fidelity vs clock register 
    size for a 4×4 system with non-integer eigenvalues
  - **Experiment 2** (`experiment_2_kappa`): Success probability and fidelity 
    scaling with condition number κ ∈ {2, 4, 8, 16, 32, 64}
  - **Experiment 3** (`experiment_3_noise`): Effect of depolarizing gate 
    noise on fidelity and post-selection probability
- Classical timing comparison (`run_classical_comparison`)
- Figure export for all plots

The Hamiltonian simulation is performed by computing e^{iAt} classically 
via `scipy.linalg.expm` and loading the result as a `UnitaryGate`. A 
Trotter-based simulation is also implemented (`build_trotter_circuit`) 
but was not used for the main results.

## Dependencies

qiskit >= 1.0
qiskit-aer
numpy
scipy
matplotlib

Install with:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib
```

## Usage

```python
import numpy as np
from HHL_Code import run_hhl

A = np.array([[1.5, 0.5],
              [0.5, 1.5]])
b = np.array([1.0, 0.0])

results = run_hhl(A, b, n_clock=2)
```

To run the three dissertation experiments:

```python
from HHL_Code import experiment_1_precision, experiment_2_kappa, experiment_3_noise

experiment_1_precision()
experiment_2_kappa()
experiment_3_noise()
```

Figures are saved to the `figures/` directory.

## Repository Structure

HHL Code.py          # Main implementation
figures/             # Output figures from experiments

## Notes

- Systems must be Hermitian, square, invertible, and of dimension 2^n
- The implementation supports systems from 2×2 up to 32×32
- Classical simulation cost scales exponentially in qubit count; 
  32×32 systems require significant runtime

  

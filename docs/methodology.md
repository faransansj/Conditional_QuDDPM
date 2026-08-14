# TFIM Simulation Methodology

## Convention

The implemented one-dimensional transverse-field Ising Hamiltonian is

`H = -J Σ_i Z_i Z_(i+1) - g Σ_i X_i`.

The MVP uses `J=1`, four qubits, and open boundaries. Dense Hermitian exact diagonalization returns the smallest eigenpair. The statevector phase is canonicalized by making its largest-amplitude component real and nonnegative, enabling deterministic artifacts.

This convention is equivalent up to spin-axis relabeling to common TFIM formulations. The thermodynamic critical point is `g/J=1`; a finite four-site chain has no sharp phase transition. Labels are therefore operational benchmark regions rather than finite-system phase proofs:

- ferromagnetic/class 0: `g/J ∈ [0.2, 0.8]`;
- paramagnetic/class 1: `g/J ∈ [1.2, 1.8]`;
- excluded initial region: `(0.8, 1.2)`.

These ranges are fixed before downstream test evaluation and can be changed only as a new versioned benchmark.

## Leakage prevention

Parameter values and IDs are sampled and assigned to class-stratified train/validation/test splits before diagonalization. State generation does not alter split membership. The manifest is the source of truth for every later generator and QCNN data loader.

## Numerical checks

Each generated state must satisfy normalization and eigenpair residual checks at the configured tolerance. Split identifiers must be pairwise disjoint. The CLI writes all checks to `validation.json` and exits nonzero on failure.

## Scaling boundary

Dense statevectors require `O(2^n)` storage and dense Hamiltonians/eigensolvers are exponential. The implementation is intentionally a transparent 4–8 qubit reference simulator, not a scalable many-body solver. Sparse methods should replace it only after dense validation and measured need.

## Literature basis

- S. Sachdev, *Quantum Phase Transitions*, Cambridge University Press (2nd ed., 2011), for the TFIM and its thermodynamic critical behavior.
- I. Cong, S. Choi, and M. D. Lukin, “Quantum Convolutional Neural Networks,” *Nature Physics* (2019), [arXiv:1810.03787](https://arxiv.org/abs/1810.03787), for QCNN phase-recognition motivation.
- TeNPy’s [`tfi_exact.py`](https://github.com/tenpy/tenpy/blob/main/examples/tfi_exact.py) was consulted as an independent exact-TFIM reference; this repository uses its own minimal Pauli/Kronecker implementation and copies no source.

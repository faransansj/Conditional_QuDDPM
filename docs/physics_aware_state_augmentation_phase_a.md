# Phase A — Physics-aware TFIM state augmentation

## Scope and repository ground truth

This phase starts from `52115e16de48b8139aeb3631564785c1aa59aacd` on
`research/tfim-state-augmentation`. It does not modify the archived QuDDPM
checkpoint, K0–K3, frozen dataset/QCNN configs, or existing datasets/results.

The dataset generator, not a documentation assumption, defines

\[
H(g)=-J\sum_{i=0}^{N-2}Z_iZ_{i+1}-g\sum_{i=0}^{N-1}X_i,
\]

with `N=4`, `J=1`, open boundaries. Class 0 uses `g in [0.2,0.8]`, class 1
uses `g in [1.2,1.8]`, and `[0.8,1.2]` is excluded. Stored examples are
normalized `complex128` exact ground-state vectors of shape `(16,)`.
Repository observables are `Mx=(1/N) sum X_i` and
`Mz2=((1/N) sum Z_i)^2`.

Both datasets have 140/30/30 train/validation/test states per class. The random
split is class-stratified. The blocked split allocates ordered `g` intervals
with a 0.02 gap. The QCNN consumes one `(16,)` vector, has 42 parameters, and
reads out `Z` on qubit 3.

## State-local method and provenance

For train state `psi`, Phase A applies

\[
\psi'=\exp(-i\epsilon\widetilde G)\psi,
\qquad \widetilde G=G/\lVert G\rVert_2.
\]

The two generators are the signed repository Hamiltonian components:

- `field`: `G=-sum_i X_i`;
- `interaction`: `G=-J sum_i Z_i Z_(i+1)`.

They are physically meaningful TFIM terms, do not individually share every
TFIM eigenvector, and spectral-norm normalization makes epsilon dimensionless.
A fixed seed chooses the perturbation sign. Every row records synthetic/source
IDs, source dataset/split/class/g, method, normalized generator, operator,
signed strength, and seed. The API rejects any source whose split is not
`train`.

These samples are derivatives of finite training data and known TFIM
structure. They are **not independent physical measurements**.

## Full-H degeneracy

Using `G=H(g)` on its exact ground state yields only `exp(-i epsilon E)` and a
global phase. At the largest tested epsilon, the maximum measured
`1-fidelity` over both datasets was `3.77e-15`. Full `H` is therefore a
degeneracy diagnostic, not an augmentation generator.

## Sweep, acceptance gate, and leakage prevention

The preregistered normalized sweep is `epsilon={0,0.1,0.2,0.4,0.8}` for both
components. `epsilon=0` is an exact identity check. Gates are fit separately
for each dataset/class using only the same ten real train states used by the
pilot:

1. norm error at most the dataset numerical tolerance (`1e-10`);
2. `Mx` and `Mz2` remain inside their train-class empirical ranges;
3. source-Hamiltonian energy excess does not exceed the maximum excess
   produced by another real state in the same train class.

A nearest-g energy scale was examined in a dry diagnostic and rejected: it
admitted no nontrivial class-1 candidates because exact nearby ground states
make that scale effectively zero. The selected rule is still entirely
empirical and train-only, but represents full observed within-class support
rather than an arbitrary constant. Validation/test states do not construct
operators, fit gates, select strengths, or supply sources. They are used only
by the frozen downstream QCNN protocol.

Accepted nonzero candidate counts were random `67/23` and blocked `48/25` for
class 0/1, enough for the maximum requested ratio. Candidate selection is
class-balanced and source-covering; accepted candidates with greatest state
movement are used first. Ratio prefixes are nested.

## Physics diagnostics

`Fnearest` is nearest same-class train fidelity. Diversity is
`1-mean pairwise fidelity` among accepted states at the specified cell; at
`epsilon=0` it reflects diversity among distinct real sources, not augmentation
movement.

| split | G | eps | accept | Fsource | Fnearest | abs dE | abs dMx | abs dMz2 | accepted diversity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blocked | field | 0.0 | 100% | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1788 |
| blocked | field | 0.1 | 85% | 0.9981 | 0.9981 | 0.0066 | 0.0000 | 0.0013 | 0.1909 |
| blocked | field | 0.2 | 80% | 0.9923 | 0.9923 | 0.0262 | 0.0000 | 0.0053 | 0.1948 |
| blocked | field | 0.4 | 35% | 0.9698 | 0.9700 | 0.1038 | 0.0000 | 0.0211 | 0.0978 |
| blocked | field | 0.8 | 0% | 0.8859 | 0.8877 | 0.3989 | 0.0000 | 0.0809 | — |
| blocked | interaction | 0.0 | 100% | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1788 |
| blocked | interaction | 0.1 | 85% | 0.9980 | 0.9980 | 0.0097 | 0.0023 | 0.0000 | 0.1842 |
| blocked | interaction | 0.2 | 35% | 0.9920 | 0.9920 | 0.0389 | 0.0090 | 0.0000 | 0.0197 |
| blocked | interaction | 0.4 | 35% | 0.9686 | 0.9686 | 0.1534 | 0.0357 | 0.0000 | 0.0465 |
| blocked | interaction | 0.8 | 10% | 0.8821 | 0.8841 | 0.5816 | 0.1353 | 0.0000 | 0.0038 |
| random | field | 0.0 | 100% | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2091 |
| random | field | 0.1 | 80% | 0.9982 | 0.9982 | 0.0064 | 0.0000 | 0.0013 | 0.2222 |
| random | field | 0.2 | 80% | 0.9928 | 0.9928 | 0.0254 | 0.0000 | 0.0051 | 0.2335 |
| random | field | 0.4 | 45% | 0.9718 | 0.9719 | 0.1007 | 0.0000 | 0.0204 | 0.1167 |
| random | field | 0.8 | 45% | 0.8933 | 0.8957 | 0.3870 | 0.0000 | 0.0783 | 0.3332 |
| random | interaction | 0.0 | 100% | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2091 |
| random | interaction | 0.1 | 85% | 0.9980 | 0.9980 | 0.0103 | 0.0022 | 0.0000 | 0.2175 |
| random | interaction | 0.2 | 45% | 0.9920 | 0.9920 | 0.0412 | 0.0089 | 0.0000 | 0.0847 |
| random | interaction | 0.4 | 40% | 0.9684 | 0.9684 | 0.1625 | 0.0352 | 0.0000 | 0.1094 |
| random | interaction | 0.8 | 30% | 0.8814 | 0.8822 | 0.6161 | 0.1337 | 0.0000 | 0.2026 |

Across all accepted nonzero strengths, overall diversity was `0.194/0.157`
(field/interaction) for blocked and `0.250/0.191` for random. Mean
source-conditioned diversity was `0.0169/0.0219` for blocked and
`0.0513/0.0393` for random. Thus the pool is not solely machine-precision
copies, though blocked source-conditioned support expansion is modest.

The exact zero drift of `Mx` under the field unitary and `Mz2` under the
interaction unitary follows the respective commutators and is expected, not a
missing measurement.

## Frozen QCNN pilot

The pilot keeps the existing architecture, parameter count, nested subset seed
`31415`, real budget 10/class, full-batch SPSA schedule (300 steps, learning
rate 0.15, perturbation 0.1, patience 40, minimum delta `1e-4`), and seed tuples
`(1000,2000)`, `(1001,2001)`, `(1002,2002)`. Ratios are
`r={0,0.5,1,2}`. Validation drives the existing early stopping; test is
reported descriptively and did not tune augmentation.

| dataset | ratio | train-real accuracy | validation accuracy | test accuracy mean ± std | test loss |
|---|---:|---:|---:|---:|---:|
| random | 0 | 0.9500 | 0.9389 | 0.9222 ± 0.0079 | 0.4115 |
| random | 0.5 | 0.9500 | 0.9444 | 0.9222 ± 0.0079 | 0.4079 |
| random | 1 | 0.9500 | 0.9500 | 0.9500 ± 0.0236 | 0.4241 |
| random | 2 | 0.9500 | 0.9444 | 0.9167 ± 0.0136 | 0.4430 |
| blocked | 0 | 1.0000 | 1.0000 | 0.9000 ± 0.0720 | 0.6190 |
| blocked | 0.5 | 1.0000 | 1.0000 | 0.6889 ± 0.1363 | 0.6501 |
| blocked | 1 | 1.0000 | 1.0000 | 0.6833 ± 0.2248 | 0.6681 |
| blocked | 2 | 1.0000 | 1.0000 | 0.6833 ± 0.2248 | 0.6697 |

Random-split improvement appears only at ratio 1 and reverses at ratio 2.
Every nonzero ratio substantially harms blocked-g test accuracy while
train/validation accuracy remains saturated. This is synthetic bias/local
regularization, not evidence of parameter-shift generalization. Harm begins at
`r=0.5` for blocked-g and at `r=2` for random.

## Separate baselines not run

`simulator_resampling` would diagonalize `H(g+delta g)` and label a new exact
ground state. It is additional simulator-generated labeled TFIM data, not a
state-local transformation of an observed state, so it was not mixed into this
method or run after the primary method met NO-GO. The naive random-Hermitian
negative control was also not run: the physics-aware method already failed the
blocked-g utility criterion, so extra QCNN training cannot rescue the Phase A
claim or distinguish a positive mechanism.

## Interpretation and decision

1. **Near-copy only?** No. Accepted cells reach mean source fidelity near
   0.88–0.97 and nonzero pairwise/source-conditioned diversity, although many
   low-strength candidates remain close and blocked diversity is modest.
2. **Physics and diversity together?** Yes at the state diagnostic level for a
   bounded accepted pool; stronger epsilon lowers acceptance sharply.
3. **QCNN improvement?** Only random split at `r=1` (`+0.0278` accuracy).
4. **Blocked-g persistence?** No; all nonzero ratios degrade substantially.
5. **When harmful?** Blocked-g from `r=0.5`; random by `r=2`.
6. **Resampling distinction?** Resampling creates a new simulator ground state
   at a new parameter; state-local augmentation applies a unitary to a finite
   observed training state and inherits its information.

# NO-GO

Physics-preserving, nontrivial states can be generated, but downstream utility
fails the stronger blocked-g test repeatedly and the isolated random-split gain
collapses under parameter shift. This exactly matches the predefined NO-GO
pattern. Phase B geometry-aware augmentation is not authorized by this result.

Machine-readable artifacts are under
`results/physics_aware_augmentation/phase_a/`.

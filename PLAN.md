# Research and Development Plan

## Governing principles

- Downstream QCNN performance is primary; generation metrics are diagnostic.
- Split Hamiltonian parameter samples before any learned component sees data.
- Compare augmentation arms with identical QCNN protocols and matched seeds.
- Start at 4 qubits, smoke-test before sweeps, and scale only after acceptance gates pass.
- Record negative outcomes without changing the protocol post hoc.
- Freeze research choices before final test evaluation; separate engineering fixes from hypothesis changes.

## Phase 0 — Repository audit ✅

**Findings (current repository):** initial commit, empty `README.md`, no source code, dependencies, configs, tests, datasets, model checkpoints, or runnable baseline. No QuDDPM/MSQuDDPM implementation is available locally to reuse. External reference implementations must be reviewed before implementation.

**Acceptance:** inventory recorded; implemented/planned status is explicit; no unsupported result claims. **Met.**

## Phase 1 — TFIM ground-truth dataset ✅ (4-qubit MVP)

- Declare Hamiltonian convention, boundary condition, `J`, `g` domain, label rule, and near-critical exclusion interval.
- Implement exact diagonalization for 4 qubits, then parameter sampling and serialization.
- Split parameter identifiers before diagonalization; persist manifests.
- Validate normalization, energy, labels, physicality, balance, and deterministic regeneration.

**Acceptance:** norm and ground-energy checks pass at declared numerical tolerances; split identifiers are disjoint; fixed seeds reproduce manifests and states; labels/counts are reported; tests pass. **Met for the 4-qubit dense MVP; 6/8-qubit scale validation remains deferred.**

## Phase 2 — QCNN baseline ✅ (4-qubit development protocol frozen)

- Select one reference QCNN topology compatible with 4/6/8 qubits.
- Implement state input, training, evaluation, and machine-readable result writing.
- Freeze architecture/tuning protocol before augmentation comparison.
- Run real-data sweep and at least 3 seeds.

**Acceptance:** end-to-end smoke run succeeds; accuracy and macro-F1 are saved per seed and aggregated; sweep is config-driven; performance is compared with the declared majority/random baseline; the same test set and protocol can be reused by augmentation arms. **Met for the frozen random/blocked 4-qubit development protocol. Current test splits are development-only; new unseen split seeds are required for final confirmatory evaluation.**

## Phase 3 — Conditional QuDDPM

- Review upstream QuDDPM and conditioning implementations/licenses; reuse rather than rederive where compatible.
- Define label injection and pure-state representation.
- Train only on the training split; generate by class.
- Validate physicality, diversity, class consistency, and reproducibility.

**Acceptance:** both labels generate samples from a fixed checkpoint; all samples meet declared physical tolerances; class-conditioned distributions are distinguishable by preregistered diagnostics; generation is seed-reproducible; no validation/test access occurs in training.

## Phase 4 — QuDDPM augmentation benchmark

- Add generated states only to QCNN training data.
- Run real-size × synthetic-size sweep with matched QCNN seeds.
- Compare real-only, simple perturbation, and Conditional QuDDPM.

**Acceptance:** all arms share split, QCNN, tuning budget, and evaluation code; per-seed and aggregate metrics are saved; `Delta Acc` and macro-F1 differences are reported, including null/negative results.

## Phase 5 — Mixed/noisy dataset

- Implement one channel first (depolarizing); add dephasing/amplitude damping only after validation.
- Apply channels after split to pure states using declared strengths.
- Validate CPTP behavior on test fixtures and density-matrix physicality.

**Acceptance:** trace, Hermiticity, and PSD checks pass at declared tolerance; zero-noise limit recovers input; fixed inputs/parameters reproduce outputs; split separation is retained.

## Phase 6 — Conditional MSQuDDPM

- Review and reproduce the published MSQuDDPM reference behavior.
- Add class conditioning with minimum divergence from the reference design.
- Generate mixed states and evaluate purity distribution, physicality, diversity, and class consistency.

**Acceptance:** class-conditioned generation works; generated matrices pass physical checks; a fixed checkpoint/seed reproduces output; a small reference task matches expected qualitative behavior.

## Phase 7 — MSQuDDPM augmentation benchmark

- Establish noisy real-only and simple-perturbation baselines.
- Run matched synthetic-ratio sweeps and robustness evaluation.

**Acceptance:** QCNN protocol is identical across noisy benchmark arms; mean/std and paired differences are stored; noise strength and augmentation effects are independently identifiable.

## Phase 8 — Cross-model analysis

- Compare models only in a matched state/noise regime.
- Analyze data efficiency, synthetic quantity, generation/downstream correlation, qubit scaling, and near-critical behavior.

**Acceptance:** comparison matrix states what is controlled; correlation analysis includes uncertainty and sample count; 6/8-qubit claims are made only for completed runs; incompatible regimes are reported separately.

## Phase 9 — Final research artifact

- Freeze configs and code revision.
- Increase seeds as allowed by a documented compute/variance decision.
- Produce figures, tables, ablations, statistical analysis, and exact reproduction commands.

**Acceptance:** a clean environment can reproduce one smoke run and all reported aggregate tables from stored configs/results; claims map to artifacts; limitations and failed hypotheses are included.

## Architecture decision

Use one Python package only when Phase 1 starts. Prefer NumPy/SciPy exact diagonalization and the already-required quantum ML framework; do not introduce parallel simulator abstractions. YAML is the planned experiment format, with one result writer shared by all arms. Dataset split manifests are immutable inputs to generators and classifiers.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Exponential state/density-matrix scaling | Gate progression at 4 → 6 → 8 qubits; estimate memory/time before each scale-up. |
| Finite-size/critical-label ambiguity | Predeclare labels and exclusion interval; report near-critical results separately. |
| Generator memorization or mode collapse | Nearest-training-state fidelity plus diversity/coverage diagnostics. |
| Leakage through tuning or generation | Split manifests, train-only data loaders, automated disjointness tests. |
| Unfair QCNN comparisons | Matched splits/seeds and frozen architecture/training protocol. |
| High synthetic ratio changes optimization steps | Control epochs/batches explicitly and report both sample and update budgets. |
| Reference-code incompatibility/license | Audit upstream APIs and licenses before porting. |
| Fidelity does not imply utility | Treat QCNN metrics as primary and test correlation rather than assume it. |
| Compute-driven selective reporting | Declare sweep grid and preserve every completed run artifact. |

## Scope exclusions for MVP

Few-step models, learned schedules, QGAN/QCBM, additional Hamiltonians, alternative classifiers, and QPU execution are roadmap items only.

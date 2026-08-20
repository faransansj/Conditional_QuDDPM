# QML augmentation track transition

> **Historical document.** This file records an earlier project transition and is superseded by [`tfim_state_augmentation_final_archive.md`](tfim_state_augmentation_final_archive.md). Its GO labels, implementation plan, and “Next execution” section are historical, not current work.

## Decision

| Track | Status | Reason |
|---|---|---|
| Conditional QuDDPM | **NO-GO** | K0–K3 did not produce an augmentation-ready generator; K3 conflict weighting was not better than uniform and physics weighting traded global MMD for physics alignment. No K4 objective engineering. |
| MSQuDDPM | **HOLD** | Not failed and not implemented in this repository. Preserve as a possible later mixed-state baseline after the primary question is answered. |
| Physics-aware perturbation | **GO** | Smallest physically valid state-level augmentation control and fastest end-to-end test of downstream utility. |
| Pure-state score model (formerly SSDM) | **GO, feasibility first** | Learned geometry-aware candidate; implement only after reproducing the current v4 formulation and passing staged gates. |

Primary question: **Under limited quantum training data, can quantum-state-level augmentation improve downstream QCNN generalization?**

Secondary question: **Does learned geometry-aware generation add benefit beyond physics-aware perturbation?**

QuDDPM evidence remains preserved under `docs/rdm_kernel_diagnostics.md`, `docs/k2_realization_conflict.md`, `docs/k3_conflict_reweighting.md`, and `results/quddpm_kernel_diagnostics/`. The NO-GO is local to the frozen 4-qubit diagnostic; it is not a universal impossibility claim. MSQuDDPM has no model, config, artifact, or test in this repository, so its status is HOLD rather than failure.

## Repository ground truth

### Data

The frozen QCNN config uses both active 4-qubit open-chain TFIM datasets (`J=1`):

| Dataset | Split strategy | States/class | Train/val/test per class |
|---|---|---:|---:|
| `data/tfim_4q_random` | random | 200 | 140 / 30 / 30 |
| `data/tfim_4q_blocked` | blocked `g`, configured gap 0.02 | 200 | 140 / 30 / 30 |

Class regions are `g∈[0.2,0.8]` and `g∈[1.2,1.8]`. Both artifacts contain normalized `(16,)` `complex128` states plus energies, labels, parameter IDs, split IDs, `g`, `Mx`, and `Mz²`. `data/tfim_4q` is a legacy 100/class dataset and is not a frozen benchmark input. `split_manifest.json` is authoritative; augmentation fitting may access train rows only. Validation may select gates/strengths. Test is reserved for final downstream evaluation.

### Frozen QCNN benchmark

`configs/qcnn/baseline_4q.yaml` freezes budgets `{10,25,50,100}` real states/class, nested subset seed `31415`, three run/init/SPSA seed triples, and 300-step full-batch SPSA (`a=0.15`, `c=0.1`, patience 40, minimum delta `1e-4`). The 42-parameter statevector QCNN in `models/qcnn.py` reduces `4→2→1`, reads `⟨Z₃⟩`, trains MSE against `2y-1`, and selects checkpoints by validation loss.

Documented real-only accuracy means are:

| States/class | Random | Blocked-g |
|---:|---:|---:|
| 10 | 0.9222 | 0.9000 |
| 25 | 0.9944 | 0.8000 |
| 50 | 0.9722 | 0.8389 |
| 100 | 0.9944 | 0.7722 |

These are development results. Final confirmatory claims require a new frozen split seed because the current test metrics have already been inspected. Exact local baseline run artifacts are absent from this checkout; the documented aggregates are available in `docs/qcnn_baseline.md`.

## Physics-aware perturbation design

### Transformation

For normalized train state `|ψ(g)⟩`, generate

```text
|ψ̃⟩ = exp[-i ε G] |ψ⟩,
G ∈ {Σ_i X_i, Σ_i Z_i Z_{i+1}}.
```

Both generators are Hermitian and commute with the TFIM global spin-flip parity `Π_i X_i`, so the unitary preserves norm and parity while producing a nontrivial neighboring ray. Using the full source Hamiltonian alone is rejected: a ground state only acquires a global phase and creates no diversity. A random-control arm uses a seeded random Hermitian generator projected into the parity-commuting subspace and matched to the physics-aware arm's Fubini–Study displacement. Matching both symmetry and displacement isolates TFIM generator structure from generic symmetry-preserving unitary regularization. Keep this control in later benchmarks if the pilot shows a material difference; otherwise record the pilot equivalence before dropping it.

The minimum API should be one function, not a class hierarchy:

```python
augment(real_states, labels, n_samples, seed, config) -> (states, labels, provenance)
```

### Data-derived strengths and gate

Do not assign arbitrary epsilon labels. On train only, compute the epsilon values whose median source Fubini–Study displacement reaches three predeclared quantiles of the within-class nearest-neighbor distance distribution; call these weak/medium/strong. Validation selects one canonical strength by the following gate, fixed before QCNN testing:

1. norm error is within the dataset numerical tolerance;
2. every synthetic sample remains closer in robust standardized `(Mx,Mz²)` distance to its source class than to the other class;
3. each class's synthetic observable distribution stays inside the validation-derived robust envelope (median/MAD with the multiplier derived from the observed inter-class gap, recorded in the config);
4. nearest-train fidelity is below one except for declared zero-strength controls;
5. pairwise Fubini–Study diversity is nonzero and reported relative to real train diversity.

If no candidate passes, the physics-aware arm stops. Selection uses no QCNN test metric. Required artifacts: resolved config, source IDs, generator/epsilon/seed, state hash, norm/fidelity/observable/diversity tables, gate decision, and explicit `test_split_used:false`.

### Pilot

Run random-control and physics-aware gates first on 10/class, one QCNN seed, augmentation ratio 1x. If valid, run the paired three-seed 10/class pilot with identical real subset, synthetic count, QCNN initialization, SPSA randomness, updates, validation/test split, and evaluation code. Expansion to 25/class and ratio sensitivity happens only after this end-to-end contract works.

## SSDM / pure-state score model feasibility

### Prior art and corrected formulation

The current source is Xu et al., **“Local-Time Riemannian Score Matching on the Quantum Pure-State Manifold,” arXiv:2605.03573v4** ([paper](https://arxiv.org/html/2605.03573v4), [record](https://export.arxiv.org/api/query?id_list=2605.03573)). Earlier versions used the title “Stochastic Schrödinger Diffusion Models for Pure-State Ensemble Generation” and the SSDM name. The plan must pin v4: it retracts stronger OU/scaling claims and calls the implementation a pure-state score model.

Represent rays on `CP^(d-1)` with Fubini–Study geometry:

```text
d_FS(ψ,φ) = arccos |⟨ψ|φ⟩|.
```

The reported forward process is drift-free Fubini–Study Brownian motion (`λ=0`), implemented by horizontal tangent Gaussian noise. The optional stochastic Schrödinger equation is an equivalent derivation, not required for the minimal implementation. The reverse SDE uses the projected score `sθ(ψ,t)`, exact projective log/exp maps, an approximately Haar terminal prior, and tangent Euler–Maruyama. Finite-time Brownian motion only approaches Haar stationarity, so S0 must measure terminal mixing against direct Haar samples and reject the configured horizon when the discrepancy exceeds its predeclared Monte Carlo uncertainty. The local teacher is normalized by diffusion-clock variance `Δτ=∫σ²dt`, not elapsed `Δt`; training also requires random global-phase augmentation.

The v4 paper is unconditional. It does not demonstrate class conditioning. For this binary task the safest first conditional baseline is **two independent class models**, because it changes no network equation or conditioning interface. A shared `sθ(ψ,t,label)` model is a later ablation only after unconditional/class-specific reproduction. Low-data instability and doubled model variance must be reported.

No maintained official SSDM/PSM package was found. Nearby references—[Riemannian score SDE](https://github.com/oxcsml/riemannian-score-sde) and [scaling Riemannian diffusion](https://github.com/louaaron/scaling-riemannian-diffusion)—provide general machinery but no drop-in complex-projective implementation. Verdict: **build a minimal, equation-pinned implementation**, using existing NumPy/SciPy first; add a neural dependency only after the S0 design identifies a real need.

### Staged gates

- **S0 feasibility:** exact FS projection/log/exp and Haar invariants; finite tiny-model loss; normalized deterministic samples; no NaN/Inf.
- **S1 quality:** use a Gaussian kernel or energy distance on flattened pure-state density matrices `|ψ⟩⟨ψ|`, an injective global-phase-invariant ray representation; also report source/nearest-neighbor fidelity, `(Mx,Mz²)`, classwise diversity, and memorization. Overlap MMD alone is forbidden because it is not characteristic.
- **S2 conditional gate:** for each generated class, compare distance to the matched held-out validation class against distance to the mismatched class, under equal sample counts. Require the matched distance to be smaller for both labels, while nearest-train fidelity and duplicate rates remain within validation-derived real-to-real reference ranges. Compare class-specific and unconditional models under equal total parameter and training-update budgets; the unconditional model is evaluated as a mixture, not asked to route labels. Freeze this gate before running QCNN.
- **S3 downstream pilot:** same 10/class real subset, 1x synthetic count, and frozen QCNN seeds as physics-aware.

One bounded diagnostic is allowed per failed gate. No architecture or objective sweep follows a failed S2.

## Canonical benchmark plan

Primary arms only:

```text
A. Real-only
B. Real + physics-aware perturbation
C. Real + pure-state score-model samples
```

Use random and blocked-g datasets, budgets `{10,25,50,100}` only after pilots, canonical ratio 1x first, identical real IDs/synthetic counts/QCNN seeds/optimizer/steps/evaluation splits, and paired per-seed deltas. Report accuracy, macro-F1, validation loss, mean/std, failures, and `Δaccuracy` against the matched real-only run. Prioritize 10/class and 25/class. The combined B+C arm is not primary and remains deferred.

## Implementation plan

1. Add a small augmentation result/provenance data contract and a QCNN runner seam that concatenates validated synthetic train states without changing validation/test arrays.
2. Implement physics-aware and matched-random unitary perturbations plus state-quality gates and focused tests.
3. Run 10/class one-seed smoke, then paired three-seed physics-aware pilot.
4. Pin SSDM/PSM v4 equations; implement/test FS geometry and forward mixing before any score network.
5. Run S0, then S1/S2 sequentially. Stop on gate failure.
6. Only after S2, run the matched SSDM QCNN pilot.
7. Freeze a new confirmatory split before making final test claims.

Required tests: normalization, deterministic seeds, invalid-state rejection, observable/label-gate calculations, train-only access, synthetic provenance/hash, QCNN integration smoke; plus SSDM phase invariance, tangent projection, log/exp round-trip, Haar statistics, finite loss, deterministic normalized sampling, class-specific routing, and split isolation.

## Next execution

Implement **only** the shared augmentation contract and physics-aware/matched-random state generator with focused tests. Do not start SSDM network code until the FS geometry S0 specification and dependency decision are reviewed. Do not run a canonical benchmark yet.

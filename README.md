# Quantum-State Augmentation for QCNNs

> **Research question:** Under limited quantum training data, can state-level augmentation improve downstream QCNN generalization, and does learned geometry-aware generation add value beyond physics-aware perturbation?

## Status

| Component | Status |
|---|---|
| Repository and research design | ✅ Documented |
| TFIM dataset generator | ✅ Implemented and tested (4-qubit dense exact diagonalization) |
| QCNN baseline | ✅ Development protocol frozen (4 qubits) |
| Conditional QuDDPM | ⛔ **NO-GO** after preserved K0–K3 diagnostics; no K4 |
| Conditional MSQuDDPM | ⏸️ **HOLD**, not evaluated as failed |
| Physics-aware perturbation | ▶️ **GO**, next implementation |
| Pure-state score model (formerly SSDM) | ▶️ **GO**, feasibility first |
| Augmentation benchmarks | 📋 Planned after method gates |

The 4-qubit TFIM simulator and real-only QCNN baseline are executable. The QCNN development protocol is frozen at commit `6861c39`; current splits are development-only and new unseen split seeds are reserved for final confirmation. Conditional QuDDPM did not reach an augmentation-ready 4-qubit generator: K2 found realization conflict and K3 found that simple reweighting did not outperform uniform aggregation. Those negative results are retained. The repository does **not** yet claim augmentation results.

## Motivation and problem statement

Quantum-state datasets may be expensive to prepare or label. This project tests whether class-conditioned quantum diffusion models can improve the data efficiency of a fixed downstream QCNN in a controlled, classically simulated setting. The objective is not to force a positive result, but to identify when augmentation helps, has no effect, or harms classification.

**Primary hypothesis:** with the same limited ground-truth TFIM training set, adding physically valid, class-conditioned synthetic states can improve held-out QCNN accuracy and F1 score relative to real-only and simple-perturbation controls.

The initial claim is deliberately limited to controlled classical simulation. Real-QPU performance is out of scope.

## Research questions

- **RQ1:** Does physics-aware state perturbation improve QCNN performance under limited training data?
- **RQ2:** Does a learned pure-state score model improve beyond the physics-aware baseline?
- **RQ3:** How does effectiveness change as the amount of real training data decreases?
- **RQ4:** How does synthetic-data quantity affect QCNN performance?
- **RQ5:** Is generative quality correlated with downstream QCNN improvement?
- **RQ6 (extension):** How does augmentation behave near the TFIM phase transition?
- **RQ7 (extension):** Do results generalize across qubit counts and classification tasks?

## Pipeline and component boundaries

```text
TFIM parameter sampling
  -> split parameter points into train / validation / test
  -> exact-diagonalization ground states
       |-> real-only QCNN baseline
       |-> train-only physics-aware perturbation -> synthetic pure states -> QCNN
       `-> train-only pure-state score model -> synthetic pure states -> QCNN
  [MSQuDDPM mixed-state work remains on HOLD]
  -> evaluate every QCNN on the untouched corresponding test set
```

- **TFIM simulator:** constructs `H = -J Σ Z_i Z_(i+1) - g Σ X_i`, computes ground states, and assigns phase labels. Initial training excludes a predeclared near-critical interval; it is never selected after observing test performance.
- **QCNN:** downstream classifier only. Its architecture, optimizer protocol, tuning budget, and test set remain fixed across augmentation arms.
- **Physics-aware perturbation:** applies small symmetry-preserving TFIM-structured unitaries selected by train/validation state-quality gates.
- **Pure-state score model:** feasibility track pinned to arXiv:2605.03573v4 and Fubini–Study geometry; no QCNN use before state-quality and conditional gates.
- **Conditional QuDDPM:** preserved negative diagnostic track, locally NO-GO after K0–K3.
- **Conditional MSQuDDPM:** preserved future mixed-state track, currently HOLD and not classified as failed.

## Experimental design

The primary benchmark compares matched pure-state arms:

1. real-only QCNN;
2. real plus physics-aware perturbations;
3. real plus pure-state score-model samples.

A symmetry- and displacement-matched random-unitary control is retained through the pilot. MSQuDDPM mixed-state comparisons remain deferred. Every reported comparison uses the same real IDs, synthetic count, split, QCNN initialization, optimizer updates, and evaluation code.

### Limited-data protocol

MVP sweeps begin at 4 qubits with real training sizes `{10, 25, 50, 100}` states per class and synthetic sizes `{0, 25, 50, 100, 250}` per class. Smoke tests use a subset. Expansion to 6 and 8 qubits occurs only after the 4-qubit pipeline is validated; density-matrix cost may limit mixed-state experiments.

At least 3 independent seeds are used during development. Final seed count is chosen before final benchmarking based on measured variance and compute budget. Dataset, split, initialization, training, and generation seeds are stored separately.

### Leakage controls

1. Sample and assign Hamiltonian parameter points to splits before state generation.
2. Fit preprocessing, generators, and hyperparameters using train/validation only.
3. Generate synthetic states from generator checkpoints trained on train only.
4. Keep the test set inaccessible until final evaluation for a declared run.
5. Store split manifests and verify disjoint parameter identifiers in tests.

Synthetic samples never enter validation or test sets. Test states are never used for generator training, early stopping, model selection, or QCNN tuning.

## Evaluation

### Primary: downstream performance

- accuracy and macro-F1 on a fixed independent test set;
- mean and standard deviation across seeds;
- paired improvement `Delta Acc = Acc_augmented - Acc_baseline` for matched splits/seeds;
- accuracy versus real-data and synthetic-data budgets (sample efficiency).

Confidence intervals and paired significance tests are added for the final benchmark after the seed count and assumptions are fixed. Near-critical accuracy and noise robustness are secondary analyses, not MVP gates.

### Minimum generative diagnostics

- **All density matrices:** trace-one error, Hermiticity error, minimum eigenvalue/PSD violation.
- **Pure-state generation:** normalization/purity, within-class fidelity or kernel-distance distribution, diversity/coverage, and class consistency.
- **Mixed-state generation:** purity distribution, fidelity/superfidelity as appropriate, diversity/coverage, and class consistency.

Wasserstein distance is used only on a declared low-dimensional observable/embedding distribution with a stated ground metric; it is not treated as a generic distance between density matrices. Generative quality is secondary to downstream performance.

## Reproducibility and result contract

Planned experiments use versioned YAML configuration and write machine-readable artifacts:

```text
results/<experiment-id>/
├── config.yaml
├── split_manifest.json
├── metrics.json
├── train.log
└── checkpoints/
```

Each artifact records code revision, library/simulator versions, qubit count, Hamiltonian boundary conditions and parameter grid, phase-label rule, critical exclusion interval, noise parameters, all seeds, augmentation ratio, architectures, optimizer, learning rate, epochs, and stopping rule.

## Repository structure

Implemented paths are shown without a planning annotation.

```text
.
├── README.md
├── PLAN.md
├── TODO.md
├── docs/
│   ├── research_plan.md
│   ├── experiment_plan.md
│   ├── methodology.md
│   ├── tfim_simulation_guide.md
│   ├── qcnn_baseline.md
│   ├── quddpm_validation.md
│   ├── quddpm_tfim_4q_gate.md
│   └── upstream_audit.md
├── configs/
│   ├── dataset/             # random and blocked TFIM configs
│   ├── qcnn/                # frozen QCNN configs
│   └── quddpm/              # Phase 3 smoke config
├── src/conditional_quddpm/
│   ├── datasets/            # TFIM generator and manifest-first loader
│   ├── models/              # QCNN and pure-state QuDDPM
│   └── experiments/         # baseline and smoke entry points
├── scripts/generate_tfim.py
├── tests/
├── data/                    # generated, not source-controlled
└── results/                 # generated, not source-controlled
```

## Installation and usage

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
uv run pytest
# Class-stratified random split
uv run generate-tfim --config configs/dataset/tfim_4q.yaml --output data/tfim_4q_random

# Contiguous g blocks with guard gaps between splits
uv run generate-tfim --config configs/dataset/tfim_4q_blocked.yaml --output data/tfim_4q_blocked
```

Each command creates 200 states per class (140 train states per class), compressed statevectors and observables, a split manifest, SHA-256 checksums, and `validation.json`.

Run the real-only QCNN baseline:

```bash
# Fast end-to-end check
uv run run-qcnn-baseline --config configs/qcnn/baseline_4q_smoke.yaml --output results/qcnn_baseline_smoke

# {10,25,50,100} states/class × 3 seeds × random/blocked
uv run run-qcnn-baseline --config configs/qcnn/baseline_4q.yaml --output results/qcnn_baseline
```

Validate unconditional and conditional QuDDPM mechanics before TFIM training:

```bash
uv run run-quddpm-smoke --config configs/quddpm/phase3_smoke.yaml --output results/quddpm_phase3_smoke

# Expected to exit nonzero until the 4-qubit TFIM learning gate passes.
uv run run-quddpm-tfim --config configs/quddpm/tfim_4q_smoke.yaml --output results/quddpm_tfim_4q_smoke
```

The QCNN loader treats `split_manifest.json` as the source of truth and stores exact training IDs with each run. The default convention is open-boundary `H = -J Σ ZZ - g Σ X`, `J=1`, with samples in `g/J ∈ [0.2,0.8]` and `[1.2,1.8]`; the finite-size critical neighborhood is excluded from this initial classification dataset. Random and blocked-g results are reported as separate benchmarks. See the [TFIM simulation guide](docs/tfim_simulation_guide.md) for artifact schemas, examples, and interpretation.

## Roadmap

1. Validate a deterministic 4-qubit TFIM dataset and leakage-safe split.
2. Establish a fixed QCNN real-only baseline and limited-data sweep.
3. Implement and gate physics-aware plus matched-random perturbations.
4. Reproduce the pinned pure-state score-model geometry and pass S0–S2 gates.
5. Run matched real-only/physics-aware/score-model comparisons.
6. Freeze a new confirmatory split, then produce paper-ready tables and reproduction commands.

See [PLAN.md](PLAN.md), [TODO.md](TODO.md), [research plan](docs/research_plan.md), [experiment plan](docs/experiment_plan.md), [QCNN baseline guide](docs/qcnn_baseline.md), [QuDDPM validation](docs/quddpm_validation.md), [4-qubit TFIM learning gate](docs/quddpm_tfim_4q_gate.md), [RDM-kernel diagnostic](docs/rdm_kernel_diagnostics.md), and [upstream diffusion audit](docs/upstream_audit.md).

## Limitations

- Exact simulation scales exponentially; mixed-state storage scales as `4^n`.
- Finite TFIM chains do not exhibit a sharp thermodynamic transition, so phase labels and the excluded critical interval are operational benchmark definitions.
- Synthetic samples are not independent ground-truth observations; effective sample size may saturate or decline at high augmentation ratios.
- A QCNN simulation result does not establish quantum advantage or hardware utility.
- Conditional QuDDPM is a preserved local NO-GO, MSQuDDPM is HOLD, and the pure-state score model has no verified drop-in implementation; future work must remain version-pinned and gate-limited.

Negative findings—including no gain, degradation at high synthetic ratios, disagreement between fidelity and downstream utility, or regime-specific model advantages—are valid outcomes.

## Future work

Few-step diffusion, learnable schedules, hardware-context conditioning, QGAN/QCBM controls, XXZ/Heisenberg and topological tasks, alternative QML classifiers, hardware-noise-informed augmentation, and real-QPU validation remain out of the MVP.

## References

1. Iris Cong, Soonwon Choi, and Mikhail D. Lukin, “Quantum Convolutional Neural Networks,” *Nature Physics* (2019), [arXiv:1810.03787](https://arxiv.org/abs/1810.03787), DOI: 10.1038/s41567-019-0648-8.
2. Bingzhi Zhang, Peng Xu, Xiaohui Chen, and Quntao Zhuang, “Generative quantum machine learning via denoising diffusion probabilistic models,” *Physical Review Letters* **132**, 100602 (2024), [arXiv:2310.05866](https://arxiv.org/abs/2310.05866), DOI: 10.1103/PhysRevLett.132.100602.
3. Gino Kwun, Bingzhi Zhang, and Quntao Zhuang, “Mixed-State Quantum Denoising Diffusion Probabilistic Model,” [arXiv:2411.17608](https://arxiv.org/abs/2411.17608) (2024; v2 2025).
4. Daniel Quinn, Lorenzo Buffoni, Stefano Gherardini, and Gabriele De Chiara, “Conditioning in Generative Quantum Denoising Diffusion Models,” [arXiv:2509.17569](https://arxiv.org/abs/2509.17569) (2025 preprint).

References 2–4 motivate model design; they do not establish the augmentation hypothesis tested here. Additional augmentation baselines will be cited only when selected and reproduced.

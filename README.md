# Conditional Quantum Diffusion Augmentation for QCNNs

> **Research question:** Can quantum-state data augmentation using Conditional QuDDPM and Conditional MSQuDDPM improve downstream QCNN classification under limited quantum training data?

## Status

| Component | Status |
|---|---|
| Repository and research design | ✅ Documented |
| TFIM dataset generator | ✅ Implemented and tested (4-qubit dense exact diagonalization) |
| QCNN baseline | 📋 Planned |
| Conditional QuDDPM | 📋 Planned |
| Conditional MSQuDDPM | 📋 Planned |
| Augmentation benchmarks | 📋 Planned |

The 4-qubit TFIM dataset simulator is executable. QCNN and diffusion-model training remain unimplemented; the repository does **not** claim augmentation results.

## Motivation and problem statement

Quantum-state datasets may be expensive to prepare or label. This project tests whether class-conditioned quantum diffusion models can improve the data efficiency of a fixed downstream QCNN in a controlled, classically simulated setting. The objective is not to force a positive result, but to identify when augmentation helps, has no effect, or harms classification.

**Primary hypothesis:** with the same limited ground-truth TFIM training set, adding physically valid, class-conditioned synthetic states can improve held-out QCNN accuracy and F1 score relative to real-only and simple-perturbation controls.

The initial claim is deliberately limited to controlled classical simulation. Real-QPU performance is out of scope.

## Research questions

- **RQ1:** Does Conditional QuDDPM augmentation improve QCNN performance under limited training data?
- **RQ2:** Does Conditional MSQuDDPM improve QCNN performance on mixed/noisy quantum-state datasets?
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
       |-> train-only Conditional QuDDPM -> synthetic pure states -> QCNN
       `-> train-only noise channels -> Conditional MSQuDDPM
             -> synthetic mixed states -> QCNN
  -> evaluate every QCNN on the untouched corresponding test set
```

- **TFIM simulator:** constructs `H = -J Σ Z_i Z_(i+1) - g Σ X_i`, computes ground states, and assigns phase labels. Initial training excludes a predeclared near-critical interval; it is never selected after observing test performance.
- **QCNN:** downstream classifier only. Its architecture, optimizer protocol, tuning budget, and test set remain fixed across augmentation arms.
- **Conditional QuDDPM:** learns the train-only pure-state distribution `p_theta(rho | y)` and generates class-conditioned pure states.
- **Conditional MSQuDDPM:** learns class-conditioned mixed/noisy states. Its initial forward process will follow the published depolarizing-channel MSQuDDPM unless evidence supports another channel.

## Experimental design

Two primary benchmarks avoid an invalid apples-to-oranges comparison:

1. **Pure-state benchmark:** real-only QCNN vs simple perturbation vs Conditional QuDDPM augmentation.
2. **Mixed/noisy benchmark:** noisy real-only QCNN vs matched simple perturbation vs Conditional MSQuDDPM augmentation.

A cross-model comparison is secondary and will use the same state representation, noise condition, real-data budget, synthetic budget, split, and QCNN protocol. It will be reported only if both models support that matched condition.

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
│   └── methodology.md
├── configs/dataset/tfim_4q.yaml
├── src/conditional_quddpm/
│   └── datasets/tfim.py     # TFIM Hamiltonian, eigensolver, dataset CLI
├── scripts/generate_tfim.py
├── tests/test_tfim.py
├── data/                    # generated, not source-controlled
└── results/                 # generated, not source-controlled
```

## Installation and usage

Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest

generate-tfim --config configs/dataset/tfim_4q.yaml --output data/tfim_4q
```

The command creates compressed statevectors, a split manifest, checksums, and `validation.json`. The default convention is open-boundary `H = -J Σ ZZ - g Σ X`, `J=1`, with samples in `g/J ∈ [0.2,0.8]` and `[1.2,1.8]`; the finite-size critical neighborhood is excluded from this initial classification dataset.

## Roadmap

1. Validate a deterministic 4-qubit TFIM dataset and leakage-safe split.
2. Establish a fixed QCNN real-only baseline and limited-data sweep.
3. Adapt the published QuDDPM conditioning approach and benchmark pure-state augmentation.
4. Build controlled noisy datasets and adapt MSQuDDPM conditioning.
5. Run matched multi-seed comparisons, scaling and near-critical analyses.
6. Produce paper-ready tables, figures, and reproduction commands.

See [PLAN.md](PLAN.md), [TODO.md](TODO.md), [research plan](docs/research_plan.md), and [experiment plan](docs/experiment_plan.md).

## Limitations

- Exact simulation scales exponentially; mixed-state storage scales as `4^n`.
- Finite TFIM chains do not exhibit a sharp thermodynamic transition, so phase labels and the excluded critical interval are operational benchmark definitions.
- Synthetic samples are not independent ground-truth observations; effective sample size may saturate or decline at high augmentation ratios.
- A QCNN simulation result does not establish quantum advantage or hardware utility.
- QuDDPM, MSQuDDPM, and conditioning choices are active research implementations; reproduction against reference code is required before modification.

Negative findings—including no gain, degradation at high synthetic ratios, disagreement between fidelity and downstream utility, or regime-specific model advantages—are valid outcomes.

## Future work

Few-step diffusion, learnable schedules, hardware-context conditioning, QGAN/QCBM controls, XXZ/Heisenberg and topological tasks, alternative QML classifiers, hardware-noise-informed augmentation, and real-QPU validation remain out of the MVP.

## References

1. Iris Cong, Soonwon Choi, and Mikhail D. Lukin, “Quantum Convolutional Neural Networks,” *Nature Physics* (2019), [arXiv:1810.03787](https://arxiv.org/abs/1810.03787), DOI: 10.1038/s41567-019-0648-8.
2. Bingzhi Zhang, Peng Xu, Xiaohui Chen, and Quntao Zhuang, “Generative quantum machine learning via denoising diffusion probabilistic models,” *Physical Review Letters* **132**, 100602 (2024), [arXiv:2310.05866](https://arxiv.org/abs/2310.05866), DOI: 10.1103/PhysRevLett.132.100602.
3. Gino Kwun, Bingzhi Zhang, and Quntao Zhuang, “Mixed-State Quantum Denoising Diffusion Probabilistic Model,” [arXiv:2411.17608](https://arxiv.org/abs/2411.17608) (2024; v2 2025).
4. Daniel Quinn, Lorenzo Buffoni, Stefano Gherardini, and Gabriele De Chiara, “Conditioning in Generative Quantum Denoising Diffusion Models,” [arXiv:2509.17569](https://arxiv.org/abs/2509.17569) (2025 preprint).

References 2–4 motivate model design; they do not establish the augmentation hypothesis tested here. Additional augmentation baselines will be cited only when selected and reproduced.

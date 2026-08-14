# Experiment Plan

## Benchmark matrix

| Benchmark | Real data | Augmentation arms | Evaluation data |
|---|---|---|---|
| Pure | TFIM ground-state vectors/density matrices | none; simple physically valid perturbation; Conditional QuDDPM | untouched pure TFIM test split |
| Mixed/noisy | channel-applied TFIM density matrices | none; matched simple perturbation; Conditional MSQuDDPM | untouched noisy test split at declared channel/strength |
| Cross-model (secondary) | matched representation and noise regime | C-QuDDPM; C-MSQuDDPM | same test split, only if both are valid for the regime |

## MVP configuration

Defaults are proposals to validate, not final scientific constants:

```yaml
system:
  qubits: 4
  J: 1.0
  boundary: open
data:
  real_per_class: [10, 25, 50, 100]
  synthetic_per_class: [0, 25, 50, 100, 250]
  seeds: [0, 1, 2]
  split_unit: hamiltonian_parameter_id
metrics:
  primary: [accuracy, macro_f1]
  report: [mean, std, paired_delta]
```

The `g/J` sampling range, phase cutoffs, critical exclusion interval, split proportions, noise strengths, and QCNN architecture require Phase 1/2 validation before freezing. Do not infer them from test performance.

## Run order

1. **Numerical smoke:** 4 qubits, a few `g` values, physicality and energy checks.
2. **Split smoke:** create manifest; prove train/val/test identifiers are disjoint and reproducible.
3. **QCNN smoke:** one seed and smallest real budget; verify artifact schema.
4. **Baseline pilot:** all real budgets, 3 seeds; estimate runtime and variance.
5. **Generator reproduction:** reproduce a small published/reference task before TFIM conditioning.
6. **Augmentation pilot:** smallest useful synthetic grid, never test-tuned.
7. **Declared sweep:** run complete grid and preserve all results.
8. **Scale gate:** estimate and approve 6-qubit, then 8-qubit cost independently.

## Data protocol

- Construct a canonical table of parameter IDs and metadata.
- Assign IDs to splits using a fixed dataset/split seed before diagonalization.
- Optionally stratify by class while retaining ID-level independence.
- Draw limited-data subsets only from train; use nested subsets to reduce variance across budgets.
- Train diffusion models on the same available real subset used by the matched QCNN arm unless a separate generator-data regime is explicitly declared.
- Fit model selection on validation only.
- Materialize test metrics once per frozen run family.

A separate research comparison may ask whether a generator pretrained on more data helps a low-data QCNN, but it must not be mislabeled as equal-ground-truth augmentation.

## Simple augmentation baseline

Use the smallest physically valid perturbation appropriate to each representation. Candidate: small label-preserving unitary rotations for pure states and a matched weak declared channel for density matrices. Calibrate strength on train/validation only. Classical duplicate oversampling may be recorded as an optimization control but is not a new-information baseline.

## Generative evaluation

### Physical validity (hard checks)

For density matrix `rho`:

- `|Tr(rho)-1|`;
- `||rho-rho†||`;
- minimum eigenvalue;
- purity `Tr(rho²)` and its valid range.

Tolerances are based on eigensolver/numeric precision measured in Phase 1, then stored in config—not chosen per sample.

### Distribution and conditioning diagnostics

- nearest-training-state fidelity to detect memorization;
- within- and between-class fidelity/observable distributions;
- diversity/coverage using a declared feature space (e.g. magnetization, energy, correlations);
- class consistency via predeclared physical observables and/or a validation-trained classifier;
- superfidelity where used by the MSQuDDPM objective and full fidelity where computationally feasible.

## Downstream evaluation

Primary outputs per run:

```json
{
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "n_test": 0,
  "split_seed": 0,
  "model_seed": 0,
  "status": "completed"
}
```

Aggregates include mean, standard deviation, and matched deltas. Save confusion counts so metrics can be independently recomputed. Secondary outputs may include near-critical accuracy, ROC-AUC for balanced binary evaluation, robustness curves, and training cost.

## Artifact and naming contract

A run ID is derived from the immutable resolved config plus code revision. Each run stores resolved config, split-manifest checksum, metrics, logs, checkpoint metadata, and failure status. Large generated states/checkpoints are excluded from Git and referenced by checksums/paths.

## Stop/go gates

- Do not implement a QCNN before TFIM and split tests pass.
- Do not benchmark augmentation before the real-only baseline is reproducible.
- Do not condition a diffusion model before reproducing a minimal upstream behavior.
- Do not scale qubits before profiling time and memory.
- Do not add more baselines until the three primary arms plus simple perturbation run end to end.

## Interpretation rules

- Positive average delta with high seed variance is inconclusive, not success.
- High fidelity with no QCNN gain indicates metric/task mismatch or redundancy.
- Degradation at high synthetic ratios is reported as a ratio effect.
- Different winners in pure and mixed regimes are expected to be analyzed separately.
- Failed runs remain in the run ledger and are not silently dropped.

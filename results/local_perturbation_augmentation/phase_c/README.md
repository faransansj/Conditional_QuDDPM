# Phase C — Local quantum-state perturbation benchmark

This is exploratory screening, not publication-level confirmatory evidence. Labels are inherited from anchors; random tangent states are not guaranteed TFIM ground states or universally label preserving. Radius calibration used only the frozen 10/class source subset and was frozen before QCNN execution.

## Reproduction

```bash
.venv/bin/python -m conditional_quddpm.experiments.local_perturbation_augmentation --config configs/augmentation/local_perturbation/phase_c.json --output results/local_perturbation_augmentation/phase_c
.venv/bin/pytest -q
(cd results/local_perturbation_augmentation/phase_c && sha256sum -c manifest.sha256)
```

Protocol: 4-qubit 4→2→1 QCNN, 42 parameters, Z3 readout, MSE, SPSA, 300 configured steps. Versions: `{"numpy": "2.5.2", "python": "3.12.13", "scipy": "1.18.0"}`.

## Frozen radius rule

Per dataset/class, `small=q25`, `medium=q50`, and `large=q75` of the 45 same-class pairwise FS distances in the frozen source subset. q10 was excluded before QCNN because blocked-g class 1 q10 implied duplicate-like infidelity below `1e-4`.

## Results

| Dataset | Radius | Ratio | Test mean | Delta vs real |
|---|---|---:|---:|---:|
| blocked-g | large | 0.5 | 0.8222 | -0.0778 |
| blocked-g | large | 1 | 0.7611 | -0.1389 |
| blocked-g | large | 2 | 0.7611 | -0.1389 |
| blocked-g | medium | 0.5 | 0.7667 | -0.1333 |
| blocked-g | medium | 1 | 0.8056 | -0.0944 |
| blocked-g | medium | 2 | 0.7667 | -0.1333 |
| blocked-g | small | 0.5 | 0.8167 | -0.0833 |
| blocked-g | small | 1 | 0.8833 | -0.0167 |
| blocked-g | small | 2 | 0.8056 | -0.0944 |
| random | large | 0.5 | 0.9333 | +0.0111 |
| random | large | 1 | 0.9222 | +0.0000 |
| random | large | 2 | 0.9167 | -0.0056 |
| random | medium | 0.5 | 0.9500 | +0.0278 |
| random | medium | 1 | 0.9333 | +0.0111 |
| random | medium | 2 | 0.9389 | +0.0167 |
| random | small | 0.5 | 0.9667 | +0.0444 |
| random | small | 1 | 0.9333 | +0.0111 |
| random | small | 2 | 0.9333 | +0.0111 |

Decision: **RESTRICTED-GO**. Full per-seed metrics, diagnostics, provenance, immutable Phase A/B comparisons, and validation fields are in the adjacent machine-readable artifacts.

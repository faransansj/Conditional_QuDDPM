# Frozen RDM-kernel diagnostic

## Previous conclusion and hypothesis shift

```text
Optimization alone insufficient
  -> terminal ensemble coverage insufficient
  -> resampled CRN insufficient
  -> realization pairing was not present
  -> gradient conflicts remain
  -> hypothesis: global fidelity may miss local TFIM structure
  -> frozen RDM-kernel diagnostic
```

The archived history is `archive/quddpm-underfitting-v1` at commit `0076752`. This study uses train/validation only and does not change or retrain an objective for comparison. Because no checkpoint/state artifact existed, the archived deterministic N=4 model was reconstructed exactly once from its frozen config and its outputs were saved to `results/quddpm_kernel_diagnostics/frozen/frozen_states.npz`.

## Kernel definitions

- **Global:** `|<psi|phi>|^2`, numerically identical to the existing implementation.
- **1-RDM:** equal mean Uhlmann fidelity over four single-qubit reduced density matrices.
- **2-RDM:** equal mean Uhlmann fidelity over all six two-qubit reduced density matrices.

All MMD values use the existing biased estimator. Classes are evaluated separately. Uhlmann-fidelity Gram matrices were empirically PSD within numerical tolerance on this sample (`min eigenvalue`: global `-1.0e-14`, 1-RDM `-1.3e-14`, 2-RDM `-1.2e-8`), but a general PSD proof is not assumed.

## K0 results

### Real validation-state class separation

| kernel | class 0 within | class 1 within | between | within-between delta |
|---|---:|---:|---:|---:|
| Global | 0.9445 | 0.9966 | 0.7225 | **0.2480** |
| 1-RDM | 0.9810 | 0.9984 | 0.8817 | 0.1081 |
| 2-RDM | 0.9597 | 0.9972 | 0.7804 | 0.1980 |

Global fidelity separates these operational TFIM classes most strongly. The local kernels, especially 1-RDM, collapse part of the class distinction.

### MMD alignment with TFIM observable error

Twenty-four frozen points cover class 0/1 across seen, unseen, and validation domains and isolated/full-rollout stages.

| kernel | Pearson | Spearman |
|---|---:|---:|
| Global | 0.8131 | 0.7061 |
| 1-RDM | 0.5607 | 0.4670 |
| 2-RDM | **0.9276** | **0.9096** |

2-RDM MMD tracks the mean absolute `Mx`/`Mz2` discrepancy substantially better than global fidelity MMD. 1-RDM is worse than global.

Representative final-rollout aggregate MMD values (raw scales are not directly comparable):

| domain | physics error | Global | 1-RDM | 2-RDM |
|---|---:|---:|---:|---:|
| Seen | 0.4105 | 0.8949 | 0.1935 | 0.6157 |
| Unseen q2 | 0.4970 | 0.9551 | 0.2537 | 0.7234 |
| Validation q2 | 0.5024 | 0.9554 | 0.3167 | 0.7114 |

### Controlled local-X perturbation

| angle | physics error | Global MMD | 1-RDM MMD | 2-RDM MMD |
|---:|---:|---:|---:|---:|
| 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.05 | 0.000112 | 0.000488 | 0.000000 | 0.000204 |
| 0.15 | 0.001002 | 0.004383 | 0.000000 | 0.001833 |
| 0.30 | 0.003987 | 0.017432 | 0.000000 | 0.007295 |

Global and 2-RDM distances increase monotonically. The biased 1-RDM MMD is clipped to zero for this ensemble-level local-unitary perturbation, so it lacks useful sensitivity in this control.

## Interpretation

The evidence is mixed:

- **For 2-RDM:** much stronger alignment with TFIM observable error and sensible perturbation sensitivity.
- **Against immediate replacement:** weaker real class separation than global fidelity; 1-RDM performs poorly; no training-gradient evidence yet shows that 2-RDM supplies a better learnable objective.

This is **partial evidence only**, not enough to select an RDM training objective. Per the stop rule, K1/T=1 training was not run. K2, K3, Haar generation, QCNN augmentation, and test evaluation were not run.

## Decision

**B. Partial evidence only.** 2-RDM is a useful diagnostic metric candidate, but the RDM-objective hypothesis is not yet sufficiently supported for training replacement.

## Next action

The smallest next diagnostic is frozen-checkpoint directional-gradient alignment for global versus 2-RDM MMD at T=1 initialization/best parameters. Proceed to objective training only if 2-RDM both preserves usable class signal and reduces the previously observed class/realization gradient conflict.

# K1 directional-gradient alignment diagnostic

## Setup

Same frozen setting as K0 (T=2, L=3, N=4 train realizations per class, 8 fixed measurement outcomes, train IDs `class-0-00125`/`class-1-00034`, identical forward realizations, conditioning angles, and deterministic PER_OUTCOME_ENSEMBLE_MMD SPSA checkpoints `initial`/`best`). 32 shared Rademacher directions, epsilon 0.15, central finite differences; identical directions, plus/minus points, source states, and measurement uniforms for both kernels, so both kernels see exactly the same generated ensembles. Derivatives are computed from **raw unclipped** biased MMD; the clip is only accounted for. Reported values are directional derivatives, not exact gradient norms. Primary step: rho1->rho0; secondary control: rho2->rho1. Test split untouched. Artifacts: `results/quddpm_kernel_diagnostics/k1_gradient/`. Pre-registered operationalizations: near-zero slack 0.05; catastrophic reversal = conflict increase > 0.05 or alignment drop > 0.10 at the controls.

## Raw-MMD validity

No raw MMD value was negative anywhere (both kernels, all 4 checkpoint/step analyses; 5200 components each). Minimum raw 2-RDM value at the primary point: 1.69e-1. Clipped fraction: 0.0. Min Gram eigenvalues: global -3.6e-15, 2-RDM -3.6e-15 (numerically PSD at these centers). The raw/clipped distinction is not a confound for this diagnostic.

## Primary result: rho1 -> rho0, best checkpoint

| signal | global | 2-RDM |
|---|---:|---:|
| center raw MMD | 0.3106 | 0.2459 |
| mean abs directional derivative | 0.1235 | 0.1273 |
| median abs derivative / std | 0.0965 / 0.1501 | 0.1236 / 0.1453 |
| near-zero fraction / descent fraction | 0.0 / 0.0 | 0.0 / 0.0 |
| class conflict rate (sign) | 0.5625 | 0.5000 |
| all class-realization conflict | 0.4648 | 0.4355 |
| corr/cos vs physics derivative | 0.464 / 0.468 | **0.829 / 0.829** |
| beneficial-direction fraction | 0.781 | **0.844** |

corr(dL_global, dL_2rdm) = 0.655. Signal ratio 2-RDM/global = 1.03. Per-class physics alignment (corr): class 0: global 0.708 / 2-RDM 0.893; class 1: global 0.636 / 2-RDM 0.929.

## Secondary and initialization controls

- rho1->rho0 initial: conflict reduction +0.035; alignment gain +0.169; ratio 1.97; 2-RDM beneficial 0.906 vs global 0.750.
- rho2->rho1 initial: alignment gain +0.303; ratio 0.99.
- rho2->rho1 best: alignment gain +0.448; ratio 1.26; conflict change **-0.020** (2-RDM conflict slightly higher: 0.449 vs 0.430); beneficial 0.781 vs 0.719.

## Decision

Fixed rule (thresholds declared in `configs/quddpm/kernel_k1.yaml` before the run):

| check | value | threshold | pass |
|---|---:|---:|---|
| validity min raw | 0.169 | >= -1e-8 | yes |
| validity clipped fraction | 0.0 | <= 0.05 | yes |
| usable signal ratio | 1.03 | >= 0.5 | yes |
| usable near-zero | 0.0 vs 0.0 | slack 0.05 | yes |
| **conflict reduction** | **0.0293** | **>= 0.05** | **no** |
| physics alignment gain | 0.365 | >= 0.10 | yes |
| beneficial not lower | 0.844 vs 0.781 | >= | yes |
| consistency | see controls | not catastrophic | yes |

**RDM OBJECTIVE: NO-GO.** 2-RDM gives a clearly better physics-aligned directional signal (+0.36 correlation gain, higher beneficial fraction), but it does not reduce the class/realization conflict that blocks the generation gate: +0.029 at the primary point (below the 0.05 bar) and slightly negative at the rho2->rho1 best control. Since the gate blocker is conflict, not physics-metric alignment, the RDM-objective hypothesis is frozen; no replacement objective is trained.

## Next action after K1

Smallest remaining bottleneck experiment: attack the class/realization conflict directly at the frozen checkpoints (e.g., conflict-aware reweighting or per-realization objective terms evaluated with the existing global MMD, measured with this same directional diagnostic), rather than swapping the kernel. K0 numbers above are unchanged. 4Q generation gate not retested; QCNN augmentation still blocked.

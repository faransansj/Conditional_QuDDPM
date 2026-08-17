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

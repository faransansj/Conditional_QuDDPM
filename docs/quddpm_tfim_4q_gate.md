# 4-qubit TFIM Conditional QuDDPM Learning Gate

## Decision

**NO-GO for QCNN augmentation.** The current small-scale Conditional QuDDPM does not yet learn the 4-qubit TFIM class distributions, despite passing physicality, reproducibility, loss-decrease, and weak validation-MMD improvement checks.

This negative result is retained as a research result. MSQuDDPM and augmentation experiments remain blocked.

## Protocol

- Dataset: `data/tfim_4q_random/`
- Train subset: 10 states/class, nested subset seed `31415`
- Training access: train IDs only
- Diagnostics: validation split only
- Test evaluation: disabled
- Data qubits: 4
- Ancillas: 2
- Diffusion steps: 2
- Reverse layers: 6
- Shared conditioning angles: class 0 → `0`, class 1 → `π`
- SPSA iterations: 1000/reverse step
- Generated samples: 32/class

Run:

```bash
uv run run-quddpm-tfim \
  --config configs/quddpm/tfim_4q_smoke.yaml \
  --output results/quddpm_tfim_4q_smoke
```

The command exits nonzero while the learning gate fails.

## Preregistered checks

A distribution is considered learned only when all are true:

1. generated states are normalized/pure;
2. serialized-checkpoint generation is deterministic;
3. every reverse-step training loss decreases;
4. generated-to-validation MMD improves over Haar for both classes;
5. generated TFIM observables preserve class ordering:
   - class 0 has larger `⟨Mz²⟩`;
   - class 1 has larger `⟨Mx⟩`.

No test metric or QCNN performance is consulted.

## Observed result

| Check | Result |
|---|---|
| Physicality | Pass, max norm error `4.44e-16` |
| Checkpoint reproducibility | Pass |
| Reverse step 0 loss | `0.8499 → 0.5316` |
| Reverse step 1 loss | `0.3974 → 0.2434` |
| Validation MMD improvement, class 0 | `0.9387 → 0.8937` |
| Validation MMD improvement, class 1 | `0.9639 → 0.9403` |
| TFIM observable class ordering | **Fail** |
| Overall learning gate | **Fail** |

Generated observables remained close to Haar rather than TFIM:

| Class | Generated `⟨Mx⟩` | Validation `⟨Mx⟩` | Generated `⟨Mz²⟩` | Validation `⟨Mz²⟩` |
|---|---:|---:|---:|---:|
| 0 | `0.0146` | `0.4485` | `0.2618` | `0.8361` |
| 1 | `0.0580` | `0.9112` | `0.2653` | `0.4317` |

Generated cross-class MMD was `0.0901`, while target cross-class MMD was `0.6499`. The reverse model therefore produced two weakly differentiated, near-Haar ensembles instead of the two TFIM distributions.

Low nearest-training fidelity (`0.0951` and `0.0874` mean) indicates that this failure is not memorization; the model is underfitting the structured distributions.

## Diagnosis

The one-qubit validation established algorithmic correctness but not 4-qubit capacity. Compared with the published many-body reference (`T=30`, 2 ancillas, `L=12`, 3001 epochs/step), this gate intentionally used a much cheaper `T=2`, `L=6`, 1000-iteration setup. The failure indicates that the current reverse capacity/schedule or SPSA optimization is insufficient at four qubits.

## Next work inside Phase 3

Before any augmentation:

1. profile a reference-scale progression over diffusion steps and circuit depth using train/validation only;
2. improve the forward schedule so terminal states approach Haar without losing a learnable stepwise path;
3. replace SPSA or add a differentiable backend for the larger parameter count;
4. validate each reverse step separately against its forward target;
5. require TFIM observable recovery and cross-class distribution separation;
6. rerun with more than one optimization seed once a configuration passes.

No QCNN test or augmentation result should be produced until this gate passes.

# K3 one-shot conflict-aware reweighting diagnostic

K2 gradient sketches were loaded verbatim and the identical frozen train-only checkpoint and CRN probe were replayed. No gradient was re-estimated, no iterative training was run, and validation/test were not evaluated.

## Weighting

- conflict score: mean pairwise cosine to all other realization gradients
- conflict weight: `softmax(tau * score)`
- physics-conflict weight: `normalize(exp(tau * score) * (0.05 + max(0, cosine(g_i, g_physics))))`
- primary tau: 1.0; sensitivity tau: [0.5, 1.0, 2.0]

| method | N_eff | class 0 weight | cancellation | cosine physics |
|---|---:|---:|---:|---:|
| uniform | 8.000 | 0.500 | 0.300 | 0.414 |
| conflict_tau_1p0 | 7.947 | 0.520 | 0.317 | 0.401 |
| physics_conflict_tau_1p0 | 3.548 | 0.228 | 0.622 | 0.839 |

## One-step probe

| method | global MMD delta | physics delta | 2-RDM delta | realization improved fraction |
|---|---:|---:|---:|---:|
| global_mmd | -0.000887 | -0.000531 | -0.003271 | 0.625 |
| 2rdm | -0.000633 | -0.002626 | -0.009094 | 0.375 |
| conflict_tau_1p0 | -0.000569 | -0.000448 | -0.003299 | 0.500 |
| physics_conflict_tau_1p0 | +0.006933 | -0.002016 | -0.001643 | 0.375 |

## Conclusion

**K3-B.** Conflict-only weighting improves both objectives locally but is weaker than the uniform global baseline on both deltas and improves fewer realizations. Physics-aligned weighting improves physics while worsening global MMD and concentrating 77% of weight in class 1 (N_eff 3.55). Conflict-aware reweighting therefore does not resolve the objective incompatibility under this frozen configuration. Further QuDDPM objective engineering is not justified; recommend stopping this augmentation track.

Limitations: one frozen checkpoint/configuration, a 32-direction sketch, and one train state per class.

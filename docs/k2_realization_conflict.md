# K2 per-realization gradient conflict diagnostic

## Scope and method

K2 tests whether the frozen global-MMD objective loses useful realization-specific directions through aggregation. It does not train a new objective. The deterministic K1 `rho1 -> rho0` best checkpoint was reconstructed from commit `451216c799e9f7abd8cdba59deeb3c9cd1a43ae3` with the same T=2, L=3, N=4 per class, R=8, SPSA training, 32 Rademacher directions, central-difference epsilon 0.15, and common measurement/perturbation randomness. Only train states `class-0-00125` (`g=0.2265828901`) and `class-1-00034` (`g=1.2840912833`) were used; each realization is one forward-noise seed of its class state. Validation and test were not evaluated. The recorded run used the repository's generic dataset loader, which materialized all split arrays; the K2 loader was corrected afterward to materialize train state rows only, without changing numerical artifacts.

For generated states `G`, targets `T`, and raw biased global-fidelity MMD, K2 allocates

```text
A_ij = [k(g_i,g_j) + k(t_i,t_j) - k(g_i,t_j) - k(t_i,g_j)] / N^2
a_i  = sum_j A_ij
```

and defines each reported realization objective as `N*a_i`. Therefore the mean over all `2N` realization objectives reconstructs the equally weighted two-class global MMD. Raw/clipped semantics are not mixed. Directional derivatives are converted to K1-compatible parameter sketches by `mean_d derivative_d * Rademacher_direction_d` in NumPy C-order; these are 32-direction sketches of 81 parameters, not exact autodiff gradients.

## Reconstruction

| check | result |
|---|---:|
| all 32 directional reconstructions | pass |
| maximum directional absolute residual | 4.996e-16 |
| global sketch relative error | 1.817e-15 |
| class 0 sketch relative error | 1.035e-15 |
| class 1 sketch relative error | 1.018e-15 |
| configured tolerance | atol 1e-10 + rtol 1e-8 |

## Conflict and cancellation

| scope | cancellation ratio |
|---|---:|
| all 8 realizations | **0.2995** |
| class 0 | 0.6077 |
| class 1 | 0.4936 |

| pair group | count | mean cosine | median | std | min | max | negative fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| within class / same `g` | 12 | 0.0864 | 0.0228 | 0.3971 | -0.5875 | 0.7088 | **0.5000** |
| between class / far `g` | 16 | -0.1352 | -0.0786 | 0.2030 | -0.5513 | 0.2167 | **0.7500** |

Because this frozen train set has one TFIM state per class, `same g` equals within-class and `far g` equals between-class; the data cannot independently separate class from `g`-distance effects. Conflict is nevertheless present inside each class, not only between labels.

Realization-to-global cosine had mean 0.2926, range -0.0453 to 0.6987, and negative fraction 0.125. Realization-to-2-RDM cosine had mean 0.1989, range -0.1146 to 0.5604, and negative fraction 0.375. Global-to-2-RDM cosine was 0.6173; global-to-physics was 0.4138; 2-RDM-to-physics was 0.8525, consistent with K1's stronger physics alignment but not reduced conflict.

## One-step probes

All probes use `theta' = theta - 0.15 * normalized_sketch` and the same fixed generated-sample randomness.

Across all eight realization directions:

| delta | mean | range | improving fraction |
|---|---:|---:|---:|
| target decomposed objective | **-0.03915** | [-0.05656, -0.01150] | **1.000** |
| target paired physics error | **-0.01377** | [-0.02930, -0.00086] | **1.000** |
| all non-target objective mean | +0.01394 | [+0.00804, +0.01888] | 0.000 |
| same-class non-target mean | +0.01142 | [-0.00253, +0.02855] | 0.125 |
| other-class mean | **+0.01583** | [+0.00475, +0.02713] | 0.000 |
| global MMD | **+0.00731** | [+0.00465, +0.00945] | 0.000 |
| aggregate physics error | +0.00075 | [-0.00207, +0.00317] | 0.250 |
| class-separation metric | -0.00269 | [-0.00854, +0.00250] | 0.250 |

Every realization direction improves its own decomposed objective and paired TFIM observable error, while every one worsens the mean non-target objective, the other class, and global MMD. Seven of eight also worsen same-class non-target objectives. This is direct structured cancellation rather than merely small realization gradients (norms 0.691-1.192).

Baseline normalized probes provide the complementary result:

| direction | global MMD delta | 2-RDM delta | aggregate physics delta |
|---|---:|---:|---:|
| global MMD | -0.000887 | -0.003271 | -0.000531 |
| class 0 aggregate | +0.004769 | +0.001028 | +0.000974 |
| class 1 aggregate | +0.003822 | +0.001072 | -0.000577 |
| 2-RDM | -0.000633 | -0.009094 | -0.002626 |

## Conclusion

**Pattern A — aggregation conflict hypothesis supported at this frozen checkpoint.** Exact reconstruction, low overall cancellation (0.2995), 50% within-class and 75% between-class negative pair cosines, and the unanimous target-improvement/non-target-harm one-step pattern jointly show that global MMD averages individually useful realization directions against one another. The evidence is bounded to one deterministic checkpoint/seed and a 32-direction sketch, so it establishes a plausible bottleneck rather than a general theorem.

`K3 one-shot conflict-aware reweighting` is justified as one bounded follow-up experiment. K3 was not implemented here. The 4Q generation gate, QuDDPM retraining, QCNN augmentation, and test evaluation remain untouched.

Artifacts: `results/quddpm_kernel_diagnostics/k2_realization/` (ignored by Git per repository convention).

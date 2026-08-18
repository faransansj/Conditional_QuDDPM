# K3 one-shot conflict-aware reweighting diagnostic

## Scope and method

K3 asks whether simple nonnegative reweighting of the frozen K2 realization-gradient sketches produces a more jointly useful local direction than uniform global MMD. It loads `results/quddpm_kernel_diagnostics/k2_realization/per_realization_gradients.npz` verbatim, preserves NumPy C-order, and verifies the uniform reconstruction error (`1.817e-15`). The deterministic `rho1 -> rho0` best checkpoint is replayed only for the one-step probe using the same train states, seeds, common measurement randomness, 32 Rademacher directions, and step size 0.15. Gradients are not re-estimated. Validation and test are not evaluated.

For realization sketches `g_i`, define

```text
s_i = mean_{j != i} cosine(g_i, g_j)
p_i = max(0, cosine(g_i, g_physics))

uniform:          w_i = 1/8
conflict:         w_i = exp(tau*s_i) / sum_j exp(tau*s_j)
physics-conflict: w_i = exp(tau*s_i)*(0.05+p_i) / sum_j exp(tau*s_j)*(0.05+p_j)
```

The predeclared primary is `tau=1.0`; `tau={0.5,1.0,2.0}` is a bounded sensitivity diagnostic, not model selection.

## Weight statistics

| method | min | max | std | N_eff | class 0 | class 1 |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.1250 | 0.1250 | 0.0000 | 8.000 | 0.500 | 0.500 |
| conflict, tau=1 | 0.1128 | 0.1436 | 0.0102 | 7.947 | 0.520 | 0.480 |
| physics-conflict, tau=1 | 0.0224 | 0.3594 | 0.1400 | 3.548 | 0.228 | 0.772 |

Sensitivity did not change the qualitative behavior. Conflict-only `N_eff` was 7.987/7.947/7.784 for tau 0.5/1/2. Physics-conflict `N_eff` was 3.514/3.548/3.616, with class-1 weight 0.781/0.772/0.751. Physics alignment therefore causes substantial realization and class concentration despite the floor.

## Aggregate gradient statistics

| method | norm | weighted cancellation | cos global | cos 2-RDM | cos physics | mean realization cos | negative fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.2526 | 0.2995 | 1.0000 | 0.6173 | 0.4138 | 0.2926 | 0.125 |
| conflict, tau=1 | 0.2665 | 0.3168 | 0.9947 | 0.6182 | 0.4008 | 0.2947 | 0.125 |
| physics-conflict, tau=1 | 0.5172 | 0.6224 | 0.4572 | 0.7156 | 0.8388 | 0.1565 | 0.500 |

Conflict-only weighting barely changes the uniform direction and slightly lowers physics alignment. Physics-conflict raises the cancellation ratio and physics alignment, but half the realization gradients become negatively aligned.

## One-step probe

All rows use `theta' = theta - 0.15*g/||g||` and identical CRN.

| method | global MMD Δ | physics Δ | 2-RDM Δ | Mx Δ | Mz² Δ | mean task Δ | worst task Δ | improved | same-class damage | other-class damage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global MMD | -0.000887 | -0.000531 | -0.003271 | +0.000783 | -0.001845 | -0.000887 | +0.012430 | 0.625 | 0.003360 | 0.005018 |
| 2-RDM | -0.000633 | -0.002626 | -0.009094 | -0.002884 | -0.002368 | -0.000633 | +0.018426 | 0.375 | 0.006777 | 0.003336 |
| conflict, tau=1 | -0.000569 | -0.000448 | -0.003299 | +0.001038 | -0.001933 | -0.000569 | +0.014702 | 0.500 | 0.003260 | 0.003860 |
| physics-conflict, tau=1 | +0.006933 | -0.002016 | -0.001643 | -0.004154 | +0.000122 | +0.006933 | +0.043406 | 0.375 | 0.016982 | 0.018241 |

Here `same-class damage` is the mean positive realization-objective delta after grouping tasks within classes; `other-class damage` is the maximum positive class-aggregate delta. The machine-readable output also contains median task delta, class-separation delta, both class deltas, and every realization result.

## Conclusion

**K3-B — MMD/physics trade-off persists.**

Conflict-only weighting is jointly descending but is weaker than uniform on both global MMD and aggregate physics and improves fewer realizations (0.50 vs 0.625). Physics-aligned weighting improves physics but reverses global MMD, worsens 5/8 realization objectives, and concentrates 77% of weight in class 1 (`N_eff=3.55`). This is not evidence that conflict-aware aggregation provides a better joint direction.

Under this frozen 4Q TFIM diagnostic configuration, simple conflict-aware reweighting does not resolve the objective incompatibility. Further QuDDPM objective engineering is not justified; recommend stopping the QuDDPM augmentation track rather than running K4 or the generation gate.

The conclusion remains limited to one frozen checkpoint/configuration, one train state per class, and a 32-direction gradient sketch. It does not claim a general TFIM or conditional-generation result.

Artifacts: `results/quddpm_kernel_diagnostics/k3_conflict_reweighting/`.

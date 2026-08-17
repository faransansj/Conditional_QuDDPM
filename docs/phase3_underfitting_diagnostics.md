# Phase 3 T=1 underfitting diagnosis

## FACT

- Branch: `diagnostic/quddpm-underfitting`
- Dataset: `data/tfim_4q_random`; train IDs only: `class-0-00125`, `class-1-00034`.
- Validation was diagnostic-only. The test split was not evaluated.
- `T=1`, two ancillas, fixed seeds/objective/conditioning. The learning gate was not changed.
- Reproduce: `uv run run-quddpm-diagnostics --config configs/quddpm/underfitting_diagnostics.yaml --output results/quddpm_underfitting_diagnostics`

## RESULT

### SPSA budget progression (`L=3`)

| iterations | source | best loss | physics gap error | Mx order | Mz2 order |
|---:|---|---:|---:|---|---|
| 8 | teacher | 1.678 | 0.455 | pass | pass |
| 8 | Haar | 1.756 | 0.576 | fail | pass |
| 32 | teacher | 1.287 | 0.438 | pass | fail |
| 32 | Haar | 1.488 | 0.700 | fail | pass |
| 128 | teacher | 0.620 | 0.168 | pass | pass |
| 128 | Haar | 0.692 | 0.364 | pass | pass |
| 512 | teacher | 0.244 | 0.142 | pass | pass |
| 512 | Haar | **0.211** | **0.079** | pass | pass |

At 512 Haar-source iterations, class 0/1 train MMD was 0.145/0.319, validation MMD 0.244/0.315, and nearest-train fidelity 0.928/0.840. Generated `(class0-class1)` gaps were `Mx=-0.569`, `Mz2=0.530`; train-reference gaps were `Mx=-0.476`, `Mz2=0.411`.

### Capacity progression (`128` SPSA iterations)

| L | source | best loss | physics gap error | ordering |
|---:|---|---:|---:|---|
| 1 | teacher | 0.927 | **0.064** | both pass |
| 1 | Haar | 1.227 | 0.519 | Mx fail |
| 3 | teacher | **0.620** | 0.168 | both pass |
| 3 | Haar | **0.692** | 0.364 | both pass |
| 6 | teacher | 0.963 | 0.288 | both pass |
| 6 | Haar | 0.861 | **0.126** | both pass |

Greater depth did not improve results systematically. `L=3` with a larger optimization budget was better than selecting depth from the 128-iteration comparison.

### Teacher-forced versus Haar source

At 8–128 iterations teacher forcing generally had lower loss/physics error. At 512 iterations Haar-source fitting recovered both observable orders and had lower loss and gap error than teacher forcing. The early source discrepancy therefore disappears with optimization and is not evidence of a persistent source-distribution mismatch at `T=1`.

### Conditioning ablation (`L=3`, 128 iterations)

| class | source | best loss | observable error | nearest-train fidelity |
|---:|---|---:|---:|---:|
| 0 | teacher | 0.693 | 0.129 | 0.498 |
| 0 | Haar | 0.372 | 0.127 | 0.709 |
| 1 | teacher | 0.296 | 0.092 | 0.852 |
| 1 | Haar | 0.179 | 0.048 | 0.911 |

Single-class fitting works better for class 1 than class 0, but the two-class conditional model also recovers both class directions at adequate budget. There is no supported conditioning-collapse classification from this run.

### Optimizer control

One bounded `L=3`, teacher-forced L-BFGS control (`maxiter=2`) reached loss 1.842 and physics error 0.568. It was deliberately bounded and is too weak to establish optimizer superiority. A broad L-BFGS sweep was not run.

### Objective versus physics

Across conditional runs, correlation between objective improvement and physics gap error was `-0.766`: larger objective improvements generally accompanied lower physics error. Individual low-budget runs were non-monotonic, but the campaign does not support a primary objective/physics mismatch at `T=1`.

## INTERPRETATION

**Best-supported classification: UNDER-OPTIMIZED.** Increasing SPSA from 8 to 512 iterations materially improved loss, MMD, nearest-target fidelity, observable gaps, and both class-order checks.

Not supported by this single-seed campaign: representation failure, systematic capacity limitation, conditioning failure, persistent source mismatch, or objective/physics mismatch. Optimizer limitation remains uncertain because the deterministic control was intentionally minimal. The result proves tiny-set learnability only; it does not recover the existing full learning gate or establish validation generalization.

## NEXT ACTION

Repeat the `T=1, L=3, 512-iteration` conditional run over the existing intended optimization seeds. Require stable train/validation distribution metrics and inferred TFIM ordering. If replicated, the next experiment is a bounded `T=2` per-step teacher-forced versus rollout diagnosis—not augmentation.

## Multi-seed reproduction and gated T=2 follow-up

### FACT

The `T=1, L=3, SPSA=512` Haar-source configuration was repeated with seed offsets `0, 1, 2`. The gate required loss improvement, both train-inferred observable orders, and per-class validation MMD better than the Haar reference. All three passed.

| seed offset | best loss | physics gap error | class ordering |
|---:|---:|---:|---|
| 0 | 0.211 | 0.079 | both pass |
| 1 | 0.232 | 0.146 | both pass |
| 2 | 0.173 | 0.045 | both pass |

A reverse-target indexing defect found before the T=2 run was corrected: transition `rho_(step+1) -> rho_step` now targets `forward[step]`; the previous diagnostic helper incorrectly used `forward[step-1]` for steps after the first. This did not affect T=1, but would have invalidated T=2.

### RESULT

The gated T=2 run used `L=3`, 512 SPSA iterations per independently teacher-forced transition. Evaluation sampled 32 ancilla-measurement outcomes rather than reusing the single optimized outcome.

| mode | transition | MMD class 0/1 | physics gap error | ordering |
|---|---|---|---:|---|
| teacher-forced | rho1 -> rho0 | 0.337 / 0.277 | 0.137 | both pass |
| teacher-forced | rho2 -> rho1 | 0.461 / 0.366 | 0.127 | both pass |
| rollout from rho2 | rho2 -> rho1 | 0.687 / 0.334 | 0.138 | both pass |
| rollout from rho2 | predicted rho1 -> rho0 | 0.905 / 0.518 | 0.469 | both pass |
| rollout from Haar | Haar -> rho1 | 1.103 / 1.212 | 0.420 | both pass |
| rollout from Haar | predicted rho1 -> rho0 | 0.953 / 1.001 | 0.605 | Mx fails |

### INTERPRETATION

T=1 tiny learnability is reproducible for the three intended diagnostic seeds. At T=2, both isolated teacher-forced transitions learn useful mappings, but chaining them materially degrades MMD and observable-gap accuracy. This is evidence of **rollout error accumulation/distribution shift**. Starting from Haar adds a separate **terminal-source mismatch** and loses Mx ordering. It is not a per-step representation failure.

### Measurement-averaged objective and off-target robustness

The prescribed follow-up kept `T=2`, `L=3`, and 512 SPSA iterations, but averaged each training objective over 8 fixed measurement outcomes. Evaluation continued to use 32 outcomes.

| mode | transition | MMD class 0/1 | physics gap error | ordering |
|---|---|---|---:|---|
| teacher-forced | rho1 -> rho0 | 0.365 / 0.289 | 0.158 | both pass |
| teacher-forced | rho2 -> rho1 | 0.111 / 0.280 | 0.087 | both pass |
| rollout from rho2 | rho2 -> rho1 | 0.110 / 0.301 | 0.077 | both pass |
| rollout from rho2 | predicted rho1 -> rho0 | 0.463 / 0.585 | 0.227 | both pass |
| rollout from Haar | Haar -> rho1 | 1.852 / 1.644 | 0.491 | Mz2 fails |
| rollout from Haar | predicted rho1 -> rho0 | 1.697 / 1.681 | 0.635 | both fail |

Averaging measurement outcomes reduced rho2-rollout final physics error from 0.469 to 0.227 and class-0 MMD from 0.905 to 0.463. It did not solve Haar initialization.

Small random-unitary off-target perturbations were tested independently at strengths 0, 0.05, 0.1, and 0.2. Both reverse transitions retained both observable orders throughout. Physics error ranges were 0.157–0.220 for rho1->rho0 and 0.073–0.106 for rho2->rho1. The maps are locally robust to these perturbations; rollout degradation therefore reflects a larger structured distribution shift rather than ordinary small input noise.

### INTERPRETATION UPDATE

The evidence now supports three simultaneous findings:

1. **Measurement-outcome overfitting contributed to rollout error**, because averaging outcomes substantially improved rho2 rollout.
2. **Residual rollout distribution shift remains**, because chained rho2 rollout is still worse than either isolated transition despite local perturbation robustness.
3. **Terminal Haar/source mismatch is dominant**, because Haar rollout has MMD above 1.6 and loses both observable orders, while rollout from the true rho2 remains useful.

### NEXT ACTION

Keep Phase 4 blocked and do not increase T. Measure the actual terminal `rho2`-to-Haar discrepancy and train/evaluate the terminal reverse map on a controlled mixture interpolating between forward `rho2` and Haar. This should determine whether the forward schedule must be changed or the reverse map must explicitly cover the generation source. Preserve the 8-outcome objective.

## Terminal-prior diagnostic

### FACT

`Haar` means independent normalized complex-Gaussian 4-qubit pure states. Classes use the same distribution law with independent deterministic seeds. Forward terminal ensembles use 64 independent forward draws from the same selected train state per class. Mixtures use Bernoulli selection between empirical q2 and Haar samples, which represents the convex density-operator mixture without linearly interpolating statevectors. Training averages 8 fixed ancilla outcomes; evaluation samples 32 outcomes.

The selected real references are class 0 `(Mx=0.182, Mz2=0.973)` and class 1 `(Mx=0.884, Mz2=0.468)`, giving gaps `Mx=-0.702`, `Mz2=0.505`.

### RESULT: forward-only convergence

| T | aggregate MMD to Haar | terminal class MMD | Mx gap | Mz2 gap | purity |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0414 | 0.0240 | -0.1407 | 0.0689 | 1.000 |
| 2 | 0.0307 | 0.0182 | -0.0861 | 0.0166 | 1.000 |
| 3 | 0.0286 | 0.0145 | -0.0225 | 0.0268 | 1.000 |
| 4 | 0.0266 | 0.0154 | -0.0434 | 0.0141 | 1.000 |
| 5 | 0.0295 | 0.0138 | -0.0070 | 0.0114 | 1.000 |

The matched independent Haar-vs-Haar class MMD is 0.0278 at this sample size. Thus q2 aggregate MMD 0.0307 is already at approximately the finite-sample Haar baseline. Pairwise terminal fidelity at T=2 is 0.0626/0.0662 by class, close to the 4-qubit Haar expectation. Nearest terminal-to-Haar fidelity is 0.282/0.275.

TFIM class gaps collapse sharply by T=1 and are nearly absent by T=2. Increasing T beyond 2 does not produce a meaningful further MMD improvement.

### RESULT: evaluation-only mixture curve

The baseline model had been trained on one exact q2 trajectory. Alpha 0 below uses fresh draws from the q2 distribution, not that memorized trajectory.

| alpha (Haar fraction) | final MMD | final physics error | Mx order | Mz2 order |
|---:|---:|---:|---|---|
| 0.00 | 0.984 | 0.613 | pass | fail |
| 0.10 | 0.976 | 0.599 | pass | fail |
| 0.25 | 0.994 | 0.596 | pass | fail |
| 0.50 | 0.913 | 0.580 | pass | pass |
| 0.75 | 0.967 | 0.576 | pass | fail |
| 1.00 | 1.004 | 0.593 | pass | fail |

Failure is already present at alpha 0 for fresh q2 samples and remains roughly flat across the curve. The previous successful “true rho2” rollout used the exact terminal states seen while fitting each reverse map; it did not establish coverage of q2 as a distribution.

### RESULT: trained-coverage control

One terminal-step control used 8 fixed source/outcome pairs sampled from a 75% q2 / 25% Haar empirical mixture. The downstream rho1->rho0 map was unchanged. Terminal-step objective decreased only from 1.896 to 1.631.

| alpha | final MMD | final physics error | ordering |
|---:|---:|---:|---|
| 0.00 | 0.956 | 0.594 | Mx fail |
| 0.10 | 0.955 | 0.604 | Mx fail |
| 0.25 | 0.939 | 0.572 | both pass |
| 0.50 | 0.955 | 0.579 | both pass |
| 0.75 | 0.971 | 0.605 | Mz2 fail |
| 1.00 | 0.958 | 0.593 | Mx fail |

The limited coverage control did not materially restore q2 or Haar rollout.

### INTERPRETATION

**Best-supported diagnosis: reverse OOD / terminal-coverage failure, not insufficient forward diffusion.** Direct metrics show q2 is already Haar-like at the finite-sample resolution, while the reverse map fails even on fresh q2 draws at alpha 0. The model learned specific terminal trajectories and fixed measurement samples rather than the support of q2. Increasing forward T would erase the small remaining class structure without resolving this coverage failure.

The mixture control's failure does not prove mixture-aware training cannot work: eight fixed source/outcome pairs and one SPSA seed are a deliberately small control, and its objective remained high. What is established is that mild exposure under the frozen budget is insufficient.

### NEXT ACTION

Keep T=2 and Phase 4 blocked. The next diagnostic should replace one-state empirical fitting with a small train-only terminal ensemble per class and test held-out forward-noise realizations from the same q2 distribution. Only after within-q2 generalization succeeds should Haar-mixture exposure be reconsidered. Do not increase T or capacity first.

## Within-q2 forward-realization generalization

### FACT

The frozen `T=2`, `L=3`, SPSA-512 model was trained with `N_train=1,4,8` independent forward realizations per selected train state/class. Each realization has its own recorded forward seed (`base + 100000*class + realization`). Holdout uses 16 disjoint train-state realizations from seed base 10121; validation uses one unseen validation state/class with 16 realizations from seed base 20121. Training averages 8 measurement-outcome vectors; evaluation uses 32 independent outcomes per realization. Seed/realization IDs are stored in the JSON artifact and were checked for overlap. The test split was not loaded.

`N=16` was omitted because fixed 512-step, 8-outcome training scales linearly; `N=1,4,8` already required about 20.6 minutes and answered the trend question.

### RESULT: isolated steps

| N | domain | rho1->rho0 MMD / physics | rho2->rho1 MMD / physics | ordering summary |
|---:|---|---|---|---|
| 1 | seen | 0.193 / 0.152 | 0.143 / 0.059 | both pass |
| 1 | unseen q2 | 0.755 / 0.529 | 0.283 / 0.374 | failures in both steps |
| 1 | validation q2 | 0.807 / 0.452 | 0.308 / 0.388 | failures in both steps |
| 4 | seen | 0.319 / 0.275 | 0.470 / 0.209 | both pass |
| 4 | unseen q2 | 0.551 / 0.401 | 0.263 / 0.372 | second step partial fail |
| 4 | validation q2 | 0.561 / 0.240 | 0.322 / 0.314 | second step partial fail |
| 8 | seen | 0.308 / 0.292 | 0.370 / 0.250 | both pass |
| 8 | unseen q2 | 0.522 / 0.384 | 0.252 / 0.354 | second step partial fail |
| 8 | validation q2 | 0.519 / 0.219 | 0.293 / 0.298 | both pass |

Forward-realization diversity improves unseen isolated rho1->rho0 MMD from 0.755 to 0.522 and restores its ordering, but improvement saturates. Unseen rho2->rho1 changes only from 0.283 to 0.252 and still fails one observable order at N=8.

### RESULT: full rollout

| N | domain | first-step MMD / physics | final MMD / physics | final ordering |
|---:|---|---|---|---|
| 1 | seen | 0.147 / 0.066 | 0.256 / 0.141 | both pass |
| 1 | unseen q2 | 0.283 / 0.374 | 0.939 / 0.565 | both pass |
| 1 | validation q2 | 0.307 / 0.388 | 0.943 / 0.485 | both fail |
| 4 | seen | 0.476 / 0.207 | 0.894 / 0.506 | both pass |
| 4 | unseen q2 | 0.262 / 0.371 | 0.936 / 0.615 | both fail |
| 4 | validation q2 | 0.322 / 0.314 | 0.955 / 0.440 | Mz2 fails |
| 8 | seen | 0.369 / 0.248 | 0.928 / 0.578 | both pass |
| 8 | unseen q2 | 0.253 / 0.358 | 0.899 / 0.606 | Mx fails |
| 8 | validation q2 | 0.292 / 0.302 | 0.912 / 0.443 | both pass |

The unseen-minus-seen final MMD gap changes from +0.683 at N=1 to +0.041 at N=4 and -0.029 at N=8. This apparent gap closure is primarily caused by seen rollout degrading (0.256 to 0.928), not unseen rollout becoming good (0.939 to 0.899). There is no successful ensemble size to replicate across optimizer seeds.

### RESULT: memorization diagnostic

For final rollout at N=1, mean nearest-training-source fidelity is 0.666/0.336 on seen trajectories but only 0.076/0.067 on unseen trajectories. At N=8 this becomes 0.204/0.218 seen and 0.191/0.179 unseen. Increased coverage removes the extreme trajectory-closeness gap, while absolute rollout quality remains poor. This is reduced memorization without successful support-level learning.

Validation-state results broadly track unseen train-state q2 results, and N=8 isolated validation transitions recover both orders. The dominant failure is therefore not validation-state-only generalization.

### INTERPRETATION

**Decision C: even unseen single-step q2 support is not learned adequately under the frozen objective/configuration.** Ensemble coverage produces a modest isolated-step improvement and reduces trajectory memorization, but it does not yield a convincing or monotonic rollout improvement. The closed seen/unseen gap at N=8 is not success because seen performance collapses toward unseen performance. Residual rollout error accumulation remains, but isolated rho2->rho1 also remains imperfect.

### NEXT ACTION

Keep `T=2`, `L=3`, and Phase 4 blocked. Before adding capacity, inspect the ensemble objective itself: determine whether fidelity-MMD over stochastic measured pure-state outputs provides a usable gradient for a multi-realization target, and compare fixed-pair versus resampled mini-batch source/outcome semantics on the existing N=4 case. Do not increase N or repeat across seeds until one configuration improves unseen performance without sacrificing seen performance.

## Fixed versus iteration-resampled objective randomness

### FACT

The frozen N=4 experiment compared only training sampling semantics. `FIXED_ALL` reused the same four forward source/target pairs and eight measurement-outcome vectors for every SPSA iteration. `RESAMPLED_CRN` bootstrapped four paired source/target realizations from the same N=4 pool and sampled new outcome vectors at every iteration. Its `theta+c*delta` and `theta-c*delta` evaluations received the exact same batch object and fingerprint; all 1,024 gradient estimates passed this CRN check, while all 513 batches per step were distinct across iterations. A regression test asserts same-batch +/- and changed next-iteration behavior.

Both conditions used 1,537 objective calls per step (3,074 total) and 98,368 class/source/outcome evaluations per step (196,736 total). Architecture, seeds outside objective sampling, SPSA schedule, realization pools, and evaluation were matched. `INDEPENDENT_PM` was skipped because the two required conditions already took 12.7 minutes and it is not a candidate configuration.

### RESULT: optimizer noise

| condition | step | initial | best | final | objective variance | directional variance | mean gradient norm | unique batches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIXED_ALL | rho1->rho0 | 1.322 | 0.352 | 0.352 | 0.0322 | 0.0576 | 1.646 | 1 |
| FIXED_ALL | rho2->rho1 | 0.630 | 0.473 | 0.473 | 0.0011 | 0.0095 | 0.700 | 1 |
| RESAMPLED_CRN | rho1->rho0 | 1.338 | 0.371 | 0.411 | 0.0379 | 0.1658 | 2.937 | 513 |
| RESAMPLED_CRN | rho2->rho1 | 0.733 | 0.417 | 0.831 | 0.0277 | 0.0855 | 1.980 | 513 |

Resampling increased directional-derivative variance by about 2.9x and 9.0x for the two steps. CRN prevented within-gradient batch mismatch, but did not make the stochastic objective stable across iterations.

### RESULT: isolated steps

| condition | domain | rho1->rho0 MMD / physics | rho2->rho1 MMD / physics | ordering |
|---|---|---|---|---|
| FIXED_ALL | seen | 0.328 / 0.322 | 0.451 / 0.196 | both pass |
| FIXED_ALL | unseen | 0.512 / 0.395 | 0.253 / 0.353 | both pass |
| FIXED_ALL | validation | 0.525 / 0.237 | 0.313 / 0.303 | rho2->rho1 Mz2 fails |
| RESAMPLED_CRN | seen | 0.361 / 0.337 | 0.358 / 0.159 | both pass |
| RESAMPLED_CRN | unseen | 0.555 / 0.429 | 0.260 / 0.332 | both pass |
| RESAMPLED_CRN | validation | 0.557 / 0.283 | 0.300 / 0.308 | rho2->rho1 Mz2 fails |

Resampling worsened unseen rho1->rho0 MMD/physics and produced only a small mixed change for rho2->rho1. Validation changes were likewise mixed.

### RESULT: rollout

| condition | domain | first MMD / physics | final MMD / physics | final ordering |
|---|---|---|---|---|
| FIXED_ALL | seen | 0.451 / 0.199 | 0.906 / 0.561 | Mz2 fails |
| FIXED_ALL | unseen | 0.256 / 0.355 | 0.936 / 0.593 | Mz2 fails |
| FIXED_ALL | validation | 0.315 / 0.305 | 0.956 / 0.435 | both pass |
| RESAMPLED_CRN | seen | 0.372 / 0.155 | 0.809 / 0.480 | both pass |
| RESAMPLED_CRN | unseen | 0.264 / 0.339 | 0.912 / 0.585 | Mz2 fails |
| RESAMPLED_CRN | validation | 0.298 / 0.311 | 0.909 / 0.438 | Mz2 fails |

Resampling modestly improves final MMD (seen -0.097, unseen -0.024, validation -0.047), but unseen physics changes only -0.008 and ordering is not restored. Validation physics is unchanged and ordering worsens. Final unseen-minus-seen MMD gap grows from 0.030 to 0.102, while the physics gap grows from 0.032 to 0.105.

### INTERPRETATION

**Decision C: fixed finite-randomness overfitting is not the primary remaining cause.** Iteration-wise CRN resampling does not improve unseen isolated-step support learning and does not materially recover unseen rollout physics or ordering. It also raises SPSA estimator variance substantially. The small rollout MMD gain is insufficient because corresponding physics/order do not improve.

This is not evidence against CRN: the implementation correctly shares randomness within each finite difference. It shows that this particular bootstrap-resampled objective is noisier than the fixed objective under the frozen SPSA schedule. `INDEPENDENT_PM` was not needed to answer the primary comparison.

### NEXT ACTION

Keep T=2, L=3, N=4, and Phase 4 blocked. The next diagnostic should examine the ensemble objective formulation without adding stochastic estimator noise: compare the current per-realization fidelity-MMD averaging against one deterministic MMD computed over the combined generated ensemble and combined target ensemble. This tests whether the loss geometry, rather than finite randomness reuse, prevents support learning. Do not increase N, T, or capacity first.

## Ensemble objective geometry

### FACT: implemented objective semantics

Code inspection and permutation tests found that the previous deterministic objective was **not realization-paired**. The accurate baseline name is `PER_OUTCOME_ENSEMBLE_MMD`, not `PER_REALIZATION_MMD`.

For class c, realization i, and measurement outcome r, let `G_c,r={g_c,i,r}_{i=1..N}` and `T_c={t_c,i}_{i=1..N}`. The baseline is:

`L_A = (1/C) sum_c (1/R) sum_r MMD_biased(G_c,r, T_c)`.

The combined condition is:

`L_B = (1/C) sum_c MMD_biased(union_r G_c,r, union_r T_c)`,

where each target is repeated R times to preserve equal empirical weighting. Both use the existing fidelity kernel `k(psi,phi)=|<psi|phi>|^2`, the biased nonnegative MMD estimator, equal class weighting, N=4 complete realizations, and eight fixed measurement outcomes. Classes are never pooled together.

Therefore A already compares generated and target ensembles distributionally within each outcome; it does not enforce `G_i->T_i` identity. B additionally includes cross-measurement-outcome generated/generated kernel terms.

### RESULT: scale and permutation controls

Across initialization plus three fixed random parameter vectors, objective scales were close:

- rho1->rho0: A 1.307–1.369; B 1.281–1.359.
- rho2->rho1: A 0.637–0.714; B 0.617–0.684.

No optimizer normalization was applied. Permuting target realization order changed A by exactly 0 and B by at most `2.22e-16`. Thus the suspected realization-pair sensitivity is absent in both objectives. Tests also verify deterministic evaluation, full-ensemble use, permutation invariance, and equal class weighting.

Both conditions used 3,074 objective calls and 196,736 class/source/outcome evaluations. Combined MMD has 8x the kernel-pair cost because it operates on 32 samples rather than eight separate four-sample kernels, but runtime increased only from 386.9s to 401.5s.

### RESULT: optimization

| objective | step | initial | best | final | parameter update | directional variance |
|---|---:|---:|---:|---:|---:|---:|
| per-outcome | rho1->rho0 | 1.307 | 0.311 | 0.311 | 3.446 | 0.0616 |
| per-outcome | rho2->rho1 | 0.637 | 0.467 | 0.467 | 1.416 | 0.0105 |
| combined | rho1->rho0 | 1.281 | 0.326 | 0.327 | 3.691 | 0.0540 |
| combined | rho2->rho1 | 0.617 | 0.369 | 0.370 | 1.655 | 0.0146 |

Both deterministic objectives optimized normally. No scale failure or unexpected stochastic variance was observed.

### RESULT: isolated steps

| objective | domain | rho1->rho0 MMD / physics | rho2->rho1 MMD / physics | ordering |
|---|---|---|---|---|
| per-outcome | seen | 0.318 / 0.277 | 0.449 / 0.197 | both pass |
| per-outcome | unseen | 0.523 / 0.367 | 0.263 / 0.368 | rho2->rho1 Mx fails |
| per-outcome | validation | 0.523 / 0.201 | 0.309 / 0.312 | rho2->rho1 Mz2 fails |
| combined | seen | 0.369 / 0.326 | 0.401 / 0.165 | both pass |
| combined | unseen | 0.550 / 0.404 | 0.231 / 0.346 | both pass |
| combined | validation | 0.528 / 0.232 | 0.288 / 0.310 | rho2->rho1 Mz2 fails |

Combined pooling improves unseen rho2->rho1 MMD/physics and restores its ordering, but worsens unseen and validation rho1->rho0. It does not consistently improve single-step support learning.

### RESULT: rollout

| objective | domain | first MMD / physics | final MMD / physics | final ordering |
|---|---|---|---|---|
| per-outcome | seen | 0.447 / 0.203 | 0.866 / 0.520 | both pass |
| per-outcome | unseen | 0.264 / 0.365 | 0.943 / 0.589 | both pass |
| per-outcome | validation | 0.311 / 0.316 | 0.945 / 0.425 | both pass |
| combined | seen | 0.386 / 0.173 | 0.827 / 0.463 | both pass |
| combined | unseen | 0.233 / 0.349 | 0.900 / 0.534 | both pass |
| combined | validation | 0.286 / 0.311 | 0.922 / 0.435 | Mz2 fails |

Combined pooling modestly improves seen and unseen rollout MMD/physics, but validation physics is unchanged/slightly worse and its Mz2 ordering is lost. Final unseen-minus-seen gaps remain essentially unchanged: MMD 0.076->0.072 and physics 0.069->0.070. The gains are not caused solely by seen degradation, but they are not a support-level recovery.

### INTERPRETATION

**Decision C: realization-pair-specific loss geometry cannot explain the failure, and combined pooling does not consistently recover unseen single steps.** The original objective is already permutation-invariant over realization identity. Pooling outcomes helps rho2->rho1 and rollout modestly, while harming rho1->rho0 and validation physics/order. There is insufficient evidence for ensemble-level objective/physics mismatch because distribution and physics changes are mixed rather than strongly divergent.

### NEXT ACTION

Keep T=2, L=3, N=4, and Phase 4 blocked. The next targeted diagnostic should inspect gradient signal/landscape separately for each reverse step and class under the two existing losses: gradient magnitude, alignment across classes/realizations, and local loss changes along SPSA directions. This can distinguish a weak fidelity-kernel signal from conditioning/class-gradient conflict before any parameterization or physics-aware loss change.

## Reverse-step gradient-signal geometry

### FACT

Both deterministic objectives were minimally reproduced to recover `initial`, iteration-256 `intermediate`, and `best` checkpoints for each reverse step. At every checkpoint, 32 shared Rademacher SPSA directions were evaluated at `theta +/- 0.15*delta` with identical fixed source/outcome randomness. The same directions were used across steps, classes, realizations, checkpoints, and objectives.

Directional signal is `(L+ - L-)/(2 epsilon)`. The SNR proxy is mean absolute aggregate directional derivative divided by the mean standard deviation of the eight per-outcome component derivatives. Class and realization conflicts use directional signs from identical perturbations. This is a relative diagnostic, not a pass threshold. A bounded component-wise finite-difference control was skipped because the full shared-direction decomposition already required 16 minutes and directly matches the production SPSA geometry.

### RESULT: aggregate signal and curvature

| trained/evaluated objective | step | checkpoint | mean abs derivative | SNR proxy | abs curvature | near-zero | descent fraction |
|---|---|---|---:|---:|---:|---:|---:|
| per-outcome | rho1->rho0 | init | 0.205 | 0.912 | 0.061 | 0% | 56% |
| per-outcome | rho1->rho0 | best | 0.124 | 0.618 | 0.631 | 0% | 0% |
| per-outcome | rho2->rho1 | init | 0.103 | 1.546 | 0.034 | 0% | 47% |
| per-outcome | rho2->rho1 | best | 0.074 | 0.697 | 0.137 | 0% | 0% |
| combined | rho1->rho0 | init | 0.250 | 1.114 | 0.056 | 0% | 53% |
| combined | rho1->rho0 | best | 0.121 | 0.607 | 0.524 | 0% | 0% |
| combined | rho2->rho1 | init | 0.130 | 1.948 | 0.019 | 0% | 88% |
| combined | rho2->rho1 | best | 0.076 | 0.481 | 0.140 | 0% | 0% |

No checkpoint is directionally barren: no sampled direction is near zero, and initialization has many descent directions. At trained checkpoints, finite epsilon moves in either sign are uphill despite nonzero central derivatives, reflected in strong positive curvature and 0% finite-step descent. Signal weakens during fitting and outcome-dispersion SNR falls below one, but the landscape is not flat.

### RESULT: class alignment

| objective | step | checkpoint | class correlation | sign agreement | conflict rate | class0/class1 mean abs signal |
|---|---|---|---:|---:|---:|---:|
| per-outcome | rho1->rho0 | init | -0.095 | 0.469 | 0.531 | 0.262 / 0.341 |
| per-outcome | rho1->rho0 | best | -0.338 | 0.438 | 0.562 | 0.227 / 0.204 |
| per-outcome | rho2->rho1 | init | -0.095 | 0.438 | 0.562 | 0.147 / 0.148 |
| per-outcome | rho2->rho1 | best | **-0.552** | 0.281 | **0.719** | 0.175 / 0.116 |
| combined | rho1->rho0 | init | 0.007 | 0.469 | 0.531 | 0.305 / 0.400 |
| combined | rho1->rho0 | best | -0.260 | 0.531 | 0.469 | 0.183 / 0.179 |
| combined | rho2->rho1 | init | 0.067 | 0.406 | 0.594 | 0.163 / 0.187 |
| combined | rho2->rho1 | best | -0.328 | 0.375 | 0.625 | 0.155 / 0.109 |

The clearest conflict is trained rho2->rho1 under the existing loss: 71.9% of directions improve one class while worsening the other, with correlation -0.552. Combined MMD reduces this to 62.5% and supplies a stronger initial signal, consistent with its modest rho2->rho1 improvement.

### RESULT: realization alignment

Mean off-diagonal realization correlations/conflict rates at best checkpoints:

| objective | step | class 0 corr/conflict | class 1 corr/conflict | all class-realization corr/conflict |
|---|---|---|---|---|
| per-outcome | rho1->rho0 | -0.107 / 0.516 | -0.228 / 0.609 | -0.095 / 0.531 |
| per-outcome | rho2->rho1 | -0.069 / 0.490 | -0.169 / 0.542 | -0.043 / 0.491 |
| combined | rho1->rho0 | -0.094 / 0.568 | -0.023 / 0.516 | -0.011 / 0.502 |
| combined | rho2->rho1 | 0.067 / 0.536 | 0.117 / 0.448 | 0.100 / 0.485 |

Individual realization signals are nonzero but mostly uncorrelated or weakly anti-correlated. Roughly half of pairwise directions conflict. For rho1->rho0, combined pooling reduces class conflict but leaves realization conflict near 50–57%, explaining why its distributional pooling did not recover that step.

### INTERPRETATION

The strongest supported mechanisms are **B: class-conditioning gradient conflict** for rho2->rho1 and **C: forward-realization gradient conflict** for both steps, especially the persistent rho1->rho0 bottleneck. There is evidence against a globally barren landscape: directional magnitudes are nonzero, initialization has usable descent directions, and trained points show curvature rather than flatness. SPSA SNR becomes weak near fitted checkpoints, but this is consistent with cancellation/heterogeneity and proximity to finite-step local minima; it is not sufficient to classify optimizer noise as primary.

Combined MMD helps rho2->rho1 because it starts with a stronger aggregate signal and ends with less class anti-alignment than the existing loss. It does not fix rho1->rho0 because realization-level directions remain conflicting even when class alignment improves.

### NEXT ACTION

Keep T=2, L=3, N=4, and Phase 4 blocked. The next diagnostic should separate class-sharing from realization conflict without changing capacity: train bounded single-class reverse controls for each step using the same N=4 ensembles, then compare their unseen support gradients/performance with the shared conditional model. If single-class controls improve, conditioning/shared-parameter conflict is causal; if rho1->rho0 still fails, realization conflict or fidelity-kernel signal is primary. Do not add a new loss or architecture yet.

## Targeted remediation controls

### FACT: single-class causal controls

Four bounded controls trained class 0 and class 1 separately for both frozen reverse steps (`T=2`, `L=3`, `N=4`, SPSA-512, eight fixed outcomes). Removing the other class also removes label-angle differentiation; these runs test whether shared conditional parameters are the causal bottleneck, not a production architecture.

| class | domain | rho1->rho0 MMD / physics | rho2->rho1 MMD / physics |
|---:|---|---|---|
| 0 | seen | 0.121 / 0.135 | 0.295 / 0.209 |
| 0 | unseen | 0.450 / 0.286 | 0.234 / 0.242 |
| 0 | validation | 0.488 / 0.357 | 0.277 / 0.340 |
| 1 | seen | 0.315 / 0.229 | 0.163 / 0.071 |
| 1 | unseen | 0.627 / 0.340 | 0.255 / 0.333 |
| 1 | validation | 0.659 / 0.321 | 0.325 / 0.286 |

Compared with the shared model, single-class rho2->rho1 improves class-0 unseen MMD from 0.281 to 0.234 and validation MMD from 0.297 to 0.277, but class 1 is unchanged/slightly worse (unseen 0.245->0.255; validation 0.322->0.325). This is not a substantial two-class causal improvement, so a class-specific adapter was not justified.

For rho1->rho0, seen fitting improves but support generalization does not: class-0 validation MMD worsens 0.428->0.488 and class-1 unseen/validation MMD worsens 0.566->0.627 and 0.619->0.659. Single-class realization conflicts remain high: 53.6–58.3% for rho1->rho0 and 54.7–59.4% for rho2->rho1, with negative mean off-diagonal correlations. Removing class competition does not remove realization conflict.

### RESULT: conflict-aware realization aggregation

Because rho1->rho0 remained realization-limited, one bounded directional aggregation control was tested. For each SPSA direction and class:

`d_c = median_i d_(c,i)` and `d = mean_c d_c`, followed by update `theta <- theta - rate*d*Delta`.

This preserves equal class weighting while preventing a minority of realization signs from cancelling the class median. The standard rho2->rho1 parameters were retained; only rho1->rho0 training changed.

| method | domain | isolated rho1->rho0 MMD / physics | final rollout MMD / physics | final ordering |
|---|---|---|---|---|
| mean | seen | 0.318 / 0.277 | 0.866 / 0.520 | both pass |
| median | seen | 0.314 / 0.286 | 0.913 / 0.513 | both pass |
| mean | unseen | 0.523 / 0.367 | 0.943 / 0.589 | both pass |
| median | unseen | 0.563 / 0.425 | 0.943 / 0.621 | both fail |
| mean | validation | 0.523 / 0.201 | 0.945 / 0.425 | both pass |
| median | validation | 0.585 / 0.280 | 0.944 / 0.470 | Mz2 fails |

The median rule lowers its fixed standard training objective to 0.304, but worsens unseen and validation MMD/physics and loses ordering. It therefore rejects this simple conflict-aware remedy. Lower train objective without external quality improvement is not a generation-quality success.

### INTERPRETATION

Phase A does not establish class conflict as sufficiently causal to justify B1 adapter parameters: only class 0 rho2->rho1 improves modestly, while class 1 does not. Phase B2 confirms that realization conflict exists but shows that majority/median directional aggregation is not an effective remedy. The failure is now narrowed to a more fundamental interaction between per-realization fidelity-kernel objectives and shared reverse-map support learning; conflicts cannot be fixed by simply removing a class or replacing the mean with a median.

No candidate qualifies for chain reassembly, multi-seed replication, or Haar generation. The Conditional QuDDPM generation-quality gate remains failed: unseen/validation MMD and quantitative TFIM observable errors remain poor despite occasional correct ordering. Phase 4 remains blocked.

### NEXT ACTION

Do not add adapters, increase depth, or run augmentation. The next experiment should be a bounded realization-aware kernel diagnostic for rho1->rho0: compare global fidelity MMD with a local/factorized fidelity kernel or another distribution metric on the same fixed generated/target ensembles, first measuring whether it produces better-aligned realization gradients. Only train a replacement objective if its gradient-alignment diagnostic improves before optimization.

## Artifacts

- `results/quddpm_q2_single_class/metrics.json`: single-class histories, metrics, and realization matrices.
- `results/quddpm_q2_single_class/single_class.csv`: compact four-control table.
- `results/quddpm_q2_conflict_aware/metrics.json`: standard/hybrid histories and evaluations.
- `results/quddpm_q2_conflict_aware/comparison.csv`: mean-vs-median comparison.
- `results/quddpm_q2_gradient_signal/metrics.json`: checkpoint, class, realization, and 8x8 alignment data.
- `results/quddpm_q2_gradient_signal/signal_summary.csv`: aggregate signal/SNR/curvature table.
- `results/quddpm_q2_gradient_signal/directions.csv`: all shared-direction derivatives.
- `results/quddpm_q2_objective_geometry/metrics.json`: probes, permutation checks, histories, evaluations, and gaps.
- `results/quddpm_q2_objective_geometry/comparison.csv`: objective/domain/step comparison.
- `results/quddpm_q2_sampling_semantics/metrics.json`: matched histories, batch fingerprints, variance diagnostics, and evaluations.
- `results/quddpm_q2_sampling_semantics/comparison.csv`: compact condition/domain/step table.
- `results/quddpm_q2_ensemble/metrics.json`: complete histories, realization IDs, per-class nearest-fidelity metrics.
- `results/quddpm_q2_ensemble/generalization.csv`: isolated/rollout domain table.
- `results/quddpm_q2_ensemble/generalization_gap.svg`: unseen-minus-seen final MMD gap.
- `results/quddpm_q2_ensemble/unseen_final_physics.svg`: unseen final physics error.
- `results/quddpm_terminal_prior/metrics.json`: full terminal, mixture, and coverage-control metrics.
- `results/quddpm_terminal_prior/terminal_distance.csv`: qT-to-Haar and TFIM-gap table.
- `results/quddpm_terminal_prior/mixture_curve.csv`: baseline alpha curve.
- `results/quddpm_terminal_prior/coverage_curve.csv`: coverage-control alpha curve.
- `results/quddpm_terminal_prior/alpha_final_mmd.svg`: alpha/MMD plot.
- `results/quddpm_terminal_prior/alpha_final_physics_error.svg`: alpha/physics plot.
- `results/quddpm_underfitting_diagnostics/metrics.json`: initial T=1 progression.
- `results/quddpm_underfitting_diagnostics/summary.csv`: initial campaign table.
- `results/quddpm_t1_multiseed_t2/metrics.json`: multi-seed gate and T=2 per-step/rollout results.
- `results/quddpm_t1_multiseed_t2/config.yaml`: resolved follow-up configuration.

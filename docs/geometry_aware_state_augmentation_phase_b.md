# Phase B — Geometry-aware quantum-state augmentation

## 1. Repository state and interpretation boundary

Phase B started on branch `research/tfim-state-augmentation` at
`c99bf84653db593d50f85f2fac2350e5295bc8e2`, with a clean tree and `64 passed`.
The implementation commit is `4e06c81`; configs and machine-readable screening
results are commit `4b1b0da`. The final documentation commit is the repository
HEAD containing this file.

> **Phase B는 method screening 및 failure-mode analysis를 위한 exploratory
> experiment이며, publication-level confirmatory evidence로 해석하지 않는다.**

The random and blocked-g test sets were already observed in Phase A. A GO or
CONDITIONAL GO would therefore require a new holdout/blocked split or expanded
paired seeds under a locked protocol. No validation/test state, observable,
distance, or metric was used to construct, filter, order, or tune candidates.
Labels are inherited under **same-class local geodesic interpolation**; the
method does not guarantee that a synthetic state is an exact TFIM ground state
or mathematically remains in the same phase.

## 2. Phase A budget audit

The audit result is **A (budget-pure)**. In
`physics_aware_augmentation.run_diagnostics`, the canonical
`nested_train_subsets(..., [10], 31415)` subset supplies all sources. Per-class
`Mx`, `Mz2`, and source-conditioned energy gates are fit on the same ten states,
and the accepted pool is generated only from those sources. Validation/test are
used only by the frozen QCNN evaluation. The sample IDs in every historical
Phase A QCNN row exactly match the Phase B IDs.

Consequently no Phase A rerun was required. The fair comparator is the
historical, already budget-matched artifact
`results/physics_aware_augmentation/phase_a/qcnn_pilot.json`; historical Phase A
artifacts and config were not changed.

## 3. Fixed geometry method

For normalized states,

\[
d_{FS}(\psi_a,\psi_b)=\arccos(\operatorname{clip}(|\langle\psi_a|\psi_b\rangle|,0,1)).
\]

For overlap `c`, `psi_b` is aligned by `exp(-i arg(c))` when `|c| > 1e-12`;
otherwise the pair is rejected. Within each class and run, only the canonical
10-state budget subset is used. Every state selects its three nearest
same-class neighbors, ties are broken by lexical sample ID, and the undirected
union graph is canonicalized by ordered pair ID. Its eligible edges satisfy
`0.04 <= d_FS <= class-budget q75` and overlap above `1e-12`.

For each eligible pair,

\[
u=\frac{\psi_b^{aligned}-\cos(\theta)\psi_a}{\sin(\theta)},\qquad
\psi_t=\cos(t\theta)\psi_a+\sin(t\theta)u,
\]

at fixed `t={0.25,0.50,0.75}`. Candidates are normalized once after recording
the pre-normalization error. Hard acceptance uses only pair eligibility,
finite/norm checks, fixed `t`, and projective duplicate rejection at infidelity
`1e-10`. `Mx`, `Mz2`, energy, and class margins are post-hoc diagnostics only.

Within each `t` stratum a SHA-256 key covers run seed, class, canonical source
pair, `t`, and the fixed namespace. Strata are round-robin interleaved. Ratio
pools are no-replacement prefixes: 5, 10, and 20 states/class. State hashes use
a global-phase canonical representative. Thresholds, `k`, `t`, ratios, and
ordering were not changed after observing diagnostics or QCNN results.

## 4. Geometry generator diagnostics

Each class has 45 unordered budget-pair distances. Counts and state statistics
are invariant across run seeds (the stable sequence order changes by seed).
`Div` is `1 - mean pairwise fidelity`; `SrcDiv` is mean anchor-conditioned
diversity; `Inf` is nearest-source infidelity. Drift columns are mean absolute
post-hoc drift relative to the nearest generating endpoint Hamiltonian.
Exact distributions and per-`t` cells are in `generator_diagnostics.json`.

| Dataset | Seeds | Class | kNN edges | Eligible pairs | Unique candidates | Coverage/10 | Pair mean/median | Inf mean/median | Div | SrcDiv | median margin excl. sources | mean abs dMx / dMz2 / dE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0,1,2 | 0 | 19 | 11 | 33 | 8 | .135785/.136142 | .001780/.001061 | .055977 | .004198 | .480978 | .044040/.027779/.006654 |
| random | 0,1,2 | 1 | 21 | 2 | 6 | 3 | .046063/.046063 | .000119/.000125 | .000212 | .000111 | .280252 | .008120/.011307/.001448 |
| blocked-g | 0,1,2 | 0 | 17 | 10 | 30 | 9 | .066594/.065327 | .000304/.000216 | .013945 | .000721 | .508397 | .022395/.012089/.001153 |
| blocked-g | 0,1,2 | 1 | 21 | **0** | **0** | 0 | — | — | — | — | — | — |

There were no finite-value, normalization, or accepted-duplicate failures.
However, blocked-g class 1 has budget q75 `0.035452 < 0.04`, making every graph
edge ineligible. Random class 1 has only two eligible pairs and six candidates.
Thus random `r=0.5` alone is feasible across both classes; random `r=1,2` and
all nonzero blocked-g ratios are infeasible. The required primary blocked-g
`r=1` pool cannot be formed, so its near-copy median is **not evaluable**. This
shortage was reported rather than repaired by threshold relaxation or
replacement. The feasible random `r=0.5` geometry median nearest-source
infidelity is `1.79e-4` to `2.21e-4` across seeds, above the `1e-4` screen.

## 5. Matched random-direction control

For every available geometry candidate, the nearest endpoint is its anchor
(lexical ID breaks a tie), and `delta` is the geometry candidate's FS distance
to that anchor. A provenance-seeded complex Gaussian vector is projected into
the anchor tangent space and normalized; the control is

\[
\psi_c=\cos(\delta)\psi+\sin(\delta)\eta_\perp.
\]

This is a **manifold-unaware, distance-matched random-direction control**, not
an invalid or non-geometric quantum state. One-to-one candidate correspondence,
class/count, anchor frequencies, run seed, and nearest-anchor displacement are
identical by construction. Deterministic redraws (maximum 100) handle a failed
projection or duplicate; none was needed beyond the recorded counters.

| Dataset | Seed | Class | Controls | Mean / max displacement error (rad) | Div | SrcDiv | median margin excl. sources |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0 | 0 | 33 | 6.79e-15 / 3.72e-14 | .081121 | .004961 | .399235 |
| random | 0 | 1 | 6 | 1.45e-14 / 4.85e-14 | .001825 | .000622 | .276164 |
| random | 1 | 0 | 33 | 6.98e-15 / 4.54e-14 | .077921 | .005372 | .417176 |
| random | 1 | 1 | 6 | 1.45e-14 / 3.88e-14 | .001709 | .000574 | .279980 |
| random | 2 | 0 | 33 | 5.76e-15 / 3.72e-14 | .079069 | .005324 | .425680 |
| random | 2 | 1 | 6 | 9.66e-15 / 1.94e-14 | .001659 | .000639 | .278067 |
| blocked-g | 0 | 0 | 30 | 8.27e-15 / 2.48e-14 | .017939 | .001177 | .507506 |
| blocked-g | 0 | 1 | 0 | — | — | — | — |
| blocked-g | 1 | 0 | 30 | 7.89e-15 / 3.06e-14 | .017891 | .001259 | .505913 |
| blocked-g | 1 | 1 | 0 | — | — | — | — |
| blocked-g | 2 | 0 | 30 | 7.86e-15 / 3.06e-14 | .018412 | .001280 | .502265 |
| blocked-g | 2 | 1 | 0 | — | — | — | — |

Across all 207 matched candidates, mean/max displacement error is
`7.72e-15 / 4.85e-14` radians. Sample counts and anchor-source counters match
exactly. Control `Mx`, `Mz2`, energy and complete class-margin distributions are
post-hoc fields in `candidate_provenance.jsonl` and
`matched_control_diagnostics.json`; none was an acceptance gate.

## 6. Frozen QCNN benchmark

The repository protocol remains a 4-qubit input, 4→2→1 QCNN, 42 parameters,
`Z` readout on qubit 3, and mean-squared loss between `<Z>` and labels mapped to
`{-1,+1}`. Full-batch SPSA uses 300 steps, learning rate `0.15`, perturbation
`0.1`, and the existing validation-loss best-checkpoint/early-stopping rule.
Seeds are `(run,init,SPSA)=(0,1000,2000),(1,1001,2001),(2,1002,2002)`.
Real sample IDs are identical across paired methods.

Only random `r=0.5` could be trained for geometry/control. Infeasible cells are
shown as `—`; no replacement, threshold tuning, or test-driven ratio selection
was used.

| Dataset | Ratio | Real-only | Matched control | Physics-aware | Geometry-aware |
|---|---:|---:|---:|---:|---:|
| random | 0 | .9222 | .9222 | .9222 | .9222 |
| random | .5 | .9222 | .9778 | .9222 | .9778 |
| random | 1 | .9222 | — | .9500 | — |
| random | 2 | .9222 | — | .9167 | — |
| blocked-g | 0 | .9000 | .9000 | .9000 | .9000 |
| blocked-g | .5 | .9000 | — | .6889 | — |
| blocked-g | 1 | .9000 | — | .6833 | — |
| blocked-g | 2 | .9000 | — | .6833 | — |

### Per-seed test accuracy

| Dataset | Ratio | Seed | Real | Control | Physics | Geometry |
|---|---:|---:|---:|---:|---:|---:|
| random | 0 | 0 | .9333 | .9333 | .9333 | .9333 |
| random | 0 | 1 | .9167 | .9167 | .9167 | .9167 |
| random | 0 | 2 | .9167 | .9167 | .9167 | .9167 |
| random | .5 | 0 | .9333 | 1.0000 | .9167 | 1.0000 |
| random | .5 | 1 | .9167 | 1.0000 | .9167 | 1.0000 |
| random | .5 | 2 | .9167 | .9333 | .9333 | .9333 |
| random | 1 | 0 | .9333 | — | .9667 | — |
| random | 1 | 1 | .9167 | — | .9167 | — |
| random | 1 | 2 | .9167 | — | .9667 | — |
| random | 2 | 0 | .9333 | — | .9000 | — |
| random | 2 | 1 | .9167 | — | .9167 | — |
| random | 2 | 2 | .9167 | — | .9333 | — |
| blocked-g | 0 | 0 | .8667 | .8667 | .8667 | .8667 |
| blocked-g | 0 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| blocked-g | 0 | 2 | .8333 | .8333 | .8333 | .8333 |
| blocked-g | .5 | 0 | .8667 | — | .8167 | — |
| blocked-g | .5 | 1 | 1.0000 | — | .5000 | — |
| blocked-g | .5 | 2 | .8333 | — | .7500 | — |
| blocked-g | 1 | 0 | .8667 | — | 1.0000 | — |
| blocked-g | 1 | 1 | 1.0000 | — | .5500 | — |
| blocked-g | 1 | 2 | .8333 | — | .5000 | — |
| blocked-g | 2 | 0 | .8667 | — | 1.0000 | — |
| blocked-g | 2 | 1 | 1.0000 | — | .5500 | — |
| blocked-g | 2 | 2 | .8333 | — | .5000 | — |

Primary blocked-g `r=1` paired deltas (`geometry-real`, `geometry-control`, and
`geometry-physics`) are all **not estimable**, because geometry/control have no
valid primary pool. They are stored as empty arrays rather than fabricated
zeros. At feasible random `r=.5`, geometry and control tie seed-by-seed, so the
observed random-split gain does not identify the same-class empirical direction
as the useful mechanism.

## 7. Train/validation/test generalization

| Dataset / method / ratio | Train real | Validation | Test | Train-test gap | Validation-test gap |
|---|---:|---:|---:|---:|---:|
| random real-only | .9500 | .9389 | .9222 | .0278 | .0167 |
| random control .5 | .9833 | .9778 | .9778 | .0056 | .0000 |
| random geometry .5 | .9667 | .9722 | .9778 | -.0111 | -.0056 |
| random physics .5 | .9500 | .9444 | .9222 | .0278 | .0222 |
| blocked-g real-only | 1.0000 | 1.0000 | .9000 | .1000 | .1000 |
| blocked-g physics 1 | 1.0000 | 1.0000 | .6833 | .3167 | .3167 |

The feasible random result improves test performance as well as fit, but both
geometry and random directions obtain exactly the same accuracies. For blocked
g, Phase B utility is not executable. Historical physics augmentation preserves
saturated train/validation accuracy while widening both blocked-g gaps; it
raises fit without parameter-shift generalization. Validation saturation cannot
select a blocked-g ratio, and test performance was not used for selection.

## 8. Explicit answers

1. **Near-copy dominated?** The full primary `r=1` pool is infeasible, so its
   preregistered median is not evaluable. Feasible random `r=.5` is above the
   `1e-4` cutoff, while class-1 diversity remains very small.
2. **Nontrivial local-geodesic states?** Yes in available cells: 33/6 random and
   30/0 blocked-g candidates by class, with nonzero FS movement and diversity.
3. **FS displacement matched?** Yes for every available one-to-one control;
   mean/max errors are `7.72e-15/4.85e-14` radians.
4. **Better than random tangent direction?** No evidence. They tie on every
   feasible random `r=.5` seed; primary blocked-g is infeasible.
5. **Random split improved?** At feasible `r=.5`, both geometry and control rise
   from `.9222` to `.9778`; this is secondary exploratory evidence only.
6. **Blocked-g improved?** Not testable: class-1 pair eligibility yields no pool.
7. **Primary `r=1` seed-consistent?** Not estimable, not seed-consistent evidence.
8. **Fit only or test generalization?** Random `.5` improves test too, but not
   specifically from geometry. Blocked physics improves/saturates fit while
   harming test; Phase B blocked utility cannot be run.
9. **Which prior is more useful than budget-matched physics?** Geometry/control
   outperform physics only in feasible random `.5`, but tie each other. No
   structural-prior ranking is supported under blocked parameter shift.
10. **Basis for Physics+Geometry Phase C?** No. Combining two methods is not
    automatically justified by a failed primary generator/utility gate.
11. **Need new holdout/expanded-seed confirmation?** Any future positive claim
    would need it, but this NO-GO does not currently justify that confirmatory
    expense without first redesigning the preregistered method.

## 9. Decision

# NO-GO

The primary generator validity gate fails: ten unique geometry candidates and
ten matched controls per dataset/seed/class are unavailable. In particular,
blocked-g class 1 has zero eligible pairs and random class 1 has only six
candidates. Therefore the predefined GO and CONDITIONAL GO utility comparisons
cannot be met, regardless of the secondary random-split result. Thresholds were
not relaxed after this observation.

The conclusion is deliberately limited:

> 현재 4-qubit TFIM, 10 states/class, fixed QCNN 및 사전 정의된
> local-geodesic augmentation 설정에서 downstream blocked-g utility를 확보하지
> 못했다.

This is not evidence that geometry-aware augmentation in general is impossible.

## 10. Next step and artifact schema

The supported next step is failure-mode analysis of state-local augmentation
under distribution shift, not automatic Phase C. Simulator resampling should
remain a distinct reference baseline, and learned augmentation a separate
research track. A future redesign must be preregistered before a new holdout.

Machine-readable artifacts are under
`results/geometry_aware_augmentation/phase_b/`. JSON/JSONL files contain all
reported tables, per-candidate provenance, selected nested prefixes, complete
QCNN best/final metrics, and run manifests. `synthetic_states.npz` maps each
geometry/control candidate ID to one `complex128[16]` state;
`qcnn_parameters.npz` maps `dataset|method|ratio|seed|{best,final}` to
`float64[42]`. `artifact_hashes.sha256` verifies every result file. Historical
Phase A final-test metrics are explicitly `null` because that immutable artifact
retained only the best-checkpoint parameters; its final train/validation history
is preserved without inventing a final test value.

# TFIM quantum-state augmentation — final archival checkpoint

## Research question

Under limited quantum training data, can quantum-state-level augmentation improve downstream QCNN generalization? This archive closes the repository's current 4-qubit TFIM track; it does not close the broader research question.

## Final scientific conclusion

> Frozen 4-qubit TFIM + QCNN + q50 local-random-tangent + synthetic ratio 1.0 + Protocol v2.3 조건에서 augmentation은 blocked-g generalization improvement criterion을 충족하지 못했다.

Execution integrity is **PASS** (48/48 scientific runs completed), while the scientific hypothesis verdict is **FAIL**. The blocked-g augmentation-minus-real-only mean paired delta is `-0.018055555555555547` (about **-1.81 percentage points**) and the frozen paired-bootstrap 95% CI is `[-0.0486111111111111, 0.0013888888888889024]`. The interval includes zero: there is no evidence of improvement, but this is not evidence that augmentation is generally harmful.

The authoritative source is `results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3_freeze/confirmatory_analysis.json` at result-freeze commit `1e3ac703ff34444de473b6ecf2e5b8641c73213b` and freeze hash `a641e912d8de85e0a9d5b03a3c6d4262f214275fbc92c453ef7a5d593e9585b4`.

## Evidence-grounded chronology

| Stage | Repository evidence | Commit evidence | Recorded outcome |
|---|---|---|---|
| 4-qubit TFIM data and QCNN baseline | `data/tfim_4q_random/`, `data/tfim_4q_blocked/`, `docs/qcnn_baseline.md` | `752afc8`, `1b96a21`, `6861c39`, `84f4194` | Random and blocked-g datasets and the real-only development benchmark were established. These inspected development results are not the later confirmatory corpus. |
| Conditional QuDDPM K0–K3 | `docs/rdm_kernel_diagnostics.md`, `docs/k2_realization_conflict.md`, `docs/k3_conflict_reweighting.md`, `results/quddpm_kernel_diagnostics/` | `7aef450`, `4435df2` through `a592fb4`, frozen at `5aec728` | No augmentation-ready conditional generator passed the frozen 4Q TFIM gates; the bounded QuDDPM path was declared NO-GO. |
| Physics-aware augmentation (Phase A) | `docs/physics_aware_state_augmentation_phase_a.md`, `results/physics_aware_augmentation/phase_a/` | `e97b27c`, conclusion `c99bf84` | Physically motivated, normalized perturbations were feasible, but the exploratory blocked-g downstream utility pattern failed. |
| Geometry-aware augmentation (Phase B) | `docs/geometry_aware_state_augmentation_phase_b.md`, `results/geometry_aware_augmentation/phase_b/` | `4e06c81`, `4b1b0da`, conclusion `2d14c75` | The primary blocked-g pool was infeasible; the feasible random result tied its distance-matched random control. Bounded NO-GO only. |
| Local random-tangent Phase C | `results/local_perturbation_augmentation/phase_c/` | `31660ee`, provenance fail-closed fix `c86b3eb` | A same-class local random-tangent exploratory baseline and its provenance artifacts were recorded. |
| Projective near-duplicate finding | `results/tfim_manifold_augmentation/confirmatory_dataset_freeze_evidence_recovery_v1/report.md`, `freshness_projective_pairs.jsonl` | archived in `51c3a53` | Distinct sample IDs, parameters, and canonical hashes still contained cross-split projective near-neighbors; v1 remained blocked and immutable. |
| FS independence/separation and constrained split | `results/tfim_manifold_augmentation/confirmatory_protocol_v2_1/fs_calibration.json`, `split_audit.json`, `protocol_manifest.json` | archived in `51c3a53` | Protocol v2.1 separated exact freshness from projective proximity, calibrated `epsilon_sep`, and introduced an FS-constrained random split. |
| Protocol v2 → v2.1 → v2.2 | corresponding `confirmatory_protocol_v2*` directories | archived in `51c3a53` | v2 froze the statistical contract; v2.1 added FS separation/split semantics; v2.2 froze fresh generation RNG and provenance. |
| Frozen confirmatory dataset | `confirmatory_dataset_freeze_v2_2/` | `51c3a53` | Random and blocked-g corpora passed the frozen dataset gate. |
| Protocol v2.3 | `confirmatory_protocol_v2_3/` | `526204e` | q50, ratio 1.0, 48-run matrix, paired seeds, final step-300 checkpoint, and 100,000-draw primary analysis were frozen before QCNN execution. |
| Confirmatory QCNN execution | `confirmatory_qcnn_v2_3/` | runner `d2dae79`; loader fixes `a8652b8`, `605e6f1`; results `1e3ac70` | 48/48 runs completed with exactly 300 SPSA updates. The archived loader failure occurred before training (`training_updates=0`) and is not a scientific failure. |
| Confirmatory and post-hoc analysis | `confirmatory_qcnn_v2_3_freeze/` | `1e3ac70` | Execution integrity PASS; confirmatory verdict FAIL; separately labeled exploratory analysis complete. |

No chronology entry above is inferred from an absent experiment. Phase-level records predating Protocol v2.3 are exploratory or methodological evidence unless explicitly labeled otherwise in their source artifact.

## Protocol evolution and split integrity

Protocol v2 froze a decision rule that required complete expected runs, blocked-g mean delta at least `0.02`, and a 95% CI lower bound above zero. Evidence recovery then showed that sample/parameter identity checks alone did not guarantee quantum-state independence: genuinely distinct samples could be extremely close as projective Hilbert-space rays. Protocol v2.1 therefore treated exact freshness and projective separation as different contracts, calibrated an FS distance threshold independently of QCNN results, and constrained the random split. Protocol v2.2 froze fresh corpus generation streams and provenance. Protocol v2.3 froze the executable 48-run QCNN matrix and statistics.

This history supports three methodological findings:

1. Parameter IDs and sample IDs may be independent while projective quantum states are not sufficiently separated for the intended split.
2. Fubini–Study geometry can provide an explicit split-separation criterion; here it prevented a hidden proximity condition from being treated as ordinary freshness.
3. Exploratory improvement and confirmatory generalization are different claims and require different evidence controls.

A normalized, physically valid pure state is not automatically a label-preserving augmentation. Phase A and Phase B diagnostics show validity and geometry can be established without thereby establishing downstream blocked-g utility. The archive does **not** claim that near-duplicate contamination caused earlier exploratory improvement; it is only a possible interpretation not identified causally by the available comparisons.

## Confirmatory design

The frozen setting used 4-qubit open-chain TFIM states, a fixed QCNN, random and blocked-g regimes, per-class budgets `{10,25,50,100}`, three repeats, paired real-only/local-random-tangent arms, q50 local radius, synthetic ratio 1.0, identical paired subset/initialization/SPSA seeds, exactly 300 SPSA updates, and final step-300 evaluation. The primary metric was test accuracy; the primary estimand was augmentation minus real-only in blocked-g; the paired bootstrap used 100,000 frozen draws. Random was supportive/exploratory only.

## Exploratory / post-hoc findings

These values come only from the frozen post-hoc artifact and do not alter the confirmatory verdict:

- random mean paired delta: `0.0`;
- blocked-g budget means: b10 `0.0`, b25 `-0.0555556`, b50 `-0.0111111`, b100 `-0.0055556`;
- blocked-g repeat means: r0 `-0.0083333`, r1 `-0.0375`, r2 `-0.0083333`;
- every leave-one-repeat-out blocked-g descriptive mean remained negative;
- all 48 runs completed 300 SPSA updates;
- no numerical instability was recorded.

No budget-level significance test, budget verdict, outlier removal, or new confirmatory claim is made.

## Interpretation and NO-GO scope

`research_track = ARCHIVED_NO_GO` means only that this repository's present chain of frozen 4Q TFIM augmentation approaches did not justify continued execution under the completed protocol. It combines two bounded outcomes without enlarging either:

- Conditional QuDDPM K0–K3 did not produce an augmentation-ready generator in its frozen diagnostic setting.
- Protocol v2.3 local-random-tangent augmentation did not meet the blocked-g improvement criterion in the exact confirmatory setting stated above.

It does **not** establish that quantum-state augmentation, geometric augmentation, local tangent augmentation generally, or augmentation for another Hamiltonian, qubit count, or classifier fails.

## Limitations and future hypotheses

The following are future hypotheses, not conclusions from the current result:

- geometrically local states may fail to be physically label-preserving;
- physics-aware and geometry-aware constraints might be useful in combination;
- physics-constrained geometric augmentation may behave differently;
- symmetry-preserving or Hamiltonian/manifold-aware perturbations may improve label fidelity;
- a learned quantum-state generator may behave differently from local perturbation;
- larger systems, other Hamiltonians, and other QML classifiers require independent validation.

Primary follow-up question: **physical manifold를 따라 perturbation해야 downstream utility가 생길 수 있는가?**

No such follow-up experiment was run for this archive.

## Immutable provenance

- branch: `research/tfim-local-perturbation`
- Protocol v2.3 hash: `bb33305486af6c998e377dd93fc74932dc3ad87bbc986dce049b792e58d72c92`
- frozen blocked-g scientific dataset hash: `cb0446120e7df9b0b6052f4575f6a1ff10742d8aaa49542054530ec6215e8867`
- frozen random scientific dataset hash: `09fd5792318cd171a3c39316adff1dbce6c402c9bc0fa66bd2df639fab73cff7`
- frozen blocked-g `states.npz` file SHA-256: `593733297e3e952ecf0cddb802b89379018c5b3c5225ee098fb2cd4237b36742`
- frozen random `states.npz` file SHA-256: `ff2f5fc4de67fe7e22e50d75617be469799964dc52282853ee4ca612dde630ff`
- execution source: `605e6f1a6f2f8c7a579895505a606bebe18110a4`
- result freeze commit: `1e3ac703ff34444de473b6ecf2e5b8641c73213b`
- result freeze hash: `a641e912d8de85e0a9d5b03a3c6d4262f214275fbc92c453ef7a5d593e9585b4`
- 48 result hashes: `results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3_freeze/freeze_manifest.json`
- implementation-failure provenance: `results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3/v2.3-random-b010-r0-real-only/implementation_failures/pre_a8652b8_directory_loader/`

The machine-readable closure is under `results/tfim_manifold_augmentation/tfim_state_augmentation_archive_v2_3/`.

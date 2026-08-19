# Confirmatory dataset-freeze failure forensics v1

## Verdict

**INSUFFICIENT_EVIDENCE.** Aggregate evidence identifies the failing checks, but deleted staging candidates prevent pair-level reconstruction and final discrimination among seed-specific genuine collision, unavoidable physical collision, and protocol contradiction.

**BLOCKED — QCNN benchmark remains prohibited.** No QCNN execution, dataset generation, eigensolve, seed change, promotion, threshold change, or source/config/test edit was performed.

## Candidate availability

Both candidate regimes are unavailable. `dataset_manifest.json` has null `artifact_paths`, `staging_artifacts_discarded=true`, and neither intended final path nor retained staging exists. Therefore states, g arrays, split assignments, and sample-level identifiers are unavailable; only aggregate hashes and audits survive.

## Gate cause

- `split_validation_pass=false`: only `random_split` failed, due to 2 cross-split projective near-duplicate pairs at fidelity `>= 1-1e-10`. Exact-g and canonical-hash cross-split overlaps were zero. `blocked_g` passed with train/val `0.023015991761299315`, train/test `0.125548482637164`, val/test/global `0.022405635927595036`; the `0.02` guard passes without rounding ambiguity.
- `freshness_validation_pass=false`: 10 directed comparisons failed (9 unordered relationships): blocked_g vs legacy blocked-g, legacy random, legacy tfim_4q, new random_split, pilot_v1; random_split vs legacy blocked-g, legacy random, legacy tfim_4q, new blocked_g, phase_a. All ID, exact-g, and canonical-hash overlaps were zero; only projective near-duplicates failed. No self-comparison/path alias was found.
- `dataset_freeze_complete=false`: derived from `all(dataset.valid)` because random_split was invalid. Scientific validation failed, so staging was discarded and promotion was correctly skipped.
- `qcnn_confirmatory_ready=false`: strict AND over seed, dataset, structural, physics, split, freshness, reproducibility, immutability, tests, review, and checksums; there is no hidden permissive path.

`random_split` here is a class-stratified physical TFIM dataset regime, not the later `distance_matched_random` augmentation arm.

## Seed and reproducibility

Seeds were frozen before generation: random `91001/92001`, blocked-g `91002/92002`; one predeclared set, zero comparative sweeps, no downstream metrics, no post-failure seed change. Both regimes reproduced exactly in semantic hashes, arrays, and serialized files. The failure is deterministic at aggregate level.

## Classification and remediation

- split validation: `GENUINE_DATASET_FAILURE` at aggregate validator level; exact physical classification `INSUFFICIENT_EVIDENCE`.
- freshness validation: `GENUINE_DATASET_FAILURE` under implemented v1 validator; physical/protocol classification `INSUFFICIENT_EVIDENCE`.
- dataset freeze: normal fail-closed derivative, not an independent defect.
- overall: `INSUFFICIENT_EVIDENCE` (medium confidence in aggregate causes; low confidence in physical/protocol cause).

Remediation decision: **INSUFFICIENT_EVIDENCE**. Do not revalidate nonexistent candidates, regenerate, change seeds, or amend thresholds. Next work must recover hash-matching candidates for read-only pair replay, or recover a trustworthy original pair export, and establish dated protocol provenance. If the criterion changes, use a versioned v2 amendment; do not rewrite v1.

## Source-attestation limitation

The generation-time source/config/tests were untracked. The config has a retained hash, but the validator source has no generation-time content hash. Current source is consistent with artifacts and tests but is not cryptographically proven to be the exact executed bytes.

## Independent review

Reviewer `aeb0add6` agreed with the conservative decision after correction of the directed-comparison count and clarification of source attestation and alternative evidence-recovery paths.

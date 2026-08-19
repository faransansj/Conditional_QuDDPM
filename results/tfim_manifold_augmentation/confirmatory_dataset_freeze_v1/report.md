# Confirmatory TFIM dataset freeze v1

Gate: **BLOCKED**; QCNN ready: **False**. No QCNN was run.

## Generated dataset evidence
- `random_split`: promotion `NOT_PROMOTED_VALIDATION_BLOCKED`, 400 generated states, semantic hash `ced581a4d29a410a90e8eb026f7b3406031d1cc11dc6f36042674228360908a6`
  - max norm error 8.882e-16; max residual 4.064e-15; minimum cross-split Δg 4.23289e-07; projective split duplicates 2
- `blocked_g`: promotion `NOT_PROMOTED_VALIDATION_BLOCKED`, 400 generated states, semantic hash `79102c54e69109a6dcdd33146f5ed398933415cf1462ea256b45359d768dc2c3`
  - max norm error 6.661e-16; max residual 5.403e-15; minimum cross-split Δg 0.0224056; projective split duplicates 0

## Protocol
Canonical generator: `conditional_quddpm.datasets.tfim.generate_dataset`.
Seeds came from `configs/augmentation/tfim_manifold_confirmatory.yaml` before generation; `seed_selection_count=1` means one predeclared seed set and `seed_sweep_count=0` means no comparative sweep. No downstream metric was inspected.
Freshness: False; reproducibility: True; immutable artifacts unchanged: True.
Base generation eigensolves: 1600 (800 primary + 800 reproducibility); pilot endpoint count remains 40.
Execution attempts recorded: 3; the latest bug-fix execution reused the same seeds and was not seed selection.

## Blocking evidence
Blocking prerequisites: dataset_freeze_complete, split_validation_pass, freshness_validation_pass.
The frozen random-split seed produces projective split duplicates and the cross-corpus freshness test fails at the unchanged tolerance. Criteria were not relaxed; blocked staging was discarded, so promotion is impossible.

This artifact establishes dataset validation evidence only. It makes no claim about augmentation utility.

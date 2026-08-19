# Confirmatory QCNN Protocol v2.3 post-experiment freeze

## Authoritative confirmatory result

- Execution integrity: **PASS** (48/48; no scientific failures)
- Blocked-g mean paired delta: **-0.018055555555555547** (-1.8056 percentage points)
- Paired bootstrap 95% CI: **[-0.0486111111111111, 0.0013888888888889024]**
- Confirmatory verdict: **FAIL**

Frozen 4-qubit TFIM, QCNN, q50 local-random-tangent, ratio 1.0, Protocol v2.3 setting에서 blocked-g 성능 향상 기준은 충족되지 않았다. CI는 0을 포함하므로 개선 증거는 없지만 augmentation이 일반적으로 해롭다고 확정할 근거도 부족하다.

## Exploratory / post-hoc

Random mean paired delta: 0.0. Blocked-g budget deltas: {"10": 0.0, "25": -0.055555555555555504, "50": -0.011111111111111108, "100": -0.005555555555555573}. These are descriptive only; random is not confirmatory evidence, n=3 cells receive no tests or verdicts, and no outlier was removed. Stored artifacts provide final train/validation losses and final metrics only; convergence trajectories were not stored and were not inferred. Calibration linkage (q50, ratio 1.0): acceptance minimum 1.0, FS displacement error maximum 1.4779844015322396e-14, anchor coverage minimum 1.0, duplicate-rate maximum 0.0, failed redraws 0. No causal or significance claim is made.

## Historical comparison boundary

Direct numerical effect comparison to pilot/Phase-C is invalid because all dataset, split, budget, seed, augmentation, and QCNN/SPSA settings do not match. FS-aware independence control 이후 기존 exploratory improvement가 blocked-g confirmatory setting에서 재현되지 않았다.

## Provenance and scope

Protocol and frozen datasets are referenced by hashes in `freeze_manifest.json`; source results remain in place and are referenced by `result_manifest.json`. The archived directory-loader implementation failure occurred before training (`training_updates = 0`) and is excluded from scientific failures. This result does not establish failure of geometric augmentation generally, quantum-state augmentation, all local tangent settings, or non-TFIM tasks.

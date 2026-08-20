# Quantum-State Augmentation for QCNNs

**Status: ARCHIVED / CLOSED**

This repository preserves the completed 4-qubit TFIM quantum-state augmentation research track. No further experiment is active on this branch.

## Final status

| Component | Final status |
|---|---|
| 4Q TFIM dataset | **COMPLETE** |
| QCNN baseline | **COMPLETE** |
| Conditional QuDDPM | **LOCAL NO-GO** |
| Physics-aware augmentation | **COMPLETED / exploratory negative** |
| Geometry-aware augmentation | **COMPLETED / bounded NO-GO** |
| Local random-tangent | **COMPLETED** |
| Confirmatory Protocol v2.3 | **COMPLETE** |
| Confirmatory QCNN | **48/48 COMPLETE** |
| Final scientific verdict | **FAIL** |
| MSQuDDPM | **HOLD / not evaluated** |

Execution integrity passed: all 48 planned scientific runs completed. The scientific verdict failed because the frozen augmentation method did not satisfy the predeclared blocked-g generalization improvement criterion. This is not an execution failure.

## Final result

Frozen scope: **4-qubit TFIM + fixed QCNN + q50 local-random-tangent augmentation + synthetic ratio 1.0 + confirmatory Protocol v2.3**.

| Primary confirmatory quantity | Value |
|---|---:|
| Blocked-g augmentation − real-only mean paired delta | `-0.018055555555555547` |
| Frozen paired-bootstrap 95% CI | `[-0.0486111111111111, 0.0013888888888889024]` |
| Confirmatory verdict | **FAIL** |

The CI includes zero. The result establishes neither improvement nor a general harmful effect. It does **not** establish that quantum-state, physics-aware, geometry-aware, local-tangent, or learned-generator augmentation fails in general.

Conditional QuDDPM is a bounded/local NO-GO for the frozen 4Q TFIM diagnostic chain. MSQuDDPM was not evaluated and remains on HOLD.

## How to read the evidence

- **Execution failure:** a run or implementation did not complete. The preserved loader failure occurred before training (`training_updates=0`) and was repaired; it is not a scientific run.
- **Scientific failure:** completed evidence failed a predeclared criterion. This is the final confirmatory outcome.
- **Methodological finding:** evidence about protocol or split design, such as distinguishing exact freshness from projective separation.
- **Exploratory result:** hypothesis-generating evidence from Phases A–C or post-hoc analysis; it does not change the confirmatory verdict.
- **Confirmatory result:** the frozen Protocol v2.3 result above.
- **Future hypothesis:** an untested possibility, not a repository conclusion.

Projective near-duplicate contamination remains a methodological concern and possible interpretation. The repository does not establish it as the cause of exploratory results.

## Authoritative artifacts

- [Final scientific archive](docs/tfim_state_augmentation_final_archive.md)
- [`confirmatory_protocol_v2_3/`](results/tfim_manifold_augmentation/confirmatory_protocol_v2_3/)
- [`confirmatory_dataset_freeze_v2_2/`](results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v2_2/)
- [`confirmatory_qcnn_v2_3_freeze/`](results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3_freeze/)
- [`tfim_state_augmentation_archive_v2_3/`](results/tfim_manifold_augmentation/tfim_state_augmentation_archive_v2_3/)

The archive document is authoritative for conclusions, provenance, metrics, hashes, and NO-GO scope. Machine-readable results and checksum manifests are preserved under `results/`.

## Research chronology

```text
TFIM/QCNN baseline
→ Conditional QuDDPM diagnostics
→ QuDDPM NO-GO
→ physics-aware Phase A
→ geometry-aware Phase B
→ local random-tangent Phase C
→ split/projective-independence audit
→ Protocol v2/v2.1/v2.2/v2.3
→ 48-run confirmatory QCNN
→ final archive
```

Detailed chronology and interpretation boundaries are in the [final archive](docs/tfim_state_augmentation_final_archive.md).

## Repository use

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --locked
uv run pytest
```

Core code is under `src/conditional_quddpm/`; frozen configs, documentation, datasets, and results remain in their existing repository paths. Validation and reproduction are appropriate; new training or sweeps are outside this archived branch.

## Scope and limitations

The result is limited to the frozen 4-qubit open-chain TFIM setting, fixed QCNN, q50 local-random-tangent method, ratio 1.0, and Protocol v2.3. It makes no claim of quantum advantage, hardware utility, other Hamiltonians, larger systems, other classifiers, or learned generators.

Exact simulation scales exponentially. Finite TFIM labels are operational benchmark definitions. Synthetic states are not independent ground-truth observations. Future hypotheses require a new research branch, an independently frozen protocol, and new evidence.

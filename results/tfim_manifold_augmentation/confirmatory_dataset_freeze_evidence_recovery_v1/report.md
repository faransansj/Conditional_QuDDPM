# Confirmatory dataset-freeze v1 evidence recovery

The same-frozen-seed candidate datasets and aggregate v1 failure were semantically reproduced, and complete pair-level evidence was recovered.

## Reproduction identity
- Overall: **BYTE_EXACT_REPRODUCTION**
- Historical attribution allowed: **True**

## Split failure pairs
- `tfim-confirmatory-v1-random_split-class-0-00046` (train, g=0.79850422517314934) vs `tfim-confirmatory-v1-random_split-class-0-00069` (val, g=0.79850464846174996): Δg=4.233e-07, F=0.99999999999991318, 1-F=8.682e-14, d_FS=2.947e-07; GENUINE_PROJECTIVE_NEAR_NEIGHBOR.
- `tfim-confirmatory-v1-random_split-class-1-00247` (train, g=1.4406610811816443) vs `tfim-confirmatory-v1-random_split-class-1-00336` (test, g=1.4406861784821114): Δg=2.510e-05, F=0.99999999996718825, 1-F=3.281e-11, d_FS=5.728e-06; GENUINE_PROJECTIVE_NEAR_NEIGHBOR.

## Freshness recovery
- Failed directed comparisons: 10; unordered corpus relationships: 9.
- Exported directed pair rows: 67; unique physical/sample relationships: 61.
- Every aggregate ID, exact-g, and canonical-hash overlap remains zero; failures are projective near-neighbors, not data reuse.

## Diagnosis
- Production and independently normalized projective metrics agree; no validator or floating-boundary defect was found.
- Random splitting has no parameter guard gap, so zero projective near-neighbors is not structurally guaranteed and acts as a hidden minimum-separation gate.
- The comparator matrix and zero-near-neighbor clauses are present in untracked implementation/config but are not generation-time cryptographically attested preregistration evidence.
- Remediation decision: **PROTOCOL_V2_REQUIRED**, confirmed by independent review. V1 remains failed and immutable; any changed freshness contract requires a separately preregistered v2.

## Gate
**BLOCKED — QCNN benchmark remains prohibited.**
QCNN runs: 0. QCNN metrics calculated: false. Final dataset promotion: false.

## Independent review finalization
- Review agreement: true. Root cause resolved: true.
- `blocked_g` split status: PASS / `NO_GATE_FAILURE`.
- Recovery generation source attested: true; historical execution source attested: false; final metadata replay attested: true.
- V1 and QCNN remain BLOCKED; no QCNN or promotion occurred.

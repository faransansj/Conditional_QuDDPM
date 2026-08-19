# Frozen Confirmatory Protocol v2

Status: **FROZEN**. Protocol SHA-256: `2e52ac26f626fb0703b06fc73e724820f23b4304ca51db8ca8c9cc0e07b959aa`.

SPSA performs exactly 300 updates without early stopping; validation is record-only and evaluation selects final step 300. Ground states with `E1-E0 <= 1e-10` are rejected, deterministically replaced, and fail closed after 100 retries. Named SeedSequence domains use NumPy Generator/PCG64DXSM. State identity normalizes complex128 vectors, removes global phase at the first amplitude above `1e-12`, rounds at tolerance precision and clears tiny/signed-zero residue, serializes little-endian contiguous bytes, and hashes SHA-256. Exact hash and Fubini--Study near-duplicate gates both apply.

Primary estimand is paired test-accuracy difference (augmentation minus real-only). Blocked-g is the mandatory generalization gate; random-only improvement cannot PASS. Missing, failed, NaN, or schema-invalid runs are INCONCLUSIVE. No outlier removal or retries are allowed.

Current gate: `protocol_v2_ready=true`, `dataset_freeze_ready=false`, `qcnn_confirmatory_ready=false`. No confirmatory dataset, QCNN run, or QCNN metric was produced.

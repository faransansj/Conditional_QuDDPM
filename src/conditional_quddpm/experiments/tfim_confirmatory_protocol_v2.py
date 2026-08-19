"""Frozen, execution-free contracts for TFIM confirmatory protocol v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

DOMAINS = (
    "dataset.parameter_sampling", "dataset.replacement_sampling", "dataset.split",
    "subset_selection", "augmentation", "qcnn.initialization", "qcnn.spsa", "statistics",
)
RUN_REQUIRED = {"protocol_version", "protocol_hash", "dataset_hash", "split", "budget", "method", "root_seed", "child_seeds", "num_spsa_updates", "evaluation_checkpoint", "primary_metric", "run_status"}
AGGREGATE_REQUIRED = {"n_expected", "n_completed", "n_failed", "paired_effects", "aggregate_effect", "confidence_interval", "decision", "decision_reason"}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def named_seed(root_seed: int, domain: str) -> int:
    if domain not in DOMAINS:
        raise ValueError(f"unknown RNG domain: {domain}")
    digest = hashlib.sha256(f"tfim-confirmatory-v2|{int(root_seed)}|{domain}".encode()).digest()
    words = np.frombuffer(digest, dtype="<u4")
    return int(np.random.SeedSequence([int(root_seed), *map(int, words)]).generate_state(1, dtype=np.uint64)[0])


def seed_manifest(root_seed: int) -> dict:
    return {"schema_version": 1, "root_seed": int(root_seed), "derivation": "SeedSequence([root_seed, uint32_le(SHA256('tfim-confirmatory-v2|root_seed|domain'))])", "domains": {name: named_seed(root_seed, name) for name in DOMAINS}}


def confirmatory_rng(seed: int, *, bit_generator: str = "PCG64DXSM") -> np.random.Generator:
    if bit_generator != "PCG64DXSM":
        raise ValueError("confirmatory RNG must use PCG64DXSM")
    return np.random.Generator(np.random.PCG64DXSM(int(seed)))


def canonical_state(state: np.ndarray, tolerance: float) -> np.ndarray:
    value = np.asarray(state, dtype=np.complex128)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("state must be a finite one-dimensional complex vector")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("zero state has no projective representative")
    value = value / norm
    pivots = np.flatnonzero(np.abs(value) > tolerance)
    if not len(pivots):
        raise ValueError("state has no amplitude above state_phase_tolerance")
    pivot = int(pivots[0])
    value *= np.exp(-1j * np.angle(value[pivot]))
    value[pivot] = abs(value[pivot]) + 0j
    decimals = max(0, int(np.ceil(-np.log10(tolerance))))
    value.real = np.round(value.real, decimals)
    value.imag = np.round(value.imag, decimals)
    value.real[np.abs(value.real) <= tolerance] = 0.0
    value.imag[np.abs(value.imag) <= tolerance] = 0.0
    return np.ascontiguousarray(value.astype("<c16", copy=False))


def state_hash(state: np.ndarray, tolerance: float) -> str:
    return hashlib.sha256(canonical_state(state, tolerance).tobytes()).hexdigest()


def ground_state_with_gap(hamiltonian: np.ndarray, tolerance: float) -> tuple[float, float, float, np.ndarray]:
    values, vectors = eigh(hamiltonian, subset_by_index=[0, 1])
    e0, e1 = map(float, values)
    gap = e1 - e0
    if gap <= tolerance:
        raise ValueError(f"degenerate_ground_state: gap={gap} <= tolerance={tolerance}")
    state = canonical_state(vectors[:, 0], tolerance)
    return e0, e1, gap, state


def sample_unique_ground_state(draw_parameter, hamiltonian, *, maximum_retries: int, tolerance: float) -> tuple[dict, list[dict]]:
    audit = []
    for attempt in range(maximum_retries + 1):
        parameter = float(draw_parameter(attempt))
        values, vectors = eigh(hamiltonian(parameter), subset_by_index=[0, 1])
        e0, e1 = map(float, values); gap = e1 - e0; accepted = gap > tolerance
        audit.append({"parameter": parameter, "E0": e0, "E1": e1, "gap": gap, "accepted": accepted, "rejection_reason": None if accepted else "degeneracy_gap_at_or_below_tolerance"})
        if accepted:
            return {"parameter": parameter, "energy": e0, "state": canonical_state(vectors[:, 0], tolerance)}, audit
    raise RuntimeError("degeneracy replacement retry exhaustion")


def validate_confirmatory_training(config: dict) -> None:
    expected = {"optimizer": "SPSA", "parameter_updates": 300, "early_stopping": False, "checkpoint_selection": "final", "evaluation_step": 300}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("confirmatory training/checkpoint contract conflict")


def validate_schema(value: dict, kind: str) -> None:
    required = RUN_REQUIRED if kind == "run" else AGGREGATE_REQUIRED if kind == "aggregate" else None
    if required is None or set(value) != required:
        raise ValueError(f"invalid {kind} schema")
    if kind == "run" and (value["num_spsa_updates"] != 300 or value["evaluation_checkpoint"] != 300):
        raise ValueError("run violates frozen training contract")
    if kind == "aggregate" and (value["decision"] not in {"PASS", "FAIL", "INCONCLUSIVE"} or value["n_completed"] + value["n_failed"] != value["n_expected"]):
        raise ValueError("aggregate violates frozen decision contract")


def gate_status(checks: dict[str, bool]) -> dict:
    protocol = bool(checks.get("protocol_v2_frozen"))
    dataset_keys = ("fresh_dataset", "provenance", "physical", "exact_duplicates", "near_duplicates", "seed_manifest", "checksums", "dataset_freeze_complete")
    dataset = protocol and all(checks.get(key, False) for key in dataset_keys)
    return {"protocol_v2_ready": protocol, "dataset_freeze_ready": dataset, "qcnn_confirmatory_ready": dataset, "status": "READY" if dataset else "BLOCKED", "blocking_reasons": [key for key in dataset_keys if not checks.get(key, False)]}


def freeze_protocol(protocol: dict, output: str | Path) -> str:
    payload = canonical_json(protocol)
    digest = hashlib.sha256(payload).hexdigest()
    Path(output).write_bytes(payload)
    return digest

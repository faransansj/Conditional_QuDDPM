"""QCNN-independent Protocol v2.1 FS calibration and constrained splitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from conditional_quddpm.datasets.tfim import SPLITS, ground_state, tfim_hamiltonian
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2 import canonical_json, confirmatory_rng

PARENT_PROTOCOL_HASH = "2e52ac26f626fb0703b06fc73e724820f23b4304ca51db8ca8c9cc0e07b959aa"


def fs_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.complex128) / np.linalg.norm(left)
    right = np.asarray(right, dtype=np.complex128) / np.linalg.norm(right)
    overlap = float(np.clip(abs(np.vdot(left, right)), 0.0, 1.0))
    return float(np.arctan2(np.sqrt(max(0.0, (1.0 - overlap) * (1.0 + overlap))), overlap))


def nearest_neighbors(states: np.ndarray, g: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for index, state in enumerate(states):
        candidates = np.flatnonzero(labels == labels[index])
        candidates = candidates[candidates != index]
        distances = np.asarray([fs_distance(state, states[j]) for j in candidates])
        nearest = int(candidates[int(np.argmin(distances))])
        rows.append({"state_index": index, "nearest_index": nearest, "fs_distance": float(distances.min()),
                     "delta_g": abs(float(g[index] - g[nearest])), "class": int(labels[index]), "g": float(g[index])})
    return rows


def numerical_floor(states: np.ndarray, g: np.ndarray, *, n_qubits: int = 4, J: float = 1.0,
                    boundary: str = "open") -> dict:
    distances = []
    for state, parameter in zip(states, g, strict=True):
        _, regenerated = ground_state(tfim_hamiltonian(n_qubits, J, float(parameter), boundary))
        distances.append(fs_distance(state, regenerated))
    values = np.asarray(distances)
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    return {"method": "deterministic ground-state regeneration at identical g using canonical TFIM eigensolver",
            "sample_count": len(values), "max": float(values.max()), "median": median,
            "upper_robust_bound": float(max(values.max(), median + 6 * 1.4826 * mad))}


def distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "min": float(values.min()),
            **{f"p{q}": float(np.percentile(values, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "max": float(values.max()), "mean": float(values.mean())}


def calibrate_epsilon(states: np.ndarray, g: np.ndarray, labels: np.ndarray) -> dict:
    floor = numerical_floor(states, g)
    rows = nearest_neighbors(states, g, labels)
    positive = np.unique([row["fs_distance"] for row in rows if row["fs_distance"] > floor["upper_robust_bound"]])
    if len(positive) < 3:
        raise RuntimeError("FS_THRESHOLD_UNRESOLVED")
    # A near-zero population is defensible only when its largest log-gap is clearly isolated.
    ratios = positive[1:] / positive[:-1]
    gap = int(np.argmax(ratios))
    if ratios[gap] < 10 or positive[gap] > np.percentile(positive, 10):
        raise RuntimeError("FS_THRESHOLD_UNRESOLVED")
    epsilon = float(np.sqrt(positive[gap] * positive[gap + 1]))
    if not floor["upper_robust_bound"] * 10 < epsilon < float(np.median(positive)) / 10:
        raise RuntimeError("FS_THRESHOLD_UNRESOLVED")
    return {"method": "largest >=10x log-gap in unique NN-FS values below the 10th percentile; geometric gap midpoint",
            "numerical_fs_floor": floor, "nearest_neighbors": rows,
            "nn_fs_distribution": distribution_summary(np.asarray([row["fs_distance"] for row in rows])),
            "epsilon_sep": epsilon,
            "selection_rationale": "QCNN-independent numerical floor plus isolated near-zero NN population and local TFIM spacing"}


def constrained_random_split(states: np.ndarray, labels: np.ndarray, target_counts: dict[str, int], epsilon: float,
                             seed: int, *, preferred_splits: np.ndarray | None = None,
                             maximum_retries: int = 32) -> tuple[np.ndarray, dict]:
    n = len(states)
    conflicts = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if fs_distance(states[i], states[j]) < epsilon:
                conflicts[i, j] = conflicts[j, i] = True
    total_conflicts = int(np.triu(conflicts, 1).sum())
    for attempt in range(maximum_retries + 1):
        retry_seed = int(np.random.SeedSequence([int(seed), attempt]).generate_state(1, dtype=np.uint64)[0])
        rng = confirmatory_rng(retry_seed)
        assigned = np.full(n, "", dtype="U5")
        failed = False
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == label); rng.shuffle(indices)
            remaining = dict(target_counts)
            for index in indices:
                alternatives = list(SPLITS); rng.shuffle(alternatives)
                preferred = None if preferred_splits is None else str(preferred_splits[index])
                preferences = ([preferred] if preferred in SPLITS else []) + [split for split in alternatives if split != preferred]
                choices = [split for split in preferences if remaining[split] and not np.any(conflicts[index] & (assigned != "") & (assigned != split))]
                if not choices:
                    failed = True; break
                split = choices[0]; assigned[index] = split; remaining[split] -= 1
            if failed or any(remaining.values()): failed = True; break
        if not failed:
            return assigned, {"assignment_attempts": attempt + 1, "conflicts": total_conflicts, "reassignments": 0,
                              "restarts": attempt, "maximum_retries": maximum_retries, "retry_seed_derivation": "SeedSequence([split_seed, attempt])"}
    raise RuntimeError("CONSTRAINED_RANDOM_SPLIT_INFEASIBLE")


def minimum_cross_split_fs(states: np.ndarray, splits: np.ndarray) -> float:
    return min(fs_distance(states[i], states[j]) for i in range(len(states)) for j in range(i + 1, len(states)) if splits[i] != splits[j])


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    points = np.sort(np.concatenate((left, right)))
    return float(np.max(np.abs(np.searchsorted(np.sort(left), points, side="right") / len(left) - np.searchsorted(np.sort(right), points, side="right") / len(right))))


def distribution_audit(g: np.ndarray, labels: np.ndarray, baseline: np.ndarray, constrained: np.ndarray) -> dict:
    diagnostics = {}; distorted = False
    for label in sorted(set(labels.tolist())):
        for split in SPLITS:
            a = g[(labels == label) & (baseline == split)]; b = g[(labels == label) & (constrained == split)]
            baseline_variance = float(np.var(a)); constrained_variance = float(np.var(b)); sd = float(np.std(a))
            mean_difference = abs(float(np.mean(b) - np.mean(a)))
            mean_shift = mean_difference / sd if sd else (0.0 if mean_difference == 0 else float("inf"))
            variance_ratio = constrained_variance / baseline_variance if baseline_variance else (1.0 if constrained_variance == 0 else float("inf"))
            row = {"count": len(b), "g_range": [float(b.min()), float(b.max())],
                   "quantiles": [float(x) for x in np.quantile(b, [0, .25, .5, .75, 1])],
                   "mean": float(b.mean()), "variance": constrained_variance, "ks_from_unconstrained": _ks(a, b),
                   "standardized_mean_shift": mean_shift, "variance_ratio": variance_ratio}
            diagnostics[f"class_{label}.{split}"] = row
            distorted |= row["ks_from_unconstrained"] > .20 or mean_shift > .25 or not .5 <= variance_ratio <= 2.0
    # A blocked-g split has disjoint per-split ranges; overlap in every class rejects that accidental regime change.
    blocked_like = any(not all(max(g[(labels == label) & (constrained == a)].min(), g[(labels == label) & (constrained == b)].min()) <=
                               min(g[(labels == label) & (constrained == a)].max(), g[(labels == label) & (constrained == b)].max())
                               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))) for label in set(labels.tolist()))
    distorted |= blocked_like
    return {"diagnostics": diagnostics, "thresholds": {"max_ks": .20, "max_standardized_mean_shift": .25, "variance_ratio": [.5, 2.0]},
            "blocked_g_like": blocked_like, "verdict": "RANDOM_SPLIT_DISTORTED" if distorted else "PASS"}


def freshness_projective_status(*, sample_identity_overlap: int, exact_parameter_overlap: int,
                                canonical_hash_overlap: int, artifact_hash_overlap: int,
                                minimum_fs: float, epsilon: float) -> dict:
    fresh = not any((sample_identity_overlap, exact_parameter_overlap, canonical_hash_overlap, artifact_hash_overlap))
    return {"freshness": "PASS" if fresh else "FAIL", "projective_separation": "PASS" if minimum_fs >= epsilon else "FAIL"}


def freeze_protocol(protocol: dict, output: str | Path) -> str:
    if protocol.get("protocol_version") != "2.1.0" or protocol.get("parent_protocol_hash") != PARENT_PROTOCOL_HASH:
        raise ValueError("invalid Protocol v2.1 lineage")
    payload = canonical_json(protocol); Path(output).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

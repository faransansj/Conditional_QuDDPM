"""Independent train-only random-tangent quantum-state augmentation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from conditional_quddpm.augmentation.geometry import (
    fubini_study_distance,
    random_tangent_control,
    stable_hash,
    state_hash,
)


RADIUS_QUANTILES = {"small": 0.25, "medium": 0.50, "large": 0.75}


def is_projective_duplicate(state: np.ndarray, others: Iterable[np.ndarray], tolerance: float = 1e-10) -> bool:
    """Return whether a state duplicates another up to global phase."""
    return any(abs(np.vdot(other, state)) ** 2 >= 1 - tolerance for other in others)


def calibrate_radii(states: np.ndarray) -> dict:
    """Calibrate radii from one class's frozen source subset only."""
    states = np.asarray(states, dtype=np.complex128)
    distances = [
        fubini_study_distance(states[i], states[j])
        for i in range(len(states)) for j in range(i + 1, len(states))
    ]
    if not distances:
        raise ValueError("radius calibration requires at least two source states")
    quantiles = {name: float(np.quantile(distances, q)) for name, q in {
        "min": 0, "q10": .1, "q25": .25, "median": .5, "q75": .75, "max": 1,
    }.items()}
    return {
        "pair_count": len(distances),
        "distance_distribution": quantiles,
        "radius_rule": "small=q25, medium=q50, large=q75",
        "radii": {name: float(np.quantile(distances, q)) for name, q in RADIUS_QUANTILES.items()},
    }


def generate_random_tangent_pool(
    states: np.ndarray,
    sample_ids: Iterable[str],
    *,
    dataset: str,
    class_label: int,
    radius_name: str,
    delta: float,
    run_seed: int,
    config_hash: str,
    code_commit_sha: str,
    source_subset_id: str,
    source_artifact_hash: str,
    count: int = 20,
    namespace: str = "phase_c_local_random_tangent",
    orthogonal_tolerance: float = 1e-12,
    duplicate_infidelity_tolerance: float = 1e-10,
    max_redraws: int = 100,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Generate a deterministic, anchor-balanced no-replacement candidate sequence."""
    states = np.asarray(states, dtype=np.complex128)
    ids = [str(value) for value in sample_ids]
    if len(states) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("states and unique sample IDs must have equal lengths")
    if radius_name not in RADIUS_QUANTILES or not 0 < delta < np.pi / 2:
        raise ValueError("radius must be a named, nonzero projective displacement")
    anchor_order = sorted(ids, key=lambda item: stable_hash(run_seed, class_label, radius_name, item, namespace))
    by_id = dict(zip(ids, states, strict=True))
    accepted: list[dict] = []
    state_by_id: dict[str, np.ndarray] = {}
    source_states = list(states)
    for index in range(count):
        anchor_id = anchor_order[index % len(anchor_order)]
        depth = index // len(anchor_order)
        seed_parts = (run_seed, class_label, anchor_id, radius_name, depth, namespace)
        synthetic, tangent, retry = random_tangent_control(
            by_id[anchor_id], delta, seed_parts=seed_parts,
            orthogonal_tolerance=orthogonal_tolerance, max_redraws=max_redraws,
            forbidden_states=[*source_states, *state_by_id.values()],
            duplicate_infidelity_tolerance=duplicate_infidelity_tolerance,
        )
        candidate_id = "local-" + stable_hash(dataset, *seed_parts)[:20]
        actual = fubini_study_distance(by_id[anchor_id], synthetic)
        source_fidelities = np.abs(states.conj() @ synthetic) ** 2
        accepted.append({
            "synthetic_sample_id": candidate_id,
            "dataset": dataset,
            "split": "train",
            "class_label": int(class_label),
            "anchor_sample_id": anchor_id,
            "radius_rule": "same-class frozen-source q25/q50/q75",
            "radius_name": radius_name,
            "delta": float(delta),
            "actual_displacement_fs": actual,
            "displacement_error": abs(actual - delta),
            "augmentation_ratio_membership": [ratio for ratio, size in ((.5, 5), (1, 10), (2, 20)) if index < size],
            "run_seed": int(run_seed),
            "generator_seed": int(stable_hash(*seed_parts, retry)[:16], 16),
            "tangent_retry_index": retry,
            "tangent_overlap_abs": float(abs(np.vdot(by_id[anchor_id], tangent))),
            "tangent_norm_error": abs(float(np.linalg.norm(tangent)) - 1),
            "normalization_error": abs(float(np.linalg.norm(synthetic)) - 1),
            "nearest_source_infidelity": 1 - float(source_fidelities.max()),
            "source_subset_id": source_subset_id,
            "source_artifact_hash": source_artifact_hash,
            "config_hash": config_hash,
            "code_commit_sha": code_commit_sha,
            "state_hash": state_hash(synthetic),
            "augmentation_method": "independent_random_tangent",
        })
        state_by_id[candidate_id] = synthetic
    return accepted, state_by_id

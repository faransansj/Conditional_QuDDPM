"""Leakage-safe projective-geodesic augmentation for normalized pure states."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np


def fubini_study_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return arccos(|<left|right>|), with clipping for roundoff."""
    overlap = abs(np.vdot(left, right))
    return float(np.arccos(np.clip(overlap, 0.0, 1.0)))


def phase_align(reference: np.ndarray, other: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Align ``other`` so its overlap with ``reference`` is real nonnegative."""
    overlap = np.vdot(reference, other)
    if abs(overlap) <= tolerance:
        raise ValueError("pair overlap is at or below overlap_tolerance")
    return np.exp(-1j * np.angle(overlap)) * other


def geodesic_interpolate(
    left: np.ndarray,
    right: np.ndarray,
    t: float,
    *,
    overlap_tolerance: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Interpolate on CP^(d-1); return the state and pre-normalization norm error."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    aligned = phase_align(left, right, overlap_tolerance)
    theta = fubini_study_distance(left, aligned)
    if theta <= overlap_tolerance:
        state = np.asarray(left, dtype=np.complex128).copy()
    else:
        tangent = (aligned - np.cos(theta) * left) / np.sin(theta)
        state = np.cos(t * theta) * left + np.sin(t * theta) * tangent
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("geodesic interpolation produced an invalid norm")
    return state / norm, abs(norm - 1.0)


def canonicalize_global_phase(state: np.ndarray) -> np.ndarray:
    """Choose a deterministic representative whose largest amplitude is positive real."""
    state = np.asarray(state, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(state)))
    if abs(state[pivot]) == 0:
        raise ValueError("zero state has no projective representative")
    canonical = state * np.exp(-1j * np.angle(state[pivot]))
    canonical[pivot] = abs(canonical[pivot]) + 0j
    return canonical


def state_hash(state: np.ndarray) -> str:
    canonical = canonicalize_global_phase(state).astype("<c16", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def stable_hash(*parts: object) -> str:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def select_local_pairs(
    states: np.ndarray,
    sample_ids: Iterable[str],
    *,
    k: int = 3,
    distance_quantile: float = 0.75,
    minimum_distance: float = 0.04,
    overlap_tolerance: float = 1e-12,
) -> dict:
    """Build the deterministic undirected kNN graph for one budget-limited class."""
    states = np.asarray(states, dtype=np.complex128)
    ids = [str(value) for value in sample_ids]
    if len(states) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("states and unique sample IDs must have equal lengths")
    if not 0 < k < len(states):
        raise ValueError("k must be positive and smaller than the class budget")
    distances = np.zeros((len(states), len(states)))
    all_pairs = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            distance = fubini_study_distance(states[i], states[j])
            distances[i, j] = distances[j, i] = distance
            pair = tuple(sorted((ids[i], ids[j])))
            all_pairs.append({"source_id_a": pair[0], "source_id_b": pair[1], "distance_fs": distance})
    cutoff = float(np.quantile([pair["distance_fs"] for pair in all_pairs], distance_quantile))
    graph_ids: set[tuple[str, str]] = set()
    for i, source_id in enumerate(ids):
        neighbors = sorted(
            ((distances[i, j], ids[j]) for j in range(len(states)) if i != j),
            key=lambda item: (item[0], item[1]),
        )[:k]
        graph_ids.update(tuple(sorted((source_id, neighbor_id))) for _, neighbor_id in neighbors)
    index = {sample_id: position for position, sample_id in enumerate(ids)}
    graph_edges = []
    eligible = []
    for a, b in sorted(graph_ids):
        distance = float(distances[index[a], index[b]])
        edge = {"source_id_a": a, "source_id_b": b, "distance_fs": distance}
        graph_edges.append(edge)
        overlap = abs(np.vdot(states[index[a]], states[index[b]]))
        if minimum_distance <= distance <= cutoff and overlap > overlap_tolerance:
            eligible.append(edge)
    return {
        "sample_ids": ids,
        "all_pair_distances": all_pairs,
        "distance_matrix": distances,
        "distance_cutoff_q75": cutoff,
        "graph_edges": graph_edges,
        "eligible_pairs": eligible,
    }


def stratified_candidate_order(candidates: list[dict], run_seed: int, class_label: int, namespace: str) -> list[dict]:
    """Stable per-t sorting followed by round-robin interleaving."""
    strata: dict[float, list[dict]] = {}
    for candidate in candidates:
        strata.setdefault(float(candidate["t"]), []).append(candidate)
    for t, items in strata.items():
        items.sort(key=lambda item: stable_hash(run_seed, class_label, item["source_id_a"], item["source_id_b"], t, namespace))
    ordered = []
    depth = 0
    values = sorted(strata)
    while any(depth < len(strata[t]) for t in values):
        ordered.extend(strata[t][depth] for t in values if depth < len(strata[t]))
        depth += 1
    return ordered


def generate_geometry_pool(
    states: np.ndarray,
    sample_ids: Iterable[str],
    *,
    dataset: str,
    run_seed: int,
    class_label: int,
    generator_config_hash: str,
    t_values: Iterable[float] = (0.25, 0.5, 0.75),
    k: int = 3,
    distance_quantile: float = 0.75,
    minimum_distance: float = 0.04,
    overlap_tolerance: float = 1e-12,
    normalization_tolerance: float = 1e-10,
    duplicate_infidelity_tolerance: float = 1e-10,
    namespace: str = "phase_b_geometry",
) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """Generate and order all unique candidates from one class budget subset."""
    states = np.asarray(states, dtype=np.complex128)
    ids = [str(value) for value in sample_ids]
    graph = select_local_pairs(
        states, ids, k=k, distance_quantile=distance_quantile,
        minimum_distance=minimum_distance, overlap_tolerance=overlap_tolerance,
    )
    by_id = dict(zip(ids, states, strict=True))
    accepted: list[dict] = []
    accepted_states: list[np.ndarray] = []
    failures = {"finite": 0, "normalization": 0, "source_duplicate": 0, "candidate_duplicate": 0}
    raw_count = 0
    for pair in graph["eligible_pairs"]:
        a, b = pair["source_id_a"], pair["source_id_b"]
        for t in t_values:
            raw_count += 1
            candidate, norm_error = geodesic_interpolate(by_id[a], by_id[b], float(t), overlap_tolerance=overlap_tolerance)
            if not np.all(np.isfinite(candidate)):
                failures["finite"] += 1
                continue
            if norm_error > normalization_tolerance or abs(np.linalg.norm(candidate) - 1.0) > normalization_tolerance:
                failures["normalization"] += 1
                continue
            source_fidelities = np.abs(states.conj() @ candidate) ** 2
            if float(source_fidelities.max()) >= 1 - duplicate_infidelity_tolerance:
                failures["source_duplicate"] += 1
                continue
            if accepted_states and float(np.max(np.abs(np.asarray(accepted_states).conj() @ candidate) ** 2)) >= 1 - duplicate_infidelity_tolerance:
                failures["candidate_duplicate"] += 1
                continue
            candidate_id = "geo-" + stable_hash(dataset, run_seed, class_label, a, b, float(t), namespace)[:20]
            distance_a = fubini_study_distance(by_id[a], candidate)
            distance_b = fubini_study_distance(candidate, by_id[b])
            accepted.append({
                "candidate_id": candidate_id,
                "dataset": dataset,
                "split": "train",
                "run_seed": int(run_seed),
                "class_label": int(class_label),
                "source_id_a": a,
                "source_id_b": b,
                "source_pair_id": [a, b],
                "pair_distance_fs": float(pair["distance_fs"]),
                "t": float(t),
                "distance_to_a_fs": distance_a,
                "distance_to_b_fs": distance_b,
                "fidelity_to_a": float(abs(np.vdot(by_id[a], candidate)) ** 2),
                "fidelity_to_b": float(abs(np.vdot(by_id[b], candidate)) ** 2),
                "nearest_source_fidelity": float(source_fidelities.max()),
                "pre_normalization_norm_error": norm_error,
                "state_hash": state_hash(candidate),
                "generator_config_hash": generator_config_hash,
                "augmentation_method": "same_class_local_projective_geodesic",
            })
            accepted_states.append(candidate)
    ordered = stratified_candidate_order(accepted, run_seed, class_label, namespace)
    state_by_id = {item["candidate_id"]: accepted_states[accepted.index(item)] for item in ordered}
    graph["raw_candidate_count"] = raw_count
    graph["failures"] = failures
    graph["duplicate_rate"] = (failures["source_duplicate"] + failures["candidate_duplicate"]) / raw_count if raw_count else 0.0
    return ordered, state_by_id, graph


def random_tangent_control(
    anchor: np.ndarray,
    displacement: float,
    *,
    seed_parts: tuple[object, ...],
    orthogonal_tolerance: float = 1e-12,
    max_redraws: int = 100,
    forbidden_states: Iterable[np.ndarray] = (),
    duplicate_infidelity_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a deterministic isotropic complex tangent direction and displaced state."""
    forbidden = list(forbidden_states)
    for counter in range(max_redraws):
        seed = int(stable_hash(*seed_parts, counter)[:16], 16)
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=anchor.shape) + 1j * rng.normal(size=anchor.shape)
        tangent = raw - anchor * np.vdot(anchor, raw)
        norm = float(np.linalg.norm(tangent))
        if norm <= orthogonal_tolerance:
            continue
        tangent /= norm
        state = np.cos(displacement) * anchor + np.sin(displacement) * tangent
        state /= np.linalg.norm(state)
        if forbidden and max(abs(np.vdot(other, state)) ** 2 for other in forbidden) >= 1 - duplicate_infidelity_tolerance:
            continue
        return state, tangent, counter
    raise RuntimeError("matched random tangent generation failed after 100 redraws")


def generate_matched_controls(
    geometry_candidates: list[dict],
    geometry_states: dict[str, np.ndarray],
    source_states: dict[str, np.ndarray],
    *,
    run_seed: int,
    class_label: int,
    duplicate_infidelity_tolerance: float = 1e-10,
    orthogonal_tolerance: float = 1e-12,
    max_redraws: int = 100,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Create one distance-matched random tangent state per geometry candidate."""
    controls: list[dict] = []
    state_by_id: dict[str, np.ndarray] = {}
    forbidden = list(source_states.values())
    for geometry in geometry_candidates:
        a, b = geometry["source_id_a"], geometry["source_id_b"]
        distances = {a: geometry["distance_to_a_fs"], b: geometry["distance_to_b_fs"]}
        anchor_id = min(distances, key=lambda key: (distances[key], key))
        displacement = float(distances[anchor_id])
        control, tangent, redraw = random_tangent_control(
            source_states[anchor_id], displacement,
            seed_parts=(run_seed, class_label, geometry["candidate_id"], "distance_matched_random_tangent"),
            orthogonal_tolerance=orthogonal_tolerance,
            max_redraws=max_redraws,
            forbidden_states=[*forbidden, *state_by_id.values()],
            duplicate_infidelity_tolerance=duplicate_infidelity_tolerance,
        )
        actual = fubini_study_distance(source_states[anchor_id], control)
        control_id = "ctrl-" + stable_hash(geometry["candidate_id"], "distance_matched_random_tangent")[:20]
        controls.append({
            "candidate_id": control_id,
            "geometry_candidate_id": geometry["candidate_id"],
            "dataset": geometry["dataset"],
            "split": "train",
            "run_seed": int(run_seed),
            "class_label": int(class_label),
            "source_id_a": a,
            "source_id_b": b,
            "source_pair_id": [a, b],
            "anchor_source_id": anchor_id,
            "target_displacement_fs": displacement,
            "actual_displacement_fs": actual,
            "displacement_matching_error": abs(actual - displacement),
            "redraw_counter": redraw,
            "tangent_overlap_abs": float(abs(np.vdot(source_states[anchor_id], tangent))),
            "state_hash": state_hash(control),
            "augmentation_method": "manifold_unaware_distance_matched_random_direction",
        })
        state_by_id[control_id] = control
    return controls, state_by_id

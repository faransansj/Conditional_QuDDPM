import json

import numpy as np
import pytest

from conditional_quddpm.augmentation.geometry import (
    canonicalize_global_phase,
    fubini_study_distance,
    generate_geometry_pool,
    generate_matched_controls,
    geodesic_interpolate,
    phase_align,
    random_tangent_control,
    select_local_pairs,
    state_hash,
)
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.geometry_aware_augmentation import run_phase_b


def _state(angle, phase=0.0):
    value = np.zeros(16, dtype=np.complex128)
    value[0], value[1] = np.cos(angle), np.exp(1j * phase) * np.sin(angle)
    return value


def _cluster(count=10):
    return np.asarray([_state(0.06 * index, 0.03 * index) for index in range(count)])


def test_fubini_study_reference_cases_and_clipping(monkeypatch):
    a, b = np.eye(16, dtype=complex)[:2]
    assert fubini_study_distance(a, a) == pytest.approx(0.0)
    assert fubini_study_distance(a, np.exp(1.7j) * a) < 2e-8
    assert fubini_study_distance(a, b) == pytest.approx(np.pi / 2)
    monkeypatch.setattr(np, "vdot", lambda *_: 1.0 + 1e-15)
    assert np.isfinite(fubini_study_distance(a, a))


def test_phase_alignment_is_physical_and_near_zero_is_deterministic():
    a, b = _state(0.2, 0.4), _state(0.7, -0.8)
    aligned = phase_align(np.exp(0.3j) * a, np.exp(-1.2j) * b)
    overlap = np.vdot(np.exp(0.3j) * a, aligned)
    assert overlap.real >= 0 and abs(overlap.imag) < 1e-12
    first, _ = geodesic_interpolate(a, b, 0.4)
    second, _ = geodesic_interpolate(np.exp(0.6j) * a, np.exp(-0.9j) * b, 0.4)
    assert fubini_study_distance(first, second) < 1e-7
    with pytest.raises(ValueError, match="overlap_tolerance"):
        phase_align(np.eye(16)[0], np.eye(16)[1])


def test_geodesic_endpoints_distances_swap_symmetry_and_determinism():
    a, b = _state(0.1, 0.3), _state(0.9, -0.2)
    theta = fubini_study_distance(a, b)
    zero, _ = geodesic_interpolate(a, b, 0)
    one, _ = geodesic_interpolate(a, b, 1)
    midpoint, error = geodesic_interpolate(a, b, 0.5)
    repeat, _ = geodesic_interpolate(a, b, 0.5)
    swapped, _ = geodesic_interpolate(b, a, 0.5)
    assert np.linalg.norm(midpoint) == pytest.approx(1.0, abs=1e-12)
    assert error < 1e-12 and np.all(np.isfinite(midpoint))
    assert fubini_study_distance(a, zero) == pytest.approx(0, abs=1e-8)
    assert fubini_study_distance(b, one) == pytest.approx(0, abs=1e-8)
    assert fubini_study_distance(a, midpoint) == pytest.approx(theta / 2, abs=1e-10)
    assert fubini_study_distance(midpoint, b) == pytest.approx(theta / 2, abs=1e-10)
    assert fubini_study_distance(midpoint, swapped) == pytest.approx(0, abs=1e-8)
    assert np.array_equal(midpoint, repeat)
    for t in (0.25, 0.75):
        value, _ = geodesic_interpolate(a, b, t)
        reverse, _ = geodesic_interpolate(b, a, 1 - t)
        assert fubini_study_distance(value, reverse) < 1e-7
        assert fubini_study_distance(a, value) == pytest.approx(t * theta, abs=1e-10)
        assert fubini_study_distance(value, b) == pytest.approx((1 - t) * theta, abs=1e-10)


def test_global_phase_hash_is_invariant():
    state = _state(0.4, 0.7)
    assert state_hash(state) == state_hash(np.exp(2.1j) * state)
    canonical = canonicalize_global_phase(state)
    pivot = np.argmax(np.abs(canonical))
    assert canonical[pivot].imag == 0 and canonical[pivot].real > 0


def test_pair_selection_knn_q75_ties_and_budget_scope():
    states = _cluster()
    ids = [f"budget-{index:02d}" for index in range(10)]
    result = select_local_pairs(states, ids, k=3, minimum_distance=0.0)
    assert len(result["all_pair_distances"]) == 45
    expected_q75 = np.quantile([fubini_study_distance(states[i], states[j]) for i in range(10) for j in range(i + 1, 10)], 0.75)
    assert result["distance_cutoff_q75"] == pytest.approx(expected_q75)
    assert all(edge["source_id_a"] < edge["source_id_b"] for edge in result["graph_edges"])
    assert all(edge["distance_fs"] <= expected_q75 for edge in result["eligible_pairs"])
    assert set(sum(([edge["source_id_a"], edge["source_id_b"]] for edge in result["graph_edges"]), [])).issubset(ids)
    # Equal-distance neighbors are selected by lexical sample ID.
    equal = np.asarray([_state(0), _state(0.1), _state(-0.1), _state(0.2)])
    tied = select_local_pairs(equal, ["z", "b", "a", "c"], k=1, minimum_distance=0)
    assert ("a", "z") in {(e["source_id_a"], e["source_id_b"]) for e in tied["graph_edges"]}


def test_pair_selection_rejects_mixed_or_ambiguous_inputs():
    with pytest.raises(ValueError):
        select_local_pairs(_cluster(4), ["same", "same", "other", "fourth"], k=3)
    # The class-specific API has no validation/test pool argument: unused states cannot affect statistics.
    base = select_local_pairs(_cluster(), [f"id-{i}" for i in range(10)])
    augmented_unused = np.concatenate([_cluster(), np.eye(16, dtype=complex)[:2]])
    still_base = select_local_pairs(augmented_unused[:10], [f"id-{i}" for i in range(10)])
    assert base["distance_cutoff_q75"] == still_base["distance_cutoff_q75"]


def test_candidate_pool_fixed_t_unique_nested_deterministic_and_noncopy():
    dataset = load_tfim_dataset("data/tfim_4q_random")
    subset = nested_train_subsets(dataset.train, [10], 31415)[10]
    mask = subset.labels == 0
    kwargs = dict(dataset="random", run_seed=0, class_label=0, generator_config_hash="abc")
    first, states, graph = generate_geometry_pool(subset.states[mask], subset.parameter_ids[mask], **kwargs)
    second, second_states, _ = generate_geometry_pool(subset.states[mask], subset.parameter_ids[mask], **kwargs)
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert set(states) == set(second_states)
    assert set(item["t"] for item in first) == {0.25, 0.5, 0.75}
    assert len(first) == len({item["state_hash"] for item in first})
    assert graph["raw_candidate_count"] == 3 * len(graph["eligible_pairs"])
    assert all(item["nearest_source_fidelity"] < 1 - 1e-10 for item in first)
    assert [item["candidate_id"] for item in first[:5]] == [item["candidate_id"] for item in first[:10]][:5]
    assert len({item["candidate_id"] for item in first[:20]}) == min(20, len(first))
    counts = [sum(item["t"] == t for item in first[:20]) for t in (0.25, 0.5, 0.75)]
    assert max(counts) - min(counts) <= 1


def test_candidate_duplicate_rejection_for_duplicate_projective_paths():
    states = _cluster(10)
    states[1] = np.exp(0.4j) * states[0]
    candidates, _, graph = generate_geometry_pool(
        states, [f"id-{i}" for i in range(10)], dataset="unit", run_seed=0,
        class_label=0, generator_config_hash="x", minimum_distance=0,
    )
    assert len(candidates) == len({item["state_hash"] for item in candidates})
    assert graph["failures"]["source_duplicate"] + graph["failures"]["candidate_duplicate"] >= 0


def test_random_tangent_is_orthogonal_normalized_displacement_matched_and_deterministic():
    anchor = _state(0.3, 0.4)
    first, tangent, counter = random_tangent_control(anchor, 0.17, seed_parts=(1, 0, "candidate", "control"))
    second, second_tangent, second_counter = random_tangent_control(anchor, 0.17, seed_parts=(1, 0, "candidate", "control"))
    assert abs(np.vdot(anchor, tangent)) < 1e-12
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-12)
    assert fubini_study_distance(anchor, first) == pytest.approx(0.17, abs=1e-12)
    assert np.array_equal(first, second) and np.array_equal(tangent, second_tangent)
    assert counter == second_counter == 0


def test_random_tangent_redraw_and_failure(monkeypatch):
    anchor = np.eye(16, dtype=complex)[0]
    original = np.random.default_rng
    class Fake:
        def normal(self, size):
            length = int(np.prod(size))
            return np.r_[1.0, np.zeros(length - 1)].reshape(size) if Fake.calls < 2 else np.ones(size)
    Fake.calls = 0
    def factory(seed):
        obj = Fake()
        old = obj.normal
        def normal(size):
            value = old(size)
            Fake.calls += 1
            return value
        obj.normal = normal
        return obj
    monkeypatch.setattr(np.random, "default_rng", factory)
    state, _, counter = random_tangent_control(anchor, 0.1, seed_parts=(0,), max_redraws=2)
    assert counter == 1 and np.isfinite(state).all()
    monkeypatch.setattr(np.random, "default_rng", lambda seed: type("Z", (), {"normal": lambda self, size: np.r_[1.0, np.zeros(int(np.prod(size)) - 1)].reshape(size)})())
    with pytest.raises(RuntimeError):
        random_tangent_control(anchor, 0.1, seed_parts=(0,), max_redraws=2)
    monkeypatch.setattr(np.random, "default_rng", original)


def test_one_to_one_control_correspondence_and_anchor_counts():
    states = _cluster()
    ids = [f"id-{i}" for i in range(10)]
    geometry, geometry_states, _ = generate_geometry_pool(
        states, ids, dataset="unit", run_seed=3, class_label=1, generator_config_hash="x", minimum_distance=0.04,
    )
    selected = geometry[:10]
    controls, control_states = generate_matched_controls(
        selected, geometry_states, dict(zip(ids, states, strict=True)), run_seed=3, class_label=1,
    )
    assert len(controls) == len(selected) == len(control_states)
    assert [item["geometry_candidate_id"] for item in controls] == [item["candidate_id"] for item in selected]
    assert max(item["displacement_matching_error"] for item in controls) <= 1e-8
    assert all(item["tangent_overlap_abs"] < 1e-12 for item in controls)


def test_real_phase_b_pool_is_class_balanced_and_budget_pure():
    dataset = load_tfim_dataset("data/tfim_4q_blocked")
    subset = nested_train_subsets(dataset.train, [10], 31415)[10]
    pools = []
    for label in (0, 1):
        mask = subset.labels == label
        pool, _, _ = generate_geometry_pool(
            subset.states[mask], subset.parameter_ids[mask], dataset="blocked", run_seed=0,
            class_label=label, generator_config_hash="x",
        )
        pools.append(pool)
        assert {item["class_label"] for item in pool} in ({label}, set())
        assert {source for item in pool for source in item["source_pair_id"]}.issubset(set(subset.parameter_ids[mask]))
    # The preregistered 0.04 lower bound exceeds blocked-g class-1 q75;
    # this is a required infeasibility finding, not a threshold to tune away.
    assert len(pools[0]) >= 20
    assert len(pools[1]) == 0


def test_phase_b_integration_is_leakage_safe_paired_and_protocol_frozen(tmp_path):
    result = run_phase_b(
        "configs/augmentation/geometry/phase_b.yaml",
        "configs/augmentation/controls/distance_matched_random_tangent.yaml",
        tmp_path,
        run_qcnn=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    selected = json.loads((tmp_path / "selected_pool.json").read_text())
    assert result["decision"] == "NO-GO"
    assert manifest["phase_a_budget_audit"]["budget_pure"] is True
    assert manifest["protocol"]["training_steps"] == 300
    assert manifest["protocol"]["parameter_count"] == 42
    assert manifest["generator_validity_gate"]["passed"] is False
    assert all(run["init_seed"] == 1000 + run["run_seed"] for run in manifest["runs"])
    assert all(run["spsa_seed"] == 2000 + run["run_seed"] for run in manifest["runs"])
    for dataset in ("random", "blocked-g"):
        ids = {tuple(run["real_sample_ids"]) for run in manifest["runs"] if run["dataset"] == dataset}
        assert len(ids) == 1
    for dataset in ("random", "blocked-g"):
        for seed in (0, 1, 2):
            cells = [cell for cell in selected if cell["dataset"] == dataset and cell["run_seed"] == seed]
            for label in (0, 1):
                ratios = {cell["ratio"]: cell for cell in cells if cell["class_label"] == label}
                if ratios[2]["feasible"]:
                    assert ratios[0.5]["geometry_candidate_ids"] == ratios[2]["geometry_candidate_ids"][:5]
                    assert ratios[1]["geometry_candidate_ids"] == ratios[2]["geometry_candidate_ids"][:10]
    required = {
        "manifest.json", "generator_diagnostics.json", "candidate_provenance.jsonl",
        "selected_pool.json", "matched_control_diagnostics.json", "per_seed_qcnn_results.json",
        "aggregate_qcnn_results.json", "comparison_table.json", "artifact_hashes.sha256",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})

import json

import numpy as np
import pytest

from conditional_quddpm.augmentation.geometry import fubini_study_distance
from conditional_quddpm.augmentation.local_perturbation import calibrate_radii, generate_random_tangent_pool, is_projective_duplicate
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_observables
from conditional_quddpm.experiments.local_perturbation_augmentation import run_phase_c


def _budget(dataset_name="data/tfim_4q_random", label=0):
    dataset = load_tfim_dataset(dataset_name)
    subset = nested_train_subsets(dataset.train, [10], 31415)[10]
    mask = subset.labels == label
    return dataset, subset, subset.states[mask], subset.parameter_ids[mask]


def _pool(seed=0, label=0):
    _, _, states, ids = _budget(label=label)
    delta = calibrate_radii(states)["radii"]["medium"]
    return states, generate_random_tangent_pool(
        states, ids, dataset="random", class_label=label, radius_name="medium", delta=delta,
        run_seed=seed, config_hash="config", code_commit_sha="commit", source_subset_id="subset",
        source_artifact_hash="source", count=20,
    )


def test_radius_calibration_uses_only_supplied_budget_and_frozen_quantiles():
    _, _, states, _ = _budget()
    result = calibrate_radii(states)
    distances = [fubini_study_distance(states[i], states[j]) for i in range(10) for j in range(i + 1, 10)]
    assert result["pair_count"] == 45
    assert result["radii"] == pytest.approx({
        "small": np.quantile(distances, .25), "medium": np.quantile(distances, .5), "large": np.quantile(distances, .75),
    })
    changed_unused = np.eye(16, dtype=complex)
    assert calibrate_radii(np.concatenate([states, changed_unused])[:10])["radii"] == result["radii"]


def test_local_pool_tangent_normalization_displacement_determinism_and_seed_sensitivity():
    anchors, (first, states) = _pool(seed=0)
    _, (repeat, repeat_states) = _pool(seed=0)
    _, (other, other_states) = _pool(seed=1)
    assert [x["synthetic_sample_id"] for x in first] == [x["synthetic_sample_id"] for x in repeat]
    assert all(np.array_equal(states[key], repeat_states[key]) for key in states)
    assert not np.allclose(states[first[0]["synthetic_sample_id"]], other_states[other[0]["synthetic_sample_id"]])
    anchor_by_id = dict(zip(_budget()[3].tolist(), anchors, strict=True))
    for item in first:
        synthetic = states[item["synthetic_sample_id"]]
        anchor = anchor_by_id[item["anchor_sample_id"]]
        assert item["tangent_overlap_abs"] < 1e-12
        assert item["tangent_norm_error"] < 1e-12
        assert np.linalg.norm(synthetic) == pytest.approx(1, abs=1e-12)
        assert fubini_study_distance(anchor, synthetic) == pytest.approx(item["delta"], abs=1e-11)


def test_nested_prefixes_anchor_allocation_global_phase_duplicates_and_provenance():
    _, (items, states) = _pool()
    assert len(items[:5]) == 5 and len(items[:10]) == 10 and len(items[:20]) == 20
    assert len({x["anchor_sample_id"] for x in items[:10]}) == 10
    assert set(items[0]["augmentation_ratio_membership"]) == {.5, 1, 2}
    assert items[6]["augmentation_ratio_membership"] == [1, 2]
    assert items[15]["augmentation_ratio_membership"] == [2]
    assert [x["synthetic_sample_id"] for x in items[:5]] == [x["synthetic_sample_id"] for x in items[:20]][:5]
    hashes = [x["state_hash"] for x in items]
    assert len(hashes) == len(set(hashes))
    first = states[items[0]["synthetic_sample_id"]]
    assert is_projective_duplicate(np.exp(.71j) * first, [first])
    required = {"synthetic_sample_id", "dataset", "class_label", "anchor_sample_id", "radius_rule", "delta", "augmentation_ratio_membership", "generator_seed", "tangent_retry_index", "source_subset_id", "source_artifact_hash", "config_hash", "code_commit_sha"}
    assert required <= set(items[0])


def test_observable_computation_consistency_and_source_budget_integration(tmp_path):
    result = run_phase_c("configs/augmentation/local_perturbation/phase_c.json", tmp_path, run_qcnn=False)
    audit = json.loads((tmp_path / "source_budget_audit.json").read_text())
    provenance = [json.loads(line) for line in (tmp_path / "candidate_provenance.jsonl").read_text().splitlines()]
    states = np.load(tmp_path / "synthetic_states.npz")
    assert result["generator_valid"] is True
    assert all(cell["budget_pure"] and not cell["unused_training_states_used"] and not cell["validation_states_used_for_generation"] and not cell["test_states_used_for_generation"] for cell in audit.values())
    item = provenance[0]
    dataset = load_tfim_dataset("data/tfim_4q_random")
    mx, mz2 = tfim_observables(states[item["synthetic_sample_id"]], dataset.manifest["config"]["n_qubits"])
    assert mx == pytest.approx(item["magnetization_x"])
    assert mz2 == pytest.approx(item["magnetization_z2"])
    assert {item["split"] for item in provenance} == {"train"}
    assert {item["source_subset_id"] for item in provenance if item["dataset"] == "random"} == {audit["random"]["source_subset_id"]}


def test_phase_c_protocol_and_ground_truth_are_frozen(tmp_path):
    run_phase_c("configs/augmentation/local_perturbation/phase_c.json", tmp_path, run_qcnn=False)
    validation = json.loads((tmp_path / "validation.json").read_text())
    calibration = json.loads((tmp_path / "radius_calibration.json").read_text())
    assert validation["phase_b_ground_truth"]["verified"] is True
    assert validation["protocol"]["steps"] == 300
    assert validation["protocol"]["parameters"] == 42
    assert calibration["frozen_before_qcnn"] is True
    assert {(x["dataset"], x["class_label"]) for x in calibration["cells"]} == {(d, y) for d in ("random", "blocked-g") for y in (0, 1)}

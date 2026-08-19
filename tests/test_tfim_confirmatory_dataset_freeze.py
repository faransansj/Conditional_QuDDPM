import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conditional_quddpm.datasets.tfim import _blocked_intervals
import conditional_quddpm.experiments.tfim_confirmatory_dataset_freeze as freeze_module
from conditional_quddpm.datasets.tfim import _records as canonical_records
from conditional_quddpm.experiments.tfim_confirmatory_dataset_freeze import (
    _projective_duplicate_count,
    canonical_state_hash,
    REQUIRED_RESULT_FILES,
    finalize_review,
    gate_status,
    generate_fresh_dataset,
    semantic_dataset_hash,
    validate_fresh_dataset,
    verify_result_checksums,
)


def canonical_config(strategy="random"):
    value = {
        "n_qubits": 4, "J": 1.0, "boundary": "open",
        "phase_regions": {"ferromagnetic": [0.2, 0.8], "paramagnetic": [1.2, 1.8]},
        "samples_per_class": 200, "dataset_seed": 91001, "split_seed": 92001,
        "split_strategy": strategy, "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "numerical_tolerance": 1e-10,
    }
    if strategy == "blocked": value.update(dataset_seed=91002, split_seed=92002, blocked_g_gap=0.02)
    return value


def test_fresh_generation_invokes_canonical_api_and_preserves_canonical_records(tmp_path, monkeypatch):
    config = canonical_config()
    expected = canonical_records(config)
    original = freeze_module.canonical_generate_dataset
    calls = []
    def spy(resolved, output):
        calls.append((resolved, Path(output)))
        return original(resolved, output)
    monkeypatch.setattr(freeze_module, "canonical_generate_dataset", spy)
    target = tmp_path / "dataset"
    assert generate_fresh_dataset(config, "random_split", target) == 400
    manifest = json.loads((target / "split_manifest.json").read_text())
    assert calls == [(config, target)]
    assert manifest["canonical_generator"] == "conditional_quddpm.datasets.tfim.generate_dataset"
    assert [(row["g"], row["label"], row["split"]) for row in manifest["records"]] == [(row["g"], row["label"], row["split"]) for row in expected]
    assert len({row["sample_id"] for row in manifest["records"]}) == 400


def test_blocked_canonical_records_have_exact_counts_and_protocol_guard_gap():
    config = canonical_config("blocked")
    rows = canonical_records(config)
    for label, region in ((0, "ferromagnetic"), (1, "paramagnetic")):
        intervals = _blocked_intervals(*config["phase_regions"][region], config["split_ratios"], config["blocked_g_gap"])
        for split in ("train", "val", "test"):
            values = [row["g"] for row in rows if row["label"] == label and row["split"] == split]
            assert len(values) == {"train": 140, "val": 30, "test": 30}[split]
            assert min(values) >= intervals[split][0] and max(values) <= intervals[split][1]
    for label in (0, 1):
        by_split = {split: np.array([row["g"] for row in rows if row["label"] == label and row["split"] == split]) for split in ("train", "val", "test")}
        assert min(np.min(np.abs(by_split[a][:, None] - by_split[b][None, :])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))) >= 0.02


def test_projective_duplicate_detection_is_global_phase_invariant():
    left = np.array([[1, 0], [0, 1]], dtype=complex)
    right = np.array([[1j, 0], [1, 1]], dtype=complex)
    right[1] /= np.linalg.norm(right[1])
    assert _projective_duplicate_count(left, right, self_comparison=False, tolerance=1e-10) == 1
    assert canonical_state_hash(left[0]) == canonical_state_hash(right[0])


def test_generation_refuses_overwrite(tmp_path):
    target = tmp_path / "dataset"; target.mkdir()
    with pytest.raises(FileExistsError): generate_fresh_dataset(canonical_config(), "random_split", target)


def test_generated_dataset_numerics_observables_labels_and_semantic_hash_are_stable(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    config = canonical_config()
    assert generate_fresh_dataset(config, "random_split", first) == 400
    assert generate_fresh_dataset(config, "random_split", second) == 400
    report = validate_fresh_dataset(first, config, "random_split")
    assert report["structural_validation_pass"] and report["physics_validation_pass"]
    assert not report["split_validation_pass"]  # frozen seed exposes near-projective cross-split duplicates
    assert report["counts"]["per_class"] == {"0": 200, "1": 200}
    assert report["split_leakage"]["cross_split_projective_duplicates"] > 0
    assert report["numerics"]["max_norm_error"] <= 1e-10
    assert report["numerics"]["max_eigenpair_residual"] <= 1e-10
    assert report["numerics"]["max_mx_recomputation_error"] <= 1e-10
    assert report["numerics"]["max_mz2_recomputation_error"] <= 1e-10
    assert report["labels"]["label_mismatch_count"] == 0
    assert semantic_dataset_hash(first) == semantic_dataset_hash(second)


def test_gate_stays_blocked_on_any_validation_or_review_failure():
    dataset = {"valid": True, "structural_validation_pass": True, "physics_validation_pass": True, "split_validation_pass": True}
    validation = {"datasets": {"random_split": dataset, "blocked_g": dataset}}
    common = {"valid": True}
    assert gate_status(validation, common, common, common, test_suite_pass=True, independent_review_pass=False)["status"] == "BLOCKED"
    assert gate_status(validation, common, common, common, test_suite_pass=True, independent_review_pass=True, checksums_validation_pass=False)["status"] == "BLOCKED"
    bad = dict(dataset, physics_validation_pass=False)
    validation = {"datasets": {"random_split": bad, "blocked_g": dataset}}
    gate = gate_status(validation, common, common, common, test_suite_pass=True, independent_review_pass=True)
    assert gate["status"] == "BLOCKED" and not gate["qcnn_confirmatory_ready"]


def test_frozen_config_preserves_pilot_accounting_immutability_scope_and_excludes_symmetry_arm():
    pilot = yaml.safe_load(Path("configs/augmentation/tfim_manifold_confirmatory.yaml").read_text())
    freeze = yaml.safe_load(Path("configs/augmentation/tfim_manifold_confirmatory_dataset_freeze.yaml").read_text())
    assert pilot["fresh_confirmatory_datasets"]["random"]["dataset_seed"] == 91001
    assert pilot["fresh_confirmatory_datasets"]["blocked-g"]["dataset_seed"] == 91002
    symmetry = json.loads(Path("results/tfim_manifold_augmentation/pilot_v1/symmetry_diagnostic.json").read_text())
    resource = json.loads(Path("results/tfim_manifold_augmentation/pilot_v1/resource_model.json").read_text())
    assert resource["endpoint_eigensolve_oracle_calls"] == 40
    assert symmetry["benchmark_arm_included"] is False
    assert freeze["immutable_paths"]["legacy_tfim_4q"] == "data/tfim_4q"


def test_finalize_review_and_checksum_verifier_fail_closed_on_incomplete_or_extra_artifacts(tmp_path):
    for name in REQUIRED_RESULT_FILES:
        (tmp_path / name).write_text("{}\n")
    assert not verify_result_checksums(tmp_path)
    with pytest.raises(RuntimeError, match="artifact set/checksums"):
        finalize_review(tmp_path, independent_review_pass=True, review_reference="review:test", test_suite_pass=True)
    (tmp_path / "unexpected.json").write_text("{}\n")
    assert not verify_result_checksums(tmp_path)


def test_finalize_refuses_changed_immutable_artifact_and_missing_retained_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze_module, "REPO_ROOT", tmp_path)
    immutable = tmp_path / "immutable"; immutable.mkdir(); evidence = immutable / "evidence.bin"; evidence.write_bytes(b"original")
    output = tmp_path / "result"; output.mkdir()
    tree = freeze_module._tree_hashes(immutable)
    dataset = {"valid": True, "structural_validation_pass": True, "physics_validation_pass": True, "split_validation_pass": True}
    artifacts = {
        "seed_manifest.json": {}, "dataset_manifest.json": {},
        "validation.json": {"datasets": {"random_split": dataset, "blocked_g": dataset}},
        "freshness_audit.json": {"valid": True}, "reproducibility_audit.json": {"valid": True},
        "resource_accounting.json": {}, "qcnn_gate.json": {}, "failure_history.json": {"attempts": []},
        "artifact_immutability_audit.json": {"valid": True, "before": {"evidence": tree}, "after": {"evidence": tree}},
    }
    for name, value in artifacts.items(): (output / name).write_text(json.dumps(value))
    (output / "report.md").write_text("pending\n")
    freeze_module._result_checksums(output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"immutable_paths": {"evidence": str(immutable)}}))
    evidence.write_bytes(b"altered")
    with pytest.raises(RuntimeError, match="immutable artifact hashes changed"):
        finalize_review(output, independent_review_pass=True, review_reference="review:test", test_suite_pass=True, config_path=config_path)
    evidence.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="validated staging was not retained"):
        finalize_review(output, independent_review_pass=True, review_reference="review:test", test_suite_pass=True, config_path=config_path)


def test_seed_manifest_defines_one_predeclared_set_and_zero_comparative_sweeps():
    seed = json.loads(Path("results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v1/seed_manifest.json").read_text())
    # Historical artifact is corrected by the bug-fix rerun; this assertion guards its contract afterward.
    assert seed["root_seed_material"]["random_split"] == {"dataset_seed": 91001, "split_seed": 92001}
    assert seed["root_seed_material"]["blocked_g"] == {"dataset_seed": 91002, "split_seed": 92002}
    assert seed["seed_selection_count"] == 1
    assert seed["seed_sweep_count"] == 0

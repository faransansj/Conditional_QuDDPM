import hashlib
import json
from pathlib import Path

import pytest
import yaml

from conditional_quddpm.experiments.tfim_manifold_confirmatory import (
    REQUIRED_ARTIFACTS,
    run_pilot,
    select_target_g,
    verify_manifest,
)


def test_fractional_endpoint_rule_is_deterministic_bounded_and_not_constant_delta_g():
    rule = {"fraction": 0.25, "direction_namespace": "test"}
    first = select_target_g("anchor-a", 0.3, (0.2, 0.8), rule)
    repeat = select_target_g("anchor-a", 0.3, (0.2, 0.8), rule)
    other = select_target_g("anchor-a", 0.5, (0.2, 0.8), rule)
    assert first == repeat
    assert 0.2 <= first[0] <= 0.8
    assert abs(first[0] - 0.3) != pytest.approx(abs(other[0] - 0.5))


def test_gate0_generator_pilot_is_paired_leakage_safe_and_stops_before_qcnn(tmp_path):
    result = run_pilot("configs/augmentation/tfim_manifold_confirmatory.yaml", tmp_path)
    validation = json.loads((tmp_path / "validation.json").read_text())
    resource = json.loads((tmp_path / "resource_model.json").read_text())
    freeze = json.loads((tmp_path / "protocol_freeze.json").read_text())
    symmetry = json.loads((tmp_path / "symmetry_diagnostic.json").read_text())
    physical = json.loads((tmp_path / "physical_generator_diagnostics.json").read_text())
    random = json.loads((tmp_path / "random_control_diagnostics.json").read_text())
    pairing = json.loads((tmp_path / "pairwise_distance_audit.json").read_text())
    budget = json.loads((tmp_path / "source_budget_audit.json").read_text())
    config = yaml.safe_load(Path("configs/augmentation/tfim_manifold_confirmatory.yaml").read_text())

    expected = 2 * 2 * config["real_states_per_class"]
    assert result == {
        "status": "PASS", "pilot_valid": True, "oracle_calls": expected,
        "manifest_valid": True, "qcnn_runs": 0, "output": str(tmp_path),
    }
    assert resource["decision"] == "RESOURCE_MODEL_A"
    assert resource["endpoint_eigensolve_oracle_calls"] == expected
    assert validation["valid"] and validation["qcnn_executed"] is False
    access = validation["dataset_access_and_scientific_use"]
    assert access == {
        "dataset_container_opened": True,
        "heldout_metadata_and_compressed_members_materialized": True,
        "validation_rows_selected": False,
        "test_rows_selected": False,
        "validation_values_used": False,
        "test_values_used": False,
    }
    assert physical["status"] == random["status"] == pairing["status"] == "PASS"
    assert pairing["pair_count"] == expected
    assert pairing["maximum_distance_mismatch"] <= config["tolerances"]["distance_match"]
    assert budget["budget_identity"] is budget["expected_arm_counts_passed"] is True
    assert budget["expected_total_states_per_synthetic_arm"] == expected
    assert all(row["real"] == row["expected_real"] == 10 for row in budget["counts"])
    assert all(row["physical"] == row["random"] == row["expected_per_synthetic_arm"] == 10 for row in budget["counts"])
    assert all(row["target_outside_blocked_val_test_support"] for row in budget["leakage_checks"])
    for audit in budget["dataset_access_audit"].values():
        assert audit["dataset_container_opened"] is True
        assert audit["test_rows_selected"] is audit["test_values_used"] is False
    assert symmetry["benchmark_arm_included"] is False
    assert symmetry["number_of_projective_duplicates"] == symmetry["number_of_transformations"]
    assert random["acceptance_checks"] == [
        "normalization", "tangent orthogonality", "pairwise FS-distance matching",
        "projective uniqueness", "anchor coverage",
    ]
    assert random["class_consistency_filter_applied"] is False
    assert all(sample["physics_label_preservation_claimed"] is False and sample["class_consistency_filter_applied"] is False for sample in random["samples"])

    payload = freeze["frozen_payload"]
    expected_freeze_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert freeze["frozen_payload_sha256"] == expected_freeze_hash
    assert payload["resolved_generator_config"] == config
    assert payload["numerical_tolerances"] == config["tolerances"]
    assert payload["random_control_construction"]["namespace"] == config["random_control"]["namespace"]
    assert payload["random_control_construction"]["maximum_redraws"] == config["random_control"]["maximum_redraws"]
    assert payload["fresh_confirmatory_dataset_configs"] == config["fresh_confirmatory_datasets"]
    assert payload["statistical_plan"] == config["statistics"]
    provenance = payload["reproducibility_and_provenance"]
    assert provenance["resolved_generator_config"] == config
    assert provenance["source_revision"]["git_commit_sha"]
    assert isinstance(provenance["source_revision"]["git_dirty"], bool)
    assert {"python", "numpy", "scipy", "pyyaml"} == set(provenance["software_versions"])
    for relative, digest in provenance["source_file_sha256"].items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == digest
    for fresh in config["fresh_confirmatory_datasets"].values():
        assert {"n_qubits", "J", "boundary", "phase_regions", "samples_per_class", "dataset_seed", "split_seed", "split_strategy", "split_ratios", "numerical_tolerance"}.issubset(fresh)

    assert set(REQUIRED_ARTIFACTS).issubset(path.name for path in tmp_path.iterdir())
    assert all(verify_manifest(tmp_path).values())
    for name in ("per_seed_results.json", "aggregate_results.json", "paired_comparisons.json", "statistical_analysis.json", "random_vs_blocked_analysis.json"):
        assert json.loads((tmp_path / name).read_text())["status"] == "NOT_RUN"


@pytest.mark.parametrize(
    ("synthetic_states_per_anchor", "augmentation_ratio", "message"),
    [(2, 2.0, "synthetic_states_per_anchor == 1"), (1, 0.5, "augmentation_ratio must equal")],
)
def test_pilot_rejects_unimplemented_or_inconsistent_synthetic_counts(
    tmp_path, synthetic_states_per_anchor, augmentation_ratio, message
):
    config = yaml.safe_load(Path("configs/augmentation/tfim_manifold_confirmatory.yaml").read_text())
    config["synthetic_states_per_anchor"] = synthetic_states_per_anchor
    config["augmentation_ratio"] = augmentation_ratio
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        run_pilot(config_path, tmp_path / "output")

import hashlib
import json
from pathlib import Path

import numpy as np

from conditional_quddpm.datasets.tfim import generate_dataset
from conditional_quddpm.experiments.tfim_confirmatory_dataset_freeze import generate_fresh_dataset, semantic_dataset_hash
from conditional_quddpm.experiments.tfim_confirmatory_dataset_freeze_recovery import historical_attribution_allowed, metric, split_pairs, verify_checksums

OUT = Path("results/tfim_manifold_augmentation/confirmatory_dataset_freeze_evidence_recovery_v1")


def small_config():
    return {"n_qubits": 4, "J": 1.0, "boundary": "open", "phase_regions": {"ferromagnetic": [0.2, 0.8], "paramagnetic": [1.2, 1.8]},
            "samples_per_class": 10, "dataset_seed": 91001, "split_seed": 92001, "split_strategy": "random",
            "split_ratios": {"train": .7, "val": .15, "test": .15}, "numerical_tolerance": 1e-10}


def test_recovery_wrapper_preserves_generation_rng_order_and_splits(tmp_path):
    canonical, recovery = tmp_path / "canonical", tmp_path / "recovery"
    generate_dataset(small_config(), canonical)
    generate_fresh_dataset(small_config(), "random_split", recovery)
    with np.load(canonical / "states.npz") as a, np.load(recovery / "states.npz") as b:
        for key in ("states", "g", "labels", "splits", "energies", "magnetization_x", "magnetization_z2"):
            assert np.array_equal(a[key], b[key])


def test_independent_metric_is_phase_invariant_stable_and_satisfies_distance_identity():
    psi = np.array([1, 1e-6], dtype=complex); psi /= np.linalg.norm(psi)
    phi = np.exp(0.7j) * psi
    row = metric(psi, phi)
    assert row["production_validator_result"] == row["independent_validator_result"]
    assert row["phase_aligned_identity_error"] < 1e-14
    assert row["stable_infidelity"] >= 0


def test_pair_exporter_separates_ids_g_hashes_and_projective_overlap():
    psi = np.array([1, 0], dtype=complex)
    records = [
        {"state": psi, "split": "train", "sample_id": "a", "g": .2, "label": 0, "energy": -1., "Mx": 0., "Mz2": 1., "state_hash": "ha", "index": 0},
        {"state": np.exp(.2j)*psi, "split": "val", "sample_id": "b", "g": .20001, "label": 0, "energy": -1., "Mx": 0., "Mz2": 1., "state_hash": "hb", "index": 1},
    ]
    pairs = split_pairs(records)
    assert len(pairs) == 1 and not pairs[0]["same_sample_ID"] and not pairs[0]["same_exact_g"]
    assert pairs[0]["production_validator_result"] and pairs[0]["same_projective_state"]


def test_recovery_identity_retention_pair_counts_threshold_and_gate():
    identity = json.loads((OUT / "reproduction_identity.json").read_text())
    reconciliation = json.loads((OUT / "pair_count_reconciliation.json").read_text())
    gate = json.loads((OUT / "recovery_gate.json").read_text())
    replay = json.loads((OUT / "replay_manifest.json").read_text())
    assert identity["historical_attribution_allowed"] and identity["overall_identity_classification"] in {"BYTE_EXACT_REPRODUCTION", "SEMANTIC_EXACT_REPRODUCTION"}
    assert reconciliation["split_pair_export_count"] == reconciliation["split_aggregate_count"] == 2
    assert reconciliation["freshness_pair_export_count"] == reconciliation["freshness_directed_aggregate_count"]
    assert reconciliation["failed_directed_comparisons"] == 10 and reconciliation["failed_unordered_relationships"] == 9
    assert gate["qcnn_status"] == "BLOCKED" and not gate["qcnn_execution_performed"]
    assert replay["seeds"]["random_split"] == {"dataset_seed": 91001, "split_seed": 92001}
    assert replay["seeds"]["blocked_g"] == {"dataset_seed": 91002, "split_seed": 92002}
    assert all(row["validation"]["split_leakage"]["cross_split_projective_duplicates"] == (2 if name == "random_split" else 0) for name, row in ((name, json.loads((OUT / "candidates" / name / "manifest.json").read_text())) for name in ("random_split", "blocked_g")))


def test_failed_candidates_are_retained_losslessly_and_not_promoted():
    for regime in ("random_split", "blocked_g"):
        candidate = OUT / "candidates" / regime
        assert {"states.npz", "metadata.json", "manifest.json"}.issubset(p.name for p in candidate.iterdir())
        with np.load(candidate / "states.npz") as data:
            assert data["states"].shape == (400, 16) and len(data["g"]) == len(data["splits"]) == 400
    assert not Path("data/tfim_4q_confirmatory_random").exists()
    assert not Path("data/tfim_4q_confirmatory_blocked_g").exists()


def test_independent_metrics_and_pair_archive_are_complete():
    audit = json.loads((OUT / "independent_metric_audit.json").read_text())
    lines = (OUT / "split_projective_pairs.jsonl").read_text().splitlines() + (OUT / "freshness_projective_pairs.jsonl").read_text().splitlines()
    with np.load(OUT / "projective_pair_states.npz") as states:
        assert len(states.files) == 2 * len(lines)
    assert audit["pair_count"] == len(lines)
    assert audit["classification_disagreement_count"] == 0
    assert audit["maximum_phase_aligned_identity_error"] < 1e-12


def test_blocked_gap_seed_sweep_threshold_immutability_and_checksums():
    replay = json.loads((OUT / "replay_manifest.json").read_text())
    resources = json.loads((OUT / "resource_accounting.json").read_text())
    immutability = json.loads((OUT / "artifact_immutability_audit.json").read_text())
    assert replay["validation"]["blocked_g"]["split_leakage"]["global_minimum_cross_split_delta_g"] == 0.022405635927595036
    assert resources["seed_sweep_count"] == 0 and resources["qcnn_runs"] == 0
    assert immutability["all_protected_artifacts_unchanged"]
    assert verify_checksums(OUT)
    assert all(json.loads(line)["duplicate_threshold"] == 1 - 1e-10 for line in (OUT / "split_projective_pairs.jsonl").read_text().splitlines())


def test_semantic_reproduction_mismatch_blocks_historical_attribution():
    row = {"identity_classification": "REPRODUCTION_MISMATCH", "seed_identity_match": True, "config_identity_match": True,
           "sample_counts_match": True, "semantic_dataset_hash_match": False, "g_label_split_ID_order_and_canonical_states_match": False,
           "aggregate_split_validation_match": True}
    assert not historical_attribution_allowed({"random_split": row}, True, True)


def test_finalized_review_attestation_and_nonfailure_classification():
    root = json.loads((OUT / "root_cause_analysis.json").read_text())
    gate = json.loads((OUT / "recovery_gate.json").read_text())
    review = json.loads((OUT / "independent_review.json").read_text())
    source = json.loads((OUT / "execution_source_manifest.json").read_text())
    decision = json.loads((OUT / "remediation_decision.json").read_text())
    assert root["blocked_g split validation"]["primary"] == "NO_GATE_FAILURE"
    assert root["blocked_g split validation"]["status"] == "PASS"
    assert root["root_cause_resolved"] and review["independent_review_pass"] and review["agreement"]
    assert gate["root_cause_resolved"] and gate["remediation_decision"] == "PROTOCOL_V2_REQUIRED"
    assert gate["qcnn_status"] == "BLOCKED" and not gate["qcnn_execution_performed"]
    assert source["recovery_generation_source_attested"] and source["final_metadata_replay_attested"]
    assert not source["historical_execution_source_attested"]
    assert decision["remediation_decision"] == "PROTOCOL_V2_REQUIRED"


def test_historical_qcnn_gate_seed_and_failure_history_remain_at_baseline_hashes():
    freeze_dir = Path("results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v1")
    baseline = json.loads((OUT / "protected_artifact_baseline.json").read_text())["baseline"]["freeze_v1"]["file_sha256"]
    for name in ("qcnn_gate.json", "seed_manifest.json", "failure_history.json"):
        path = freeze_dir / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == baseline[str(path.resolve().relative_to(Path.cwd().resolve()))]

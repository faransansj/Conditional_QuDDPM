import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_1 import (
    PARENT_PROTOCOL_HASH, calibrate_epsilon, constrained_random_split, distribution_audit,
    freeze_protocol, freshness_projective_status, fs_distance, minimum_cross_split_fs,
    nearest_neighbors, numerical_floor,
)

V2 = Path("results/tfim_manifold_augmentation/confirmatory_protocol_v2")
V21 = Path("results/tfim_manifold_augmentation/confirmatory_protocol_v2_1")


def test_fs_floor_nn_and_calibration_are_deterministic():
    g = np.array([.2, .200001, .3, .5])
    states = np.array([[np.cos(x), np.sin(x)] for x in (0, 1e-8, 1e-4, 1e-2)], dtype=complex)
    labels = np.zeros(4, dtype=int)
    rows = nearest_neighbors(states, g, labels)
    assert rows[0]["nearest_index"] == 1 and rows[0]["delta_g"] == pytest.approx(1e-6)
    assert fs_distance(states[0], states[1]) >= 0
    # Repository calibration artifact exercises deterministic eigensolver regeneration.
    artifact = json.loads((V21 / "fs_calibration.json").read_text())
    assert artifact["numerical_fs_floor"] == artifact["repeat_calibration"]["numerical_fs_floor"]
    assert artifact["epsilon_sep"] == artifact["repeat_calibration"]["epsilon_sep"]
    pool = np.load("results/tfim_manifold_augmentation/confirmatory_dataset_freeze_evidence_recovery_v1/candidates/random_split/states.npz")
    assert numerical_floor(pool["states"][:1], pool["g"][:1])["sample_count"] == 1
    first = calibrate_epsilon(pool["states"], pool["g"], pool["labels"])
    second = calibrate_epsilon(pool["states"], pool["g"], pool["labels"])
    assert first["epsilon_sep"] == second["epsilon_sep"] == artifact["epsilon_sep"]


def test_constrained_assignment_is_deterministic_and_enforces_separation():
    angles = np.array([0, 1e-5, .2, .4, .6, .8])
    states = np.array([[np.cos(x), np.sin(x)] for x in angles], complex)
    labels = np.zeros(6, dtype=int); counts = {"train": 2, "val": 2, "test": 2}
    first, audit = constrained_random_split(states, labels, counts, 1e-4, 7)
    second, _ = constrained_random_split(states, labels, counts, 1e-4, 7)
    assert np.array_equal(first, second) and first[0] == first[1]
    assert minimum_cross_split_fs(states, first) >= 1e-4
    assert audit["assignment_attempts"] >= 1


def test_constrained_assignment_retry_failure_is_fail_closed():
    states = np.array([[1, 0]] * 3, complex); labels = np.zeros(3, int)
    with pytest.raises(RuntimeError, match="CONSTRAINED_RANDOM_SPLIT_INFEASIBLE"):
        constrained_random_split(states, labels, {"train": 1, "val": 1, "test": 1}, 1e-3, 7, maximum_retries=2)


def test_distribution_audit_detects_preservation_and_blocked_like_distortion():
    g = np.arange(12, dtype=float); labels = np.zeros(12, int)
    random = np.array(["train", "val", "test"] * 4)
    assert distribution_audit(g, labels, random, random)["verdict"] == "PASS"
    constant = np.ones(12); changed = constant.copy(); changed[-1] = 2
    assert distribution_audit(changed, labels, random, random)["verdict"] == "PASS"
    assert distribution_audit(changed, labels, random, np.roll(random, 1))["verdict"] == "RANDOM_SPLIT_DISTORTED"
    blocked = np.array(["train"] * 4 + ["val"] * 4 + ["test"] * 4)
    assert distribution_audit(g, labels, random, blocked)["verdict"] == "RANDOM_SPLIT_DISTORTED"


def test_freshness_and_projective_separation_are_independent():
    status = freshness_projective_status(sample_identity_overlap=0, exact_parameter_overlap=0,
        canonical_hash_overlap=0, artifact_hash_overlap=0, minimum_fs=1e-7, epsilon=1e-6)
    assert status == {"freshness": "PASS", "projective_separation": "FAIL"}


def test_protocol_v2_immutable_v21_hash_reproducible_and_qcnn_fail_closed(tmp_path):
    checksums = dict(line.split("  ", 1)[::-1] for line in (V2 / "checksums.sha256").read_text().splitlines())
    assert checksums["protocol_manifest.json"] == PARENT_PROTOCOL_HASH
    assert all(hashlib.sha256((V2 / name).read_bytes()).hexdigest() == digest for name, digest in checksums.items())
    protocol = json.loads((V21 / "protocol_manifest.json").read_text()); target = tmp_path / "protocol.json"
    digest = freeze_protocol(protocol, target)
    gate = json.loads((V21 / "gate.json").read_text())
    assert digest == gate["protocol_hash"] == hashlib.sha256((V21 / "protocol_manifest.json").read_bytes()).hexdigest()
    assert gate["qcnn_run_count"] == 0 and not gate["qcnn_confirmatory_ready"]
    assert not gate["dataset_generation_performed"]

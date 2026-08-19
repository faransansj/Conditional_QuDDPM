import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_3 import (
    BUDGETS, build_contract, confirmatory_tangent_batch, freeze, paired_bootstrap, validate_contract,
)

V23 = Path("results/tfim_manifold_augmentation/confirmatory_protocol_v2_3")


def test_v23_freezes_complete_paired_execution_contract(tmp_path):
    artifacts = build_contract()
    validate_contract(artifacts)
    runs = artifacts["run_matrix.json"]["runs"]
    assert len(runs) == len({row["run_id"] for row in runs}) == 48
    for regime in ("random", "blocked_g"):
        for repeat in range(3):
            rows = [row for row in artifacts["execution_config.json"]["subsets"] if row["regime"] == regime and row["repeat"] == repeat]
            for label in ("0", "1"):
                subsets = [set(next(row for row in rows if row["budget"] == budget)["sample_ids_by_class"][label]) for budget in BUDGETS]
                assert all(left < right for left, right in zip(subsets, subsets[1:]))
    digest = freeze(tmp_path)
    assert digest == hashlib.sha256((tmp_path / "protocol_manifest.json").read_bytes()).hexdigest()
    assert json.loads((tmp_path / "gate.json").read_text())["qcnn_run_count"] == 0


def test_confirmatory_augmentation_and_statistics_are_fully_frozen():
    states = np.eye(4, dtype=complex)
    ids = np.array(["b", "a", "d", "c"]); labels = np.array([0, 0, 1, 1])
    first, records = confirmatory_tangent_batch(states, ids, labels, {0: .1, 1: .2}, 7)
    second, _ = confirmatory_tangent_batch(states, ids, labels, {0: .1, 1: .2}, 7)
    assert np.array_equal(first, second) and [row["anchor_sample_id"] for row in records] == ["a", "b", "c", "d"]
    assert np.allclose(np.linalg.norm(first, axis=1), 1)
    mean, interval = paired_bootstrap(np.arange(12) / 100)
    assert mean == pytest.approx(.055) and interval == pytest.approx([.035833333333333335, .07416666666666667])
    with pytest.raises(ValueError):
        paired_bootstrap(np.arange(11))


def test_committed_v23_artifacts_are_reproducible_and_frozen(tmp_path):
    digest = freeze(tmp_path)
    gate = json.loads((V23 / "gate.json").read_text())
    assert gate["status"] == "FROZEN" and gate["qcnn_confirmatory_ready"]
    assert gate["qcnn_run_count"] == 0 and gate["run_matrix_validated"] == 48
    assert digest == gate["protocol_hash"]
    for line in (V23 / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((V23 / name).read_bytes()).hexdigest() == expected
    for name in ("protocol_manifest.json", "run_matrix.json", "seed_mapping.json", "execution_config.json",
                 "statistical_plan.json", "calibration_linkage.json", "provenance.json"):
        assert (tmp_path / name).read_bytes() == (V23 / name).read_bytes()

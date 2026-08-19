import hashlib
import json
from pathlib import Path

from conditional_quddpm.experiments.tfim_confirmatory_qcnn_v2_3_freeze import EXPECTED_PRIMARY, OUTPUT, build, sha256


def test_freeze_reproduces_primary_and_preserves_sources(tmp_path):
    before = {path: sha256(path) for path in OUTPUT.rglob("*") if path.is_file()}
    manifest = build(destination=tmp_path / "freeze", starting_status="frozen starting state")
    after = {path: sha256(path) for path in OUTPUT.rglob("*") if path.is_file()}
    assert before == after
    assert len(manifest["result_hashes"]) == 48
    primary = json.loads((tmp_path / "freeze/confirmatory_analysis.json").read_text())
    assert (primary["blocked_g_mean_paired_delta"], primary["paired_bootstrap_95_ci"], primary["verdict"]) == EXPECTED_PRIMARY
    assert primary["execution_integrity"] == "PASS" and primary["completed_runs"] == 48


def test_freeze_contracts_and_checksums_are_reproducible(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = build(destination=first, starting_status="same")
    b = build(destination=second, starting_status="same")
    assert a["freeze_hash"] == b["freeze_hash"]
    results = json.loads((first / "result_manifest.json").read_text())
    assert results["count"] == results["unique_run_ids"] == 48
    assert all(row["updates_completed"] == 300 and row["evaluation_checkpoint"] == "final step 300" for row in results["results"])
    assert len({row["run_id"] for row in results["results"]}) == 48
    for line in (first / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest


def test_exploratory_is_separate_and_implementation_failure_excluded(tmp_path):
    build(destination=tmp_path, starting_status="same")
    primary = json.loads((tmp_path / "confirmatory_analysis.json").read_text())
    exploratory = json.loads((tmp_path / "exploratory_analysis.json").read_text())
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert primary["analysis_class"] == "confirmatory" and "budget_level" not in primary
    assert exploratory["analysis_class"] == "exploratory/post-hoc" and not exploratory["confirmatory_evidence"]
    assert all(cell["n"] == 3 and "descriptive only" in cell["interpretation"] for cell in exploratory["budget_level"].values())
    assert provenance["implementation_failure"]["training_updates"] == 0
    assert primary["failed_scientific_runs"] == 0

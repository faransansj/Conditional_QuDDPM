import hashlib
import json
from pathlib import Path

ARCHIVE = Path("results/tfim_manifold_augmentation/tfim_state_augmentation_archive_v2_3")
FREEZE = Path("results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3_freeze")
RESULTS = Path("results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_archive_preserves_frozen_protocol_datasets_results_and_failure():
    archive = json.loads((ARCHIVE / "archival_manifest.json").read_text())
    freeze = json.loads((FREEZE / "freeze_manifest.json").read_text())
    assert archive["protocol_v2_3_hash"] == freeze["protocol_hash"] == "bb33305486af6c998e377dd93fc74932dc3ad87bbc986dce049b792e58d72c92"
    assert archive["dataset_state_file_sha256"] == freeze["dataset_hashes"] == {
        "blocked_g": "593733297e3e952ecf0cddb802b89379018c5b3c5225ee098fb2cd4237b36742",
        "random": "ff2f5fc4de67fe7e22e50d75617be469799964dc52282853ee4ca612dde630ff",
    }
    assert archive["dataset_scientific_hashes"] == {
        "blocked_g": "cb0446120e7df9b0b6052f4575f6a1ff10742d8aaa49542054530ec6215e8867",
        "random": "09fd5792318cd171a3c39316adff1dbce6c402c9bc0fa66bd2df639fab73cff7",
    }
    assert archive["result_freeze_manifest_sha256"] == sha(FREEZE / "freeze_manifest.json")
    assert archive["result_freeze_hash"] == freeze["freeze_hash"] == "a641e912d8de85e0a9d5b03a3c6d4262f214275fbc92c453ef7a5d593e9585b4"
    assert archive["result_hashes"] == freeze["result_hashes"] and len(archive["result_hashes"]) == 48
    for run_id, digest in archive["result_hashes"].items():
        assert sha(RESULTS / run_id / "result.json") == digest
    failure = archive["implementation_failure_archive"]
    assert failure["training_updates"] == 0
    assert sha(failure["path"] + "/failure.json") == failure["failure_sha256"]
    assert sha(failure["path"] + "/recovery.json") == failure["recovery_sha256"]


def test_archive_reproduces_frozen_numbers_and_separates_claims():
    verdict = json.loads((ARCHIVE / "final_verdict.json").read_text())
    assert verdict["execution_integrity"] == "PASS"
    assert verdict["confirmatory_verdict"] == "FAIL"
    assert verdict["result_freeze"] == "FROZEN"
    assert verdict["research_track"] == "ARCHIVED_NO_GO"
    assert verdict["primary"]["mean_paired_delta"] == -0.018055555555555547
    assert verdict["primary"]["paired_bootstrap_95_ci"] == [-0.0486111111111111, 0.0013888888888889024]
    assert verdict["exploratory_post_hoc"]["confirmatory_evidence"] is False
    assert verdict["exploratory_post_hoc"]["random_mean_paired_delta"] == 0.0
    assert verdict["exploratory_post_hoc"]["leave_one_repeat_out_all_negative"] is True
    assert verdict["exploratory_post_hoc"]["runs_at_300_updates"] == 48


def test_archive_checksums_and_documentation_are_complete():
    for line in (ARCHIVE / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(ARCHIVE / name) == digest
    manifest = json.loads((ARCHIVE / "archival_manifest.json").read_text())
    assert manifest["no_qcnn_rerun"] is True and manifest["no_new_scientific_result"] is True
    assert all(sha(path) == digest for path, digest in manifest["evidence_artifacts"].items())
    report = Path("docs/tfim_state_augmentation_final_archive.md").read_text()
    for text in ("-0.018055555555555547", "-0.0486111111111111", "0.0013888888888889024",
                 "Execution integrity is **PASS**", "scientific hypothesis verdict is **FAIL**"):
        assert text in report

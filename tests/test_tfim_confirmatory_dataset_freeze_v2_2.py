import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conditional_quddpm.experiments.tfim_confirmatory_dataset_freeze_v2_2 import (
    EPSILON_SEP, PROTOCOL_HASH, _initial_parameters, scientific_hash, validate_corpus,
)
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_2 import generation_seed_manifest

OUT = Path("results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v2_2")


def test_rng_reference_fixtures_are_repeatable_and_frozen():
    first, preferred_first = _initial_parameters("random"); second, preferred_second = _initial_parameters("random")
    assert first == second and np.array_equal(preferred_first, preferred_second)
    assert first[:3] == [
        (0, 0.45679550895117677, None),
        (0, 0.6190167913326607, None),
        (0, 0.5290257223368748, None),
    ]
    blocked, _ = _initial_parameters("blocked_g")
    assert blocked[:2] == [(0, 0.38273735808853204, "train"), (0, 0.5247147673891661, "train")]
    assert generation_seed_manifest()["bit_generator"] == "PCG64DXSM"


def test_frozen_dataset_artifacts_validate_and_reproduce():
    gate = json.loads((OUT / "freeze_gate.json").read_text())
    assert gate["status"] in {"PENDING_INDEPENDENT_REVIEW", "PENDING_INDEPENDENT_REREVIEW", "PENDING_FINAL_INDEPENDENT_REVIEW", "FROZEN"} and gate["protocol_hash"] == PROTOCOL_HASH
    assert gate["qcnn_run_count"] == 0 and gate["projective_separation_pass"]
    random_validation = json.loads((OUT / "random" / "validation.json").read_text())
    assert random_validation["projective_separation"]["epsilon_sep"] == EPSILON_SEP
    assert random_validation["projective_separation"]["violation_count"] == 0
    assert random_validation["distribution"]["verdict"] == "PASS"
    for regime in ("random", "blocked_g"):
        assert scientific_hash(OUT / regime) == gate["dataset_hashes"][regime]
        assert json.loads((OUT / regime / "validation.json").read_text())["valid"]


def test_validation_recomputes_data_and_provenance_and_fails_closed(tmp_path):
    calibration = Path("results/tfim_manifold_augmentation/confirmatory_dataset_freeze_evidence_recovery_v1/candidates/random_split/states.npz")
    clean = tmp_path / "clean"; shutil.copytree(OUT / "random", clean)
    assert validate_corpus(clean, "random", calibration_path=calibration)["valid"]
    for case in ("state_hash", "nan_observable", "provenance", "domain", "schema", "top_seeds", "replacement_event", "generator_commit", "coordinated_commit"):
        corpus = tmp_path / case; shutil.copytree(OUT / "random", corpus)
        if case in {"provenance", "domain", "schema", "top_seeds", "replacement_event", "generator_commit", "coordinated_commit"}:
            manifest = json.loads((corpus / "manifest.json").read_text())
            if case == "provenance": manifest["records"][0]["protocol_version"] = "wrong"
            elif case == "domain": manifest["records"][0]["replacement_domain"] = "wrong"
            elif case == "schema": manifest["schema_version"] = 2
            elif case == "top_seeds": manifest["seeds"]["root_seed"] = 1
            elif case == "replacement_event": manifest["records"][0]["replacement_events"] = [{"attempt":1,"rejected_g":manifest["records"][0]["g"],"E0":0.0,"E1":0.0,"gap":0.0,"replacement_g":manifest["records"][0]["g"]}]; manifest["records"][0]["replacement_status"] = "replaced"
            elif case == "generator_commit": manifest["records"][0]["generator_commit"] = "0" * 40
            else:
                manifest["generator_commit"] = "0" * 40
                for record in manifest["records"]: record["generator_commit"] = "0" * 40
            (corpus / "manifest.json").write_text(json.dumps(manifest))
        else:
            with np.load(corpus / "states.npz") as source:
                arrays = {name: np.asarray(source[name]) for name in source.files}
            if case == "state_hash": arrays["state_hashes"] = arrays["state_hashes"].copy(); arrays["state_hashes"][0] = "0" * 64
            else: arrays["magnetization_x"] = arrays["magnetization_x"].copy(); arrays["magnetization_x"][0] = np.nan
            np.savez_compressed(corpus / "states.npz", **arrays)
        assert not validate_corpus(corpus, "random", calibration_path=calibration)["valid"]


def test_freeze_checksums_cover_every_artifact():
    rows = [line.split("  ", 1) for line in (OUT / "checksums.sha256").read_text().splitlines()]
    covered = {name for _, name in rows}
    actual = {str(path.relative_to(OUT)) for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256"}
    assert covered == actual and all(hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest for digest, name in rows)
    for regime in ("random", "blocked_g"):
        checks = json.loads((OUT / regime / "checksums.json").read_text())
        assert all(hashlib.sha256((OUT / regime / name).read_bytes()).hexdigest() == digest for name, digest in checks.items())

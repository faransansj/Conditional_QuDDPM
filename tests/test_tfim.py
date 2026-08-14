import json

import numpy as np
import pytest

from conditional_quddpm.datasets.tfim import (
    X,
    Z,
    generate_dataset,
    ground_state,
    tfim_hamiltonian,
    validate_dataset,
    verify_checksums,
    write_checksums,
)


def config(strategy="random", samples=10):
    result = {
        "n_qubits": 4,
        "J": 1.0,
        "boundary": "open",
        "phase_regions": {"ferromagnetic": [0.2, 0.8], "paramagnetic": [1.2, 1.8]},
        "samples_per_class": samples,
        "dataset_seed": 3,
        "split_seed": 5,
        "split_strategy": strategy,
        "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    if strategy == "blocked":
        result["blocked_g_gap"] = 0.02
    return result


def build_validated_dataset(config, path):
    generate_dataset(config, path)
    report = validate_dataset(path)
    write_checksums(path)
    return report


def test_two_qubit_hamiltonian_matches_definition():
    J, g = 1.3, 0.4
    expected = -J * np.kron(Z, Z) - g * (np.kron(X, np.eye(2)) + np.kron(np.eye(2), X))
    actual = tfim_hamiltonian(2, J=J, g=g)
    assert np.allclose(actual, expected)
    assert np.allclose(actual, actual.conj().T)


def test_ground_state_is_normalized_eigenvector():
    hamiltonian = tfim_hamiltonian(4, g=0.7)
    energy, state = ground_state(hamiltonian)
    assert np.isclose(np.vdot(state, state), 1.0)
    assert np.linalg.norm(hamiltonian @ state - energy * state) < 1e-10


def test_invalid_phase_regions_are_rejected(tmp_path):
    invalid = config()
    invalid["phase_regions"] = {"ferromagnetic": [0.2, 1.1], "paramagnetic": [1.0, 1.8]}
    with pytest.raises(ValueError, match="phase regions"):
        generate_dataset(invalid, tmp_path / "invalid")


def test_random_dataset_is_reproducible_and_leakage_free(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    legacy_config = config()
    legacy_config.pop("split_strategy")  # Existing smoke configs default to random.
    report = build_validated_dataset(legacy_config, first)
    build_validated_dataset(legacy_config, second)
    assert (first / "split_manifest.json").read_bytes() == (second / "split_manifest.json").read_bytes()
    assert report["valid"]
    assert report["split_strategy"] == "random"
    assert report["split_counts"] == {"train": 12, "val": 4, "test": 4}

    manifest = json.loads((first / "split_manifest.json").read_text())
    split_ids = {
        split: {r["parameter_id"] for r in manifest["records"] if r["split"] == split}
        for split in ("train", "val", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["val"] | split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])


def test_blocked_split_enforces_cross_split_g_gap(tmp_path):
    path = tmp_path / "blocked"
    report = build_validated_dataset(config("blocked", samples=20), path)
    assert report["valid"]
    assert report["split_strategy"] == "blocked"
    assert report["minimum_cross_split_g_gap"] >= 0.02

    data = np.load(path / "states.npz")
    for label in (0, 1):
        maxima = {split: data["g"][(data["labels"] == label) & (data["splits"] == split)].max() for split in ("train", "val")}
        minima = {split: data["g"][(data["labels"] == label) & (data["splits"] == split)].min() for split in ("val", "test")}
        assert maxima["train"] < minima["val"]
        assert maxima["val"] < minima["test"]


def test_observables_separate_tfim_classes(tmp_path):
    report = build_validated_dataset(config(samples=20), tmp_path / "observables")
    ferro = report["observable_means"]["ferromagnetic"]
    para = report["observable_means"]["paramagnetic"]
    assert ferro["magnetization_z2"] > para["magnetization_z2"]
    assert para["magnetization_x"] > ferro["magnetization_x"]


@pytest.mark.parametrize("filename", ["states.npz", "split_manifest.json", "validation.json"])
def test_checksums_cover_and_detect_changes(tmp_path, filename):
    path = tmp_path / filename.replace(".", "-")
    build_validated_dataset(config(), path)
    assert verify_checksums(path) == {
        "states.npz": True,
        "split_manifest.json": True,
        "validation.json": True,
    }

    with (path / filename).open("ab") as artifact:
        artifact.write(b"tampered")
    assert verify_checksums(path)[filename] is False

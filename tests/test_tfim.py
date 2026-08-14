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
)


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
    config = {
        "n_qubits": 4,
        "J": 1.0,
        "boundary": "open",
        "phase_regions": {"ferromagnetic": [0.2, 1.1], "paramagnetic": [1.0, 1.8]},
        "samples_per_class": 2,
        "dataset_seed": 3,
        "split_seed": 5,
        "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
    }
    with pytest.raises(ValueError, match="phase regions"):
        generate_dataset(config, tmp_path / "invalid")


def test_dataset_is_reproducible_and_leakage_free(tmp_path):
    config = {
        "n_qubits": 4,
        "J": 1.0,
        "boundary": "open",
        "phase_regions": {"ferromagnetic": [0.2, 0.8], "paramagnetic": [1.2, 1.8]},
        "samples_per_class": 10,
        "dataset_seed": 3,
        "split_seed": 5,
        "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    first, second = tmp_path / "first", tmp_path / "second"
    generate_dataset(config, first)
    generate_dataset(config, second)
    assert (first / "split_manifest.json").read_bytes() == (second / "split_manifest.json").read_bytes()
    assert validate_dataset(first)["valid"]

    manifest = json.loads((first / "split_manifest.json").read_text())
    split_ids = {
        split: {r["parameter_id"] for r in manifest["records"] if r["split"] == split}
        for split in ("train", "val", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["val"] | split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])

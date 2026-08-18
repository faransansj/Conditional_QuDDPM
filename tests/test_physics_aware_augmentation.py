import json

import numpy as np
import pytest

from conditional_quddpm.augmentation.physics import (
    accepted,
    augment_state,
    fit_acceptance_gate,
    state_diagnostics,
    tfim_components,
)
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_hamiltonian
from conditional_quddpm.experiments.physics_aware_augmentation import run_diagnostics


def _source():
    dataset = load_tfim_dataset("data/tfim_4q_random")
    subset = nested_train_subsets(dataset.train, [2], 31415)[2]
    records = {record["parameter_id"]: record for record in dataset.manifest["records"]}
    record = records[str(subset.parameter_ids[0])]
    metadata = {
        "source_state_id": record["parameter_id"],
        "source_split": record["split"],
        "source_class": int(subset.labels[0]),
        "source_g": record["g"],
    }
    return dataset, subset, record, metadata


def test_component_operators_reconstruct_repository_hamiltonian():
    components = tfim_components(4, J=1.0, boundary="open")
    for operator in components.values():
        assert np.allclose(operator, operator.conj().T)
    assert np.allclose(components["interaction"] + 0.73 * components["field"], tfim_hamiltonian(4, 1.0, 0.73, "open"))


def test_generation_is_deterministic_normalized_nontrivial_and_provenanced():
    _, subset, _, metadata = _source()
    field = tfim_components(4)["field"]
    first, first_meta = augment_state(subset.states[0], metadata, field, operator="field", epsilon=0.2, seed=17)
    second, second_meta = augment_state(subset.states[0], metadata, field, operator="field", epsilon=0.2, seed=17)
    assert np.allclose(first, second)
    assert first_meta == second_meta
    assert np.isclose(np.vdot(first, first), 1.0, atol=1e-12)
    assert abs(np.vdot(subset.states[0], first)) ** 2 < 1 - 1e-6
    assert first_meta == {
        "synthetic_id": first_meta["synthetic_id"],
        "source_state_id": metadata["source_state_id"],
        "source_dataset": None,
        "source_split": "train",
        "source_class": metadata["source_class"],
        "source_g": metadata["source_g"],
        "augmentation_method": "state_local_physics",
        "generator": "spectral_norm_normalized",
        "operator": "field",
        "perturbation_strength": first_meta["perturbation_strength"],
        "random_seed": 17,
    }


def test_epsilon_zero_is_identity_and_nontrain_sources_are_rejected():
    _, subset, _, metadata = _source()
    state, _ = augment_state(subset.states[0], metadata, tfim_components(4)["interaction"], operator="interaction", epsilon=0, seed=5)
    assert np.allclose(state, subset.states[0], atol=1e-12)
    with pytest.raises(ValueError, match="train split"):
        augment_state(subset.states[0], {**metadata, "source_split": "val"}, tfim_components(4)["field"], operator="field", epsilon=0.1, seed=5)


def test_full_h_is_global_phase_but_components_move_ground_state():
    _, subset, record, metadata = _source()
    full = tfim_hamiltonian(4, 1.0, record["g"], "open")
    full_state, _ = augment_state(subset.states[0], metadata, full, operator="full_h", epsilon=0.4, seed=1)
    component_state, _ = augment_state(subset.states[0], metadata, tfim_components(4)["field"], operator="field", epsilon=0.4, seed=1)
    assert np.isclose(abs(np.vdot(subset.states[0], full_state)) ** 2, 1.0, atol=1e-12)
    assert abs(np.vdot(subset.states[0], component_state)) ** 2 < 1 - 1e-5


def test_diagnostics_and_train_fitted_acceptance_gate():
    _, subset, record, _ = _source()
    records = {item["parameter_id"]: item for item in load_tfim_dataset("data/tfim_4q_random").manifest["records"]}
    label = int(subset.labels[0])
    mask = subset.labels == label
    g_values = np.asarray([records[str(source_id)]["g"] for source_id in subset.parameter_ids[mask]])
    gate = fit_acceptance_gate(subset.states[mask], g_values, n_qubits=4, J=1.0, boundary="open")
    diagnostics = state_diagnostics(subset.states[0], subset.states[0], subset.states, subset.labels, label, record["g"], 4, 1.0, "open")
    assert diagnostics["source_fidelity"] == pytest.approx(1.0)
    assert diagnostics["nearest_same_class_train_fidelity"] == pytest.approx(1.0)
    assert diagnostics["energy_drift"] == pytest.approx(0.0)
    assert accepted(diagnostics, gate)
    assert not accepted({**diagnostics, "magnetization_x": gate.mx_range[1] + 1}, gate)


def test_diagnostic_artifacts_round_trip(tmp_path):
    config = {
        "datasets": {"random": "data/tfim_4q_random"},
        "real_states_per_class": 2,
        "subset_seed": 31415,
        "operators": ["field", "interaction"],
        "epsilon_sweep": [0.0, 0.1],
        "generation_seed": 7000,
        "augmentation_ratios": [0, 0.5],
    }
    result, _ = run_diagnostics(config, tmp_path)
    assert json.loads((tmp_path / "diagnostics.json").read_text())["git_sha"] == result["git_sha"]
    assert (tmp_path / "per_sample_diagnostics.csv").read_text().startswith("dataset,epsilon,synthetic_id")
    with np.load(tmp_path / "synthetic_states.npz") as states:
        assert len(states.files) == 16
        assert all(states[name].shape == (16,) for name in states.files)

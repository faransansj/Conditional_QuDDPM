import json

import numpy as np

from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.qcnn_baseline import run_baseline
from conditional_quddpm.models.qcnn import (
    CNOT,
    X,
    Y,
    Z,
    _euler,
    _interaction,
    qcnn_expectation,
    qcnn_state,
    train_qcnn_spsa,
)


def _assert_unitary(matrix):
    assert np.allclose(matrix.conj().T @ matrix, np.eye(len(matrix)), atol=1e-12)


def _explicit_apply(state, gate, wires):
    """Independent basis-index implementation used as a simulator cross-check."""
    output = np.zeros_like(state)
    for column, amplitude in enumerate(state):
        input_bits = [(column >> (3 - wire)) & 1 for wire in wires]
        gate_column = sum(bit << (len(wires) - 1 - index) for index, bit in enumerate(input_bits))
        for gate_row in range(2 ** len(wires)):
            row = column
            for index, wire in enumerate(wires):
                bit = (gate_row >> (len(wires) - 1 - index)) & 1
                mask = 1 << (3 - wire)
                row = row | mask if bit else row & ~mask
            output[row] += gate[gate_row, gate_column] * amplitude
    return output


def _reference_expectation(state, parameters):
    output = state.copy()

    def convolution(wires, values):
        nonlocal output
        output = _explicit_apply(output, _euler(values[0:3]), (wires[0],))
        output = _explicit_apply(output, _euler(values[3:6]), (wires[1],))
        for pauli, angle in zip((X, Y, Z), values[6:9], strict=True):
            output = _explicit_apply(output, _interaction(pauli, angle), wires)
        output = _explicit_apply(output, _euler(values[9:12]), (wires[0],))
        output = _explicit_apply(output, _euler(values[12:15]), (wires[1],))

    def pool(source, sink, values):
        nonlocal output
        sink_basis = _euler(values[3:6])
        output = _explicit_apply(output, _euler(values[0:3]), (source,))
        output = _explicit_apply(output, sink_basis, (sink,))
        output = _explicit_apply(output, CNOT, (source, sink))
        output = _explicit_apply(output, sink_basis.conj().T, (sink,))

    for wires in ((0, 1), (2, 3), (1, 2), (3, 0)):
        convolution(wires, parameters[0:15])
    for source, sink in ((0, 2), (1, 3)):
        pool(source, sink, parameters[15:21])
    convolution((2, 3), parameters[21:36])
    pool(2, 3, parameters[36:42])
    probabilities = np.abs(output.reshape(2, 2, 2, 2)) ** 2
    return float(probabilities[:, :, :, 0].sum() - probabilities[:, :, :, 1].sum())


def test_qcnn_gates_are_unitary():
    _assert_unitary(CNOT)
    _assert_unitary(_euler(np.array([0.2, -0.4, 0.7])))
    for pauli in (X, Y, Z):
        _assert_unitary(_interaction(pauli, 0.37))


def test_qcnn_preserves_norm_and_matches_independent_reference():
    rng = np.random.default_rng(91)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    parameters = rng.normal(size=42)
    assert np.isclose(np.linalg.norm(qcnn_state(state, parameters)), 1.0, atol=1e-12)
    assert np.isclose(qcnn_expectation(state, parameters), _reference_expectation(state, parameters), atol=1e-12)


def test_zero_parameter_basis_states_measure_parity():
    expectations = [qcnn_expectation(np.eye(16, dtype=complex)[index], np.zeros(42)) for index in range(16)]
    expected = [1 if index.bit_count() % 2 == 0 else -1 for index in range(16)]
    assert np.allclose(expectations, expected)


def test_fixed_parameter_golden_output():
    rng = np.random.default_rng(12345)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    parameters = np.linspace(-0.7, 0.9, 42)
    assert np.isclose(qcnn_expectation(state, parameters), 0.2019068564919677, atol=1e-14)


def test_manifest_drives_leakage_free_dataset_loading():
    dataset = load_tfim_dataset("data/tfim_4q_random")
    manifest = dataset.manifest["records"]
    expected = {
        split: [record["parameter_id"] for record in manifest if record["split"] == split]
        for split in ("train", "val", "test")
    }
    assert dataset.train.parameter_ids.tolist() == expected["train"]
    assert dataset.val.parameter_ids.tolist() == expected["val"]
    assert dataset.test.parameter_ids.tolist() == expected["test"]
    assert set(expected["train"]).isdisjoint(expected["val"] + expected["test"])


def test_training_subsets_are_balanced_and_nested():
    train = load_tfim_dataset("data/tfim_4q_random").train
    subsets = nested_train_subsets(train, [10, 25, 50, 100], seed=7)
    previous = set()
    for size, subset in subsets.items():
        assert np.bincount(subset.labels, minlength=2).tolist() == [size, size]
        current = set(subset.parameter_ids.tolist())
        assert previous.issubset(current)
        previous = current


def test_qcnn_expectation_and_training_are_deterministic():
    dataset = load_tfim_dataset("data/tfim_4q_random")
    subset = nested_train_subsets(dataset.train, [2], seed=1)[2]
    parameters = np.zeros(42)
    expectation = qcnn_expectation(subset.states[0], parameters)
    assert -1.0 <= expectation <= 1.0

    kwargs = dict(
        init_seed=3,
        spsa_seed=5,
        steps=1,
        learning_rate=0.1,
        perturbation=0.1,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-6,
    )
    first = train_qcnn_spsa(subset.states, subset.labels, dataset.val.states[:4], dataset.val.labels[:4], **kwargs)
    second = train_qcnn_spsa(subset.states, subset.labels, dataset.val.states[:4], dataset.val.labels[:4], **kwargs)
    assert np.allclose(first.parameters, second.parameters)
    assert first.history == second.history


def test_smoke_experiment_writes_comparable_artifacts(tmp_path):
    config = {
        "datasets": {"random": "data/tfim_4q_random", "blocked": "data/tfim_4q_blocked"},
        "real_states_per_class": [2],
        "subset_seed": 4,
        "seeds": [{"run_seed": 0, "init_seed": 3, "spsa_seed": 5}],
        "evaluate_test": True,
        "training": {
            "optimizer": "spsa",
            "steps": 1,
            "learning_rate": 0.1,
            "perturbation": 0.1,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 1e-6,
        },
    }
    summary = run_baseline(config, tmp_path)
    assert summary["completed_runs"] == 2
    assert summary["failure_rate"] == 0.0
    assert summary["evaluation_split"] == "test"
    assert all("accuracy_std" in aggregate for aggregate in summary["aggregates"])
    assert all("macro_f1_std" in aggregate for aggregate in summary["aggregates"])
    assert all("loss_std" in aggregate for aggregate in summary["aggregates"])
    assert all(aggregate["majority_baseline_accuracy"] == 0.5 for aggregate in summary["aggregates"])
    for dataset in ("random", "blocked"):
        path = tmp_path / dataset / "real-2" / "seed-0" / "metrics.json"
        metrics = json.loads(path.read_text())
        assert metrics["method"] == "real_only"
        assert metrics["status"] == "completed"
        assert metrics["test"]["samples"] == 60
        assert metrics["git_sha"]
        assert set(metrics["dataset_checksums"]) == {"states.npz", "split_manifest.json", "validation.json"}
        assert metrics["init_seed"] == 3
        assert metrics["spsa_seed"] == 5
        assert len(metrics["train_parameter_ids"]) == 4


def test_validation_only_tuning_never_evaluates_test(tmp_path):
    config = {
        "datasets": {"random": "data/tfim_4q_random"},
        "real_states_per_class": [2],
        "subset_seed": 4,
        "seeds": [{"run_seed": 0, "init_seed": 3, "spsa_seed": 5}],
        "evaluate_test": False,
        "training": {
            "optimizer": "spsa",
            "steps": 0,
            "learning_rate": 0.1,
            "perturbation": 0.1,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 1e-6,
        },
    }
    summary = run_baseline(config, tmp_path)
    metrics = json.loads((tmp_path / "random" / "real-2" / "seed-0" / "metrics.json").read_text())
    assert metrics["test"] is None
    assert summary["evaluation_split"] == "validation"

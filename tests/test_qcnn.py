import json

import numpy as np

from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.qcnn_baseline import run_baseline
from conditional_quddpm.models.qcnn import qcnn_expectation, train_qcnn_spsa


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

    kwargs = dict(seed=3, steps=1, learning_rate=0.1, perturbation=0.1)
    first = train_qcnn_spsa(subset.states, subset.labels, dataset.val.states[:4], dataset.val.labels[:4], **kwargs)
    second = train_qcnn_spsa(subset.states, subset.labels, dataset.val.states[:4], dataset.val.labels[:4], **kwargs)
    assert np.allclose(first.parameters, second.parameters)
    assert first.history == second.history


def test_smoke_experiment_writes_comparable_artifacts(tmp_path):
    config = {
        "datasets": {"random": "data/tfim_4q_random", "blocked": "data/tfim_4q_blocked"},
        "real_states_per_class": [2],
        "subset_seed": 4,
        "model_seeds": [0],
        "training": {"optimizer": "spsa", "steps": 1, "learning_rate": 0.1, "perturbation": 0.1},
    }
    summary = run_baseline(config, tmp_path)
    assert summary["runs"] == 2
    for dataset in ("random", "blocked"):
        path = tmp_path / dataset / "real-2" / "seed-0" / "metrics.json"
        metrics = json.loads(path.read_text())
        assert metrics["method"] == "real_only"
        assert metrics["test"]["samples"] == 60
        assert len(metrics["train_parameter_ids"]) == 4

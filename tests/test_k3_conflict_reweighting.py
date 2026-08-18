import hashlib
from pathlib import Path

import numpy as np
import pytest

from conditional_quddpm.datasets.loader import QuantumSplit
from conditional_quddpm.experiments import k3_conflict_reweighting as k3


def test_uniform_reconstructs_k2_global_and_preserves_parameter_order():
    with np.load("results/quddpm_kernel_diagnostics/k2_realization/per_realization_gradients.npz") as data:
        tasks = data["realization_raw"]
        expected = data["global_raw"]
        assert tasks.shape[1:] == expected.shape == (81,)
        actual = k3.aggregate_gradient(tasks, np.full(len(tasks), 1 / len(tasks)))
    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-8)


def test_weight_methods_are_nonnegative_normalized_deterministic_and_have_valid_class_totals():
    vectors = np.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., 0.]])
    first, scores, alignment = k3.build_methods(vectors, [1., 1.], [0.5, 1., 2.], 0.05, 1e-10)
    second = k3.build_methods(vectors, [1., 1.], [0.5, 1., 2.], 0.05, 1e-10)[0]
    assert set(first) == set(second) and np.all(np.isfinite(scores)) and np.all(np.isfinite(alignment))
    classes = np.asarray([0, 0, 1, 1])
    for name, weights in first.items():
        assert np.allclose(weights, second[name])
        assert np.all(weights >= 0) and weights.sum() == pytest.approx(1)
        stats = k3._weight_stats(weights, classes)
        assert stats["class_0_total"] + stats["class_1_total"] == pytest.approx(1)


def test_effective_sample_size_weighted_aggregate_and_cancellation_reference():
    weights = np.asarray([0.5, 0.25, 0.25])
    vectors = np.asarray([[1., 0.], [0., 2.], [-1., 1.]])
    expected = (weights[:, None] * vectors).sum(axis=0)
    assert k3.effective_sample_size(weights) == pytest.approx(1 / np.sum(weights**2))
    assert np.allclose(k3.aggregate_gradient(vectors, weights), expected)
    expected_c = np.linalg.norm(expected) / np.sum(weights * np.linalg.norm(vectors, axis=1))
    assert k3.weighted_cancellation(vectors, weights) == pytest.approx(expected_c)
    with pytest.raises(ValueError): k3.aggregate_gradient(vectors, [1, 0, 1])


def test_zero_gradients_do_not_produce_nan_or_weight_collapse():
    vectors = np.zeros((4, 3))
    methods, scores, alignment = k3.build_methods(vectors, np.zeros(3), [1.0], 0.05, 1e-10)
    assert np.all(scores == 0) and np.all(alignment == 0)
    assert np.allclose(methods["conflict_tau_1p0"], 0.25)
    assert np.allclose(methods["physics_conflict_tau_1p0"], 0.25)
    assert k3.weighted_cancellation(vectors, methods["uniform"]) is None


def test_frozen_context_uses_train_only_loader(monkeypatch):
    split = QuantumSplit(np.ones((2, 16), dtype=complex), np.asarray([0, 1]), np.asarray(["a", "b"]))
    calls = []
    monkeypatch.setattr(k3.k2, "load_train_split", lambda path: (calls.append(path) or split, {}))
    monkeypatch.setattr(k3.k2, "nested_train_subsets", lambda train, sizes, seed: {1: train})
    fake_path = {c: [np.ones((1, 16), dtype=complex)] * 3 for c in (0, 1)}
    monkeypatch.setattr(k3, "trajectories", lambda states, n, steps, seed: (fake_path, {}))
    parameters = np.zeros((3, 27))
    monkeypatch.setattr(k3, "train", lambda path, config, objective: (None, None, [{"best": parameters}, {"best": parameters}]))
    config = {"k2_config": "configs/quddpm/kernel_k2.yaml", "step": 1, "checkpoint": "best"}
    actual = k3.frozen_context(config)[0]
    assert calls == ["data/tfim_4q_random"] and actual is parameters


def test_classification_requires_pareto_improvement_over_global_and_checks_collapse():
    baseline = {"method": "global_mmd", "global_mmd_delta": -0.01, "aggregate_physics_delta": -0.01}
    rdm = {"method": "2rdm", "global_mmd_delta": 0.0, "aggregate_physics_delta": -0.02}
    weaker = {"method": "conflict_tau_1p0", "global_mmd_delta": -0.005, "aggregate_physics_delta": -0.005}
    weights = [{"method": "conflict_tau_1p0", "effective_sample_size": 8.0}]
    assert k3.classify([baseline, rdm, weaker], weights, 8) == "K3-B"
    stronger = {**weaker, "global_mmd_delta": -0.02, "aggregate_physics_delta": -0.02}
    assert k3.classify([baseline, rdm, stronger], weights, 8) == "K3-A"
    weights[0]["effective_sample_size"] = 2.0
    assert k3.classify([baseline, rdm, stronger], weights, 8) == "K3-C"
    other = {**stronger, "method": "physics_conflict_tau_1p0"}
    mixed = [*weights, {"method": other["method"], "effective_sample_size": 8.0}]
    assert k3.classify([baseline, rdm, stronger, other], mixed, 8) == "K3-A"


def test_k2_artifacts_are_read_only_during_weight_calculation():
    root = Path("results/quddpm_kernel_diagnostics/k2_realization")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    with np.load(root / "per_realization_gradients.npz") as data:
        k3.build_methods(data["realization_raw"], data["physics_raw"], [1.0], 0.05, 1e-10)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    assert before == after

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conditional_quddpm.experiments import k2_realization_conflict as k2
from conditional_quddpm.models.quddpm import condition_angles, haar_states, reverse_parameter_count
from conditional_quddpm.models.rdm_kernels import kernel_mmd_raw


def _problem(seed=8, n=2, direction_count=3):
    path = {c: [haar_states(n, seed + 20*c + t, 4) for t in range(3)] for c in (0, 1)}
    angles = condition_angles([0, 1])
    uniforms = {c: np.random.default_rng(seed + 100 + c).random((2, n)) for c in (0, 1)}
    parameters = np.random.default_rng(seed + 200).normal(0, 0.15, (1, reverse_parameter_count(4, 1)))
    rng = np.random.default_rng(seed + 300)
    directions = [rng.choice((-1.0, 1.0), size=parameters.shape) for _ in range(direction_count)]
    metadata = [{"realization_id": f"c{c}-r{i}", "state_id": f"c{c}", "class": c, "g": float(c),
                 "split": "train", "forward_seed": seed+i+100*c, "measurement_seed": seed+100+c}
                for c in (0, 1) for i in range(n)]
    return path, angles, uniforms, parameters, directions, metadata


def _config(seed=301, directions=2):
    return {"schema_version": k2.SCHEMA_VERSION, "dataset": "data/tfim_4q_random", "subset_seed": 31415,
            "train_states_per_class": 1, "train_realizations": 2, "diffusion_steps": 2, "layers": 1,
            "iterations": 0, "ancillas": 1, "measurement_outcomes": 1,
            "trained_objective": "PER_OUTCOME_ENSEMBLE_MMD", "checkpoint": "best", "step": 1,
            "directions": directions, "epsilon": 0.15, "directional_step": 0.15,
            "near_zero_threshold": 1e-10, "reconstruction": {"atol": 1e-10, "rtol": 1e-8},
            "spsa": {"learning_rate": 0.5, "perturbation": 0.15},
            "seeds": {"train_forward": 121, "init": 140, "spsa": 150, "measurement": 160,
                      "scale_probe": 40160, "permutation": 50160, "directions": seed}}


def test_additive_mmd_rows_reconstruct_raw_biased_estimator():
    generated = haar_states(3, 4, 4)
    target = haar_states(3, 5, 4)
    matrix, rows = k2.mmd_contributions(generated, target)
    assert matrix.sum() == pytest.approx(kernel_mmd_raw(generated, target, "global"), abs=1e-14)
    assert rows.sum() == pytest.approx(matrix.sum(), abs=1e-14)
    permutation = [2, 0, 1]
    permuted_matrix, permuted_rows = k2.mmd_contributions(generated[permutation], target[permutation])
    assert permuted_matrix.sum() == pytest.approx(matrix.sum())
    assert np.allclose(permuted_rows, rows[permutation])


def test_gradient_and_class_reconstruction_parameter_order_and_determinism():
    path, angles, uniforms, parameters, directions, metadata = _problem()
    first = k2.analyze(parameters, path, 0, angles, uniforms, directions, _config(directions=len(directions)), metadata)
    second = k2.analyze(parameters, path, 0, angles, uniforms, directions, _config(directions=len(directions)), metadata)
    assert first["reconstruction"] == second["reconstruction"]
    assert first["reconstruction"]["directional_all_pass"]
    assert first["reconstruction"]["global_relative_error"] < 1e-12
    assert max(first["reconstruction"]["class_relative_error"]) < 1e-12
    vectors = [first["global"], *first["classes"], *first["tasks"], first["2rdm"], first["physics"]]
    assert all(record["raw"].shape == (parameters.size,) for record in vectors)
    assert np.allclose(np.mean([r["raw"] for r in first["tasks"]], axis=0), first["global"]["raw"])
    n = 2
    for c in (0, 1):
        assert np.allclose(np.mean([first["tasks"][c*n+i]["raw"] for i in range(n)], axis=0), first["classes"][c]["raw"])


def test_safe_cosine_and_pairwise_grouping():
    assert k2.safe_cosine([0, 0], [1, 0])["status"] == "left_near_zero"
    vectors = np.asarray([[1, 0], [1, 1], [-1, 0], [-1, 1]], dtype=float)
    metadata = [{"realization_id": str(i), "class": i // 2, "g": float(i // 2)} for i in range(4)]
    matrix, rows, stats = k2.pairwise_analysis(vectors, metadata, 1e-12)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.allclose(np.diag(matrix), 1.0)
    assert stats["within_class"]["count"] == 2
    assert stats["between_class"]["count"] == 4
    assert len(rows) == 6


def test_schema_validation_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        k2.validate_required({"schema_version": k2.SCHEMA_VERSION}, k2.REQUIRED_MANIFEST, "manifest")


def test_run_is_reproducible_records_seed_and_never_materializes_nontrain_states(tmp_path, monkeypatch):
    dataset_path = Path("data/tfim_4q_random")
    manifest = json.loads((dataset_path / "split_manifest.json").read_text())
    original_np_load = np.load
    with original_np_load(dataset_path / "states.npz") as data:
        ids = data["parameter_ids"].tolist()
    allowed = {ids.index(record["parameter_id"]) for record in manifest["records"] if record["split"] == "train"}
    original_rows = k2._read_state_rows
    def guarded_rows(path, indices):
        assert set(indices) <= allowed
        return original_rows(path, indices)
    class MetadataOnlyNPZ:
        def __init__(self, loaded): self.loaded = loaded
        def __enter__(self): return self
        def __exit__(self, *args): self.loaded.close()
        def __getitem__(self, key):
            if key == "states": raise AssertionError("full state array materialized")
            return self.loaded[key]
    monkeypatch.setattr(k2, "_read_state_rows", guarded_rows)
    monkeypatch.setattr(k2.np, "load", lambda path: MetadataOnlyNPZ(original_np_load(path)))
    historical = [p for root in (Path("results/quddpm_kernel_diagnostics/frozen"), Path("results/quddpm_kernel_diagnostics/k1_gradient"))
                  for p in root.glob("*") if p.is_file()]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in historical}
    first = k2.run(_config(seed=301, directions=1), tmp_path / "first")
    second = k2.run(_config(seed=301, directions=1), tmp_path / "second")
    different = k2.run(_config(seed=302, directions=1), tmp_path / "different")
    assert first["summary"] == second["summary"]
    assert first["manifest"]["seeds"]["directions"] == 301
    assert different["manifest"]["seeds"]["directions"] == 302
    assert first["manifest"]["config_hash"] != different["manifest"]["config_hash"]
    assert first["manifest"]["test_split_used"] is False and first["manifest"]["validation_split_used"] is False
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in historical}
    expected = {"config.yaml", "run_manifest.json", "gradient_provenance.json", "per_realization_gradients.npz",
                "per_realization_summary.csv", "pairwise_cosine_matrix.npz", "pairwise_cosines.csv",
                "directional_probe_results.csv", "directional_reconstruction.csv", "k2_summary.json", "report.md"}
    assert expected == {p.name for p in (tmp_path / "first").iterdir()}

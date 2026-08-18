import json
import numpy as np
import pytest
import yaml
from pathlib import Path
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.final_support_diagnostic import aggregate, categorize, decide, run, train_oracle
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.models.quddpm import reverse_parameter_count


def tiny_config(tmp_path):
    config = yaml.safe_load(Path("configs/quddpm/final_support_diagnostic.yaml").read_text())
    config.update({"iterations": 4, "holdout_realizations": 2, "validation_realizations": 2, "evaluation_outcomes": 2})
    return config


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("final_support")
    return run(tiny_config(out), out), out


def test_train_and_holdout_realizations_are_disjoint():
    ds = load_tfim_dataset("data/tfim_4q_random")
    split = nested_train_subsets(ds.train, [1], 31415)[1]
    states = {c: split.states[split.labels == c] for c in (0, 1)}
    train_path, train_ids = trajectories(states, 4, 2, 121)
    holdout_path, holdout_ids = trajectories(states, 4, 2, 10121)
    train_seeds = {r["forward_seed"] for ids in train_ids.values() for r in ids}
    holdout_seeds = {r["forward_seed"] for ids in holdout_ids.values() for r in ids}
    assert train_seeds.isdisjoint(holdout_seeds)
    for c in (0, 1):
        assert np.allclose(train_path[c][0], holdout_path[c][0])  # same initial state by construction
        for t in (1, 2):  # diffused states must differ (independent forward noise)
            assert not np.allclose(train_path[c][t], holdout_path[c][t])


def test_oracle_training_is_deterministic():
    config = tiny_config(None)
    rng = np.random.default_rng(0)
    shape = (config["layers"], reverse_parameter_count(4, config["ancillas"]))
    initial = rng.normal(0, 0.15, shape)
    source = rng.normal(0, 1, (1, 16)); source /= np.linalg.norm(source, axis=1, keepdims=True)
    target = rng.normal(0, 1, (1, 16)); target /= np.linalg.norm(target, axis=1, keepdims=True)
    uniforms = rng.random((2, 1))
    first = train_oracle(initial, source, target, 0.3, uniforms, config, 70160)
    second = train_oracle(initial, source, target, 0.3, uniforms, config, 70160)
    assert np.array_equal(first["best"], second["best"])
    assert first["history"] == second["history"]


def test_categorize_covers_all_preregistered_branches():
    t = {"seen_fit_ratio_max": 0.5, "min_loss_improvement": 0.2, "unseen_advantage_min": 0.05}
    def summary(seen_ratio, loss_improvement, unseen_advantage):
        return {"shared_seen_mmd": {"mean": 1.0}, "oracle_seen_mmd": {"mean": seen_ratio},
                "oracle_mean_loss_improvement": loss_improvement, "unseen_advantage": unseen_advantage}
    assert categorize(summary(0.8, 0.9, 0.5), t) == "FUNDAMENTAL REVERSE-LEARNING BOTTLENECK"
    assert categorize(summary(0.2, 0.1, 0.5), t) == "FUNDAMENTAL REVERSE-LEARNING BOTTLENECK"
    assert categorize(summary(0.2, 0.9, 0.2), t) == "SHARED SUPPORT BOTTLENECK"
    assert categorize(summary(0.2, 0.9, -0.1), t) == "TRAJECTORY MEMORIZATION / SUPPORT GENERALIZATION FAILURE"
    step_summaries = {1: {"all": summary(0.2, 0.9, 0.2)}, 2: {"all": summary(0.2, 0.9, -0.1)}}
    assert decide(step_summaries, t)["diagnosis"] == "INCONCLUSIVE"
    step_summaries[2] = step_summaries[1]
    assert decide(step_summaries, t)["diagnosis"] == "SHARED SUPPORT BOTTLENECK"


def test_aggregate_mean_std_are_correct():
    cells = [{"mmd": 1.0}, {"mmd": 2.0}, {"mmd": 4.0}]
    result = aggregate(cells, "mmd")
    assert result["mean"] == pytest.approx(7.0 / 3)
    assert result["std"] == pytest.approx(float(np.std([1.0, 2.0, 4.0])))


def test_tiny_run_integrity(tiny_run):
    result, out = tiny_run
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["test_split_used"] is False
    assert manifest["train_ids"] == ["class-0-00125", "class-1-00034"]
    assert set(manifest["realization_ids"]) == {"train", "holdout", "validation"}  # no test-pool entry
    train_seeds = {r["forward_seed"] for ids in manifest["realization_ids"]["train"].values() for r in ids}
    holdout_seeds = {r["forward_seed"] for ids in manifest["realization_ids"]["holdout"].values() for r in ids}
    assert train_seeds.isdisjoint(holdout_seeds)
    # exact realization provenance: every oracle run records its training realization
    assert len(result["oracle_runs"]) == 2 * 2 * 4
    for key, oracle in result["oracle_runs"].items():
        step, c, i = (int(x) for x in key.split("_"))
        assert oracle["train_realization_id"] == manifest["realization_ids"]["train"][str(c)][i]
        assert oracle["spsa_seed"] == manifest["seeds"]["oracle_spsa"] + 1000 * (step) + 10 * c + i
    # configuration equivalence except realization scope
    equiv = manifest["configuration_equivalence"]
    assert equiv["oracle_init"] == "shared model step-initial parameters"
    shared_model = manifest["model"]
    assert (shared_model["T"], shared_model["L"], shared_model["iterations"]) == (2, 3, 4)
    # metric aggregation: overall mean equals mean of the 8 cells
    cells = result["cells"]
    assert len(cells) == 16
    for step in (1, 2):
        subset = [x for x in cells if x["step"] == step]
        summary = result["step_summaries"][str(step)]["all"]
        assert summary["oracle_seen_mmd"]["mean"] == pytest.approx(float(np.mean([x["oracle_seen_mmd"] for x in subset])))
        assert summary["unseen_advantage"] == pytest.approx(float(np.mean([x["shared_unseen_mmd"] - x["oracle_unseen_mmd"] for x in subset])))
        class0 = [x for x in subset if x["class"] == 0]
        assert result["step_summaries"][str(step)]["class_0"]["shared_gap_mmd"] == pytest.approx(
            float(np.mean([x["shared_unseen_mmd"] - x["shared_seen_mmd"] for x in class0])))
    # deterministic rerun of one cell's evaluation values is implied by seeded eval; check finiteness
    for cell in cells:
        for key in ("shared_seen_mmd", "oracle_unseen_mmd", "oracle_unseen_physics_error"):
            assert np.isfinite(cell[key])


def test_tiny_run_is_reproducible(tiny_run, tmp_path):
    _, out = tiny_run
    rerun_out = tmp_path / "rerun"
    second = run(tiny_config(rerun_out), rerun_out)
    first_metrics = json.loads((out / "metrics.json").read_text())
    assert second["cells"] == first_metrics["cells"]
    assert second["decision"] == first_metrics["decision"]

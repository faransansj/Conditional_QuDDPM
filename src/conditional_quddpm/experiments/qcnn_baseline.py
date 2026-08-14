"""Config-driven real-only QCNN baseline experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import yaml

from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.models.qcnn import metrics, predict_expectations, train_qcnn_spsa


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        true_positive = int(np.sum((labels == label) & (predictions == label)))
        false_positive = int(np.sum((labels != label) & (predictions == label)))
        false_negative = int(np.sum((labels == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _evaluate(states: np.ndarray, labels: np.ndarray, parameters: np.ndarray) -> dict[str, float | int]:
    base = metrics(states, labels, parameters)
    predictions = (predict_expectations(states, parameters) >= 0).astype(np.int8)
    return {**base, "macro_f1": _macro_f1(labels, predictions), "samples": len(labels)}


def _git_provenance() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return sha, dirty


def _seed_specs(config: dict) -> list[dict[str, int]]:
    if "seeds" in config:
        return config["seeds"]
    return [
        {"run_seed": seed, "init_seed": seed, "spsa_seed": seed}
        for seed in config.get("model_seeds", [])
    ]


def _aggregate(runs: list[dict], dataset_name: str, size: int, metric_split: str) -> dict:
    selected = [
        run for run in runs
        if run["status"] == "completed"
        and run["dataset"] == dataset_name
        and run["real_states_per_class"] == size
    ]
    per_seed = [
        {
            "run_seed": run["run_seed"],
            "init_seed": run["init_seed"],
            "spsa_seed": run["spsa_seed"],
            **run[metric_split],
        }
        for run in selected
    ]
    result = {
        "dataset": dataset_name,
        "method": "real_only",
        "real_states_per_class": size,
        "evaluation_split": metric_split,
        "completed_runs": len(selected),
        "per_seed": per_seed,
        "majority_baseline_accuracy": 0.5,
        "random_baseline_accuracy": 0.5,
    }
    for metric in ("accuracy", "macro_f1", "loss"):
        values = [entry[metric] for entry in per_seed]
        result[f"{metric}_mean"] = float(np.mean(values)) if values else None
        result[f"{metric}_std"] = float(np.std(values)) if values else None
    return result


def run_baseline(config: dict, output: str | Path) -> dict:
    """Run every configured dataset, nested real-data size, and paired seed spec."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = json.loads(json.dumps(config))
    (output / "config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=True))
    git_sha, git_dirty = _git_provenance()
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    all_runs: list[dict] = []
    failed_runs: list[dict] = []
    evaluate_test = config.get("evaluate_test", True)
    seed_specs = _seed_specs(config)

    for dataset_name, dataset_path in config["datasets"].items():
        dataset_path = Path(dataset_path)
        dataset_checksums = json.loads((dataset_path / "checksums.json").read_text())
        dataset = load_tfim_dataset(dataset_path)
        subsets = nested_train_subsets(dataset.train, config["real_states_per_class"], config["subset_seed"])
        for size, subset in subsets.items():
            for seed_spec in seed_specs:
                run_payload = {
                    "method": "real_only",
                    "architecture": "tfq-inspired-qcnn-4q-42p",
                    "dataset": dataset_name,
                    "dataset_checksums": dataset_checksums,
                    "real_states_per_class": size,
                    "train_parameter_ids": subset.parameter_ids.tolist(),
                    "subset_seed": config["subset_seed"],
                    **seed_spec,
                    "training": config["training"],
                    "git_sha": git_sha,
                }
                run_id = hashlib.sha256(
                    json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:16]
                run_dir = output / dataset_name / f"real-{size}" / f"seed-{seed_spec['run_seed']}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "resolved_config.json").write_text(
                    json.dumps(run_payload, indent=2, sort_keys=True) + "\n"
                )
                try:
                    training = config["training"]
                    result = train_qcnn_spsa(
                        subset.states,
                        subset.labels,
                        dataset.val.states,
                        dataset.val.labels,
                        init_seed=seed_spec["init_seed"],
                        spsa_seed=seed_spec["spsa_seed"],
                        steps=training["steps"],
                        learning_rate=training["learning_rate"],
                        perturbation=training["perturbation"],
                        early_stopping_patience=training["early_stopping_patience"],
                        early_stopping_min_delta=training["early_stopping_min_delta"],
                    )
                    np.save(run_dir / "parameters.npy", result.parameters)
                    (run_dir / "history.json").write_text(json.dumps(result.history, indent=2) + "\n")
                    run_metrics = {
                        **run_payload,
                        "run_id": run_id,
                        "status": "completed",
                        "git_dirty": git_dirty,
                        "versions": versions,
                        "dataset_path": str(dataset_path),
                        "split_strategy": dataset.manifest["config"].get("split_strategy", "random"),
                        "best_step": result.best_step,
                        "steps_completed": result.history[-1]["step"],
                        "stopped_early": result.stopped_early,
                        "train": _evaluate(subset.states, subset.labels, result.parameters),
                        "validation": _evaluate(dataset.val.states, dataset.val.labels, result.parameters),
                        "test": _evaluate(dataset.test.states, dataset.test.labels, result.parameters) if evaluate_test else None,
                    }
                    (run_dir / "metrics.json").write_text(
                        json.dumps(run_metrics, indent=2, sort_keys=True) + "\n"
                    )
                    all_runs.append(run_metrics)
                except Exception as error:
                    failure = {
                        **run_payload,
                        "run_id": run_id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    (run_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                    failed_runs.append(failure)

    metric_split = "test" if evaluate_test else "validation"
    aggregates = [
        _aggregate(all_runs, dataset_name, size, metric_split)
        for dataset_name in config["datasets"]
        for size in sorted(set(config["real_states_per_class"]))
    ]
    planned_runs = len(config["datasets"]) * len(set(config["real_states_per_class"])) * len(seed_specs)
    summary = {
        "status": "completed" if not failed_runs else "completed_with_failures",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "versions": versions,
        "evaluation_split": metric_split,
        "planned_runs": planned_runs,
        "completed_runs": len(all_runs),
        "failed_runs": len(failed_runs),
        "failure_rate": len(failed_runs) / planned_runs,
        "failures": failed_runs,
        "aggregates": aggregates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-only 4-qubit QCNN baselines")
    parser.add_argument("--config", default="configs/qcnn/baseline_4q.yaml")
    parser.add_argument("--output", default="results/qcnn_baseline")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    summary = run_baseline(config, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

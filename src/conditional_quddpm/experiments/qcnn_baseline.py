"""Config-driven real-only QCNN baseline experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def run_baseline(config: dict, output: str | Path) -> dict:
    """Run every configured dataset, nested real-data size, and model seed."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    all_runs = []

    for dataset_name, dataset_path in config["datasets"].items():
        dataset = load_tfim_dataset(dataset_path)
        subsets = nested_train_subsets(dataset.train, config["real_states_per_class"], config["subset_seed"])
        for size, subset in subsets.items():
            for model_seed in config["model_seeds"]:
                result = train_qcnn_spsa(
                    subset.states,
                    subset.labels,
                    dataset.val.states,
                    dataset.val.labels,
                    seed=model_seed,
                    steps=config["training"]["steps"],
                    learning_rate=config["training"]["learning_rate"],
                    perturbation=config["training"]["perturbation"],
                )
                run_dir = output / dataset_name / f"real-{size}" / f"seed-{model_seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "parameters.npy", result.parameters)
                (run_dir / "history.json").write_text(json.dumps(result.history, indent=2) + "\n")
                run_metrics = {
                    "method": "real_only",
                    "architecture": "tfq-inspired-qcnn-4q-42p",
                    "dataset": dataset_name,
                    "dataset_path": str(dataset_path),
                    "split_strategy": dataset.manifest["config"].get("split_strategy", "random"),
                    "real_states_per_class": size,
                    "subset_seed": config["subset_seed"],
                    "model_seed": model_seed,
                    "best_step": result.best_step,
                    "train_parameter_ids": subset.parameter_ids.tolist(),
                    "train": _evaluate(subset.states, subset.labels, result.parameters),
                    "validation": _evaluate(dataset.val.states, dataset.val.labels, result.parameters),
                    "test": _evaluate(dataset.test.states, dataset.test.labels, result.parameters),
                }
                (run_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2, sort_keys=True) + "\n")
                all_runs.append(run_metrics)

    aggregates = []
    for dataset_name in config["datasets"]:
        for size in sorted(set(config["real_states_per_class"])):
            runs = [run for run in all_runs if run["dataset"] == dataset_name and run["real_states_per_class"] == size]
            aggregates.append({
                "dataset": dataset_name,
                "method": "real_only",
                "real_states_per_class": size,
                "seeds": len(runs),
                "test_accuracy_mean": float(np.mean([run["test"]["accuracy"] for run in runs])),
                "test_accuracy_std": float(np.std([run["test"]["accuracy"] for run in runs])),
                "test_macro_f1_mean": float(np.mean([run["test"]["macro_f1"] for run in runs])),
                "test_loss_mean": float(np.mean([run["test"]["loss"] for run in runs])),
            })
    summary = {"runs": len(all_runs), "aggregates": aggregates}
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

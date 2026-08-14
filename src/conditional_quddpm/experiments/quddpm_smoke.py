"""Functional validation for unconditional and conditional pure-state QuDDPM."""

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

from conditional_quddpm.models.quddpm import (
    bloch_z,
    fidelity_matrix,
    fidelity_mmd,
    generate_quddpm,
    haar_states,
    load_quddpm_checkpoint,
    pole_clusters,
    save_quddpm_checkpoint,
    train_stepwise_quddpm,
)


def _git_provenance() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return sha, dirty


def _distribution_metrics(generated: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> dict:
    norms = np.sum(np.abs(generated) ** 2, axis=1)
    pairwise = fidelity_matrix(generated, generated)
    off_diagonal = pairwise[~np.eye(len(generated), dtype=bool)]
    nearest_target = fidelity_matrix(generated, target).max(axis=1)
    generated_mmd = fidelity_mmd(generated, target)
    baseline_mmd = fidelity_mmd(baseline, target)
    return {
        "samples": len(generated),
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "min_purity": float(np.min(norms**2)),
        "max_purity": float(np.max(norms**2)),
        "generated_target_mmd": generated_mmd,
        "haar_target_mmd": baseline_mmd,
        "mmd_improvement": baseline_mmd - generated_mmd,
        "mean_pairwise_fidelity_off_diagonal": float(np.mean(off_diagonal)),
        "mean_nearest_target_fidelity": float(np.mean(nearest_target)),
        "max_nearest_target_fidelity": float(np.max(nearest_target)),
        "mean_bloch_z": float(np.mean(bloch_z(generated))),
    }


def _run_task(name: str, labels: list[int], config: dict, output: Path) -> dict:
    target_config = config["target"]
    model_config = config["model"]
    training_config = config["training"]
    seeds = config["seeds"]
    generation_config = config["generation"]
    targets = pole_clusters(target_config["samples"], labels, target_config["seed"], target_config["spread"])
    training_steps = training_config[f"{name}_steps"]
    result, forward = train_stepwise_quddpm(
        targets,
        diffusion_steps=model_config["diffusion_steps"],
        layers=model_config["layers"],
        samples=target_config["samples"],
        forward_seed=seeds["forward"],
        source_seed=seeds["source"],
        init_seed=seeds["init"],
        spsa_seed=seeds["spsa"],
        measurement_seed=seeds["measurement"],
        training_steps=training_steps,
        learning_rate=training_config["learning_rate"],
        perturbation=training_config["perturbation"],
    )
    generated = generate_quddpm(
        result,
        labels,
        generation_config["samples_per_class"],
        generation_config["source_seed"],
        generation_config["measurement_seed"],
    )
    task_dir = output / name
    task_dir.mkdir(parents=True, exist_ok=True)
    save_quddpm_checkpoint(task_dir / "checkpoint.npz", result)
    repeated = generate_quddpm(
        load_quddpm_checkpoint(task_dir / "checkpoint.npz"),
        labels,
        generation_config["samples_per_class"],
        generation_config["source_seed"],
        generation_config["measurement_seed"],
    )
    np.savez_compressed(
        task_dir / "generated_states.npz",
        **{f"class_{label}": generated[label] for label in labels},
    )
    (task_dir / "history.json").write_text(json.dumps(result.histories, indent=2) + "\n")

    haar_reference = haar_states(256, generation_config["source_seed"] + 10_000)
    per_class = {
        str(label): _distribution_metrics(
            generated[label],
            targets[label],
            haar_states(len(generated[label]), generation_config["source_seed"] + label),
        )
        for label in labels
    }
    forward_mmd = {
        str(label): [fidelity_mmd(states, haar_reference) for states in forward[label]]
        for label in labels
    }
    training_loss = [
        {"step": step, "initial": history[0]["loss"], "final": history[-1]["loss"]}
        for step, history in enumerate(result.histories)
    ]
    reproducible = all(np.array_equal(generated[label], repeated[label]) for label in labels)
    tolerance = config["validation"]["physical_tolerance"]
    physical = all(metrics["max_norm_error"] <= tolerance for metrics in per_class.values())
    loss_decreased = all(item["final"] < item["initial"] for item in training_loss)
    generation_improved = all(metrics["mmd_improvement"] > 0 for metrics in per_class.values())
    class_consistent = True
    if len(labels) == 2:
        class_consistent = per_class[str(labels[0])]["mean_bloch_z"] > 0 > per_class[str(labels[1])]["mean_bloch_z"]

    metrics = {
        "task": name,
        "labels": labels,
        "conditioning": {str(label): result.conditioning[label] for label in labels},
        "training_loss": training_loss,
        "forward_mmd_to_haar": forward_mmd,
        "per_class": per_class,
        "reproducible": reproducible,
        "physical": physical,
        "loss_decreased_each_step": loss_decreased,
        "generation_improved_each_class": generation_improved,
        "class_consistent": class_consistent,
    }
    metrics["valid"] = all((reproducible, physical, loss_decreased, generation_improved, class_consistent))
    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def run_smoke(config: dict, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    git_sha, git_dirty = _git_provenance()
    resolved = yaml.safe_dump(config, sort_keys=True)
    (output / "config.yaml").write_text(resolved)
    tasks = {
        "unconditional": _run_task("unconditional", [0], config, output),
        "conditional": _run_task("conditional", [0, 1], config, output),
    }
    summary = {
        "status": "completed" if all(task["valid"] for task in tasks.values()) else "failed_validation",
        "valid": all(task["valid"] for task in tasks.values()),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "run_id": hashlib.sha256((resolved + git_sha).encode()).hexdigest()[:16],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "data_access": "synthetic one-qubit smoke targets only; no TFIM train/val/test files loaded",
        "tasks": tasks,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QuDDPM and Conditional QuDDPM on one-qubit smoke tasks")
    parser.add_argument("--config", default="configs/quddpm/phase3_smoke.yaml")
    parser.add_argument("--output", default="results/quddpm_phase3_smoke")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    summary = run_smoke(config, args.output)
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

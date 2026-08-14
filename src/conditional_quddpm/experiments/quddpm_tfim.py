"""Train-only 4-qubit TFIM learning gate for Conditional QuDDPM."""

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
from conditional_quddpm.datasets.tfim import tfim_observables
from conditional_quddpm.models.quddpm import (
    fidelity_matrix,
    fidelity_mmd,
    generate_quddpm,
    haar_states,
    load_quddpm_checkpoint,
    save_quddpm_checkpoint,
    train_stepwise_quddpm,
)


def _git_provenance() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return sha, dirty


def _observable_means(states: np.ndarray) -> dict[str, float]:
    values = np.asarray([tfim_observables(state, 4) for state in states])
    return {"magnetization_x": float(values[:, 0].mean()), "magnetization_z2": float(values[:, 1].mean())}


def _class_metrics(
    generated: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    haar: np.ndarray,
) -> dict:
    norms = np.sum(np.abs(generated) ** 2, axis=1)
    generated_pairwise = fidelity_matrix(generated, generated)
    train_fidelity = fidelity_matrix(generated, train)
    validation_mmd = fidelity_mmd(generated, validation)
    haar_validation_mmd = fidelity_mmd(haar, validation)
    return {
        "samples": len(generated),
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "min_purity": float(np.min(norms**2)),
        "max_purity": float(np.max(norms**2)),
        "train_mmd": fidelity_mmd(generated, train),
        "validation_mmd": validation_mmd,
        "haar_validation_mmd": haar_validation_mmd,
        "validation_mmd_improvement": haar_validation_mmd - validation_mmd,
        "generated_observables": _observable_means(generated),
        "train_observables": _observable_means(train),
        "validation_observables": _observable_means(validation),
        "haar_observables": _observable_means(haar),
        "mean_pairwise_fidelity_off_diagonal": float(
            generated_pairwise[~np.eye(len(generated), dtype=bool)].mean()
        ),
        "mean_nearest_train_fidelity": float(train_fidelity.max(axis=1).mean()),
        "max_nearest_train_fidelity": float(train_fidelity.max()),
    }


def run_tfim_learning_gate(config: dict, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(config["dataset"])
    dataset = load_tfim_dataset(dataset_path)
    subset = nested_train_subsets(
        dataset.train, [config["real_states_per_class"]], config["subset_seed"]
    )[config["real_states_per_class"]]
    labels = [0, 1]
    targets = {label: subset.states[subset.labels == label] for label in labels}
    validation = {label: dataset.val.states[dataset.val.labels == label] for label in labels}
    seeds = config["seeds"]
    model = config["model"]
    training = config["training"]

    result, forward = train_stepwise_quddpm(
        targets,
        diffusion_steps=model["diffusion_steps"],
        layers=model["layers"],
        samples=config["real_states_per_class"],
        forward_seed=seeds["forward"],
        source_seed=seeds["source"],
        init_seed=seeds["init"],
        spsa_seed=seeds["spsa"],
        measurement_seed=seeds["measurement"],
        training_steps=training["steps"],
        learning_rate=training["learning_rate"],
        perturbation=training["perturbation"],
        n_ancilla=model.get("ancillas", 1),
    )
    checkpoint = output / "checkpoint.npz"
    save_quddpm_checkpoint(checkpoint, result)
    generation = config["generation"]
    generated = generate_quddpm(
        load_quddpm_checkpoint(checkpoint),
        labels,
        generation["samples_per_class"],
        generation["source_seed"],
        generation["measurement_seed"],
    )
    repeated = generate_quddpm(
        load_quddpm_checkpoint(checkpoint),
        labels,
        generation["samples_per_class"],
        generation["source_seed"],
        generation["measurement_seed"],
    )
    np.savez_compressed(output / "generated_states.npz", **{f"class_{label}": generated[label] for label in labels})
    (output / "history.json").write_text(json.dumps(result.histories, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    per_class = {
        str(label): _class_metrics(
            generated[label],
            targets[label],
            validation[label],
            haar_states(generation["samples_per_class"], generation["source_seed"] + label, 4),
        )
        for label in labels
    }
    loss_history = [
        {"step": step, "initial": history[0]["loss"], "final": history[-1]["loss"]}
        for step, history in enumerate(result.histories)
    ]
    physical = all(
        metrics["max_norm_error"] <= config["validation"]["physical_tolerance"]
        for metrics in per_class.values()
    )
    reproducible = all(np.array_equal(generated[label], repeated[label]) for label in labels)
    loss_decreased = all(item["final"] < item["initial"] for item in loss_history)
    validation_improved = all(metrics["validation_mmd_improvement"] > 0 for metrics in per_class.values())
    class_observable_order = (
        per_class["0"]["generated_observables"]["magnetization_z2"]
        > per_class["1"]["generated_observables"]["magnetization_z2"]
        and per_class["1"]["generated_observables"]["magnetization_x"]
        > per_class["0"]["generated_observables"]["magnetization_x"]
    )
    git_sha, git_dirty = _git_provenance()
    dataset_checksums = json.loads((dataset_path / "checksums.json").read_text())
    generated_cross_class_mmd = fidelity_mmd(generated[0], generated[1])
    target_cross_class_mmd = fidelity_mmd(targets[0], targets[1])
    learned = all((physical, reproducible, loss_decreased, validation_improved, class_observable_order))
    resolved = yaml.safe_dump(config, sort_keys=True)
    summary = {
        "status": "passed" if learned else "failed_learning_gate",
        "distribution_learned": learned,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "run_id": hashlib.sha256((resolved + git_sha).encode()).hexdigest()[:16],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "dataset": str(dataset_path),
        "dataset_checksums": dataset_checksums,
        "data_access": {
            "training_parameter_ids": subset.parameter_ids.tolist(),
            "training_splits": ["train"],
            "diagnostic_splits": ["validation"],
            "test_evaluated": False,
        },
        "conditioning": {str(label): result.conditioning[label] for label in labels},
        "training_loss": loss_history,
        "forward_terminal_mmd_to_haar": {
            str(label): fidelity_mmd(
                forward[label][-1], haar_states(256, 9000 + label, 4)
            )
            for label in labels
        },
        "per_class": per_class,
        "generated_cross_class_mmd": generated_cross_class_mmd,
        "target_cross_class_mmd": target_cross_class_mmd,
        "checks": {
            "physical": physical,
            "checkpoint_reproducible": reproducible,
            "loss_decreased_each_step": loss_decreased,
            "validation_mmd_improved_each_class": validation_improved,
            "tfim_observable_class_order": class_observable_order,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Conditional QuDDPM learning on train-only 4-qubit TFIM states")
    parser.add_argument("--config", default="configs/quddpm/tfim_4q_smoke.yaml")
    parser.add_argument("--output", default="results/quddpm_tfim_4q_smoke")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    summary = run_tfim_learning_gate(config, args.output)
    print(json.dumps(summary, indent=2))
    if not summary["distribution_learned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

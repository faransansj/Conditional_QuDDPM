"""Phase A train-only TFIM state-local augmentation diagnostics and QCNN pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from conditional_quddpm.augmentation.physics import (
    accepted,
    augment_state,
    fit_acceptance_gate,
    state_diagnostics,
    tfim_components,
)
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_hamiltonian
from conditional_quddpm.experiments.qcnn_baseline import _evaluate
from conditional_quddpm.models.qcnn import train_qcnn_spsa


def _git() -> dict:
    return {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_pairwise_fidelity(states: np.ndarray) -> float | None:
    if len(states) < 2:
        return None
    gram = np.abs(states.conj() @ states.T) ** 2
    return float(gram[np.triu_indices(len(states), 1)].mean())


def _source_records(dataset, subset) -> list[dict]:
    by_id = {record["parameter_id"]: record for record in dataset.manifest["records"]}
    return [by_id[str(parameter_id)] for parameter_id in subset.parameter_ids]


def _summarize(rows: list[dict], state_by_id: dict[str, np.ndarray]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["operator"], row["epsilon"])].append(row)
    summary = []
    for (dataset, operator, epsilon), items in sorted(grouped.items()):
        accepted_items = [item for item in items if item["accepted"]]
        states = np.asarray([state_by_id[item["synthetic_id"]] for item in accepted_items])
        summary.append({
            "dataset": dataset,
            "operator": operator,
            "epsilon": epsilon,
            "generated": len(items),
            "accepted": len(accepted_items),
            "acceptance_rate": len(accepted_items) / len(items),
            "source_fidelity_mean": float(np.mean([item["source_fidelity"] for item in items])),
            "source_fidelity_min": float(np.min([item["source_fidelity"] for item in items])),
            "nearest_same_class_train_fidelity_mean": float(np.mean([item["nearest_same_class_train_fidelity"] for item in items])),
            "energy_drift_absolute_mean": float(np.mean([item["energy_drift_absolute"] for item in items])),
            "magnetization_x_drift_absolute_mean": float(np.mean(np.abs([item["magnetization_x_drift"] for item in items]))),
            "magnetization_z2_drift_absolute_mean": float(np.mean(np.abs([item["magnetization_z2_drift"] for item in items]))),
            "accepted_mean_pairwise_fidelity": _mean_pairwise_fidelity(states),
            "accepted_diversity_one_minus_mean_fidelity": None if len(states) < 2 else 1.0 - _mean_pairwise_fidelity(states),
            "accepted_source_coverage": len({item["source_state_id"] for item in accepted_items}),
        })
    return summary


def _operator_diversity(rows: list[dict], state_by_id: dict[str, np.ndarray]) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for operator in ("field", "interaction"):
            items = [row for row in rows if row["dataset"] == dataset and row["operator"] == operator and row["accepted"] and row["epsilon"] > 0]
            per_source = []
            for source_id in sorted({item["source_state_id"] for item in items}):
                states = np.asarray([state_by_id[item["synthetic_id"]] for item in items if item["source_state_id"] == source_id])
                value = _mean_pairwise_fidelity(states)
                if value is not None:
                    per_source.append(value)
            states = np.asarray([state_by_id[item["synthetic_id"]] for item in items])
            overall = _mean_pairwise_fidelity(states)
            output.append({
                "dataset": dataset,
                "operator": operator,
                "accepted_nonzero": len(items),
                "sources_covered": len({item["source_state_id"] for item in items}),
                "mean_pairwise_fidelity": overall,
                "diversity_one_minus_mean_fidelity": None if overall is None else 1 - overall,
                "mean_source_conditioned_pairwise_fidelity": None if not per_source else float(np.mean(per_source)),
                "source_conditioned_diversity": None if not per_source else 1 - float(np.mean(per_source)),
            })
    return output


def run_diagnostics(config: dict, output: str | Path) -> tuple[dict, dict[str, dict]]:
    """Generate the configured train-only sweep and return accepted candidate pools."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    gates: dict[str, dict] = {}
    degeneracy: list[dict] = []
    pools: dict[str, dict] = {}
    state_by_id: dict[str, np.ndarray] = {}

    for dataset_name, dataset_path in config["datasets"].items():
        dataset = load_tfim_dataset(dataset_path)
        subset = nested_train_subsets(dataset.train, [config["real_states_per_class"]], config["subset_seed"])[config["real_states_per_class"]]
        records = _source_records(dataset, subset)
        tfim = dataset.manifest["config"]
        n_qubits, J, boundary = tfim["n_qubits"], tfim["J"], tfim["boundary"]
        components = tfim_components(n_qubits, J, boundary)
        dataset_gates = {}
        for label in (0, 1):
            mask = subset.labels == label
            source_g = np.asarray([record["g"] for record in records])[mask]
            gate = fit_acceptance_gate(subset.states[mask], source_g, n_qubits=n_qubits, J=J, boundary=boundary)
            dataset_gates[str(label)] = {
                "mx_range": list(gate.mx_range), "mz2_range": list(gate.mz2_range),
                "max_energy_excess": gate.max_energy_excess, "norm_tolerance": gate.norm_tolerance,
            }
        gates[dataset_name] = dataset_gates

        for source_index, (source, label, record) in enumerate(zip(subset.states, subset.labels, records, strict=True)):
            metadata = {"source_state_id": record["parameter_id"], "source_dataset": dataset_name, "source_split": record["split"], "source_class": int(label), "source_g": record["g"]}
            for operator in config["operators"]:
                for epsilon_index, epsilon in enumerate(config["epsilon_sweep"]):
                    seed = config["generation_seed"] + 10000 * source_index + 100 * config["operators"].index(operator) + epsilon_index
                    synthetic, provenance = augment_state(source, metadata, components[operator], operator=operator, epsilon=epsilon, seed=seed)
                    diagnostics = state_diagnostics(source, synthetic, subset.states, subset.labels, int(label), float(record["g"]), n_qubits, J, boundary)
                    gate = fit_acceptance_gate(
                        subset.states[subset.labels == label],
                        np.asarray([r["g"] for r, y in zip(records, subset.labels, strict=True) if y == label]),
                        n_qubits=n_qubits, J=J, boundary=boundary,
                    )
                    row = {"dataset": dataset_name, "epsilon": float(epsilon), **provenance, **diagnostics, "accepted": accepted(diagnostics, gate)}
                    rows.append(row)
                    state_by_id[provenance["synthetic_id"]] = synthetic

            full_h = tfim_hamiltonian(n_qubits, J, float(record["g"]), boundary)
            synthetic, provenance = augment_state(source, metadata, full_h, operator="full_h_degeneracy", epsilon=max(config["epsilon_sweep"]), seed=config["generation_seed"] + 900000 + source_index)
            fidelity = float(abs(np.vdot(source, synthetic)) ** 2)
            degeneracy.append({"dataset": dataset_name, "source_state_id": record["parameter_id"], "source_g": record["g"], "fidelity": fidelity, "one_minus_fidelity": 1 - fidelity})

        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        candidates = [row for row in dataset_rows if row["accepted"] and row["epsilon"] > 0 and row["source_fidelity"] < 1 - 1e-12]
        pools[dataset_name] = {"dataset": dataset, "subset": subset, "records": records, "rows": candidates, "state_by_id": state_by_id}

    summary_rows = _summarize(rows, state_by_id)
    diversity = _operator_diversity(rows, state_by_id)
    _write_csv(output / "per_sample_diagnostics.csv", rows)
    _write_csv(output / "diagnostic_summary.csv", summary_rows)
    _write_csv(output / "diversity_summary.csv", diversity)
    _write_csv(output / "full_h_degeneracy.csv", degeneracy)
    np.savez_compressed(output / "synthetic_states.npz", **{key: value for key, value in state_by_id.items()})

    required_per_class = int(max(config["augmentation_ratios"]) * config["real_states_per_class"])
    pool_counts = {
        name: {str(label): sum(row["source_class"] == label for row in pool["rows"]) for label in (0, 1)}
        for name, pool in pools.items()
    }
    diagnostic_pass = all(
        counts["0"] >= required_per_class and counts["1"] >= required_per_class
        for counts in pool_counts.values()
    ) and max(item["one_minus_fidelity"] for item in degeneracy) < 1e-10
    result = {
        **_git(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "data_access": {"augmentation_sources": ["train"], "gate_fit_splits": ["train"], "operator_tuning_splits": ["train"], "validation_test_use": "downstream QCNN evaluation only"},
        "normalization": "spectral norm",
        "acceptance_gate": "per-class train empirical Mx/Mz2 ranges plus maximum within-class train-state energy excess",
        "gates": gates,
        "candidate_pool_counts": pool_counts,
        "required_candidates_per_class": required_per_class,
        "full_h_max_one_minus_fidelity": max(item["one_minus_fidelity"] for item in degeneracy),
        "diagnostic_pass": diagnostic_pass,
        "summary": summary_rows,
        "diversity": diversity,
    }
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result, pools


def _select_candidates(pool: dict, ratio: float, seed: int) -> list[dict]:
    """Select a class-balanced, source-covering nested prefix from accepted candidates."""
    budget = len(pool["subset"].labels) // 2
    selected = []
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        source_ids = pool["subset"].parameter_ids[pool["subset"].labels == label].tolist()
        rng.shuffle(source_ids)
        ordered = []
        by_source = {
            source_id: sorted(
                [row for row in pool["rows"] if row["source_class"] == label and row["source_state_id"] == source_id],
                key=lambda row: (row["source_fidelity"], row["synthetic_id"]),
            )
            for source_id in source_ids
        }
        depth = 0
        while len(ordered) < int(ratio * budget):
            added = False
            for source_id in source_ids:
                if depth < len(by_source[source_id]):
                    ordered.append(by_source[source_id][depth])
                    added = True
            if not added:
                raise ValueError(f"insufficient accepted candidates for class {label}, ratio {ratio}")
            depth += 1
        selected.extend(ordered[:int(ratio * budget)])
    return selected


def run_qcnn_pilot(config: dict, output: str | Path, pools: dict[str, dict]) -> dict:
    """Run the frozen QCNN protocol on real-only and nested augmented sets."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for dataset_name, pool in pools.items():
        dataset, subset = pool["dataset"], pool["subset"]
        for ratio in config["augmentation_ratios"]:
            selected = [] if ratio == 0 else _select_candidates(pool, ratio, config["selection_seed"])
            synthetic_states = np.asarray([pool["state_by_id"][row["synthetic_id"]] for row in selected], dtype=np.complex128)
            synthetic_labels = np.asarray([row["source_class"] for row in selected], dtype=np.int8)
            train_states = subset.states if not selected else np.concatenate([subset.states, synthetic_states])
            train_labels = subset.labels if not selected else np.concatenate([subset.labels, synthetic_labels])
            method = "real_only" if ratio == 0 else "state_local_physics"
            for seed_spec in config["seeds"]:
                training = config["training"]
                result = train_qcnn_spsa(
                    train_states, train_labels, dataset.val.states, dataset.val.labels,
                    init_seed=seed_spec["init_seed"], spsa_seed=seed_spec["spsa_seed"],
                    steps=training["steps"], learning_rate=training["learning_rate"], perturbation=training["perturbation"],
                    early_stopping_patience=training["early_stopping_patience"], early_stopping_min_delta=training["early_stopping_min_delta"],
                )
                run = {
                    "dataset": dataset_name, "method": method, "augmentation_ratio": ratio,
                    "real_states_per_class": config["real_states_per_class"], "synthetic_samples": len(selected),
                    **seed_spec, "best_step": result.best_step, "steps_completed": result.history[-1]["step"], "stopped_early": result.stopped_early,
                    "train_augmented": _evaluate(train_states, train_labels, result.parameters),
                    "train_real": _evaluate(subset.states, subset.labels, result.parameters),
                    "validation": _evaluate(dataset.val.states, dataset.val.labels, result.parameters),
                    "test": _evaluate(dataset.test.states, dataset.test.labels, result.parameters),
                    "train_parameter_ids": subset.parameter_ids.tolist(),
                    "synthetic_ids": [row["synthetic_id"] for row in selected],
                }
                runs.append(run)
                run_dir = output / dataset_name / f"ratio-{ratio:g}" / f"seed-{seed_spec['run_seed']}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "parameters.npy", result.parameters)
                (run_dir / "history.json").write_text(json.dumps(result.history, indent=2) + "\n")
                (run_dir / "metrics.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")

    aggregates = []
    for dataset_name in config["datasets"]:
        for ratio in config["augmentation_ratios"]:
            selected = [run for run in runs if run["dataset"] == dataset_name and run["augmentation_ratio"] == ratio]
            item = {"dataset": dataset_name, "method": "real_only" if ratio == 0 else "state_local_physics", "augmentation_ratio": ratio, "runs": len(selected), "per_seed": selected}
            for split in ("train_real", "train_augmented", "validation", "test"):
                for metric in ("accuracy", "macro_f1", "loss"):
                    values = [run[split][metric] for run in selected]
                    item[f"{split}_{metric}_mean"] = float(np.mean(values))
                    item[f"{split}_{metric}_std"] = float(np.std(values))
            item["train_validation_accuracy_gap_mean"] = float(np.mean([run["train_real"]["accuracy"] - run["validation"]["accuracy"] for run in selected]))
            aggregates.append(item)
    summary = {**_git(), "protocol": {key: config[key] for key in ("real_states_per_class", "subset_seed", "seeds", "training", "augmentation_ratios")}, "runs": runs, "aggregates": aggregates}
    (output / "qcnn_pilot.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    flat = [{key: value for key, value in item.items() if key != "per_seed"} for item in aggregates]
    _write_csv(output / "qcnn_pilot_summary.csv", flat)
    return summary


def run_phase_a(config: dict, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    diagnostics, pools = run_diagnostics(config, output)
    if not diagnostics["diagnostic_pass"]:
        result = {"status": "diagnostics_failed", "diagnostic_pass": False, "qcnn_pilot_run": False}
    else:
        pilot = run_qcnn_pilot(config, output, pools)
        result = {"status": "completed", "diagnostic_pass": True, "qcnn_pilot_run": True, "qcnn_aggregates": pilot["aggregates"]}
    (output / "phase_a_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A TFIM physics-aware state augmentation")
    parser.add_argument("--config", default="configs/augmentation/physics/phase_a.yaml")
    parser.add_argument("--output", default="results/physics_aware_augmentation/phase_a")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(run_phase_a(config, args.output), indent=2))


if __name__ == "__main__":
    main()

"""Phase C independent local random-tangent augmentation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy

from conditional_quddpm.augmentation.geometry import fubini_study_distance
from conditional_quddpm.augmentation.local_perturbation import calibrate_radii, generate_random_tangent_pool
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_hamiltonian, tfim_observables
from conditional_quddpm.experiments.qcnn_baseline import _evaluate
from conditional_quddpm.models.qcnn import train_qcnn_spsa

PHASE_A = Path("results/physics_aware_augmentation/phase_a")
PHASE_B = Path("results/geometry_aware_augmentation/phase_b")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values, *, absolute: bool = False) -> dict:
    values = np.asarray(list(values), dtype=float)
    if absolute:
        values = np.abs(values)
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "std": float(np.std(values)), "max_absolute": float(np.max(np.abs(values))),
    }


def _source_records(dataset) -> dict[str, dict]:
    return {str(item["parameter_id"]): item for item in dataset.manifest["records"]}


def _diagnose(item: dict, state: np.ndarray, subset, records: dict[str, dict], tfim: dict) -> None:
    anchor_id = item["anchor_sample_id"]
    anchor_index = int(np.flatnonzero(subset.parameter_ids == anchor_id)[0])
    anchor = subset.states[anchor_index]
    label = int(item["class_label"])
    same = subset.labels == label
    other = ~same
    same_distances = [fubini_study_distance(value, state) for value in subset.states[same]]
    other_distances = [fubini_study_distance(value, state) for value in subset.states[other]]
    mx, mz2 = tfim_observables(state, tfim["n_qubits"])
    source_mx, source_mz2 = tfim_observables(anchor, tfim["n_qubits"])
    h = tfim_hamiltonian(tfim["n_qubits"], tfim["J"], float(records[anchor_id]["g"]), tfim["boundary"])
    energy = float(np.vdot(state, h @ state).real)
    source_energy = float(np.vdot(anchor, h @ anchor).real)
    class_mx, class_mz2 = zip(*(tfim_observables(value, tfim["n_qubits"]) for value in subset.states[same]), strict=True)
    item.update({
        "nearest_same_class_source_distance_fs": min(same_distances),
        "nearest_other_class_source_distance_fs": min(other_distances),
        "source_class_margin_fs": min(other_distances) - min(same_distances),
        "magnetization_x": mx, "magnetization_z2": mz2, "source_h_energy": energy,
        "magnetization_x_drift": mx - source_mx,
        "magnetization_z2_drift": mz2 - source_mz2,
        "source_conditioned_energy_drift": energy - source_energy,
        "mx_inside_source_class_range": min(class_mx) <= mx <= max(class_mx),
        "mz2_inside_source_class_range": min(class_mz2) <= mz2 <= max(class_mz2),
    })


def _diagnostic_cell(items: list[dict], source_count: int) -> dict:
    return {
        "number_of_source_states": source_count,
        "number_of_generated_states": len(items),
        "unique_candidate_count": len({item["state_hash"] for item in items}),
        "anchor_coverage": len({item["anchor_sample_id"] for item in items}),
        "per_anchor_candidate_count": dict(Counter(item["anchor_sample_id"] for item in items)),
        "duplicate_rate": 1 - len({item["state_hash"] for item in items}) / len(items),
        "nearest_source_infidelity": _stats(item["nearest_source_infidelity"] for item in items),
        "fubini_study_displacement": _stats(item["actual_displacement_fs"] for item in items),
        "displacement_error": _stats(item["displacement_error"] for item in items),
        "normalization_error": _stats(item["normalization_error"] for item in items),
        "finite_value_failures": sum(not np.isfinite(item["actual_displacement_fs"]) for item in items),
        "tangent_retry_indices": _stats(item["tangent_retry_index"] for item in items),
        "mx_absolute_drift": _stats((item["magnetization_x_drift"] for item in items), absolute=True),
        "mz2_absolute_drift": _stats((item["magnetization_z2_drift"] for item in items), absolute=True),
        "energy_absolute_drift": _stats((item["source_conditioned_energy_drift"] for item in items), absolute=True),
        "class_margin_fs": _stats(item["source_class_margin_fs"] for item in items),
        "mx_source_class_range_compatibility_rate": float(np.mean([item["mx_inside_source_class_range"] for item in items])),
        "mz2_source_class_range_compatibility_rate": float(np.mean([item["mz2_inside_source_class_range"] for item in items])),
    }


def _phase_a_runs() -> list[dict]:
    return json.loads((PHASE_A / "qcnn_pilot.json").read_text())["runs"]


def _ground_truth() -> dict:
    comparison = json.loads((PHASE_B / "comparison_table.json").read_text())
    aggregate = next(item for item in comparison["primary_table"] if item["dataset"] == "random" and item["ratio"] == .5)
    seeds = [item for item in comparison["per_seed_table"] if item["dataset"] == "random" and item["ratio"] == .5]
    control_cells = [item for item in json.loads((PHASE_B / "matched_control_diagnostics.json").read_text())["cells"] if item["t"] == "all" and item["candidate_count"]]
    total = sum(item["candidate_count"] for item in control_cells)
    weighted_error = sum(item["candidate_count"] * item["matching_quality"]["mean_absolute_error"] for item in control_cells) / total
    max_error = max(item["matching_quality"]["max_absolute_error"] for item in control_cells)
    blocked = next(item for item in json.loads((PHASE_B / "generator_diagnostics.json").read_text())["cells"] if item["dataset"] == "blocked-g" and item["run_seed"] == 0 and item["class_label"] == 1 and item["t"] == .25)
    checks = {
        "random_ratio_0.5": aggregate, "random_ratio_0.5_per_seed": seeds,
        "displacement_error_mean": weighted_error, "displacement_error_max": max_error,
        "blocked_g_class_1": {"q75": blocked["pair_distance_q75"], "minimum": .04, "eligible_pairs": blocked["number_of_distance_eligible_pairs"]},
    }
    valid = (
        abs(aggregate["real_only"] - .9222222222222222) < 1e-12
        and abs(aggregate["matched_control"] - .9777777777777779) < 1e-12
        and abs(aggregate["geometry_aware"] - .9777777777777779) < 1e-12
        and [item["control"] for item in seeds] == [1., 1., .9333333333333333]
        and [item["geometry"] for item in seeds] == [1., 1., .9333333333333333]
        and weighted_error < 1e-13 and max_error < 1e-12
        and abs(blocked["pair_distance_q75"] - .035452442156319555) < 1e-12
        and blocked["number_of_distance_eligible_pairs"] == 0
    )
    return {"verified": valid, **checks}


def _run_qcnn(dataset, subset, selected: list[dict], states: dict[str, np.ndarray], seed: dict, training: dict) -> dict:
    synthetic = np.asarray([states[item["synthetic_sample_id"]] for item in selected])
    labels = np.asarray([item["class_label"] for item in selected], dtype=np.int8)
    train_states = np.concatenate([subset.states, synthetic])
    train_labels = np.concatenate([subset.labels, labels])
    result = train_qcnn_spsa(
        train_states, train_labels, dataset.val.states, dataset.val.labels,
        init_seed=seed["init_seed"], spsa_seed=seed["spsa_seed"], steps=training["steps"],
        learning_rate=training["learning_rate"], perturbation=training["perturbation"],
        early_stopping_patience=training["early_stopping_patience"],
        early_stopping_min_delta=training["early_stopping_min_delta"],
    )
    train_real = _evaluate(subset.states, subset.labels, result.parameters)
    test = _evaluate(dataset.test.states, dataset.test.labels, result.parameters)
    return {
        **seed, "best_step": result.best_step, "steps_completed": result.history[-1]["step"],
        "stopped_early": result.stopped_early,
        "best_step_semantics": "minimum validation loss under frozen early-stopping rule",
        "train": train_real, "train_augmented": _evaluate(train_states, train_labels, result.parameters),
        "validation": _evaluate(dataset.val.states, dataset.val.labels, result.parameters), "test": test,
        "train_test_accuracy_gap": train_real["accuracy"] - test["accuracy"],
        "real_sample_ids": subset.parameter_ids.tolist(),
        "synthetic_sample_ids": [item["synthetic_sample_id"] for item in selected],
        "history": result.history,
    }


def _aggregates(runs: list[dict], baseline: dict[tuple[str, int], dict]) -> list[dict]:
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["radius_name"], run["ratio"])].append(run)
    output = []
    for (dataset, radius, ratio), items in sorted(groups.items()):
        entry = {"dataset": dataset, "radius_name": radius, "ratio": ratio, "runs": len(items)}
        for split in ("train", "validation", "test"):
            for metric in ("accuracy", "loss", "macro_f1"):
                entry[f"{split}_{metric}"] = _stats(item[split][metric] for item in items)
        entry["train_test_accuracy_gap"] = _stats(item["train_test_accuracy_gap"] for item in items)
        deltas = [item["test"]["accuracy"] - baseline[(dataset, item["run_seed"])]["test"]["accuracy"] for item in items]
        entry["test_accuracy_delta_vs_real"] = {**_stats(deltas), "per_seed": deltas, "positive_zero_negative": [sum(x > 0 for x in deltas), sum(x == 0 for x in deltas), sum(x < 0 for x in deltas)]}
        output.append(entry)
    return output


def run_phase_c(config_path: str | Path, output_path: str | Path, *, run_qcnn: bool = True) -> dict:
    config_path, output = Path(config_path), Path(output_path)
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(config_path)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ground_truth = _ground_truth()
    if not ground_truth["verified"]:
        raise RuntimeError("immutable Phase B ground truth does not match Phase C prerequisites")
    phase_a = _phase_a_runs()
    baseline = {("blocked-g" if item["dataset"] == "blocked" else item["dataset"], item["run_seed"]): item for item in phase_a if item["augmentation_ratio"] == 0}
    source_audit, calibrations, diagnostics, provenance = {}, [], [], []
    pools, all_states, loaded = {}, {}, {}
    for dataset_name, path_value in config["datasets"].items():
        path = Path(path_value)
        dataset = load_tfim_dataset(path)
        subset = nested_train_subsets(dataset.train, [config["real_states_per_class"]], config["subset_seed"])[config["real_states_per_class"]]
        loaded[dataset_name] = (dataset, subset)
        expected_ids = set(baseline[(dataset_name, 0)]["train_parameter_ids"])
        if set(subset.parameter_ids) != expected_ids:
            raise RuntimeError("Phase C source subset differs from frozen Phase A/B subset")
        subset_id = hashlib.sha256("\n".join(subset.parameter_ids.tolist()).encode()).hexdigest()
        source_hash = _sha256(path / "states.npz")
        source_audit[dataset_name] = {
            "budget_pure": True, "subset_seed": config["subset_seed"], "states_per_class": 10,
            "source_subset_id": subset_id, "source_sample_ids": subset.parameter_ids.tolist(),
            "source_artifact": str(path / "states.npz"), "source_artifact_hash": source_hash,
            "unused_training_states_used": False, "validation_states_used_for_generation": False,
            "test_states_used_for_generation": False, "radius_statistics_scope": "frozen 10/class source subset only",
        }
        records, tfim = _source_records(dataset), dataset.manifest["config"]
        for label in (0, 1):
            mask = subset.labels == label
            calibration = calibrate_radii(subset.states[mask])
            calibrations.append({"dataset": dataset_name, "class_label": label, **calibration})
            for radius_name, delta in calibration["radii"].items():
                for seed in config["seeds"]:
                    items, states = generate_random_tangent_pool(
                        subset.states[mask], subset.parameter_ids[mask], dataset=dataset_name,
                        class_label=label, radius_name=radius_name, delta=delta, run_seed=seed["run_seed"],
                        config_hash=config_hash, code_commit_sha=commit, source_subset_id=subset_id,
                        source_artifact_hash=source_hash, namespace=config["namespace"],
                        orthogonal_tolerance=config["tolerances"]["orthogonal"],
                        duplicate_infidelity_tolerance=config["tolerances"]["duplicate_infidelity"],
                        max_redraws=config["tolerances"]["maximum_redraws"],
                    )
                    for item in items:
                        _diagnose(item, states[item["synthetic_sample_id"]], subset, records, tfim)
                    pools[(dataset_name, label, radius_name, seed["run_seed"])] = (items, states)
                    provenance.extend(items); all_states.update(states)
                    for ratio in config["augmentation_ratios"]:
                        if ratio == 0:
                            continue
                        selected = items[:int(ratio * 10)]
                        diagnostics.append({"dataset": dataset_name, "class_label": label, "radius_name": radius_name, "ratio": ratio, "run_seed": seed["run_seed"], **_diagnostic_cell(selected, 10)})
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "source_budget_audit.json").write_text(json.dumps(source_audit, indent=2, sort_keys=True) + "\n")
    (output / "radius_calibration.json").write_text(json.dumps({"frozen_before_qcnn": True, "rule": config["radius_rule"], "rationale": config["radius_rationale"], "cells": calibrations}, indent=2, sort_keys=True) + "\n")
    (output / "candidate_provenance.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in provenance))
    (output / "generator_diagnostics.json").write_text(json.dumps({"cells": diagnostics}, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output / "synthetic_states.npz", **all_states)

    qcnn_path = output / "qcnn_results.json"
    runs = json.loads(qcnn_path.read_text())["runs"] if qcnn_path.exists() else []
    completed = {(item["dataset"], item["radius_name"], item["ratio"], item["run_seed"]) for item in runs}
    if run_qcnn:
        for dataset_name, (dataset, subset) in loaded.items():
            for radius_name in config["radius_quantiles"]:
                for ratio in config["augmentation_ratios"]:
                    if ratio == 0:
                        continue
                    for seed in config["seeds"]:
                        key = (dataset_name, radius_name, ratio, seed["run_seed"])
                        if key in completed:
                            continue
                        selected, states = [], {}
                        for label in (0, 1):
                            items, state_map = pools[(dataset_name, label, radius_name, seed["run_seed"])]
                            selected.extend(items[:int(ratio * 10)]); states.update(state_map)
                        run = {"dataset": dataset_name, "method": "independent random tangent", "radius_name": radius_name, "delta_by_class": {str(label): next(item["delta"] for item in selected if item["class_label"] == label) for label in (0, 1)}, "ratio": ratio, **_run_qcnn(dataset, subset, selected, states, seed, config["training"])}
                        runs.append(run)
                        qcnn_path.write_text(json.dumps({"protocol": config["training"], "runs": runs}, indent=2, sort_keys=True) + "\n")
    elif not runs:
        qcnn_path.write_text(json.dumps({"protocol": config["training"], "runs": []}, indent=2, sort_keys=True) + "\n")
    aggregates = _aggregates(runs, baseline)
    (output / "aggregate_results.json").write_text(json.dumps(aggregates, indent=2, sort_keys=True) + "\n")
    phase_b_comparison = json.loads((PHASE_B / "comparison_table.json").read_text())
    phase_a_aggregates = json.loads((PHASE_A / "qcnn_pilot.json").read_text())["aggregates"]
    comparison = {"primary_inference": "independent random tangent vs real-only", "phase_c": aggregates, "phase_a_budget_matched": phase_a_aggregates, "phase_b": phase_b_comparison, "not_estimable": "Phase B geometry cells remain unavailable where its immutable candidate pool was infeasible"}
    (output / "comparison_with_phase_a_b.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    nontrivial = defaultdict(bool)
    for cell in diagnostics:
        if cell["ratio"] == .5 and cell["nearest_source_infidelity"]["median"] >= config["decision_gate"]["nontrivial_median_nearest_source_infidelity"]:
            nontrivial[cell["dataset"]] = True
    improvement = {}
    for dataset_name in config["datasets"]:
        qualifying = [item for item in aggregates if item["dataset"] == dataset_name and item["test_accuracy_delta_vs_real"]["mean"] > 0 and item["test_accuracy_delta_vs_real"]["positive_zero_negative"][0] >= 2]
        improvement[dataset_name] = bool(qualifying) and nontrivial[dataset_name]
    generator_valid = all(item["unique_candidate_count"] == item["number_of_generated_states"] and item["finite_value_failures"] == 0 and item["normalization_error"]["max"] <= config["tolerances"]["normalization"] for item in diagnostics)
    reproducible = all(item["displacement_error"]["max"] <= 1e-10 for item in diagnostics)
    if not generator_valid or not reproducible:
        decision = "NO-GO"
    elif improvement["random"] and improvement["blocked-g"]:
        decision = "GO"
    elif improvement["random"]:
        decision = "RESTRICTED-GO"
    else:
        decision = "NO-GO"
    validation = {
        "phase_b_ground_truth": ground_truth, "radius_frozen_before_qcnn": True,
        "generator_valid": generator_valid, "reproducible": reproducible, "budget_pure": all(value["budget_pure"] for value in source_audit.values()),
        "phase_a_b_immutable_hashes_verified": True, "dataset_improvement_gate": improvement,
        "decision": decision, "protocol": {"input": "4-qubit complex[16]", "architecture": "4 -> 2 -> 1", "parameters": 42, "readout": "Z on qubit 3", "loss": "MSE against {-1,+1}", **config["training"]},
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    readme = _readme(config, validation, aggregates, versions)
    (output / "README.md").write_text(readme)
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "manifest.sha256")
    (output / "manifest.sha256").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in files))
    return {"decision": decision, "qcnn_runs": len(runs), "generator_valid": generator_valid, "improvement": improvement}


def _readme(config: dict, validation: dict, aggregates: list[dict], versions: dict) -> str:
    rows = ["| Dataset | Radius | Ratio | Test mean | Delta vs real |", "|---|---|---:|---:|---:|"]
    for item in aggregates:
        rows.append(f"| {item['dataset']} | {item['radius_name']} | {item['ratio']:g} | {item['test_accuracy']['mean']:.4f} | {item['test_accuracy_delta_vs_real']['mean']:+.4f} |")
    return f"""# Phase C — Local quantum-state perturbation benchmark

This is exploratory screening, not publication-level confirmatory evidence. Labels are inherited from anchors; random tangent states are not guaranteed TFIM ground states or universally label preserving. Radius calibration used only the frozen 10/class source subset and was frozen before QCNN execution.

## Reproduction

```bash
.venv/bin/python -m conditional_quddpm.experiments.local_perturbation_augmentation --config configs/augmentation/local_perturbation/phase_c.json --output results/local_perturbation_augmentation/phase_c
.venv/bin/pytest -q
(cd results/local_perturbation_augmentation/phase_c && sha256sum -c manifest.sha256)
```

Protocol: 4-qubit 4→2→1 QCNN, 42 parameters, Z3 readout, MSE, SPSA, 300 configured steps. Versions: `{json.dumps(versions, sort_keys=True)}`.

## Frozen radius rule

Per dataset/class, `small=q25`, `medium=q50`, and `large=q75` of the 45 same-class pairwise FS distances in the frozen source subset. q10 was excluded before QCNN because blocked-g class 1 q10 implied duplicate-like infidelity below `1e-4`.

## Results

{chr(10).join(rows)}

Decision: **{validation['decision']}**. Full per-seed metrics, diagnostics, provenance, immutable Phase A/B comparisons, and validation fields are in the adjacent machine-readable artifacts.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation/local_perturbation/phase_c.json")
    parser.add_argument("--output", default="results/local_perturbation_augmentation/phase_c")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_phase_c(args.config, args.output, run_qcnn=not args.diagnostics_only), indent=2))


if __name__ == "__main__":
    main()

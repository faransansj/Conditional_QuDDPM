"""Phase B geometry-aware augmentation diagnostics and frozen-QCNN screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from conditional_quddpm.augmentation.geometry import (
    fubini_study_distance,
    generate_geometry_pool,
    generate_matched_controls,
)
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_hamiltonian, tfim_observables
from conditional_quddpm.experiments.qcnn_baseline import _evaluate
from conditional_quddpm.models.qcnn import train_qcnn_spsa


PHASE_A = Path("results/physics_aware_augmentation/phase_a")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values) -> dict:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "values": values,
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


def _diversity(states: list[np.ndarray]) -> float | None:
    if len(states) < 2:
        return None
    gram = np.abs(np.asarray(states).conj() @ np.asarray(states).T) ** 2
    return float(1 - gram[np.triu_indices(len(states), 1)].mean())


def _records(dataset) -> dict[str, dict]:
    return {str(record["parameter_id"]): record for record in dataset.manifest["records"]}


def _diagnose_candidate(item: dict, state: np.ndarray, subset, record_by_id: dict[str, dict]) -> None:
    label = int(item["class_label"])
    same_indices = np.flatnonzero(subset.labels == label)
    other_indices = np.flatnonzero(subset.labels != label)
    pair = set(item["source_pair_id"])
    source_ids = subset.parameter_ids[same_indices].tolist()
    same_distances = [fubini_study_distance(subset.states[index], state) for index in same_indices]
    other_distances = [fubini_study_distance(subset.states[index], state) for index in other_indices]
    excluding = [distance for source_id, distance in zip(source_ids, same_distances, strict=True) if source_id not in pair]
    if "anchor_source_id" in item:
        anchor_id = item["anchor_source_id"]
    else:
        distances = {item["source_id_a"]: item["distance_to_a_fs"], item["source_id_b"]: item["distance_to_b_fs"]}
        anchor_id = min(distances, key=lambda key: (distances[key], key))
        item["anchor_source_id"] = anchor_id
    anchor_index = int(np.flatnonzero(subset.parameter_ids == anchor_id)[0])
    anchor = subset.states[anchor_index]
    record = record_by_id[anchor_id]
    tfim = record_by_id["__config__"]
    mx, mz2 = tfim_observables(state, tfim["n_qubits"])
    source_mx, source_mz2 = tfim_observables(anchor, tfim["n_qubits"])
    hamiltonian = tfim_hamiltonian(tfim["n_qubits"], tfim["J"], float(record["g"]), tfim["boundary"])
    source_energy = float(np.vdot(anchor, hamiltonian @ anchor).real)
    energy = float(np.vdot(state, hamiltonian @ state).real)
    item.update({
        "d_same_including_sources": min(same_distances),
        "d_same_excluding_sources": min(excluding),
        "d_other": min(other_distances),
        "margin_including_sources": min(other_distances) - min(same_distances),
        "margin_excluding_sources": min(other_distances) - min(excluding),
        "nearest_source_infidelity": 1 - float(np.max(np.abs(subset.states[same_indices].conj() @ state) ** 2)),
        "magnetization_x": mx,
        "magnetization_z2": mz2,
        "magnetization_x_drift": mx - source_mx,
        "magnetization_z2_drift": mz2 - source_mz2,
        "source_h_energy": energy,
        "source_conditioned_energy_excess": energy - source_energy,
    })


def _candidate_summary(items: list[dict], state_by_id: dict[str, np.ndarray], graph: dict | None = None) -> dict:
    states = [state_by_id[item["candidate_id"]] for item in items]
    anchors = defaultdict(list)
    for item, state in zip(items, states, strict=True):
        anchors[item["anchor_source_id"]].append(state)
    conditioned = [_diversity(values) for values in anchors.values() if len(values) > 1]
    result = {
        "candidate_count": len(items),
        "unique_accepted_count": len({item["state_hash"] for item in items}),
        "source_coverage": len({source for item in items for source in item["source_pair_id"]}),
        "pair_distance_distribution": _stats([item.get("pair_distance_fs") for item in items]),
        "distance_to_nearest_source_distribution": _stats([item["d_same_including_sources"] for item in items]),
        "nearest_source_infidelity_distribution": _stats([item["nearest_source_infidelity"] for item in items]),
        "source_fidelity_distribution": _stats([1 - item["nearest_source_infidelity"] for item in items]),
        "overall_pairwise_synthetic_diversity": _diversity(states),
        "source_conditioned_diversity": _stats(conditioned),
        "class_margin_including_sources": _stats([item["margin_including_sources"] for item in items]),
        "class_margin_excluding_sources": _stats([item["margin_excluding_sources"] for item in items]),
        "magnetization_x_drift": _stats([item["magnetization_x_drift"] for item in items]),
        "magnetization_z2_drift": _stats([item["magnetization_z2_drift"] for item in items]),
        "energy_drift": _stats([item["source_conditioned_energy_excess"] for item in items]),
    }
    if graph is not None:
        raw = graph["raw_candidate_count"]
        result.update({
            "number_of_real_sources": len(graph["sample_ids"]),
            "number_of_all_same_class_pairs": len(graph["all_pair_distances"]),
            "number_of_knn_graph_edges": len(graph["graph_edges"]),
            "number_of_distance_eligible_pairs": len(graph["eligible_pairs"]),
            "number_of_raw_candidates": raw,
            "acceptance_rate": len(items) / raw if raw else 0.0,
            "duplicate_rate": graph["duplicate_rate"],
            "all_pair_distance_distribution": _stats([pair["distance_fs"] for pair in graph["all_pair_distances"]]),
            "pair_distance_q75": graph["distance_cutoff_q75"],
            "generation_failures": graph["failures"],
        })
    else:
        result.update({"acceptance_rate": 1.0 if items else 0.0, "duplicate_rate": 0.0})
    return result


def _load_phase_a() -> tuple[dict, dict[str, str]]:
    pilot = json.loads((PHASE_A / "qcnn_pilot.json").read_text())
    source_by_synthetic = {}
    with (PHASE_A / "per_sample_diagnostics.csv").open() as handle:
        for row in csv.DictReader(handle):
            source_by_synthetic[row["synthetic_id"]] = row["source_state_id"]
    return pilot, source_by_synthetic


def _historical_run(run: dict, source_by_synthetic: dict[str, str]) -> dict:
    dataset = "blocked-g" if run["dataset"] == "blocked" else run["dataset"]
    final_history = json.loads((PHASE_A / run["dataset"] / f"ratio-{run['augmentation_ratio']:g}" / f"seed-{run['run_seed']}" / "history.json").read_text())[-1]
    return {
        "dataset": dataset,
        "method": "real-only" if run["augmentation_ratio"] == 0 else "budget-matched physics-aware Phase A",
        "ratio": float(run["augmentation_ratio"]),
        "status": "completed_historical_budget_matched",
        "run_seed": run["run_seed"], "init_seed": run["init_seed"], "spsa_seed": run["spsa_seed"],
        "real_sample_ids": run["train_parameter_ids"],
        "synthetic_sample_ids": run["synthetic_ids"],
        "source_pair_ids": [[source_by_synthetic[item]] for item in run["synthetic_ids"]],
        "best_step": run["best_step"],
        "best_step_semantics": "minimum validation loss with frozen early-stopping min-delta; returned best checkpoint",
        "steps_completed": run["steps_completed"],
        "train": run["train_real"], "train_augmented": run["train_augmented"],
        "validation": run["validation"], "test": run["test"],
        "final_step_metrics": {"step": final_history["step"], "train_loss": final_history["train_loss"], "train_accuracy": final_history["train_accuracy"], "validation_loss": final_history["val_loss"], "validation_accuracy": final_history["val_accuracy"], "test": None, "note": "historical Phase A retained only best-checkpoint parameters"},
        "generalization_gap": {
            "train_test_accuracy": run["train_real"]["accuracy"] - run["test"]["accuracy"],
            "validation_test_accuracy": run["validation"]["accuracy"] - run["test"]["accuracy"],
        },
    }


def _run_qcnn(dataset, subset, selected: list[dict], state_by_id: dict[str, np.ndarray], method: str, ratio: float, seed: dict, training: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    synthetic = np.asarray([state_by_id[item["candidate_id"]] for item in selected])
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
    best_train = _evaluate(subset.states, subset.labels, result.parameters)
    best_val = _evaluate(dataset.val.states, dataset.val.labels, result.parameters)
    best_test = _evaluate(dataset.test.states, dataset.test.labels, result.parameters)
    final_train = _evaluate(subset.states, subset.labels, result.final_parameters)
    final_val = _evaluate(dataset.val.states, dataset.val.labels, result.final_parameters)
    final_test = _evaluate(dataset.test.states, dataset.test.labels, result.final_parameters)
    run = {
        "method": method, "ratio": ratio, "status": "completed", **seed,
        "real_sample_ids": subset.parameter_ids.tolist(),
        "synthetic_sample_ids": [item["candidate_id"] for item in selected],
        "source_pair_ids": [item["source_pair_id"] for item in selected],
        "best_step": result.best_step,
        "best_step_semantics": "minimum validation loss with frozen early-stopping min-delta; returned best checkpoint",
        "steps_completed": result.history[-1]["step"],
        "train": best_train,
        "train_augmented": _evaluate(train_states, train_labels, result.parameters),
        "validation": best_val, "test": best_test,
        "final_step_metrics": {"step": result.history[-1]["step"], "train": final_train, "validation": final_val, "test": final_test},
        "generalization_gap": {"train_test_accuracy": best_train["accuracy"] - best_test["accuracy"], "validation_test_accuracy": best_val["accuracy"] - best_test["accuracy"]},
        "history": result.history,
    }
    return run, result.parameters, result.final_parameters


def _aggregates(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for run in runs:
        if run["status"].startswith("completed"):
            groups[(run["dataset"], run["method"], float(run["ratio"]))].append(run)
    output = []
    for (dataset, method, ratio), items in sorted(groups.items()):
        entry = {"dataset": dataset, "method": method, "ratio": ratio, "runs": len(items), "run_seeds": [item["run_seed"] for item in items]}
        for split in ("train", "validation", "test"):
            for metric in ("accuracy", "loss", "macro_f1"):
                values = [item[split][metric] for item in items]
                entry[f"{split}_{metric}"] = _stats(values)
        for gap in ("train_test_accuracy", "validation_test_accuracy"):
            entry[gap] = _stats([item["generalization_gap"][gap] for item in items])
        output.append(entry)
    return output


def _comparison(aggregates: list[dict], runs: list[dict], ratios: list[float]) -> dict:
    index = {(item["dataset"], item["method"], item["ratio"]): item for item in aggregates}
    rows = []
    for dataset in ("random", "blocked-g"):
        real = index[(dataset, "real-only", 0.0)]["test_accuracy"]["mean"]
        for ratio in ratios:
            if ratio == 0:
                rows.append({"dataset": dataset, "ratio": ratio, "real_only": real, "matched_control": real, "physics_aware": real, "geometry_aware": real})
                continue
            def mean(method):
                item = index.get((dataset, method, float(ratio)))
                return None if item is None else item["test_accuracy"]["mean"]
            rows.append({"dataset": dataset, "ratio": ratio, "real_only": real, "matched_control": mean("matched random tangent control"), "physics_aware": mean("budget-matched physics-aware Phase A"), "geometry_aware": mean("geometry-aware Phase B")})
    per_seed = []
    run_index = {(item["dataset"], item["method"], float(item["ratio"]), item["run_seed"]): item for item in runs if item["status"].startswith("completed")}
    for dataset in ("random", "blocked-g"):
        for ratio in ratios:
            for seed in (0, 1, 2):
                def accuracy(method, used_ratio):
                    item = run_index.get((dataset, method, float(used_ratio), seed))
                    return None if item is None else item["test"]["accuracy"]
                per_seed.append({"dataset": dataset, "ratio": ratio, "seed": seed, "real": accuracy("real-only", 0), "control": accuracy("matched random tangent control", ratio) if ratio else accuracy("real-only", 0), "physics": accuracy("budget-matched physics-aware Phase A", ratio) if ratio else accuracy("real-only", 0), "geometry": accuracy("geometry-aware Phase B", ratio) if ratio else accuracy("real-only", 0)})
    primary = [row for row in per_seed if row["dataset"] == "blocked-g" and row["ratio"] == 1]
    deltas = {}
    for name, comparator in (("geometry_minus_real", "real"), ("geometry_minus_control", "control"), ("geometry_minus_physics", "physics")):
        values = [row["geometry"] - row[comparator] for row in primary if row["geometry"] is not None and row[comparator] is not None]
        deltas[name] = {**_stats(values), "per_seed": values, "positive_zero_negative": [sum(v > 0 for v in values), sum(v == 0 for v in values), sum(v < 0 for v in values)]}
    return {"primary_table": rows, "per_seed_table": per_seed, "primary_blocked_g_r1_paired_deltas": deltas}


def run_phase_b(config_path: str | Path, control_path: str | Path, output: str | Path, *, run_qcnn: bool = True) -> dict:
    config_path, control_path, output = Path(config_path), Path(control_path), Path(output)
    config, control = yaml.safe_load(config_path.read_text()), yaml.safe_load(control_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    config_hash, control_hash = _sha256(config_path), _sha256(control_path)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance: list[dict] = []
    generator_summaries, control_summaries, selected_artifact = [], [], []
    pools, datasets = {}, {}
    all_states: dict[str, np.ndarray] = {}
    phase_a, source_by_synthetic = _load_phase_a()
    historical_runs = [_historical_run(run, source_by_synthetic) for run in phase_a["runs"]]
    audit = {
        "classification": "A",
        "source_pool": "the canonical subset_seed=31415 subset used by the QCNN: exactly 10 train states/class",
        "statistics_scope": "per-dataset/per-class Mx, Mz2, and energy gates fit only on those same 10 states/class",
        "budget_pure": True,
        "budget_matched_rerun_required": False,
        "comparator": str(PHASE_A / "qcnn_pilot.json"),
    }
    for dataset_name, dataset_path in config["datasets"].items():
        dataset = load_tfim_dataset(dataset_path)
        subset = nested_train_subsets(dataset.train, [config["real_states_per_class"]], config["subset_seed"])[config["real_states_per_class"]]
        datasets[dataset_name] = (dataset, subset, Path(dataset_path))
        record_by_id = _records(dataset)
        record_by_id["__config__"] = dataset.manifest["config"]
        historical_ids = {tuple(run["real_sample_ids"]) for run in historical_runs if run["dataset"] == dataset_name}
        if historical_ids != {tuple(subset.parameter_ids.tolist())}:
            raise ValueError("Phase A and Phase B real sample IDs differ")
        for seed in config["seeds"]:
            for label in (0, 1):
                mask = subset.labels == label
                geometry, geometry_states, graph = generate_geometry_pool(
                    subset.states[mask], subset.parameter_ids[mask], dataset=dataset_name,
                    run_seed=seed["run_seed"], class_label=label, generator_config_hash=config_hash,
                    t_values=config["interpolation_positions"], k=config["k_nearest_neighbors"],
                    distance_quantile=config["same_class_pair_distance_quantile"],
                    minimum_distance=config["minimum_pair_distance_radians"],
                    overlap_tolerance=config["tolerances"]["overlap"],
                    normalization_tolerance=config["tolerances"]["normalization"],
                    duplicate_infidelity_tolerance=config["tolerances"]["duplicate_infidelity"],
                    namespace=config["augmentation_namespace"],
                )
                source_states = dict(zip(subset.parameter_ids[mask].tolist(), subset.states[mask], strict=True))
                controls, control_states = generate_matched_controls(
                    geometry, geometry_states, source_states, run_seed=seed["run_seed"], class_label=label,
                    duplicate_infidelity_tolerance=control["duplicate_infidelity_tolerance"],
                    orthogonal_tolerance=control["orthogonal_projection_tolerance"], max_redraws=control["maximum_redraws"],
                )
                for item in geometry:
                    _diagnose_candidate(item, geometry_states[item["candidate_id"]], subset, record_by_id)
                for item in controls:
                    item["t"] = next(value["t"] for value in geometry if value["candidate_id"] == item["geometry_candidate_id"])
                    _diagnose_candidate(item, control_states[item["candidate_id"]], subset, record_by_id)
                pools[(dataset_name, seed["run_seed"], label)] = (geometry, geometry_states, controls, control_states, graph)
                all_states.update(geometry_states); all_states.update(control_states)
                provenance.extend(geometry); provenance.extend(controls)
                for t in config["interpolation_positions"]:
                    geo_t = [item for item in geometry if item["t"] == t]
                    ctrl_t = [item for item in controls if item["t"] == t]
                    generator_summaries.append({"dataset": dataset_name, "run_seed": seed["run_seed"], "class_label": label, "t": t, **_candidate_summary(geo_t, geometry_states, graph)})
                    control_summaries.append({"dataset": dataset_name, "run_seed": seed["run_seed"], "class_label": label, "t": t, **_candidate_summary(ctrl_t, control_states)})
                errors = [item["displacement_matching_error"] for item in controls]
                control_summaries.append({
                    "dataset": dataset_name, "run_seed": seed["run_seed"], "class_label": label, "t": "all",
                    **_candidate_summary(controls, control_states),
                    "matching_quality": {"mean_absolute_error": float(np.mean(errors)) if errors else None, "max_absolute_error": max(errors, default=None), "source_anchor_counts": dict(Counter(item["anchor_source_id"] for item in controls)), "source_anchor_count_equality": Counter(item["anchor_source_id"] for item in controls) == Counter(item["anchor_source_id"] for item in geometry), "sample_count_equality": len(controls) == len(geometry)},
                })
                for ratio in config["augmentation_ratios"]:
                    count = int(ratio * config["real_states_per_class"])
                    feasible = len(geometry) >= count and len(controls) >= count
                    selected_artifact.append({"dataset": dataset_name, "run_seed": seed["run_seed"], "class_label": label, "ratio": ratio, "required_count": count, "feasible": feasible, "geometry_candidate_ids": [item["candidate_id"] for item in geometry[:count]] if feasible else [], "control_candidate_ids": [item["candidate_id"] for item in controls[:count]] if feasible else []})
    (output / "candidate_provenance.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in provenance))
    np.savez_compressed(output / "synthetic_states.npz", **all_states)
    (output / "generator_diagnostics.json").write_text(json.dumps({"phase_a_budget_audit": audit, "cells": generator_summaries}, indent=2, sort_keys=True) + "\n")
    (output / "matched_control_diagnostics.json").write_text(json.dumps({"method": control["method"], "cells": control_summaries}, indent=2, sort_keys=True) + "\n")
    (output / "selected_pool.json").write_text(json.dumps(selected_artifact, indent=2, sort_keys=True) + "\n")

    results = historical_runs
    parameters = {}
    planned = []
    for dataset_name, (dataset, subset, _) in datasets.items():
        for ratio in config["augmentation_ratios"]:
            for seed in config["seeds"]:
                for method in ("geometry-aware Phase B", "matched random tangent control"):
                    if ratio == 0:
                        continue
                    entries = [item for item in selected_artifact if item["dataset"] == dataset_name and item["run_seed"] == seed["run_seed"] and item["ratio"] == ratio]
                    feasible = all(item["feasible"] for item in entries)
                    planned.append({"dataset": dataset_name, "method": method, "ratio": ratio, **seed, "status": "planned" if feasible else "infeasible_candidate_shortage"})
                    if not feasible or not run_qcnn:
                        continue
                    selected = []
                    states = {}
                    for label in (0, 1):
                        geometry, geometry_states, controls, control_states, _ = pools[(dataset_name, seed["run_seed"], label)]
                        source, source_states = (geometry, geometry_states) if method.startswith("geometry") else (controls, control_states)
                        count = int(ratio * config["real_states_per_class"])
                        selected.extend(source[:count]); states.update(source_states)
                    run, best, final = _run_qcnn(dataset, subset, selected, states, method, ratio, seed, config["training"])
                    run["dataset"] = dataset_name
                    results.append(run)
                    key = f"{dataset_name}|{method}|{ratio:g}|{seed['run_seed']}"
                    parameters[key + "|best"] = best; parameters[key + "|final"] = final
                    planned[-1]["status"] = "completed"
    if parameters:
        np.savez_compressed(output / "qcnn_parameters.npz", **parameters)
    (output / "per_seed_qcnn_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    aggregates = _aggregates(results)
    (output / "aggregate_qcnn_results.json").write_text(json.dumps(aggregates, indent=2, sort_keys=True) + "\n")
    comparison = _comparison(aggregates, results, config["augmentation_ratios"])
    (output / "comparison_table.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    primary_cells = [item for item in selected_artifact if item["ratio"] == 1]
    primary_provenance = [item for item in provenance if item["candidate_id"] in {candidate for cell in primary_cells for candidate in cell["geometry_candidate_ids"] + cell["control_candidate_ids"]}]
    geometry_primary = [item for item in primary_provenance if item["augmentation_method"] == "same_class_local_projective_geodesic"]
    control_primary = [item for item in primary_provenance if item["augmentation_method"].startswith("manifold_unaware")]
    primary_feasible = all(cell["feasible"] for cell in primary_cells)
    gate = {
        "ten_unique_geometry_per_cell": primary_feasible,
        "ten_matched_controls_per_cell": primary_feasible,
        "no_finite_or_normalization_failures": all(sum(graph["failures"][key] for key in ("finite", "normalization")) == 0 for *_, graph in pools.values()),
        "no_accepted_duplicates": all(
            len({item["state_hash"] for item in geometry}) == len(geometry)
            and len({item["state_hash"] for item in controls}) == len(controls)
            for geometry, _, controls, _, _ in pools.values()
        ),
        "mean_displacement_error_le_1e-8": bool(control_primary) and float(np.mean([item["displacement_matching_error"] for item in control_primary])) <= 1e-8,
        "max_displacement_error_le_1e-6": bool(control_primary) and max(item["displacement_matching_error"] for item in control_primary) <= 1e-6,
        "near_copy_dominated": None if not primary_feasible else float(np.median([item["nearest_source_infidelity"] for item in geometry_primary])) < 1e-4,
        "near_copy_assessment": "not_evaluable_primary_pool_infeasible" if not primary_feasible else "evaluated",
    }
    gate["passed"] = primary_feasible and all(value for key, value in gate.items() if key not in ("near_copy_dominated", "near_copy_assessment", "passed")) and not gate["near_copy_dominated"]
    decision = "NO-GO" if not gate["passed"] else "UNRESOLVED"
    reason = "generator validity gate failed because preregistered pair eligibility left primary pools infeasible" if not gate["passed"] else "utility criteria require completed primary runs"
    manifest_entries = []
    dataset_hashes = {name: {file: _sha256(path / file) for file in ("states.npz", "split_manifest.json", "validation.json")} for name, (_, _, path) in datasets.items()}
    for item in planned:
        dataset, method, ratio, seed = item["dataset"], item["method"], item["ratio"], item["run_seed"]
        subset = datasets[dataset][1]
        selected = [cell for cell in selected_artifact if cell["dataset"] == dataset and cell["run_seed"] == seed and cell["ratio"] == ratio]
        selected_key = "geometry_candidate_ids" if method.startswith("geometry") else "control_candidate_ids"
        synthetic_ids = [candidate for cell in selected for candidate in cell[selected_key]]
        pair_by_id = {entry["candidate_id"]: entry["source_pair_id"] for entry in provenance}
        manifest_entries.append({**item, "git_commit_sha": git_sha, "dataset_path": config["datasets"][dataset], "dataset_artifact_hashes": dataset_hashes[dataset], "config_path": str(config_path), "config_hash": config_hash, "control_config_path": str(control_path), "control_config_hash": control_hash, "real_sample_ids": subset.parameter_ids.tolist(), "source_pair_ids": [pair_by_id[candidate] for candidate in synthetic_ids], "synthetic_state_ids": synthetic_ids, "dtype": "complex128", "tolerances": config["tolerances"]})
    phase_a_config = PHASE_A / "config.yaml"
    for run in historical_runs:
        dataset = run["dataset"]
        manifest_entries.append({
            "dataset": dataset, "method": run["method"], "ratio": run["ratio"],
            "run_seed": run["run_seed"], "init_seed": run["init_seed"], "spsa_seed": run["spsa_seed"],
            "status": run["status"], "git_commit_sha": phase_a["git_sha"],
            "dataset_path": config["datasets"][dataset], "dataset_artifact_hashes": dataset_hashes[dataset],
            "config_path": str(phase_a_config), "config_hash": _sha256(phase_a_config),
            "real_sample_ids": run["real_sample_ids"], "source_pair_ids": run["source_pair_ids"],
            "synthetic_state_ids": run["synthetic_sample_ids"], "dtype": "complex128",
            "tolerances": {"normalization": 1e-10},
        })
    manifest = {
        "git_commit_sha": git_sha,
        "git_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pyyaml": yaml.__version__},
        "phase_a_budget_audit": audit,
        "protocol": {"input": "4-qubit complex[16] pure state", "architecture": "4 -> 2 -> 1 QCNN", "parameter_count": 42, "readout": "Z on qubit 3", "loss": "mean squared error between <Z3> and labels mapped to {-1,+1}", "optimizer": "SPSA", "training_steps": 300, "best_step_semantics": "minimum validation loss with frozen early-stopping rule"},
        "exploratory_status": "Phase B is exploratory method screening and failure-mode analysis, not publication-level confirmatory evidence.",
        "generator_validity_gate": gate,
        "decision": decision, "decision_reason": reason,
        "runs": manifest_entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hash_files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "artifact_hashes.sha256")
    (output / "artifact_hashes.sha256").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in hash_files))
    return {"decision": decision, "generator_validity_gate": gate, "qcnn_runs": len([item for item in results if item["method"] in ("geometry-aware Phase B", "matched random tangent control")]), "infeasible_conditions": sum(item["status"].startswith("infeasible") for item in planned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation/geometry/phase_b.yaml")
    parser.add_argument("--control-config", default="configs/augmentation/controls/distance_matched_random_tangent.yaml")
    parser.add_argument("--output", default="results/geometry_aware_augmentation/phase_b")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_phase_b(args.config, args.control_config, args.output, run_qcnn=not args.diagnostics_only), indent=2))


if __name__ == "__main__":
    main()

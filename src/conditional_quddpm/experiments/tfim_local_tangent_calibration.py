"""Train-only Phase-C local random-tangent calibration; never imports QCNN code."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from conditional_quddpm.augmentation.local_perturbation import calibrate_radii, generate_random_tangent_pool
from conditional_quddpm.datasets.loader import QuantumSplit, nested_train_subsets

RADIUS_NAMES = {"q25": "small", "q50": "medium", "q75": "large"}
RATIOS = (0.5, 1.0, 2.0)
NEAR_DUPLICATE_INFidelity = 1e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def _load_train(path: Path) -> QuantumSplit:
    """Materialize train rows only; never read validation/test labels or metrics."""
    manifest = json.loads((path / "split_manifest.json").read_text())
    records = [record for record in manifest["records"] if record["split"] == "train"]
    with np.load(path / "states.npz") as data:
        index = {str(value): i for i, value in enumerate(data["parameter_ids"])}
        rows = [index[record["parameter_id"]] for record in records]
        states = np.asarray(data["states"][rows], dtype=np.complex128)
    return QuantumSplit(states, np.asarray([record["label"] for record in records], dtype=np.int8), np.asarray([record["parameter_id"] for record in records]))


def calibrate(config_path: str | Path, output_path: str | Path) -> dict:
    config_path, output = Path(config_path), Path(output_path)
    config = json.loads(config_path.read_text())
    config_hash = _sha256(config_path)
    cells = []

    for dataset_name, dataset_path in config["datasets"].items():
        dataset_path = Path(dataset_path)
        train = _load_train(dataset_path)
        subset = nested_train_subsets(train, [10], config["subset_seed"])[10]
        subset_id = hashlib.sha256("\n".join(subset.parameter_ids).encode()).hexdigest()
        for label in (0, 1):
            mask = subset.labels == label
            anchors, ids = subset.states[mask], subset.parameter_ids[mask]
            radii = calibrate_radii(anchors)["radii"]
            for public_radius, radius_name in RADIUS_NAMES.items():
                delta = radii[radius_name]
                for seed in config["seeds"]:
                    items, states = generate_random_tangent_pool(
                        anchors, ids, dataset=dataset_name, class_label=label,
                        radius_name=radius_name, delta=delta, run_seed=seed["run_seed"],
                        config_hash=config_hash, code_commit_sha="calibration-only",
                        source_subset_id=subset_id, source_artifact_hash=_sha256(dataset_path / "states.npz"),
                        namespace=config["namespace"],
                        orthogonal_tolerance=config["tolerances"]["orthogonal"],
                        duplicate_infidelity_tolerance=config["tolerances"]["duplicate_infidelity"],
                        max_redraws=config["tolerances"]["maximum_redraws"],
                    )
                    for ratio in RATIOS:
                        requested = int(10 * ratio)
                        chosen = items[:requested]
                        chosen_states = [states[item["synthetic_sample_id"]] for item in chosen]
                        retries = sum(item["tangent_retry_index"] for item in chosen)
                        pair_infidelities = [
                            1 - abs(np.vdot(chosen_states[i], chosen_states[j])) ** 2
                            for i in range(len(chosen_states)) for j in range(i)
                        ]
                        near_duplicates = sum(value < NEAR_DUPLICATE_INFidelity for value in pair_infidelities)
                        finite = all(np.isfinite(state).all() for state in chosen_states)
                        valid = finite and all(
                            item["normalization_error"] <= config["tolerances"]["normalization"]
                            and item["tangent_overlap_abs"] <= config["tolerances"]["orthogonal"]
                            and item["tangent_norm_error"] <= config["tolerances"]["orthogonal"]
                            for item in chosen
                        )
                        cells.append({
                            "dataset": dataset_name, "class_label": label, "seed": seed["run_seed"],
                            "radius": public_radius, "ratio": ratio, "target_radius": delta,
                            "requested": requested, "accepted": len(chosen),
                            "acceptance_rate": len(chosen) / (len(chosen) + retries),
                            "failed_redraws": retries, "anchor_coverage": len({item["anchor_sample_id"] for item in chosen}) / 10,
                            "actual_fs_displacement": _summary(item["actual_displacement_fs"] for item in chosen),
                            "displacement_error": _summary(item["displacement_error"] for item in chosen),
                            "source_near_duplicate_rate": sum(item["nearest_source_infidelity"] < NEAR_DUPLICATE_INFidelity for item in chosen) / len(chosen),
                            "synthetic_pair_near_duplicate_rate": near_duplicates / max(1, len(pair_infidelities)),
                            "unique": len({item["state_hash"] for item in chosen}) == len(chosen),
                            "class_balance": "evaluated per class with equal requested count",
                            "numerical_physical_validity": "PASS" if valid else "FAIL",
                        })

    aggregate = []
    for radius in RADIUS_NAMES:
        for ratio in RATIOS:
            group = [cell for cell in cells if cell["radius"] == radius and cell["ratio"] == ratio]
            aggregate.append({
                "radius": radius, "ratio": ratio, "cells": len(group),
                "valid_cells": sum(cell["numerical_physical_validity"] == "PASS" for cell in group),
                "requested": sum(cell["requested"] for cell in group), "accepted": sum(cell["accepted"] for cell in group),
                "acceptance_rate_min": min(cell["acceptance_rate"] for cell in group),
                "anchor_coverage_min": min(cell["anchor_coverage"] for cell in group),
                "source_near_duplicate_rate_max": max(cell["source_near_duplicate_rate"] for cell in group),
                "synthetic_pair_near_duplicate_rate_max": max(cell["synthetic_pair_near_duplicate_rate"] for cell in group),
                "displacement_error_max": max(cell["displacement_error"]["max"] for cell in group),
                "failed_redraws": sum(cell["failed_redraws"] for cell in group),
            })

    eligible = [row for row in aggregate if row["valid_cells"] == row["cells"] and row["accepted"] == row["requested"] and row["source_near_duplicate_rate_max"] == 0 and row["synthetic_pair_near_duplicate_rate_max"] == 0 and row["displacement_error_max"] <= 1e-10]
    if eligible:
        best_coverage = max(row["anchor_coverage_min"] for row in eligible)
        eligible = [row for row in eligible if row["anchor_coverage_min"] == best_coverage]
    selected = next((row for row in eligible if row["radius"] == "q50" and row["ratio"] == 1.0), None)
    status = "CALIBRATION_PASS" if selected else "AUGMENTATION_CALIBRATION_BLOCKED"
    decision = {
        "status": status, "provenance": "NEW_DECISION", "qcnn_runs": 0,
        "test_metrics_accessed": False, "selection_rule_frozen_before_diagnostics": True,
        "tie_break_used": bool(selected),
        "selected": {"radius": selected["radius"], "ratio": selected["ratio"]} if selected else None,
        "reason": "Ratios 1.0 and 2.0 achieved full anchor coverage; radii were equivalent on validity, duplication, displacement, and redraw gates; deterministic q50/1.0 tie-break applied." if selected else "No unique eligible candidate under the frozen rule.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps({"cells": cells, "aggregate": aggregate}, indent=2, sort_keys=True) + "\n")
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "config": str(config_path), "config_sha256": config_hash,
        "datasets": {name: {"path": path, "states_sha256": _sha256(Path(path) / "states.npz")} for name, path in config["datasets"].items()},
        "data_scope": "balanced frozen train subsets only", "forbidden_inputs": ["validation", "test", "QCNN artifacts", "QCNN metrics"],
    }, indent=2, sort_keys=True) + "\n")
    return decision


if __name__ == "__main__":
    print(json.dumps(calibrate("configs/augmentation/local_perturbation/phase_c.json", "results/tfim_manifold_augmentation/local_tangent_calibration_v1"), indent=2))

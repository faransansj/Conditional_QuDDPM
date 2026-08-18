"""K2: frozen per-realization gradient conflict diagnostic for global MMD.

The biased MMD kernel summands are allocated by rows.  For class c and one
measurement outcome,

  A_ij = (k(g_i,g_j)+k(t_i,t_j)-k(g_i,t_j)-k(t_i,g_j))/N^2,
  a_i = sum_j A_ij.

Each reported realization objective is N*a_i.  Consequently its mean over all
2N class/realization tasks exactly reconstructs the equally class-weighted
global MMD.  No clipping is applied.  Gradients are K1-compatible central
finite-difference directional sketches, not autodiff gradients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.kernel_diagnostics import provenance
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.experiments.q2_objective_geometry import train
from conditional_quddpm.models.quddpm import condition_angles, reverse_parameter_count, reverse_step
from conditional_quddpm.models.rdm_kernels import kernel_matrix, kernel_mmd_raw
from conditional_quddpm.datasets.tfim import tfim_observables

SCHEMA_VERSION = "k2-realization-conflict-v1"
REQUIRED_CONFIG = {
    "schema_version", "dataset", "subset_seed", "train_states_per_class",
    "train_realizations", "diffusion_steps", "layers", "iterations", "ancillas",
    "measurement_outcomes", "trained_objective", "checkpoint", "step", "directions",
    "epsilon", "directional_step", "near_zero_threshold", "reconstruction", "spsa", "seeds",
}
REQUIRED_MANIFEST = {
    "schema_version", "git_commit", "branch", "checkpoint", "checkpoint_hash",
    "dataset", "dataset_hashes", "train_state_ids", "realizations", "config_hash",
    "seeds", "gradient_estimator", "epsilon", "directional_step", "parameter_count",
    "parameter_order", "dtype", "reconstruction_tolerance", "test_split_used", "run_id",
}


def validate_required(mapping, required, name):
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(missing)}")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def safe_cosine(left, right, threshold=1e-12):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    ln, rn = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if ln <= threshold or rn <= threshold:
        reason = "both_near_zero" if ln <= threshold and rn <= threshold else "left_near_zero" if ln <= threshold else "right_near_zero"
        return {"cosine": None, "status": reason}
    return {"cosine": float(np.dot(left, right) / (ln * rn)), "status": "valid"}


def mmd_contributions(generated, target, kernel="global"):
    """Return the exact row allocation of one raw biased empirical MMD."""
    n = len(target)
    if len(generated) != n or n == 0:
        raise ValueError("generated and target must have the same nonzero realization count")
    gg = kernel_matrix(generated, generated, kernel)
    tt = kernel_matrix(target, target, kernel)
    gt = kernel_matrix(generated, target, kernel)
    matrix = (gg + tt - gt - gt.T) / (n * n)
    return matrix, matrix.sum(axis=1)


def _state_observables(states):
    return np.asarray([tfim_observables(state, 4) for state in states], dtype=float)


def _class_separation(generated):
    within = []
    for c in (0, 1):
        gram = kernel_matrix(generated[c], generated[c], "global")
        mask = ~np.eye(len(gram), dtype=bool)
        within.append(float(gram[mask].mean()) if mask.any() else 1.0)
    between = float(kernel_matrix(generated[0], generated[1], "global").mean())
    return float(np.mean(within) - between)


def evaluate_point(parameters, path, step, angles, uniforms):
    """Evaluate all K2 objectives from one common generated ensemble."""
    n = len(path[0][step])
    class_global, class_2rdm, class_physics = {}, {}, {}
    class_mx, class_mz2 = {}, {}
    tasks = np.empty((2, n), dtype=float)
    task_physics = np.empty((2, n), dtype=float)
    pooled = {}
    for c in (0, 1):
        target = path[c][step]
        target_obs = _state_observables(target)
        generated = [reverse_step(path[c][step + 1], parameters, angles[c], row) for row in uniforms[c]]
        pooled[c] = np.concatenate(generated)
        global_values, rdm_values, contribution_values, physics_values = [], [], [], []
        per_mx, per_mz2, paired_physics = [], [], []
        target_mean = target_obs.mean(axis=0)
        for batch in generated:
            matrix, rows = mmd_contributions(batch, target, "global")
            global_values.append(float(matrix.sum()))
            contribution_values.append(n * rows)
            rdm_values.append(kernel_mmd_raw(batch, target, "2-rdm"))
            generated_obs = _state_observables(batch)
            delta = generated_obs - target_obs
            paired_physics.append(0.5 * np.sum(delta * delta, axis=1))
            mean_delta = generated_obs.mean(axis=0) - target_mean
            per_mx.append(float(mean_delta[0] ** 2))
            per_mz2.append(float(mean_delta[1] ** 2))
            physics_values.append(float(0.5 * np.sum(mean_delta * mean_delta)))
        class_global[c] = float(np.mean(global_values))
        class_2rdm[c] = float(np.mean(rdm_values))
        class_physics[c] = float(np.mean(physics_values))
        class_mx[c] = float(np.mean(per_mx))
        class_mz2[c] = float(np.mean(per_mz2))
        tasks[c] = np.mean(contribution_values, axis=0)
        task_physics[c] = np.mean(paired_physics, axis=0)
    return {
        "global": float(np.mean(list(class_global.values()))),
        "class_global": class_global,
        "tasks": tasks,
        "2rdm": float(np.mean(list(class_2rdm.values()))),
        "class_2rdm": class_2rdm,
        "physics": float(np.mean(list(class_physics.values()))),
        "class_physics": class_physics,
        "mx_error": float(np.mean(list(class_mx.values()))),
        "mz2_error": float(np.mean(list(class_mz2.values()))),
        "task_physics": task_physics,
        "class_separation": _class_separation(pooled),
    }


def directional_sketch(derivatives, directions):
    derivatives = np.asarray(derivatives, dtype=float)
    flat_directions = np.asarray(directions, dtype=float).reshape(len(directions), -1)
    return np.mean(derivatives[..., None] * flat_directions, axis=-2)


def gradient_record(vector, source, near_zero):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return {
        "source_objective": source,
        "raw": vector,
        "norm": norm,
        "normalized": vector / norm if norm > near_zero else np.zeros_like(vector),
        "status": "valid" if norm > near_zero else "near_zero",
    }


def _relative_error(actual, expected, floor=1e-15):
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / max(np.linalg.norm(expected), floor))


def _pair_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: None for key in ("count", "mean", "median", "std", "min", "max", "negative_fraction", "positive_fraction")}
    return {
        "count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
        "negative_fraction": float(np.mean(values < 0)), "positive_fraction": float(np.mean(values > 0)),
    }


def pairwise_analysis(task_vectors, metadata, near_zero):
    count = len(task_vectors)
    matrix = np.full((count, count), np.nan)
    rows = []
    groups = {"within_class": [], "between_class": [], "same_g": [], "far_g": []}
    for i in range(count):
        for j in range(count):
            result = safe_cosine(task_vectors[i], task_vectors[j], near_zero)
            if result["cosine"] is not None:
                matrix[i, j] = result["cosine"]
            if j <= i:
                continue
            same_class = metadata[i]["class"] == metadata[j]["class"]
            g_gap = abs(metadata[i]["g"] - metadata[j]["g"])
            group = "within_class" if same_class else "between_class"
            band = "same_g" if np.isclose(g_gap, 0.0) else "far_g"
            row = {
                "left_id": metadata[i]["realization_id"], "right_id": metadata[j]["realization_id"],
                "left_class": metadata[i]["class"], "right_class": metadata[j]["class"],
                "g_gap": g_gap, "group": group, "g_band": band,
                "cosine": result["cosine"], "status": result["status"],
            }
            rows.append(row)
            if result["cosine"] is not None:
                groups[group].append(result["cosine"])
                groups[band].append(result["cosine"])
    return matrix, rows, {name: _pair_stats(values) for name, values in groups.items()}


def cancellation_ratio(vectors, near_zero):
    vectors = np.asarray(vectors, dtype=float)
    denominator = float(np.mean(np.linalg.norm(vectors, axis=1)))
    numerator = float(np.linalg.norm(vectors.mean(axis=0)))
    return None if denominator <= near_zero else numerator / denominator


def _delta(after, before):
    return float(after - before)


def probe_direction(name, record, parameters, center, path, step, angles, uniforms, step_size,
                    target_index=None, class_index=None):
    row = {"direction": name, "source_status": record["status"], "source_norm": record["norm"],
           "target_index": target_index, "class": class_index}
    if record["status"] != "valid":
        row["probe_status"] = "skipped_near_zero"
        return row
    candidate = parameters - step_size * record["normalized"].reshape(parameters.shape)
    after = evaluate_point(candidate, path, step, angles, uniforms)
    row.update({
        "probe_status": "evaluated", "global_mmd_delta": _delta(after["global"], center["global"]),
        "2rdm_delta": _delta(after["2rdm"], center["2rdm"]),
        "physics_delta": _delta(after["physics"], center["physics"]),
        "mx_error_delta": _delta(after["mx_error"], center["mx_error"]),
        "mz2_error_delta": _delta(after["mz2_error"], center["mz2_error"]),
        "class_separation_delta": _delta(after["class_separation"], center["class_separation"]),
    })
    before_tasks, after_tasks = center["tasks"].reshape(-1), after["tasks"].reshape(-1)
    task_delta = after_tasks - before_tasks
    if target_index is not None:
        target_class = target_index // center["tasks"].shape[1]
        row.update({
            "target_objective_delta": float(task_delta[target_index]),
            "non_target_mean_delta": float(np.delete(task_delta, target_index).mean()),
            "same_class_non_target_mean_delta": float(np.delete(task_delta.reshape(2, -1)[target_class], target_index % center["tasks"].shape[1]).mean()),
            "other_class_mean_delta": float(task_delta.reshape(2, -1)[1 - target_class].mean()),
            "target_physics_delta": float(after["task_physics"].reshape(-1)[target_index] - center["task_physics"].reshape(-1)[target_index]),
        })
    elif class_index is not None:
        row.update({
            "target_objective_delta": float(after["class_global"][class_index] - center["class_global"][class_index]),
            "non_target_mean_delta": float(after["class_global"][1 - class_index] - center["class_global"][1 - class_index]),
            "same_class_non_target_mean_delta": None, "other_class_mean_delta": float(after["class_global"][1 - class_index] - center["class_global"][1 - class_index]),
            "target_physics_delta": float(after["class_physics"][class_index] - center["class_physics"][class_index]),
        })
    else:
        row.update({"target_objective_delta": None, "non_target_mean_delta": float(task_delta.mean()),
                    "same_class_non_target_mean_delta": None, "other_class_mean_delta": None,
                    "target_physics_delta": None})
    return row


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_g(dataset, state_ids):
    records = {record["parameter_id"]: record for record in dataset.manifest["records"]}
    return {state_id: float(records[state_id]["g"]) for state_id in state_ids}


def analyze(parameters, path, step, angles, uniforms, directions, config, metadata):
    epsilon = config["epsilon"]
    near_zero = config["near_zero_threshold"]
    center = evaluate_point(parameters, path, step, angles, uniforms)
    global_d, class_d, task_d, rdm_d, physics_d = [], [[], []], [[[] for _ in range(config["train_realizations"])] for _ in range(2)], [], []
    direction_rows = []
    atol, rtol = config["reconstruction"]["atol"], config["reconstruction"]["rtol"]
    for index, direction in enumerate(directions):
        plus = evaluate_point(parameters + epsilon * direction, path, step, angles, uniforms)
        minus = evaluate_point(parameters - epsilon * direction, path, step, angles, uniforms)
        gd = (plus["global"] - minus["global"]) / (2 * epsilon)
        rd = (plus["2rdm"] - minus["2rdm"]) / (2 * epsilon)
        pd = (plus["physics"] - minus["physics"]) / (2 * epsilon)
        td = (plus["tasks"] - minus["tasks"]) / (2 * epsilon)
        cd = [(plus["class_global"][c] - minus["class_global"][c]) / (2 * epsilon) for c in (0, 1)]
        reconstructed = float(td.mean())
        residual = reconstructed - gd
        tolerance = atol + rtol * abs(gd)
        global_d.append(gd); rdm_d.append(rd); physics_d.append(pd)
        for c in (0, 1):
            class_d[c].append(cd[c])
            for i in range(config["train_realizations"]): task_d[c][i].append(float(td[c, i]))
        direction_rows.append({"direction": index, "global_derivative": gd, "class0_derivative": cd[0], "class1_derivative": cd[1],
                               "2rdm_derivative": rd, "physics_derivative": pd, "reconstructed_global": reconstructed,
                               "reconstruction_residual": residual, "reconstruction_tolerance": tolerance,
                               "reconstruction_pass": abs(residual) <= tolerance})
    global_record = gradient_record(directional_sketch(global_d, directions), "global_mmd", near_zero)
    class_records = [gradient_record(directional_sketch(class_d[c], directions), f"class_{c}_global_mmd", near_zero) for c in (0, 1)]
    task_records = []
    for c in (0, 1):
        for i in range(config["train_realizations"]):
            task_records.append(gradient_record(directional_sketch(task_d[c][i], directions), f"realization_{c}_{i}_decomposed_global_mmd", near_zero))
    rdm_record = gradient_record(directional_sketch(rdm_d, directions), "2rdm_mmd", near_zero)
    physics_record = gradient_record(directional_sketch(physics_d, directions), "tfim_observable_error", near_zero)
    reconstructed = np.mean([record["raw"] for record in task_records], axis=0)
    class_reconstructed = [np.mean([task_records[c * config["train_realizations"] + i]["raw"] for i in range(config["train_realizations"])], axis=0) for c in (0, 1)]
    reconstruction = {
        "directional_all_pass": bool(all(row["reconstruction_pass"] for row in direction_rows)),
        "max_directional_abs_error": float(max(abs(row["reconstruction_residual"]) for row in direction_rows)),
        "global_relative_error": _relative_error(reconstructed, global_record["raw"]),
        "class_relative_error": [_relative_error(class_reconstructed[c], class_records[c]["raw"]) for c in (0, 1)],
        "atol": atol, "rtol": rtol,
    }
    matrix, pair_rows, pair_stats = pairwise_analysis([record["raw"] for record in task_records], metadata, near_zero)
    cancellation = {
        "overall": cancellation_ratio([record["raw"] for record in task_records], near_zero),
        "class_0": cancellation_ratio([record["raw"] for record in task_records[:config["train_realizations"]]], near_zero),
        "class_1": cancellation_ratio([record["raw"] for record in task_records[config["train_realizations"]:]], near_zero),
    }
    summary_rows = []
    for index, (record, meta) in enumerate(zip(task_records, metadata, strict=True)):
        summary_rows.append({**meta, "gradient_norm": record["norm"], "gradient_status": record["status"],
                             "cosine_global": safe_cosine(record["raw"], global_record["raw"], near_zero)["cosine"],
                             "cosine_2rdm": safe_cosine(record["raw"], rdm_record["raw"], near_zero)["cosine"],
                             "cosine_class": safe_cosine(record["raw"], class_records[meta["class"]]["raw"], near_zero)["cosine"],
                             "center_objective": float(center["tasks"].reshape(-1)[index]),
                             "center_physics": float(center["task_physics"].reshape(-1)[index])})
    probes = []
    for index, record in enumerate(task_records):
        probes.append(probe_direction(f"realization_{metadata[index]['realization_id']}", record, parameters, center, path, step, angles, uniforms, config["directional_step"], index, metadata[index]["class"]))
    probes.append(probe_direction("global_mmd", global_record, parameters, center, path, step, angles, uniforms, config["directional_step"]))
    for c in (0, 1):
        probes.append(probe_direction(f"class_{c}_aggregate", class_records[c], parameters, center, path, step, angles, uniforms, config["directional_step"], class_index=c))
    probes.append(probe_direction("2rdm_mmd", rdm_record, parameters, center, path, step, angles, uniforms, config["directional_step"]))
    return {
        "center": center, "global": global_record, "classes": class_records, "tasks": task_records,
        "2rdm": rdm_record, "physics": physics_record, "reconstruction": reconstruction,
        "pairwise_matrix": matrix, "pairwise_rows": pair_rows, "pairwise_stats": pair_stats,
        "cancellation": cancellation, "summary_rows": summary_rows, "direction_rows": direction_rows,
        "probes": probes,
    }


def _json_gradient(record):
    return {"source_objective": record["source_objective"], "norm": record["norm"], "status": record["status"]}


def _report(summary):
    p = summary["pairwise_stats"]
    r = summary["reconstruction"]
    c = summary["cancellation"]
    probes = [row for row in summary["probes"] if row["direction"].startswith("realization_") and row["probe_status"] == "evaluated"]
    target = float(np.mean([row["target_physics_delta"] for row in probes]))
    other = float(np.mean([row["other_class_mean_delta"] for row in probes]))
    within = p["within_class"]; between = p["between_class"]
    return f"""# K2 per-realization gradient conflict diagnostic

Frozen global-MMD diagnostic at rho1->rho0 best checkpoint; train split only. The K1 central finite-difference directional sketch and common randomness are reused.

## Reconstruction

- Global sketch relative error: {r['global_relative_error']:.3e}
- Class sketch relative errors: {r['class_relative_error'][0]:.3e}, {r['class_relative_error'][1]:.3e}
- Maximum directional residual: {r['max_directional_abs_error']:.3e}

## Conflict

- Cancellation ratio overall / class 0 / class 1: {c['overall']:.4f} / {c['class_0']:.4f} / {c['class_1']:.4f}
- Within-class cosine mean / negative fraction: {within['mean']:.4f} / {within['negative_fraction']:.4f}
- Between-class cosine mean / negative fraction: {between['mean']:.4f} / {between['negative_fraction']:.4f}
- Mean realization-step target physics delta: {target:+.6f}
- Mean realization-step other-class objective delta: {other:+.6f}

No K3 training, generation gate, QCNN evaluation, or test split access was performed.
"""


def run(config, output):
    validate_required(config, REQUIRED_CONFIG, "config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {config['schema_version']}")
    started = time.perf_counter()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    dataset = load_tfim_dataset(config["dataset"])
    split = nested_train_subsets(dataset.train, [config["train_states_per_class"]], config["subset_seed"])[config["train_states_per_class"]]
    states = {c: split.states[split.labels == c] for c in (0, 1)}
    path, realization_ids = trajectories(states, config["train_realizations"], config["diffusion_steps"], config["seeds"]["train_forward"])
    angles = condition_angles([0, 1])
    step = config["step"] - 1
    n = config["train_realizations"]
    uniforms = {c: np.random.default_rng(config["seeds"]["measurement"] + 1000 * step + c).random((config["measurement_outcomes"], n)) for c in (0, 1)}
    shape = (config["layers"], reverse_parameter_count(4, config["ancillas"]))
    rng = np.random.default_rng(config["seeds"]["directions"])
    directions = [rng.choice((-1.0, 1.0), size=shape) for _ in range(config["directions"])]
    model, _, checkpoints = train(path, config, config["trained_objective"])
    parameters = checkpoints[step][config["checkpoint"]]
    state_ids = {c: split.parameter_ids[split.labels == c][0] for c in (0, 1)}
    g_values = _manifest_g(dataset, state_ids.values())
    metadata = []
    for c in (0, 1):
        for item in realization_ids[str(c)]:
            metadata.append({"realization_id": f"{state_ids[c]}/forward-{item['realization']}", "state_id": str(state_ids[c]),
                             "class": c, "g": g_values[state_ids[c]], "split": "train",
                             "forward_seed": item["forward_seed"], "measurement_seed": config["seeds"]["measurement"] + 1000 * step + c})
    analysis = analyze(parameters, path, step, angles, uniforms, directions, config, metadata)
    config_text = yaml.safe_dump(config, sort_keys=True)
    dataset_hashes = json.loads((Path(config["dataset"]) / "checksums.json").read_text())
    git = provenance()
    run_id = f"k2-{git['git_commit'][:8]}-s{config['step']}-{config['checkpoint']}"
    manifest = {
        "schema_version": SCHEMA_VERSION, **git, "run_id": run_id,
        "checkpoint": {"identifier": f"rho{config['step']}->rho{config['step']-1}/{config['checkpoint']}", "construction": "deterministic K1 SPSA replay"},
        "checkpoint_hash": sha256_bytes(np.ascontiguousarray(parameters).tobytes()),
        "dataset": config["dataset"], "dataset_hashes": dataset_hashes,
        "train_state_ids": [str(x) for x in split.parameter_ids], "realizations": metadata,
        "config_hash": sha256_bytes(config_text.encode()), "seeds": config["seeds"],
        "gradient_estimator": {"name": "central finite-difference Rademacher directional sketch", "directions": config["directions"],
                               "sketch": "mean_d directional_derivative[d] * rademacher_direction[d]", "common_random_numbers": True,
                               "mmd": "raw biased empirical global MMD; no clipping", "realization_allocation": "N times row sum of A_ij; mean over 2N tasks equals equal-class global MMD"},
        "epsilon": config["epsilon"], "directional_step": config["directional_step"],
        "parameter_count": int(parameters.size), "parameter_shape": list(parameters.shape), "parameter_order": "NumPy C-order flatten",
        "dtype": str(parameters.dtype), "reconstruction_tolerance": config["reconstruction"],
        "test_split_used": False, "validation_split_used": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    validate_required(manifest, REQUIRED_MANIFEST, "manifest")
    gradient_provenance = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "checkpoint_hash": manifest["checkpoint_hash"],
        "parameter_order": manifest["parameter_order"], "parameter_shape": manifest["parameter_shape"],
        "directions_seed": config["seeds"]["directions"], "measurement_seeds": sorted({m["measurement_seed"] for m in metadata}),
        "gradients": [_json_gradient(analysis["global"])] + [_json_gradient(x) for x in analysis["classes"]] +
                     [_json_gradient(x) for x in analysis["tasks"]] + [_json_gradient(analysis["2rdm"]), _json_gradient(analysis["physics"])],
    }
    summary = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "reconstruction": analysis["reconstruction"], "cancellation": analysis["cancellation"],
        "pairwise_stats": analysis["pairwise_stats"],
        "alignment": {
            "realization_vs_global": _pair_stats([row["cosine_global"] for row in analysis["summary_rows"] if row["cosine_global"] is not None]),
            "realization_vs_2rdm": _pair_stats([row["cosine_2rdm"] for row in analysis["summary_rows"] if row["cosine_2rdm"] is not None]),
            "global_vs_2rdm": safe_cosine(analysis["global"]["raw"], analysis["2rdm"]["raw"], config["near_zero_threshold"]),
            "global_vs_physics": safe_cosine(analysis["global"]["raw"], analysis["physics"]["raw"], config["near_zero_threshold"]),
            "2rdm_vs_physics": safe_cosine(analysis["2rdm"]["raw"], analysis["physics"]["raw"], config["near_zero_threshold"]),
        },
        "center": {key: analysis["center"][key] for key in ("global", "2rdm", "physics", "mx_error", "mz2_error", "class_separation")},
        "probes": analysis["probes"], "interpretation": "diagnostic_only_no_post_hoc_threshold",
    }
    arrays = {
        "global_raw": analysis["global"]["raw"], "global_normalized": analysis["global"]["normalized"],
        "class_raw": np.asarray([x["raw"] for x in analysis["classes"]]),
        "class_normalized": np.asarray([x["normalized"] for x in analysis["classes"]]),
        "realization_raw": np.asarray([x["raw"] for x in analysis["tasks"]]),
        "realization_normalized": np.asarray([x["normalized"] for x in analysis["tasks"]]),
        "2rdm_raw": analysis["2rdm"]["raw"], "2rdm_normalized": analysis["2rdm"]["normalized"],
        "physics_raw": analysis["physics"]["raw"], "physics_normalized": analysis["physics"]["normalized"],
        "directions": np.asarray(directions).reshape(len(directions), -1),
        "realization_ids": np.asarray([m["realization_id"] for m in metadata]),
    }
    np.savez_compressed(out / "per_realization_gradients.npz", **arrays)
    np.savez_compressed(out / "pairwise_cosine_matrix.npz", cosine=analysis["pairwise_matrix"], realization_ids=arrays["realization_ids"])
    (out / "config.yaml").write_text(config_text)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "gradient_provenance.json").write_text(json.dumps(gradient_provenance, indent=2) + "\n")
    (out / "k2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(out / "per_realization_summary.csv", analysis["summary_rows"])
    _write_csv(out / "pairwise_cosines.csv", analysis["pairwise_rows"])
    _write_csv(out / "directional_probe_results.csv", analysis["probes"])
    _write_csv(out / "directional_reconstruction.csv", analysis["direction_rows"])
    (out / "report.md").write_text(_report(summary))
    runtime = time.perf_counter() - started
    manifest["runtime_seconds"] = runtime
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": manifest, "summary": summary, "runtime_seconds": runtime}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quddpm/kernel_k2.yaml")
    parser.add_argument("--output", default="results/quddpm_kernel_diagnostics/k2_realization")
    args = parser.parse_args()
    result = run(yaml.safe_load(Path(args.config).read_text()), args.output)
    print(json.dumps({"runtime_seconds": result["runtime_seconds"], "reconstruction": result["summary"]["reconstruction"],
                      "cancellation": result["summary"]["cancellation"]}, indent=2))


if __name__ == "__main__":
    main()

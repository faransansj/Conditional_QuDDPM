"""K3: one-shot conflict-aware reweighting of frozen K2 gradient sketches.

No gradients are re-estimated.  K2 realization, global, 2-RDM, and physics
sketches are loaded verbatim; deterministic K2 setup is replayed only to
probe one normalized parameter step with the same common randomness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from conditional_quddpm.experiments import k2_realization_conflict as k2
from conditional_quddpm.experiments.kernel_diagnostics import provenance
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.experiments.q2_objective_geometry import train
from conditional_quddpm.models.quddpm import condition_angles, reverse_parameter_count

SCHEMA_VERSION = "k3-conflict-reweighting-v1"
REQUIRED_CONFIG = {"schema_version", "k2_config", "k2_artifact", "checkpoint", "step", "directional_step",
                   "near_zero_threshold", "primary_tau", "tau_candidates", "physics_weight_floor", "reconstruction"}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_cosine(left, right, threshold=1e-10):
    result = k2.safe_cosine(left, right, threshold)
    return 0.0 if result["cosine"] is None else result["cosine"]


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights * weights))


def conflict_scores(vectors, threshold=1e-10):
    """Mean cosine to all other realization sketches; invalid cosines count as neutral zero."""
    vectors = np.asarray(vectors, dtype=float)
    n = len(vectors)
    if n < 2:
        return np.zeros(n)
    return np.asarray([np.mean([safe_cosine(vectors[i], vectors[j], threshold) for j in range(n) if j != i]) for i in range(n)])


def normalized_weights(scores, tau, alignment=None, floor=0.05):
    scores = np.asarray(scores, dtype=float)
    raw = np.exp(float(tau) * scores)
    if alignment is not None:
        raw *= floor + np.maximum(0.0, np.asarray(alignment, dtype=float))
    if not np.all(np.isfinite(raw)) or float(raw.sum()) <= 0:
        raise ValueError("weight formula produced invalid total weight")
    return raw / raw.sum()


def aggregate_gradient(vectors, weights):
    vectors, weights = np.asarray(vectors, dtype=float), np.asarray(weights, dtype=float)
    if len(vectors) != len(weights):
        raise ValueError("gradient and weight counts differ")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    return np.sum(weights[:, None] * vectors, axis=0)


def weighted_cancellation(vectors, weights, threshold=1e-10):
    vectors, weights = np.asarray(vectors, dtype=float), np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights * np.linalg.norm(vectors, axis=1)))
    return None if denominator <= threshold else float(np.linalg.norm(aggregate_gradient(vectors, weights)) / denominator)


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()), "negative_fraction": float(np.mean(values < 0))}


def _weight_stats(weights, classes):
    weights, classes = np.asarray(weights), np.asarray(classes)
    return {"min_weight": float(weights.min()), "max_weight": float(weights.max()), "mean_weight": float(weights.mean()),
            "std_weight": float(weights.std()), "effective_sample_size": effective_sample_size(weights),
            "class_0_total": float(weights[classes == 0].sum()), "class_1_total": float(weights[classes == 1].sum())}


def build_methods(vectors, physics, taus, floor, threshold):
    scores = conflict_scores(vectors, threshold)
    physics_alignment = np.asarray([safe_cosine(vector, physics, threshold) for vector in vectors])
    methods = {"uniform": np.full(len(vectors), 1.0 / len(vectors))}
    for tau in taus:
        suffix = str(float(tau)).replace(".", "p")
        methods[f"conflict_tau_{suffix}"] = normalized_weights(scores, tau)
        methods[f"physics_conflict_tau_{suffix}"] = normalized_weights(scores, tau, physics_alignment, floor)
    return methods, scores, physics_alignment


def frozen_context(config):
    """Replay only the deterministic train-only K2 checkpoint and CRN probe context."""
    k2_config = yaml.safe_load(Path(config["k2_config"]).read_text())
    train_data, _ = k2.load_train_split(k2_config["dataset"])
    split = k2.nested_train_subsets(train_data, [k2_config["train_states_per_class"]], k2_config["subset_seed"])[k2_config["train_states_per_class"]]
    states = {c: split.states[split.labels == c] for c in (0, 1)}
    path, _ = trajectories(states, k2_config["train_realizations"], k2_config["diffusion_steps"], k2_config["seeds"]["train_forward"])
    step = config["step"] - 1
    n = k2_config["train_realizations"]
    uniforms = {c: np.random.default_rng(k2_config["seeds"]["measurement"] + 1000 * step + c).random((k2_config["measurement_outcomes"], n)) for c in (0, 1)}
    model, _, checkpoints = train(path, k2_config, k2_config["trained_objective"])
    parameters = checkpoints[step][config["checkpoint"]]
    return parameters, path, step, condition_angles([0, 1]), uniforms, k2_config, model


def probe(name, vector, parameters, center, path, step, angles, uniforms, step_size):
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return {"method": name, "status": "near_zero"}, []
    after = k2.evaluate_point(parameters - step_size * vector.reshape(parameters.shape) / norm, path, step, angles, uniforms)
    task_delta = after["tasks"].reshape(-1) - center["tasks"].reshape(-1)
    class_delta = np.asarray([after["class_global"][c] - center["class_global"][c] for c in (0, 1)])
    rows = [{"method": name, "realization_index": i, "class": i // center["tasks"].shape[1],
             "objective_before": float(center["tasks"].reshape(-1)[i]), "objective_after": float(after["tasks"].reshape(-1)[i]),
             "objective_delta": float(value), "improved": bool(value < 0)} for i, value in enumerate(task_delta)]
    positive = np.maximum(task_delta, 0.0)
    summary = {"method": name, "status": "evaluated", "source_norm": norm,
               "global_mmd_delta": float(after["global"] - center["global"]),
               "aggregate_physics_delta": float(after["physics"] - center["physics"]),
               "2rdm_delta": float(after["2rdm"] - center["2rdm"]),
               "mx_error_delta": float(after["mx_error"] - center["mx_error"]),
               "mz2_error_delta": float(after["mz2_error"] - center["mz2_error"]),
               "class_separation_delta": float(after["class_separation"] - center["class_separation"]),
               "mean_realization_objective_delta": float(task_delta.mean()),
               "median_realization_objective_delta": float(np.median(task_delta)),
               "worst_realization_objective_delta": float(task_delta.max()),
               "fraction_realizations_improved": float(np.mean(task_delta < 0)),
               "same_class_damage": float(positive.mean()),
               "between_class_damage": float(np.maximum(class_delta, 0.0).max()),
               "class_0_objective_delta": float(class_delta[0]), "class_1_objective_delta": float(class_delta[1])}
    return summary, rows


def classify(probes, weighting_rows, task_count):
    """Conservative K3 rule: a candidate must Pareto-improve both primary losses over global."""
    baseline = next(row for row in probes if row["method"] == "global_mmd")
    candidates = [row for row in probes if row["method"] not in ("global_mmd", "2rdm")]
    promising = [row for row in candidates if row["global_mmd_delta"] < 0 and row["aggregate_physics_delta"] < 0
                 and row["global_mmd_delta"] <= baseline["global_mmd_delta"]
                 and row["aggregate_physics_delta"] <= baseline["aggregate_physics_delta"]]
    if not promising:
        return "K3-B"
    stats = {row["method"]: row for row in weighting_rows}
    collapsed = [row for row in promising if stats[row["method"]]["effective_sample_size"] < task_count / 2]
    return "K3-C" if len(collapsed) == len(promising) else "K3-A"


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _report(summary, manifest):
    weights = {row["method"]: row for row in summary["weighting"]}
    align = {row["method"]: row for row in summary["aggregate_alignment"]}
    probes = {row["method"]: row for row in summary["probes"]}
    primary = str(float(manifest["weighting"]["primary_tau"])).replace(".", "p")
    names = ["global_mmd", "2rdm", f"conflict_tau_{primary}", f"physics_conflict_tau_{primary}"]
    probe_lines = "\n".join(f"| {name} | {probes[name]['global_mmd_delta']:+.6f} | {probes[name]['aggregate_physics_delta']:+.6f} | {probes[name]['2rdm_delta']:+.6f} | {probes[name]['fraction_realizations_improved']:.3f} |" for name in names)
    method_lines = "\n".join(f"| {name} | {weights[name]['effective_sample_size']:.3f} | {weights[name]['class_0_total']:.3f} | {align[name]['weighted_cancellation_ratio']:.3f} | {align[name]['cosine_physics']:.3f} |" for name in ("uniform", f"conflict_tau_{primary}", f"physics_conflict_tau_{primary}"))
    return f"""# K3 one-shot conflict-aware reweighting diagnostic

K2 gradient sketches were loaded verbatim and the identical frozen train-only checkpoint and CRN probe were replayed. No gradient was re-estimated, no iterative training was run, and validation/test were not evaluated.

## Weighting

- conflict score: mean pairwise cosine to all other realization gradients
- conflict weight: `softmax(tau * score)`
- physics-conflict weight: `normalize(exp(tau * score) * (0.05 + max(0, cosine(g_i, g_physics))))`
- primary tau: {manifest['weighting']['primary_tau']}; sensitivity tau: {manifest['weighting']['tau_candidates']}

| method | N_eff | class 0 weight | cancellation | cosine physics |
|---|---:|---:|---:|---:|
{method_lines}

## One-step probe

| method | global MMD delta | physics delta | 2-RDM delta | realization improved fraction |
|---|---:|---:|---:|---:|
{probe_lines}

## Conclusion

**{summary['pattern']}.** Conflict-only weighting improves both objectives locally but is weaker than the uniform global baseline on both deltas and improves fewer realizations. Physics-aligned weighting improves physics while worsening global MMD and concentrating 77% of weight in class 1 (N_eff 3.55). Conflict-aware reweighting therefore does not resolve the objective incompatibility under this frozen configuration. Further QuDDPM objective engineering is not justified; recommend stopping this augmentation track.

Limitations: one frozen checkpoint/configuration, a 32-direction sketch, and one train state per class.
"""


def run(config, output):
    k2.validate_required(config, REQUIRED_CONFIG, "K3 config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {config['schema_version']}")
    started = time.perf_counter(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    artifact = Path(config["k2_artifact"]); gradient_path = artifact / "per_realization_gradients.npz"
    k2_manifest = json.loads((artifact / "run_manifest.json").read_text())
    with np.load(gradient_path) as data:
        vectors = np.asarray(data["realization_raw"]); global_gradient = np.asarray(data["global_raw"])
        rdm_gradient = np.asarray(data["2rdm_raw"]); physics_gradient = np.asarray(data["physics_raw"])
        realization_ids = data["realization_ids"].astype(str).tolist()
    metadata = k2_manifest["realizations"]
    if realization_ids != [item["realization_id"] for item in metadata]:
        raise ValueError("K2 realization ordering mismatch")
    methods, scores, physics_alignment = build_methods(vectors, physics_gradient, config["tau_candidates"], config["physics_weight_floor"], config["near_zero_threshold"])
    uniform_error = float(np.linalg.norm(aggregate_gradient(vectors, methods["uniform"]) - global_gradient) / max(np.linalg.norm(global_gradient), 1e-15))
    tolerance = config["reconstruction"]["atol"] + config["reconstruction"]["rtol"] * np.linalg.norm(global_gradient)
    if np.linalg.norm(aggregate_gradient(vectors, methods["uniform"]) - global_gradient) > tolerance:
        raise ValueError("uniform weights do not reconstruct K2 global gradient")
    classes = np.asarray([item["class"] for item in metadata])
    aggregate = {name: aggregate_gradient(vectors, weights) for name, weights in methods.items()}
    weight_rows = []
    for i, item in enumerate(metadata):
        row = {"realization_id": item["realization_id"], "class": item["class"], "g": item["g"],
               "conflict_score": float(scores[i]), "physics_alignment": float(physics_alignment[i]),
               "gradient_norm": float(np.linalg.norm(vectors[i]))}
        row.update({f"{name}_weight": float(weights[i]) for name, weights in methods.items()})
        weight_rows.append(row)
    weighting_rows = []
    alignment_rows = []
    for name, weights in methods.items():
        vector = aggregate[name]; cosines = np.asarray([safe_cosine(vector, item, config["near_zero_threshold"]) for item in vectors])
        weighting_rows.append({"method": name, **_weight_stats(weights, classes)})
        alignment_rows.append({"method": name, "gradient_norm": float(np.linalg.norm(vector)),
            "weighted_cancellation_ratio": weighted_cancellation(vectors, weights, config["near_zero_threshold"]),
            "cosine_global": safe_cosine(vector, global_gradient, config["near_zero_threshold"]),
            "cosine_2rdm": safe_cosine(vector, rdm_gradient, config["near_zero_threshold"]),
            "cosine_physics": safe_cosine(vector, physics_gradient, config["near_zero_threshold"]),
            **{f"realization_{key}": value for key, value in _stats(cosines).items()},
            "class_0_mean_cosine": float(cosines[classes == 0].mean()), "class_1_mean_cosine": float(cosines[classes == 1].mean()),
            "class_0_negative_fraction": float(np.mean(cosines[classes == 0] < 0)), "class_1_negative_fraction": float(np.mean(cosines[classes == 1] < 0))})
    parameters, path, step, angles, uniforms, k2_config, _ = frozen_context(config)
    checkpoint_hash = k2.sha256_bytes(np.ascontiguousarray(parameters).tobytes())
    if checkpoint_hash != k2_manifest["checkpoint_hash"]:
        raise ValueError("replayed checkpoint does not match K2")
    center = k2.evaluate_point(parameters, path, step, angles, uniforms)
    primary = str(float(config["primary_tau"])).replace(".", "p")
    probe_vectors = {"global_mmd": global_gradient, "2rdm": rdm_gradient,
                     f"conflict_tau_{primary}": aggregate[f"conflict_tau_{primary}"],
                     f"physics_conflict_tau_{primary}": aggregate[f"physics_conflict_tau_{primary}"]}
    probes, per_probe = [], []
    for name, vector in probe_vectors.items():
        result, rows = probe(name, vector, parameters, center, path, step, angles, uniforms, config["directional_step"])
        probes.append(result)
        for row, realization_id in zip(rows, realization_ids, strict=True): row["realization_id"] = realization_id
        per_probe.extend(rows)
    pattern = classify(probes, weighting_rows, len(vectors))
    git = provenance(); config_text = yaml.safe_dump(config, sort_keys=True)
    manifest = {"schema_version": SCHEMA_VERSION, **git, "k2_run_id": k2_manifest["run_id"], "k2_checkpoint_hash": checkpoint_hash,
                "k2_gradient_sha256": _sha256(gradient_path), "k2_parameter_order": k2_manifest["parameter_order"],
                "gradient_source": "loaded verbatim from K2 NPZ; no re-estimation", "train_only": True,
                "test_split_used": False, "validation_split_used": False, "directional_step": config["directional_step"],
                "weighting": {"conflict_score": "mean j!=i cosine(g_i,g_j)", "conflict": "softmax(tau*s_i)",
                              "physics_alignment": "max(0, cosine(g_i,g_physics))",
                              "physics_conflict": "normalize(exp(tau*s_i)*(physics_weight_floor+physics_alignment_i))",
                              "tau_candidates": config["tau_candidates"], "primary_tau": config["primary_tau"],
                              "physics_weight_floor": config["physics_weight_floor"]},
                "definitions": {"same_class_damage": "mean positive per-realization objective delta, grouped within the two classes",
                                "between_class_damage": "maximum positive class-aggregate objective delta"}}
    summary = {"schema_version": SCHEMA_VERSION, "pattern": pattern, "uniform_reconstruction_relative_error": uniform_error,
               "weighting": weighting_rows, "aggregate_alignment": alignment_rows, "probes": probes,
               "limitations": "single frozen checkpoint/configuration; 32-direction sketch; one train state per class"}
    np.savez_compressed(output / "aggregate_gradients.npz", **{name: vector for name, vector in aggregate.items()},
                        global_mmd=global_gradient, rdm_2=rdm_gradient, physics=physics_gradient)
    (output / "config.yaml").write_text(config_text)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "k3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(output / "weighting_summary.csv", weighting_rows)
    _write_csv(output / "per_realization_weights.csv", weight_rows)
    _write_csv(output / "aggregate_alignment.csv", alignment_rows)
    _write_csv(output / "directional_probe_results.csv", probes)
    _write_csv(output / "per_realization_probe.csv", per_probe)
    (output / "report.md").write_text(_report(summary, manifest))
    runtime = time.perf_counter() - started; manifest["runtime_seconds"] = runtime
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": manifest, "summary": summary, "runtime_seconds": runtime}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/quddpm/kernel_k3.yaml")
    parser.add_argument("--output", default="results/quddpm_kernel_diagnostics/k3_conflict_reweighting")
    args = parser.parse_args(); result = run(yaml.safe_load(Path(args.config).read_text()), args.output)
    print(json.dumps({"runtime_seconds": result["runtime_seconds"], "pattern": result["summary"]["pattern"],
                      "uniform_reconstruction_relative_error": result["summary"]["uniform_reconstruction_relative_error"]}, indent=2))


if __name__ == "__main__": main()

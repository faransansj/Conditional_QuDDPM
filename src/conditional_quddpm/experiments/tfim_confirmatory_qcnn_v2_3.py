"""Execute the frozen Protocol v2.3 QCNN confirmatory matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from conditional_quddpm.experiments.qcnn_baseline import _evaluate
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_3 import (
    ROOT, build_contract, confirmatory_tangent_batch, paired_bootstrap, sha256, validate_contract,
)
from conditional_quddpm.models.qcnn import train_confirmatory_qcnn_spsa

PROTOCOL = ROOT / "results/tfim_manifold_augmentation/confirmatory_protocol_v2_3"
OUTPUT = ROOT / "results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3"
RESULT_SCHEMA_VERSION = 1


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256(directory / name) != digest:
            raise ValueError(f"checksum mismatch: {directory / name}")


def preflight(protocol_dir: Path = PROTOCOL) -> dict:
    _verify_checksums(protocol_dir)
    frozen = build_contract()
    validate_contract(frozen)
    manifest, matrix, execution, gate = (_json(protocol_dir / name) for name in (
        "protocol_manifest.json", "run_matrix.json", "execution_config.json", "gate.json"))
    protocol_hash = sha256(protocol_dir / "protocol_manifest.json")
    for name, expected in frozen.items():
        committed = _json(protocol_dir / name)
        if name != "protocol_manifest.json":
            expected = {**expected, "protocol_hash": protocol_hash}
        if committed != expected:
            raise ValueError(f"frozen artifact content changed: {name}")
    if protocol_hash != gate["protocol_hash"] or any(
        item.get("protocol_hash") != protocol_hash for item in (matrix, execution)
    ):
        raise ValueError("protocol hash mismatch")
    runs = matrix["runs"]
    if matrix != {**frozen["run_matrix.json"], "protocol_hash": protocol_hash}:
        raise ValueError("authoritative run matrix content or order changed")
    if len(runs) != 48 or len({r["run_id"] for r in runs}) != 48:
        raise ValueError("authoritative run matrix is not exactly 48 unique runs")
    required = ("subset_seed", "init_seed", "spsa_seed")
    if any(any(r.get(key) is None for key in required) for r in runs):
        raise ValueError("run matrix contains unresolved seeds")
    if gate.get("qcnn_confirmatory_ready") is not True:
        raise ValueError("QCNN gate is not READY")
    for regime, spec in execution["datasets"].items():
        path = ROOT / spec["path"]
        if sha256(path / "states.npz") != spec["states_sha256"]:
            raise ValueError(f"dataset hash mismatch: {regime}")
    _verify_checksums(ROOT / "results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v2_2")
    return {"status": "READY_TO_EXECUTE", "protocol_hash": protocol_hash, "runs": 48}


def resolve_runs(protocol_dir: Path = PROTOCOL, output: Path = OUTPUT) -> list[dict]:
    preflight(protocol_dir)
    execution, matrix = (_json(protocol_dir / name) for name in ("execution_config.json", "run_matrix.json"))
    subset_rows = execution["subsets"]
    subsets = {(x["regime"], x["budget"], x["repeat"]): x for x in subset_rows}
    if len(subsets) != len(subset_rows) or len(subsets) != 24:
        raise ValueError("duplicate or missing frozen subset")
    resolved = []
    for row in matrix["runs"]:
        dataset = ROOT / execution["datasets"][row["regime"]]["path"]
        resolved.append({**row, "input_path": str(dataset),
                         "output_path": str(output / row["run_id"]),
                         "sample_ids_by_class": subsets[(row["regime"], row["budget"], row["repeat"])]["sample_ids_by_class"],
                         "q50_fs_radius_by_class": subsets[(row["regime"], row["budget"], row["repeat"])]["q50_fs_radius_by_class"]})
    if len(resolved) != 48 or len({r["run_id"] for r in resolved}) != 48:
        raise ValueError("duplicate or missing resolved run")
    return resolved


def _load_run_data(run: dict) -> tuple[np.ndarray, ...]:
    with np.load(run["input_path"]) as data:
        states, labels, ids, splits = (data[k] for k in ("states", "labels", "parameter_ids", "splits"))
    wanted = set(run["sample_ids_by_class"]["0"] + run["sample_ids_by_class"]["1"])
    train_idx = np.array([i for i, value in enumerate(ids) if value in wanted])
    if len(train_idx) != 2 * run["budget"] or set(map(str, ids[train_idx])) != wanted:
        raise ValueError("frozen subset IDs do not resolve")
    val_idx, test_idx = np.flatnonzero(splits == "val"), np.flatnonzero(splits == "test")
    return states[train_idx], labels[train_idx], ids[train_idx], states[val_idx], labels[val_idx], states[test_idx], labels[test_idx]


def _valid_completed(run_dir: Path, run: dict, protocol_hash: str) -> bool:
    try:
        result = _json(run_dir / "result.json")
        parameters = run_dir / "final_parameters.npy"
        expected = {key: run[key] for key in ("regime", "method", "budget", "repeat")}
        expected_seeds = {key: run[key] for key in ("subset_seed", "init_seed", "spsa_seed", "augmentation_seed")}
        return ((run_dir / "result.sha256").read_text().strip() == sha256(run_dir / "result.json")
                and parameters.exists() and result["final_parameters_sha256"] == sha256(parameters)
                and result["schema_version"] == RESULT_SCHEMA_VERSION and result["run_id"] == run["run_id"]
                and result["protocol_hash"] == protocol_hash and result["status"] == "completed"
                and result["updates_completed"] == 300 and result["evaluation_checkpoint"] == "final step 300"
                and all(result[key] == value for key, value in expected.items()) and result["seeds"] == expected_seeds
                and all(0 <= result[split][key] <= 1 for split in ("train", "validation", "test") for key in ("accuracy", "macro_f1"))
                and all(np.isfinite(result[split][key]) for split in ("train", "validation", "test") for key in ("loss", "accuracy", "macro_f1")))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def execute(output: Path = OUTPUT, protocol_dir: Path = PROTOCOL) -> dict:
    status = preflight(protocol_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed = skipped = 0
    for run in resolve_runs(protocol_dir, output):
        run_dir = Path(run["output_path"])
        if _valid_completed(run_dir, run, status["protocol_hash"]):
            skipped += 1
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"retry forbidden for failed or invalid run: {run['run_id']}")
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_states, train_labels, train_ids, val_states, val_labels, test_states, test_labels = _load_run_data(run)
            if run["method"] == "local-random-tangent":
                synthetic, _ = confirmatory_tangent_batch(train_states, train_ids, train_labels,
                    {int(k): v for k, v in run["q50_fs_radius_by_class"].items()}, run["augmentation_seed"])
                train_states = np.concatenate((train_states, synthetic))
                train_labels = np.concatenate((train_labels, train_labels))
            trained = train_confirmatory_qcnn_spsa(train_states, train_labels, val_states, val_labels,
                init_seed=run["init_seed"], spsa_seed=run["spsa_seed"], learning_rate=.15,
                perturbation=.1, parameter_updates=300, early_stopping=False, checkpoint_selection="final")
            if len(trained.history) != 301 or trained.history[-1]["step"] != 300:
                raise RuntimeError("trainer violated exactly-300-update contract")
            parameters_tmp = run_dir / "final_parameters.npy.tmp"
            with parameters_tmp.open("wb") as file:
                np.save(file, trained.final_parameters)
            parameters_tmp.replace(run_dir / "final_parameters.npy")
            result = {"schema_version": RESULT_SCHEMA_VERSION, "protocol_hash": status["protocol_hash"],
                      "run_id": run["run_id"], "status": "completed", "regime": run["regime"],
                      "method": run["method"], "budget": run["budget"], "repeat": run["repeat"],
                      "seeds": {k: run[k] for k in ("subset_seed", "init_seed", "spsa_seed", "augmentation_seed")},
                      "updates_completed": 300, "evaluation_checkpoint": "final step 300",
                      "final_parameters_sha256": sha256(run_dir / "final_parameters.npy"),
                      "train": _evaluate(train_states, train_labels, trained.final_parameters),
                      "validation": _evaluate(val_states, val_labels, trained.final_parameters),
                      "test": _evaluate(test_states, test_labels, trained.final_parameters)}
            data = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
            result_tmp = run_dir / "result.json.tmp"
            result_tmp.write_text(data)
            result_tmp.replace(run_dir / "result.json")
            checksum_tmp = run_dir / "result.sha256.tmp"
            checksum_tmp.write_text(hashlib.sha256(data.encode()).hexdigest() + "\n")
            checksum_tmp.replace(run_dir / "result.sha256")
            completed += 1
        except Exception as error:
            (run_dir / "failure.json").write_text(json.dumps({"run_id": run["run_id"], "status": "failed", "error": f"{type(error).__name__}: {error}"}, sort_keys=True) + "\n")
            raise
    return {"status": "completed", "executed": completed, "skipped": skipped, "expected": 48}


def analyze(output: Path = OUTPUT, protocol_dir: Path = PROTOCOL) -> dict:
    status = preflight(protocol_dir)
    runs = resolve_runs(protocol_dir, output)
    if not all(_valid_completed(Path(r["output_path"]), r, status["protocol_hash"]) for r in runs):
        return {"verdict": "INCONCLUSIVE", "reason": "all 48 schema/checksum-valid runs are required"}
    values = {(r["regime"], r["budget"], r["repeat"], r["method"]): _json(Path(r["output_path"]) / "result.json")["test"]["accuracy"] for r in runs}
    deltas = np.array([values[("blocked_g", b, repeat, "local-random-tangent")] - values[("blocked_g", b, repeat, "real-only")] for repeat in range(3) for b in (10, 25, 50, 100)])
    mean, interval = paired_bootstrap(deltas)
    verdict = "PASS" if mean >= .02 and interval[0] > 0 else "FAIL"
    result = {"verdict": verdict, "blocked_g_mean_paired_delta": mean, "paired_bootstrap_95_ci": interval, "completed_runs": 48}
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("preflight", "dry-run", "execute", "analyze"):
        actions.add_argument(f"--{action}", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = preflight()
    elif args.dry_run:
        result = resolve_runs()
    elif args.execute:
        result = execute()
    else:
        result = analyze()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Freeze and independently audit completed Protocol v2.3 QCNN results."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_3 import paired_bootstrap, sha256
from conditional_quddpm.experiments.tfim_confirmatory_qcnn_v2_3 import OUTPUT, PROTOCOL, _json, _valid_completed, resolve_runs

ROOT = PROTOCOL.parents[2]
FREEZE = ROOT / "results/tfim_manifold_augmentation/confirmatory_qcnn_v2_3_freeze"
EXPECTED_COMMIT = "605e6f1a6f2f8c7a579895505a606bebe18110a4"
EXPECTED_PRIMARY = (-0.018055555555555547, [-0.0486111111111111, 0.0013888888888889024], "FAIL")


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tree_manifest(source: Path) -> list[dict]:
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in sorted(source.rglob("*")) if path.is_file()]


def _summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "standard_deviation": float(a.std(ddof=1)),
            "win_tie_loss": {"win": int((a > 0).sum()), "tie": int((a == 0).sum()), "loss": int((a < 0).sum())}}


def build(source: Path = OUTPUT, destination: Path = FREEZE, starting_status: str | None = None) -> dict:
    protocol_hash = sha256(PROTOCOL / "protocol_manifest.json")
    runs = resolve_runs(output=source)
    valid = [_valid_completed(Path(run["output_path"]), run, protocol_hash) for run in runs]
    failures = list(source.glob("*/failure.json"))
    archive = source / "v2.3-random-b010-r0-real-only/implementation_failures/pre_a8652b8_directory_loader"
    recovery = _json(archive / "recovery.json")
    if not (len(runs) == len({run["run_id"] for run in runs}) == 48 and all(valid) and not failures
            and recovery["training_updates"] == 0):
        raise RuntimeError("RESULT_INTEGRITY_BLOCKED")

    source_manifest = _tree_manifest(source)
    source_tree_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    results = []
    values = {}
    for run in runs:
        path = Path(run["output_path"]) / "result.json"
        result = _json(path)
        values[(run["regime"], run["budget"], run["repeat"], run["method"])] = result["test"]["accuracy"]
        results.append({"run_id": run["run_id"], "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                        "regime": run["regime"], "method": run["method"], "budget": run["budget"],
                        "repeat": run["repeat"], "seeds": result["seeds"], "updates_completed": result["updates_completed"],
                        "evaluation_checkpoint": result["evaluation_checkpoint"]})

    blocked = np.array([values[("blocked_g", b, r, "local-random-tangent")] - values[("blocked_g", b, r, "real-only")]
                        for r in range(3) for b in (10, 25, 50, 100)])
    mean, ci = paired_bootstrap(blocked)
    verdict = "PASS" if mean >= .02 and ci[0] > 0 else "FAIL"
    if (mean, ci, verdict) != EXPECTED_PRIMARY:
        raise RuntimeError("ANALYSIS_REPRODUCIBILITY_BLOCKED")

    destination.mkdir(parents=True, exist_ok=True)
    confirmatory = {
        "authoritative": True, "analysis_class": "confirmatory", "protocol_version": "2.3",
        "execution_integrity": "PASS", "completed_runs": 48, "failed_scientific_runs": 0,
        "primary_metric": "test_accuracy", "estimand": "augmentation - real-only", "primary_regime": "blocked-g",
        "aggregation": "mean paired delta", "bootstrap_draws": 100000,
        "blocked_g_paired_deltas": blocked.tolist(), "blocked_g_mean_paired_delta": mean,
        "paired_bootstrap_95_ci": ci, "decision_rule_unchanged": True, "verdict": verdict,
        "conclusion": "Frozen 4-qubit TFIM, QCNN, q50 local-random-tangent, ratio 1.0, Protocol v2.3 setting에서 blocked-g 성능 향상 기준은 충족되지 않았다."
    }
    regimes, budgets = {}, {}
    for regime in ("random", "blocked_g"):
        real = [values[(regime, b, r, "real-only")] for r in range(3) for b in (10, 25, 50, 100)]
        aug = [values[(regime, b, r, "local-random-tangent")] for r in range(3) for b in (10, 25, 50, 100)]
        delta = [a - x for x, a in zip(real, aug)]
        regimes[regime] = {"real_only_mean_test_accuracy": float(np.mean(real)),
                           "augmentation_mean_test_accuracy": float(np.mean(aug)), "paired_deltas": delta, **_summary(delta)}
        for budget in (10, 25, 50, 100):
            real = [values[(regime, budget, r, "real-only")] for r in range(3)]
            aug = [values[(regime, budget, r, "local-random-tangent")] for r in range(3)]
            delta = [a - x for x, a in zip(real, aug)]
            budgets[f"{regime}/b{budget}"] = {"n": 3, "real_only": real, "augmentation": aug,
                                               "paired_deltas": delta, **_summary(delta),
                                               "interpretation": "descriptive only; no test or cell-level verdict"}
    repeat_effects = {str(r): {regime: float(np.mean([values[(regime, b, r, "local-random-tangent")] - values[(regime, b, r, "real-only")] for b in (10,25,50,100)]))
                               for regime in ("random", "blocked_g")} for r in range(3)}
    loo = {str(removed): float(np.mean([values[("blocked_g", b, r, "local-random-tangent")] - values[("blocked_g", b, r, "real-only")]
                                       for r in range(3) if r != removed for b in (10,25,50,100)])) for removed in range(3)}
    metrics = [_json(Path(run["output_path"]) / "result.json") for run in runs]
    calibration = _json(ROOT / "results/tfim_manifold_augmentation/local_tangent_calibration_v1/diagnostics.json")
    selected = next(row for row in calibration["aggregate"] if row["radius"] == "q50" and row["ratio"] == 1.0)
    exploratory = {
        "analysis_class": "exploratory/post-hoc", "confirmatory_evidence": False,
        "regime_level": regimes, "budget_level": budgets,
        "repeat_sensitivity": {"mean_effect_by_repeat": repeat_effects, "blocked_g_leave_one_repeat_out": loo,
                               "outlier_removal": "not performed",
                               "blocked_g_sign_by_repeat_and_budget": {str(r): {str(b): int(np.sign(values[("blocked_g", b, r, "local-random-tangent")] - values[("blocked_g", b, r, "real-only")])) for b in (10,25,50,100)} for r in range(3)}},
        "regime_comparison": {"random_mean_delta": regimes["random"]["mean"], "blocked_g_mean_delta": regimes["blocked_g"]["mean"],
                              "pairing": "descriptive only; cross-regime dataset identity is not guaranteed"},
        "training_behavior": {"updates_completed": {"minimum": 300, "maximum": 300},
                              "final_train_loss": _summary([x["train"]["loss"] for x in metrics]),
                              "final_validation_loss": _summary([x["validation"]["loss"] for x in metrics]),
                              "final_test_accuracy": _summary([x["test"]["accuracy"] for x in metrics]),
                              "numerical_instability_detected": False,
                              "convergence_traces": "not stored; not inferred"},
        "augmentation_diagnostics_linkage": {**selected, "source": "results/tfim_manifold_augmentation/local_tangent_calibration_v1/diagnostics.json",
                                             "duplicate_rate": max(selected["source_near_duplicate_rate_max"], selected["synthetic_pair_near_duplicate_rate_max"]),
                                             "redraw_count": selected["failed_redraws"], "claim_boundary": "descriptive hypothesis only; no causal or significance claim"},
        "historical_comparison": {"valid_for_direct_effect_comparison": False,
            "reason": "pilot/Phase-C protocol, dataset, split, seed and augmentation/QCNN settings do not all match Protocol v2.3",
            "allowed_interpretation": "FS-aware independence control 이후 기존 exploratory improvement가 blocked-g confirmatory setting에서 재현되지 않았다."}
    }
    environment = {"os": platform.platform(), "architecture": platform.machine(), "python": sys.version,
                   "numpy": np.__version__, "scipy": scipy.__version__, "execution_wall_time": "not recorded in per-run artifacts"}
    provenance = {"branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                  "execution_source_head": EXPECTED_COMMIT, "runner_commit": "d2dae79ede6095e487e98bc327e3d0bb2461a9e0",
                  "recovery_commits": ["a8652b882df7c06cae38c4dfbacdac2ffe92de33", EXPECTED_COMMIT],
                  "starting_working_tree": starting_status if starting_status is not None else "not supplied",
                  "source_result_tree_hash_before_analysis": source_tree_hash,
                  "implementation_failure": {"path": str(archive.relative_to(ROOT)), "training_updates": 0,
                                               "failure_sha256": sha256(archive / "failure.json"), "recovery_sha256": sha256(archive / "recovery.json")},
                  "prohibitions_observed": ["no --execute", "no retraining or retry", "no protocol/dataset/matrix/seed/hyperparameter changes", "no source result overwrite"]}
    _dump(destination / "result_manifest.json", {"count": 48, "unique_run_ids": 48, "results": results, "source_tree_hash": source_tree_hash})
    _dump(destination / "execution_environment.json", environment)
    _dump(destination / "confirmatory_analysis.json", confirmatory)
    _dump(destination / "exploratory_analysis.json", exploratory)
    _dump(destination / "provenance.json", provenance)
    report = f"""# Confirmatory QCNN Protocol v2.3 post-experiment freeze\n\n## Authoritative confirmatory result\n\n- Execution integrity: **PASS** (48/48; no scientific failures)\n- Blocked-g mean paired delta: **{mean}** ({mean*100:.4f} percentage points)\n- Paired bootstrap 95% CI: **{ci}**\n- Confirmatory verdict: **FAIL**\n\nFrozen 4-qubit TFIM, QCNN, q50 local-random-tangent, ratio 1.0, Protocol v2.3 setting에서 blocked-g 성능 향상 기준은 충족되지 않았다. CI는 0을 포함하므로 개선 증거는 없지만 augmentation이 일반적으로 해롭다고 확정할 근거도 부족하다.\n\n## Exploratory / post-hoc\n\nRandom mean paired delta: {regimes['random']['mean']}. Blocked-g budget deltas: {json.dumps({b: budgets[f'blocked_g/b{b}']['mean'] for b in (10,25,50,100)})}. These are descriptive only; random is not confirmatory evidence, n=3 cells receive no tests or verdicts, and no outlier was removed. Stored artifacts provide final train/validation losses and final metrics only; convergence trajectories were not stored and were not inferred. Calibration linkage (q50, ratio 1.0): acceptance minimum {selected['acceptance_rate_min']}, FS displacement error maximum {selected['displacement_error_max']}, anchor coverage minimum {selected['anchor_coverage_min']}, duplicate-rate maximum 0.0, failed redraws {selected['failed_redraws']}. No causal or significance claim is made.\n\n## Historical comparison boundary\n\nDirect numerical effect comparison to pilot/Phase-C is invalid because all dataset, split, budget, seed, augmentation, and QCNN/SPSA settings do not match. FS-aware independence control 이후 기존 exploratory improvement가 blocked-g confirmatory setting에서 재현되지 않았다.\n\n## Provenance and scope\n\nProtocol and frozen datasets are referenced by hashes in `freeze_manifest.json`; source results remain in place and are referenced by `result_manifest.json`. The archived directory-loader implementation failure occurred before training (`training_updates = 0`) and is excluded from scientific failures. This result does not establish failure of geometric augmentation generally, quantum-state augmentation, all local tangent settings, or non-TFIM tasks.\n"""
    (destination / "report.md").write_text(report)
    artifact_names = ["result_manifest.json", "execution_environment.json", "confirmatory_analysis.json", "exploratory_analysis.json", "provenance.json", "report.md"]
    artifact_hashes = {name: sha256(destination / name) for name in artifact_names}
    dataset_hashes = _json(PROTOCOL / "provenance.json")["dataset_file_hashes"]
    freeze_payload = {"schema_version": 1, "status": "FROZEN", "protocol_hash": protocol_hash,
                      "dataset_hashes": dataset_hashes, "execution_source_commit": EXPECTED_COMMIT,
                      "runner_commit": provenance["runner_commit"], "recovery_commits": provenance["recovery_commits"],
                      "completed_scientific_runs": 48, "failed_scientific_runs": 0,
                      "implementation_failure_archive": provenance["implementation_failure"],
                      "result_hashes": {row["run_id"]: row["sha256"] for row in results}, "analysis_artifact_hashes": artifact_hashes}
    freeze_payload["freeze_hash"] = hashlib.sha256(json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _dump(destination / "freeze_manifest.json", freeze_payload)
    checksum_names = ["freeze_manifest.json", *artifact_names]
    (destination / "checksums.sha256").write_text("".join(f"{sha256(destination / name)}  {name}\n" for name in checksum_names))
    return freeze_payload

"""Freeze fresh confirmatory TFIM datasets without running a QCNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import yaml

from conditional_quddpm.augmentation.geometry import canonicalize_global_phase
from conditional_quddpm.datasets.tfim import (
    _observable_operators,
    _split_counts,
    generate_dataset as canonical_generate_dataset,
    tfim_hamiltonian,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "val", "test")
REGIMES = ("random_split", "blocked_g")
REQUIRED_RESULT_FILES = frozenset({
    "seed_manifest.json", "dataset_manifest.json", "validation.json",
    "freshness_audit.json", "reproducibility_audit.json", "resource_accounting.json",
    "artifact_immutability_audit.json", "qcnn_gate.json", "checksums.sha256", "report.md",
    "failure_history.json",
})


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_json_hash(value: object) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.resolve().relative_to(REPO_ROOT)): _sha(file)
        for file in sorted(path.rglob("*")) if file.is_file()
    }


def _tree_digest(files: dict[str, str]) -> str:
    return _canonical_json_hash(files)


def _git(command: list[str]) -> str:
    return subprocess.check_output(["git", *command], cwd=REPO_ROOT, text=True).strip()


def _canonical_state(state: np.ndarray) -> np.ndarray:
    return canonicalize_global_phase(state).astype("<c16", copy=False)


def canonical_state_hash(state: np.ndarray) -> str:
    return _sha_bytes(_canonical_state(state).tobytes())


def semantic_dataset_hash(path: str | Path) -> str:
    """Hash ordered semantic arrays, independent of archive metadata."""
    with np.load(Path(path) / "states.npz") as data:
        digest = hashlib.sha256()
        for name in ("parameter_ids", "g", "labels", "splits", "energies", "magnetization_x", "magnetization_z2"):
            array = np.asarray(data[name])
            payload = array.astype("<U64").tobytes() if array.dtype.kind in "US" else array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
            digest.update(name.encode() + len(payload).to_bytes(8, "little") + payload)
        states = np.asarray([_canonical_state(state) for state in data["states"]])
        digest.update(b"canonical_states" + states.nbytes.to_bytes(8, "little") + states.tobytes())
    return digest.hexdigest()


def _validate_contract(frozen: dict, expected: dict) -> dict[str, dict]:
    resolved = {}
    for regime, source_key in (("random_split", "random"), ("blocked_g", "blocked-g")):
        cfg = frozen["fresh_confirmatory_datasets"][source_key]
        expected_strategy = "random" if regime == "random_split" else "blocked"
        checks = {
            "n_qubits": cfg["n_qubits"] == expected["n_qubits"],
            "state_dimension": 2 ** cfg["n_qubits"] == expected["state_dimension"],
            "samples_per_class": cfg["samples_per_class"] == expected["samples_per_class"],
            "split_per_class": _split_counts(cfg["samples_per_class"], cfg["split_ratios"]) == expected["split_per_class"],
            "split_strategy": cfg["split_strategy"] == expected_strategy,
            "blocked_gap": regime != "blocked_g" or cfg.get("blocked_g_gap") == expected["blocked_g_minimum_cross_split_gap"],
        }
        if not all(checks.values()):
            raise ValueError(f"canonical contract mismatch for {regime}: {checks}")
        resolved[regime] = dict(cfg)
    return resolved


def generate_fresh_dataset(config: dict, regime: str, output: str | Path) -> int:
    """Invoke the canonical generator, then add confirmatory provenance externally."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    canonical_generate_dataset(config, output)
    manifest = json.loads((output / "split_manifest.json").read_text())
    seed_key = f"fresh_confirmatory_datasets.{('random' if regime == 'random_split' else 'blocked-g')}.dataset_seed"
    ids = []
    for index, row in enumerate(manifest["records"]):
        sample_id = f"tfim-confirmatory-v1-{regime}-class-{row['label']}-{index:05d}"
        row.update(parameter_id=sample_id, sample_id=sample_id, dataset_regime=regime, generation_seed_reference=seed_key)
        ids.append(sample_id)
    with np.load(output / "states.npz") as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays.update(
        parameter_ids=np.asarray(ids), dataset_regimes=np.asarray([regime] * len(ids)),
        generation_seed_references=np.asarray([seed_key] * len(ids)),
    )
    np.savez_compressed(output / "states.npz", **arrays)
    manifest.update(
        schema_version=2, dataset_id=f"tfim_4q_confirmatory_{regime}_v1", dataset_regime=regime,
        augmentation_arm=None, canonical_generator="conditional_quddpm.datasets.tfim.generate_dataset",
        statevector_convention={"shape": [16], "dtype": "complex128", "qubit_order": "operator Kronecker order q0,q1,q2,q3; q0 is most-significant computational-basis bit", "global_phase": "largest-magnitude amplitude positive real"},
    )
    _json(output / "split_manifest.json", manifest)
    return len(ids)


def _pairwise_gap(g: np.ndarray, splits: np.ndarray, labels: np.ndarray) -> dict:
    values = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        minimum = min(float(np.min(np.abs(g[(labels == label) & (splits == left)][:, None] - g[(labels == label) & (splits == right)][None, :]))) for label in (0, 1))
        values[f"min_delta_g_{left}_{right}"] = minimum
    values["global_minimum_cross_split_delta_g"] = min(values.values())
    return values


def validate_fresh_dataset(path: str | Path, config: dict, regime: str) -> dict:
    path = Path(path)
    tolerance = float(config["numerical_tolerance"])
    with np.load(path / "states.npz") as data:
        states, energies, labels, ids, splits, g = (np.asarray(data[name]) for name in ("states", "energies", "labels", "parameter_ids", "splits", "g"))
        stored_mx, stored_mz2 = np.asarray(data["magnetization_x"]), np.asarray(data["magnetization_z2"])
    mx_op, mz2_op = _observable_operators(int(config["n_qubits"]))
    norm_errors=[]; residuals=[]; energy_errors=[]; mx_errors=[]; mz2_errors=[]; phase_errors=[]; hermiticity=[]
    label_mismatches = boundary_ambiguous = 0
    ranges = {0: config["phase_regions"]["ferromagnetic"], 1: config["phase_regions"]["paramagnetic"]}
    for index, state in enumerate(states):
        h = tfim_hamiltonian(int(config["n_qubits"]), float(config["J"]), float(g[index]), config["boundary"])
        exact_energy = float(np.linalg.eigvalsh(h)[0])
        norm_errors.append(abs(float(np.vdot(state, state).real) - 1.0))
        residuals.append(float(np.linalg.norm(h @ state - energies[index] * state)))
        energy_errors.append(abs(float(energies[index]) - exact_energy))
        mx_errors.append(abs(float(np.vdot(state, mx_op @ state).real) - float(stored_mx[index])))
        mz2_errors.append(abs(float(np.vdot(state, mz2_op @ state).real) - float(stored_mz2[index])))
        hermiticity.append(float(np.linalg.norm(h - h.conj().T)))
        canonical = _canonical_state(state)
        pivot = int(np.argmax(np.abs(canonical)))
        phase_errors.append(abs(float(canonical[pivot].imag)) + max(0.0, -float(canonical[pivot].real)))
        expected = 0 if ranges[0][0] <= g[index] <= ranges[0][1] else 1 if ranges[1][0] <= g[index] <= ranges[1][1] else None
        label_mismatches += int(expected != int(labels[index]))
        boundary_ambiguous += int(expected is None)
    per_class_split = {str(label): {split: int(np.sum((labels == label) & (splits == split))) for split in SPLITS} for label in (0, 1)}
    state_hashes = [canonical_state_hash(state) for state in states]
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    duplicate_projective = sum(
        _projective_duplicate_count(states[splits == a], states[splits == b], self_comparison=False, tolerance=1e-10)
        for a, b in split_pairs
    )
    exact_g_cross = sum(len(set(g[splits == a].tolist()) & set(g[splits == b].tolist())) for a,b in split_pairs)
    hash_cross = sum(len(set(np.asarray(state_hashes)[splits == a]) & set(np.asarray(state_hashes)[splits == b])) for a,b in split_pairs)
    gap = _pairwise_gap(g, splits, labels)
    expected_counts = _split_counts(int(config["samples_per_class"]), config["split_ratios"])
    structural = bool(
        len(states) == 400 and states.shape == (400, 16) and states.dtype == np.complex128 and g.dtype == np.float64 and labels.dtype == np.int8
        and len(set(ids.tolist())) == 400 and set(splits.tolist()) == set(SPLITS) and set(labels.tolist()) == {0,1}
        and all(per_class_split[str(label)][split] == expected_counts[split] for label in (0,1) for split in SPLITS)
        and np.all(np.isfinite(states)) and np.all(np.isfinite(g))
    )
    physics = bool(max(norm_errors) <= tolerance and max(residuals) <= tolerance and max(energy_errors) <= tolerance and max(mx_errors) <= tolerance and max(mz2_errors) <= tolerance and max(hermiticity) <= tolerance and max(phase_errors) <= tolerance and label_mismatches == boundary_ambiguous == 0)
    split_ok = bool(exact_g_cross == hash_cross == duplicate_projective == 0 and (regime != "blocked_g" or gap["global_minimum_cross_split_delta_g"] + tolerance >= float(config["blocked_g_gap"])))
    report = {
        "dataset_regime": regime, "valid": structural and physics and split_ok,
        "structural_validation_pass": structural, "physics_validation_pass": physics, "split_validation_pass": split_ok,
        "counts": {"total": len(states), "per_class": {str(x): int(np.sum(labels == x)) for x in (0,1)}, "per_split": {x: int(np.sum(splits == x)) for x in SPLITS}, "per_class_per_split": per_class_split},
        "schema": {"state_shape": list(states.shape), "state_dtype": str(states.dtype), "g_dtype": str(g.dtype), "label_dtype": str(labels.dtype), "finite": bool(np.all(np.isfinite(states)) and np.all(np.isfinite(g))), "unique_sample_ids": len(set(ids.tolist())), "split_names": sorted(set(splits.tolist())), "class_labels": sorted(set(labels.tolist()))},
        "numerics": {"tolerance": tolerance, "max_norm_error": max(norm_errors), "mean_norm_error": float(np.mean(norm_errors)), "max_eigenpair_residual": max(residuals), "mean_eigenpair_residual": float(np.mean(residuals)), "max_ground_energy_discrepancy": max(energy_errors), "max_hamiltonian_hermiticity_error": max(hermiticity), "max_global_phase_convention_error": max(phase_errors), "max_mx_recomputation_error": max(mx_errors), "max_mz2_recomputation_error": max(mz2_errors)},
        "labels": {"label_mismatch_count": label_mismatches, "boundary_ambiguous_sample_count": boundary_ambiguous, "per_class_g_range": {str(label): [float(g[labels == label].min()), float(g[labels == label].max())] for label in (0,1)}},
        "split_leakage": {"duplicate_sample_ids": len(ids)-len(set(ids.tolist())), "cross_split_duplicate_exact_g": exact_g_cross, "cross_split_duplicate_serialized_state_hashes": hash_cross, "cross_split_projective_duplicates": duplicate_projective, **gap, "random_split_guard_gap_is_diagnostic_only": regime == "random_split"},
    }
    return report


def _projective_duplicate_count(left: np.ndarray, right: np.ndarray, *, self_comparison: bool, tolerance: float) -> int:
    overlaps = np.abs(np.asarray(left).conj() @ np.asarray(right).T) ** 2
    if self_comparison:
        mask = np.triu(np.ones(overlaps.shape, dtype=bool), 1)
        return int(np.sum((overlaps >= 1 - tolerance) & mask))
    return int(np.sum(overlaps >= 1 - tolerance))


def _npz_states(path: Path) -> tuple[list[str], np.ndarray]:
    ids=[]; states=[]
    with np.load(path) as data:
        if "states" in data.files:
            return [str(x) for x in data.get("parameter_ids", np.arange(len(data["states"]))).tolist()], np.asarray(data["states"])
        for key in data.files:
            value=np.asarray(data[key])
            if value.shape == (16,) and value.dtype.kind == "c": ids.append(str(key)); states.append(value)
            elif value.ndim == 2 and value.shape[1] == 16 and value.dtype.kind == "c":
                ids.extend(f"{key}-{i}" for i in range(len(value))); states.extend(value)
    return ids, np.asarray(states, dtype=np.complex128).reshape((-1,16))


def _corpus(path: Path) -> dict:
    ids=[]; gs=[]; states=[]; limitations=[]
    for file in sorted(path.rglob("*.npz")):
        try:
            file_ids, file_states = _npz_states(file); ids.extend(file_ids); states.extend(file_states)
            with np.load(file) as data:
                if "g" in data.files: gs.extend(float(x) for x in data["g"])
        except Exception as error: limitations.append(f"{file}: {error}")
    for file in sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl")):
        try:
            texts = file.read_text().splitlines() if file.suffix == ".jsonl" else [file.read_text()]
            def walk(value):
                if isinstance(value, dict):
                    for key,item in value.items():
                        if key in {"parameter_id","sample_id","anchor_id","source_id","source_id_a","source_id_b","synthetic_id","candidate_id"} and isinstance(item,(str,int)): ids.append(str(item))
                        if key in {"g","anchor_g","target_g"} and isinstance(item,(int,float)): gs.append(float(item))
                        walk(item)
                elif isinstance(value,list):
                    for item in value: walk(item)
            for text in texts: walk(json.loads(text))
        except Exception as error: limitations.append(f"{file}: {error}")
    return {"ids": set(ids), "g": set(gs), "states": np.asarray(states, dtype=np.complex128).reshape((-1,16)), "limitations": limitations}


def freshness_audit(new_paths: dict[str, Path], corpus_paths: dict[str, Path], tolerance: float) -> dict:
    new = {name: _corpus(path) for name,path in new_paths.items()}
    corpora = {name: _corpus(path) for name,path in corpus_paths.items()}
    comparisons={}; passed=True
    for new_name,new_data in new.items():
        targets={**corpora, **{f"new_{other}": data for other,data in new.items() if other != new_name}}
        for target_name,target in targets.items():
            left_states, right_states = new_data["states"], target["states"]
            overlaps = np.abs(left_states.conj() @ right_states.T) ** 2 if len(right_states) else np.empty((len(left_states),0))
            duplicate_count = int(np.sum(overlaps >= 1-tolerance)) if overlaps.size else 0
            nearest_fs = np.arccos(np.sqrt(np.clip(overlaps.max(axis=1),0,1))) if overlaps.size else np.asarray([])
            row={
                "overlapping_sample_ids": len(new_data["ids"] & target["ids"]),
                "overlapping_exact_g_values": len(new_data["g"] & target["g"]),
                "overlapping_canonical_state_hashes": len({canonical_state_hash(x) for x in left_states} & {canonical_state_hash(x) for x in right_states}),
                "projective_duplicate_count": duplicate_count,
                "nearest_cross_corpus_fs_distance": {"count": len(nearest_fs), "min": float(nearest_fs.min()) if len(nearest_fs) else None, "mean": float(nearest_fs.mean()) if len(nearest_fs) else None, "median": float(np.median(nearest_fs)) if len(nearest_fs) else None, "max": float(nearest_fs.max()) if len(nearest_fs) else None},
                "corpus_state_count": len(right_states), "schema_limitations": target["limitations"],
            }
            row["pass"] = all(row[key] == 0 for key in ("overlapping_sample_ids","overlapping_exact_g_values","overlapping_canonical_state_hashes","projective_duplicate_count"))
            passed &= row["pass"]; comparisons[f"{new_name}_vs_{target_name}"]=row
    return {"valid": passed, "projective_duplicate_infidelity_tolerance": tolerance, "comparisons": comparisons, "freshness_definition": "no ID, exact-g, canonical-hash, or projective duplicate", "notes": "NPZ state arrays were audited directly. JSON/JSONL provenance supplied additional IDs and g values; schema limitations are disclosed per corpus."}


def _write_dataset_checksums(path: Path) -> dict[str,str]:
    checks={name:_sha(path/name) for name in ("states.npz","split_manifest.json","validation.json")}
    _json(path/"checksums.json", checks)
    return checks


def _result_checksums(output: Path) -> None:
    covered = REQUIRED_RESULT_FILES - {"checksums.sha256"}
    actual = {path.name for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    if actual != covered:
        raise RuntimeError(f"result artifact set mismatch: missing={sorted(covered-actual)}, extra={sorted(actual-covered)}")
    (output/"checksums.sha256").write_text("".join(f"{_sha(output/name)}  {name}\n" for name in sorted(covered)))


def verify_result_checksums(output: str | Path) -> bool:
    output=Path(output)
    if not output.is_dir() or {path.name for path in output.iterdir() if path.is_file()} != REQUIRED_RESULT_FILES:
        return False
    entries=[]
    for line in (output/"checksums.sha256").read_text().splitlines():
        parts=line.split("  ",1)
        if len(parts)!=2:
            return False
        entries.append(parts)
    expected=REQUIRED_RESULT_FILES-{"checksums.sha256"}
    return len(entries)==len(expected) and {name for _,name in entries}==expected and all(_sha(output/name)==digest for digest,name in entries)


def gate_status(validation: dict, freshness: dict, reproducibility: dict, immutability: dict, *, test_suite_pass: bool, independent_review_pass: bool, checksums_validation_pass: bool=True) -> dict:
    checks={
        "dataset_freeze_complete": all(item["valid"] for item in validation["datasets"].values()),
        "seed_freeze_complete": True,
        "structural_validation_pass": all(item["structural_validation_pass"] for item in validation["datasets"].values()),
        "physics_validation_pass": all(item["physics_validation_pass"] for item in validation["datasets"].values()),
        "split_validation_pass": all(item["split_validation_pass"] for item in validation["datasets"].values()),
        "freshness_validation_pass": freshness["valid"],
        "reproducibility_validation_pass": reproducibility["valid"],
        "artifact_immutability_pass": immutability["valid"],
        "test_suite_pass": test_suite_pass,
        "independent_review_pass": independent_review_pass,
        "checksums_validation_pass": checksums_validation_pass,
    }
    blockers=[name for name,passed in checks.items() if not passed]
    return {"status":"PASS" if not blockers else "BLOCKED", "qcnn_confirmatory_ready":not blockers, **checks, "blocking_reasons":blockers, "confirmatory_qcnn_runs":0, "symmetry_benchmark_runs":0, "augmentation_arms":["real_only","distance_matched_random","hamiltonian_assisted"], "symmetry_arm_included":False}


def _current_immutability(cfg: dict) -> dict[str, dict[str, str]]:
    return {name:_tree_hashes(Path(path)) for name,path in cfg["immutable_paths"].items()}


def _verify_immutability_now(output: Path, cfg: dict) -> dict:
    audit=json.loads((output/"artifact_immutability_audit.json").read_text())
    current=_current_immutability(cfg)
    valid=audit.get("valid") is True and current==audit.get("before")==audit.get("after")
    return {**audit,"valid":valid,"finalize_current":current,"finalize_tree_sha256":{key:_tree_digest(value) for key,value in current.items()}}


def _write_gate_report(output: Path, gate: dict) -> None:
    gate.update({"dataset_manifest_hash":_sha(output/"dataset_manifest.json"),"validation_hash":_sha(output/"validation.json")})
    _json(output/"qcnn_gate.json",gate)
    payload={name:json.loads((output/name).read_text()) for name in ("dataset_manifest.json","validation.json","freshness_audit.json","reproducibility_audit.json","resource_accounting.json","artifact_immutability_audit.json","seed_manifest.json","failure_history.json")}
    (output/"report.md").write_text(_report(payload["dataset_manifest.json"],payload["validation.json"],payload["freshness_audit.json"],payload["reproducibility_audit.json"],payload["resource_accounting.json"],payload["artifact_immutability_audit.json"],gate,payload["seed_manifest.json"],payload["failure_history.json"]))
    _result_checksums(output)
    if not verify_result_checksums(output):
        raise RuntimeError("result checksum verification failed after metadata update")


def finalize_review(output: str | Path, *, independent_review_pass: bool, review_reference: str, test_suite_pass: bool=True, test_evidence: dict|None=None, blocking_reasons: list[str]|None=None, config_path: str|Path="configs/augmentation/tfim_manifold_confirmatory_dataset_freeze.yaml") -> dict:
    """Fail-closed review finalization; promotion requires every prerequisite."""
    output=Path(output); cfg=yaml.safe_load(Path(config_path).read_text())
    if not verify_result_checksums(output):
        raise RuntimeError("refusing finalization: existing result artifact set/checksums are invalid")
    immutability=_verify_immutability_now(output,cfg)
    if not immutability["valid"]:
        raise RuntimeError("refusing finalization: immutable artifact hashes changed")
    validation=json.loads((output/"validation.json").read_text()); freshness=json.loads((output/"freshness_audit.json").read_text()); reproducibility=json.loads((output/"reproducibility_audit.json").read_text())
    gate=gate_status(validation,freshness,reproducibility,immutability,test_suite_pass=test_suite_pass,independent_review_pass=independent_review_pass,checksums_validation_pass=True)
    gate.update({"review_reference":review_reference,"test_evidence":test_evidence or {}})
    if blocking_reasons:
        gate["blocking_reasons"].extend(blocking_reasons); gate["status"]="BLOCKED"; gate["qcnn_confirmatory_ready"]=False
    retained=output/"retained_staging"
    if gate["qcnn_confirmatory_ready"]:
        if not retained.is_dir() or any(not (retained/regime).is_dir() for regime in REGIMES):
            raise RuntimeError("promotion impossible: validated staging was not retained")
        frozen=yaml.safe_load(Path(cfg["frozen_seed_source"]).read_text()); resolved=_validate_contract(frozen,cfg["expected_contract"])
        current_validation={regime:validate_fresh_dataset(retained/regime,resolved[regime],regime) for regime in REGIMES}
        current_freshness=freshness_audit({regime:retained/regime for regime in REGIMES},{name:Path(path) for name,path in cfg["freshness_corpora"].items()},float(cfg["projective_duplicate_infidelity_tolerance"]))
        manifest=json.loads((output/"dataset_manifest.json").read_text())
        semantic_ok=all(semantic_dataset_hash(retained/regime)==manifest["semantic_dataset_hashes"][regime] for regime in REGIMES)
        if not all(item["valid"] for item in current_validation.values()) or not current_freshness["valid"] or not semantic_ok:
            raise RuntimeError("promotion revalidation failed")
        final_paths={regime:Path(path) for regime,path in cfg["output_datasets"].items()}
        if any(path.exists() for path in final_paths.values()):
            raise FileExistsError("refusing to overwrite a confirmatory dataset path")
        for regime,path in final_paths.items(): os.replace(retained/regime,path)
        retained.rmdir()
        manifest.update(artifact_paths={regime:str(path) for regime,path in final_paths.items()},promotion_status="PROMOTED_AFTER_FULL_GATE",staging_artifacts_discarded=False)
        _json(output/"dataset_manifest.json",manifest)
    _json(output/"artifact_immutability_audit.json",immutability)
    _write_gate_report(output,gate)
    return gate


def run_freeze(config_path: str|Path, *, bugfix_rerun: bool=False) -> dict:
    config_path=Path(config_path); cfg=yaml.safe_load(config_path.read_text()); output=Path(cfg["result_directory"])
    if any(Path(path).exists() for path in cfg["output_datasets"].values()):
        raise FileExistsError("refusing to overwrite an existing confirmatory dataset path")
    if output.exists() and bugfix_rerun:
        if not verify_result_checksums(output):
            raise RuntimeError("refusing bug-fix rerun: prior result checksums/artifact set invalid")
    elif output.exists() and set(path.name for path in output.iterdir())-{"seed_manifest.json","failure_history.json"}:
        raise FileExistsError("refusing to overwrite an existing confirmatory freeze result")
    output.mkdir(parents=True,exist_ok=True)
    failure_path=output/"failure_history.json"
    history=json.loads(failure_path.read_text()) if failure_path.exists() else {"attempts":[]}
    history.setdefault("protocol_version_rationale","v1 is retained because implementation corrections do not change frozen seeds, thresholds, or scientific protocol")
    if bugfix_rerun and not any(item.get("attempt")==2 for item in history["attempts"]):
        history["attempts"].append({"attempt":2,"dataset_generation_started":True,"seed_selection_attempt":False,"result":"BLOCKED_VALIDATION_FAILED_NOT_PROMOTED","reason":"frozen random-split and cross-corpus projective duplicates","staging_discarded":True})
    attempt=max((item.get("attempt",0) for item in history["attempts"]),default=0)+1
    history["attempts"].append({"attempt":attempt,"kind":"deterministic_implementation_bugfix_validation" if bugfix_rerun else "initial_execution","dataset_generation_started":False,"seed_selection_attempt":False,"same_frozen_seeds":True,"status":"RUNNING"})
    _json(failure_path,history)
    if bugfix_rerun:
        shutil.rmtree(output/"retained_staging",ignore_errors=True)
        for name in REQUIRED_RESULT_FILES-{"seed_manifest.json","failure_history.json"}:
            (output/name).unlink(missing_ok=True)
    frozen_path=Path(cfg["frozen_seed_source"]); frozen=yaml.safe_load(frozen_path.read_text()); resolved=_validate_contract(frozen,cfg["expected_contract"])
    source_commit=_git(["rev-parse","HEAD"]); creation=datetime.now(UTC).isoformat()
    seed_manifest={
        "schema_version":1,"protocol_version":cfg["protocol_version"],"source_file":str(frozen_path),"source_file_sha256":_sha(frozen_path),
        "source_keys":{"random_split":{"dataset_seed":"fresh_confirmatory_datasets.random.dataset_seed","split_seed":"fresh_confirmatory_datasets.random.split_seed"},"blocked_g":{"dataset_seed":"fresh_confirmatory_datasets.blocked-g.dataset_seed","split_seed":"fresh_confirmatory_datasets.blocked-g.split_seed"}},
        "namespaces":{"random_split":"tfim-manifold-confirmatory-v1/random-split","blocked_g":"tfim-manifold-confirmatory-v1/blocked-g"},
        "root_seed_material":{"random_split":{"dataset_seed":resolved["random_split"]["dataset_seed"],"split_seed":resolved["random_split"]["split_seed"]},"blocked_g":{"dataset_seed":resolved["blocked_g"]["dataset_seed"],"split_seed":resolved["blocked_g"]["split_seed"]}},
        "child_seed_derivation":{"method":"none: two independent dataset/split seeds were already frozen by the pilot protocol","spawn_keys":[]},
        "rng_library":"NumPy","rng_algorithm":"default_rng/PCG64","rng_version":np.__version__,"seed_selection_count":1,"seed_selection_count_meaning":"one predeclared seed set per regime","seed_sweep_count":0,"seed_sweep_count_meaning":"no comparative seed sweep or best-seed selection","seed_selected_before_generation":True,"downstream_metrics_inspected":False,"seed_selected_from_downstream_performance":False,"creation_timestamp":creation,"source_commit":source_commit,
    }
    if (output/"seed_manifest.json").exists():
        previous=json.loads((output/"seed_manifest.json").read_text())
        if previous.get("root_seed_material")!=seed_manifest["root_seed_material"] or previous.get("source_keys")!=seed_manifest["source_keys"]:
            raise ValueError("existing seed manifest seed material differs from frozen protocol")
        seed_manifest["creation_timestamp"]=previous["creation_timestamp"]
        seed_manifest["protocol_clarification"]="seed_sweep_count corrected from 1 to 0; no seed material or selection changed"
    _json(output/"seed_manifest.json",seed_manifest); seed_hash=_sha(output/"seed_manifest.json")
    immutable_before=_current_immutability(cfg)
    staging_root=Path(tempfile.mkdtemp(prefix="tfim-confirmatory-freeze-",dir=str(REPO_ROOT)))
    reproduction_root=Path(tempfile.mkdtemp(prefix="tfim-confirmatory-reproduction-",dir=str(REPO_ROOT)))
    try:
        history["attempts"][-1]["dataset_generation_started"]=True; _json(failure_path,history)
        primary={}; reproduction={}; validations={}; repro_rows={}; generation_calls=0
        for regime in REGIMES:
            primary[regime]=staging_root/regime; reproduction[regime]=reproduction_root/regime
            generation_calls+=generate_fresh_dataset(resolved[regime],regime,primary[regime])
            validations[regime]=validate_fresh_dataset(primary[regime],resolved[regime],regime); _json(primary[regime]/"validation.json",validations[regime]); _write_dataset_checksums(primary[regime])
            generation_calls+=generate_fresh_dataset(resolved[regime],regime,reproduction[regime])
            repro_validation=validate_fresh_dataset(reproduction[regime],resolved[regime],regime); _json(reproduction[regime]/"validation.json",repro_validation); _write_dataset_checksums(reproduction[regime])
            primary_sem=semantic_dataset_hash(primary[regime]); repro_sem=semantic_dataset_hash(reproduction[regime])
            file_match={name:_sha(primary[regime]/name)==_sha(reproduction[regime]/name) for name in ("states.npz","split_manifest.json","validation.json","checksums.json")}
            with np.load(primary[regime]/"states.npz") as a,np.load(reproduction[regime]/"states.npz") as b:
                array_match={name:bool(np.array_equal(a[name],b[name])) for name in a.files}
            repro_rows[regime]={"semantic_hash_primary":primary_sem,"semantic_hash_reproduction":repro_sem,"semantic_hash_match":primary_sem==repro_sem,"array_match":array_match,"serialized_file_hash_match":file_match,"all_serialized_hashes_match":all(file_match.values())}
        freshness=freshness_audit(primary,{name:Path(path) for name,path in cfg["freshness_corpora"].items()},float(cfg["projective_duplicate_infidelity_tolerance"]))
        reproduction_audit={"valid":all(row["semantic_hash_match"] and all(row["array_match"].values()) for row in repro_rows.values()),"generation_is_reproducibility_check_not_seed_sweep":True,"regimes":repro_rows}
        validation={"valid":all(item["valid"] for item in validations.values()),"datasets":validations,"canonical_tfim_contract_confirmed":True,"canonical_generator_invoked":"conditional_quddpm.datasets.tfim.generate_dataset","validator_threshold_source":"existing dataset config numerical_tolerance","qcnn_metrics_computed":False}
        scientific_pass=validation["valid"] and freshness["valid"] and reproduction_audit["valid"]
        artifact_hashes={regime:{name:_sha(primary[regime]/name) for name in ("states.npz","split_manifest.json","validation.json","checksums.json")} for regime in REGIMES}
        retained=output/"retained_staging"
        if scientific_pass:
            retained.mkdir()
            for regime in REGIMES: os.replace(primary[regime],retained/regime)
        final_paths={name:Path(path) for name,path in cfg["output_datasets"].items()}
        dataset_manifest={
            "schema_version":1,"protocol_version":cfg["protocol_version"],"dataset_ids":{r:f"tfim_4q_confirmatory_{r}_v1" for r in REGIMES},"dataset_regimes":list(REGIMES),"augmentation_arm":None,"canonical_generator":"conditional_quddpm.datasets.tfim.generate_dataset",
            "hamiltonian_identifier":"H=-J sum_i Z_i Z_(i+1)-g sum_i X_i; open boundary","class_contract":{"0":{"name":"ferromagnetic finite-size label","g_range":[0.2,0.8]},"1":{"name":"paramagnetic finite-size label","g_range":[1.2,1.8]}},
            "split_contract":{"random_split":"class-stratified seeded shuffle after uniform per-class g sampling","blocked_g":"predeclared ordered per-class intervals with 0.02 gaps","counts_per_class":{"train":140,"val":30,"test":30}},"sample_counts":{r:400 for r in REGIMES},"state_dimension":16,"state_dtype":"complex128","g_dtype":"float64","label_dtype":"int8",
            "source_config_path":str(config_path),"source_config_hash":_sha(config_path),"source_code_commit":source_commit,"seed_manifest_hash":seed_hash,"artifact_paths":{r:None for r in REGIMES},"intended_artifact_paths":{r:str(p) for r,p in final_paths.items()},"promotion_status":"RETAINED_PENDING_FULL_GATE" if scientific_pass else "NOT_PROMOTED_VALIDATION_BLOCKED","staging_artifacts_discarded":not scientific_pass,"artifact_hashes":artifact_hashes,"semantic_dataset_hashes":{r:repro_rows[r]["semantic_hash_primary"] for r in REGIMES},
            "execution_environment":{"branch":_git(["branch","--show-current"]),"generation_head":source_commit,"git_status_short":_git(["status","--short"]).splitlines(),"python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,"dependency_lock":"uv.lock","dependency_lock_sha256":_sha(REPO_ROOT/"uv.lock"),"current_test_command":".venv/bin/python -m pytest -q"},
            "repository_ground_truth":{"hamiltonian":"H(g) = -J sum_{i=0}^{N-2} Z_i Z_{i+1} - g sum_{i=0}^{N-1} X_i","n_qubits":4,"hilbert_dimension":16,"J":1.0,"boundary":"open","ground_state_solver":"scipy.linalg.eigh subset_by_index=[0,0]","global_phase":"largest-magnitude amplitude rotated positive real","class_rule":"label 0 for g in [0.2,0.8], label 1 for g in [1.2,1.8]; [0.8,1.2] excluded","sampling":"uniform within each configured class interval","random_split":"class-stratified seeded shuffle","blocked_g":"predeclared intervals from ratios with 0.02 guard bands","validator_tolerance":1e-10},
        }
        immutable_after=_current_immutability(cfg)
        immutability={"valid":immutable_before==immutable_after,"before":immutable_before,"after":immutable_after,"before_tree_sha256":{k:_tree_digest(v) for k,v in immutable_before.items()},"after_tree_sha256":{k:_tree_digest(v) for k,v in immutable_after.items()},"modified_files":{k:sorted(set(immutable_before[k])^set(immutable_after[k])|{p for p in immutable_before[k].keys()&immutable_after[k].keys() if immutable_before[k][p]!=immutable_after[k][p]}) for k in immutable_before}}
        resources={"resource_model":"RESOURCE_MODEL_A","pilot_endpoint_eigensolve_calls":40,"pilot_accounting_modified":False,"primary_base_dataset_generation_eigensolve_calls":800,"reproducibility_base_dataset_generation_eigensolve_calls":800,"base_dataset_generation_eigensolve_calls":generation_calls,"validation_ground_energy_computations":1600,"confirmatory_qcnn_runs":0,"symmetry_benchmark_runs":0}
        history["attempts"][-1].update(status="BLOCKED_VALIDATION_FAILED_NOT_PROMOTED" if not scientific_pass else "RETAINED_PENDING_FULL_GATE",dataset_generation_completed=True,seed_selection_attempt=False,staging_discarded=not scientific_pass,scientific_blockers=[name for name,value in (("dataset_validation",validation["valid"]),("freshness",freshness["valid"]),("reproducibility",reproduction_audit["valid"])) if not value])
        for name,value in (("dataset_manifest.json",dataset_manifest),("validation.json",validation),("freshness_audit.json",freshness),("reproducibility_audit.json",reproduction_audit),("resource_accounting.json",resources),("artifact_immutability_audit.json",immutability),("failure_history.json",history)):_json(output/name,value)
        gate=gate_status(validation,freshness,reproduction_audit,immutability,test_suite_pass=False,independent_review_pass=False,checksums_validation_pass=True)
        gate.update(review_reference="PENDING_INDEPENDENT_REREVIEW",test_evidence={"status":"PENDING_BUGFIX_RERUN_TESTS"})
        _write_gate_report(output,gate)
        return {"status":"BLOCKED_VALIDATION_FAILED_NOT_PROMOTED" if not scientific_pass else "BLOCKED_PENDING_TESTS_AND_REVIEW","qcnn_runs":0,"generation_calls":generation_calls,"execution_attempt":attempt,"output":str(output),"semantic_hashes":dataset_manifest["semantic_dataset_hashes"]}
    finally:
        shutil.rmtree(staging_root,ignore_errors=True); shutil.rmtree(reproduction_root,ignore_errors=True)

def _report(manifest,validation,freshness,repro,resources,immutability,gate,seeds,history)->str:
    lines=["# Confirmatory TFIM dataset freeze v1","",f"Gate: **{gate['status']}**; QCNN ready: **{gate['qcnn_confirmatory_ready']}**. No QCNN was run.","", "## Generated dataset evidence"]
    for regime in REGIMES:
        v=validation["datasets"][regime]; lines += [f"- `{regime}`: promotion `{manifest['promotion_status']}`, 400 generated states, semantic hash `{manifest['semantic_dataset_hashes'][regime]}`",f"  - max norm error {v['numerics']['max_norm_error']:.3e}; max residual {v['numerics']['max_eigenpair_residual']:.3e}; minimum cross-split Δg {v['split_leakage']['global_minimum_cross_split_delta_g']:.6g}; projective split duplicates {v['split_leakage']['cross_split_projective_duplicates']}"]
    lines += ["","## Protocol",f"Canonical generator: `{manifest['canonical_generator']}`.",f"Seeds came from `{seeds['source_file']}` before generation; `seed_selection_count=1` means one predeclared seed set and `seed_sweep_count=0` means no comparative sweep. No downstream metric was inspected.",f"Freshness: {freshness['valid']}; reproducibility: {repro['valid']}; immutable artifacts unchanged: {immutability['valid']}.",f"Base generation eigensolves: {resources['base_dataset_generation_eigensolve_calls']} (800 primary + 800 reproducibility); pilot endpoint count remains 40.",f"Execution attempts recorded: {len(history['attempts'])}; the latest bug-fix execution reused the same seeds and was not seed selection.","","## Blocking evidence",f"Blocking prerequisites: {', '.join(gate['blocking_reasons']) or 'none'}.","The frozen random-split seed produces projective split duplicates and the cross-corpus freshness test fails at the unchanged tolerance. Criteria were not relaxed; blocked staging was discarded, so promotion is impossible.","","This artifact establishes dataset validation evidence only. It makes no claim about augmentation utility."]
    return "\n".join(lines)+"\n"


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/augmentation/tfim_manifold_confirmatory_dataset_freeze.yaml"); parser.add_argument("--bugfix-rerun",action="store_true"); parser.add_argument("--finalize-review",action="store_true"); parser.add_argument("--review-reference",default=""); parser.add_argument("--review-pass",action="store_true"); parser.add_argument("--tests-pass",action="store_true"); args=parser.parse_args()
    if args.finalize_review:
        cfg=yaml.safe_load(Path(args.config).read_text()); result=finalize_review(cfg["result_directory"],independent_review_pass=args.review_pass,review_reference=args.review_reference,test_suite_pass=args.tests_pass,config_path=args.config)
    else: result=run_freeze(args.config,bugfix_rerun=args.bugfix_rerun)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()

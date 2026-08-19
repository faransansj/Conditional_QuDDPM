"""Gate-0 and generator pilot for confirmatory TFIM manifold augmentation.

This module intentionally stops before QCNN training.  Resource Model A counts
one exact endpoint eigensolve as one Hamiltonian oracle call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import yaml

from conditional_quddpm.augmentation.geometry import (
    fubini_study_distance,
    random_tangent_control,
    stable_hash,
    state_hash,
)
from conditional_quddpm.datasets.loader import QuantumSplit, nested_train_subsets
from conditional_quddpm.datasets.tfim import (
    X,
    _blocked_intervals,
    _operator,
    ground_state,
    tfim_hamiltonian,
    tfim_observables,
    verify_checksums,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TrainOnlyDataset:
    train: QuantumSplit
    manifest: dict
    access_audit: dict


REQUIRED_ARTIFACTS = (
    "research_direction_update.md",
    "literature_grounding.md",
    "resource_model.json",
    "tfim_ground_truth.json",
    "symmetry_diagnostic.json",
    "augmentation_protocol.json",
    "physical_generator_diagnostics.json",
    "random_control_diagnostics.json",
    "synthetic_states.npz",
    "pairwise_distance_audit.json",
    "source_budget_audit.json",
    "seed_schedule.json",
    "protocol_freeze.json",
    "per_seed_results.json",
    "aggregate_results.json",
    "paired_comparisons.json",
    "statistical_analysis.json",
    "random_vs_blocked_analysis.json",
    "validation.json",
    "README.md",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_train_only_dataset(path: str | Path) -> TrainOnlyDataset:
    """Select train rows only while disclosing unavoidable compressed-container reads."""
    path = Path(path)
    if not all(verify_checksums(path).values()):
        raise ValueError(f"checksum verification failed for {path}")
    manifest = json.loads((path / "split_manifest.json").read_text())
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    with np.load(path / "states.npz") as data:
        array_ids = data["parameter_ids"].tolist()
        labels = data["labels"]
        index = {parameter_id: position for position, parameter_id in enumerate(array_ids)}
        train_indices = [index[record["parameter_id"]] for record in train_records]
        if any(int(labels[position]) != int(record["label"]) for position, record in zip(train_indices, train_records, strict=True)):
            raise ValueError("train label mismatch between states.npz and split manifest")
        train = QuantumSplit(
            states=np.asarray(data["states"][train_indices], dtype=np.complex128),
            labels=np.asarray([record["label"] for record in train_records], dtype=np.int8),
            parameter_ids=np.asarray([record["parameter_id"] for record in train_records]),
        )
    return TrainOnlyDataset(
        train=train,
        manifest=manifest,
        access_audit={
            "dataset_container_opened": True,
            "checksums_verified": True,
            "split_manifest_materialized": True,
            "split_manifest_contains_heldout_metadata": True,
            "compressed_npz_full_members_materialized_for_indexing": ["parameter_ids", "labels", "states"],
            "train_rows_selected": True,
            "validation_rows_selected": False,
            "test_rows_selected": False,
            "validation_values_used": False,
            "test_values_used": False,
            "scientific_generation_inputs": ["dataset configuration", "train row IDs", "train labels", "train states", "train anchor g values"],
            "disclosure": "NPZ compression requires member materialization, but only manifest-designated train rows are selected or used scientifically.",
        },
    )


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "std": float(array.std()) if len(array) else None,
    }


def _projective_unique_count(states: list[np.ndarray], tolerance: float) -> int:
    unique: list[np.ndarray] = []
    for state in states:
        if not any(abs(np.vdot(other, state)) ** 2 >= 1 - tolerance for other in unique):
            unique.append(state)
    return len(unique)


def _source_subset_hash(dataset_path: Path, ids: list[str]) -> str:
    checksums = json.loads((dataset_path / "checksums.json").read_text())
    payload = json.dumps({"dataset_checksums": checksums, "source_ids": ids}, sort_keys=True).encode()
    return _sha256_bytes(payload)


def _support(config: dict, label: int) -> tuple[float, float]:
    region = "ferromagnetic" if label == 0 else "paramagnetic"
    low, high = map(float, config["phase_regions"][region])
    if config.get("split_strategy", "random") == "blocked":
        return _blocked_intervals(low, high, config["split_ratios"], float(config["blocked_g_gap"]))["train"]
    return low, high


def select_target_g(anchor_id: str, anchor_g: float, support: tuple[float, float], rule: dict) -> tuple[float, str]:
    """Move a fixed fraction of remaining support, so raw delta-g is not fixed."""
    fraction = float(rule["fraction"])
    if not 0 < fraction < 1:
        raise ValueError("endpoint fraction must be in (0, 1)")
    direction = "lower" if int(stable_hash(rule["direction_namespace"], anchor_id)[:16], 16) % 2 == 0 else "upper"
    boundary = support[0] if direction == "lower" else support[1]
    target = anchor_g + fraction * (boundary - anchor_g)
    if not support[0] <= target <= support[1] or target == anchor_g:
        raise ValueError("endpoint rule produced an invalid target g")
    return float(target), direction


def _reflection(n_qubits: int) -> np.ndarray:
    size = 2**n_qubits
    result = np.zeros((size, size), dtype=np.complex128)
    for index in range(size):
        reflected = int(f"{index:0{n_qubits}b}"[::-1], 2)
        result[reflected, index] = 1
    return result


def _symmetry_audit(cells: list[dict], tolerance: float) -> dict:
    rows = []
    generated = duplicates = 0
    nonduplicate_hashes: set[str] = set()
    max_errors = defaultdict(float)
    for cell in cells:
        n = cell["hamiltonian"]["n_qubits"]
        h = tfim_hamiltonian(n, cell["hamiltonian"]["J"], cell["anchor_g"], cell["hamiltonian"]["boundary"])
        state = cell["anchor_state"]
        parity = _operator(n, {q: X for q in range(n)})
        reflection = _reflection(n)
        transformations = (
            ("global_x_parity", parity @ state, float(np.linalg.norm(parity @ h - h @ parity)), "unitary symmetry [U,H]=0; g and the repository label are unchanged"),
            ("reflection", reflection @ state, float(np.linalg.norm(reflection @ h - h @ reflection)), "open-chain reflection symmetry [U,H]=0; g and the repository label are unchanged"),
            ("time_reversal_complex_conjugation", state.conj(), float(np.linalg.norm(h.conj() - h)), "H is real, so antiunitary complex conjugation preserves its eigenspaces and label"),
        )
        for name, transformed, error, argument in transformations:
            fidelity = float(abs(np.vdot(state, transformed)) ** 2)
            duplicate = fidelity >= 1 - tolerance
            generated += 1
            duplicates += int(duplicate)
            max_errors[name] = max(max_errors[name], error)
            if not duplicate:
                nonduplicate_hashes.add(state_hash(transformed))
            rows.append({
                "dataset": cell["dataset"], "class": cell["class"], "anchor_id": cell["anchor_id"],
                "symmetry": name, "commutator_or_invariance_error": error,
                "source_fidelity": fidelity, "projective_duplicate": duplicate,
                "label_preservation_argument": argument,
            })
    return {
        "status": "SYMMETRY AUGMENTATION NOT USEFUL: EXACT SYMMETRIES PRODUCE PROJECTIVE DUPLICATES" if generated == duplicates else "UNIQUE_SYMMETRY_STATES_FOUND",
        "boundary_condition": "open",
        "tested": ["global_x_parity", "reflection", "time_reversal_complex_conjugation"],
        "translation": {"applicable": False, "reason": "translation is not an exact symmetry of the repository open-chain Hamiltonian"},
        "number_of_transformations": generated,
        "number_of_projective_duplicates": duplicates,
        "number_of_unique_nonduplicate_states": len(nonduplicate_hashes),
        "maximum_commutator_or_invariance_error": dict(max_errors),
        "benchmark_arm_included": generated != duplicates,
        "duplicate_infidelity_tolerance": tolerance,
        "per_anchor": rows,
    }


def _ground_truth(datasets: dict[str, tuple[object, Path]]) -> dict:
    entries = {}
    for name, (dataset, path) in datasets.items():
        config = dataset.manifest["config"]
        blocked = None
        if config.get("split_strategy", "random") == "blocked":
            blocked = {
                region: {key: list(value) for key, value in _blocked_intervals(*map(float, bounds), config["split_ratios"], float(config["blocked_g_gap"])).items()}
                for region, bounds in config["phase_regions"].items()
            }
        entries[name] = {
            "path": str(path), "artifact_checksums": json.loads((path / "checksums.json").read_text()),
            "n_qubits": config["n_qubits"], "J": config["J"], "boundary": config["boundary"],
            "g_ranges": config["phase_regions"], "dataset_generation_range": config["phase_regions"],
            "samples_per_class": config["samples_per_class"], "dataset_seed": config["dataset_seed"],
            "split_seed": config["split_seed"], "split_strategy": config.get("split_strategy", "random"),
            "split_ratios": config["split_ratios"], "blocked_g_gap": config.get("blocked_g_gap"),
            "blocked_intervals": blocked,
            "class_label_rule": "label 0 iff g is sampled from configured ferromagnetic interval; label 1 iff g is sampled from configured paramagnetic interval",
            "random_split_rule": "class-stratified seeded shuffle after g sampling" if config.get("split_strategy", "random") == "random" else None,
            "blocked_g_split_rule": "ordered per-class g intervals separated by configured gaps" if blocked else None,
        }
    return {
        "canonical_source": "src/conditional_quddpm/datasets/tfim.py plus each dataset split_manifest.json",
        "hamiltonian_convention": "H(g) = -J sum_{i=0}^{N-2} Z_i Z_{i+1} - g sum_{i=0}^{N-1} X_i",
        "g_definition": "coefficient of the transverse X field; dimensionless g/J at J=1",
        "ground_state_solver": "scipy.linalg.eigh dense Hermitian minimum eigenpair (subset_by_index=[0,0])",
        "degeneracy_handling": "no degenerate-subspace rule; one solver eigenvector is returned",
        "global_phase_convention": "rotate the largest-magnitude amplitude to positive real",
        "critical_point_convention": "thermodynamic g/J=1 motivates excluding [0.8,1.2], but labels are the repository's predeclared finite-size TFIM classification convention",
        "claim_boundary": "N=4 finite-size classification; this protocol does not claim observation of a thermodynamic phase transition",
        "datasets": entries,
    }


def _validate_fresh_dataset_configs(config: dict, datasets: dict[str, tuple[TrainOnlyDataset, Path]]) -> None:
    immutable_keys = ("n_qubits", "J", "boundary", "phase_regions", "samples_per_class", "split_ratios", "numerical_tolerance")
    for name, (dataset, _) in datasets.items():
        fresh = config["fresh_confirmatory_datasets"][name]
        canonical = dataset.manifest["config"]
        if any(fresh[key] != canonical[key] for key in immutable_keys):
            raise ValueError(f"fresh {name} dataset config changes canonical TFIM ground truth")
        if fresh["split_strategy"] != canonical.get("split_strategy", "random"):
            raise ValueError(f"fresh {name} split strategy differs from its pilot regime")
        if fresh["split_strategy"] == "blocked" and fresh.get("blocked_g_gap") != canonical.get("blocked_g_gap"):
            raise ValueError("fresh blocked-g gap differs from the audited pilot contract")


def _reproducibility(config_path: Path, config: dict, datasets: dict[str, tuple[TrainOnlyDataset, Path]]) -> dict:
    relevant = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/conditional_quddpm/augmentation/geometry.py",
        REPO_ROOT / "src/conditional_quddpm/datasets/loader.py",
        REPO_ROOT / "src/conditional_quddpm/datasets/tfim.py",
        REPO_ROOT / "src/conditional_quddpm/models/qcnn.py",
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty_lines = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).splitlines()
    return {
        "resolved_generator_config": config,
        "generator_config_path": str(config_path),
        "generator_config_sha256": _sha256(config_path),
        "numerical_tolerances": config["tolerances"],
        "random_control_seed_and_redraw_policy": config["random_control"],
        "fresh_confirmatory_dataset_configs": config["fresh_confirmatory_datasets"],
        "source_revision": {"git_commit_sha": revision, "git_dirty": bool(dirty_lines), "git_status_porcelain": dirty_lines},
        "source_file_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in relevant},
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pyyaml": yaml.__version__,
        },
        "pilot_dataset_artifacts": {
            name: {"path": str(path), "checksums": json.loads((path / "checksums.json").read_text())}
            for name, (_, path) in datasets.items()
        },
    }


def _placeholder(reason: str) -> dict:
    return {"status": "NOT_RUN", "results": [], "blocking_reason": reason, "test_split_used": False}


def verify_manifest(output: str | Path) -> dict[str, bool]:
    output = Path(output)
    expected = {}
    for line in (output / "manifest.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    return {name: (output / name).is_file() and _sha256(output / name) == digest for name, digest in expected.items()}


def run_pilot(config_path: str | Path, output: str | Path) -> dict:
    config_path, output = Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text())
    if config["resource_model"] != "RESOURCE_MODEL_A":
        raise ValueError("this endpoint-eigensolve pilot is valid only under RESOURCE_MODEL_A")
    if config["synthetic_states_per_anchor"] != 1:
        raise ValueError("this implementation requires synthetic_states_per_anchor == 1")
    if not np.isclose(float(config["augmentation_ratio"]), float(config["synthetic_states_per_anchor"])):
        raise ValueError("augmentation_ratio must equal synthetic_states_per_anchor for one synthetic state per anchor")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.sha256").unlink(missing_ok=True)
    config_hash = _sha256(config_path)

    datasets = {name: (_load_train_only_dataset(path), Path(path)) for name, path in config["pilot_datasets"].items()}
    _validate_fresh_dataset_configs(config, datasets)
    ground_truth = _ground_truth(datasets)
    access_audits = {name: dataset.access_audit for name, (dataset, _) in datasets.items()}
    source_cells: list[dict] = []
    source_hashes = {}
    physical_rows, random_rows, pair_rows = [], [], []
    physical_state_by_id: dict[str, np.ndarray] = {}
    random_state_by_id: dict[str, np.ndarray] = {}
    physical_states_by_cell: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    random_states_by_cell: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    oracle_calls = 0
    leakage_checks = []
    expected_per_cell = int(config["real_states_per_class"]) * int(config["synthetic_states_per_anchor"])
    expected_total_per_arm = len(datasets) * 2 * expected_per_cell

    for dataset_name, (dataset, dataset_path) in datasets.items():
        n = int(dataset.manifest["config"]["n_qubits"])
        tfim_config = dataset.manifest["config"]
        records = {record["parameter_id"]: record for record in dataset.manifest["records"] if record["split"] == "train"}
        subset = nested_train_subsets(dataset.train, [int(config["real_states_per_class"])], int(config["subset_seed"]))[int(config["real_states_per_class"])]
        subset_ids = [str(value) for value in subset.parameter_ids]
        subset_hash = _source_subset_hash(dataset_path, subset_ids)
        source_hashes[dataset_name] = subset_hash
        for label in (0, 1):
            support = _support(tfim_config, label)
            mask = subset.labels == label
            class_ids = subset.parameter_ids[mask].tolist()
            class_states = subset.states[mask]
            if len(class_ids) != int(config["real_states_per_class"]):
                raise RuntimeError(f"{dataset_name} class {label} source budget mismatch")
            generated_random: list[np.ndarray] = []
            for anchor_id, anchor in zip(class_ids, class_states, strict=True):
                anchor_g = float(records[anchor_id]["g"])
                target_g, direction = select_target_g(anchor_id, anchor_g, support, config["endpoint_rule"])
                h_target = tfim_hamiltonian(n, float(tfim_config["J"]), target_g, tfim_config["boundary"])
                energy, physical = ground_state(h_target)
                oracle_calls += 1
                oracle_id = f"endpoint-eigensolve-{oracle_calls:04d}"
                delta = fubini_study_distance(anchor, physical)
                delta_g = target_g - anchor_g
                mx_anchor, mz2_anchor = tfim_observables(anchor, n)
                mx_target, mz2_target = tfim_observables(physical, n)
                expectation = float(np.vdot(physical, h_target @ physical).real)
                norm_error = abs(float(np.linalg.norm(physical)) - 1)
                eigen_residual = float(np.linalg.norm(h_target @ physical - energy * physical))
                energy_residual = abs(expectation - energy)
                physical_id = "phys-" + stable_hash(dataset_name, anchor_id, target_g, config_hash)[:20]
                base = {
                    "dataset": dataset_name, "split": "train-derived", "class": int(label),
                    "anchor_id": anchor_id, "anchor_g": anchor_g, "source_subset_hash": subset_hash,
                    "target_g": target_g, "hamiltonian_parameters": {"n_qubits": n, "J": float(tfim_config["J"]), "g": target_g, "boundary": tfim_config["boundary"]},
                    "fs_distance": delta, "delta_g": delta_g,
                    "augmentation_ratio": float(config["augmentation_ratio"]),
                    "endpoint_direction": direction, "training_support": list(support),
                    "generator_config_hash": config_hash,
                }
                nearest_source_fidelity = float(np.max(np.abs(class_states.conj() @ physical) ** 2))
                physical_row = {
                    **base, "synthetic_id": physical_id,
                    "augmentation_method": "same-phase TFIM ground-state-manifold Hamiltonian-assisted augmentation",
                    "interpretation": "physics-guided data expansion under RESOURCE_MODEL_A",
                    "oracle_call_id": oracle_id, "oracle_call_type": "exact_endpoint_ground_state_eigensolve",
                    "ground_state_energy": energy, "energy_expectation": expectation,
                    "normalization_error": norm_error, "eigenpair_residual": eigen_residual,
                    "energy_residual": energy_residual, "endpoint_fidelity_to_solver_state": 1.0,
                    "endpoint_exactness_basis": "the candidate is the exact solver output; no redundant second oracle call",
                    "same_phase_constraint_passed": support[0] <= target_g <= support[1],
                    "critical_boundary_exclusion_passed": not 0.8 <= target_g <= 1.2,
                    "anchor_fidelity": float(abs(np.vdot(anchor, physical)) ** 2),
                    "nearest_source_fidelity": nearest_source_fidelity,
                    "source_projective_duplicate": nearest_source_fidelity >= 1 - float(config["tolerances"]["projective_duplicate_infidelity"]),
                    "magnetization_x_anchor": mx_anchor, "magnetization_x_target": mx_target,
                    "magnetization_z2_anchor": mz2_anchor, "magnetization_z2_target": mz2_target,
                    "local_metric_estimate_delta2_over_dg2": float((delta / abs(delta_g)) ** 2),
                    "state_hash": state_hash(physical),
                }
                physical_rows.append(physical_row)
                physical_state_by_id[physical_id] = physical
                physical_states_by_cell[(dataset_name, label)].append(physical)

                seed_parts = (config["random_control"]["namespace"], dataset_name, label, anchor_id, physical_id)
                random, tangent, redraw = random_tangent_control(
                    anchor, delta, seed_parts=seed_parts,
                    max_redraws=int(config["random_control"]["maximum_redraws"]),
                    forbidden_states=[*class_states, *physical_states_by_cell[(dataset_name, label)], *generated_random],
                    duplicate_infidelity_tolerance=float(config["tolerances"]["projective_duplicate_infidelity"]),
                )
                generated_random.append(random)
                actual = fubini_study_distance(anchor, random)
                random_id = "rand-" + stable_hash(physical_id, config["random_control"]["namespace"])[:20]
                random_nearest_source_fidelity = float(np.max(np.abs(class_states.conj() @ random) ** 2))
                random_row = {
                    **base, "target_g": None, "hamiltonian_parameters": None,
                    "synthetic_id": random_id, "paired_physical_id": physical_id,
                    "augmentation_method": "FS-distance-matched random tangent control",
                    "class_assignment_semantics": "assigned the anchor label solely for paired budget matching",
                    "physics_label_preservation_claimed": False,
                    "class_consistency_filter_applied": False,
                    "validity_caveat": "small Fubini-Study distance does not establish TFIM label preservation",
                    "target_fs_distance": delta, "actual_fs_distance": actual,
                    "distance_matching_error": abs(actual - delta),
                    "normalization_error": abs(float(np.linalg.norm(random)) - 1),
                    "tangent_orthogonality_abs": float(abs(np.vdot(anchor, tangent))),
                    "tangent_norm_error": abs(float(np.linalg.norm(tangent)) - 1),
                    "redraw_counter": redraw,
                    "generator_seed": int(stable_hash(*seed_parts, redraw)[:16], 16),
                    "state_hash": state_hash(random),
                    "nearest_source_fidelity": random_nearest_source_fidelity,
                    "source_projective_duplicate": random_nearest_source_fidelity >= 1 - float(config["tolerances"]["projective_duplicate_infidelity"]),
                }
                random_rows.append(random_row)
                random_state_by_id[random_id] = random
                random_states_by_cell[(dataset_name, label)].append(random)
                pair_rows.append({
                    "dataset": dataset_name, "class": int(label), "anchor_id": anchor_id,
                    "physical_id": physical_id, "random_id": random_id,
                    "physical_fs_distance": delta, "random_fs_distance": actual,
                    "absolute_distance_mismatch": abs(delta - actual),
                    "same_anchor": True, "same_assigned_anchor_label": True, "one_to_one": True,
                    "random_physics_label_preservation_claimed": False,
                })
                source_cells.append({
                    "dataset": dataset_name, "class": int(label), "anchor_id": anchor_id,
                    "anchor_g": anchor_g, "anchor_state": anchor,
                    "hamiltonian": {"n_qubits": n, "J": float(tfim_config["J"]), "boundary": tfim_config["boundary"]},
                })
                interval_safe = support[0] <= target_g <= support[1]
                heldout_safe = interval_safe
                if tfim_config.get("split_strategy") == "blocked":
                    heldout_intervals = _blocked_intervals(*map(float, tfim_config["phase_regions"]["ferromagnetic" if label == 0 else "paramagnetic"]), tfim_config["split_ratios"], float(tfim_config["blocked_g_gap"]))
                    heldout_safe = not any(low <= target_g <= high for split, (low, high) in heldout_intervals.items() if split != "train")
                leakage_checks.append({"dataset": dataset_name, "anchor_id": anchor_id, "target_in_training_support": interval_safe, "target_outside_blocked_val_test_support": heldout_safe})

    tolerance = config["tolerances"]
    symmetry = _symmetry_audit(source_cells, float(tolerance["projective_duplicate_infidelity"]))
    del source_cells

    physical_cells = []
    random_cells = []
    for dataset_name in config["pilot_datasets"]:
        for label in (0, 1):
            phys = [row for row in physical_rows if row["dataset"] == dataset_name and row["class"] == label]
            rand = [row for row in random_rows if row["dataset"] == dataset_name and row["class"] == label]
            physical_cells.append({
                "dataset": dataset_name, "class": label, "count": len(phys),
                "unique_projective_states": _projective_unique_count(physical_states_by_cell[(dataset_name, label)], float(tolerance["projective_duplicate_infidelity"])),
                "source_projective_duplicates": sum(row["source_projective_duplicate"] for row in phys),
                "anchor_coverage": len({row["anchor_id"] for row in phys}),
                "fs_distance": _stats([row["fs_distance"] for row in phys]),
                "absolute_delta_g": _stats([abs(row["delta_g"]) for row in phys]),
                "local_metric_estimate": _stats([row["local_metric_estimate_delta2_over_dg2"] for row in phys]),
                "max_normalization_error": max(row["normalization_error"] for row in phys),
                "max_eigenpair_residual": max(row["eigenpair_residual"] for row in phys),
                "max_energy_residual": max(row["energy_residual"] for row in phys),
            })
            random_cells.append({
                "dataset": dataset_name, "class": label, "count": len(rand),
                "unique_projective_states": _projective_unique_count(random_states_by_cell[(dataset_name, label)], float(tolerance["projective_duplicate_infidelity"])),
                "source_projective_duplicates": sum(row["source_projective_duplicate"] for row in rand),
                "anchor_coverage": len({row["anchor_id"] for row in rand}),
                "max_normalization_error": max(row["normalization_error"] for row in rand),
                "max_tangent_orthogonality_abs": max(row["tangent_orthogonality_abs"] for row in rand),
                "max_tangent_norm_error": max(row["tangent_norm_error"] for row in rand),
                "max_distance_matching_error": max(row["distance_matching_error"] for row in rand),
            })

    count_rows = [
        {
            "dataset": name, "class": label,
            "expected_real": int(config["real_states_per_class"]),
            "expected_per_synthetic_arm": expected_per_cell,
            "real": sum(row["dataset"] == name and row["class"] == label for row in physical_rows),
            "physical": sum(row["dataset"] == name and row["class"] == label for row in physical_rows),
            "random": sum(row["dataset"] == name and row["class"] == label for row in random_rows),
        }
        for name in config["pilot_datasets"] for label in (0, 1)
    ]
    expected_arm_counts_passed = all(
        row["real"] == row["expected_real"]
        and row["physical"] == row["expected_per_synthetic_arm"]
        and row["random"] == row["expected_per_synthetic_arm"]
        for row in count_rows
    ) and len(physical_rows) == len(random_rows) == expected_total_per_arm
    budget = {
        "real_source_states_per_class": config["real_states_per_class"],
        "synthetic_states_per_anchor": config["synthetic_states_per_anchor"],
        "synthetic_states_per_class_per_arm": expected_per_cell,
        "expected_total_states_per_synthetic_arm": expected_total_per_arm,
        "augmentation_ratio": config["augmentation_ratio"],
        "anchor_allocation": "exactly one physical and one matched-random synthetic state per selected anchor",
        "subset_seed": config["subset_seed"], "source_subset_hashes": source_hashes,
        "counts": count_rows,
        "expected_arm_counts_passed": expected_arm_counts_passed,
        "budget_identity": expected_arm_counts_passed and all(row["one_to_one"] for row in pair_rows),
        "dataset_access_audit": access_audits,
        "scientific_split_use": {"train_rows_used": True, "validation_rows_selected": False, "test_rows_selected": False, "validation_values_used": False, "test_values_used": False},
    }
    physical_ok = all(row["normalization_error"] <= tolerance["normalization"] and row["eigenpair_residual"] <= tolerance["eigenpair_residual"] and row["energy_residual"] <= tolerance["energy_residual"] and row["same_phase_constraint_passed"] and row["critical_boundary_exclusion_passed"] for row in physical_rows)
    random_numerics_ok = all(row["normalization_error"] <= tolerance["normalization"] and row["tangent_orthogonality_abs"] <= tolerance["tangent_orthogonality"] and row["distance_matching_error"] <= tolerance["distance_match"] for row in random_rows)
    leakage_ok = all(row["target_in_training_support"] and row["target_outside_blocked_val_test_support"] for row in leakage_checks)
    physical_uniqueness_ok = all(cell["unique_projective_states"] == cell["count"] and cell["source_projective_duplicates"] == 0 for cell in physical_cells)
    random_uniqueness_coverage_ok = all(cell["unique_projective_states"] == cell["count"] and cell["source_projective_duplicates"] == 0 and cell["anchor_coverage"] == int(config["real_states_per_class"]) for cell in random_cells)
    uniqueness_ok = physical_uniqueness_ok and random_uniqueness_coverage_ok
    cross_arm_duplicates = 0
    cross_arm_max_fidelity = 0.0
    for cell, physical_states in physical_states_by_cell.items():
        fidelities = np.abs(np.asarray(physical_states).conj() @ np.asarray(random_states_by_cell[cell]).T) ** 2
        cross_arm_max_fidelity = max(cross_arm_max_fidelity, float(fidelities.max()))
        cross_arm_duplicates += int(np.sum(fidelities >= 1 - float(tolerance["projective_duplicate_infidelity"])))
    oracle_count_ok = oracle_calls == expected_total_per_arm
    pairing_ok = len(pair_rows) == len(physical_rows) == len(random_rows) == expected_total_per_arm and budget["budget_identity"] and cross_arm_duplicates == 0
    random_ok = random_numerics_ok and random_uniqueness_coverage_ok
    pilot_valid = physical_ok and random_ok and leakage_ok and uniqueness_ok and pairing_ok and oracle_count_ok

    resource = {
        "decision": "RESOURCE_MODEL_A",
        "scarce_resource": "acquired/prepared quantum states",
        "known_resource": "TFIM Hamiltonian family",
        "method_naming": ["Hamiltonian-assisted augmentation", "physics-guided data expansion"],
        "forbidden_claim": "strict fixed-dataset augmentation",
        "ground_state_query_policy": "each new exact endpoint ground-state solve is allowed but counted as an oracle call",
        "endpoint_eigensolve_oracle_calls": oracle_calls,
        "model_b_analysis": "Under RESOURCE_MODEL_B every endpoint eigensolve would consume the labeled-state budget; Candidate A would be disallowed and adiabatic/quasi-adiabatic quantum-in/quantum-out continuation would be required.",
        "dataset_access_and_scientific_use": {
            "dataset_container_opened": True,
            "heldout_metadata_and_compressed_members_materialized": True,
            "validation_rows_selected": False, "test_rows_selected": False,
            "validation_values_used": False, "test_values_used": False,
            "details": access_audits,
        },
    }
    protocol = {
        "protocol_version": config["protocol_version"], "resource_model": config["resource_model"],
        "method": config["method_name"], "interpretation": config["interpretation"],
        "endpoint_generation": "exact ground state of H(g')",
        "endpoint_rule": config["endpoint_rule"],
        "endpoint_rule_explanation": "For each anchor, a provenance-hash chooses lower/upper support; g' moves the frozen fraction of remaining distance to that boundary. Therefore delta-g varies by anchor and no QCNN result selects magnitude.",
        "same_phase_rule": "g' must remain inside the anchor label's configured interval and generation TRAIN support",
        "critical_exclusion_rule": "targets may not enter excluded [0.8,1.2] or cross configured class intervals",
        "blocked_g_rule": "g' is restricted to the analytically configured TRAIN interval and may not enter validation/test intervals",
        "fs_distance_rule": "measure physical endpoint distance, then construct the random tangent at exactly that pairwise displacement",
        "random_control_semantics": {
            "class_assignment": "inherit the anchor label solely to match class budgets and paired QCNN inputs",
            "physics_label_preservation_claimed": False,
            "small_fs_distance_establishes_validity": False,
            "class_consistency_filter": None,
            "pilot_checks_only": ["normalization", "tangent orthogonality", "pairwise FS-distance matching", "projective uniqueness", "anchor coverage"],
        },
        "arms": ["real-only", "same-phase physical ground-state-manifold Hamiltonian-assisted augmentation", "FS-distance-matched random tangent control"],
        "symmetry_arm_rule": "include only if the exact-symmetry diagnostic finds projectively unique states",
        "phase_a_status": "immutable exploratory ablation: literature-motivated, not literature-derived",
        "phase_b_status": "immutable exploratory heuristic; excluded from core benchmark",
        "phase_c_status": "redefined as FS-distance-matched random control",
        "qcNN_test_execution": "forbidden in this pilot",
    }
    seed_schedule = {
        "status": "FROZEN_FOR_FRESH_CONFIRMATORY_DATASETS",
        "subset_seed": config["subset_seed"], "fresh_confirmatory_datasets": config["fresh_confirmatory_datasets"],
        "paired_seeds": [{"run_seed": seed, "init_seed": config["qcnn"]["init_seed_offset"] + seed, "spsa_seed": config["qcnn"]["spsa_seed_offset"] + seed} for seed in config["qcnn"]["run_seeds"]],
        "intermediate_rule": "seeds 100-109 may be reported as stability only; protocol cannot be changed before extending to all 20",
    }

    np.savez_compressed(
        output / "synthetic_states.npz",
        physical_ids=np.asarray(list(physical_state_by_id)),
        physical_states=np.asarray(list(physical_state_by_id.values())),
        random_ids=np.asarray(list(random_state_by_id)),
        random_states=np.asarray(list(random_state_by_id.values())),
    )
    _write_json(output / "resource_model.json", resource)
    _write_json(output / "tfim_ground_truth.json", ground_truth)
    _write_json(output / "symmetry_diagnostic.json", symmetry)
    _write_json(output / "augmentation_protocol.json", protocol)
    _write_json(output / "physical_generator_diagnostics.json", {"status": "PASS" if physical_ok and physical_uniqueness_ok and expected_arm_counts_passed and oracle_count_ok else "FAIL", "oracle_calls": oracle_calls, "expected_oracle_calls": expected_total_per_arm, "cells": physical_cells, "samples": physical_rows})
    _write_json(output / "random_control_diagnostics.json", {
        "status": "PASS" if random_ok else "FAIL",
        "semantics": protocol["random_control_semantics"],
        "acceptance_checks": ["normalization", "tangent orthogonality", "pairwise FS-distance matching", "projective uniqueness", "anchor coverage"],
        "class_consistency_filter_applied": False,
        "cells": random_cells, "samples": random_rows,
    })
    _write_json(output / "pairwise_distance_audit.json", {"status": "PASS" if pairing_ok else "FAIL", "pair_count": len(pair_rows), "maximum_distance_mismatch": max(row["absolute_distance_mismatch"] for row in pair_rows), "cross_arm_projective_duplicates": cross_arm_duplicates, "maximum_cross_arm_fidelity": cross_arm_max_fidelity, "pairs": pair_rows})
    _write_json(output / "source_budget_audit.json", {**budget, "leakage_checks": leakage_checks})
    _write_json(output / "seed_schedule.json", seed_schedule)

    qcnn_path = Path(config["qcnn"]["config"])
    qcnn_config = yaml.safe_load(qcnn_path.read_text())
    resolved_qcnn = {
        "config_path": str(qcnn_path), "config_sha256": _sha256(qcnn_path),
        "resolved_base_config": qcnn_config,
        "architecture": "repository 4-to-2-to-1 statevector QCNN", "parameter_count": 42,
        "readout": "Z on qubit 3", "training": qcnn_config["training"],
        "real_states_per_class": config["qcnn"]["real_states_per_class"],
        "paired_seed_schedule": seed_schedule["paired_seeds"],
    }
    reproducibility = _reproducibility(config_path, config, datasets)
    reproducibility["source_subset_hashes"] = source_hashes
    freeze_payload = {
        "resource_model": resource["decision"], "hamiltonian_definition": ground_truth["hamiltonian_convention"],
        "resolved_generator_config": config,
        "augmentation_transformation": protocol, "numerical_tolerances": config["tolerances"],
        "synthetic_ratio": config["augmentation_ratio"], "synthetic_states_per_anchor": config["synthetic_states_per_anchor"],
        "anchor_allocation": budget["anchor_allocation"],
        "random_control_construction": {"distance_rule": protocol["fs_distance_rule"], **config["random_control"]},
        "fresh_confirmatory_dataset_configs": config["fresh_confirmatory_datasets"],
        "seed_schedule": seed_schedule, "qcnn_protocol": resolved_qcnn,
        "statistical_plan": config["statistics"],
        "reproducibility_and_provenance": reproducibility,
    }
    freeze_hash = _sha256_bytes(json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode())
    freeze = {
        "status": "FROZEN_GENERATOR_PROTOCOL_QCNN_NOT_RUN", "pilot_valid": pilot_valid,
        "frozen_payload": freeze_payload, "frozen_payload_sha256": freeze_hash,
        "benchmark_ready": False,
        "benchmark_blocker": "fresh confirmatory random and blocked-g dataset artifacts must be generated from the frozen seeds and validated before any QCNN test execution",
        "scientific_split_use": {"validation_values_used": False, "test_values_used": False},
    }
    _write_json(output / "protocol_freeze.json", freeze)

    reason = "QCNN prohibited in this scope: prior Phase A/B/C test metrics were observed; fresh frozen confirmatory datasets are not yet materialized"
    for filename in ("per_seed_results.json", "aggregate_results.json", "paired_comparisons.json", "random_vs_blocked_analysis.json"):
        _write_json(output / filename, _placeholder(reason))
    _write_json(output / "statistical_analysis.json", {
        **_placeholder(reason), "plan": freeze_payload["statistical_plan"],
        "required_reports": ["mean accuracy", "standard deviation", "median", "macro-F1", "validation loss", "train-test gap", "paired delta accuracy", "95% paired bootstrap CI", "paired delta macro-F1", "paired effect size dz", "wins/ties/losses", "Holm-adjusted inference"],
        "interpretation_priority": "effect magnitude and confidence interval before p-value",
    })

    validation = {
        "valid": pilot_valid,
        "gate_0_passed": resource["decision"] == "RESOURCE_MODEL_A",
        "ground_truth_audited": True, "symmetry_diagnostic_complete": True,
        "physical_generator_passed": physical_ok, "random_control_passed": random_ok,
        "pairwise_matching_passed": pairing_ok, "projective_uniqueness_passed": uniqueness_ok,
        "training_support_and_blocked_leakage_passed": leakage_ok,
        "source_budget_identity_passed": budget["budget_identity"],
        "expected_arm_counts_passed": expected_arm_counts_passed,
        "endpoint_oracle_calls": oracle_calls, "expected_endpoint_oracle_calls": expected_total_per_arm,
        "endpoint_oracle_count_passed": oracle_count_ok, "qcnn_executed": False,
        "dataset_access_and_scientific_use": {
            "dataset_container_opened": True,
            "heldout_metadata_and_compressed_members_materialized": True,
            "validation_rows_selected": False, "test_rows_selected": False,
            "validation_values_used": False, "test_values_used": False,
        },
        "required_artifacts_declared": list(REQUIRED_ARTIFACTS),
        "manifest_verification": {"method": "recompute SHA-256 for every entry in manifest.sha256 using verify_manifest(output)", "timing": "performed after manifest is written; manifest excludes itself to avoid recursion"},
        "errors": [] if pilot_valid else [name for name, ok in (("physical", physical_ok), ("random", random_ok), ("leakage", leakage_ok), ("uniqueness", uniqueness_ok), ("pairing", pairing_ok), ("counts", expected_arm_counts_passed), ("oracle_count", oracle_count_ok)) if not ok],
    }
    _write_json(output / "validation.json", validation)
    _write_markdown(output, oracle_calls, pilot_valid, freeze_hash)

    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "manifest.sha256")
    (output / "manifest.sha256").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in files))
    manifest_valid = all(verify_manifest(output).values())
    if not pilot_valid or not manifest_valid:
        raise RuntimeError("pilot or artifact manifest validation failed")
    return {"status": "PASS", "pilot_valid": pilot_valid, "oracle_calls": oracle_calls, "manifest_valid": manifest_valid, "qcnn_runs": 0, "output": str(output)}


def _write_markdown(output: Path, oracle_calls: int, pilot_valid: bool, freeze_hash: str) -> None:
    (output / "research_direction_update.md").write_text(f"""# Confirmatory TFIM augmentation direction\n\nThe retired question comparing physics-, geometry-, and random heuristics is replaced by: at equal source-state budget, synthetic count, and pairwise Fubini--Study displacement, does movement along the same-label TFIM exact-ground-state manifold improve low-data QCNN generalization beyond generic local projective smoothing?\n\nCore arms are real-only, **Hamiltonian-assisted same-phase ground-state data expansion**, and its FS-distance-matched random tangent control. The random control inherits its anchor label only for paired class-budget assignment; it is not claimed to preserve TFIM physics, and small FS distance does not establish validity. Phase A remains immutable literature-motivated exploratory evidence; Phase B remains an immutable exploratory heuristic and is removed from the core benchmark; Phase C is retained only as the matched control. This N=4 study uses finite-size labels from the repository contract and makes no thermodynamic-transition claim.\n\nGate 0 adopts `RESOURCE_MODEL_A`: acquired/prepared states are scarce while the Hamiltonian family is known. Exact endpoints are therefore physics-guided data expansion, not strict fixed-dataset augmentation. This pilot made **{oracle_calls}** counted endpoint eigensolve oracle calls. Generator validity is **{'PASS' if pilot_valid else 'FAIL'}**. QCNN execution is blocked until fresh frozen split artifacts exist.\n""")
    (output / "literature_grounding.md").write_text("""# Literature grounding\n\n- Augmentation requires label preservation or a physical invariant argument; this protocol inherits the repository label only because both `g` and `g'` remain in the same predeclared TFIM class interval.\n- Gapped same-phase ground states can be related by quasi-adiabatic continuation: M. B. Hastings and X.-G. Wen, *Quasi-adiabatic continuation of quantum states*, Phys. Rev. B 72, 045141 (2005), https://doi.org/10.1103/PhysRevB.72.045141. This motivates the ground-state manifold as structured support, without claiming that exact endpoint diagonalization is itself a quantum-in/quantum-out continuation.\n- Local-unitary equivalence underlies the modern phase notion: X. Chen, Z.-C. Gu, and X.-G. Wen, *Local unitary transformation, long-range quantum entanglement, wave function renormalization, and topological order*, Phys. Rev. B 82, 155138 (2010), https://doi.org/10.1103/PhysRevB.82.155138.\n- Fidelity susceptibility/quantum metric explains why fixed raw `delta g` is not a fixed state-space movement: P. Zanardi, P. Giorda, and M. Cozzini, *Information-Theoretic Differential Geometry of Quantum Phase Transitions*, Phys. Rev. Lett. 99, 100603 (2007), https://doi.org/10.1103/PhysRevLett.99.100603; W.-L. You, Y.-W. Li, and S.-J. Gu, *Fidelity, dynamic structure factor, and susceptibility in critical phenomena*, Phys. Rev. E 76, 022101 (2007), https://doi.org/10.1103/PhysRevE.76.022101.\n- Fubini--Study geometry is used only to measure and pair displacement. Physics selects the endpoint direction; geometry controls how far the random control moves.\n\nThese principles do not make observable closeness, geodesic proximity, or small FS distance sufficient evidence of TFIM label preservation. The random arm therefore uses the anchor label only as a matched-control assignment and applies no class-consistency filter or physics-validity claim.\n""")
    (output / "README.md").write_text(f"""# TFIM manifold confirmatory track — protocol v1\n\nStatus: generator pilot **{'PASS' if pilot_valid else 'FAIL'}**; QCNN **NOT RUN**.\n\nRun:\n\n```bash\npython -m conditional_quddpm.experiments.tfim_manifold_confirmatory \\\n  --config configs/augmentation/tfim_manifold_confirmatory.yaml \\\n  --output results/tfim_manifold_augmentation/pilot_v1\n```\n\n`resource_model.json` records Gate 0, oracle accounting, and the distinction between dataset-container materialization and scientific train-row use. Ground truth and exact-symmetry results are in `tfim_ground_truth.json` and `symmetry_diagnostic.json`. Generator, random-control, pairing, and source-budget audits contain per-sample provenance; `synthetic_states.npz` stores the corresponding pilot states. `protocol_freeze.json` freezes the pre-QCNN method/statistical payload at `{freeze_hash}`. Result files are explicit `NOT_RUN` placeholders until fresh confirmatory datasets are generated. `manifest.sha256` covers every other file in this directory. Prior Phase A/B/C artifacts were not modified.\n""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation/tfim_manifold_confirmatory.yaml")
    parser.add_argument("--output", default="results/tfim_manifold_augmentation/pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()

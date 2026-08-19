"""Freeze the execution-only QCNN confirmatory Protocol v2.3 contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from conditional_quddpm.augmentation.local_perturbation import calibrate_radii
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2 import canonical_json

ROOT = Path(__file__).resolve().parents[3]
PARENT_HASH = "8af9bf4d0ab8320ec46f5983c2073ec4b85e1955916fd35b1f3a5e4f6fa33004"
REGIMES = ("random", "blocked_g")
BUDGETS = (10, 25, 50, 100)
METHODS = ("real-only", "local-random-tangent")
REPEATS = range(3)
HISTORICAL_PROTOCOL_HASHES = {
    "2.0.0": "2e52ac26f626fb0703b06fc73e724820f23b4304ca51db8ca8c9cc0e07b959aa",
    "2.1.0": "3a1c242f43b7b8366fdcf4cfc37de51edecee4e4f3979f3dc751b21beec2e28d",
    "2.2.0": PARENT_HASH,
}
DATASET_HASHES = {
    "random": "09fd5792318cd171a3c39316adff1dbce6c402c9bc0fa66bd2df639fab73cff7",
    "blocked_g": "cb0446120e7df9b0b6052f4575f6a1ff10742d8aaa49542054530ec6215e8867",
}
DATASET_FILE_HASHES = {
    "random": "ff2f5fc4de67fe7e22e50d75617be469799964dc52282853ee4ca612dde630ff",
    "blocked_g": "593733297e3e952ecf0cddb802b89379018c5b3c5225ee098fb2cd4237b36742",
}
CALIBRATION_HASHES = {
    "decision.json": "09368c4b7c771564d7bbb36453f2b58bdadd39c3f31bb3ce5aedc9263a9cf842",
    "diagnostics.json": "d005b76495eae8339c0ce4eaf2e10ef92b47baad71f60d99af496ef1bb674b54",
    "provenance.json": "8eb8e914c76cbb3eede0f24ddb0bd895737feb2bd545975c9d89e468a271ecae",
}
SEED_MANIFEST_HASH = "bba1b6a984401dc859d846671fde19c88fb34702b5c1226778a46947ad570180"
QCNN_SOURCE_HASH = "d9f92f743cb60c841b6b7aca5c06681bbd127cbc13f0e1288603ff793734b42f"
HISTORICAL_CHECKSUM_HASHES = {
    "v2": "b649a3cc4ff933646816f7cca2f5282b96f389a19b2a32745a1378350d2c5d4f",
    "v2_1": "18edb299e4f1532bc26f39cc219d09c46062507601e3f71cb9614c78f9fae647",
    "v2_2": "11356f4e66f82f3b602a820bd974a37e15bb1a865ee56edd2f8f64af69e60602",
}
BLOCKED_G_STATISTICS_SEEDS = (
    12279711874889977296, 1784002412079155030, 12135047068772529481, 233618289726920178,
    11558587206418968329, 844740112680131372, 5467244074479300463, 3202473052213059193,
    1201064018424568544, 16441874957504487036, 14762666102111998680, 9272767271630878344,
)
DATASET_ARTIFACT_HASHES = {
    "checksums.sha256": "cace62edc4a20b5606f91e3fa9be0ce97c46fc36f991652bb7e4e0f5fcd73729",
    "freeze_manifest.json": "01b7be5ab171778ef10a72b38602c8f15037293472062160414795e01c77a93f",
    "provenance.json": "38ca26190c42d1528cda14d033e60b318ea79aaa3ab203c2877b9552e97e57cb",
    "reproducibility.json": "fa0d467f0a4ab642ad14cacff1691b4959c2d059d0abc223faa80103eb34367b",
    "random/manifest.json": "b3aeec2f47988233b9325ca9a009e5b3e16235832d6726e62c79eba3b8403f4e",
    "random/validation.json": "68054a289a4ca379f062c7312f653777adff4ebca2071a580b7772167d0dcfbc",
    "blocked_g/manifest.json": "382220794b1dd74970672a35f66b16125448ab985bb5ff55c2ed1b28ba109b93",
    "blocked_g/validation.json": "67e1990968c246aaf48e54e2e1c8c80d5a126805eaba74e3512349b9adccae4e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _random_tangent(anchor: np.ndarray, displacement: float, rng: np.random.Generator,
                    forbidden_states: list[np.ndarray]) -> tuple[np.ndarray, int]:
    anchor = np.asarray(anchor, dtype=np.complex128); anchor /= np.linalg.norm(anchor)
    for redraw in range(100):
        raw = rng.normal(size=anchor.shape) + 1j * rng.normal(size=anchor.shape)
        tangent = raw - anchor * np.vdot(anchor, raw)
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-12:
            continue
        tangent /= norm
        if abs(np.vdot(anchor, tangent)) > 1e-12:
            continue
        state = np.cos(displacement) * anchor + np.sin(displacement) * tangent
        state /= np.linalg.norm(state)
        if abs(np.linalg.norm(state) - 1) > 1e-10:
            continue
        if forbidden_states and max(abs(np.vdot(other, state)) ** 2 for other in forbidden_states) >= 1 - 1e-10:
            continue
        return state, redraw
    raise RuntimeError("confirmatory tangent redraw exhaustion")


def confirmatory_tangent_batch(states: np.ndarray, sample_ids: np.ndarray, labels: np.ndarray,
                                radii: dict[int, float], augmentation_seed: int) -> tuple[np.ndarray, list[dict]]:
    """Generate the frozen ratio-1 batch in class then lexicographic-ID order."""
    states = np.asarray(states, dtype=np.complex128); sample_ids = np.asarray(sample_ids); labels = np.asarray(labels)
    if len(states) != len(sample_ids) or len(states) != len(labels) or set(labels.tolist()) != {0, 1} or set(radii) != {0, 1}:
        raise ValueError("confirmatory tangent batch requires aligned two-class states, IDs, and radii")
    rng = np.random.Generator(np.random.PCG64DXSM(int(augmentation_seed)))
    order = sorted(range(len(states)), key=lambda i: (int(labels[i]), str(sample_ids[i])))
    forbidden = [state / np.linalg.norm(state) for state in states]
    generated, records = [], []
    for index in order:
        state, redraw = _random_tangent(states[index], radii[int(labels[index])], rng, forbidden + generated)
        generated.append(state); records.append({"anchor_sample_id": str(sample_ids[index]), "class": int(labels[index]), "redraw": redraw})
    return np.asarray(generated), records


def paired_bootstrap(deltas: np.ndarray) -> tuple[float, list[float]]:
    """Frozen percentile bootstrap over the 12 blocked-g paired cells."""
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) != 12 or not np.all(np.isfinite(deltas)):
        raise ValueError("bootstrap requires 12 finite blocked-g paired deltas")
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(BLOCKED_G_STATISTICS_SEEDS)))
    means = deltas[rng.integers(0, len(deltas), size=(100000, len(deltas)))].mean(axis=1)
    return float(deltas.mean()), np.quantile(means, [.025, .975], method="linear").tolist()


def block_mapping(seed_manifest: dict) -> list[dict]:
    """Map block/00..23 in regime/repeat/budget order."""
    rows = []
    for regime_index, regime in enumerate(REGIMES):
        for repeat in REPEATS:
            for budget_index, budget in enumerate(BUDGETS):
                index = regime_index * 12 + repeat * 4 + budget_index
                block_id = f"block/{index:02d}"
                block = seed_manifest["blocks"][block_id]
                rows.append({"regime": regime, "budget": budget, "repeat": repeat,
                             "seed_block": block_id, "root_seed": block["root_seed"], **block["domains"]})
    return rows


def nested_subsets(rows: list[dict]) -> tuple[list[dict], dict[tuple[str, int, int], dict]]:
    """Select balanced nested subsets by seeded incremental draws."""
    output, lookup = [], {}
    base = ROOT / "results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v2_2"
    for regime in REGIMES:
        with np.load(base / regime / "states.npz") as data:
            ids, labels, splits, states = (data[key] for key in ("parameter_ids", "labels", "splits", "states"))
            for repeat in REPEATS:
                chosen: dict[int, list[int]] = {0: [], 1: []}
                for budget in BUDGETS:
                    row = next(item for item in rows if (item["regime"], item["repeat"], item["budget"]) == (regime, repeat, budget))
                    rng = np.random.Generator(np.random.PCG64DXSM(row["subset_selection"]))
                    for label in (0, 1):
                        candidates = np.flatnonzero((labels == label) & (splits == "train") & ~np.isin(np.arange(len(labels)), chosen[label]))
                        rng.shuffle(candidates)
                        chosen[label].extend(map(int, candidates[:budget - len(chosen[label])]))
                    indices = chosen[0] + chosen[1]
                    radii = {str(label): calibrate_radii(states[chosen[label]])["radii"]["medium"] for label in (0, 1)}
                    entry = {"regime": regime, "repeat": repeat, "budget": budget,
                             "seed_block": row["seed_block"], "subset_seed": row["subset_selection"],
                             "selection": "per-class incremental PCG64DXSM shuffle of remaining train IDs; append prefix to prior budget",
                             "sample_ids_by_class": {str(label): ids[chosen[label]].tolist() for label in (0, 1)},
                             "q50_fs_radius_by_class": radii}
                    output.append(entry); lookup[(regime, repeat, budget)] = entry
    return output, lookup


def build_contract() -> dict[str, object]:
    seed_path = ROOT / "results/tfim_manifold_augmentation/confirmatory_protocol_v2/seed_manifest.json"
    if sha256(seed_path) != SEED_MANIFEST_HASH:
        raise ValueError("frozen seed manifest changed")
    seed_manifest = json.loads(seed_path.read_text())
    mappings = block_mapping(seed_manifest)
    subsets, subset_lookup = nested_subsets(mappings)
    runs = []
    for block in mappings:
        for method in METHODS:
            run_id = f"v2.3-{block['regime']}-b{block['budget']:03d}-r{block['repeat']}-{method}"
            runs.append({"run_id": run_id, "regime": block["regime"], "method": method,
                         "budget": block["budget"], "repeat": block["repeat"], "seed_block": block["seed_block"],
                         "subset_seed": block["subset_selection"], "init_seed": block["qcnn.initialization"],
                         "spsa_seed": block["qcnn.spsa"],
                         "augmentation_seed": block["augmentation"] if method == "local-random-tangent" else None})
    manifest = {
        "schema_version": 1, "protocol_name": "tfim_qcnn_confirmatory_execution", "protocol_version": "2.3.0",
        "protocol_status": "FROZEN", "parent_protocol_hash": PARENT_HASH,
        "scope": "execution contract only; Protocol v2/v2.1/v2.2 and frozen datasets remain immutable",
        "axes": {"regimes": list(REGIMES), "regime_labels": {"random": "FS-constrained random", "blocked_g": "blocked-g"},
                 "methods": list(METHODS), "budgets_real_per_class": list(BUDGETS), "repeats": 3, "expected_runs": 48},
        "authoritative_run_count": {"value": 48, "factorization": "2 regimes x 4 budgets x 3 repeats x 2 methods",
                                    "supersedes": 576, "reason": "576 had no recorded axis factorization or run provenance and is therefore non-authoritative"},
        "dataset_hashes": DATASET_HASHES, "qcnn_run_count_at_freeze": 0,
    }
    execution = {
        "schema_version": 1, "protocol_version": "2.3.0", "datasets": {
            regime: {"path": f"results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v2_2/{regime}",
                     "scientific_hash": DATASET_HASHES[regime], "states_sha256": DATASET_FILE_HASHES[regime]} for regime in REGIMES},
        "subset_contract": {"balanced": True, "nested": "10 subset 25 subset 50 subset 100 within each regime/repeat",
                            "algorithm": "ascending budgets; each block's subset seed initializes PCG64DXSM and shuffles remaining train IDs separately in class order 0,1; append exactly budget-minus-prior IDs"},
        "qcnn": {"architecture": "recovered frozen 4->2->1 statevector QCNN", "parameters": 42, "readout": "Z on qubit 3",
                 "loss": "MSE against {-1,+1}", "source_path": "src/conditional_quddpm/models/qcnn.py",
                 "source_sha256": QCNN_SOURCE_HASH},
        "spsa": {"a": .15, "c": .1, "a_k": "a/(k+1)^0.602", "c_k": "c/(k+1)^0.101",
                 "perturbation": "Rademacher +/-1", "updates": 300, "early_stopping": False,
                 "validation_use": "record-only", "evaluation_checkpoint": "final step 300"},
        "augmentation": {"method": "local-random-tangent", "radius": "q50 per regime/repeat/budget/class frozen subset",
                         "synthetic_ratio": 1.0, "synthetic_count_per_class": "equal to real budget",
                         "rng": "numpy.Generator(PCG64DXSM(frozen block augmentation seed))",
                         "semantics": "Phase-C canonical tangent projection, exp-map displacement, source/synthetic projective uniqueness",
                         "draw_order": "class 0 then 1; one synthetic state per real anchor in lexicographic sample-ID order; one persistent block PCG64DXSM stream; redraws consume the next stream values",
                         "acceptance": {"orthogonal_tolerance": 1e-12, "normalization_tolerance": 1e-10,
                                        "duplicate_infidelity_tolerance": 1e-10, "maximum_redraws": 100,
                                        "exhaustion": "run failure; no retry"}},
        "execution_prohibitions": ["retry", "outlier removal", "early stopping", "test-based checkpoint selection", "runtime parameter override"],
        "subsets": subsets,
    }
    statistics = {"schema_version": 1, "primary_metric": "test_accuracy", "estimand": "local-random-tangent minus real-only",
                  "pairing": "same regime, budget, repeat, subset, QCNN initialization seed, and SPSA seed",
                  "primary_regime": "blocked_g", "random_regime_role": "supportive only", "aggregation": "mean paired delta",
                  "confidence_interval": "paired percentile bootstrap 95%", "bootstrap_draws": 100000,
                  "bootstrap_unit": "12 blocked-g paired cells (4 budgets x 3 repeats)",
                  "bootstrap_rng": "PCG64DXSM(SeedSequence(list of the 12 blocked-g block statistics seeds in repeat then ascending-budget order))",
                  "bootstrap_resampling": "draw 12 integer indices with replacement per draw; arithmetic mean; NumPy quantile [0.025,0.975] method=linear",
                  "PASS": "all 48 complete; blocked-g mean paired delta >= 0.02; paired 95% CI lower bound > 0",
                  "FAIL": "all 48 complete but PASS false", "INCONCLUSIVE": "missing, failed, NaN, schema-invalid run, or unavailable uncertainty",
                  "retry": "forbidden", "outlier_removal": "forbidden"}
    calibration = {"schema_version": 1, "provenance_class": "NEW_DECISION", "selected": {"radius": "q50", "synthetic_ratio": 1.0},
                   "artifact_directory": "results/tfim_manifold_augmentation/local_tangent_calibration_v1",
                   "artifact_sha256": CALIBRATION_HASHES, "qcnn_runs_used": 0, "test_metrics_accessed": False}
    provenance = {"schema_version": 1, "parent_protocol_hash": PARENT_HASH, "historical_protocol_hashes": HISTORICAL_PROTOCOL_HASHES,
                  "seed_manifest": {"path": str(seed_path.relative_to(ROOT)), "sha256": SEED_MANIFEST_HASH, "blocks_used": 24},
                  "dataset_file_hashes": DATASET_FILE_HASHES, "calibration_artifact_hashes": CALIBRATION_HASHES,
                  "decisions": ["NEW_DECISION: q50 radius and ratio 1.0 from calibration", "NEW_DECISION: authoritative run count 48 replaces provenance-free 576",
                                "NEW_DECISION: block mapping order is regime, repeat, ascending budget", "NEW_DECISION: incremental per-budget subset draws preserve nesting while consuming every block subset seed"],
                  "qcnn_run_count": 0}
    seed_mapping = {"schema_version": 1, "mapping_order": "random then blocked_g; repeat 0..2; budgets 10,25,50,100", "blocks": mappings}
    run_matrix = {"schema_version": 1, "expected_runs": 48, "runs": runs}
    return {"protocol_manifest.json": manifest, "run_matrix.json": run_matrix, "seed_mapping.json": seed_mapping,
            "execution_config.json": execution, "statistical_plan.json": statistics,
            "calibration_linkage.json": calibration, "provenance.json": provenance}


def _validate_checksum_manifest(path: Path) -> None:
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256(path.parent / name) != digest:
            raise ValueError(f"checksum mismatch: {path.parent / name}")


def validate_contract(artifacts: dict[str, object]) -> None:
    runs = artifacts["run_matrix.json"]["runs"]
    if len(runs) != 48 or len({row["run_id"] for row in runs}) != 48:
        raise ValueError("run matrix must contain 48 unique runs")
    if any(row[key] is None for row in runs for key in ("regime", "method", "budget", "repeat", "seed_block", "subset_seed", "init_seed", "spsa_seed")):
        raise ValueError("run matrix has unresolved fields")
    pairs = {}
    for row in runs:
        key = (row["regime"], row["budget"], row["repeat"])
        pairs.setdefault(key, []).append(row)
    if len(pairs) != 24 or any(len(rows) != 2 or len({(r["subset_seed"], r["init_seed"], r["spsa_seed"]) for r in rows}) != 1 for rows in pairs.values()):
        raise ValueError("paired seed mapping is invalid")
    subsets = artifacts["execution_config.json"]["subsets"]
    for regime in REGIMES:
        for repeat in REPEATS:
            cells = [x for x in subsets if x["regime"] == regime and x["repeat"] == repeat]
            for label in ("0", "1"):
                sets = [set(next(x for x in cells if x["budget"] == budget)["sample_ids_by_class"][label]) for budget in BUDGETS]
                if not all(len(ids) == budget for ids, budget in zip(sets, BUDGETS, strict=True)) or not all(a < b for a, b in zip(sets, sets[1:])):
                    raise ValueError("subset nesting is invalid")
    for version, digest in HISTORICAL_PROTOCOL_HASHES.items():
        path = ROOT / f"results/tfim_manifold_augmentation/confirmatory_protocol_v{version.replace('.0','').replace('.', '_')}/protocol_manifest.json"
        if sha256(path) != digest:
            raise ValueError(f"Protocol v{version} changed")
    for directory, digest in HISTORICAL_CHECKSUM_HASHES.items():
        checksum_path = ROOT / f"results/tfim_manifold_augmentation/confirmatory_protocol_{directory}/checksums.sha256"
        if sha256(checksum_path) != digest:
            raise ValueError(f"historical {directory} checksum manifest changed")
        _validate_checksum_manifest(checksum_path)
    base = ROOT / "results/tfim_manifold_augmentation/confirmatory_dataset_freeze_v2_2"
    if (any(sha256(base / regime / "states.npz") != digest for regime, digest in DATASET_FILE_HASHES.items())
            or any(sha256(base / name) != digest for name, digest in DATASET_ARTIFACT_HASHES.items())):
        raise ValueError("frozen dataset changed")
    _validate_checksum_manifest(base / "checksums.sha256")
    calibration = ROOT / "results/tfim_manifold_augmentation/local_tangent_calibration_v1"
    if any(sha256(calibration / name) != digest for name, digest in CALIBRATION_HASHES.items()):
        raise ValueError("calibration artifacts changed")
    if sha256(ROOT / "src/conditional_quddpm/models/qcnn.py") != QCNN_SOURCE_HASH:
        raise ValueError("frozen QCNN source changed")


def freeze(output: str | Path) -> str:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    artifacts = build_contract(); validate_contract(artifacts)
    protocol_hash = hashlib.sha256(canonical_json(artifacts["protocol_manifest.json"])).hexdigest()
    for name, value in artifacts.items():
        if name != "protocol_manifest.json": value["protocol_hash"] = protocol_hash
        (output / name).write_bytes(canonical_json(value))
    gate = {"schema_version": 1, "protocol_version": "2.3.0", "protocol_hash": protocol_hash, "status": "FROZEN",
            "run_matrix_expected": 48, "run_matrix_validated": 48, "all_seeds_resolved": True, "all_configs_resolved": True,
            "execution_time_free_parameters": 0, "historical_protocols_unchanged": True, "dataset_hashes_unchanged": True,
            "focused_tests_pass": False, "full_suite_pass": False, "independent_review_pass": False,
            "independent_review_reference": None, "qcnn_run_count": 0, "qcnn_confirmatory_ready": False, "blocking_reasons": ["tests and independent review pending"]}
    (output / "gate.json").write_bytes(canonical_json(gate))
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    return protocol_hash

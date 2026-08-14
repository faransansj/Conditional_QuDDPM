"""Exact-diagonalization datasets for the 1D transverse-field Ising model.

The convention is H = -J sum_i Z_i Z_(i+1) - g sum_i X_i.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import eigh

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
CHECKSUM_FILES = ("states.npz", "split_manifest.json", "validation.json")
SPLITS = ("train", "val", "test")


def _operator(n_qubits: int, operators: dict[int, np.ndarray]) -> np.ndarray:
    result = np.array([[1]], dtype=np.complex128)
    for qubit in range(n_qubits):
        result = np.kron(result, operators.get(qubit, I))
    return result


def tfim_hamiltonian(
    n_qubits: int, J: float = 1.0, g: float = 1.0, boundary: str = "open"
) -> np.ndarray:
    """Return the dense TFIM Hamiltonian for open or periodic boundaries."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    size = 2**n_qubits
    hamiltonian = np.zeros((size, size), dtype=np.complex128)
    bonds = [(i, i + 1) for i in range(n_qubits - 1)]
    if boundary == "periodic" and n_qubits > 2:
        bonds.append((n_qubits - 1, 0))

    for left, right in bonds:
        hamiltonian -= J * _operator(n_qubits, {left: Z, right: Z})
    for qubit in range(n_qubits):
        hamiltonian -= g * _operator(n_qubits, {qubit: X})
    return hamiltonian


def ground_state(hamiltonian: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the minimum eigenvalue and a deterministic normalized eigenvector."""
    eigenvalues, eigenvectors = eigh(hamiltonian, subset_by_index=[0, 0])
    state = eigenvectors[:, 0]
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    return float(eigenvalues[0]), state


def _observable_operators(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    mx = sum(_operator(n_qubits, {q: X}) for q in range(n_qubits)) / n_qubits
    mz = sum(_operator(n_qubits, {q: Z}) for q in range(n_qubits)) / n_qubits
    return mx, mz @ mz


def tfim_observables(state: np.ndarray, n_qubits: int) -> tuple[float, float]:
    """Return transverse magnetization <Mx> and longitudinal <Mz^2>."""
    mx, mz2 = _observable_operators(n_qubits)
    return float(np.vdot(state, mx @ state).real), float(np.vdot(state, mz2 @ state).real)


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = int(total * ratios["train"])
    val = int(total * ratios["val"])
    return {"train": train, "val": val, "test": total - train - val}


def _validate_config(config: dict) -> None:
    ferro = config["phase_regions"]["ferromagnetic"]
    para = config["phase_regions"]["paramagnetic"]
    if not (ferro[0] < ferro[1] < para[0] < para[1]):
        raise ValueError("phase regions must be ordered, non-overlapping intervals")
    ratios = config["split_ratios"]
    if any(ratios[name] < 0 for name in SPLITS) or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("train/val/test split ratios must be nonnegative and sum to 1")
    if config["samples_per_class"] < 1 or any(count < 1 for count in _split_counts(config["samples_per_class"], ratios).values()):
        raise ValueError("samples_per_class must produce at least one sample in every split")
    strategy = config.get("split_strategy", "random")
    if strategy not in {"random", "blocked"}:
        raise ValueError("split_strategy must be 'random' or 'blocked'")
    gap = config.get("blocked_g_gap", 0.0)
    if gap < 0:
        raise ValueError("blocked_g_gap must be nonnegative")
    if strategy == "blocked" and any(2 * gap >= high - low for low, high in (ferro, para)):
        raise ValueError("blocked_g_gap leaves no room for blocked split intervals")


def _blocked_intervals(low: float, high: float, ratios: dict[str, float], gap: float) -> dict[str, tuple[float, float]]:
    usable = high - low - 2 * gap
    train_end = low + usable * ratios["train"]
    val_start = train_end + gap
    val_end = val_start + usable * ratios["val"]
    return {
        "train": (low, train_end),
        "val": (val_start, val_end),
        "test": (val_end + gap, high),
    }


def _records(config: dict) -> list[dict]:
    _validate_config(config)
    rng = np.random.default_rng(config["dataset_seed"])
    strategy = config.get("split_strategy", "random")
    ratios = config["split_ratios"]
    counts = _split_counts(config["samples_per_class"], ratios)
    records: list[dict] = []

    for label, region_name in ((0, "ferromagnetic"), (1, "paramagnetic")):
        low, high = config["phase_regions"][region_name]
        if strategy == "blocked":
            intervals = _blocked_intervals(low, high, ratios, config.get("blocked_g_gap", 0.0))
            samples = [
                (float(g), split)
                for split in SPLITS
                for g in rng.uniform(*intervals[split], counts[split])
            ]
            for index, (g, split) in enumerate(samples):
                records.append({
                    "parameter_id": f"class-{label}-{index:05d}",
                    "g": g,
                    "label": label,
                    "split": split,
                })
        else:
            for index, g in enumerate(rng.uniform(low, high, config["samples_per_class"])):
                records.append({
                    "parameter_id": f"class-{label}-{index:05d}",
                    "g": float(g),
                    "label": label,
                })

    if strategy == "random":
        # Preserve the original class-stratified random assignment for smoke-config compatibility.
        split_rng = np.random.default_rng(config["split_seed"])
        for label in (0, 1):
            indices = np.array([i for i, record in enumerate(records) if record["label"] == label])
            split_rng.shuffle(indices)
            for position, index in enumerate(indices):
                records[index]["split"] = (
                    "train" if position < counts["train"] else "val" if position < counts["train"] + counts["val"] else "test"
                )
    return records


def generate_dataset(config: dict, output: str | Path) -> Path:
    """Generate states and a leakage-auditable manifest from a resolved config."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = _records(config)  # split assignment occurs before diagonalization
    mx_operator, mz2_operator = _observable_operators(config["n_qubits"])

    states, energies, mx_values, mz2_values = [], [], [], []
    for record in records:
        hamiltonian = tfim_hamiltonian(
            config["n_qubits"], config["J"], record["g"], config["boundary"]
        )
        energy, state = ground_state(hamiltonian)
        mx = float(np.vdot(state, mx_operator @ state).real)
        mz2 = float(np.vdot(state, mz2_operator @ state).real)
        states.append(state)
        energies.append(energy)
        mx_values.append(mx)
        mz2_values.append(mz2)
        record.update({"energy": energy, "magnetization_x": mx, "magnetization_z2": mz2})

    np.savez_compressed(
        output / "states.npz",
        states=np.asarray(states),
        energies=np.asarray(energies),
        labels=np.asarray([r["label"] for r in records], dtype=np.int8),
        parameter_ids=np.asarray([r["parameter_id"] for r in records]),
        splits=np.asarray([r["split"] for r in records]),
        g=np.asarray([r["g"] for r in records]),
        magnetization_x=np.asarray(mx_values),
        magnetization_z2=np.asarray(mz2_values),
    )
    manifest = {"config": config, "records": records}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def _minimum_cross_split_g_gap(data: np.lib.npyio.NpzFile) -> float:
    minimum = np.inf
    for label in (0, 1):
        values = {split: data["g"][(data["labels"] == label) & (data["splits"] == split)] for split in SPLITS}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            minimum = min(minimum, float(np.min(np.abs(values[left][:, None] - values[right][None, :]))))
    return minimum


def validate_dataset(path: str | Path, tolerance: float = 1e-10) -> dict:
    """Validate numerical accuracy, split integrity, and phase observables."""
    path = Path(path)
    manifest = json.loads((path / "split_manifest.json").read_text())
    data = np.load(path / "states.npz")
    config = manifest["config"]
    errors: list[str] = []

    expected_length = len(manifest["records"])
    if any(len(data[name]) != expected_length for name in data.files):
        errors.append("dataset arrays and manifest have different lengths")
    ids_by_split = {
        split: {r["parameter_id"] for r in manifest["records"] if r["split"] == split}
        for split in SPLITS
    }
    if any(ids_by_split[a] & ids_by_split[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        errors.append("split parameter IDs overlap")

    max_norm_error = 0.0
    max_residual = 0.0
    max_observable_error = 0.0
    computed_mx_values, computed_mz2_values = [], []
    observables_stored = {"magnetization_x", "magnetization_z2"}.issubset(data.files)
    mx_operator, mz2_operator = _observable_operators(config["n_qubits"])
    for index, (state, energy, g) in enumerate(zip(data["states"], data["energies"], data["g"], strict=True)):
        max_norm_error = max(max_norm_error, abs(float(np.vdot(state, state).real) - 1.0))
        hamiltonian = tfim_hamiltonian(config["n_qubits"], config["J"], float(g), config["boundary"])
        max_residual = max(max_residual, float(np.linalg.norm(hamiltonian @ state - energy * state)))
        computed_mx = float(np.vdot(state, mx_operator @ state).real)
        computed_mz2 = float(np.vdot(state, mz2_operator @ state).real)
        computed_mx_values.append(computed_mx)
        computed_mz2_values.append(computed_mz2)
        if observables_stored:
            max_observable_error = max(
                max_observable_error,
                abs(computed_mx - data["magnetization_x"][index]),
                abs(computed_mz2 - data["magnetization_z2"][index]),
            )
    if max_norm_error > tolerance:
        errors.append(f"normalization error {max_norm_error} exceeds {tolerance}")
    if max_residual > tolerance:
        errors.append(f"eigenpair residual {max_residual} exceeds {tolerance}")
    if max_observable_error > tolerance:
        errors.append(f"stored observable error {max_observable_error} exceeds {tolerance}")

    observable_means = {}
    mx_values = data["magnetization_x"] if observables_stored else np.asarray(computed_mx_values)
    mz2_values = data["magnetization_z2"] if observables_stored else np.asarray(computed_mz2_values)
    for label, name in ((0, "ferromagnetic"), (1, "paramagnetic")):
        mask = data["labels"] == label
        observable_means[name] = {
            "magnetization_x": float(np.mean(mx_values[mask])),
            "magnetization_z2": float(np.mean(mz2_values[mask])),
        }
    ferro = observable_means["ferromagnetic"]
    para = observable_means["paramagnetic"]
    if not ferro["magnetization_z2"] > para["magnetization_z2"]:
        errors.append("classes are not separated by mean longitudinal magnetization squared")
    if not para["magnetization_x"] > ferro["magnetization_x"]:
        errors.append("classes are not separated by mean transverse magnetization")

    minimum_g_gap = _minimum_cross_split_g_gap(data)
    required_gap = config.get("blocked_g_gap", 0.0) if config.get("split_strategy", "random") == "blocked" else 0.0
    if minimum_g_gap + tolerance < required_gap:
        errors.append(f"cross-split g gap {minimum_g_gap} is smaller than required {required_gap}")

    report = {
        "valid": not errors,
        "errors": errors,
        "samples": len(data["states"]),
        "split_strategy": config.get("split_strategy", "random"),
        "split_counts": {split: len(ids) for split, ids in ids_by_split.items()},
        "max_norm_error": max_norm_error,
        "max_eigenpair_residual": max_residual,
        "max_observable_error": max_observable_error,
        "observables_stored": observables_stored,
        "minimum_cross_split_g_gap": minimum_g_gap,
        "observable_means": observable_means,
    }
    (path / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def write_checksums(path: str | Path) -> dict[str, str]:
    """Write SHA-256 digests for all benchmark dataset artifacts."""
    path = Path(path)
    checksums = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in CHECKSUM_FILES}
    (path / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    return checksums


def verify_checksums(path: str | Path) -> dict[str, bool]:
    """Verify every required artifact against checksums.json."""
    path = Path(path)
    expected = json.loads((path / "checksums.json").read_text())
    return {
        name: name in expected and hashlib.sha256((path / name).read_bytes()).hexdigest() == expected[name]
        for name in CHECKSUM_FILES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate an exact TFIM dataset")
    parser.add_argument("--config", default="configs/dataset/tfim_4q.yaml")
    parser.add_argument("--output", default="data/tfim_4q")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    path = generate_dataset(config, args.output)
    report = validate_dataset(path, config.get("numerical_tolerance", 1e-10))
    write_checksums(path)
    checksum_validation = verify_checksums(path)
    report["checksums_valid"] = checksum_validation
    print(json.dumps(report, indent=2))
    if not report["valid"] or not all(checksum_validation.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

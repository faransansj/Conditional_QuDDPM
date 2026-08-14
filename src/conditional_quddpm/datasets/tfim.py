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


def _validate_config(config: dict) -> None:
    ferro = config["phase_regions"]["ferromagnetic"]
    para = config["phase_regions"]["paramagnetic"]
    if not (ferro[0] < ferro[1] < para[0] < para[1]):
        raise ValueError("phase regions must be ordered, non-overlapping intervals")
    ratios = config["split_ratios"]
    if any(ratios[name] < 0 for name in ("train", "val", "test")) or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("train/val/test split ratios must be nonnegative and sum to 1")
    if config["samples_per_class"] < 1:
        raise ValueError("samples_per_class must be positive")


def _records(config: dict) -> list[dict]:
    _validate_config(config)
    rng = np.random.default_rng(config["dataset_seed"])
    records: list[dict] = []
    for label, region_name in ((0, "ferromagnetic"), (1, "paramagnetic")):
        low, high = config["phase_regions"][region_name]
        for index, g in enumerate(rng.uniform(low, high, config["samples_per_class"])):
            records.append({
                "parameter_id": f"class-{label}-{index:05d}",
                "g": float(g),
                "label": label,
            })

    split_rng = np.random.default_rng(config["split_seed"])
    ratios = config["split_ratios"]
    for label in (0, 1):
        indices = np.array([i for i, record in enumerate(records) if record["label"] == label])
        split_rng.shuffle(indices)
        n_train = int(len(indices) * ratios["train"])
        n_val = int(len(indices) * ratios["val"])
        for position, index in enumerate(indices):
            records[index]["split"] = (
                "train" if position < n_train else "val" if position < n_train + n_val else "test"
            )
    return records


def generate_dataset(config: dict, output: str | Path) -> Path:
    """Generate states and a leakage-auditable manifest from a resolved config."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = _records(config)  # split assignment occurs before diagonalization

    states, energies = [], []
    for record in records:
        hamiltonian = tfim_hamiltonian(
            config["n_qubits"], config["J"], record["g"], config["boundary"]
        )
        energy, state = ground_state(hamiltonian)
        energies.append(energy)
        states.append(state)
        record["energy"] = energy

    np.savez_compressed(
        output / "states.npz",
        states=np.asarray(states),
        energies=np.asarray(energies),
        labels=np.asarray([r["label"] for r in records], dtype=np.int8),
        parameter_ids=np.asarray([r["parameter_id"] for r in records]),
        splits=np.asarray([r["split"] for r in records]),
        g=np.asarray([r["g"] for r in records]),
    )
    manifest = {"config": config, "records": records}
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output / "checksums.json").write_text(json.dumps({"split_manifest.json": checksum}, indent=2) + "\n")
    return output


def validate_dataset(path: str | Path, tolerance: float = 1e-10) -> dict:
    """Validate normalization, residuals, and disjoint split identifiers."""
    path = Path(path)
    manifest = json.loads((path / "split_manifest.json").read_text())
    data = np.load(path / "states.npz")
    config = manifest["config"]
    errors: list[str] = []

    ids_by_split = {
        split: {r["parameter_id"] for r in manifest["records"] if r["split"] == split}
        for split in ("train", "val", "test")
    }
    if any(ids_by_split[a] & ids_by_split[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        errors.append("split parameter IDs overlap")

    max_norm_error = 0.0
    max_residual = 0.0
    for state, energy, g in zip(data["states"], data["energies"], data["g"], strict=True):
        max_norm_error = max(max_norm_error, abs(float(np.vdot(state, state).real) - 1.0))
        hamiltonian = tfim_hamiltonian(config["n_qubits"], config["J"], float(g), config["boundary"])
        max_residual = max(max_residual, float(np.linalg.norm(hamiltonian @ state - energy * state)))
    if max_norm_error > tolerance:
        errors.append(f"normalization error {max_norm_error} exceeds {tolerance}")
    if max_residual > tolerance:
        errors.append(f"eigenpair residual {max_residual} exceeds {tolerance}")

    report = {
        "valid": not errors,
        "errors": errors,
        "samples": len(data["states"]),
        "split_counts": {split: len(ids) for split, ids in ids_by_split.items()},
        "max_norm_error": max_norm_error,
        "max_eigenpair_residual": max_residual,
    }
    (path / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate an exact TFIM dataset")
    parser.add_argument("--config", default="configs/dataset/tfim_4q.yaml")
    parser.add_argument("--output", default="data/tfim_4q")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    path = generate_dataset(config, args.output)
    report = validate_dataset(path, config.get("numerical_tolerance", 1e-10))
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

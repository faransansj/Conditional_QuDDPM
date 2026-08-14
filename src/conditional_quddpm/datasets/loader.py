"""Leakage-safe loading and nested subset selection for generated TFIM datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tfim import verify_checksums


@dataclass(frozen=True)
class QuantumSplit:
    states: np.ndarray
    labels: np.ndarray
    parameter_ids: np.ndarray


@dataclass(frozen=True)
class TFIMDataset:
    train: QuantumSplit
    val: QuantumSplit
    test: QuantumSplit
    manifest: dict


def load_tfim_dataset(path: str | Path, verify: bool = True) -> TFIMDataset:
    """Load arrays in manifest order; the manifest alone defines split membership."""
    path = Path(path)
    if verify and not all(verify_checksums(path).values()):
        raise ValueError(f"checksum verification failed for {path}")

    manifest = json.loads((path / "split_manifest.json").read_text())
    data = np.load(path / "states.npz")
    array_ids = data["parameter_ids"].tolist()
    manifest_ids = [record["parameter_id"] for record in manifest["records"]]
    if len(array_ids) != len(set(array_ids)) or set(array_ids) != set(manifest_ids):
        raise ValueError("states.npz IDs do not match unique manifest IDs")

    array_index = {parameter_id: index for index, parameter_id in enumerate(array_ids)}
    grouped: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for record in manifest["records"]:
        split = record["split"]
        if split not in grouped:
            raise ValueError(f"unknown manifest split: {split}")
        index = array_index[record["parameter_id"]]
        if int(data["labels"][index]) != int(record["label"]):
            raise ValueError(f"label mismatch for {record['parameter_id']}")
        grouped[split].append(index)
        labels[split].append(int(record["label"]))
        ids[split].append(record["parameter_id"])

    if any(set(ids[a]) & set(ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("manifest split IDs overlap")

    def build(split: str) -> QuantumSplit:
        return QuantumSplit(
            states=np.asarray(data["states"][grouped[split]], dtype=np.complex128),
            labels=np.asarray(labels[split], dtype=np.int8),
            parameter_ids=np.asarray(ids[split]),
        )

    return TFIMDataset(train=build("train"), val=build("val"), test=build("test"), manifest=manifest)


def nested_train_subsets(
    train: QuantumSplit, sizes_per_class: list[int], seed: int
) -> dict[int, QuantumSplit]:
    """Return class-balanced prefix subsets from one seeded ordering."""
    sizes = sorted(set(sizes_per_class))
    if not sizes or sizes[0] < 1:
        raise ValueError("subset sizes must be positive")
    rng = np.random.default_rng(seed)
    ordered: dict[int, np.ndarray] = {}
    for label in (0, 1):
        indices = np.flatnonzero(train.labels == label)
        rng.shuffle(indices)
        if sizes[-1] > len(indices):
            raise ValueError(f"requested {sizes[-1]} states for class {label}, only {len(indices)} available")
        ordered[label] = indices

    subsets = {}
    for size in sizes:
        indices = np.concatenate([ordered[0][:size], ordered[1][:size]])
        indices.sort()
        subsets[size] = QuantumSplit(
            states=train.states[indices],
            labels=train.labels[indices],
            parameter_ids=train.parameter_ids[indices],
        )
    return subsets

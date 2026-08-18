"""Leakage-safe, state-local TFIM unitary augmentation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from conditional_quddpm.datasets.tfim import X, Z, _operator, tfim_hamiltonian, tfim_observables


@dataclass(frozen=True)
class AcceptanceGate:
    """Train-only empirical physics envelope for one class."""

    mx_range: tuple[float, float]
    mz2_range: tuple[float, float]
    max_energy_excess: float
    norm_tolerance: float = 1e-10


def tfim_components(n_qubits: int, J: float = 1.0, boundary: str = "open") -> dict[str, np.ndarray]:
    """Return the signed interaction and field terms used by the repository TFIM."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")
    bonds = [(i, i + 1) for i in range(n_qubits - 1)]
    if boundary == "periodic" and n_qubits > 2:
        bonds.append((n_qubits - 1, 0))
    interaction = -J * sum(_operator(n_qubits, {left: Z, right: Z}) for left, right in bonds)
    field = -sum(_operator(n_qubits, {qubit: X}) for qubit in range(n_qubits))
    return {"interaction": interaction, "field": field}


def normalize_generator(generator: np.ndarray) -> np.ndarray:
    """Normalize a Hermitian generator by spectral norm so epsilon is dimensionless."""
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be square")
    if not np.allclose(generator, generator.conj().T, atol=1e-12):
        raise ValueError("generator must be Hermitian")
    scale = float(np.max(np.abs(np.linalg.eigvalsh(generator))))
    if scale == 0:
        raise ValueError("generator must be nonzero")
    return generator / scale


def _unitary_action(state: np.ndarray, generator: np.ndarray, epsilon: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return vectors @ (np.exp(-1j * epsilon * values) * (vectors.conj().T @ state))


def augment_state(
    state: np.ndarray,
    metadata: dict,
    generator: np.ndarray,
    *,
    operator: str,
    epsilon: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Apply one normalized component unitary and return the required provenance."""
    if metadata.get("source_split") != "train":
        raise ValueError("augmentation sources must belong to the train split")
    required = {"source_state_id", "source_class"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"missing source metadata: {sorted(missing)}")
    state = np.asarray(state, dtype=np.complex128)
    dimension = generator.shape[0]
    if state.shape != (dimension,):
        raise ValueError("state and generator dimensions do not match")
    direction = 1 if epsilon == 0 else int(np.random.default_rng(seed).choice((-1, 1)))
    strength = float(direction * epsilon)
    synthetic = state.copy() if epsilon == 0 else _unitary_action(state, normalize_generator(generator), strength)
    token = f"{metadata.get('source_dataset', '')}|{metadata['source_state_id']}|{operator}|{strength:.17g}|{seed}"
    provenance = {
        "synthetic_id": "syn-" + hashlib.sha256(token.encode()).hexdigest()[:16],
        "source_state_id": str(metadata["source_state_id"]),
        "source_dataset": metadata.get("source_dataset"),
        "source_split": "train",
        "source_class": int(metadata["source_class"]),
        "source_g": None if metadata.get("source_g") is None else float(metadata["source_g"]),
        "augmentation_method": "state_local_physics",
        "generator": "spectral_norm_normalized",
        "operator": operator,
        "perturbation_strength": strength,
        "random_seed": int(seed),
    }
    return synthetic, provenance


def state_diagnostics(
    source: np.ndarray,
    synthetic: np.ndarray,
    train_states: np.ndarray,
    train_labels: np.ndarray,
    source_class: int,
    source_g: float,
    n_qubits: int,
    J: float,
    boundary: str,
) -> dict[str, float]:
    """Compute validity, fidelity, energy, and repository TFIM observables."""
    fidelities = np.abs(train_states.conj() @ synthetic) ** 2
    same = train_labels == source_class
    hamiltonian = tfim_hamiltonian(n_qubits, J, source_g, boundary)
    source_energy = float(np.vdot(source, hamiltonian @ source).real)
    synthetic_energy = float(np.vdot(synthetic, hamiltonian @ synthetic).real)
    source_mx, source_mz2 = tfim_observables(source, n_qubits)
    mx, mz2 = tfim_observables(synthetic, n_qubits)
    return {
        "norm": float(np.vdot(synthetic, synthetic).real),
        "purity": float(np.vdot(synthetic, synthetic).real ** 2),
        "source_fidelity": float(abs(np.vdot(source, synthetic)) ** 2),
        "nearest_train_fidelity": float(np.max(fidelities)),
        "nearest_same_class_train_fidelity": float(np.max(fidelities[same])),
        "nearest_other_class_train_fidelity": float(np.max(fidelities[~same])),
        "energy": synthetic_energy,
        "source_energy": source_energy,
        "energy_drift": synthetic_energy - source_energy,
        "energy_drift_absolute": abs(synthetic_energy - source_energy),
        "magnetization_x": mx,
        "magnetization_z2": mz2,
        "magnetization_x_drift": mx - source_mx,
        "magnetization_z2_drift": mz2 - source_mz2,
    }


def fit_acceptance_gate(
    states: np.ndarray,
    g_values: np.ndarray,
    *,
    n_qubits: int,
    J: float,
    boundary: str,
) -> AcceptanceGate:
    """Fit empirical observable ranges and within-class energy variation."""
    if len(states) < 2:
        raise ValueError("acceptance gate needs at least two train states from a class")
    observables = np.asarray([tfim_observables(state, n_qubits) for state in states])
    excesses = []
    for index, (state, g) in enumerate(zip(states, g_values, strict=True)):
        hamiltonian = tfim_hamiltonian(n_qubits, J, float(g), boundary)
        ground_energy = float(np.vdot(state, hamiltonian @ state).real)
        excesses.extend(
            float(np.vdot(peer, hamiltonian @ peer).real) - ground_energy
            for peer_index, peer in enumerate(states)
            if peer_index != index
        )
    return AcceptanceGate(
        mx_range=(float(observables[:, 0].min()), float(observables[:, 0].max())),
        mz2_range=(float(observables[:, 1].min()), float(observables[:, 1].max())),
        max_energy_excess=float(max(excesses)),
    )


def accepted(diagnostics: dict[str, float], gate: AcceptanceGate) -> bool:
    """Apply only train-fitted physics limits; val/test statistics never enter."""
    return bool(
        abs(diagnostics["norm"] - 1.0) <= gate.norm_tolerance
        and gate.mx_range[0] <= diagnostics["magnetization_x"] <= gate.mx_range[1]
        and gate.mz2_range[0] <= diagnostics["magnetization_z2"] <= gate.mz2_range[1]
        and diagnostics["energy_drift"] <= gate.max_energy_excess + 1e-12
    )

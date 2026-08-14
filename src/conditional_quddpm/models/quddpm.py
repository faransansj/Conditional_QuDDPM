"""Minimal pure-state QuDDPM/C-QuDDPM reference implementation.

This is an independent NumPy implementation of the paper-defined method: a
random-unitary forward process and an ancilla-assisted, measured, non-unitary
reverse process trained one diffusion step at a time with fidelity-kernel MMD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)


def rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    return np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * pauli


def haar_states(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(count, 2)) + 1j * rng.normal(size=(count, 2))
    return states / np.linalg.norm(states, axis=1, keepdims=True)


def pole_clusters(count: int, labels: list[int], seed: int, spread: float = 0.22) -> dict[int, np.ndarray]:
    """Create reproducible one-qubit clusters around the north/south poles."""
    rng = np.random.default_rng(seed)
    clusters = {}
    for label in labels:
        center = 0.0 if label == 0 else np.pi
        theta = np.clip(rng.normal(center, spread, count), 0.0, np.pi)
        phi = rng.uniform(0.0, 2 * np.pi, count)
        clusters[label] = np.column_stack([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2),
        ])
    return clusters


def _random_su2(rng: np.random.Generator, strength: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    pauli = axis[0] * X + axis[1] * Y + axis[2] * Z
    return rotation(pauli, rng.uniform(-np.pi, np.pi) * strength)


def forward_diffusion(
    targets: dict[int, np.ndarray], steps: int, seed: int
) -> dict[int, list[np.ndarray]]:
    """Return target-to-noise ensembles at every random-unitary forward step."""
    rng = np.random.default_rng(seed)
    trajectories = {label: [states.copy()] for label, states in targets.items()}
    for step in range(steps):
        strength = (step + 1) / steps
        for label, states in targets.items():
            previous = trajectories[label][-1]
            trajectories[label].append(np.asarray([_random_su2(rng, strength) @ state for state in previous]))
    return trajectories


def fidelity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left.conj() @ right.T) ** 2


def fidelity_mmd(left: np.ndarray, right: np.ndarray) -> float:
    """Biased, nonnegative MMD² using state fidelity as the kernel."""
    value = fidelity_matrix(left, left).mean() + fidelity_matrix(right, right).mean()
    value -= 2 * fidelity_matrix(left, right).mean()
    return float(max(value, 0.0))


def _two_qubit_rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    product = np.kron(pauli, pauli)
    return np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * product


def _reverse_unitary(parameters: np.ndarray) -> np.ndarray:
    """Expressive RX/RY plus XX/YY/ZZ ancilla-data denoising ansatz."""
    unitary = np.eye(4, dtype=np.complex128)
    for layer in parameters:
        local = np.kron(rotation(Y, layer[1]) @ rotation(X, layer[0]), rotation(Y, layer[3]) @ rotation(X, layer[2]))
        entangler = _two_qubit_rotation(Z, layer[6]) @ _two_qubit_rotation(Y, layer[5]) @ _two_qubit_rotation(X, layer[4])
        unitary = CZ @ entangler @ local @ unitary
    return unitary


def reverse_step(
    states: np.ndarray,
    parameters: np.ndarray,
    condition_angle: float,
    measurement_uniforms: np.ndarray,
) -> np.ndarray:
    """Apply one conditioned reverse step and sample the ancilla measurement."""
    if len(states) != len(measurement_uniforms):
        raise ValueError("one measurement uniform is required per state")
    ancilla = rotation(X, condition_angle) @ np.array([1.0, 0.0], dtype=np.complex128)
    unitary = _reverse_unitary(parameters)
    outputs = []
    for state, uniform in zip(states, measurement_uniforms, strict=True):
        joint = (unitary @ np.kron(ancilla, state)).reshape(2, 2)
        probabilities = np.sum(np.abs(joint) ** 2, axis=1)
        outcome = int(uniform >= probabilities[0])
        output = joint[outcome]
        outputs.append(output / np.linalg.norm(output))
    return np.asarray(outputs)


def condition_angles(labels: list[int]) -> dict[int, float]:
    if len(labels) == 1:
        return {labels[0]: 0.0}
    return {label: 2 * np.pi * index / len(labels) for index, label in enumerate(sorted(labels))}


@dataclass(frozen=True)
class QuDDPMTrainingResult:
    parameters: np.ndarray
    histories: list[list[dict[str, float]]]
    conditioning: dict[int, float]


def save_quddpm_checkpoint(path: str | Path, result: QuDDPMTrainingResult) -> None:
    labels = np.asarray(sorted(result.conditioning), dtype=np.int64)
    np.savez_compressed(
        path,
        parameters=result.parameters,
        labels=labels,
        condition_angles=np.asarray([result.conditioning[int(label)] for label in labels]),
    )


def load_quddpm_checkpoint(path: str | Path) -> QuDDPMTrainingResult:
    checkpoint = np.load(path)
    conditioning = {
        int(label): float(angle)
        for label, angle in zip(checkpoint["labels"], checkpoint["condition_angles"], strict=True)
    }
    return QuDDPMTrainingResult(checkpoint["parameters"], [], conditioning)


def train_stepwise_quddpm(
    targets: dict[int, np.ndarray],
    *,
    diffusion_steps: int,
    layers: int,
    samples: int,
    forward_seed: int,
    source_seed: int,
    init_seed: int,
    spsa_seed: int,
    measurement_seed: int,
    training_steps: int,
    learning_rate: float,
    perturbation: float,
) -> tuple[QuDDPMTrainingResult, dict[int, list[np.ndarray]]]:
    """Train reverse maps from T→0, sharing each step's parameters across labels."""
    labels = sorted(targets)
    if any(len(targets[label]) != samples for label in labels):
        raise ValueError("every class must contain exactly samples states")
    forward = forward_diffusion(targets, diffusion_steps, forward_seed)
    angles = condition_angles(labels)
    init_rng = np.random.default_rng(init_seed)
    spsa_rng = np.random.default_rng(spsa_seed)
    measurement_rng = np.random.default_rng(measurement_seed)
    parameters = init_rng.normal(0.0, 0.15, size=(diffusion_steps, layers, 7))
    source = {label: haar_states(samples, source_seed + label) for label in labels}
    current = {label: source[label].copy() for label in labels}
    uniforms = {
        (step, label): measurement_rng.random(samples)
        for step in range(diffusion_steps)
        for label in labels
    }
    histories: list[list[dict[str, float]]] = [[] for _ in range(diffusion_steps)]

    for step in range(diffusion_steps - 1, -1, -1):
        target_at_step = {label: forward[label][step] for label in labels}

        def loss(candidate: np.ndarray) -> float:
            losses = [
                fidelity_mmd(
                    reverse_step(current[label], candidate, angles[label], uniforms[(step, label)]),
                    target_at_step[label],
                )
                for label in labels
            ]
            return float(np.mean(losses))

        step_parameters = parameters[step].copy()
        for iteration in range(training_steps + 1):
            value = loss(step_parameters)
            histories[step].append({"iteration": iteration, "loss": value})
            if iteration == training_steps:
                break
            delta = spsa_rng.choice((-1.0, 1.0), size=step_parameters.shape)
            scale = perturbation / (iteration + 1) ** 0.101
            rate = learning_rate / (iteration + 1) ** 0.602
            gradient = (loss(step_parameters + scale * delta) - loss(step_parameters - scale * delta)) / (2 * scale) * delta
            step_parameters -= rate * gradient
        parameters[step] = step_parameters
        current = {
            label: reverse_step(current[label], step_parameters, angles[label], uniforms[(step, label)])
            for label in labels
        }

    return QuDDPMTrainingResult(parameters, histories, angles), forward


def generate_quddpm(
    result: QuDDPMTrainingResult,
    labels: list[int],
    count: int,
    source_seed: int,
    measurement_seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(measurement_seed)
    generated = {}
    for label in labels:
        states = haar_states(count, source_seed + label)
        for step in range(len(result.parameters) - 1, -1, -1):
            states = reverse_step(states, result.parameters[step], result.conditioning[label], rng.random(count))
        generated[label] = states
    return generated


def bloch_z(states: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("bi,ij,bj->b", states.conj(), Z, states))

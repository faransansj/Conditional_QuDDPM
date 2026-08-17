"""Minimal pure-state QuDDPM/C-QuDDPM reference implementation.

Independent NumPy implementation of a random-unitary forward process and an
ancilla-assisted, measured reverse process trained one diffusion step at a
time with fidelity-kernel MMD. Supports one or more data qubits.
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


def _two_qubit_rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    product = np.kron(pauli, pauli)
    return np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * product


def _qubit_count(state_dimension: int) -> int:
    n_qubits = int(np.log2(state_dimension))
    if 2**n_qubits != state_dimension:
        raise ValueError("state dimension must be a power of two")
    return n_qubits


def _apply_gate(state: np.ndarray, gate: np.ndarray, wires: tuple[int, ...], n_qubits: int) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    remaining = tuple(qubit for qubit in range(n_qubits) if qubit not in wires)
    permutation = wires + remaining
    inverse = np.argsort(permutation)
    transformed = np.transpose(tensor, permutation).reshape(2 ** len(wires), -1)
    transformed = gate @ transformed
    return np.transpose(transformed.reshape((2,) * n_qubits), inverse).reshape(-1)


def haar_states(count: int, seed: int, n_qubits: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(count, 2**n_qubits)) + 1j * rng.normal(size=(count, 2**n_qubits))
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
    return rotation(axis[0] * X + axis[1] * Y + axis[2] * Z, rng.uniform(-np.pi, np.pi) * strength)


def _scramble_state(state: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    n_qubits = _qubit_count(len(state))
    output = state
    for qubit in range(n_qubits):
        output = _apply_gate(output, _random_su2(rng, strength), (qubit,), n_qubits)
    for qubit in range(n_qubits - 1):
        angle = rng.uniform(-np.pi / 2, np.pi / 2) * strength
        output = _apply_gate(output, _two_qubit_rotation(Z, angle), (qubit, qubit + 1), n_qubits)
    return output


def forward_diffusion(
    targets: dict[int, np.ndarray], steps: int, seed: int
) -> dict[int, list[np.ndarray]]:
    """Return target-to-noise ensembles at every random-unitary forward step."""
    rng = np.random.default_rng(seed)
    trajectories = {label: [states.copy()] for label, states in targets.items()}
    for step in range(steps):
        strength = (step + 1) / steps
        for label in targets:
            trajectories[label].append(
                np.asarray([_scramble_state(state, rng, strength) for state in trajectories[label][-1]])
            )
    return trajectories


def fidelity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left.conj() @ right.T) ** 2


def fidelity_mmd(left: np.ndarray, right: np.ndarray) -> float:
    """Biased, nonnegative MMD² using state fidelity as the kernel."""
    value = fidelity_matrix(left, left).mean() + fidelity_matrix(right, right).mean()
    value -= 2 * fidelity_matrix(left, right).mean()
    return float(max(value, 0.0))


def reverse_parameter_count(n_data: int, n_ancilla: int = 1) -> int:
    n_total = n_data + n_ancilla
    return 2 * n_total + 3 * (n_total - 1)


def _apply_reverse_circuit(
    joint: np.ndarray, parameters: np.ndarray, n_data: int, n_ancilla: int = 1
) -> np.ndarray:
    n_total = n_data + n_ancilla
    expected = reverse_parameter_count(n_data, n_ancilla)
    if parameters.shape[1] != expected:
        raise ValueError(f"expected {expected} parameters per layer, got {parameters.shape[1]}")
    output = joint
    for layer in parameters:
        offset = 0
        for qubit in range(n_total):
            output = _apply_gate(output, rotation(X, layer[offset]), (qubit,), n_total)
            output = _apply_gate(output, rotation(Y, layer[offset + 1]), (qubit,), n_total)
            offset += 2
        for left in range(n_total - 1):
            wires = (left, left + 1)
            for pauli in (X, Y, Z):
                output = _apply_gate(output, _two_qubit_rotation(pauli, layer[offset]), wires, n_total)
                offset += 1
            output = _apply_gate(output, CZ, wires, n_total)
    return output


def _reverse_unitary(parameters: np.ndarray) -> np.ndarray:
    """Materialize the reverse unitary for validation tests."""
    width = parameters.shape[1]
    n_total = (width + 3) // 5
    if 5 * n_total - 3 != width or n_total < 2:
        raise ValueError("invalid reverse parameter width")
    n_data = n_total - 1
    basis = np.eye(2**n_total, dtype=np.complex128)
    return np.column_stack([_apply_reverse_circuit(column, parameters, n_data) for column in basis])


def reverse_step(
    states: np.ndarray,
    parameters: np.ndarray,
    condition_angle: float,
    measurement_uniforms: np.ndarray,
) -> np.ndarray:
    """Apply one conditioned reverse step and sample the ancilla measurement."""
    if len(states) != len(measurement_uniforms):
        raise ValueError("one measurement uniform is required per state")
    n_data = _qubit_count(states.shape[1])
    width = parameters.shape[1]
    n_total = (width + 3) // 5
    n_ancilla = n_total - n_data
    if n_ancilla < 1 or reverse_parameter_count(n_data, n_ancilla) != width:
        raise ValueError("reverse parameter width is incompatible with the data dimension")
    ancilla_qubit = rotation(X, condition_angle) @ np.array([1.0, 0.0], dtype=np.complex128)
    ancilla = ancilla_qubit
    for _ in range(n_ancilla - 1):
        ancilla = np.kron(ancilla, ancilla_qubit)
    outputs = []
    for state, uniform in zip(states, measurement_uniforms, strict=True):
        joint = _apply_reverse_circuit(
            np.kron(ancilla, state), parameters, n_data, n_ancilla
        ).reshape(2**n_ancilla, -1)
        probabilities = np.sum(np.abs(joint) ** 2, axis=1)
        outcome = int(np.searchsorted(np.cumsum(probabilities), uniform, side="right"))
        outcome = min(outcome, len(probabilities) - 1)
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
    n_data: int
    n_ancilla: int = 1


def save_quddpm_checkpoint(path: str | Path, result: QuDDPMTrainingResult) -> None:
    labels = np.asarray(sorted(result.conditioning), dtype=np.int64)
    np.savez_compressed(
        path,
        parameters=result.parameters,
        labels=labels,
        condition_angles=np.asarray([result.conditioning[int(label)] for label in labels]),
        n_data=np.asarray(result.n_data),
        n_ancilla=np.asarray(result.n_ancilla),
    )


def load_quddpm_checkpoint(path: str | Path) -> QuDDPMTrainingResult:
    checkpoint = np.load(path)
    conditioning = {
        int(label): float(angle)
        for label, angle in zip(checkpoint["labels"], checkpoint["condition_angles"], strict=True)
    }
    n_data = int(checkpoint["n_data"]) if "n_data" in checkpoint else (checkpoint["parameters"].shape[-1] + 3) // 5 - 1
    n_ancilla = int(checkpoint["n_ancilla"]) if "n_ancilla" in checkpoint else 1
    return QuDDPMTrainingResult(checkpoint["parameters"], [], conditioning, n_data, n_ancilla)


def train_single_reverse_steps(
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
    n_ancilla: int = 1,
    optimizer: str = "spsa",
    source_mode: str = "haar",
    measurement_repeats: int = 1,
    forward_trajectories: dict[int, list[np.ndarray]] | None = None,
) -> tuple[QuDDPMTrainingResult, dict[int, list[np.ndarray]], list[dict]]:
    """Fit each q_t -> q_(t-1) map independently from a Haar source.

    Unlike the chain trainer, each step starts from Haar and targets the
    adjacent forward ensemble; this isolates ansatz/optimizer failures.
    """
    labels = sorted(targets)
    n_data = _qubit_count(targets[labels[0]].shape[1])
    forward = forward_trajectories if forward_trajectories is not None else forward_diffusion(targets, diffusion_steps, forward_seed)
    angles = condition_angles(labels)
    rng = np.random.default_rng(init_seed)
    spsa_rng = np.random.default_rng(spsa_seed)
    measurement_rng = np.random.default_rng(measurement_seed)
    parameters = rng.normal(0, 0.15, (diffusion_steps, layers, reverse_parameter_count(n_data, n_ancilla)))
    histories, diagnostics = [[] for _ in range(diffusion_steps)], []
    from scipy.optimize import minimize
    for step in range(diffusion_steps):
        if source_mode == "haar":
            source = {label: haar_states(samples, source_seed + 100 * step + label, n_data) for label in labels}
        elif source_mode == "teacher_forced":
            source = {label: forward[label][step + 1][:samples].copy() for label in labels}
        else:
            raise ValueError(f"unknown source_mode: {source_mode}")
        uniforms = {
            label: [measurement_rng.random(samples) for _ in range(measurement_repeats)]
            for label in labels
        }
        target = {label: forward[label][step] for label in labels}
        def loss(p):
            return float(np.mean([
                fidelity_mmd(reverse_step(source[label], p, angles[label], repeat), target[label])
                for label in labels for repeat in uniforms[label]
            ]))
        p = parameters[step].copy()
        initial_parameters = p.copy()
        initial = loss(p)
        if optimizer == "lbfgs":
            shape = p.shape
            result = minimize(lambda flat: loss(flat.reshape(shape)), p.ravel(), method="L-BFGS-B", options={"maxiter": training_steps})
            p = result.x.reshape(shape)
            history = [{"iteration": 0, "loss": initial}, {"iteration": training_steps, "loss": loss(p), "parameter_update_norm": float(np.linalg.norm(p - initial_parameters))}]
        else:
            local = np.random.default_rng(spsa_seed + step)
            history = []
            for iteration in range(training_steps + 1):
                value = loss(p)
                record = {"iteration": iteration, "loss": value, "parameter_update_norm": float(np.linalg.norm(p - initial_parameters))}
                if iteration == training_steps:
                    history.append(record)
                    break
                delta = local.choice((-1.0, 1.0), size=p.shape)
                scale = perturbation / (iteration + 1) ** 0.101
                rate = learning_rate / (iteration + 1) ** 0.602
                loss_plus, loss_minus = loss(p + scale * delta), loss(p - scale * delta)
                record.update({"learning_rate": rate, "perturbation": scale, "loss_plus": loss_plus, "loss_minus": loss_minus})
                history.append(record)
                grad = (loss_plus - loss_minus) / (2 * scale) * delta
                p -= rate * grad
        parameters[step] = p; histories[step] = history
        outputs = {label: reverse_step(source[label], p, angles[label], uniforms[label][0]) for label in labels}
        from conditional_quddpm.datasets.tfim import tfim_observables
        diagnostics.append({"step": step + 1, "initial_mmd": initial, "final_mmd": loss(p), "best_mmd": min(item["loss"] for item in history), "parameter_update_norm": float(np.linalg.norm(p - initial_parameters)), "optimizer": optimizer, "source_mode": source_mode,
                            "output_observables": {str(label): np.asarray([tfim_observables(s, n_data) for s in outputs[label]]).mean(axis=0).tolist() for label in labels},
                            "target_observables": {str(label): np.asarray([tfim_observables(s, n_data) for s in target[label]]).mean(axis=0).tolist() for label in labels}})
    return QuDDPMTrainingResult(parameters, histories, angles, n_data, n_ancilla), forward, diagnostics


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
    n_ancilla: int = 1,
) -> tuple[QuDDPMTrainingResult, dict[int, list[np.ndarray]]]:
    """Train reverse maps from T→0, sharing each step's parameters across labels."""
    labels = sorted(targets)
    if any(len(targets[label]) != samples for label in labels):
        raise ValueError("every class must contain exactly samples states")
    dimensions = {targets[label].shape[1] for label in labels}
    if len(dimensions) != 1:
        raise ValueError("all target classes must have the same state dimension")
    n_data = _qubit_count(dimensions.pop())
    forward = forward_diffusion(targets, diffusion_steps, forward_seed)
    angles = condition_angles(labels)
    init_rng = np.random.default_rng(init_seed)
    spsa_rng = np.random.default_rng(spsa_seed)
    measurement_rng = np.random.default_rng(measurement_seed)
    parameters = init_rng.normal(
        0.0, 0.15, size=(diffusion_steps, layers, reverse_parameter_count(n_data, n_ancilla))
    )
    current = {label: haar_states(samples, source_seed + label, n_data) for label in labels}
    uniforms = {
        (step, label): measurement_rng.random(samples)
        for step in range(diffusion_steps)
        for label in labels
    }
    histories: list[list[dict[str, float]]] = [[] for _ in range(diffusion_steps)]

    for step in range(diffusion_steps - 1, -1, -1):
        target_at_step = {label: forward[label][step] for label in labels}

        def loss(candidate: np.ndarray) -> float:
            return float(np.mean([
                fidelity_mmd(
                    reverse_step(current[label], candidate, angles[label], uniforms[(step, label)]),
                    target_at_step[label],
                )
                for label in labels
            ]))

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

    return QuDDPMTrainingResult(parameters, histories, angles, n_data, n_ancilla), forward


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
        states = haar_states(count, source_seed + label, result.n_data)
        for step in range(len(result.parameters) - 1, -1, -1):
            states = reverse_step(states, result.parameters[step], result.conditioning[label], rng.random(count))
        generated[label] = states
    return generated


def bloch_z(states: np.ndarray) -> np.ndarray:
    if states.shape[1] != 2:
        raise ValueError("bloch_z is defined only for one-qubit states")
    return np.real(np.einsum("bi,ij,bj->b", states.conj(), Z, states))

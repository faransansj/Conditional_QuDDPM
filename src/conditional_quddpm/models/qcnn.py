"""A minimal 4-qubit QCNN inspired by the TensorFlow Quantum QCNN tutorial.

Each convolution shares one 15-parameter two-qubit unitary across its edges;
each pooling layer shares one 6-parameter source-to-sink block. The 4→2→1
architecture has 42 trainable parameters and reads out Z on qubit 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)


def _rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * pauli


def _euler(angles: np.ndarray) -> np.ndarray:
    return _rotation(Z, angles[2]) @ _rotation(Y, angles[1]) @ _rotation(X, angles[0])


def _interaction(pauli: np.ndarray, angle: float) -> np.ndarray:
    product = np.kron(pauli, pauli)
    return np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * product


def _apply_gate(state: np.ndarray, gate: np.ndarray, wires: tuple[int, ...], n_qubits: int = 4) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    remaining = tuple(q for q in range(n_qubits) if q not in wires)
    permutation = wires + remaining
    inverse = np.argsort(permutation)
    transformed = np.transpose(tensor, permutation).reshape(2 ** len(wires), -1)
    transformed = gate @ transformed
    return np.transpose(transformed.reshape((2,) * n_qubits), inverse).reshape(-1)


def _convolution(state: np.ndarray, wires: tuple[int, int], parameters: np.ndarray) -> np.ndarray:
    left, right = wires
    state = _apply_gate(state, _euler(parameters[0:3]), (left,))
    state = _apply_gate(state, _euler(parameters[3:6]), (right,))
    for pauli, angle in zip((X, Y, Z), parameters[6:9], strict=True):
        state = _apply_gate(state, _interaction(pauli, angle), wires)
    state = _apply_gate(state, _euler(parameters[9:12]), (left,))
    return _apply_gate(state, _euler(parameters[12:15]), (right,))


def _pool(state: np.ndarray, source: int, sink: int, parameters: np.ndarray) -> np.ndarray:
    source_basis = _euler(parameters[0:3])
    sink_basis = _euler(parameters[3:6])
    state = _apply_gate(state, source_basis, (source,))
    state = _apply_gate(state, sink_basis, (sink,))
    state = _apply_gate(state, CNOT, (source, sink))
    return _apply_gate(state, sink_basis.conj().T, (sink,))


def qcnn_expectation(state: np.ndarray, parameters: np.ndarray) -> float:
    """Return <Z_3> for one normalized 4-qubit statevector."""
    if state.shape != (16,):
        raise ValueError(f"QCNN expects a 4-qubit statevector with shape (16,), got {state.shape}")
    if parameters.shape != (42,):
        raise ValueError(f"QCNN expects 42 parameters, got {parameters.shape}")

    output = np.asarray(state, dtype=np.complex128)
    for wires in ((0, 1), (2, 3), (1, 2), (3, 0)):
        output = _convolution(output, wires, parameters[0:15])
    for source, sink in ((0, 2), (1, 3)):
        output = _pool(output, source, sink, parameters[15:21])
    output = _convolution(output, (2, 3), parameters[21:36])
    output = _pool(output, 2, 3, parameters[36:42])
    probabilities = np.abs(output.reshape(2, 2, 2, 2)) ** 2
    return float(probabilities[:, :, :, 0].sum() - probabilities[:, :, :, 1].sum())


def predict_expectations(states: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return np.asarray([qcnn_expectation(state, parameters) for state in states])


def metrics(states: np.ndarray, labels: np.ndarray, parameters: np.ndarray) -> dict[str, float]:
    expectations = predict_expectations(states, parameters)
    targets = 2 * labels.astype(float) - 1
    predictions = (expectations >= 0).astype(np.int8)
    return {
        "loss": float(np.mean((expectations - targets) ** 2)),
        "accuracy": float(np.mean(predictions == labels)),
    }


@dataclass(frozen=True)
class QCNNTrainingResult:
    parameters: np.ndarray
    history: list[dict[str, float]]
    best_step: int


def train_qcnn_spsa(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    val_states: np.ndarray,
    val_labels: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    perturbation: float,
) -> QCNNTrainingResult:
    """Train full-batch with two objective evaluations per SPSA step."""
    rng = np.random.default_rng(seed)
    parameters = rng.normal(0.0, 0.1, 42)
    best_parameters = parameters.copy()
    best_step = 0
    best_val_loss = np.inf
    history: list[dict[str, float]] = []

    targets = 2 * train_labels.astype(float) - 1

    def loss(candidate: np.ndarray) -> float:
        predictions = predict_expectations(train_states, candidate)
        return float(np.mean((predictions - targets) ** 2))

    for step in range(steps + 1):
        train_metrics = metrics(train_states, train_labels, parameters)
        val_metrics = metrics(val_states, val_labels, parameters)
        history.append({
            "step": step,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_parameters = parameters.copy()
            best_step = step
        if step == steps:
            break

        delta = rng.choice((-1.0, 1.0), size=parameters.shape)
        scale = perturbation / (step + 1) ** 0.101
        rate = learning_rate / (step + 1) ** 0.602
        gradient = (loss(parameters + scale * delta) - loss(parameters - scale * delta)) / (2 * scale) * delta
        parameters = parameters - rate * gradient

    return QCNNTrainingResult(best_parameters, history, best_step)

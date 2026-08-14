import numpy as np

from conditional_quddpm.models.quddpm import (
    _reverse_unitary,
    bloch_z,
    condition_angles,
    fidelity_mmd,
    forward_diffusion,
    generate_quddpm,
    haar_states,
    load_quddpm_checkpoint,
    pole_clusters,
    reverse_step,
    save_quddpm_checkpoint,
    train_stepwise_quddpm,
)


def test_reverse_circuit_is_unitary_and_measurement_preserves_pure_states():
    rng = np.random.default_rng(2)
    parameters = rng.normal(size=(3, 7))
    unitary = _reverse_unitary(parameters)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-12)

    states = haar_states(12, 3)
    output = reverse_step(states, parameters, np.pi / 3, np.linspace(0.01, 0.99, len(states)))
    assert np.allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-12)


def test_forward_diffusion_and_measurement_are_reproducible():
    targets = pole_clusters(16, [0, 1], seed=4)
    first = forward_diffusion(targets, steps=2, seed=5)
    second = forward_diffusion(targets, steps=2, seed=5)
    assert all(np.array_equal(first[label][step], second[label][step]) for label in targets for step in range(3))

    parameters = np.zeros((2, 7))
    uniforms = np.linspace(0.1, 0.9, 16)
    assert np.array_equal(
        reverse_step(targets[0], parameters, 0.0, uniforms),
        reverse_step(targets[0], parameters, 0.0, uniforms),
    )


def test_fidelity_mmd_and_condition_encoding():
    targets = pole_clusters(20, [0, 1], seed=6)
    assert fidelity_mmd(targets[0], targets[0]) < 1e-12
    assert fidelity_mmd(targets[0], targets[1]) > 0.5
    assert condition_angles([0]) == {0: 0.0}
    assert condition_angles([0, 1]) == {0: 0.0, 1: np.pi}


def test_conditional_smoke_training_separates_classes_and_reproduces_generation(tmp_path):
    labels = [0, 1]
    targets = pole_clusters(24, labels, seed=11)
    result, _ = train_stepwise_quddpm(
        targets,
        diffusion_steps=2,
        layers=3,
        samples=24,
        forward_seed=12,
        source_seed=13,
        init_seed=14,
        spsa_seed=15,
        measurement_seed=16,
        training_steps=300,
        learning_rate=0.35,
        perturbation=0.12,
    )
    first = generate_quddpm(result, labels, 48, source_seed=101, measurement_seed=102)
    checkpoint = tmp_path / "quddpm.npz"
    save_quddpm_checkpoint(checkpoint, result)
    second = generate_quddpm(load_quddpm_checkpoint(checkpoint), labels, 48, source_seed=101, measurement_seed=102)
    assert all(np.array_equal(first[label], second[label]) for label in labels)
    assert all(history[-1]["loss"] < history[0]["loss"] for history in result.histories)
    assert np.mean(bloch_z(first[0])) > 0 > np.mean(bloch_z(first[1]))
    assert all(fidelity_mmd(first[label], targets[label]) < fidelity_mmd(haar_states(48, 101 + label), targets[label]) for label in labels)

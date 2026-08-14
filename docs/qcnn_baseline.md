# Real-only 4-qubit QCNN Baseline

## Architecture basis

The implementation follows the simplified QCNN pattern in the TensorFlow Quantum QCNN tutorial and the convolution/pooling hierarchy introduced by Cong, Choi, and Lukin:

- shared 15-parameter two-qubit convolution blocks;
- shared 6-parameter pooling blocks;
- 4→2→1 hierarchy;
- final `Z` expectation readout on qubit 3.

For four qubits, convolution edges are `(0,1)`, `(2,3)`, `(1,2)`, `(3,0)`, followed by pooling `0→2`, `1→3`. A second convolution acts on `(2,3)`, followed by pooling `2→3`. Separate layers use separate parameters, giving 42 trainable parameters. The implementation uses the existing NumPy statevector simulator rather than adding TensorFlow Quantum solely for this small benchmark.

References:

- I. Cong, S. Choi, and M. D. Lukin, “Quantum Convolutional Neural Networks,” *Nature Physics* (2019), [arXiv:1810.03787](https://arxiv.org/abs/1810.03787).
- [TensorFlow Quantum QCNN tutorial](https://www.tensorflow.org/quantum/tutorials/qcnn), consulted for the concrete convolution/pooling parameter-sharing pattern.

## Leakage controls

`split_manifest.json` is the only source of split membership. The loader:

1. verifies all dataset checksums;
2. requires a one-to-one match between NPZ and manifest parameter IDs;
3. orders and groups samples from manifest records;
4. rejects label mismatches, duplicate IDs, unknown splits, and overlap.

The `{10,25,50,100}` states/class sweep uses prefixes of one seeded class-specific ordering. Consequently every smaller training set is a strict subset of every larger set. Validation selects the best optimization step; test data is used only for final metrics of each declared run.

## Training and artifacts

The baseline uses full-batch SPSA, requiring two train-loss evaluations per step independent of parameter count. Configs record dataset paths, subset seed, model seeds, steps, learning rate, and perturbation. Every run writes:

```text
results/qcnn_baseline/<dataset>/real-<n>/seed-<seed>/
├── history.json
├── metrics.json
└── parameters.npy
```

`metrics.json` records the exact training parameter IDs and train/validation/test accuracy, macro-F1, loss, and sample count. This contract is shared by future augmentation arms; only `method` and the training-state provenance should change.

## Commands

Smoke run:

```bash
uv run run-qcnn-baseline \
  --config configs/qcnn/baseline_4q_smoke.yaml \
  --output results/qcnn_baseline_smoke
```

Full real-only sweep:

```bash
uv run run-qcnn-baseline \
  --config configs/qcnn/baseline_4q.yaml \
  --output results/qcnn_baseline
```

## Circuit verification

The research baseline is protected by five independent checks:

1. every Euler, Pauli-interaction, and CNOT gate is unitary;
2. the full QCNN preserves state norm;
3. zero-parameter computational-basis outputs match the circuit's parity mapping;
4. a fixed state/parameter pair has a committed golden expectation;
5. an independent explicit basis-index simulator matches the optimized tensor-axis simulator.

A direct TensorFlow Quantum/PennyLane numerical equivalence test was not added because it would introduce a large framework solely for testing. The explicit simulator cross-check validates wire ordering, CNOT direction, gate application, pooling, and readout independently while retaining the minimal NumPy environment.

## Optimizer freeze study

The initial 60-step pilot placed 15/24 best checkpoints at the final step, so it was not treated as a frozen training protocol. A replacement convergence study used **train and validation only** (`evaluate_test: false`):

```bash
uv run run-qcnn-baseline \
  --config configs/qcnn/baseline_4q_convergence.yaml \
  --output results/qcnn_convergence
```

The study covered random/blocked datasets, 10 and 100 real states/class, and three paired seed specifications. With a 300-step cap, patience 40, and minimum validation improvement `1e-4`:

- test evaluation was absent (`test: null`) for all 12 runs;
- 5/12 runs stopped early;
- 2/12 best checkpoints occurred at the final step, down from 15/24;
- 3/12 occurred within the final five steps;
- final ten-step validation-loss ranges were approximately `6.6e-4` to `6.3e-3`.

The remaining late checkpoints showed small validation changes rather than the systematic boundary concentration of the pilot. The frozen development protocol is therefore:

- full-batch SPSA;
- maximum 300 steps;
- learning rate `0.15`;
- perturbation `0.1`;
- patience `40`;
- validation `min_delta = 1e-4`;
- separate initialization and SPSA seeds;
- validation-loss checkpoint selection.

No test metric was used to select these settings.

## Provenance contract

Every frozen run records an immutable run ID, status, Git SHA/dirty flag, all three dataset checksums, Python/NumPy/SciPy versions, resolved run config, exact training IDs, separate initialization/SPSA seeds, history, checkpoint, and train/validation/test metrics. Summary artifacts include per-seed values, mean and population standard deviation for accuracy/macro-F1/loss, failure rate, and the balanced majority/random baseline of `0.5`.

## Development and confirmatory benchmarks

The current random/blocked datasets, split seeds, nested subset seed, and any test metrics already observed are designated **development benchmarks**. The frozen protocol must not be changed in response to their test results. Final confirmatory evaluation will generate new random and blocked manifests using previously unused dataset/split seeds, apply the frozen protocol once, and keep those test states unopened until model and augmentation protocols are fixed.

## Initial pilot result (superseded for protocol freeze)

Pilot implementation commit: `1b96a21`. Three model seeds, 60 SPSA steps, and one fixed nested-subset seed:

| Split | Real/class | Accuracy mean ± std | Macro-F1 mean | Test loss mean |
|---|---:|---:|---:|---:|
| Random | 10 | 0.9667 ± 0.0471 | 0.9663 | 0.5273 |
| Random | 25 | 0.9722 ± 0.0393 | 0.9720 | 0.4987 |
| Random | 50 | 0.9889 ± 0.0079 | 0.9889 | 0.4868 |
| Random | 100 | 0.9944 ± 0.0079 | 0.9944 | 0.4885 |
| Blocked | 10 | 0.8333 ± 0.2357 | 0.7778 | 0.6972 |
| Blocked | 25 | 0.8333 ± 0.2357 | 0.7778 | 0.6991 |
| Blocked | 50 | 0.8333 ± 0.2357 | 0.7778 | 0.6958 |
| Blocked | 100 | 0.8333 ± 0.2357 | 0.7778 | 0.7187 |

These are baseline engineering results, not final research claims. Blocked-split variance is high with three seeds, and SPSA/config tuning was not expanded after observing test metrics. Future augmentation comparisons must reuse the exact split, nested-subset IDs, model seeds, architecture, optimizer settings, and test protocol.

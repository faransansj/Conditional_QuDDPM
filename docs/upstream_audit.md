# Upstream QuDDPM and Conditioning Audit

Audit date: 2026-08-14

## Sources

### Original QuDDPM

- Paper: Bingzhi Zhang, Peng Xu, Xiaohui Chen, and Quntao Zhuang, “Generative quantum machine learning via denoising diffusion probabilistic models,” *Physical Review Letters* **132**, 100602 (2024), [arXiv:2310.05866](https://arxiv.org/abs/2310.05866).
- Official repository: [Francis-Hsu/QuantGenMdl](https://github.com/Francis-Hsu/QuantGenMdl).
- Audited repository commit: `afa16b893465aae07d5a4f669a44d18bc4016395`.
- Repository license: **none found**. No `LICENSE`, `COPYING`, or license statement was present. Public visibility does not grant reuse rights; source will be treated as read-only reference unless the authors add a compatible license or grant permission.

### Conditional extension

- Paper: Daniel Quinn, Lorenzo Buffoni, Stefano Gherardini, and Gabriele De Chiara, “Conditioning in Generative Quantum Denoising Diffusion Models,” *Quantum Science and Technology* **11**, 035047 (2026), [arXiv:2509.17569](https://arxiv.org/abs/2509.17569), DOI: 10.1088/2058-9565/ae8881.
- Paper license: CC BY 4.0 for the article.
- Code availability: the paper states that code will be made openly available with the final accepted version, but no official repository was discoverable during this audit. Article licensing does not license unpublished source code.

## Original QuDDPM definition

### Forward process

For each target pure state, stepwise random scrambling circuits move the ensemble toward a Haar-random distribution. The official JAX implementation exposes:

- `scrambleCircuitOneQubit`
- `scrambleCircuitMultiQubit`
- `setDiffusionDataMultiQubit`

For `n` qubits and step schedule `diff_hs`, the multi-qubit helper samples per-state single-qubit rotation angles and entangling angles, applies the accumulated scrambling circuit, and stores intermediate ensembles. In audited source lines 98–105, rotations are sampled in `[-π/8, π/8]`, entangling coefficients in `[0.4, 0.6]`, and both are scaled by the diffusion schedule.

### Reverse process

Each reverse step uses a parameterized circuit over data and ancilla qubits:

1. initialize ancillas in `|0…0⟩`;
2. apply `L` hardware-efficient layers;
3. each layer applies `RX` and `RY` to every data/ancilla qubit;
4. apply staggered nearest-neighbor `CZ` gates;
5. projectively measure ancillas;
6. retain and normalize the conditional data-register state;
7. repeat from step `T` to zero.

The ancilla measurement makes the effective map on the data register non-unitary. Generation begins from Haar-random pure states.

### Training and loss

Training is divide-and-conquer: reverse step `t` is trained against the corresponding forward-diffused target ensemble while later reverse steps remain fixed. The official many-body notebook uses:

- natural-distance/MMD-style ensemble loss from pairwise state fidelities;
- JAX automatic differentiation;
- Optax Adam, learning rate `5e-4`;
- Gaussian parameter initialization with seed 42;
- 4 data qubits, 2 ancillas, 30 diffusion steps, 12 circuit layers;
- 100 generated samples per reverse step and 3001 epochs per step.

The repository also implements Wasserstein distance using infidelity `1-|⟨ψ|ϕ⟩|²` as the transport cost, although the phase notebook imports `naturalDistance` for training.

## Conditioning definition

The conditional paper keeps one shared reverse-model parameter set across classes. A class is encoded by a conditioning angle `μ` applied during ancilla preparation; the conditioned model adds one gate per ancilla qubit. For two classes, the paper considers angles `0` and `π` and also compares computational-basis ancilla conditioning. The reverse ansatz remains layered `RX`/`RY` rotations plus entanglers and ancilla measurement.

At every reverse step, the conditioned loss sums or averages an ensemble distance over classes. The paper defines both:

- MMD built from pairwise fidelity;
- Wasserstein distance with pairwise infidelity cost.

Distances may be normalized by the distance between each target class and the Haar ensemble. Training proceeds one reverse step at a time with Adam and shared conditioned parameters.

## Framework and dependency assessment

The official repository uses TensorCircuit with separate JAX, TensorFlow, and PyTorch implementations/notebooks. The phase task specifically requires JAX, Optax, TensorCircuit, `opt_einsum`, NumPy, and plotting/LaTeX tooling. Distance modules additionally reference POT and OTT.

Risks:

- no pinned dependency file;
- no Python version declaration;
- all three backends are listed as prerequisites for all notebooks;
- JAX source sets x64 but TensorCircuit state dtype to `complex64`;
- notebook execution depends on pre-generated files under `data/phase`;
- notebook plotting enables external LaTeX.

Introducing this entire stack into the main QCNN environment would be unnecessary. Phase 3 should use a separate optional dependency group or isolated environment after a minimal backend is selected.

## Reuse decision

| Component | Decision | Reason |
|---|---|---|
| Official source files | Do not copy | No repository license found. |
| Forward schedule/circuit concepts | Reimplement from paper | Scientific method is documented; implementation must be independently written and cited. |
| Reverse `RX/RY/CZ` ansatz | Reimplement minimally | Small, paper-defined circuit; avoids three-backend dependency. |
| Ancilla measurement semantics | Reimplement and unit-test | Core non-unitary operation requires explicit probability/reproducibility tests. |
| `naturalDistance`/MMD formula | Independently implement | Formula is paper-defined; source code is unlicensed. |
| Wasserstein/POT path | Defer | Not required for first smoke task; MMD/fidelity is sufficient initially. |
| Conditional ancilla angle | Reimplement from paper | No official conditioned source was available. |
| Official pre-generated phase data | Do not reuse | This project already has leakage-controlled TFIM datasets. |

## Official smoke-task reproducibility

**Current status: not reproducible as a clean automated smoke test without repair.**

Evidence:

- no CLI or documented smoke command;
- no lockfile/requirements file;
- notebooks are the only experiment entry points;
- the audited `QDDPM_phase.ipynb` calls `model.backwardOutput_t(input_tplus1, params_t)` while the current JAX source signature requires an additional `key` argument;
- model preparation/generation creates PRNG seeds from wall-clock timestamps;
- the full reference phase configuration is expensive: 30 reverse steps × 3001 epochs.

Before TFIM conditioning, Phase 3 should independently reproduce a much smaller official-style task:

1. one-qubit ring or two-cluster ensemble;
2. 1–3 diffusion steps;
3. one ancilla and shallow reverse circuit;
4. fixed explicit PRNG keys;
5. demonstrated decrease in train/validation MMD;
6. physical and deterministic generated states.

This is a methodology reproduction, not a source-code port.

## Phase 3 entry decision

The literature and upstream architecture are sufficiently specified for an independent minimal implementation. Direct code reuse is blocked by absent licensing and reproducibility issues. Phase 3 may begin only after the QCNN protocol freeze commit is recorded; the first Phase 3 deliverable is the isolated one-qubit reproduction above, not TFIM augmentation.

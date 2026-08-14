# Atomic TODO Ledger

Priority order is top-to-bottom within each section. Check an item only after its acceptance criteria are validated.

## P0 — Start here

- [x] Audit upstream QuDDPM, MSQuDDPM, and conditioning repositories
  - Goal: identify reusable code, APIs, licenses, framework/version constraints, and reference smoke tasks.
  - Inputs: papers arXiv:2310.05866, 2411.17608, 2509.17569 and linked official repositories.
  - Outputs: `docs/upstream_audit.md` with commit URLs, license findings, reuse decision, and risks.
  - Acceptance criteria: each model has a verified source or is marked unavailable; no code is copied before license compatibility is recorded.
  - Dependencies: none.

- [x] Freeze the 4-qubit TFIM benchmark convention
  - Goal: define Hamiltonian sign, open/periodic boundary, `J`, `g/J` domain/grid, class regions, critical exclusion interval, and split proportions.
  - Inputs: QCNN/TFIM literature and finite-size checks.
  - Outputs: one dataset YAML config plus rationale in `docs/methodology.md`.
  - Acceptance criteria: every convention is explicit; class regions do not overlap; choices are made without test-result inspection.
  - Dependencies: none.

- [x] Select the minimum Python environment
  - Goal: choose one simulation/QML stack compatible with reusable upstream code.
  - Inputs: upstream audit and local Python/platform constraints.
  - Outputs: `pyproject.toml` or one environment file with pinned direct dependencies.
  - Acceptance criteria: clean install succeeds; versions and Python requirement are documented; no redundant quantum frameworks are added.
  - Dependencies: upstream audit.

- [x] Implement the 4-qubit TFIM Hamiltonian
  - Goal: construct the declared dense Hamiltonian for configurable `J`, `g`, and boundary condition.
  - Inputs: frozen dataset config.
  - Outputs: `src/datasets/tfim.py` and one focused test.
  - Acceptance criteria: matrix has shape `2^n × 2^n`, is Hermitian, and matches hand-checkable 2-qubit terms.
  - Dependencies: benchmark convention; environment.

- [x] Implement the TFIM ground-state solver
  - Goal: return normalized ground-state vector and minimum eigenvalue.
  - Inputs: TFIM Hamiltonian.
  - Outputs: solver function and focused numerical test.
  - Acceptance criteria: norm error and residual `||Hψ-Eψ||` meet a declared numerical tolerance; repeated input is deterministic.
  - Dependencies: TFIM Hamiltonian.

## P1 — Leakage-safe dataset

- [x] Define canonical parameter IDs and metadata schema
  - Goal: uniquely identify each Hamiltonian sample independently of serialization order.
  - Inputs: frozen benchmark config.
  - Outputs: schema fields for parameter ID, `n`, `J`, `g`, boundary, label, and split.
  - Acceptance criteria: IDs are stable and unique for the configured grid.
  - Dependencies: benchmark convention.

- [x] Implement parameter-level train/validation/test splitting
  - Goal: split IDs before state generation with optional class stratification.
  - Inputs: parameter metadata and split seed.
  - Outputs: `split_manifest.json` generator.
  - Acceptance criteria: pairwise split intersections are empty; union equals all IDs; fixed seed reproduces byte-equivalent assignments.
  - Dependencies: parameter schema.

- [x] Implement operational phase labeling
  - Goal: map declared `g/J` regions to ferromagnetic class 0 and paramagnetic class 1 while excluding critical points.
  - Inputs: dataset config.
  - Outputs: label function and boundary tests.
  - Acceptance criteria: class regions and excluded points exactly match config; invalid/ambiguous points fail clearly.
  - Dependencies: benchmark convention.

- [x] Serialize the 4-qubit dataset and manifest
  - Goal: save states and metadata without losing complex precision or provenance.
  - Inputs: split manifest and ground-state solver.
  - Outputs: versioned dataset artifact with checksum.
  - Acceptance criteria: round-trip preserves states within tolerance; metadata includes config/checksum; no duplicate IDs.
  - Dependencies: solver; split; labels.

- [x] Add dataset physicality and leakage validation command
  - Goal: run normalization, energy residual, class-count, determinism, and split-disjointness checks together.
  - Inputs: serialized dataset.
  - Outputs: machine-readable validation report and nonzero exit on failure.
  - Acceptance criteria: a valid fixture passes and an intentionally overlapping manifest fails.
  - Dependencies: serialized dataset.

## P2 — Reproducible QCNN baseline

- [x] Select and document one QCNN topology
  - Goal: fix a minimal architecture that supports the planned qubit counts.
  - Inputs: QCNN reference and chosen framework.
  - Outputs: architecture diagram/parameter table and rationale.
  - Acceptance criteria: input/readout conventions and parameter count are explicit; no augmentation-specific branch exists.
  - Dependencies: environment; upstream audit.

- [x] Implement QCNN forward pass
  - Goal: map a supported quantum state to a binary prediction.
  - Inputs: fixed QCNN topology.
  - Outputs: model module and shape/range test.
  - Acceptance criteria: pure-state input runs at 4 qubits; output is finite and in the declared prediction domain.
  - Dependencies: topology.

- [x] Implement QCNN training and validation selection
  - Goal: train without test access using separate initialization/training seeds.
  - Inputs: train/validation datasets and QCNN config.
  - Outputs: checkpoint and training history.
  - Acceptance criteria: fixed config/seeds reproduce metrics within declared deterministic limits; test loader is absent from training API.
  - Dependencies: QCNN forward pass; dataset validation.

- [x] Implement metric and result artifact writer
  - Goal: save accuracy, macro-F1, confusion counts, resolved config, seeds, code revision, and status.
  - Inputs: predictions and run metadata.
  - Outputs: `results/<id>/config.yaml` and `metrics.json` contract.
  - Acceptance criteria: metrics recompute from saved counts; interrupted/failed runs are represented explicitly.
  - Dependencies: environment.

- [x] Run the real-only limited-data smoke benchmark
  - Goal: verify end-to-end operation at the smallest budget and one seed.
  - Inputs: 4-qubit dataset and QCNN.
  - Outputs: one complete result directory.
  - Acceptance criteria: run completes, artifacts validate, and performance is compared with declared random/majority guessing without imposing an arbitrary pass threshold.
  - Dependencies: QCNN training; result writer.

- [x] Run the 3-seed real-data pilot sweep
  - Goal: estimate baseline variance/runtime over `{10,25,50,100}` real states per class.
  - Inputs: frozen baseline configs.
  - Outputs: per-seed runs and aggregate mean/std table.
  - Acceptance criteria: every planned cell is present or has recorded failure; no test-driven hyperparameter changes occur between cells.
  - Dependencies: baseline smoke.

## P3 — Pure-state augmentation

- [x] Reproduce a minimal upstream QuDDPM task
  - Goal: verify the reused implementation before TFIM modifications.
  - Inputs: pinned upstream code/config.
  - Outputs: reproduction log and diagnostic comparison.
  - Acceptance criteria: declared qualitative/quantitative reference behavior is met or discrepancy is documented before proceeding.
  - Dependencies: upstream audit; environment.

- [x] Specify the class-conditioning mechanism
  - Goal: choose label injection consistent with the conditioning reference and current QuDDPM architecture.
  - Inputs: upstream reproduction and conditioning paper.
  - Outputs: design note and config fields.
  - Acceptance criteria: parameter sharing, label encoding, training loss, and sampling API are explicit.
  - Dependencies: QuDDPM reproduction.

- [ ] Train Conditional QuDDPM on train-only TFIM states
  - Goal: produce one checkpoint for both classes.
  - Inputs: one limited-data training subset; no validation/test states in fitting.
  - Outputs: checkpoint, resolved config, train log.
  - Acceptance criteria: data-access audit lists only train IDs; run is reproducible under fixed seeds.
  - Dependencies: conditioning mechanism; validated dataset.

- [ ] Generate class-conditioned pure states
  - Goal: materialize requested counts per class with provenance.
  - Inputs: fixed checkpoint and generation seed.
  - Outputs: synthetic-state artifact and metadata.
  - Acceptance criteria: requested counts/labels are exact; normalization/purity checks pass; fixed seed reproduces outputs.
  - Dependencies: trained Conditional QuDDPM.

- [ ] Implement pure-state generative diagnostics
  - Goal: measure physicality, memorization, diversity/coverage, and class consistency.
  - Inputs: real train/validation and synthetic states.
  - Outputs: metrics JSON.
  - Acceptance criteria: nearest-training fidelity and declared observable distributions are reported by class; no test states are loaded.
  - Dependencies: generated pure states.

- [ ] Implement a simple pure-state perturbation baseline
  - Goal: create a low-complexity physically valid augmentation control.
  - Inputs: training states and train/validation-selected perturbation strength.
  - Outputs: synthetic control states and config.
  - Acceptance criteria: normalization passes; labels/provenance are retained; strength selection never uses test data.
  - Dependencies: validated dataset.

- [ ] Run the pure augmentation pilot and declared sweep
  - Goal: compare real-only, perturbation, and Conditional QuDDPM across matched budgets/seeds.
  - Inputs: frozen QCNN and augmentation configs.
  - Outputs: complete run matrix, paired deltas, mean/std.
  - Acceptance criteria: all arms use matched splits/QCNN seeds; missing/failed cells remain visible; null/negative deltas are reported.
  - Dependencies: baseline pilot; diagnostics; perturbation baseline.

## P4 — Mixed-state augmentation

- [ ] Implement a depolarizing channel for density matrices
  - Goal: create the first controlled mixed-state dataset transformation.
  - Inputs: pure density matrix and channel strength.
  - Outputs: channel function and analytic fixtures.
  - Acceptance criteria: zero-noise identity, trace preservation, Hermiticity, PSD, and expected one-qubit behavior pass.
  - Dependencies: environment.

- [ ] Build and validate the noisy split-preserving dataset
  - Goal: apply noise independently after split without changing IDs.
  - Inputs: pure dataset manifest and declared strengths.
  - Outputs: noisy dataset artifacts and reports.
  - Acceptance criteria: all outputs are physical; IDs/splits match source manifest; fixed settings reproduce artifacts.
  - Dependencies: depolarizing channel; validated pure dataset.

- [ ] Reproduce a minimal upstream MSQuDDPM task
  - Goal: verify published mixed-state behavior before conditioning.
  - Inputs: pinned upstream source.
  - Outputs: reproduction report.
  - Acceptance criteria: reference behavior is met or discrepancy is documented and resolved/accepted.
  - Dependencies: upstream audit; environment.

- [ ] Add and document class conditioning to MSQuDDPM
  - Goal: generate both TFIM labels with one shared conditional model.
  - Inputs: reproduced MSQuDDPM and conditioning design.
  - Outputs: conditional model/config and focused sampling test.
  - Acceptance criteria: both labels sample; conditioning path is tested; unconditioned reference behavior is not silently broken.
  - Dependencies: MSQuDDPM reproduction; conditioning specification.

- [ ] Train and sample Conditional MSQuDDPM using train IDs only
  - Goal: create reproducible class-conditioned mixed synthetic states.
  - Inputs: noisy training subset.
  - Outputs: checkpoint, synthetic artifact, provenance.
  - Acceptance criteria: no val/test training IDs; requested class counts; fixed-seed reproducibility.
  - Dependencies: conditional MSQuDDPM; noisy dataset.

- [ ] Implement mixed-state generative diagnostics
  - Goal: validate trace, Hermiticity, PSD, purity distribution, diversity, and class consistency.
  - Inputs: mixed real and synthetic states.
  - Outputs: metrics JSON.
  - Acceptance criteria: every physical violation is counted; per-class distribution diagnostics and nearest-training comparisons are saved.
  - Dependencies: MSQuDDPM samples.

- [ ] Run the noisy baseline and MSQuDDPM augmentation sweep
  - Goal: compare matched noisy real-only, simple perturbation, and MSQuDDPM arms.
  - Inputs: frozen noisy benchmark configs.
  - Outputs: run matrix and aggregate paired comparisons.
  - Acceptance criteria: same QCNN protocol/test set across arms; noise and synthetic budgets are explicit; negative findings retained.
  - Dependencies: mixed diagnostics; noisy QCNN baseline.

## P5 — Analysis and extensions

- [ ] Quantify generation-quality/downstream association
  - Goal: answer RQ5 without treating correlation as causation.
  - Inputs: completed run-level generation and QCNN metrics.
  - Outputs: correlation estimates, uncertainty, scatter plots, sample count.
  - Acceptance criteria: unit of analysis and multiple-comparison policy are documented; missing/failed runs are disclosed.
  - Dependencies: pure and mixed sweeps.

- [ ] Profile and gate 6-qubit then 8-qubit scaling
  - Goal: expand only when memory/time estimates are acceptable.
  - Inputs: 4-qubit profiles and target configs.
  - Outputs: profile report and explicit go/no-go per qubit/model.
  - Acceptance criteria: statevector and density-matrix memory/time are estimated; no scale claim precedes completed validation.
  - Dependencies: 4-qubit end-to-end benchmark.

- [ ] Add near-critical evaluation
  - Goal: answer RQ6 on a held-out, predeclared critical-region test slice.
  - Inputs: frozen critical interval and completed primary pipeline.
  - Outputs: separate near-critical metrics/figures.
  - Acceptance criteria: critical points were not used to tune labels/models; results are not merged into the primary metric post hoc.
  - Dependencies: primary benchmark.

- [ ] Prepare final reproducibility artifact
  - Goal: reproduce reported tables/figures from frozen configs and stored run artifacts.
  - Inputs: final runs and code revision.
  - Outputs: reproduction commands, figures, tables, methodology, and failure ledger.
  - Acceptance criteria: clean-environment smoke passes; every claim maps to an artifact; all completed and failed declared cells are accounted for.
  - Dependencies: final benchmark.

## Deferred roadmap (not MVP)

- [ ] Evaluate few-step QuDDPM/MSQuDDPM.
- [ ] Evaluate learnable quantum noise schedules.
- [ ] Add hardware-context and hardware-noise conditioning.
- [ ] Compare QGAN/QCBM augmentation.
- [ ] Add XXZ, Heisenberg, and topological-phase datasets.
- [ ] Compare alternative downstream QML classifiers.
- [ ] Design real-QPU validation after simulation conclusions stabilize.

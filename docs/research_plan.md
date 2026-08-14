# Research Plan

## Objective

Determine whether class-conditioned QuDDPM or MSQuDDPM synthetic quantum states improve QCNN phase classification when ground-truth TFIM training states are scarce, and identify the regimes in which augmentation helps or harms.

## Study claims

The strongest initial claim permitted is:

> In controlled classical simulations of finite TFIM systems, under declared data, noise, model, and compute conditions, conditional quantum diffusion augmentation changes held-out QCNN performance by a measured amount.

The study does not claim quantum advantage, QPU feasibility, or guaranteed augmentation benefit.

## Hypotheses

- **H1:** Conditional QuDDPM augmentation yields positive paired `Delta Acc` and macro-F1 improvement in at least part of the limited pure-state regime.
- **H2:** Conditional MSQuDDPM augmentation yields positive paired improvement in at least part of the controlled noisy/mixed-state regime.
- **H3:** Benefit is larger at smaller real-data budgets but may saturate or reverse as synthetic quantity increases.
- **H4:** Better generative diagnostics are associated with, but do not necessarily cause, better downstream performance.

Null and negative results are retained. Hypotheses, sweep grids, exclusion rules, and primary metrics are frozen before final test evaluation.

## Operational definitions

- **Ground truth:** exact-diagonalization TFIM states at predeclared parameter points.
- **Phase label:** operational class assigned from the declared `g/J` region, not an assertion of a sharp finite-size transition.
- **Limited data:** a fixed per-class subset of the training split, nested where practical.
- **Augmentation:** synthetic states added only to that subset; the real-state count never changes between matched arms.
- **Improvement:** paired augmented-minus-baseline test metric using the same split and QCNN seed.
- **Class consistency:** agreement with a classifier/observable rule fitted or defined without test-set access.

## Experimental units and independence

Hamiltonian parameter points—not duplicated serialized states—are assigned to train, validation, and test. Random seeds do not create new physical test points; uncertainty reports distinguish variability over training seeds from variability over dataset splits. Final design should use multiple split seeds if compute permits.

## Fairness controls

- QCNN architecture, initialization seed, optimizer, stopping rule, preprocessing, and test set are matched across arms.
- Validation tuning budgets are equal.
- Generator compute is reported separately; downstream comparisons do not hide extra synthetic-generation cost.
- If augmentation increases batches per epoch, both epoch-matched and update-budget-aware interpretations are recorded.
- Pure and mixed benchmarks are primary; cross-model claims require a genuinely matched input regime.

## Analysis plan

For each real budget, synthetic budget, method, noise condition, qubit count, split seed, and model seed:

1. save accuracy and macro-F1;
2. pair augmented and baseline runs by split/QCNN seed;
3. report mean, standard deviation, and paired differences;
4. plot performance against real and synthetic counts;
5. preserve all completed runs, including failures with failure reason;
6. examine generation-quality/downstream association only after primary results.

The final significance method and confidence interval are selected after checking design assumptions, then applied consistently. Effect sizes and uncertainty take precedence over binary significance.

## Research versus engineering decisions

**Research decisions (freeze before final benchmark):** Hamiltonian convention, boundary conditions, parameter regions, split policy, metrics, sweep grid, QCNN protocol, noise channels/strengths, seed count, statistical analysis.

**Engineering decisions:** module layout, serialization implementation, logging library, CLI naming, batching, checkpoint format. These may change if artifacts remain reproducible.

## Threats to validity

- Small finite chains may make labels architecture- or convention-dependent.
- A generator can memorize training states while appearing high-fidelity.
- A QCNN may exploit simulator artifacts rather than phase structure.
- Hyperparameter tuning can favor one augmentation arm.
- Three seeds support development diagnostics, not strong inferential claims.
- Density-matrix simulation may prevent matched scaling.
- Conditional diffusion literature is recent; implementation baselines require direct reproduction.

## Extension questions

Near-critical behavior, 6/8-qubit scaling, alternative Hamiltonians/classifiers, few-step diffusion, learned schedules, QGAN/QCBM baselines, hardware-context conditioning, and QPU validation remain explicitly outside the initial confirmatory study.

# Conditional QuDDPM 4Q TFIM NO-GO checkpoint

## Scope-correct conclusion

> Under the current implementation and frozen 4-qubit TFIM experimental setting, Conditional QuDDPM did not reach an augmentation-ready conditional generator. The evidence does not justify the cost of further objective engineering, so this research path stops here.

This is not a claim that QuDDPM can never support quantum augmentation.

## Rationale

1. Individual TFIM realization directions contain useful signal: K2 one-step probes improved each target objective and paired physics metric for all 8/8 realizations.
2. Realization gradients conflict: within-class and between-class negative-cosine fractions were 0.50 and 0.75.
3. Global MMD averaging cancels signal: the overall cancellation ratio was 0.2995.
4. Conflict-score reweighting did not stably improve the uniform baseline; its one-step global-MMD and physics improvements were both smaller.
5. Physics-heavy weighting improved aggregate physics but worsened global MMD (`+0.006933`).
6. Physics-heavy weighting concentrated weight in class 1 (0.772) and reduced effective realization count to 3.548/8.
7. No stable class-conditioned synthetic TFIM state generator passed an augmentation-ready gate; no QCNN augmentation claim was made.
8. Therefore K4 and further Conditional QuDDPM objective engineering are out of scope for this track.

## Frozen evidence

- K0/K1: `docs/rdm_kernel_diagnostics.md`
- K2: `docs/k2_realization_conflict.md`
- K3: `docs/k3_conflict_reweighting.md`
- Frozen configs: `configs/quddpm/kernel_k0.yaml` through `kernel_k3.yaml`
- Machine-readable results: `results/quddpm_kernel_diagnostics/`
- Integrity manifest: `docs/quddpm_nogo_artifacts.sha256`

The result directory is intentionally committed at this checkpoint despite the repository-wide `results/` ignore rule, so the archive branch is self-contained. Future augmentation work must not overwrite these paths.

## Lineage

```text
Conditional QuDDPM
  -> 4Q TFIM learning diagnostics
  -> RDM and directional-gradient diagnostics
  -> realization conflict and cancellation analysis
  -> one-shot conflict-aware reweighting
  -> 4Q TFIM NO-GO checkpoint
  -> archive/quddpm-tfim-nogo
  -> research/tfim-state-augmentation
       -> physics-aware augmentation
       -> geometry-aware augmentation
       -> QCNN downstream benchmark
```

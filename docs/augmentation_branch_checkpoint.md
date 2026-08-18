# State-augmentation branch checkpoint

```text
New research branch: research/tfim-state-augmentation
Starting SHA: 5aec728faf992205b4cf493bcfa5c757513d5154
Parent NO-GO SHA: 5aec728faf992205b4cf493bcfa5c757513d5154
Archive branch: archive/quddpm-tfim-nogo
Working tree at branch creation: clean
```

The archive and new branch share only the frozen starting checkpoint. `archive/quddpm-tfim-nogo` must remain fixed. New physics-aware and geometry-aware configs and outputs use `configs/augmentation/` and method-specific result namespaces; they must not modify `configs/quddpm/` or `results/quddpm_kernel_diagnostics/`.

```text
Conditional QuDDPM
  -> 4Q TFIM learning diagnostics
  -> gradient conflict analysis
  -> conflict-aware objective experiments
  -> NO-GO checkpoint
  -> archive/quddpm-tfim-nogo
  -> research/tfim-state-augmentation
  -> physics-aware augmentation
  -> geometry-aware augmentation
  -> QCNN downstream benchmark
```

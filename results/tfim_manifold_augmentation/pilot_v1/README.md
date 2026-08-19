# TFIM manifold confirmatory track — protocol v1

Status: generator pilot **PASS**; QCNN **NOT RUN**.

Run:

```bash
python -m conditional_quddpm.experiments.tfim_manifold_confirmatory \
  --config configs/augmentation/tfim_manifold_confirmatory.yaml \
  --output results/tfim_manifold_augmentation/pilot_v1
```

`resource_model.json` records Gate 0, oracle accounting, and the distinction between dataset-container materialization and scientific train-row use. Ground truth and exact-symmetry results are in `tfim_ground_truth.json` and `symmetry_diagnostic.json`. Generator, random-control, pairing, and source-budget audits contain per-sample provenance; `synthetic_states.npz` stores the corresponding pilot states. `protocol_freeze.json` freezes the pre-QCNN method/statistical payload at `eea815d1233cb17bcf085e018dc46bc4b3455be3ff38e4831e421426b3a5c922`. Result files are explicit `NOT_RUN` placeholders until fresh confirmatory datasets are generated. `manifest.sha256` covers every other file in this directory. Prior Phase A/B/C artifacts were not modified.

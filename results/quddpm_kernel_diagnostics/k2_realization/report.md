# K2 per-realization gradient conflict diagnostic

Frozen global-MMD diagnostic at rho1->rho0 best checkpoint; train split only. The K1 central finite-difference directional sketch and common randomness are reused.

## Reconstruction

- Global sketch relative error: 1.817e-15
- Class sketch relative errors: 1.035e-15, 1.018e-15
- Maximum directional residual: 4.996e-16

## Conflict

- Cancellation ratio overall / class 0 / class 1: 0.2995 / 0.6077 / 0.4936
- Within-class cosine mean / negative fraction: 0.0864 / 0.5000
- Between-class cosine mean / negative fraction: -0.1352 / 0.7500
- Mean realization-step target physics delta: -0.013774
- Mean realization-step other-class objective delta: +0.015834

No K3 training, generation gate, QCNN evaluation, or validation/test evaluation was performed. This recorded run used the generic dataset loader, which materialized all split arrays; only train rows were used numerically. The K2 loader was corrected afterward without rerunning or changing numerical artifacts.

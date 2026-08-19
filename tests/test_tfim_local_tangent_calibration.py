import json

from conditional_quddpm.experiments.tfim_local_tangent_calibration import calibrate


def test_train_only_calibration_selects_by_frozen_tie_break(tmp_path):
    decision = calibrate("configs/augmentation/local_perturbation/phase_c.json", tmp_path)
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert decision == json.loads((tmp_path / "decision.json").read_text())
    assert decision["status"] == "CALIBRATION_PASS"
    assert decision["selected"] == {"radius": "q50", "ratio": 1.0}
    assert decision["qcnn_runs"] == 0 and decision["test_metrics_accessed"] is False
    assert len(diagnostics["aggregate"]) == 9
    assert all(row["valid_cells"] == row["cells"] and row["accepted"] == row["requested"] for row in diagnostics["aggregate"])

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conditional_quddpm.experiments import tfim_confirmatory_qcnn_v2_3 as runner


def test_preflight_hash_mismatch_fails_closed(tmp_path):
    protocol = tmp_path / "protocol"
    shutil.copytree(runner.PROTOCOL, protocol)
    data = json.loads((protocol / "run_matrix.json").read_text())
    data["runs"][0]["budget"] = 11
    (protocol / "run_matrix.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.preflight(protocol)


def test_preflight_rejects_self_rechecksummed_execution_change(tmp_path):
    protocol = tmp_path / "protocol"
    shutil.copytree(runner.PROTOCOL, protocol)
    path = protocol / "execution_config.json"
    data = json.loads(path.read_text())
    data["spsa"]["updates"] = 299
    path.write_text(json.dumps(data))
    lines = (protocol / "checksums.sha256").read_text().splitlines()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (protocol / "checksums.sha256").write_text("\n".join(f"{digest}  execution_config.json" if line.endswith("  execution_config.json") else line for line in lines) + "\n")
    with pytest.raises(ValueError, match="frozen artifact content changed"):
        runner.preflight(protocol)


def test_dry_run_resolves_exact_authoritative_matrix_and_seed_pairing(tmp_path):
    runs = runner.resolve_runs(output=tmp_path)
    assert len(runs) == len({r["run_id"] for r in runs}) == 48
    assert all(set(("regime", "method", "budget", "repeat", "subset_seed", "init_seed", "spsa_seed", "input_path", "output_path")) <= r.keys() for r in runs)
    for regime in ("random", "blocked_g"):
        for budget in (10, 25, 50, 100):
            for repeat in range(3):
                pair = [r for r in runs if (r["regime"], r["budget"], r["repeat"]) == (regime, budget, repeat)]
                assert len(pair) == 2
                assert len({(r["subset_seed"], r["init_seed"], r["spsa_seed"]) for r in pair}) == 1


def test_execute_enforces_300_final_schema_and_integrity_resume(tmp_path, monkeypatch):
    runs = runner.resolve_runs(output=tmp_path)[:1]
    monkeypatch.setattr(runner, "resolve_runs", lambda protocol_dir=runner.PROTOCOL, output=runner.OUTPUT: [{**runs[0], "output_path": str(output / runs[0]["run_id"])}])
    states = np.eye(16, dtype=complex)[:2]
    monkeypatch.setattr(runner, "_load_run_data", lambda run: (states, np.array([0, 1]), np.array(["a", "b"]), states, np.array([0, 1]), states, np.array([0, 1])))
    calls = []
    def train(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(history=[{"step": i} for i in range(301)], final_parameters=np.zeros(42))
    monkeypatch.setattr(runner, "train_confirmatory_qcnn_spsa", train)
    monkeypatch.setattr(runner, "_evaluate", lambda *args: {"loss": 0.0, "accuracy": 0.5, "macro_f1": 0.5, "samples": 2})
    assert runner.execute(tmp_path)["executed"] == 1
    assert calls == [{"init_seed": runs[0]["init_seed"], "spsa_seed": runs[0]["spsa_seed"], "learning_rate": .15, "perturbation": .1, "parameter_updates": 300, "early_stopping": False, "checkpoint_selection": "final"}]
    result = json.loads((tmp_path / runs[0]["run_id"] / "result.json").read_text())
    assert result["updates_completed"] == 300 and result["evaluation_checkpoint"] == "final step 300"
    assert runner.execute(tmp_path)["skipped"] == 1 and len(calls) == 1
    (tmp_path / runs[0]["run_id"] / "result.sha256").write_text("bad\n")
    with pytest.raises(RuntimeError, match="retry forbidden"):
        runner.execute(tmp_path)
    assert len(calls) == 1


def test_analysis_requires_all_48_valid_runs(tmp_path):
    result = runner.analyze(tmp_path)
    assert result["verdict"] == "INCONCLUSIVE"
    assert not (tmp_path / "analysis.json").exists()


def test_confirmatory_trainer_rejects_hidden_hyperparameter_defaults():
    from conditional_quddpm.models.qcnn import train_confirmatory_qcnn_spsa
    states, labels = np.eye(16, dtype=complex)[:2], np.array([0, 1])
    with pytest.raises(ValueError, match="300 updates"):
        train_confirmatory_qcnn_spsa(states, labels, states, labels, init_seed=1, spsa_seed=2, learning_rate=.14, perturbation=.1)

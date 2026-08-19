import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conditional_quddpm.augmentation.geometry import fubini_study_distance
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2 import (
    DOMAINS, canonical_json, confirmatory_rng, freeze_protocol, gate_status,
    ground_state_with_gap, named_seed, sample_unique_ground_state, seed_manifest,
    state_hash, validate_confirmatory_training, validate_schema,
)
from conditional_quddpm.models import qcnn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tfim_manifold_augmentation/confirmatory_protocol_v2"
CONFIG = ROOT / "configs/augmentation/tfim_hamiltonian_assisted_confirmatory_v2.yaml"


def test_exactly_300_updates_final_checkpoint_and_no_early_stopping(monkeypatch):
    calls = []
    monkeypatch.setattr(qcnn, "predict_expectations", lambda states, parameters: calls.append(parameters.copy()) or np.zeros(len(states)))
    monkeypatch.setattr(qcnn, "metrics", lambda states, labels, parameters: {"loss": float(300-len(calls)), "accuracy": 0.0})
    result = qcnn.train_confirmatory_qcnn_spsa(np.zeros((1,16)), np.array([0]), np.zeros((1,16)), np.array([0]), init_seed=1, spsa_seed=2, learning_rate=.1, perturbation=.1)
    assert len(result.history) == 301 and result.best_step == 300 and not result.stopped_early
    assert np.array_equal(result.parameters, result.final_parameters)
    for conflict in ({"parameter_updates":299}, {"early_stopping":True}, {"checkpoint_selection":"best_validation"}):
        args={"parameter_updates":300,"early_stopping":False,"checkpoint_selection":"final",**conflict}
        with pytest.raises(ValueError): qcnn.train_confirmatory_qcnn_spsa(np.zeros((1,16)),np.array([0]),np.zeros((1,16)),np.array([0]),init_seed=1,spsa_seed=2,learning_rate=.1,perturbation=.1,**args)


def test_training_config_conflicts_fail_closed():
    good={"optimizer":"SPSA","parameter_updates":300,"early_stopping":False,"checkpoint_selection":"final","evaluation_step":300}
    validate_confirmatory_training(good)
    with pytest.raises(ValueError): validate_confirmatory_training({**good,"evaluation_step":299})


def test_degeneracy_unique_near_and_retry_exhaustion():
    e0,e1,gap,state=ground_state_with_gap(np.diag([0.,1.]),1e-10)
    assert (e0,e1,gap)==(0.,1.,1.) and np.linalg.norm(state)==pytest.approx(1)
    for matrix in (np.diag([0.,0.]),np.diag([0.,5e-11])):
        with pytest.raises(ValueError): ground_state_with_gap(matrix,1e-10)
    sample,audit=sample_unique_ground_state(lambda i:i,lambda g:np.diag([0.,0.]) if g<2 else np.diag([0.,1.]),maximum_retries=2,tolerance=1e-10)
    assert sample["parameter"]==2 and [row["accepted"] for row in audit]==[False,False,True]
    with pytest.raises(RuntimeError): sample_unique_ground_state(lambda i:i,lambda g:np.diag([0.,0.]),maximum_retries=1,tolerance=1e-10)


def test_named_seed_tree_is_order_independent_and_manifest_reproducible():
    first=seed_manifest(42); second=seed_manifest(42)
    assert first==second and list(first["domains"])==list(DOMAINS)
    existing={name:named_seed(42,name) for name in DOMAINS}
    unrelated=hashlib.sha256(b"unrelated-domain").digest()  # adding unrelated material cannot consume/spawn existing streams
    assert unrelated and existing=={name:named_seed(42,name) for name in reversed(DOMAINS)}
    assert canonical_json(first)==canonical_json(second)


def test_pcg64dxsm_factory_enforced_and_reproducible():
    a=confirmatory_rng(7); b=confirmatory_rng(7)
    assert isinstance(a.bit_generator,np.random.PCG64DXSM) and np.array_equal(a.random(8),b.random(8))
    with pytest.raises(ValueError): confirmatory_rng(7,bit_generator="PCG64")


def test_projective_hash_canonicalization_edge_cases():
    psi=np.array([0,1+2j,-3j],dtype=complex); expected=state_hash(psi,1e-12)
    assert expected==state_hash(-psi,1e-12)==state_hash(np.exp(.73j)*psi,1e-12)==state_hash(9*psi,1e-12)
    assert expected==state_hash(np.array([-0.+0.j,1+2j,-3j]),1e-12)
    assert state_hash(np.array([1e-14,1,0j]),1e-12)==state_hash(np.array([0,1,0j]),1e-12)
    assert expected!=state_hash(np.array([0,1+2j,-2j]),1e-12)
    with pytest.raises(ValueError): state_hash(np.array([np.nan,1]),1e-12)


def test_exact_and_near_duplicate_rules_remain_distinct():
    psi=np.array([1.,0.]); phase=np.exp(.2j)*psi; near=np.array([np.sqrt(1-5e-11),np.sqrt(5e-11)])
    assert state_hash(psi,1e-12)==state_hash(phase,1e-12)
    assert state_hash(psi,1e-12)!=state_hash(near,1e-12)
    assert fubini_study_distance(psi,near)**2 < 1e-10


def test_run_and_aggregate_schemas_fail_closed():
    run=dict.fromkeys(["protocol_version","protocol_hash","dataset_hash","split","budget","method","root_seed","child_seeds","primary_metric","run_status"]); run.update(num_spsa_updates=300,evaluation_checkpoint=300)
    validate_schema(run,"run")
    aggregate={"n_expected":2,"n_completed":2,"n_failed":0,"paired_effects":[.1,.2],"aggregate_effect":.15,"confidence_interval":[.1,.2],"decision":"PASS","decision_reason":"frozen rule"}
    validate_schema(aggregate,"aggregate")
    with pytest.raises(ValueError): validate_schema({**run,"evaluation_checkpoint":100},"run")
    with pytest.raises(ValueError): validate_schema({**aggregate,"decision":"GO"},"aggregate")


def test_protocol_hash_and_checked_artifacts_are_reproducible(tmp_path):
    protocol=json.loads((OUT/"protocol_manifest.json").read_text()); path=tmp_path/"protocol.json"
    digest=freeze_protocol(protocol,path)
    assert digest==hashlib.sha256((OUT/"protocol_manifest.json").read_bytes()).hexdigest()
    cfg=yaml.safe_load(CONFIG.read_text()); assert cfg["protocol"]["protocol_hash"]==digest
    entries=dict(line.split("  ",1)[::-1] for line in (OUT/"checksums.sha256").read_text().splitlines())
    assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==value for name,value in entries.items())


def test_gate_is_protocol_ready_but_dataset_and_qcnn_blocked():
    gate=gate_status({"protocol_v2_frozen":True})
    assert gate["protocol_v2_ready"] and not gate["dataset_freeze_ready"] and not gate["qcnn_confirmatory_ready"] and gate["status"]=="BLOCKED"
    frozen=json.loads((OUT/"gate.json").read_text())
    assert frozen["protocol_v2_ready"] is True and frozen["dataset_freeze_ready"] is frozen["qcnn_confirmatory_ready"] is False
    assert frozen["confirmatory_dataset_generated"] is frozen["confirmatory_qcnn_executed"] is frozen["qcnn_metrics_generated"] is False


def test_unknown_seed_domain_fails_closed():
    with pytest.raises(ValueError): named_seed(1,"future.unregistered")


def test_degeneracy_audit_records_required_fields():
    _,audit=sample_unique_ground_state(lambda i:.5,lambda g:np.diag([0.,1.]),maximum_retries=0,tolerance=1e-10)
    assert set(audit[0])=={"parameter","E0","E1","gap","accepted","rejection_reason"}


def test_complete_dataset_prerequisites_are_the_only_qcnn_unblock_path():
    checks={key:True for key in ("protocol_v2_frozen","fresh_dataset","provenance","physical","exact_duplicates","near_duplicates","seed_manifest","checksums","dataset_freeze_complete")}
    assert gate_status(checks)["qcnn_confirmatory_ready"] is True
    checks["near_duplicates"]=False
    assert gate_status(checks)["status"]=="BLOCKED"


def test_statistical_plan_freezes_blocked_g_and_failed_run_policy():
    plan=json.loads((OUT/"statistical_plan.json").read_text())
    assert plan["primary_metric"]=="test_accuracy" and plan["primary_regime"]=="blocked-g"
    assert "cannot PASS" in plan["split_decision"] and plan["retry_policy"].startswith("none")

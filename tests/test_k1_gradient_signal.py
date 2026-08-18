import numpy as np
import pytest
from conditional_quddpm.datasets.loader import load_tfim_dataset
from conditional_quddpm.datasets.tfim import tfim_observables
from conditional_quddpm.experiments import k1_gradient_signal as k1
from conditional_quddpm.models.quddpm import condition_angles, haar_states, reverse_parameter_count, reverse_step


def _tiny_problem(seed=7, directions=2):
    path={c:[haar_states(2,seed+10*c+t,4) for t in range(3)] for c in (0,1)}
    angles=condition_angles([0,1])
    uniforms={c:np.random.default_rng(seed+100+c).random((2,2)) for c in (0,1)}
    parameters=np.random.default_rng(seed+200).normal(0,0.15,(1,reverse_parameter_count(4,1)))
    rng=np.random.default_rng(seed+300); deltas=[rng.choice((-1.0,1.0),size=parameters.shape) for _ in range(directions)]
    return path,angles,uniforms,parameters,deltas


def _tiny_config():
    return {"dataset":"data/tfim_4q_random","subset_seed":31415,"train_states_per_class":1,"train_realizations":2,
      "diffusion_steps":2,"layers":1,"iterations":1,"ancillas":1,"measurement_outcomes":2,
      "trained_objective":"PER_OUTCOME_ENSEMBLE_MMD","kernels":["global","2-rdm"],"checkpoints":["initial","best"],
      "directions":2,"epsilon":0.15,"near_zero_threshold":1.0e-6,
      "spsa":{"learning_rate":0.5,"perturbation":0.15},
      "decision":{"validity_min_raw":-1.0e-8,"validity_max_clipped_fraction":0.05,"usable_min_signal_ratio":0.5,
        "usable_near_zero_slack":0.05,"conflict_min_reduction":0.05,"physics_min_alignment_gain":0.10,
        "consistency_conflict_tolerance":0.05,"consistency_alignment_tolerance":0.10},
      "seeds":{"train_forward":121,"init":140,"spsa":150,"measurement":160,"scale_probe":40160,"permutation":50160,"directions":60160}}


def test_deterministic_rerun_under_fixed_seeds():
    path,angles,uniforms,parameters,deltas=_tiny_problem()
    first=k1.analyze_checkpoint(parameters,path,0,angles,uniforms,deltas,0.15,1e-6,["global","2-rdm"])
    second=k1.analyze_checkpoint(parameters,path,0,angles,uniforms,deltas,0.15,1e-6,["global","2-rdm"])
    assert first==second


def test_identical_directions_and_plus_minus_randomness_across_kernels():
    path,angles,uniforms,parameters,deltas=_tiny_problem()
    both=k1.analyze_checkpoint(parameters,path,0,angles,uniforms,deltas,0.15,1e-6,["global","2-rdm"])
    only=k1.analyze_checkpoint(parameters,path,0,angles,uniforms,deltas,0.15,1e-6,["global"])
    assert [r["derivative_global"] for r in both["directions"]]==[r["derivative_global"] for r in only["directions"]]
    assert [r["direction"] for r in both["directions"]]==list(range(len(deltas)))
    shared=k1.ensemble_objectives(parameters,path,0,angles,uniforms,["global","2-rdm"])
    single=k1.ensemble_objectives(parameters,path,0,angles,uniforms,["global"])
    assert shared["aggregate"]["global"]==single["aggregate"]["global"]
    assert shared["class"]["global"]==single["class"]["global"]


def test_clipping_accounting_counts_raw_values():
    s=k1.clipping_accounting([0.1,-0.2,-1e-9,0.0,-3e-8,1e-8])
    assert s["count"]==6 and s["min_raw"]==pytest.approx(-0.2)
    assert s["below_zero_count"]==3 and s["below_zero_fraction"]==pytest.approx(0.5)
    assert s["below_neg1e8_count"]==2 and s["below_neg1e8_fraction"]==pytest.approx(2/6)
    assert s["clipped_count"]==3 and s["clipped_fraction"]==pytest.approx(0.5)


def test_physics_error_and_directional_derivative_fixture():
    path,angles,uniforms,parameters,deltas=_tiny_problem()
    result=k1.ensemble_objectives(parameters,path,0,angles,uniforms,["global"])
    for c in (0,1):
        target_obs=np.asarray([tfim_observables(s,4) for s in path[c][0]]).mean(axis=0)
        expected=[]
        for row in uniforms[c]:
            generated=reverse_step(path[c][1],parameters,angles[c],row)
            generated_obs=np.asarray([tfim_observables(s,4) for s in generated]).mean(axis=0)
            expected.append(0.5*((generated_obs[0]-target_obs[0])**2+(generated_obs[1]-target_obs[1])**2))
        assert result["physics"]["class"][c]==pytest.approx(np.mean(expected))
    epsilon=0.15; delta=deltas[0]
    plus=k1.ensemble_objectives(parameters+epsilon*delta,path,0,angles,uniforms,["global"])["physics"]["aggregate"]
    minus=k1.ensemble_objectives(parameters-epsilon*delta,path,0,angles,uniforms,["global"])["physics"]["aggregate"]
    analysis=k1.analyze_checkpoint(parameters,path,0,angles,uniforms,[delta],epsilon,1e-6,["global"])
    assert analysis["directions"][0]["derivative_physics"]==pytest.approx((plus-minus)/(2*epsilon))


def test_run_records_ids_is_deterministic_and_never_touches_test_split(tmp_path,monkeypatch):
    dataset=load_tfim_dataset("data/tfim_4q_random")
    class Guard:
        def __getattr__(self,name):
            if name=="test": raise AssertionError("test split accessed")
            return getattr(dataset,name)
    monkeypatch.setattr(k1,"load_tfim_dataset",lambda path:Guard())
    first=k1.run(_tiny_config(),tmp_path/"a"); second=k1.run(_tiny_config(),tmp_path/"b")
    assert first["analyses"]==second["analyses"] and first["decision"]==second["decision"]
    manifest=first["manifest"]
    assert manifest["train_ids"]==["class-0-00125","class-1-00034"]
    assert manifest["realization_ids"]["0"]==[{"realization":0,"forward_seed":121},{"realization":1,"forward_seed":122}]
    assert manifest["realization_ids"]["1"]==[{"realization":0,"forward_seed":100121},{"realization":1,"forward_seed":100122}]
    assert manifest["test_split_used"] is False
    assert first["decision"]["decision"] in ("GO","NO-GO")
    for name in ("config.yaml","run_manifest.json","metrics.json","signal_summary.csv","directions.csv","clipping_summary.csv"):
        assert (tmp_path/"a"/name).exists()

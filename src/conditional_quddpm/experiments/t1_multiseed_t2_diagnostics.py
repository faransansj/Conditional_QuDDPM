"""Gate T=2 diagnostics on reproducible T=1 tiny learnability."""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.underfitting_diagnostics import add_gaps, fit, metrics, obs, provenance
from conditional_quddpm.models.quddpm import _scramble_state, haar_states, reverse_step, train_single_reverse_steps


def passed(run):
    return (run["best_loss"] < run["initial_loss"] and all(run["class_order_agreement"].values())
        and all(m["validation_mmd"] < m["haar_validation_mmd"] for m in run["per_class"].values()))


def evaluate_step(model, forward, train, validation, reference, config, step, source):
    seeds=config["seeds"]; rng=np.random.default_rng(seeds["evaluation_measurement"] + 100*step)
    per_class={}
    for c in (0,1):
        expanded=np.repeat(source[c][:1],config["evaluation_samples"],axis=0)
        generated=reverse_step(expanded,model.parameters[step],model.conditioning[c],rng.random(len(expanded)))
        per_class[str(c)]=metrics(generated,forward[c][step],validation[c],haar_states(config["haar_samples"],seeds["haar"]+c,4))
    result={"step":step+1,"transition":f"rho_{step+1}->rho_{step}","per_class":per_class}
    add_gaps(result,reference if step==0 else {str(c):obs(forward[c][step]) for c in (0,1)})
    return result


def t2_diagnostic(train,validation,reference,config):
    seeds=config["seeds"]; started=time.perf_counter()
    model,forward,training=train_single_reverse_steps(train,diffusion_steps=2,layers=3,samples=len(train[0]),forward_seed=seeds["forward"],
      source_seed=seeds["source"],init_seed=seeds["init"],spsa_seed=seeds["spsa"],measurement_seed=seeds["measurement"],
      training_steps=config["iterations"],learning_rate=config["spsa"]["learning_rate"],perturbation=config["spsa"]["perturbation"],
      n_ancilla=config["ancillas"],source_mode="teacher_forced",measurement_repeats=config["measurement_objective_samples"])
    teacher=[evaluate_step(model,forward,train,validation,reference,config,step,{c:forward[c][step+1] for c in (0,1)}) for step in (0,1)]
    def rollout(initial,name):
        current={c:np.repeat(initial[c][:1],config["evaluation_samples"],axis=0) for c in (0,1)}; stages=[]; rng=np.random.default_rng(seeds["rollout_measurement"])
        for step in (1,0):
            current={c:reverse_step(current[c],model.parameters[step],model.conditioning[c],rng.random(len(current[c]))) for c in (0,1)}
            target_reference={str(c):obs(forward[c][step]) for c in (0,1)}
            stage={"step":step+1,"transition":f"rho_{step+1}->rho_{step}","per_class":{str(c):metrics(current[c],forward[c][step],validation[c],haar_states(config["haar_samples"],seeds["haar"]+c,4)) for c in (0,1)}}
            add_gaps(stage,reference if step==0 else target_reference); stages.append(stage)
        return {"source":name,"stages":stages}
    robustness=[]
    for step in (0,1):
      for strength in config["off_target_strengths"]:
        perturb_rng=np.random.default_rng(seeds["perturbation"]+100*step+round(1000*strength))
        source={c:np.asarray([_scramble_state(forward[c][step+1][0],perturb_rng,strength) for _ in range(config["evaluation_samples"])]) for c in (0,1)}
        eval_rng=np.random.default_rng(seeds["evaluation_measurement"]+1000+100*step+round(1000*strength))
        per_class={}
        for c in (0,1):
          generated=reverse_step(source[c],model.parameters[step],model.conditioning[c],eval_rng.random(len(source[c])))
          per_class[str(c)]=metrics(generated,forward[c][step],validation[c],haar_states(config["haar_samples"],seeds["haar"]+c,4))
        item={"step":step+1,"transition":f"rho_{step+1}->rho_{step}","strength":strength,"per_class":per_class}
        add_gaps(item,reference if step==0 else {str(c):obs(forward[c][step]) for c in (0,1)}); robustness.append(item)
    return {"T":2,"L":3,"iterations":config["iterations"],"measurement_objective_samples":config["measurement_objective_samples"],
      "runtime_seconds":time.perf_counter()-started,"training_history":model.histories,"step_training_diagnostics":training,
      "teacher_forced":teacher,"rollouts":[rollout({c:forward[c][2] for c in (0,1)},"rho_2"),
      rollout({c:haar_states(len(train[c]),seeds["source"]+c,4) for c in (0,1)},"haar")],"off_target_robustness":robustness}


def run(config,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); size=config["tiny_train_states_per_class"]
    subset=nested_train_subsets(ds.train,[size],config["subset_seed"])[size]
    train={c:subset.states[subset.labels==c] for c in (0,1)}; validation={c:ds.val.states[ds.val.labels==c] for c in (0,1)}; reference={str(c):obs(train[c]) for c in (0,1)}
    result={"provenance":provenance(),"data_access":{"train_ids":subset.parameter_ids.tolist(),"validation_diagnostics":True,"test_evaluated":False},"t1_runs":[]}
    for offset in config["seed_offsets"]:
        local=copy.deepcopy(config); local["seeds"].update({k:v+offset for k,v in config["seeds"].items() if k in ("source","init","spsa","measurement")})
        run1=fit(train,validation,reference,local,name="t1_multiseed",labels=[0,1],layers=3,iterations=config["iterations"],source_mode="haar")
        run1["seed_offset"]=offset; run1["passed_reproducibility_gate"]=passed(run1); result["t1_runs"].append(run1)
    result["t1_gate_passed"]=all(r["passed_reproducibility_gate"] for r in result["t1_runs"])
    result["t2_ran"]=result["t1_gate_passed"]
    if result["t1_gate_passed"]: result["t2"]=t2_diagnostic(train,validation,reference,config)
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/t1_multiseed_t2.yaml"); p.add_argument("--output",default="results/quddpm_t1_multiseed_t2"); a=p.parse_args()
    result=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"t1_gate_passed":result["t1_gate_passed"],"t2_ran":result["t2_ran"]},indent=2))
if __name__=="__main__": main()

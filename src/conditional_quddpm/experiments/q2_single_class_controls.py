"""Causal single-class controls for frozen T=2 reverse steps."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.q2_ensemble_generalization import summarize, trajectories
from conditional_quddpm.experiments.underfitting_diagnostics import obs, provenance
from conditional_quddpm.models.quddpm import fidelity_mmd, reverse_parameter_count, reverse_step, train_single_reverse_steps


def single_loss(parameters,source,target,angle,uniforms):
    return float(np.mean([fidelity_mmd(reverse_step(source,parameters,angle,row),target) for row in uniforms]))


def realization_alignment(model,path,step,config):
    N=config["train_realizations"]; R=config["measurement_outcomes"]; seeds=config["seeds"]; label=next(iter(model.conditioning)); angle=model.conditioning[label]
    uniforms=np.random.default_rng(seeds["measurement"]+1000*step+label).random((R,N)); rng=np.random.default_rng(seeds["directions"]+step); shape=model.parameters[step].shape; epsilon=config["gradient_epsilon"]; signatures=[[] for _ in range(N)]
    for _ in range(config["gradient_directions"]):
      delta=rng.choice((-1.0,1.0),size=shape)
      for i in range(N):
        plus=single_loss(model.parameters[step]+epsilon*delta,path[label][step+1][i:i+1],path[label][step][i:i+1],angle,uniforms[:,i:i+1]); minus=single_loss(model.parameters[step]-epsilon*delta,path[label][step+1][i:i+1],path[label][step][i:i+1],angle,uniforms[:,i:i+1]); signatures[i].append((plus-minus)/(2*epsilon))
    corr=[]; conflict=[]
    for i in range(N):
      corr.append([]); conflict.append([])
      for j in range(N):
        left,right=np.asarray(signatures[i]),np.asarray(signatures[j]); corr[i].append(float(np.corrcoef(left,right)[0,1])); conflict[i].append(float(np.mean(left*right<0)))
    mask=~np.eye(N,dtype=bool)
    return {"correlation":corr,"conflict_rate":conflict,"mean_off_diagonal_correlation":float(np.asarray(corr)[mask].mean()),"mean_off_diagonal_conflict":float(np.asarray(conflict)[mask].mean()),"mean_abs_signal_by_realization":[float(np.mean(abs(np.asarray(x)))) for x in signatures]}


def evaluate(model,path,training_path,step,config,seed):
    label=next(iter(model.conditioning)); outcomes=config["evaluation_measurement_outcomes"]; source=np.repeat(path[label][step+1],outcomes,axis=0); target=np.repeat(path[label][step],outcomes,axis=0); rng=np.random.default_rng(seed)
    generated=reverse_step(source,model.parameters[step],model.conditioning[label],rng.random(len(source))); result=summarize(generated,target,training_path[label][step+1],training_path[label][step]); generated_obs=obs(generated); target_obs=obs(target)
    result.update({"generated_observables":generated_obs,"target_observables":target_obs,"observable_absolute_error":{key:abs(generated_obs[key]-target_obs[key]) for key in ("Mx","Mz2")},"physics_error":float(np.mean([abs(generated_obs[key]-target_obs[key]) for key in ("Mx","Mz2")]))})
    return result


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); seeds=config["seeds"]; T=config["diffusion_steps"]
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]; val_split=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_split.states[val_split.labels==c] for c in (0,1)}; result={"provenance":provenance(),"data_access":{"train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_split.parameter_ids.tolist(),"test_evaluated":False},"classes":[]}
    for label in (0,1):
      train_path,train_ids=trajectories({label:train_states[label]},config["train_realizations"],T,seeds["train_forward"]); holdout,holdout_ids=trajectories({label:train_states[label]},config["holdout_realizations"],T,seeds["holdout_forward"]); validation,val_ids=trajectories({label:val_states[label]},config["validation_realizations"],T,seeds["validation_forward"]); expanded={label:train_path[label][0]}; class_started=time.perf_counter()
      model,_,diagnostics=train_single_reverse_steps(expanded,diffusion_steps=T,layers=config["layers"],samples=config["train_realizations"],forward_seed=seeds["train_forward"],source_seed=seeds["source"],init_seed=seeds["init"],spsa_seed=seeds["spsa"],measurement_seed=seeds["measurement"],training_steps=config["iterations"],learning_rate=config["spsa"]["learning_rate"],perturbation=config["spsa"]["perturbation"],n_ancilla=config["ancillas"],source_mode="teacher_forced",measurement_repeats=config["measurement_outcomes"],forward_trajectories=train_path)
      evaluations={name:[evaluate(model,path,train_path,step,config,seeds["evaluation"]+100000*label+10000*di+100*step) for step in (0,1)] for di,(name,path) in enumerate((("seen",train_path),("unseen",holdout),("validation",validation)))}
      result["classes"].append({"label":label,"runtime_seconds":time.perf_counter()-class_started,"realization_ids":{"train":train_ids,"holdout":holdout_ids,"validation":val_ids},"training_diagnostics":diagnostics,"training_history":model.histories,"realization_gradient_alignment":[realization_alignment(model,train_path,step,config) for step in (0,1)],"evaluations":evaluations})
    result["runtime_seconds"]=time.perf_counter()-started; (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    rows=[]
    for item in result["classes"]:
      for domain,steps in item["evaluations"].items():
       for step,m in enumerate(steps): rows.append({"class":item["label"],"domain":domain,"step":step+1,"mmd":m["mmd"],"physics_error":m["physics_error"],"Mx_error":m["observable_absolute_error"]["Mx"],"Mz2_error":m["observable_absolute_error"]["Mz2"],"diversity":m["mean_pairwise_fidelity"],"nearest_training_source_fidelity":m["nearest_training_source_fidelity"]})
    with (out/"single_class.csv").open("w",newline="") as f: w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/q2_single_class_controls.yaml"); p.add_argument("--output",default="results/quddpm_q2_single_class"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

"""Bounded realization-conflict-aware SPSA control for rho1->rho0."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.q2_ensemble_generalization import evaluate_step, rollout, trajectories
from conditional_quddpm.experiments.q2_objective_geometry import objective, train
from conditional_quddpm.experiments.underfitting_diagnostics import obs, provenance
from conditional_quddpm.models.quddpm import QuDDPMTrainingResult, fidelity_mmd, reverse_step


def task_losses(parameters,path,angles,uniforms):
    values=np.empty((2,len(path[0][0])))
    for c in (0,1):
      losses=np.zeros(len(path[c][0]))
      for row in uniforms[c]:
        generated=reverse_step(path[c][1],parameters,angles[c],row)
        losses += np.asarray([fidelity_mmd(generated[i:i+1],path[c][0][i:i+1]) for i in range(len(generated))])
      values[c]=losses/len(uniforms[c])
    return values


def train_robust_step0(standard_model,path,config):
    p=np.random.default_rng(config["seeds"]["init"]).normal(0,0.15,standard_model.parameters.shape)[0]; initial=p.copy(); delta_rng=np.random.default_rng(config["seeds"]["spsa"]); R=config["measurement_outcomes"]; N=config["train_realizations"]; angles=standard_model.conditioning
    uniforms={c:np.random.default_rng(config["seeds"]["measurement"]+c).random((R,N)) for c in (0,1)}; history=[]
    for iteration in range(config["iterations"]+1):
      scalar=objective(p,path,0,angles,uniforms,"PER_OUTCOME_ENSEMBLE_MMD"); item={"iteration":iteration,"standard_objective":scalar,"parameter_update_norm":float(np.linalg.norm(p-initial))}
      if iteration==config["iterations"]: history.append(item); break
      delta=delta_rng.choice((-1.0,1.0),size=p.shape); scale=config["spsa"]["perturbation"]/(iteration+1)**0.101; rate=config["spsa"]["learning_rate"]/(iteration+1)**0.602
      plus,minus=task_losses(p+scale*delta,path,angles,uniforms),task_losses(p-scale*delta,path,angles,uniforms); directional=(plus-minus)/(2*scale); class_medians=np.median(directional,axis=1); robust=float(class_medians.mean()); mean=float(directional.mean())
      item.update({"mean_directional":mean,"robust_directional":robust,"class_medians":class_medians.tolist(),"within_class_conflict":[float(np.mean(directional[c,:,None]*directional[c,None,:]<0)) for c in (0,1)],"learning_rate":rate,"perturbation":scale}); history.append(item); p-=rate*robust*delta
    parameters=standard_model.parameters.copy(); parameters[0]=p
    return QuDDPMTrainingResult(parameters,standard_model.histories,standard_model.conditioning,standard_model.n_data,standard_model.n_ancilla),history


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); seeds=config["seeds"]; N=config["train_realizations"]
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]; val_split=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_split.states[val_split.labels==c] for c in (0,1)}; train_path,train_ids=trajectories(train_states,N,2,seeds["train_forward"]); holdout,holdout_ids=trajectories(train_states,config["holdout_realizations"],2,seeds["holdout_forward"]); validation,val_ids=trajectories(val_states,config["validation_realizations"],2,seeds["validation_forward"])
    standard,_,_=train(train_path,config,"PER_OUTCOME_ENSEMBLE_MMD"); hybrid,robust_history=train_robust_step0(standard,train_path,config); result={"provenance":provenance(),"data_access":{"train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_split.parameter_ids.tolist(),"test_evaluated":False},"realization_ids":{"train":train_ids,"holdout":holdout_ids,"validation":val_ids},"rule":"d_class=median_i d_class,i; d_aggregate=mean_class d_class; update=d_aggregate*Delta","conditions":[]}
    for ci,(name,model) in enumerate((("STANDARD_MEAN",standard),("CLASS_MEDIAN_REALIZATION",hybrid))):
      evaluations={}
      for di,(domain,path) in enumerate((("seen",train_path),("unseen",holdout),("validation",validation))):
        refs={str(step):{str(c):obs(path[c][step]) for c in (0,1)} for step in (0,1)}; evaluations[domain]={"steps":[evaluate_step(model,step,path,path,train_path,refs[str(step)],seeds["evaluation"]+100000*ci+10000*di+100*step,config["evaluation_measurement_outcomes"]) for step in (0,1)],"rollout":rollout(model,path,train_path,refs,seeds["evaluation"]+100000*ci+10000*di,config["evaluation_measurement_outcomes"])}
      result["conditions"].append({"name":name,"evaluations":evaluations,"training_history":standard.histories if name=="STANDARD_MEAN" else [robust_history,standard.histories[1]]})
    result["runtime_seconds"]=time.perf_counter()-started; (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    rows=[]
    for condition in result["conditions"]:
      for domain,e in condition["evaluations"].items():
       for mode,items in (("isolated",e["steps"]),("rollout",e["rollout"])):
        for item in items: rows.append({"condition":condition["name"],"domain":domain,"mode":mode,"step":item["step"],"aggregate_mmd":item["aggregate_mmd"],"physics_error":item["physics_error"],"Mx_order":item["class_order_agreement"]["Mx"],"Mz2_order":item["class_order_agreement"]["Mz2"]})
    with (out/"comparison.csv").open("w",newline="") as f: w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/q2_conflict_aware_step0.yaml"); p.add_argument("--output",default="results/quddpm_q2_conflict_aware"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

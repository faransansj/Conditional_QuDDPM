"""Compare fixed versus iteration-resampled CRN objectives at frozen N=4/T=2."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.q2_ensemble_generalization import evaluate_step, rollout, trajectories
from conditional_quddpm.experiments.underfitting_diagnostics import obs, provenance
from conditional_quddpm.models.quddpm import QuDDPMTrainingResult, condition_angles, fidelity_mmd, reverse_parameter_count, reverse_step


def _batch(pool_size,batch_size,repeats,labels,rng):
    indices=rng.integers(pool_size,size=batch_size)
    uniforms={c:rng.random((repeats,batch_size)) for c in labels}
    raw=indices.tobytes()+b"".join(uniforms[c].tobytes() for c in labels)
    return {"indices":indices,"uniforms":uniforms,"id":hashlib.sha256(raw).hexdigest()[:16]}


def spsa_pair_batches(condition,fixed,pool_size,batch_size,repeats,labels,rng):
    """Return the same object for +/-; RESAMPLED_CRN changes it on the next call."""
    batch=fixed if condition=="FIXED_ALL" else _batch(pool_size,batch_size,repeats,labels,rng)
    return batch,batch


def train_condition(path,config,condition):
    labels=(0,1); T=config["diffusion_steps"]; N=config["train_realizations"]; repeats=config["measurement_outcomes"]; seeds=config["seeds"]
    angles=condition_angles(list(labels)); width=reverse_parameter_count(4,config["ancillas"]); parameters=np.random.default_rng(seeds["init"]).normal(0,0.15,(T,config["layers"],width)); histories=[]
    for step in range(T):
      p=parameters[step].copy(); initial=p.copy(); delta_rng=np.random.default_rng(seeds["spsa"]+step); batch_rng=np.random.default_rng(seeds["resampling"]+1000*step)
      fixed=_batch(N,N,repeats,labels,np.random.default_rng(seeds["fixed_measurement"]+1000*step)); fixed["indices"]=np.arange(N); history=[]
      def loss(candidate,batch):
        idx=batch["indices"]
        return float(np.mean([fidelity_mmd(reverse_step(path[c][step+1][idx],candidate,angles[c],batch["uniforms"][c][repeat]),path[c][step][idx]) for c in labels for repeat in range(repeats)]))
      for iteration in range(config["iterations"]+1):
        plus_batch,minus_batch=spsa_pair_batches(condition,fixed,N,N,repeats,labels,batch_rng); value=loss(p,plus_batch)
        item={"iteration":iteration,"objective":value,"batch_id":plus_batch["id"],"parameter_update_norm":float(np.linalg.norm(p-initial))}
        if iteration==config["iterations"]: history.append(item); break
        delta=delta_rng.choice((-1.0,1.0),size=p.shape); scale=config["spsa"]["perturbation"]/(iteration+1)**0.101; rate=config["spsa"]["learning_rate"]/(iteration+1)**0.602
        plus,minus=loss(p+scale*delta,plus_batch),loss(p-scale*delta,minus_batch); directional=(plus-minus)/(2*scale); gradient=directional*delta
        item.update({"plus_objective":plus,"minus_objective":minus,"plus_batch_id":plus_batch["id"],"minus_batch_id":minus_batch["id"],"directional_derivative":directional,"gradient_norm":float(np.linalg.norm(gradient)),"perturbation":scale,"learning_rate":rate}); history.append(item); p-=rate*gradient
      parameters[step]=p; histories.append(history)
    return QuDDPMTrainingResult(parameters,histories,angles,4,config["ancillas"])


def training_summary(histories,N,config):
    calls_per_step=3*config["iterations"]+1; sample_outcomes_per_call=2*N*config["measurement_outcomes"]
    return {"objective_calls_per_step":calls_per_step,"objective_calls_total":2*calls_per_step,
      "source_outcome_evaluations_per_step":calls_per_step*sample_outcomes_per_call,"source_outcome_evaluations_total":2*calls_per_step*sample_outcomes_per_call,
      "steps":[{"initial_objective":h[0]["objective"],"final_objective":h[-1]["objective"],"best_objective":min(x["objective"] for x in h),
        "parameter_update_norm":h[-1]["parameter_update_norm"],"objective_variance":float(np.var([x["objective"] for x in h])),
        "directional_derivative_variance":float(np.var([x["directional_derivative"] for x in h[:-1]])),"gradient_norm_mean":float(np.mean([x["gradient_norm"] for x in h[:-1]])),
        "gradient_norm_variance":float(np.var([x["gradient_norm"] for x in h[:-1]])),"crn_verified":all(x["plus_batch_id"]==x["minus_batch_id"] for x in h[:-1]),"unique_batches":len({x["batch_id"] for x in h})} for h in histories]}


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); N=config["train_realizations"]; T=config["diffusion_steps"]; seeds=config["seeds"]
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]; val_split=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_split.states[val_split.labels==c] for c in (0,1)}
    train_path,train_ids=trajectories(train_states,N,T,seeds["train_forward"]); holdout,holdout_ids=trajectories(train_states,config["holdout_realizations"],T,seeds["holdout_forward"]); validation,val_ids=trajectories(val_states,config["validation_realizations"],T,seeds["validation_forward"])
    result={"provenance":provenance(),"data_access":{"train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_split.parameter_ids.tolist(),"test_evaluated":False},"train_realization_ids":train_ids,"holdout_realization_ids":holdout_ids,"validation_realization_ids":val_ids,"conditions":[]}
    for ci,condition in enumerate(config["conditions"]):
      condition_started=time.perf_counter(); model=train_condition(train_path,config,condition); domains={"seen":train_path,"unseen":holdout,"validation":validation}; evaluations={}
      for di,(name,path) in enumerate(domains.items()):
        refs={str(step):{str(c):obs(path[c][step]) for c in (0,1)} for step in (0,1)}
        evaluations[name]={"steps":[evaluate_step(model,step,path,path,train_path,refs[str(step)],seeds["evaluation"]+100000*ci+10000*di+100*step,config["evaluation_measurement_outcomes"]) for step in (0,1)],"rollout":rollout(model,path,train_path,refs,seeds["evaluation"]+100000*ci+10000*di,config["evaluation_measurement_outcomes"])}
      gaps={}
      for step in (0,1):
        gaps[f"isolated_step_{step+1}"]={metric:evaluations["unseen"]["steps"][step][key]-evaluations["seen"]["steps"][step][key] for metric,key in (("mmd","aggregate_mmd"),("physics","physics_error"))}
      gaps["final_rollout"]={metric:evaluations["unseen"]["rollout"][-1][key]-evaluations["seen"]["rollout"][-1][key] for metric,key in (("mmd","aggregate_mmd"),("physics","physics_error"))}
      result["conditions"].append({"name":condition,"runtime_seconds":time.perf_counter()-condition_started,"training_summary":training_summary(model.histories,N,config),"training_history":model.histories,"evaluations":evaluations,"generalization_gaps":gaps})
    result["runtime_seconds"]=time.perf_counter()-started; (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    rows=[]
    for condition in result["conditions"]:
      for domain,e in condition["evaluations"].items():
       for mode,items in (("isolated",e["steps"]),("rollout",e["rollout"])):
        for item in items: rows.append({"condition":condition["name"],"domain":domain,"mode":mode,"step":item["step"],"aggregate_mmd":item["aggregate_mmd"],"physics_error":item["physics_error"],"Mx_order":item["class_order_agreement"]["Mx"],"Mz2_order":item["class_order_agreement"]["Mz2"]})
    with (out/"comparison.csv").open("w",newline="") as f: w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/q2_sampling_semantics.yaml"); p.add_argument("--output",default="results/quddpm_q2_sampling_semantics"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

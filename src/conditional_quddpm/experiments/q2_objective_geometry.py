"""Compare deterministic per-outcome and pooled class-conditional ensemble MMD."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.q2_ensemble_generalization import evaluate_step, rollout, trajectories
from conditional_quddpm.experiments.underfitting_diagnostics import obs, provenance
from conditional_quddpm.models.quddpm import QuDDPMTrainingResult, condition_angles, fidelity_mmd, reverse_parameter_count, reverse_step


def per_outcome_ensemble_mmd(generated_by_outcome,target):
    """Mean biased fidelity-MMD over outcomes; each outcome contains all realizations."""
    return float(np.mean([fidelity_mmd(generated,target) for generated in generated_by_outcome]))


def combined_ensemble_mmd(generated_by_outcome,target):
    """One biased fidelity-MMD over all realization/outcome outputs and repeated targets."""
    generated=np.concatenate(generated_by_outcome); repeated_target=np.tile(target,(len(generated_by_outcome),1))
    return fidelity_mmd(generated,repeated_target)


def equal_class_objective(per_class_losses):
    return float(np.mean(list(per_class_losses.values())))


def generated_outcomes(source,parameters,angle,uniforms):
    return [reverse_step(source,parameters,angle,row) for row in uniforms]


def objective(candidate,path,step,angles,uniforms,condition):
    losses={}
    for c in (0,1):
      generated=generated_outcomes(path[c][step+1],candidate,angles[c],uniforms[c])
      losses[c]=(per_outcome_ensemble_mmd(generated,path[c][step]) if condition=="PER_OUTCOME_ENSEMBLE_MMD" else combined_ensemble_mmd(generated,path[c][step]))
    return equal_class_objective(losses)


def scale_and_permutation(path,parameters,step,angles,uniforms,seed):
    rng=np.random.default_rng(seed); original={}; permuted={}; permutation=rng.permutation(len(path[0][step]))
    for condition in ("PER_OUTCOME_ENSEMBLE_MMD","COMBINED_ENSEMBLE_MMD"):
      original[condition]=objective(parameters,path,step,angles,uniforms,condition)
      changed={c:list(path[c]) for c in (0,1)}
      changed={c:[array.copy() for array in changed[c]] for c in changed}
      for c in (0,1): changed[c][step]=changed[c][step][permutation]
      permuted[condition]=objective(parameters,changed,step,angles,uniforms,condition)
    return {"permutation":permutation.tolist(),"original":original,"permuted":permuted,"absolute_change":{k:abs(original[k]-permuted[k]) for k in original}}


def train(path,config,condition):
    labels=(0,1); T=config["diffusion_steps"]; N=config["train_realizations"]; R=config["measurement_outcomes"]; seeds=config["seeds"]; angles=condition_angles(list(labels)); width=reverse_parameter_count(4,config["ancillas"])
    initial_parameters=np.random.default_rng(seeds["init"]).normal(0,0.15,(T,config["layers"],width)); parameters=initial_parameters.copy(); histories=[]; probes=[]; checkpoints=[]
    for step in range(T):
      p=parameters[step].copy(); initial=p.copy(); delta_rng=np.random.default_rng(seeds["spsa"]+step); uniforms={c:np.random.default_rng(seeds["measurement"]+1000*step+c).random((R,N)) for c in labels}
      random_rng=np.random.default_rng(seeds["scale_probe"]+step); vectors=[p]+[random_rng.normal(0,0.15,p.shape) for _ in range(3)]
      probes.append({"step":step+1,"vectors":[{"PER_OUTCOME_ENSEMBLE_MMD":objective(v,path,step,angles,uniforms,"PER_OUTCOME_ENSEMBLE_MMD"),"COMBINED_ENSEMBLE_MMD":objective(v,path,step,angles,uniforms,"COMBINED_ENSEMBLE_MMD")} for v in vectors],"permutation":scale_and_permutation(path,p,step,angles,uniforms,seeds["permutation"]+step)})
      history=[]; intermediate=None; best_value=float("inf"); best_parameters=p.copy()
      for iteration in range(config["iterations"]+1):
        value=objective(p,path,step,angles,uniforms,condition); item={"iteration":iteration,"objective":value,"parameter_update_norm":float(np.linalg.norm(p-initial))}
        if value < best_value: best_value, best_parameters = value, p.copy()
        if iteration == config["iterations"] // 2: intermediate = p.copy()
        if iteration==config["iterations"]: history.append(item); break
        delta=delta_rng.choice((-1.0,1.0),size=p.shape); scale=config["spsa"]["perturbation"]/(iteration+1)**0.101; rate=config["spsa"]["learning_rate"]/(iteration+1)**0.602
        plus=objective(p+scale*delta,path,step,angles,uniforms,condition); minus=objective(p-scale*delta,path,step,angles,uniforms,condition); directional=(plus-minus)/(2*scale); gradient=directional*delta
        item.update({"plus_objective":plus,"minus_objective":minus,"directional_derivative":directional,"gradient_norm":float(np.linalg.norm(gradient)),"perturbation":scale,"learning_rate":rate}); history.append(item); p-=rate*gradient
      parameters[step]=p; histories.append(history); checkpoints.append({"initial":initial.copy(),"intermediate":intermediate,"best":best_parameters})
    return QuDDPMTrainingResult(parameters,histories,angles,4,config["ancillas"]),probes,checkpoints


def training_summary(histories,N,R,config,condition):
    calls=3*config["iterations"]+1; kernel_multiplier=1 if condition=="PER_OUTCOME_ENSEMBLE_MMD" else R
    return {"objective_calls_total":2*calls,"source_outcome_evaluations_total":2*calls*2*N*R,"relative_kernel_pair_cost":kernel_multiplier,
      "steps":[{"initial":h[0]["objective"],"best":min(x["objective"] for x in h),"final":h[-1]["objective"],"parameter_update_norm":h[-1]["parameter_update_norm"],"objective_variance":float(np.var([x["objective"] for x in h])),"directional_variance":float(np.var([x["directional_derivative"] for x in h[:-1]])),"gradient_norm_mean":float(np.mean([x["gradient_norm"] for x in h[:-1]]))} for h in histories]}


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); N=config["train_realizations"]; T=config["diffusion_steps"]; seeds=config["seeds"]
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]; val_split=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_split.states[val_split.labels==c] for c in (0,1)}; train_path,train_ids=trajectories(train_states,N,T,seeds["train_forward"]); holdout,holdout_ids=trajectories(train_states,config["holdout_realizations"],T,seeds["holdout_forward"]); validation,val_ids=trajectories(val_states,config["validation_realizations"],T,seeds["validation_forward"])
    result={"provenance":provenance(),"data_access":{"train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_split.parameter_ids.tolist(),"test_evaluated":False},"realizations":{"train":train_ids,"holdout":holdout_ids,"validation":val_ids},"conditions":[]}
    for ci,condition in enumerate(config["conditions"]):
      condition_started=time.perf_counter(); model,probes,_=train(train_path,config,condition); evaluations={}
      for di,(name,path) in enumerate((("seen",train_path),("unseen",holdout),("validation",validation))):
        refs={str(step):{str(c):obs(path[c][step]) for c in (0,1)} for step in (0,1)}; evaluations[name]={"steps":[evaluate_step(model,step,path,path,train_path,refs[str(step)],seeds["evaluation"]+100000*ci+10000*di+100*step,config["evaluation_measurement_outcomes"]) for step in (0,1)],"rollout":rollout(model,path,train_path,refs,seeds["evaluation"]+100000*ci+10000*di,config["evaluation_measurement_outcomes"])}
      gaps={key:{metric:evaluations["unseen"][section][index][field]-evaluations["seen"][section][index][field] for metric,field in (("mmd","aggregate_mmd"),("physics","physics_error"))} for key,section,index in (("isolated_rho1_to_rho0","steps",0),("isolated_rho2_to_rho1","steps",1),("final_rollout","rollout",1))}
      result["conditions"].append({"name":condition,"runtime_seconds":time.perf_counter()-condition_started,"objective_probes":probes,"training_summary":training_summary(model.histories,N,config["measurement_outcomes"],config,condition),"training_history":model.histories,"evaluations":evaluations,"generalization_gaps":gaps})
    result["runtime_seconds"]=time.perf_counter()-started; (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    rows=[]
    for condition in result["conditions"]:
      for domain,e in condition["evaluations"].items():
       for mode,items in (("isolated",e["steps"]),("rollout",e["rollout"])):
        for item in items: rows.append({"condition":condition["name"],"domain":domain,"mode":mode,"step":item["step"],"aggregate_mmd":item["aggregate_mmd"],"physics_error":item["physics_error"],"Mx_order":item["class_order_agreement"]["Mx"],"Mz2_order":item["class_order_agreement"]["Mz2"]})
    with (out/"comparison.csv").open("w",newline="") as f: w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/q2_objective_geometry.yaml"); p.add_argument("--output",default="results/quddpm_q2_objective_geometry"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

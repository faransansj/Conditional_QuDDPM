"""Train-only forward-realization ensemble generalization at frozen T=2."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.underfitting_diagnostics import add_gaps, obs, provenance
from conditional_quddpm.models.quddpm import fidelity_matrix, fidelity_mmd, forward_diffusion, reverse_step, train_single_reverse_steps


def trajectories(states,count,T,base_seed):
    """One independent forward RNG seed per class/state realization."""
    result={}
    ids={}
    for c,state in states.items():
      paths=[forward_diffusion({c:state},T,base_seed+100000*c+i)[c] for i in range(count)]
      result[c]=[np.concatenate([path[t] for path in paths]) for t in range(T+1)]
      ids[str(c)]=[{"realization":i,"forward_seed":base_seed+100000*c+i} for i in range(count)]
    return result,ids


def summarize(generated,target,training_source,training_target):
    pair=fidelity_matrix(generated,generated); off=pair[~np.eye(len(generated),dtype=bool)]
    return {"mmd":fidelity_mmd(generated,target),"observables":obs(generated),
      "nearest_training_source_fidelity":float(fidelity_matrix(generated,training_source).max(axis=1).mean()),
      "nearest_training_target_fidelity":float(fidelity_matrix(generated,training_target).max(axis=1).mean()),
      "nearest_target_fidelity":float(fidelity_matrix(generated,target).max(axis=1).mean()),
      "mean_pairwise_fidelity":float(off.mean()) if len(off) else None}


def evaluate_step(model,step,source_path,target_path,training_path,reference,seed,outcomes):
    rng=np.random.default_rng(seed); per={}
    for c in (0,1):
      source=np.repeat(source_path[c][step+1],outcomes,axis=0); target=np.repeat(target_path[c][step],outcomes,axis=0)
      generated=reverse_step(source,model.parameters[step],model.conditioning[c],rng.random(len(source)))
      per[str(c)]=summarize(generated,target,training_path[c][step+1],training_path[c][step])
    item={"step":step+1,"transition":f"rho_{step+1}->rho_{step}","per_class":per}; add_gaps(item,reference); item["aggregate_mmd"]=float(np.mean([m["mmd"] for m in per.values()])); return item


def rollout(model,path,training_path,reference,seed,outcomes):
    current={c:np.repeat(path[c][2],outcomes,axis=0) for c in (0,1)}; rng=np.random.default_rng(seed); stages=[]
    for step in (1,0):
      current={c:reverse_step(current[c],model.parameters[step],model.conditioning[c],rng.random(len(current[c]))) for c in (0,1)}
      per={str(c):summarize(current[c],np.repeat(path[c][step],outcomes,axis=0),training_path[c][step+1],training_path[c][step]) for c in (0,1)}
      item={"step":step+1,"transition":f"rho_{step+1}->rho_{step}","per_class":per}; add_gaps(item,reference[str(step)]); item["aggregate_mmd"]=float(np.mean([m["mmd"] for m in per.values()])); stages.append(item)
    return stages


def svg(path,rows,key,title):
    w,h,p=640,360,45; xs=[r["N_train"] for r in rows]; ys=[r[key] for r in rows]; xmax=max(xs); ymax=max(ys) or 1
    pts=" ".join(f'{p+x/xmax*(w-2*p):.1f},{h-p-y/ymax*(h-2*p):.1f}' for x,y in zip(xs,ys))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{p}" y="25">{title}</text><line x1="{p}" y1="{h-p}" x2="{w-p}" y2="{h-p}" stroke="black"/><line x1="{p}" y1="{p}" x2="{p}" y2="{h-p}" stroke="black"/><polyline fill="none" stroke="#c00" stroke-width="2" points="{pts}"/></svg>')


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"])
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]
    val_split=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_split.states[val_split.labels==c] for c in (0,1)}; seeds=config["seeds"]; T=config["diffusion_steps"]
    holdout,holdout_ids=trajectories(train_states,config["holdout_realizations"],T,seeds["holdout_forward"]); validation,val_ids=trajectories(val_states,config["validation_realizations"],T,seeds["validation_forward"])
    result={"provenance":provenance(),"data_access":{"train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_split.parameter_ids.tolist(),"test_evaluated":False},
      "semantics":{"forward":"one independent RNG seed and scrambling path per realization","measurement":"8 fixed outcome vectors per training objective; independent evaluation RNG","holdout_overlap_verified":True},"holdout_realization_ids":holdout_ids,"validation_realization_ids":val_ids,"runs":[]}
    for N in config["train_realizations"]:
      train_path,train_ids=trajectories(train_states,N,T,seeds["train_forward"]); expanded={c:train_path[c][0] for c in (0,1)}; fit_started=time.perf_counter()
      model,_,diagnostics=train_single_reverse_steps(expanded,diffusion_steps=T,layers=config["layers"],samples=N,forward_seed=seeds["train_forward"],source_seed=seeds["source"],init_seed=seeds["init"],spsa_seed=seeds["spsa"],measurement_seed=seeds["measurement"],training_steps=config["iterations"],learning_rate=config["spsa"]["learning_rate"],perturbation=config["spsa"]["perturbation"],n_ancilla=config["ancillas"],source_mode="teacher_forced",measurement_repeats=config["measurement_objective_samples"],forward_trajectories=train_path)
      domains={"seen":train_path,"unseen":holdout,"validation":validation}; evaluations={}
      for di,(name,path) in enumerate(domains.items()):
        refs={str(step):{str(c):obs(path[c][step]) for c in (0,1)} for step in (0,1)}
        evaluations[name]={"steps":[evaluate_step(model,step,path,path,train_path,refs[str(step)],config["evaluation_measurement_seed"]+10000*di+100*step+N,config["evaluation_measurement_outcomes"]) for step in (0,1)],
          "rollout":rollout(model,path,train_path,refs,config["evaluation_measurement_seed"]+10000*di+N,config["evaluation_measurement_outcomes"])}
      unseen_final=evaluations["unseen"]["rollout"][-1]; seen_final=evaluations["seen"]["rollout"][-1]
      result["runs"].append({"N_train":N,"train_realization_ids":train_ids,"runtime_seconds":time.perf_counter()-fit_started,"training_diagnostics":diagnostics,"evaluations":evaluations,
        "final_rollout_mmd_gap":unseen_final["aggregate_mmd"]-seen_final["aggregate_mmd"],"final_rollout_physics_gap":unseen_final["physics_error"]-seen_final["physics_error"]})
    result["runtime_seconds"]=time.perf_counter()-started; (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    rows=[]
    for run_item in result["runs"]:
      for domain,e in run_item["evaluations"].items():
       for mode,items in (("isolated",e["steps"]),("rollout",e["rollout"])):
        for item in items: rows.append({"N_train":run_item["N_train"],"domain":domain,"mode":mode,"step":item["step"],"aggregate_mmd":item["aggregate_mmd"],"physics_error":item["physics_error"],"Mx_order":item["class_order_agreement"]["Mx"],"Mz2_order":item["class_order_agreement"]["Mz2"]})
    with (out/"generalization.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)
    plotrows=[{"N_train":r["N_train"],"gap":r["final_rollout_mmd_gap"],"physics":r["evaluations"]["unseen"]["rollout"][-1]["physics_error"]} for r in result["runs"]]
    svg(out/"generalization_gap.svg",plotrows,"gap","unseen-seen final MMD gap"); svg(out/"unseen_final_physics.svg",plotrows,"physics","unseen final physics error")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/q2_ensemble_generalization.yaml"); p.add_argument("--output",default="results/quddpm_q2_ensemble"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

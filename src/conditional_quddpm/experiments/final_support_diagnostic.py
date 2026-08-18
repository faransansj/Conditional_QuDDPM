"""Final bounded support diagnostic: shared reverse map vs per-realization oracles.

Question: can the current reverse architecture/objective fit individual
forward realizations (per-realization oracle controls), and do those fits
generalize to same-state unseen forward realizations, while the shared map
covering N=4 realizations per class fails? Global fidelity MMD only; frozen
T=2/L=3/N=4/R=8 SPSA semantics; train/holdout/validation splits only.
Diagnosis decision rules are pre-registered in the config, not fitted to
results.
"""
from __future__ import annotations
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import verify_checksums
from conditional_quddpm.experiments.kernel_diagnostics import obs, provenance
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.experiments.q2_objective_geometry import per_outcome_ensemble_mmd, train
from conditional_quddpm.models.quddpm import condition_angles, fidelity_matrix, fidelity_mmd, reverse_step

DOMAINS=("seen","unseen","validation")


def oracle_objective(parameters,source,target,angle,uniforms):
    """Same PER_OUTCOME_ENSEMBLE_MMD semantics as the shared model, restricted to one realization."""
    generated=[reverse_step(source,parameters,angle,row) for row in uniforms]
    return per_outcome_ensemble_mmd(generated,target)


def train_oracle(initial,source,target,angle,uniforms,config,spsa_seed):
    """SPSA loop identical to the shared trainer; only the realization scope differs."""
    p=initial.copy(); start=p.copy(); rng=np.random.default_rng(spsa_seed); history=[]; best_value=float("inf"); best=p.copy()
    for iteration in range(config["iterations"]+1):
        value=oracle_objective(p,source,target,angle,uniforms)
        history.append({"iteration":iteration,"objective":value,"parameter_update_norm":float(np.linalg.norm(p-start))})
        if value<best_value: best_value,best=value,p.copy()
        if iteration==config["iterations"]: break
        delta=rng.choice((-1.0,1.0),size=p.shape)
        scale=config["spsa"]["perturbation"]/(iteration+1)**0.101
        rate=config["spsa"]["learning_rate"]/(iteration+1)**0.602
        plus=oracle_objective(p+scale*delta,source,target,angle,uniforms)
        minus=oracle_objective(p-scale*delta,source,target,angle,uniforms)
        p-=rate*((plus-minus)/(2*scale))*delta
    return {"best":best,"history":history,
      "training":{"initial_loss":history[0]["objective"],"best_loss":best_value,"final_loss":history[-1]["objective"],
        "loss_improvement":float((history[0]["objective"]-best_value)/history[0]["objective"]) if history[0]["objective"]>0 else None,
        "parameter_update_norm":history[-1]["parameter_update_norm"]}}


def evaluate_cell(parameters,angle,source,target,repeats,seed):
    """Evaluate one model on identical source/target slices; same seed => same uniforms across models."""
    rng=np.random.default_rng(seed); src=np.repeat(source,repeats,axis=0); tgt=np.repeat(target,repeats,axis=0)
    generated=reverse_step(src,parameters,angle,rng.random(len(src)))
    go,to=obs(generated),obs(tgt)
    return {"mmd":fidelity_mmd(generated,tgt),
      "physics_error":float(0.5*(abs(go["Mx"]-to["Mx"])+abs(go["Mz2"]-to["Mz2"]))),
      "Mx_generated":go["Mx"],"Mx_target":to["Mx"],"Mz2_generated":go["Mz2"],"Mz2_target":to["Mz2"],
      "nearest_target_fidelity":float(fidelity_matrix(generated,tgt).max(axis=1).mean())}


def aggregate(cells,key):
    values=np.asarray([c[key] for c in cells],dtype=float)
    return {"mean":float(values.mean()),"std":float(values.std()),"min":float(values.min()),"max":float(values.max())}


def summarize_step(cells,classes=(0,1)):
    """Aggregate shared-vs-oracle comparison over (class, realization) cells of one step."""
    out={}
    for scope,subset in [("all",cells)]+[(f"class_{c}",[x for x in cells if x["class"]==c]) for c in classes]:
        out[scope]={}
        for model in ("shared","oracle"):
            for domain in DOMAINS:
                for key in ("mmd","physics_error"):
                    out[scope][f"{model}_{domain}_{key}"]=aggregate(subset,f"{model}_{domain}_{key}")
        out[scope]["seen_advantage"]=float(np.mean([x["shared_seen_mmd"]-x["oracle_seen_mmd"] for x in subset]))
        out[scope]["unseen_advantage"]=float(np.mean([x["shared_unseen_mmd"]-x["oracle_unseen_mmd"] for x in subset]))
        out[scope]["unseen_advantage_physics"]=float(np.mean([x["shared_unseen_physics_error"]-x["oracle_unseen_physics_error"] for x in subset]))
        out[scope]["oracle_gap_mmd"]=float(np.mean([x["oracle_unseen_mmd"]-x["oracle_seen_mmd"] for x in subset]))
        out[scope]["shared_gap_mmd"]=float(np.mean([x["shared_unseen_mmd"]-x["shared_seen_mmd"] for x in subset]))
    oracle_cells=[x for x in cells if x["oracle_loss_improvement"] is not None]
    for scope in out:
        subset=cells if scope=="all" else [x for x in cells if x["class"]==int(scope.split("_")[1])]
        out[scope]["oracle_mean_loss_improvement"]=float(np.mean([x["oracle_loss_improvement"] for x in subset])) if oracle_cells else None
    return out


def categorize(summary,thresholds):
    """Pre-registered per-step rule; thresholds come from the config, declared before the run."""
    seen_ratio=float("inf") if summary["shared_seen_mmd"]["mean"]<=0 else summary["oracle_seen_mmd"]["mean"]/summary["shared_seen_mmd"]["mean"]
    fit_ok=bool(seen_ratio<=thresholds["seen_fit_ratio_max"] and summary["oracle_mean_loss_improvement"]>=thresholds["min_loss_improvement"])
    if not fit_ok: return "FUNDAMENTAL REVERSE-LEARNING BOTTLENECK"
    if summary["unseen_advantage"]>=thresholds["unseen_advantage_min"]: return "SHARED SUPPORT BOTTLENECK"
    return "TRAJECTORY MEMORIZATION / SUPPORT GENERALIZATION FAILURE"


def decide(step_summaries,thresholds):
    categories={str(step):categorize(summary["all"],thresholds) for step,summary in step_summaries.items()}
    primary=categories["1"]
    diagnosis=primary if categories.get("2")==primary else "INCONCLUSIVE"
    return {"thresholds":thresholds,"primary":"step 1 (rho1->rho0)","step_categories":categories,"diagnosis":diagnosis,
      "rule":"fit_ok := oracle_seen_mmd <= seen_fit_ratio_max * shared_seen_mmd AND oracle_mean_loss_improvement >= min_loss_improvement; "
        "not fit_ok -> FUNDAMENTAL; fit_ok AND unseen_advantage >= unseen_advantage_min -> SHARED SUPPORT BOTTLENECK; "
        "fit_ok otherwise -> TRAJECTORY MEMORIZATION / SUPPORT GENERALIZATION FAILURE; primary/secondary disagreement -> INCONCLUSIVE"}


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    ds=load_tfim_dataset(config["dataset"]); seeds=config["seeds"]; N=config["train_realizations"]; T=config["diffusion_steps"]; R=config["measurement_outcomes"]
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]
    val_split=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_split.states[val_split.labels==c] for c in (0,1)}
    train_path,train_ids=trajectories(train_states,N,T,seeds["train_forward"])
    holdout,holdout_ids=trajectories(train_states,config["holdout_realizations"],T,seeds["holdout_forward"])
    validation,val_ids=trajectories(val_states,config["validation_realizations"],T,seeds["validation_forward"])
    angles=condition_angles([0,1])
    model,_,checkpoints=train(train_path,config,config["trained_objective"])
    uniforms=[{c:np.random.default_rng(seeds["measurement"]+1000*step+c).random((R,N)) for c in (0,1)} for step in range(T)]
    cells=[]; oracle_runs={}
    for step in range(T):
      for c in (0,1):
        for i in range(N):
          t0=time.perf_counter()
          oracle=train_oracle(checkpoints[step]["initial"],train_path[c][step+1][i:i+1],train_path[c][step][i:i+1],angles[c],uniforms[step][c][:,i:i+1],config,seeds["oracle_spsa"]+1000*step+10*c+i)
          oracle_runs[f"{step}_{c}_{i}"]={"training":oracle["training"],"history":oracle["history"],"runtime_seconds":time.perf_counter()-t0,
            "spsa_seed":seeds["oracle_spsa"]+1000*step+10*c+i,"train_realization_id":train_ids[str(c)][i]}
          cell={"step":step+1,"transition":f"rho_{step+1}->rho_{step}","class":c,"realization":i,
            "oracle_initial_loss":oracle["training"]["initial_loss"],"oracle_best_loss":oracle["training"]["best_loss"],
            "oracle_final_loss":oracle["training"]["final_loss"],"oracle_loss_improvement":oracle["training"]["loss_improvement"],
            "oracle_parameter_update_norm":oracle["training"]["parameter_update_norm"]}
          for di,domain in enumerate(DOMAINS):
            if domain=="seen": source,target=train_path[c][step+1][i:i+1],train_path[c][step][i:i+1]
            elif domain=="unseen": source,target=holdout[c][step+1],holdout[c][step]
            else: source,target=validation[c][step+1],validation[c][step]
            seed=seeds["evaluation"]+10000*di+1000*step+100*c+10*i
            shared_eval=evaluate_cell(checkpoints[step]["best"],angles[c],source,target,config["evaluation_outcomes"],seed)
            oracle_eval=evaluate_cell(oracle["best"],angles[c],source,target,config["evaluation_outcomes"],seed)
            for m,values in (("shared",shared_eval),("oracle",oracle_eval)):
                for key,value in values.items(): cell[f"{m}_{domain}_{key}"]=value
          cells.append(cell)
    step_summaries={str(step+1):summarize_step([x for x in cells if x["step"]==step+1]) for step in range(T)}
    decision=decide(step_summaries,config["decision"])
    shared_training=[{"step":s+1,"initial_objective":h[0]["objective"],"best_objective":min(x["objective"] for x in h),
      "final_objective":h[-1]["objective"],"parameter_update_norm":h[-1]["parameter_update_norm"]} for s,h in enumerate(model.histories)]
    manifest={**provenance(),"dataset":config["dataset"],"dataset_checksums":verify_checksums(config["dataset"]),
      "train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_split.parameter_ids.tolist(),
      "realization_ids":{"train":train_ids,"holdout":holdout_ids,"validation":val_ids},
      "model":{"T":T,"L":config["layers"],"optimizer":"SPSA","iterations":config["iterations"],"objective":config["trained_objective"],
        "measurement_outcomes":R,"train_realizations":N,"evaluation_outcomes":config["evaluation_outcomes"]},
      "configuration_equivalence":{"shared_vs_oracle":"identical L, SPSA learning rate/perturbation/iterations, measurement outcomes, "
        "conditioning angles and step-initial parameters; only the realization scope differs (N=4 shared vs N=1 oracle). "
        "Oracle measurement uniforms are the realization column of the shared training uniforms; oracle SPSA uses a disjoint recorded seed stream.",
        "oracle_init":"shared model step-initial parameters","oracle_spsa_seed_formula":"oracle_spsa + 1000*step + 10*class + realization"},
      "evaluation":{"seen":"training realization slice","unseen":"same train state, holdout forward-noise pool (prefix of the K0 holdout seeds)",
        "validation":"validation state pool (prefix of the K0 validation seeds); secondary evidence","identical_eval_seeds_across_models":True},
      "decision_rules":decision["rule"],"decision_thresholds":config["decision"],
      "seeds":seeds,"test_split_used":False,"dtype":"complex128","versions":{"python":platform.python_version(),"numpy":np.__version__}}
    result={"manifest":manifest,"shared_training":shared_training,"cells":cells,"step_summaries":step_summaries,
      "oracle_runs":oracle_runs,"decision":decision,"runtime_seconds":time.perf_counter()-started}
    manifest["runtime_seconds"]=result["runtime_seconds"]
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    eval_keys=["mmd","physics_error","Mx_generated","Mx_target","Mz2_generated","Mz2_target","nearest_target_fidelity"]
    per_rows=[]
    for cell in cells:
      for m in ("shared","oracle"):
        for domain in DOMAINS:
          row={"step":cell["step"],"transition":cell["transition"],"class":cell["class"],"realization":cell["realization"],"model":m,"domain":domain}
          row.update({key:cell[f"{m}_{domain}_{key}"] for key in eval_keys})
          for k in ("oracle_initial_loss","oracle_best_loss","oracle_final_loss","oracle_loss_improvement","oracle_parameter_update_norm"):
            row[k]=cell[k] if m=="oracle" else None
          per_rows.append(row)
    with (out/"per_realization.csv").open("w",newline="") as f: w=csv.DictWriter(f,per_rows[0].keys()); w.writeheader(); w.writerows(per_rows)
    svo=[]
    for cell in cells:
      row={k:cell[k] for k in ("step","transition","class","realization","oracle_initial_loss","oracle_best_loss","oracle_final_loss","oracle_loss_improvement","oracle_parameter_update_norm")}
      for domain in DOMAINS:
        for key in ("mmd","physics_error"):
          row[f"shared_{domain}_{key}"]=cell[f"shared_{domain}_{key}"]; row[f"oracle_{domain}_{key}"]=cell[f"oracle_{domain}_{key}"]
      row["seen_advantage_mmd"]=cell["shared_seen_mmd"]-cell["oracle_seen_mmd"]
      row["unseen_advantage_mmd"]=cell["shared_unseen_mmd"]-cell["oracle_unseen_mmd"]
      row["seen_advantage_physics"]=cell["shared_seen_physics_error"]-cell["oracle_seen_physics_error"]
      row["unseen_advantage_physics"]=cell["shared_unseen_physics_error"]-cell["oracle_unseen_physics_error"]
      svo.append(row)
    with (out/"shared_vs_oracle.csv").open("w",newline="") as f: w=csv.DictWriter(f,svo[0].keys()); w.writeheader(); w.writerows(svo)
    gap=[]
    for cell in cells:
      for m in ("shared","oracle"):
        gap.append({"step":cell["step"],"transition":cell["transition"],"class":cell["class"],"realization":cell["realization"],"model":m,"scope":"cell",
          "seen_mmd":cell[f"{m}_seen_mmd"],"unseen_mmd":cell[f"{m}_unseen_mmd"],"gap_mmd":cell[f"{m}_unseen_mmd"]-cell[f"{m}_seen_mmd"],
          "seen_physics":cell[f"{m}_seen_physics_error"],"unseen_physics":cell[f"{m}_unseen_physics_error"],"gap_physics":cell[f"{m}_unseen_physics_error"]-cell[f"{m}_seen_physics_error"]})
    for step in (1,2):
      for m in ("shared","oracle"):
        subset=[x for x in cells if x["step"]==step]
        for scope,group in [("overall",subset),("class_0",[x for x in subset if x["class"]==0]),("class_1",[x for x in subset if x["class"]==1])]:
          gap.append({"step":step,"transition":f"rho_{step+1}->rho_{step}","class":"","realization":"","model":m,"scope":scope,
            "seen_mmd":float(np.mean([x[f"{m}_seen_mmd"] for x in group])),"unseen_mmd":float(np.mean([x[f"{m}_unseen_mmd"] for x in group])),
            "gap_mmd":float(np.mean([x[f"{m}_unseen_mmd"]-x[f"{m}_seen_mmd"] for x in group])),
            "seen_physics":float(np.mean([x[f"{m}_seen_physics_error"] for x in group])),"unseen_physics":float(np.mean([x[f"{m}_unseen_physics_error"] for x in group])),
            "gap_physics":float(np.mean([x[f"{m}_unseen_physics_error"]-x[f"{m}_seen_physics_error"] for x in group]))})
    with (out/"generalization_gap.csv").open("w",newline="") as f: w=csv.DictWriter(f,gap[0].keys()); w.writeheader(); w.writerows(gap)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/final_support_diagnostic.yaml")
    p.add_argument("--output",default="results/quddpm_final_support_diagnostic"); a=p.parse_args()
    r=run(yaml.safe_load(Path(a.config).read_text()),a.output)
    print(json.dumps({"runtime_seconds":r["runtime_seconds"],"diagnosis":r["decision"]["diagnosis"],"step_categories":r["decision"]["step_categories"]},indent=2))
if __name__=="__main__": main()

"""Frozen K0 comparison of global, one-RDM, and two-RDM TFIM kernels."""
from __future__ import annotations
import argparse, csv, json, platform, subprocess, time
from pathlib import Path
import numpy as np, scipy, yaml
from scipy.stats import pearsonr,spearmanr
from conditional_quddpm.datasets.loader import load_tfim_dataset,nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_observables
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.experiments.q2_objective_geometry import train
from conditional_quddpm.models.quddpm import X,_apply_gate,reverse_step,rotation
from conditional_quddpm.models.rdm_kernels import kernel_matrix,kernel_mmd


def obs(states):
    values=np.asarray([tfim_observables(s,4) for s in states]); return {"Mx":float(values[:,0].mean()),"Mz2":float(values[:,1].mean())}


def provenance():
    return {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())}


def generated_groups(model,paths,config):
    groups={}; arrays={}; outcomes=config["evaluation_measurement_outcomes"]; base=config["seeds"]["evaluation"]
    for di,(domain,path) in enumerate(paths.items()):
      for step in (0,1):
        rng=np.random.default_rng(base+10000*di+100*step); generated={}; target={}
        for c in (0,1):
          source=np.repeat(path[c][step+1],outcomes,axis=0); target[c]=np.repeat(path[c][step],outcomes,axis=0); generated[c]=reverse_step(source,model.parameters[step],model.conditioning[c],rng.random(len(source)))
        groups[f"{domain}/isolated_step_{step+1}"]=(generated,target)
      rng=np.random.default_rng(base+10000*di); current={c:np.repeat(path[c][2],outcomes,axis=0) for c in (0,1)}
      for step in (1,0):
        current={c:reverse_step(current[c],model.parameters[step],model.conditioning[c],rng.random(len(current[c]))) for c in (0,1)}; target={c:np.repeat(path[c][step],outcomes,axis=0) for c in (0,1)}; groups[f"{domain}/rollout_step_{step+1}"]=({c:current[c].copy() for c in (0,1)},target)
    for name,(generated,target) in groups.items():
      key=name.replace("/","__")
      for c in (0,1): arrays[f"{key}__generated_{c}"]=generated[c]; arrays[f"{key}__target_{c}"]=target[c]
    return groups,arrays


def subset_pair(generated,target,count,seed):
    rng=np.random.default_rng(seed); left=rng.choice(len(generated),min(count,len(generated)),replace=False); right=rng.choice(len(target),min(count,len(target)),replace=False)
    return generated[left],target[right]


def safe_correlation(x,y,method):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return None
    value=pearsonr(x,y).statistic if method=="pearson" else spearmanr(x,y).statistic
    return float(value)


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); seeds=config["seeds"]
    train_split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]; val_one=nested_train_subsets(ds.val,[config["validation_states_per_class"]],config["subset_seed"])[config["validation_states_per_class"]]
    train_states={c:train_split.states[train_split.labels==c] for c in (0,1)}; val_states={c:val_one.states[val_one.labels==c] for c in (0,1)}; train_path,train_ids=trajectories(train_states,config["train_realizations"],2,seeds["train_forward"]); holdout,holdout_ids=trajectories(train_states,config["holdout_realizations"],2,seeds["holdout_forward"]); validation,val_ids=trajectories(val_states,config["validation_realizations"],2,seeds["validation_forward"])
    model,_,_=train(train_path,config,"PER_OUTCOME_ENSEMBLE_MMD"); groups,arrays=generated_groups(model,{"seen":train_path,"unseen":holdout,"validation":validation},config); np.savez_compressed(out/"frozen_states.npz",**arrays)
    separation={}; separation_states={c:ds.val.states[ds.val.labels==c][:config["class_separation_samples_per_class"]] for c in (0,1)}
    for kernel in config["kernels"]:
      within=[]
      for c in (0,1):
        matrix=kernel_matrix(separation_states[c],separation_states[c],kernel); within.append(float(matrix[~np.eye(len(matrix),dtype=bool)].mean()))
      between=float(kernel_matrix(separation_states[0],separation_states[1],kernel).mean()); gram=kernel_matrix(np.concatenate([separation_states[0],separation_states[1]]),np.concatenate([separation_states[0],separation_states[1]]),kernel)
      separation[kernel]={"class0_within":within[0],"class1_within":within[1],"within_mean":float(np.mean(within)),"between":between,"delta":float(np.mean(within)-between),"gram_min_eigenvalue":float(np.linalg.eigvalsh((gram+gram.T)/2).min())}
    rows=[]; correlation_points={kernel:{"mmd":[],"physics":[]} for kernel in config["kernels"]}
    for gi,(name,(generated,target)) in enumerate(groups.items()):
      for c in (0,1):
        left,right=subset_pair(generated[c],target[c],config["kernel_evaluation_samples"],seeds["kernel_subsample"]+1000*gi+c); go,to=obs(left),obs(right); physics=float(np.mean([abs(go[k]-to[k]) for k in ("Mx","Mz2")]))
        for kernel in config["kernels"]:
          mmd=kernel_mmd(left,right,kernel); rows.append({"group":name,"class":c,"kernel":kernel,"mmd":mmd,"physics_error":physics,"generated_Mx":go["Mx"],"target_Mx":to["Mx"],"generated_Mz2":go["Mz2"],"target_Mz2":to["Mz2"]}); correlation_points[kernel]["mmd"].append(mmd); correlation_points[kernel]["physics"].append(physics)
    correlations={kernel:{method:safe_correlation(values["mmd"],values["physics"],method) for method in ("pearson","spearman")} for kernel,values in correlation_points.items()}
    perturb=[]
    for strength in config["perturbation_strengths"]:
      for c in (0,1):
        original=separation_states[c]; perturbed=np.asarray([_apply_gate(state,rotation(X,strength),(0,),4) for state in original]); oo,po=obs(original),obs(perturbed); physics=float(np.mean([abs(oo[k]-po[k]) for k in ("Mx","Mz2")]))
        for kernel in config["kernels"]: perturb.append({"strength":strength,"class":c,"kernel":kernel,"mmd":kernel_mmd(original,perturbed,kernel),"physics_error":physics})
    manifest={**provenance(),"dataset":config["dataset"],"train_ids":train_split.parameter_ids.tolist(),"validation_ids":val_one.parameter_ids.tolist(),"realization_ids":{"train":train_ids,"holdout":holdout_ids,"validation":val_ids},"model":{"T":2,"L":config["layers"],"optimizer":"SPSA","iterations":config["iterations"],"measurement_samples":config["measurement_outcomes"]},"kernels":{"global":"squared pure-state fidelity","1-rdm":"equal mean Uhlmann fidelity over 4 one-qubit RDMs","2-rdm":"equal mean Uhlmann fidelity over 6 two-qubit RDMs","version":1,"note":"mixed-state Uhlmann-fidelity Gram PSD is checked empirically, not assumed"},"dtype":"complex128","test_split_used":False,"versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}}
    result={"manifest":manifest,"class_separation":separation,"physics_correlations":correlations,"comparisons":rows,"perturbation":perturb,"runtime_seconds":time.perf_counter()-started}; manifest["runtime_seconds"]=result["runtime_seconds"]
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n"); (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with (out/"comparison.csv").open("w",newline="") as f: writer=csv.DictWriter(f,rows[0].keys(),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    with (out/"perturbation.csv").open("w",newline="") as f: writer=csv.DictWriter(f,perturb[0].keys(),lineterminator="\n"); writer.writeheader(); writer.writerows(perturb)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/quddpm/kernel_k0.yaml"); parser.add_argument("--output",default="results/quddpm_kernel_diagnostics/frozen"); args=parser.parse_args(); result=run(yaml.safe_load(Path(args.config).read_text()),args.output); print(json.dumps({"runtime_seconds":result["runtime_seconds"],"correlations":result["physics_correlations"]},indent=2))
if __name__=="__main__": main()

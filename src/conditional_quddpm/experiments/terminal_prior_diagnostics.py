"""Forward-terminal prior and bounded reverse coverage diagnostics; never loads test data."""
from __future__ import annotations
import argparse, copy, csv, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.underfitting_diagnostics import add_gaps, metrics, obs, provenance
from conditional_quddpm.models.quddpm import (condition_angles, fidelity_matrix, fidelity_mmd, forward_diffusion,
    haar_states, reverse_parameter_count, reverse_step, train_single_reverse_steps)


def diversity(states):
    matrix=fidelity_matrix(states,states); off=matrix[~np.eye(len(states),dtype=bool)]
    return {"mean_pairwise_fidelity":float(off.mean()),"pairwise_fidelity_std":float(off.std()),
      "purity_mean":float(np.mean(np.sum(abs(states)**2,axis=1)**2))}


def terminal_metrics(terminal,haar):
    result={}
    for c in (0,1):
      cross=fidelity_matrix(terminal[c],haar[c])
      result[str(c)]={"mmd_to_haar":fidelity_mmd(terminal[c],haar[c]),"terminal":diversity(terminal[c]),"haar":diversity(haar[c]),
        "nearest_cross_fidelity_terminal_to_haar":float(cross.max(axis=1).mean()),
        "nearest_cross_fidelity_haar_to_terminal":float(cross.max(axis=0).mean()),"terminal_observables":obs(terminal[c]),"haar_observables":obs(haar[c])}
    result["aggregate_mmd_to_haar"]=float(np.mean([result[str(c)]["mmd_to_haar"] for c in (0,1)]))
    result["terminal_class_mmd"]=fidelity_mmd(terminal[0],terminal[1])
    result["haar_class_mmd"]=fidelity_mmd(haar[0],haar[1])
    result["terminal_gaps"]={k:result["0"]["terminal_observables"][k]-result["1"]["terminal_observables"][k] for k in ("Mx","Mz2")}
    return result


def empirical_mixture(terminal,haar,alpha,count,seed):
    """Sample the convex empirical mixture: choose a q2 or Haar pure state per draw."""
    rng=np.random.default_rng(seed); use_haar=rng.random(count)<alpha; output={}
    for c in (0,1):
      ti=rng.integers(len(terminal[c]),size=count); hi=rng.integers(len(haar[c]),size=count)
      output[c]=np.asarray([haar[c][hi[i]] if use_haar[i] else terminal[c][ti[i]] for i in range(count)])
    return output


def rollout(model,source,forward,train,validation,reference,config,seed):
    current=copy.deepcopy(source); stages=[]; rng=np.random.default_rng(seed)
    for step in (1,0):
      current={c:reverse_step(current[c],model.parameters[step],model.conditioning[c],rng.random(len(current[c]))) for c in (0,1)}
      target={c:forward[c][step] for c in (0,1)}
      per={str(c):metrics(current[c],target[c],validation[c],haar_states(len(current[c]),config["seeds"]["haar"]+c,4)) for c in (0,1)}
      item={"step":step+1,"per_class":per}; add_gaps(item,reference if step==0 else {str(c):obs(forward[c][1]) for c in (0,1)}); stages.append(item)
    return stages


def train_terminal_coverage(base_model,terminal_pool,haar_pool,targets,config):
    """Train only rho2->rho1 on fixed q2/Haar source-outcome pairs; keep rho1->rho0 unchanged."""
    cfg=config; seeds=cfg["seeds"]; labels=(0,1); repeats=cfg["reverse"]["measurement_objective_samples"]
    rng=np.random.default_rng(seeds["coverage"]); alpha=cfg["coverage_training_alpha"]
    sources={}; uniforms={}
    for c in labels:
      choose_haar=rng.random(repeats)<alpha
      sources[c]=np.asarray([haar_pool[c][rng.integers(len(haar_pool[c]))] if flag else terminal_pool[c][rng.integers(len(terminal_pool[c]))] for flag in choose_haar])
      uniforms[c]=rng.random(repeats)
    angles=condition_angles(list(labels)); p=base_model.parameters[1].copy(); initial=p.copy(); local=np.random.default_rng(seeds["spsa"])
    def loss(candidate):
      return float(np.mean([fidelity_mmd(reverse_step(sources[c][i:i+1],candidate,angles[c],uniforms[c][i:i+1]),targets[c]) for c in labels for i in range(repeats)]))
    history=[]
    for iteration in range(cfg["reverse"]["iterations"]+1):
      value=loss(p); history.append({"iteration":iteration,"loss":value,"parameter_update_norm":float(np.linalg.norm(p-initial))})
      if iteration==cfg["reverse"]["iterations"]: break
      delta=local.choice((-1.0,1.0),size=p.shape); scale=cfg["spsa"]["perturbation"]/(iteration+1)**0.101; rate=cfg["spsa"]["learning_rate"]/(iteration+1)**0.602
      plus,minus=loss(p+scale*delta),loss(p-scale*delta); history[-1].update({"loss_plus":plus,"loss_minus":minus,"perturbation":scale,"learning_rate":rate})
      p-=rate*(plus-minus)/(2*scale)*delta
    parameters=base_model.parameters.copy(); parameters[1]=p
    return type(base_model)(parameters,base_model.histories,base_model.conditioning,base_model.n_data,base_model.n_ancilla),history


def svg_plot(path,rows,key,title):
    width,height=640,360; pad=45; xs=[r["alpha"] for r in rows]; ys=[r[key] for r in rows]; ymax=max(ys) or 1
    points=" ".join(f'{pad+x*(width-2*pad):.1f},{height-pad-y/ymax*(height-2*pad):.1f}' for x,y in zip(xs,ys))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="{pad}" y="25">{title}</text><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/><polyline fill="none" stroke="#c00" stroke-width="2" points="{points}"/></svg>')


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"])
    subset=nested_train_subsets(ds.train,[config["tiny_train_states_per_class"]],config["subset_seed"])[config["tiny_train_states_per_class"]]
    train={c:subset.states[subset.labels==c] for c in (0,1)}; validation={c:ds.val.states[ds.val.labels==c] for c in (0,1)}; reference={str(c):obs(train[c]) for c in (0,1)}; seeds=config["seeds"]
    expanded={c:np.repeat(train[c],config["forward_terminal_samples"],axis=0) for c in (0,1)}; forward_study=[]; pools={}
    for T in config["forward_steps"]:
      terminal={c:forward_diffusion({c:expanded[c]},T,seeds["forward"])[c][-1] for c in (0,1)}; haar={c:haar_states(config["forward_terminal_samples"],seeds["haar"]+c,4) for c in (0,1)}
      item={"T":T,**terminal_metrics(terminal,haar)}; forward_study.append(item); pools[T]=(terminal,haar)
    rev=config["reverse"]; model,forward,_=train_single_reverse_steps(train,diffusion_steps=2,layers=rev["layers"],samples=1,forward_seed=seeds["forward"],source_seed=seeds["source"],init_seed=seeds["init"],spsa_seed=seeds["spsa"],measurement_seed=seeds["measurement"],training_steps=rev["iterations"],learning_rate=config["spsa"]["learning_rate"],perturbation=config["spsa"]["perturbation"],n_ancilla=rev["ancillas"],source_mode="teacher_forced",measurement_repeats=rev["measurement_objective_samples"])
    terminal,haar=pools[2]; curves=[]
    for alpha in config["mixture_alphas"]:
      source=empirical_mixture(terminal,haar,alpha,rev["evaluation_samples"],seeds["mixture"]+round(alpha*1000)); stages=rollout(model,source,forward,train,validation,reference,config,seeds["evaluation"]+round(alpha*1000)); final=stages[-1]
      curves.append({"alpha":alpha,"stages":stages,"final_mmd":float(np.mean([m["train_mmd"] for m in final["per_class"].values()])),"final_physics_error":final["physics_error"],"final_ordering":final["class_order_agreement"]})
    coverage_model,history=train_terminal_coverage(model,terminal,haar,{c:forward[c][1] for c in (0,1)},config); coverage=[]
    for alpha in config["mixture_alphas"]:
      source=empirical_mixture(terminal,haar,alpha,rev["evaluation_samples"],seeds["mixture"]+round(alpha*1000)); stages=rollout(coverage_model,source,forward,train,validation,reference,config,seeds["evaluation"]+round(alpha*1000)); final=stages[-1]
      coverage.append({"alpha":alpha,"stages":stages,"final_mmd":float(np.mean([m["train_mmd"] for m in final["per_class"].values()])),"final_physics_error":final["physics_error"],"final_ordering":final["class_order_agreement"]})
    result={"provenance":provenance(),"data_access":{"train_ids":subset.parameter_ids.tolist(),"test_evaluated":False},"sampling_semantics":{"haar":"independent normalized complex Gaussian pure states; same distribution law per class, independent seeds","mixture":"empirical convex mixture via Bernoulli selection of a q2 or Haar pure state per draw","measurement":"8 fixed outcomes averaged in training; 32 sampled outcomes in evaluation"},"forward_study":forward_study,"baseline_mixture_curve":curves,"coverage_training":{"alpha":config["coverage_training_alpha"],"history":history,"curve":coverage},"runtime_seconds":time.perf_counter()-started}
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,rows in (("terminal_distance",forward_study),("mixture_curve",curves),("coverage_curve",coverage)):
      with (out/f"{name}.csv").open("w",newline="") as f:
       flat=[]
       if name=="terminal_distance": flat=[{"T":r["T"],"aggregate_mmd_to_haar":r["aggregate_mmd_to_haar"],"terminal_class_mmd":r["terminal_class_mmd"],"Mx_gap":r["terminal_gaps"]["Mx"],"Mz2_gap":r["terminal_gaps"]["Mz2"]} for r in rows]
       else: flat=[{"alpha":r["alpha"],"final_mmd":r["final_mmd"],"final_physics_error":r["final_physics_error"],"Mx_order":r["final_ordering"]["Mx"],"Mz2_order":r["final_ordering"]["Mz2"]} for r in rows]
       w=csv.DictWriter(f,flat[0].keys()); w.writeheader(); w.writerows(flat)
    svg_plot(out/"alpha_final_mmd.svg",curves,"final_mmd","alpha vs final MMD"); svg_plot(out/"alpha_final_physics_error.svg",curves,"final_physics_error","alpha vs final physics error")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/terminal_prior_diagnostics.yaml"); p.add_argument("--output",default="results/quddpm_terminal_prior"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

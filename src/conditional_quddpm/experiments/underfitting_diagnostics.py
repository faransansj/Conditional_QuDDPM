"""Bounded T=1, train/validation-only QuDDPM learnability diagnostics."""
from __future__ import annotations
import argparse, csv, json, platform, subprocess, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.datasets.tfim import tfim_observables
from conditional_quddpm.models.quddpm import fidelity_matrix, fidelity_mmd, haar_states, reverse_step, train_single_reverse_steps


def obs(states):
    x = np.asarray([tfim_observables(s, 4) for s in states])
    return {"Mx": float(x[:, 0].mean()), "Mz2": float(x[:, 1].mean())}


def provenance():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return {"git_sha": sha, "git_dirty": dirty, "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(), "python": platform.python_version()}


def metrics(generated, train, validation, haar):
    pair = fidelity_matrix(generated, generated)
    offdiag = pair[~np.eye(len(generated), dtype=bool)]
    return {"train_mmd": fidelity_mmd(generated, train), "validation_mmd": fidelity_mmd(generated, validation),
        "near_haar_mmd": fidelity_mmd(generated, haar), "haar_validation_mmd": fidelity_mmd(haar, validation),
        "pairwise_generated_fidelity": float(offdiag.mean()) if len(offdiag) else None,
        "nearest_train_fidelity": float(fidelity_matrix(generated, train).max(axis=1).mean()),
        "observables": obs(generated), "max_norm_error": float(np.max(np.abs(np.sum(abs(generated)**2, axis=1)-1)))}


def add_gaps(run, reference):
    generated = {name: run["per_class"]["0"]["observables"][name] - run["per_class"]["1"]["observables"][name] for name in ("Mx", "Mz2")}
    expected = {name: reference["0"][name] - reference["1"][name] for name in ("Mx", "Mz2")}
    run["generated_gaps"] = generated
    run["real_gaps"] = expected
    run["gap_absolute_error"] = {name: abs(generated[name] - expected[name]) for name in expected}
    run["class_order_agreement"] = {name: bool(np.sign(generated[name]) == np.sign(expected[name])) for name in expected}
    run["physics_error"] = float(np.mean(list(run["gap_absolute_error"].values())))


def fit(train, validation, reference, config, *, name, labels, layers, iterations, source_mode, optimizer="spsa"):
    targets = {c: train[c] for c in labels}; seeds=config["seeds"]; started=time.perf_counter()
    model, forward, diagnostics = train_single_reverse_steps(targets, diffusion_steps=1, layers=layers, samples=len(next(iter(targets.values()))),
        forward_seed=seeds["forward"], source_seed=seeds["source"], init_seed=seeds["init"], spsa_seed=seeds["spsa"],
        measurement_seed=seeds["measurement"], training_steps=iterations, learning_rate=config["spsa"]["learning_rate"],
        perturbation=config["spsa"]["perturbation"], n_ancilla=config["ancillas"], optimizer=optimizer, source_mode=source_mode)
    rng=np.random.default_rng(seeds["measurement"])
    per_class={}
    for c in labels:
        source = haar_states(len(train[c]), seeds["source"]+c, 4) if source_mode=="haar" else forward[c][1]
        generated = reverse_step(source, model.parameters[0], model.conditioning[c], rng.random(len(train[c])))
        per_class[str(c)] = metrics(generated, train[c], validation[c], haar_states(config["haar_samples"], seeds["haar"]+c, 4))
    history=model.histories[0]
    run={"name":name,"T":1,"L":layers,"iterations":iterations,"optimizer":optimizer,"source_mode":source_mode,"labels":labels,
        "conditioning":{str(k):v for k,v in model.conditioning.items()},"runtime_seconds":time.perf_counter()-started,
        "initial_loss":history[0]["loss"],"final_loss":history[-1]["loss"],"best_loss":min(x["loss"] for x in history),
        "parameter_update_norm":history[-1].get("parameter_update_norm",0.0),"loss_trajectory":history,"per_class":per_class,"step_diagnostics":diagnostics}
    if labels == [0,1]: add_gaps(run, reference)
    else:
        c=str(labels[0]); run["observable_absolute_error"]={k:abs(per_class[c]["observables"][k]-reference[c][k]) for k in ("Mx","Mz2")}
        run["physics_error"]=float(np.mean(list(run["observable_absolute_error"].values())))
    return run


def correlation(runs, x, y):
    a=np.asarray([r[x] for r in runs]); b=np.asarray([r[y] for r in runs])
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def run(config, output):
    if config["diffusion_steps"] != 1: raise ValueError("this campaign is intentionally frozen at T=1")
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    ds=load_tfim_dataset(config["dataset"]); size=config["tiny_train_states_per_class"]
    subset=nested_train_subsets(ds.train,[size],config["subset_seed"])[size]
    train={c:subset.states[subset.labels==c] for c in (0,1)}; validation={c:ds.val.states[ds.val.labels==c] for c in (0,1)}
    reference={str(c):obs(train[c]) for c in (0,1)}
    result={"provenance":provenance(),"experiment":{"dataset":config["dataset"],"train_subset_ids":subset.parameter_ids.tolist(),
      "test_evaluated":False,"reference":"selected train subset","reference_observables":reference,"config":config},"runs":[]}
    for budget in config["spsa_iteration_budgets"]:
      for source in ("teacher_forced","haar"):
        result["runs"].append(fit(train,validation,reference,config,name="iteration_progression",labels=[0,1],layers=3,iterations=budget,source_mode=source))
    budget=config["capacity_iteration_budget"]
    for layers in config["capacity_layers"]:
      if layers == 3: continue
      for source in ("teacher_forced","haar"):
        result["runs"].append(fit(train,validation,reference,config,name="capacity_progression",labels=[0,1],layers=layers,iterations=budget,source_mode=source))
    selected_layers=config["conditioning_layers"]
    for c in (0,1):
      for source in ("teacher_forced","haar"):
        result["runs"].append(fit(train,validation,reference,config,name="single_class_ablation",labels=[c],layers=selected_layers,iterations=budget,source_mode=source))
    if config.get("optimizer_control",{}).get("enabled"):
      ctl=config["optimizer_control"]
      result["runs"].append(fit(train,validation,reference,config,name="optimizer_control",labels=[0,1],layers=ctl["layers"],iterations=ctl["iterations"],source_mode="teacher_forced",optimizer="lbfgs"))
    conditional=[r for r in result["runs"] if r["labels"]==[0,1]]
    result["analysis"]={"loss_improvement_vs_physics_error_correlation":correlation([{**r,"loss_improvement":r["initial_loss"]-r["best_loss"]} for r in conditional],"loss_improvement","physics_error"),
      "best_by_physics_error":min(conditional,key=lambda r:r["physics_error"])["name"]+":"+str(min(conditional,key=lambda r:r["physics_error"])["L"])+":"+str(min(conditional,key=lambda r:r["physics_error"])["iterations"])+":"+min(conditional,key=lambda r:r["physics_error"])["source_mode"]}
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"summary.csv").open("w",newline="") as f:
      fields=["name","T","L","iterations","optimizer","source_mode","labels","runtime_seconds","initial_loss","final_loss","best_loss","parameter_update_norm","physics_error"]
      w=csv.DictWriter(f,fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in result["runs"])
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/underfitting_diagnostics.yaml"); p.add_argument("--output",default="results/quddpm_underfitting_diagnostics"); a=p.parse_args()
    print(json.dumps(run(yaml.safe_load(Path(a.config).read_text()),a.output)["analysis"],indent=2))
if __name__=="__main__": main()

"""Directional loss-signal decomposition for frozen T=2/N=4 reverse maps."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.experiments.q2_objective_geometry import combined_ensemble_mmd, per_outcome_ensemble_mmd, train
from conditional_quddpm.experiments.underfitting_diagnostics import provenance
from conditional_quddpm.models.quddpm import condition_angles, fidelity_mmd, reverse_parameter_count, reverse_step


def correlation(left,right):
    return float(np.corrcoef(left,right)[0,1]) if np.std(left)>0 and np.std(right)>0 else None


def alignment(signatures):
    count=len(signatures); corr=[]; agree=[]; conflict=[]
    for i in range(count):
      corr.append([]); agree.append([]); conflict.append([])
      for j in range(count):
        left,right=np.asarray(signatures[i]),np.asarray(signatures[j])
        corr[i].append(correlation(left,right)); agree[i].append(float(np.mean(np.sign(left)==np.sign(right)))); conflict[i].append(float(np.mean(left*right<0)))
    return {"correlation":corr,"sign_agreement":agree,"conflict_rate":conflict}


def loss_parts(parameters,path,step,angles,uniforms,objective_name):
    class_losses={}; realization_losses={}; outcome_losses=[]
    for c in (0,1):
      generated=[reverse_step(path[c][step+1],parameters,angles[c],row) for row in uniforms[c]]
      class_losses[c]=per_outcome_ensemble_mmd(generated,path[c][step]) if objective_name=="PER_OUTCOME_ENSEMBLE_MMD" else combined_ensemble_mmd(generated,path[c][step])
      realization_losses[c]=[]
      for i in range(len(path[c][step])):
        singleton=[g[i:i+1] for g in generated]
        realization_losses[c].append(per_outcome_ensemble_mmd(singleton,path[c][step][i:i+1]) if objective_name=="PER_OUTCOME_ENSEMBLE_MMD" else combined_ensemble_mmd(singleton,path[c][step][i:i+1]))
      per_outcome=[fidelity_mmd(generated[r],path[c][step]) for r in range(len(generated))]
      outcome_losses.append(per_outcome)
    return {"aggregate":float(np.mean(list(class_losses.values()))),"class":class_losses,"realization":realization_losses,"outcome_components":np.mean(outcome_losses,axis=0).tolist()}


def analyze_checkpoint(parameters,path,step,angles,uniforms,directions,epsilon,threshold,objective_name):
    center=loss_parts(parameters,path,step,angles,uniforms,objective_name); rows=[]; class_d={0:[],1:[]}; realization_d={c:[[] for _ in range(len(path[c][step]))] for c in (0,1)}; pair_d=[[] for _ in range(8)]; aggregate=[]; curvature=[]; outcome_noise=[]; descent=[]
    for index,delta in enumerate(directions):
      plus=loss_parts(parameters+epsilon*delta,path,step,angles,uniforms,objective_name); minus=loss_parts(parameters-epsilon*delta,path,step,angles,uniforms,objective_name); d=(plus["aggregate"]-minus["aggregate"])/(2*epsilon); curv=plus["aggregate"]+minus["aggregate"]-2*center["aggregate"]
      aggregate.append(d); curvature.append(curv); descent.append(min(plus["aggregate"],minus["aggregate"])<center["aggregate"])
      outcome_derivatives=(np.asarray(plus["outcome_components"])-np.asarray(minus["outcome_components"]))/(2*epsilon); outcome_noise.append(float(np.std(outcome_derivatives)))
      for c in (0,1):
        cd=(plus["class"][c]-minus["class"][c])/(2*epsilon); class_d[c].append(cd)
        for i in range(4):
          rd=(plus["realization"][c][i]-minus["realization"][c][i])/(2*epsilon); realization_d[c][i].append(rd); pair_d[4*c+i].append(rd)
      rows.append({"direction":index,"derivative":d,"curvature":curv,"outcome_noise":outcome_noise[-1],"class0_derivative":class_d[0][-1],"class1_derivative":class_d[1][-1]})
    a=np.asarray(aggregate); class0=np.asarray(class_d[0]); class1=np.asarray(class_d[1]); noise=float(np.mean(outcome_noise))
    return {"center_loss":center["aggregate"],"mean_abs_derivative":float(np.mean(abs(a))),"median_abs_derivative":float(np.median(abs(a))),"derivative_std":float(np.std(a)),"near_zero_fraction":float(np.mean(abs(a)<threshold)),"descent_fraction":float(np.mean(descent)),"curvature_mean":float(np.mean(curvature)),"curvature_abs_mean":float(np.mean(abs(np.asarray(curvature)))),"outcome_noise_proxy":noise,"snr_proxy":float(np.mean(abs(a))/noise) if noise else None,
      "class_alignment":{"correlation":correlation(class0,class1),"sign_agreement":float(np.mean(np.sign(class0)==np.sign(class1))),"conflict_rate":float(np.mean(class0*class1<0)),"class0_mean_abs":float(np.mean(abs(class0))),"class1_mean_abs":float(np.mean(abs(class1)))},
      "realization_alignment":{str(c):alignment(realization_d[c]) for c in (0,1)},"class_realization_alignment":alignment(pair_d),"directions":rows}


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True); ds=load_tfim_dataset(config["dataset"]); split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]
    states={c:split.states[split.labels==c] for c in (0,1)}; path,ids=trajectories(states,config["train_realizations"],config["diffusion_steps"],config["seeds"]["train_forward"]); angles=condition_angles([0,1]); R=config["measurement_outcomes"]; N=config["train_realizations"]
    uniforms_by_step=[{c:np.random.default_rng(config["seeds"]["measurement"]+1000*step+c).random((R,N)) for c in (0,1)} for step in range(2)]
    shape=(config["layers"],reverse_parameter_count(4,config["ancillas"])); rng=np.random.default_rng(config["seeds"]["directions"]); directions=[rng.choice((-1.0,1.0),size=shape) for _ in range(config["directions"])]
    result={"provenance":provenance(),"data_access":{"train_ids":split.parameter_ids.tolist(),"test_evaluated":False},"realization_ids":ids,"method":{"directions":config["directions"],"epsilon":config["epsilon"],"common_directions":True,"common_plus_minus_randomness":True,"snr_proxy":"mean absolute aggregate directional derivative / mean std of per-outcome component derivatives"},"trained_models":[]}
    flat=[]
    for trained_objective in config["objectives"]:
      model,_,checkpoints=train(path,config,trained_objective); analyses=[]
      for step in range(2):
       for checkpoint_name in ("initial","intermediate","best"):
        for evaluated_objective in config["objectives"]:
          analysis=analyze_checkpoint(checkpoints[step][checkpoint_name],path,step,angles,uniforms_by_step[step],directions,config["epsilon"],config["near_zero_threshold"],evaluated_objective)
          analyses.append({"step":step+1,"transition":f"rho_{step+1}->rho_{step}","checkpoint":checkpoint_name,"evaluated_objective":evaluated_objective,"metrics":analysis})
          flat.append({"trained_objective":trained_objective,"step":step+1,"checkpoint":checkpoint_name,"evaluated_objective":evaluated_objective,**{k:analysis[k] for k in ("center_loss","mean_abs_derivative","median_abs_derivative","derivative_std","near_zero_fraction","descent_fraction","curvature_mean","curvature_abs_mean","outcome_noise_proxy","snr_proxy")},**{f"class_{k}":v for k,v in analysis["class_alignment"].items()}})
      result["trained_models"].append({"trained_objective":trained_objective,"training_history":model.histories,"analyses":analyses})
    result["runtime_seconds"]=time.perf_counter()-started; (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"signal_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,flat[0].keys()); w.writeheader(); w.writerows(flat)
    direction_rows=[]
    for model in result["trained_models"]:
      for item in model["analyses"]:
       for row in item["metrics"]["directions"]: direction_rows.append({"trained_objective":model["trained_objective"],"step":item["step"],"checkpoint":item["checkpoint"],"evaluated_objective":item["evaluated_objective"],**row})
    with (out/"directions.csv").open("w",newline="") as f: w=csv.DictWriter(f,direction_rows[0].keys()); w.writeheader(); w.writerows(direction_rows)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/q2_gradient_signal.yaml"); p.add_argument("--output",default="results/quddpm_q2_gradient_signal"); a=p.parse_args(); r=run(yaml.safe_load(Path(a.config).read_text()),a.output); print(json.dumps({"runtime_seconds":r["runtime_seconds"]},indent=2))
if __name__=="__main__": main()

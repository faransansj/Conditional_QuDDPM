"""K1: directional-gradient alignment of global vs 2-RDM MMD at frozen checkpoints.

Reuses the q2 controlled directional-derivative semantics: shared Rademacher
directions, identical theta +/- epsilon*delta, identical source states and
measurement uniforms, and the same deterministic global-MMD training
checkpoints. Derivatives are computed from raw unclipped MMD values; clipping
is only accounted for, never silently applied. This is a directional-gradient
diagnostic, not an exact full-gradient calculation.
"""
from __future__ import annotations
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np, yaml
from conditional_quddpm.datasets.loader import load_tfim_dataset, nested_train_subsets
from conditional_quddpm.experiments.kernel_diagnostics import obs, provenance
from conditional_quddpm.experiments.q2_ensemble_generalization import trajectories
from conditional_quddpm.experiments.q2_gradient_signal import alignment, correlation
from conditional_quddpm.experiments.q2_objective_geometry import train
from conditional_quddpm.models.quddpm import condition_angles, reverse_parameter_count, reverse_step
from conditional_quddpm.models.rdm_kernels import kernel_matrix, kernel_mmd_raw


def cosine(left,right):
    a=np.asarray(left); b=np.asarray(right); norm=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/norm) if norm>0 else None


def clipping_accounting(raw_values):
    a=np.asarray(raw_values,dtype=float); below=a<0; deep=a<-1e-8
    return {"count":int(a.size),"min_raw":float(a.min()) if a.size else None,
      "below_zero_count":int(below.sum()),"below_zero_fraction":float(below.mean()) if a.size else 0.0,
      "below_neg1e8_count":int(deep.sum()),"below_neg1e8_fraction":float(deep.mean()) if a.size else 0.0,
      "clipped_count":int(below.sum()),"clipped_fraction":float(below.mean()) if a.size else 0.0}


def ensemble_objectives(parameters,path,step,angles,uniforms,kernels,raw_sink=None):
    """Per-outcome ensemble objectives for every kernel plus physics error from one shared generated set."""
    class_losses={k:{} for k in kernels}; realization_losses={k:{} for k in kernels}; physics={}
    for c in (0,1):
        generated=[reverse_step(path[c][step+1],parameters,angles[c],row) for row in uniforms[c]]
        target=path[c][step]; to=obs(target)
        physics[c]=float(np.mean([0.5*((obs(g)["Mx"]-to["Mx"])**2+(obs(g)["Mz2"]-to["Mz2"])**2) for g in generated]))
        for k in kernels:
            per_outcome=[kernel_mmd_raw(g,target,k) for g in generated]
            class_losses[k][c]=float(np.mean(per_outcome))
            reals=[]
            for i in range(len(target)):
                values=[kernel_mmd_raw(g[i:i+1],target[i:i+1],k) for g in generated]
                reals.append(float(np.mean(values)))
                if raw_sink is not None: raw_sink[k].extend(values)
            realization_losses[k][c]=reals
            if raw_sink is not None: raw_sink[k].extend(per_outcome)
    return {"aggregate":{k:float(np.mean(list(class_losses[k].values()))) for k in kernels},
      "class":class_losses,"realization":realization_losses,
      "physics":{"aggregate":float(np.mean(list(physics.values()))),"class":physics}}


def gram_min_eigenvalues(parameters,path,step,angles,uniforms,kernels):
    result={}
    for k in kernels:
        per_class={}
        for c in (0,1):
            generated=np.concatenate([reverse_step(path[c][step+1],parameters,angles[c],row) for row in uniforms[c]])
            states=np.concatenate([generated,path[c][step]]); gram=kernel_matrix(states,states,k)
            per_class[str(c)]=float(np.linalg.eigvalsh((gram+gram.T)/2).min())
        result[k]={"per_class":per_class,"min":float(min(per_class.values()))}
    return result


def _signal_stats(derivatives,threshold,descent):
    a=np.asarray(derivatives)
    return {"mean_abs_derivative":float(np.mean(np.abs(a))),"median_abs_derivative":float(np.median(np.abs(a))),
      "derivative_std":float(np.std(a)),"near_zero_fraction":float(np.mean(np.abs(a)<threshold)),
      "descent_fraction":float(np.mean(descent))}


def _compare(dg,dr,dp,dclass,dphys_class):
    dg,dr,dp=np.asarray(dg),np.asarray(dr),np.asarray(dp)
    def versus(d):
        return {"correlation":correlation(d,dp),"cosine":cosine(d,dp),"beneficial_direction_fraction":float(np.mean(d*dp>0))}
    gvp,rvp=versus(dg),versus(dr)
    gain_corr=None if gvp["correlation"] is None or rvp["correlation"] is None else rvp["correlation"]-gvp["correlation"]
    gain_cos=None if gvp["cosine"] is None or rvp["cosine"] is None else rvp["cosine"]-gvp["cosine"]
    per_class={}
    for k,dc in dclass.items():
        per_class[k]={str(c):{"correlation":correlation(dc[c],dphys_class[c]),"cosine":cosine(dc[c],dphys_class[c]),
          "beneficial_direction_fraction":float(np.mean(np.asarray(dc[c])*np.asarray(dphys_class[c])>0))} for c in (0,1)}
    return {"global_vs_2rdm":{"correlation":correlation(dg,dr),"cosine":cosine(dg,dr)},
      "global_vs_physics":gvp,"2rdm_vs_physics":rvp,
      "alignment_gain_correlation":gain_corr,"alignment_gain_cosine":gain_cos,
      "signal_ratio_2rdm_over_global":float(np.mean(np.abs(dr))/np.mean(np.abs(dg))) if np.mean(np.abs(dg))>0 else None,
      "per_class":per_class}


def analyze_checkpoint(parameters,path,step,angles,uniforms,directions,epsilon,threshold,kernels):
    raw={k:[] for k in kernels}
    center=ensemble_objectives(parameters,path,step,angles,uniforms,kernels,raw)
    d={k:[] for k in kernels}; class_d={k:{0:[],1:[]} for k in kernels}
    real_d={k:{c:[[] for _ in range(len(path[c][step]))] for c in (0,1)} for k in kernels}
    dphys=[]; dphys_class={0:[],1:[]}; descent={k:[] for k in kernels}; rows=[]
    for index,delta in enumerate(directions):
        plus=ensemble_objectives(parameters+epsilon*delta,path,step,angles,uniforms,kernels,raw)
        minus=ensemble_objectives(parameters-epsilon*delta,path,step,angles,uniforms,kernels,raw)
        row={"direction":index}
        for k in kernels:
            dv=(plus["aggregate"][k]-minus["aggregate"][k])/(2*epsilon); d[k].append(dv)
            descent[k].append(bool(min(plus["aggregate"][k],minus["aggregate"][k])<center["aggregate"][k]))
            row[f"derivative_{k}"]=dv
            for c in (0,1):
                cd=(plus["class"][k][c]-minus["class"][k][c])/(2*epsilon); class_d[k][c].append(cd); row[f"class{c}_{k}"]=cd
                for i in range(len(path[c][step])):
                    rd=(plus["realization"][k][c][i]-minus["realization"][k][c][i])/(2*epsilon); real_d[k][c][i].append(rd); row[f"real{c}{i}_{k}"]=rd
        dp=(plus["physics"]["aggregate"]-minus["physics"]["aggregate"])/(2*epsilon); dphys.append(dp); row["derivative_physics"]=dp
        for c in (0,1):
            cp=(plus["physics"]["class"][c]-minus["physics"]["class"][c])/(2*epsilon); dphys_class[c].append(cp); row[f"class{c}_physics"]=cp
        rows.append(row)
    kernel_results={}
    for k in kernels:
        class0,class1=np.asarray(class_d[k][0]),np.asarray(class_d[k][1])
        pair_d=[real_d[k][c][i] for c in (0,1) for i in range(len(path[c][step]))]
        kernel_results[k]={"center_raw":center["aggregate"][k],"center_clipped":float(max(center["aggregate"][k],0.0)),
          **_signal_stats(d[k],threshold,descent[k]),
          "class_alignment":{"correlation":correlation(class0,class1),"sign_agreement":float(np.mean(np.sign(class0)==np.sign(class1))),
            "conflict_rate":float(np.mean(class0*class1<0)),"class0_mean":float(class0.mean()),"class1_mean":float(class1.mean()),
            "class0_mean_abs":float(np.mean(np.abs(class0))),"class1_mean_abs":float(np.mean(np.abs(class1)))},
          "realization_alignment":{str(c):alignment(real_d[k][c]) for c in (0,1)},
          "class_realization_alignment":alignment(pair_d),
          "all_class_realization_conflict_rate":float(np.mean(alignment(pair_d)["conflict_rate"])),
          "clipping":clipping_accounting(raw[k]),"gram_min_eigenvalue":gram_min_eigenvalues(parameters,path,step,angles,uniforms,[k])[k]}
    return {"kernels":kernel_results,
      "physics":{"center":center["physics"]["aggregate"],**{key:_signal_stats(dphys,threshold,[True]*len(dphys))[key] for key in ("mean_abs_derivative","median_abs_derivative","derivative_std")}},
      "comparison":_compare(d[kernels[0]],d[kernels[1]],dphys,{k:class_d[k] for k in kernels},dphys_class) if len(kernels)==2 else None,
      "directions":rows}


def _find(analyses,step,checkpoint):
    return next(a for a in analyses if a["step"]==step and a["checkpoint"]==checkpoint)


def _alignment_gain(comparison):
    gains=[g for g in (comparison["alignment_gain_correlation"],comparison["alignment_gain_cosine"]) if g is not None]
    return max(gains) if gains else None


def decide(analyses,thresholds):
    """Fixed decision rule; thresholds are read from the config, not fitted post hoc."""
    primary=_find(analyses,1,"best"); g=primary["kernels"]["global"]; r=primary["kernels"]["2-rdm"]; comp=primary["comparison"]
    comp["conflict_reduction_all_class_realization"]=g["all_class_realization_conflict_rate"]-r["all_class_realization_conflict_rate"]
    gain=_alignment_gain(comp)
    def conflict_ok(a): return a["comparison"]["conflict_reduction_all_class_realization"]>-thresholds["consistency_conflict_tolerance"]
    def align_ok(a):
        ga=_alignment_gain(a["comparison"]); return ga is None or ga>-thresholds["consistency_alignment_tolerance"]
    controls=[_find(analyses,1,"initial"),_find(analyses,2,"best")]
    for a in controls:
        a["comparison"]["conflict_reduction_all_class_realization"]=a["kernels"]["global"]["all_class_realization_conflict_rate"]-a["kernels"]["2-rdm"]["all_class_realization_conflict_rate"]
    checks={
      "validity_min_raw":{"value":r["clipping"]["min_raw"],"pass":bool(r["clipping"]["min_raw"]>=thresholds["validity_min_raw"])},
      "validity_clipped_fraction":{"value":r["clipping"]["clipped_fraction"],"threshold":thresholds["validity_max_clipped_fraction"],"pass":bool(r["clipping"]["clipped_fraction"]<=thresholds["validity_max_clipped_fraction"])},
      "usable_signal_ratio":{"value":comp["signal_ratio_2rdm_over_global"],"threshold":thresholds["usable_min_signal_ratio"],"pass":bool(comp["signal_ratio_2rdm_over_global"] is not None and comp["signal_ratio_2rdm_over_global"]>=thresholds["usable_min_signal_ratio"])},
      "usable_near_zero":{"value":r["near_zero_fraction"],"reference":g["near_zero_fraction"],"slack":thresholds["usable_near_zero_slack"],"pass":bool(r["near_zero_fraction"]<=g["near_zero_fraction"]+thresholds["usable_near_zero_slack"])},
      "conflict_reduction":{"value":comp["conflict_reduction_all_class_realization"],"threshold":thresholds["conflict_min_reduction"],"pass":bool(comp["conflict_reduction_all_class_realization"]>=thresholds["conflict_min_reduction"])},
      "physics_alignment_gain":{"value":gain,"threshold":thresholds["physics_min_alignment_gain"],"pass":bool(gain is not None and gain>=thresholds["physics_min_alignment_gain"])},
      "beneficial_not_lower":{"value":comp["2rdm_vs_physics"]["beneficial_direction_fraction"],"reference":comp["global_vs_physics"]["beneficial_direction_fraction"],"pass":bool(comp["2rdm_vs_physics"]["beneficial_direction_fraction"]>=comp["global_vs_physics"]["beneficial_direction_fraction"])},
      "consistency":{"controls":[{"step":a["step"],"checkpoint":a["checkpoint"],"conflict_change":a["comparison"]["conflict_reduction_all_class_realization"],"alignment_gain":_alignment_gain(a["comparison"])} for a in controls],"pass":bool(all(conflict_ok(a) and align_ok(a) for a in controls))}}
    return {"primary":"step 1 (rho1->rho0), best checkpoint","decision":"GO" if all(c["pass"] for c in checks.values()) else "NO-GO","checks":checks}


def run(config,output):
    started=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    ds=load_tfim_dataset(config["dataset"]); seeds=config["seeds"]; R=config["measurement_outcomes"]; N=config["train_realizations"]
    split=nested_train_subsets(ds.train,[config["train_states_per_class"]],config["subset_seed"])[config["train_states_per_class"]]
    states={c:split.states[split.labels==c] for c in (0,1)}
    path,ids=trajectories(states,N,config["diffusion_steps"],seeds["train_forward"])
    angles=condition_angles([0,1])
    uniforms_by_step=[{c:np.random.default_rng(seeds["measurement"]+1000*step+c).random((R,N)) for c in (0,1)} for step in range(config["diffusion_steps"])]
    shape=(config["layers"],reverse_parameter_count(4,config["ancillas"]))
    rng=np.random.default_rng(seeds["directions"]); directions=[rng.choice((-1.0,1.0),size=shape) for _ in range(config["directions"])]
    model,_,checkpoints=train(path,config,config["trained_objective"])
    analyses=[]; clipping_rows=[]; direction_rows=[]
    for step in range(config["diffusion_steps"]):
        for checkpoint_name in config["checkpoints"]:
            analysis=analyze_checkpoint(checkpoints[step][checkpoint_name],path,step,angles,uniforms_by_step[step],directions,config["epsilon"],config["near_zero_threshold"],config["kernels"])
            analysis["step"]=step+1; analysis["transition"]=f"rho_{step+1}->rho_{step}"; analysis["checkpoint"]=checkpoint_name
            analyses.append(analysis)
            for k in config["kernels"]:
                clipping_rows.append({"step":step+1,"checkpoint":checkpoint_name,"kernel":k,**analysis["kernels"][k]["clipping"],"gram_min_eigenvalue":analysis["kernels"][k]["gram_min_eigenvalue"]["min"]})
            for row in analysis["directions"]: direction_rows.append({"step":step+1,"checkpoint":checkpoint_name,**row})
    training=[{"step":s+1,"checkpoint_identity":{"initial":"SPSA iteration 0 parameters (init seed)","best":"lowest-objective parameters over the deterministic SPSA run"},
      "initial_objective":h[0]["objective"],"best_objective":min(x["objective"] for x in h),
      "best_iteration":int(min(range(len(h)),key=lambda i:h[i]["objective"])),"final_objective":h[-1]["objective"]} for s,h in enumerate(model.histories)]
    decision=decide(analyses,config["decision"])
    manifest={**provenance(),"dataset":config["dataset"],"train_ids":split.parameter_ids.tolist(),"realization_ids":ids,
      "model":{"T":config["diffusion_steps"],"L":config["layers"],"optimizer":"SPSA","iterations":config["iterations"],"trained_objective":config["trained_objective"],"measurement_outcomes":R,"train_realizations":N},
      "checkpoints":config["checkpoints"],"kernels":config["kernels"],"directions":config["directions"],"epsilon":config["epsilon"],
      "near_zero_threshold":config["near_zero_threshold"],"seeds":seeds,"decision_thresholds":config["decision"],
      "method":{"directional_derivative":"central finite difference over shared Rademacher directions; identical directions, plus/minus points, source states and measurement uniforms for both kernels; not an exact full-gradient norm",
        "mmd_semantics":"biased empirical MMD; derivatives computed from raw unclipped values; clipped=max(raw,0) reported alongside","class_weighting":"equal","aggregation":"per-outcome ensemble mean"},
      "test_split_used":False,"dtype":"complex128","versions":{"python":platform.python_version(),"numpy":np.__version__}}
    result={"manifest":manifest,"training":training,"analyses":analyses,"decision":decision,"runtime_seconds":time.perf_counter()-started}
    manifest["runtime_seconds"]=result["runtime_seconds"]
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n"); (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    flat=[]
    for a in analyses:
        row={"step":a["step"],"transition":a["transition"],"checkpoint":a["checkpoint"]}
        for k in config["kernels"]:
            kk=a["kernels"][k]
            row.update({f"{k}_{key}":kk[key] for key in ("center_raw","center_clipped","mean_abs_derivative","median_abs_derivative","derivative_std","near_zero_fraction","descent_fraction")})
            row.update({f"{k}_class_{key}":kk["class_alignment"][key] for key in ("correlation","sign_agreement","conflict_rate","class0_mean_abs","class1_mean_abs")})
            row[f"{k}_all_class_realization_conflict"]=kk["all_class_realization_conflict_rate"]
            row[f"{k}_min_raw"]=kk["clipping"]["min_raw"]; row[f"{k}_clipped_fraction"]=kk["clipping"]["clipped_fraction"]
        comp=a["comparison"]
        row.update({"physics_mean_abs_derivative":a["physics"]["mean_abs_derivative"],
          "corr_global_2rdm":comp["global_vs_2rdm"]["correlation"],"cos_global_2rdm":comp["global_vs_2rdm"]["cosine"],
          "corr_global_physics":comp["global_vs_physics"]["correlation"],"cos_global_physics":comp["global_vs_physics"]["cosine"],
          "corr_2rdm_physics":comp["2rdm_vs_physics"]["correlation"],"cos_2rdm_physics":comp["2rdm_vs_physics"]["cosine"],
          "beneficial_global":comp["global_vs_physics"]["beneficial_direction_fraction"],"beneficial_2rdm":comp["2rdm_vs_physics"]["beneficial_direction_fraction"],
          "alignment_gain_correlation":comp["alignment_gain_correlation"],"alignment_gain_cosine":comp["alignment_gain_cosine"],
          "signal_ratio_2rdm_over_global":comp["signal_ratio_2rdm_over_global"],"conflict_reduction":comp.get("conflict_reduction_all_class_realization")})
        flat.append(row)
    with (out/"signal_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,flat[0].keys()); w.writeheader(); w.writerows(flat)
    with (out/"directions.csv").open("w",newline="") as f: w=csv.DictWriter(f,direction_rows[0].keys()); w.writeheader(); w.writerows(direction_rows)
    with (out/"clipping_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,clipping_rows[0].keys()); w.writeheader(); w.writerows(clipping_rows)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/quddpm/kernel_k1.yaml"); p.add_argument("--output",default="results/quddpm_kernel_diagnostics/k1_gradient"); a=p.parse_args()
    r=run(yaml.safe_load(Path(a.config).read_text()),a.output)
    print(json.dumps({"runtime_seconds":r["runtime_seconds"],"decision":r["decision"]["decision"],"checks":{k:v["pass"] for k,v in r["decision"]["checks"].items()}},indent=2))
if __name__=="__main__": main()

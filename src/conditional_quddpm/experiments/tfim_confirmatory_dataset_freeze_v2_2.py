"""Generate and validate frozen Protocol v2.2 TFIM corpora without QCNN execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from conditional_quddpm.datasets.tfim import SPLITS, _blocked_intervals, _observable_operators, tfim_hamiltonian
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2 import canonical_state, state_hash
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_1 import (
    constrained_random_split, distribution_audit, fs_distance,
)
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_2 import (
    CALIBRATION_POOL_HASH, generation_rng, generation_seed_manifest,
)

PROTOCOL_HASH = "8af9bf4d0ab8320ec46f5983c2073ec4b85e1955916fd35b1f3a5e4f6fa33004"
GENERATOR_COMMIT = "c86b3ebbd878902db2d8153190d8aa8ce71aa0c5"
EPSILON_SEP = 1.2991569026968808e-06
COUNTS = {"train": 140, "val": 30, "test": 30}
REGIONS = {0: (0.2, 0.8), 1: (1.2, 1.8)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _solve(g: float) -> tuple[float, float, np.ndarray]:
    values, vectors = eigh(tfim_hamiltonian(4, 1.0, g, "open"), subset_by_index=[0, 1])
    return float(values[0]), float(values[1]), canonical_state(vectors[:, 0], 1e-12)


def _initial_parameters(regime: str) -> tuple[list[tuple[int, float, str | None]], np.ndarray | None]:
    seeds = generation_seed_manifest(); domain = f"confirmatory.{regime}.parameter_generation"
    rng = generation_rng(seeds["domains"][domain]); rows: list[tuple[int, float, str | None]] = []
    if regime == "random":
        for label in (0, 1):
            rows.extend((label, float(g), None) for g in rng.uniform(*REGIONS[label], 200))
        split_rng = generation_rng(seeds["frozen_random_split_seed"]); preferred = np.full(400, "", dtype="U5")
        for label in (0, 1):
            indices = np.arange(label * 200, (label + 1) * 200); split_rng.shuffle(indices)
            for split, section in zip(SPLITS, np.split(indices, [140, 170]), strict=True): preferred[section] = split
        return rows, preferred
    intervals = {label: _blocked_intervals(*REGIONS[label], {"train": .7, "val": .15, "test": .15}, .02) for label in (0, 1)}
    for label in (0, 1):
        for split in SPLITS:
            rows.extend((label, float(g), split) for g in rng.uniform(*intervals[label][split], COUNTS[split]))
    return rows, None


def generate_corpus(regime: str, output: str | Path, *, generator_commit: str) -> dict:
    if regime not in {"random", "blocked_g"}: raise ValueError("unknown regime")
    if generator_commit != GENERATOR_COMMIT: raise ValueError("generator commit differs from frozen execution commit")
    output = Path(output)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    seeds = generation_seed_manifest(); rows, preferred = _initial_parameters(regime)
    replacement_domain = f"confirmatory.{regime}.replacement_sampling"
    replacement_rng = generation_rng(seeds["domains"][replacement_domain])
    states=[]; energies=[]; e1s=[]; gaps=[]; replacements=[]; labels=[]; gs=[]; provisional_splits=[]
    for index, (label, initial_g, split) in enumerate(rows):
        g=initial_g; events=[]
        for attempt in range(101):
            e0,e1,state=_solve(g); gap=e1-e0
            if gap > 1e-10: break
            if attempt == 100: raise RuntimeError("DEGENERACY_REPLACEMENT_EXHAUSTED")
            interval = REGIONS[label] if regime == "random" else _blocked_intervals(*REGIONS[label], {"train":.7,"val":.15,"test":.15}, .02)[str(split)]
            replacement=float(replacement_rng.uniform(*interval)); events.append({"attempt":attempt+1,"rejected_g":g,"E0":e0,"E1":e1,"gap":gap,"replacement_g":replacement}); g=replacement
        states.append(state); energies.append(e0); e1s.append(e1); gaps.append(gap); replacements.append(events); labels.append(label); gs.append(g); provisional_splits.append(split or "")
    states=np.asarray(states); labels=np.asarray(labels,dtype=np.int8); gs=np.asarray(gs)
    if regime == "random":
        splits, assignment = constrained_random_split(states, labels, COUNTS, EPSILON_SEP, seeds["frozen_random_split_seed"], preferred_splits=preferred)
    else: splits=np.asarray(provisional_splits); assignment={"assignment_attempts":1,"conflicts":0,"restarts":0,"reassignments":0}
    mx,mz2=_observable_operators(4); mxs=np.asarray([float(np.vdot(s,mx@s).real) for s in states]); mz2s=np.asarray([float(np.vdot(s,mz2@s).real) for s in states])
    hashes=np.asarray([state_hash(s,1e-12) for s in states]); corpus_id=f"tfim-confirmatory-v2.2-{regime.replace('_','-')}"
    ids=np.asarray([f"{corpus_id}-class-{int(label)}-{i:05d}" for i,label in enumerate(labels)])
    np.savez_compressed(output/"states.npz",states=states,energies=np.asarray(energies),E1=np.asarray(e1s),gaps=np.asarray(gaps),g=gs,labels=labels,splits=splits,parameter_ids=ids,state_hashes=hashes,magnetization_x=mxs,magnetization_z2=mz2s)
    provenance=[]
    for i in range(len(states)):
        provenance.append({"sample_id":str(ids[i]),"corpus_id":corpus_id,"dataset_realization_id":corpus_id,"dataset_regime":regime,"regime":regime,"class":int(labels[i]),"g":float(gs[i]),"parameter_generation_seed":seeds["domains"][f"confirmatory.{regime}.parameter_generation"],"parameter_generation_domain":f"confirmatory.{regime}.parameter_generation","split_seed":seeds["frozen_random_split_seed"] if regime=="random" else None,"replacement_seed":seeds["domains"][replacement_domain],"replacement_domain":replacement_domain,"replacement_events":replacements[i],"replacement_status":"replaced" if replacements[i] else "not_required","E0":float(energies[i]),"E1":float(e1s[i]),"gap":float(gaps[i]),"degeneracy_status":"nondegenerate","state_hash":str(hashes[i]),"generator_commit":generator_commit,"generator_version":"tfim-confirmatory-v2.2","protocol_version":"2.2.0","protocol_hash":PROTOCOL_HASH,"source_calibration_pool_relationship":"independent fresh generation; no reuse","calibration_pool_relationship":"independent fresh generation; no reuse"})
    manifest={"schema_version":1,"protocol_version":"2.2.0","protocol_hash":PROTOCOL_HASH,"corpus_id":corpus_id,"dataset_realization_id":corpus_id,"dataset_regime":regime,"regime":regime,"seeds":seeds,"generator_commit":generator_commit,"generator_version":"tfim-confirmatory-v2.2","sample_count":len(states),"assignment":assignment,"records":provenance}
    _dump(output/"manifest.json",manifest)
    return manifest


def validate_corpus(path: str | Path, regime: str, *, calibration_path: str | Path) -> dict:
    if regime not in {"random", "blocked_g"}: raise ValueError("unknown regime")
    path=Path(path); calibration_path=Path(calibration_path)
    if _sha(calibration_path) != CALIBRATION_POOL_HASH: raise ValueError("calibration pool integrity failure")
    with np.load(calibration_path) as calibration: calibration_states=np.asarray(calibration["states"]); calibration_g=np.asarray(calibration["g"])
    manifest=json.loads((path/"manifest.json").read_text())
    with np.load(path/"states.npz") as d: a={k:np.asarray(d[k]) for k in d.files}
    required={"states","energies","E1","gaps","g","labels","splits","parameter_ids","state_hashes","magnetization_x","magnetization_z2"}
    records=manifest.get("records",[]); seeds=generation_seed_manifest(); corpus_id=f"tfim-confirmatory-v2.2-{regime.replace('_','-')}"
    schema=bool(set(a)==required and a["states"].shape==(400,16) and a["states"].dtype==np.complex128 and a["g"].shape==(400,) and a["g"].dtype==np.float64 and a["labels"].dtype==np.int8 and all(a[name].ndim>=1 and len(a[name])==400 for name in required)
                and manifest.get("schema_version")==1 and manifest.get("protocol_version")=="2.2.0" and manifest.get("protocol_hash")==PROTOCOL_HASH and manifest.get("regime")==regime and manifest.get("dataset_regime")==regime and manifest.get("dataset_realization_id")==corpus_id and manifest.get("corpus_id")==corpus_id and manifest.get("seeds")==seeds and manifest.get("generator_commit")==GENERATOR_COMMIT and manifest.get("generator_version")=="tfim-confirmatory-v2.2" and manifest.get("sample_count")==400 and len(records)==400)
    states,g,labels,splits=a["states"],a["g"],a["labels"],a["splits"]
    norm=[]; residual=[]; energy_error=[]; observable_error=[]; gap_error=[]
    mx,mz2=_observable_operators(4)
    for i,state in enumerate(states):
        h=tfim_hamiltonian(4,1.0,float(g[i]),"open"); values=np.linalg.eigvalsh(h)[:2]
        norm.append(abs(float(np.vdot(state,state).real)-1)); residual.append(float(np.linalg.norm(h@state-a["energies"][i]*state)))
        energy_error.append(abs(float(values[0]-a["energies"][i]))); gap_error.append(abs(float(values[1]-values[0]-a["gaps"][i])))
        observable_error.append(max(abs(float(np.vdot(state,mx@state).real)-a["magnetization_x"][i]),abs(float(np.vdot(state,mz2@state).real)-a["magnetization_z2"][i])))
    recomputed_hashes=[state_hash(s,1e-12) for s in states]
    numeric_arrays=(states,g,a["energies"],a["E1"],a["gaps"],a["magnetization_x"],a["magnetization_z2"])
    errors=np.asarray(norm+residual+energy_error+observable_error+gap_error,dtype=float)
    region_ok=all(REGIONS[int(labels[i])][0] <= g[i] < REGIONS[int(labels[i])][1] for i in range(len(g)) if int(labels[i]) in REGIONS)
    required_record_fields={"sample_id","corpus_id","dataset_realization_id","dataset_regime","class","g","parameter_generation_seed","parameter_generation_domain","split_seed","replacement_seed","replacement_domain","replacement_events","replacement_status","E0","E1","gap","degeneracy_status","state_hash","generator_commit","generator_version","protocol_version","protocol_hash","source_calibration_pool_relationship"}
    parameter_domain=f"confirmatory.{regime}.parameter_generation"; replacement_domain=f"confirmatory.{regime}.replacement_sampling"
    expected_rows,expected_preferred=_initial_parameters(regime); replacement_replay=generation_rng(seeds["domains"][replacement_domain]); provenance_ok=schema
    for i,row in enumerate(records):
        events=row.get("replacement_events"); initial_label,initial_g,initial_split=expected_rows[i]; split=str(splits[i]); interval=REGIONS[int(labels[i])] if regime=="random" else _blocked_intervals(*REGIONS[int(labels[i])],{"train":.7,"val":.15,"test":.15},.02)[split]
        event_ok=isinstance(events,list) and len(events)<=100 and row.get("replacement_status")==("replaced" if events else "not_required")
        previous=initial_g
        if event_ok:
            for position,event in enumerate(events,1):
                if set(event)!={"attempt","rejected_g","E0","E1","gap","replacement_g"}: event_ok=False; break
                e0,e1,_=_solve(float(event["rejected_g"])); expected_replacement=float(replacement_replay.uniform(*interval))
                event_ok &= event["attempt"]==position and event["rejected_g"]==previous and abs(event["E0"]-e0)<=1e-12 and abs(event["E1"]-e1)<=1e-12 and abs(event["gap"]-(e1-e0))<=1e-12 and event["gap"]<=1e-10 and event["replacement_g"]==expected_replacement
                previous=event["replacement_g"]
        event_ok &= row.get("g")==previous
        provenance_ok &= required_record_fields <= set(row) and initial_label==int(labels[i]) and (initial_split is None or initial_split==split) and row["sample_id"]==str(a["parameter_ids"][i]) and row["corpus_id"]==row["dataset_realization_id"]==corpus_id and row["dataset_regime"]==regime and row["class"]==int(labels[i]) and row["g"]==float(g[i]) and row["E0"]==float(a["energies"][i]) and row["E1"]==float(a["E1"][i]) and row["gap"]==float(a["gaps"][i]) and row["state_hash"]==recomputed_hashes[i] and row["protocol_version"]=="2.2.0" and row["protocol_hash"]==PROTOCOL_HASH and row["parameter_generation_domain"]==parameter_domain and row["parameter_generation_seed"]==seeds["domains"][parameter_domain] and row["replacement_domain"]==replacement_domain and row["replacement_seed"]==seeds["domains"][replacement_domain] and row["split_seed"]==(seeds["frozen_random_split_seed"] if regime=="random" else None) and row["degeneracy_status"]=="nondegenerate" and row["generator_version"]==manifest.get("generator_version") and row["generator_commit"]==manifest.get("generator_commit") and row["source_calibration_pool_relationship"]=="independent fresh generation; no reuse" and event_ok
    if regime=="random" and schema:
        expected_splits,_=constrained_random_split(states,labels,COUNTS,EPSILON_SEP,seeds["frozen_random_split_seed"],preferred_splits=expected_preferred)
        provenance_ok &= bool(np.array_equal(splits,expected_splits))
    physical=bool(schema and provenance_ok and region_ok and all(np.all(np.isfinite(x)) for x in numeric_arrays) and np.all(np.isfinite(errors)) and float(errors.max())<=1e-10 and np.all(a["gaps"]>1e-10)
                  and np.allclose(a["E1"]-a["energies"],a["gaps"],rtol=0,atol=1e-10) and recomputed_hashes==a["state_hashes"].tolist()
                  and set(labels.tolist())=={0,1} and set(splits.tolist())==set(SPLITS) and len(set(a["parameter_ids"].tolist()))==400)
    hashes=recomputed_hashes; duplicate_within={s:len(x)-len(set(x)) for s in SPLITS for x in [[hashes[i] for i in np.flatnonzero(splits==s)]]}
    cross_hash=sum(len(set(a["state_hashes"][splits==x])&set(a["state_hashes"][splits==y])) for x,y in (("train","val"),("train","test"),("val","test")))
    pair_min={}; argmin=None; violations=0
    if regime=="random":
        overall=float("inf")
        for left,right in (("train","val"),("train","test"),("val","test")):
            best=(float("inf"),None)
            for i in np.flatnonzero(splits==left):
                for j in np.flatnonzero(splits==right):
                    distance=fs_distance(states[i],states[j]); violations+=int(distance<EPSILON_SEP)
                    if distance<best[0]: best=(distance,[int(i),int(j)])
            pair_min[f"{left}-{right}"]={"minimum_fs":best[0],"argmin_indices":best[1]}
            if best[0]<overall: overall,argmin=best
        preferred=np.full(len(states),"",dtype="U5"); rows,_preferred=_initial_parameters("random"); preferred[:]=_preferred
        distribution=distribution_audit(g,labels,preferred,splits)
    else: overall=None; violations=None; pair_min={}; argmin=None; distribution={"verdict":"NOT_APPLICABLE_BLOCKED_G","blocked_g_like":True}
    calibration_hash_overlap=len(set(hashes)&{state_hash(x,1e-12) for x in calibration_states}); calibration_g_overlap=len(set(g.tolist())&set(calibration_g.tolist()))
    freshness=calibration_hash_overlap==calibration_g_overlap==0 and _sha(path/"states.npz")!=CALIBRATION_POOL_HASH
    counts={str(label):{s:int(np.sum((labels==label)&(splits==s))) for s in SPLITS} for label in (0,1)}
    blocked_ok=True
    if regime=="blocked_g":
        intervals={label:_blocked_intervals(*REGIONS[label],{"train":.7,"val":.15,"test":.15},.02) for label in (0,1)}
        blocked_ok=all(np.all((g[(labels==label)&(splits==s)]>=intervals[label][s][0])&(g[(labels==label)&(splits==s)]<intervals[label][s][1])) for label in (0,1) for s in SPLITS)
        blocked_ok &= min(abs(g[(labels==label)&(splits==x)][:,None]-g[(labels==label)&(splits==y)][None,:]).min() for label in (0,1) for x,y in (("train","val"),("train","test"),("val","test")))>=.02
    projective_ok=regime!="random" or violations==0
    distribution_ok=regime!="random" or distribution["verdict"]=="PASS"
    valid=physical and not any(duplicate_within.values()) and cross_hash==0 and projective_ok and distribution_ok and freshness and blocked_ok and all(counts[str(c)]==COUNTS for c in (0,1))
    report={"valid":bool(valid),"schema_pass":schema,"provenance_records_pass":bool(provenance_ok),"physical_pass":physical,"exact_duplicates":{"within_split":duplicate_within,"cross_split":cross_hash},"projective_separation":{"epsilon_sep":EPSILON_SEP,"overall_minimum_fs":overall,"argmin_pair":argmin,"pair_minima":pair_min,"violation_count":violations},"freshness":{"pass":freshness,"calibration_state_hash_overlap":calibration_hash_overlap,"calibration_exact_g_overlap":calibration_g_overlap},"distribution":distribution,"blocked_g_contract_pass":bool(blocked_ok),"counts":counts,"numerics":{"max_norm_error":max(norm),"max_residual":max(residual),"max_energy_error":max(energy_error),"max_observable_error":max(observable_error),"max_gap_error":max(gap_error)},"manifest_protocol_hash":manifest["protocol_hash"]}
    _dump(path/"validation.json",report); return report


def scientific_hash(path: str | Path) -> str:
    path=Path(path); digest=hashlib.sha256()
    with np.load(path/"states.npz") as d:
        for name in sorted(d.files):
            array=np.ascontiguousarray(d[name]); metadata=json.dumps({"name":name,"dtype":array.dtype.str,"shape":array.shape},sort_keys=True,separators=(",",":")).encode(); payload=array.tobytes()
            digest.update(len(metadata).to_bytes(8,"little")+metadata+len(payload).to_bytes(8,"little")+payload)
    for name in ("manifest.json","validation.json"):
        payload=json.dumps(json.loads((path/name).read_text()),sort_keys=True,separators=(",",":")).encode(); digest.update(len(name).to_bytes(8,"little")+name.encode()+len(payload).to_bytes(8,"little")+payload)
    return digest.hexdigest()

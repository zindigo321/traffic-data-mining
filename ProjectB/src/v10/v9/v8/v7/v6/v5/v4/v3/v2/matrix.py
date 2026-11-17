"""Execute preregistered matrices and freeze model choices using development only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import statistics
import sys

import pandas as pd

from common import CODE, digest, json_read, json_write, now, record, settings
from prepare_chicago import fingerprint_keys, key_matrix, local_dates, minute_values, read_npz, times_for

RECIPES = ("V0","V1","V2","V3")


def get_folds(root,cfg):
    primary = cfg["main_folds"]
    if all((root/"prepared"/f"{f}.npz").is_file() for f in primary):
        return primary
    return cfg["fallback_folds"]


def label(recipe,pool,seed,algo="gt",center=False):
    if algo!="gt":
        return f"{algo}_{recipe}_s{seed}"
    return f"{recipe}_{pool}_s{seed}" + ("_center" if center else "")


def job(root,cfg,fold,recipe,pool,seed,algo="gt",center=False,epoch=None):
    name = label(recipe,pool,seed,algo,center)
    out = root/"results"/("calibration" if fold=="final" else f"dev/{fold}")/name
    opt = dict(recipe=recipe,pool=pool,seed=seed,algo=algo,central=center,fixed_epoch=epoch)
    if (out/"status.json").exists():
        saved = json_read(out/"options.json")
        if saved!=opt:
            raise ValueError(f"Existing job has different options: {name}")
        return out
    args = [sys.executable,"-B","-X","utf8",CODE/"experiment.py","--fold",fold,"--recipe",recipe,
            "--pool",pool,"--seed",seed,"--algo",algo,"--destination",out]
    if center:
        args.append("--central")
    if epoch is not None:
        args.extend(["--fixed-epoch",epoch])
    rc = record(root,f"{fold}_{name}",args)
    if rc:
        raise RuntimeError(f"Failed job {fold}/{name}, exit {rc}")
    return out


def execute(root,cfg,jobs):
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        return list(pool.map(lambda j:job(root,cfg,*j),jobs))


def load_row(path, mode):
    row = json_read(path/"status.json")
    if row["status"] != "complete":
        raise ValueError(f"Study not complete: {path}")
    return {k:v for k,v in row.items() if k!="scored"} | row["scored"][mode] | {"route":mode,"job":path.name}


def stability(rows):
    aps, f1s = [[r[k] for r in rows] for k in ("AP","f1")]
    return dict(mean_AP=statistics.mean(aps),mean_f1=statistics.mean(f1s),
                sd_AP=statistics.stdev(aps) if len(aps)>1 else 0.,
                mean_roc=statistics.mean(r["ROC_AUC"] for r in rows), runs=len(rows))


def key(s):
    return (round(s["mean_AP"],6),round(s["mean_f1"],6),-round(s["sd_AP"],6))


def screen(root,cfg):
    folds = get_folds(root,cfg)
    outputs = execute(root,cfg,[(f,r,"max",cfg["seeds"][0]) for f,r in product(folds,RECIPES)])
    raw = [load_row(p,"raw") for p in outputs]
    aggregates = {r:stability([x for x in raw if x["recipe"]==r]) for r in RECIPES}
    best = max(RECIPES[1:],key=lambda r:key(aggregates[r]))
    top_paths = [p for p in outputs if p.name.startswith(best+"_")]
    evidence = [json_read(root/"evidence"/f"fold_{f}.json")["parts"][1] for f in folds]
    gate_allowed = all(e["positive"]>0 and e["high_risk_recovered"]>0 for e in evidence)
    route = "raw"
    routed = []
    if gate_allowed:
        routed = [load_row(p,"routed") for p in top_paths]
        if key(stability(routed)) > key(aggregates[best]):
            route = "routed"
    pd.DataFrame(raw+routed).to_csv(root/"results/screen_metrics.csv",index=False,na_rep="N/A")
    rule = "6-decimal AP, then F1, then smaller sample SD of AP, then fixed iteration order"
    decision = dict(folds=folds, screening_pool="max", screening_seed=cfg["seeds"][0],
                    choices=aggregates, chosen_enhancement=best, route=route, rule=rule,
                    routed_challenger=stability(routed) if routed else None,
                    v0_is_reference=True, holdout_labels_used=False, completed_utc=now())
    json_write(root/"results/screen_selection.json",decision)
    print(decision,flush=True)


def confirm(root,cfg):
    sr = json_read(root/"results/screen_selection.json")
    folds, r, mode = [sr[k] for k in ("folds","chosen_enhancement","route")]
    jobs = [(f,r,p,s) for f,p,s in product(folds,cfg["pools"],cfg["seeds"])]
    jobs += [(f,r,"max",cfg["seeds"][0],algo) for f,algo in product(folds,("logreg","rule"))]
    outs = execute(root,cfg,jobs)
    gt = [load_row(p,mode) for p in outs if p.name.startswith(r+"_")]
    # Take the SD across fold means, not across 9 correlated seed/fold values.
    by_pool = {}
    for pool in cfg["pools"]:
        by_fold = [stability([x for x in gt if x["pool"]==pool and x["fold"]==f]) for f in folds]
        means = [row["mean_AP"] for row in by_fold]
        by_pool[pool] = dict(mean_AP=statistics.mean(means),mean_f1=statistics.mean(x["mean_f1"] for x in by_fold),
                             sd_AP=statistics.stdev(means),folds=by_fold,runs=sum(x["runs"] for x in by_fold))
    best_pool = max(cfg["pools"],key=lambda p:key(by_pool[p]))
    centers = execute(root,cfg,[(f,r,best_pool,cfg["seeds"][0],"gt",True) for f in folds])
    comparisons = [load_row(p,"raw") for p in outs if not p.name.startswith(r+"_")]
    comparisons += [load_row(p,"raw") for p in centers]
    epoch = {p:int(statistics.median(x["best_epoch"] for x in gt if x["pool"]==p)) for p in cfg["pools"]}
    epoch["center"] = int(statistics.median(load_row(p,"raw")["best_epoch"] for p in centers))
    refs = [root/"results/dev"/f/label("V0","max",cfg["seeds"][0]) for f in folds]
    epoch["reference"] = int(statistics.median(load_row(p,"raw")["best_epoch"] for p in refs))
    pd.DataFrame(gt+comparisons).to_csv(root/"results/confirmation_metrics.csv",index=False,na_rep="N/A")
    decision = dict(status="development_frozen",folds=folds, recipe=r,route=mode,pooling=best_pool,
                    epochs=epoch,by_pool=by_pool,rule=sr["rule"],holdout_labels_used=False,completed_utc=now(),
                    search_count=dict(screen=4,screen_routed_challenger=int(sr["routed_challenger"] is not None),
                                      pool_choices=3,seeds_per_pool=3))
    json_write(root/"results/development_selection.json",decision)
    print(decision,flush=True)


def freeze(root,cfg,config_path):
    evidence = root/"evidence"
    if (evidence/"holdout_unlock.json").exists() or (evidence/"freeze.json").exists():
        raise ValueError("Holdout already frozen or used. Choose a new untouched holdout for model changes.")
    select = json_read(root/"results/development_selection.json")
    scope = evidence/"download_holdout_sealed.json"
    if (not (root/"prepared/final.npz").exists() or not scope.is_file() or
            json_read(scope)["status"]!="complete" or select["folds"]!=cfg["main_folds"]):
        json_write(evidence/"holdout_status.json",dict(status="blocked", reason="Fixed extension or preregistered forward folds incomplete"))
        return
    r,p,mode = [select[k] for k in ("recipe","pooling","route")]
    epochs = select["epochs"]
    jobs = [("final",r,q,s,"gt",False,epochs[q]) for q,s in product(cfg["pools"],cfg["seeds"])]
    jobs += [("final","V0","max",s,"gt",False,epochs["reference"]) for s in cfg["seeds"]]
    jobs += [("final",r,p,s,"gt",True,epochs["center"]) for s in cfg["seeds"]]
    jobs += [("final",r,p,cfg["seeds"][0],a) for a in ("logreg","rule")]
    paths = execute(root,cfg,jobs)
    models = []
    for directory in paths:
        state = json_read(directory/"status.json")
        if state["status"]!="complete":
            json_write(evidence/"holdout_status.json",dict(status="blocked",reason=f"Calibration unsupported for {directory.name}"))
            return
        mode_names = [mode] if state["algo"]=="gt" and state["recipe"]==r and not state["central"] else ["raw"]
        if state["algo"]=="gt" and state["pool"]==p and state["recipe"]==r and not state["central"]:
            alternate = "routed" if mode=="raw" else "raw"
            mode_names.append(alternate)
        files = ["status.json","options.json"]
        if state["weight"]:
            files += [state["weight"],"preprocessor.npz","sampling_ids.npz"]
        for tag in mode_names:
            verify = {f:digest(directory/f) for f in files+[f"assessment_{tag}.npz"]}
            selected = (state["recipe"]==r and state["pool"]==p and not state["central"] and state["algo"]=="gt" and tag==mode)
            models.append(dict(job=directory.name,mode=tag,selected=selected,recipe=state["recipe"],pool=state["pool"],
                               algo=state["algo"],central=state["central"],seed=state["seed"],
                               threshold=state["scored"][tag]["threshold"],platt=state["scored"][tag]["platt"],files=verify))
    roads = read_npz(root/"prepared/topology.npz")["road_ids"]
    grid = key_matrix(minute_values(times_for(local_dates([cfg["holdout"]]),cfg)),roads)
    sha_files = {s:digest(root/s) for s in (
        "evidence/roads_v1.csv","evidence/download_holdout_sealed.json","prepared/frozen_history.npz",
        "prepared/topology.npz","results/screen_selection.json","results/development_selection.json"
    )}
    code_files = {f.relative_to(CODE).as_posix():digest(f) for f in sorted(CODE.rglob("*.py"))}
    contract = dict(
        frozen_utc=now(), dataset="Chicago 4g9f-3jbs", holdout_range=cfg["holdout"],
        reference="V0 max, same final dates and evaluation keys", selection=select, cfg_sha256=digest(config_path),
        expected_grid_key_sha256=fingerprint_keys(grid),expected_grid_bins=grid.size, road_ids=roads.tolist(),
        declared_main_support="All known y and lag1-eligible, no risk-based exclusions",
        prereg_plan_sha256=digest(root/"plan.md") if (root/"plan.md").exists() else None,
        calibration_known_keys_sha256=fingerprint_keys(read_npz(paths[0]/"assessment_raw.npz")["sample_id"]),
        calibration_range=cfg["calibration"],threshold_rule="Full calibration F1, tie closest .5 then lower",
        actual_model_count=len(paths), score_variant_count=len(models), models=models, digests=sha_files,
        code_digests=code_files, holdout_observation_values_read=False,
    )
    json_write(evidence/"freeze.json",contract)
    json_write(evidence/"freeze_digest.json",dict(sha256=digest(evidence/"freeze.json"), frozen_utc=contract["frozen_utc"]))
    print(f"Frozen {len(models)} score variants before parsing holdout observations",flush=True)


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("stage",choices=["screen","confirm","freeze"])
    arg = p.parse_args()
    root,cfg = settings()
    if arg.stage=="screen":
        screen(root,cfg)
    elif arg.stage=="confirm":
        confirm(root,cfg)
    else:
        freeze(root,cfg,Path(os.environ.get("GT2_CONFIG",CODE/"config.json")))


if __name__ == "__main__":
    main()




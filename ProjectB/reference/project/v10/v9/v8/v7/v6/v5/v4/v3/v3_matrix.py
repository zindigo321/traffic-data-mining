"""Pre-registered development matrix, pool confirmation and holdout freeze."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os
from pathlib import Path
import statistics
import sys

from v3_io import CODE,context,load,now,record,save,sha
from v3_prepare import fingerprint_keys,key_matrix,local_dates,minute_values,read_npz,times_for
import pandas as pd


def name(recipe,pool,seed):
    return f"{recipe}_{pool}_s{seed}"


def run_one(root,cfg,fold,recipe,pool,seed,fixed_epoch=None):
    tag=name(recipe,pool,seed)
    dest=root/"results"/("calibration" if fold=="final" else f"dev/{fold}")/tag
    opt=dict(fold=fold,recipe=recipe,pool=pool,seed=seed,fixed_epoch=fixed_epoch)
    if (dest/"status.json").is_file():
        if load(dest/"options.json")!=opt:
            raise ValueError("Resumed job options differ")
        return dest
    args=[sys.executable,"-B","-X","utf8",CODE/"v3_train.py","--fold",fold,"--recipe",recipe,
          "--pool",pool,"--seed",seed,"--output",dest]
    if fixed_epoch is not None:
        args+=["--fixed-epoch",fixed_epoch]
    if record(root,f"{fold}_{tag}",args):
        raise RuntimeError(f"Fit failed: {fold}/{tag}")
    return dest


def execute(root,cfg,jobs):
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        outputs=list(pool.map(lambda row:run_one(root,cfg,*row),jobs))
    rows=[load(p/"status.json") for p in outputs]
    if not all(r["status"]=="complete" for r in rows):
        save(root/"evidence/blocked.json",dict(stage="fit",reason="Training or assessment missing one class",
             blocked=[str(p) for p,r in zip(outputs,rows,strict=True) if r["status"]!="complete"]))
        raise ValueError("A planned fit was blocked")
    return outputs,rows


def aggregate(rows):
    aps=[row["AP"] for row in rows]
    return dict(AP=statistics.mean(aps),F1=statistics.mean(row["f1"] for row in rows),
                AP_sd=statistics.stdev(aps) if len(aps)>1 else 0.,
                FP_per_day=statistics.mean(row["fp_per_day"] for row in rows),runs=len(aps))


def criterion(metrics):
    return (round(metrics["AP"],6),round(metrics["F1"],6),-round(metrics["AP_sd"],6))


def screen(root,cfg):
    folds=list(cfg["folds"])
    if not all((root/"prepared"/f"{f}.npz").is_file() for f in folds):
        save(root/"evidence/blocked.json",dict(stage="screen",reason="Three fixed forward folds were not prepared"))
        return
    specs=["A",*cfg["recipes"]]
    _,results=execute(root,cfg,[(f,r,"mean",cfg["seeds"][0]) for f,r in product(folds,specs)])
    data={r:aggregate([row for row in results if row["recipe"]==r]) for r in specs}
    pick=max(cfg["recipes"],key=lambda r:criterion(data[r]))
    pd.DataFrame(results).drop(columns=["platt"]).to_csv(root/"results/screen.csv",index=False,na_rep="N/A")
    receipt=dict(recipe=pick,results=data,folds=folds,seed=cfg["seeds"][0],pool="mean",
                 rule="AP to 6 decimals, F1 to 6 decimals, smaller AP SD, registered order",
                 holdout_used=False,at_utc=now())
    save(root/"results/screen_selection.json",receipt)
    print(receipt,flush=True)


def confirm(root,cfg):
    if not (root/"results/screen_selection.json").is_file():
        return
    preliminary=load(root/"results/screen_selection.json")
    recipe=preliminary["recipe"]
    folds=preliminary["folds"]
    _,rows=execute(root,cfg,[(f,recipe,p,s) for f,p,s in product(folds,cfg["pools"],cfg["seeds"])])
    stats={}
    epochs={}
    for p in cfg["pools"]:
        current=[r for r in rows if r["pool"]==p]
        grouped=[aggregate([r for r in current if r["fold"]==f]) for f in folds]
        aps=[d["AP"] for d in grouped]
        stats[p]=dict(AP=statistics.mean(aps),F1=statistics.mean(d["F1"] for d in grouped),
                      AP_sd=statistics.stdev(aps),folds=grouped,runs=len(current))
        epochs[p]=int(statistics.median(r["best_epoch"] for r in current))
    pool=max(cfg["pools"],key=lambda p:criterion(stats[p]))
    ref=[load(root/"results/dev"/f/name("A","mean",cfg["seeds"][0])/"status.json") for f in folds]
    epochs["A"]=int(statistics.median(r["best_epoch"] for r in ref))
    pd.DataFrame(rows).drop(columns=["platt"]).to_csv(root/"results/confirmation.csv",index=False,na_rep="N/A")
    receipt=dict(recipe=recipe,pool=pool,epochs=epochs,folds=folds,results=stats,holdout_used=False,
                 rule=preliminary["rule"],at_utc=now())
    save(root/"results/development_selection.json",receipt)
    print(receipt,flush=True)


def freeze(root,cfg):
    evidence=root/"evidence"
    choice=root/"results/development_selection.json"
    download=evidence/"download_holdout_sealed.json"
    if (evidence/"freeze.json").exists() or (evidence/"holdout_unlock.json").exists():
        raise ValueError("Holdout was already frozen or opened")
    if not choice.exists() or not download.is_file() or load(download)["status"]!="complete":
        save(evidence/"blocked.json",dict(stage="freeze",reason="Fixed development or holdout scope incomplete"))
        return
    if load(download)["observation_values_parsed"] is not False:
        raise ValueError("Holdout transport included observation parsing")
    if (root/"prepared/holdout.npz").exists():
        raise ValueError("A holdout array already exists before freeze")
    selected=load(choice)
    epochs=selected["epochs"]
    r=selected["recipe"]
    jobs=[("final",r,p,s,epochs[p]) for p,s in product(cfg["pools"],cfg["seeds"])]
    jobs+=[("final","A","mean",s,epochs["A"]) for s in cfg["seeds"]]
    dev=load(evidence/"prepared_development.json")
    if sha(root/"prepared/final.npz")!=dev["hashes"]["final"]:
        raise ValueError("Final training/calibration array changed")
    paths,rows=execute(root,cfg,jobs)
    models=[]
    cal_support=None
    for path,row in zip(paths,rows,strict=True):
        keys=fingerprint_keys(read_npz(path/"assessment.npz")["sample_id"])
        if cal_support is None:
            cal_support=keys
        elif cal_support!=keys:
            raise ValueError("Different calibration support")
        models.append(dict(job=path.name,recipe=row["recipe"],pool=row["pool"],seed=row["seed"],threshold=row["threshold"],
                           platt=row["platt"],selected=row["recipe"]==r and row["pool"]==selected["pool"],
                           files={f:sha(path/f) for f in ("weight.pt","preprocessor.npz","assessment.npz","options.json",
                                                         "sampling_ids.npz","status.json","sampling.json")}))
    road=read_npz(root/"prepared/topology.npz")["road_ids"]
    grid=key_matrix(minute_values(times_for(local_dates([cfg["holdout"]]),cfg)),road)
    frozen_files=("evidence/roads.csv","evidence/prior_inputs.json","evidence/download_holdout_sealed.json",
                  "evidence/prepared_development.json","prepared/frozen_history.npz","prepared/topology.npz",
                  "results/screen_selection.json","results/development_selection.json")
    config=Path(os.environ.get("GT3_CONFIG",CODE/"config.json"))
    result=dict(frozen_utc=now(),selection=selected,config_sha256=sha(config),plan_sha256=sha(root/"plan.md"),
                code_hashes={p.relative_to(CODE).as_posix():sha(p) for p in sorted(CODE.rglob("*.py"))},
                digests={f:sha(root/f) for f in frozen_files},holdout=cfg["holdout"],calibration=cfg["calibration"],
                threshold_rule="Max F1 over complete calibration, tie nearest .5 then lower",
                main_support="All known labels with unchanged lag1 qualification",
                expected_keys_sha256=fingerprint_keys(grid),expected_bins=grid.size,road_ids=road.tolist(),
                calibration_known_ids_sha256=cal_support,models=models,holdout_values_read=False)
    save(evidence/"freeze.json",result)
    save(evidence/"freeze_sha256.json",dict(sha256=sha(evidence/"freeze.json"),frozen_utc=result["frozen_utc"]))
    print(f"Frozen {len(models)} model configurations",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["screen","confirm","freeze"])
    a=p.parse_args()
    globals()[a.stage](*context())


if __name__=="__main__":
    main()


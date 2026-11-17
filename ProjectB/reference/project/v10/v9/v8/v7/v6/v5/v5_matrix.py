"""Preregistered development selection and calibration-only fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import statistics

import pandas as pd

from v5_io import load, now, python, record, save, settings, sha
from v5_train import read_npz
from v4_matrix import average, criterion


def folder(root,fold,recipe,pool,seed):
    base=root/"results"
    parent=base/"calibration" if fold=="final" else base/"dev"/fold
    return parent/f"{recipe}_{pool}_s{seed}"


def single(root,job):
    f,r,p,s,epoch=job
    out=folder(root,f,r,p,s)
    result=out/"status.json"
    if not result.is_file():
        command=python("v5_train.py","--fold",f,"--recipe",r,"--pool",p,"--seed",s,"--output",out)
        if epoch is not None:
            command+=["--fixed-epoch",str(epoch)]
        record(root,f"{f}_{out.name}",command)
    row=load(result)
    if row["status"]!="complete" or tuple(row[k] for k in ("fold","recipe","pool","seed","fixed_epoch"))!=job:
        raise ValueError(f"Unexpected job {out}")
    return row


def execute(root,cfg,jobs):
    if (root/"evidence/unlock.json").exists():
        raise PermissionError("Training is closed after holdout unlock")
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        return list(pool.map(lambda job:single(root,job),jobs))


def screen(root,cfg):
    variants=["B",*cfg["recipes"]]
    rows=execute(root,cfg,[(f,r,"mean",cfg["seeds"][0],None) for f,r in product(cfg["folds"],variants)])
    means={r:average([x for x in rows if x["recipe"]==r]) for r in variants}
    best=max(cfg["recipes"],key=lambda r:criterion(means[r]))
    result=dict(at_utc=now(),recipe=best,summary=means,rule="AP at 6 decimals, lower FP/day at 60% recall, F1, registry order",
                holdout_used=False,folds=list(cfg["folds"]))
    save(root/"results/screen.json",result)
    pd.DataFrame(rows).to_csv(root/"results/screen.csv",index=False,na_rep="N/A")
    print(dict(chosen=best,means={r:{k:m[k] for k in ("AP","f1","FP60_per_day")} for r,m in means.items()}),flush=True)


def confirm(root,cfg):
    initial=load(root/"results/screen.json")
    chosen=initial["recipe"]
    specs=[(f,chosen,p,s,None) for f,p,s in product(cfg["folds"],cfg["pools"],cfg["seeds"])]
    specs.extend((f,"B","mean",s,None) for f,s in product(cfg["folds"],cfg["seeds"]))
    rows=execute(root,cfg,specs)
    means,epochs={},{}
    for r,p in [(chosen,p) for p in cfg["pools"]]+[("B","mean")]:
        key=f"{r}_{p}"
        cohort=[x for x in rows if (x["recipe"],x["pool"])==(r,p)]
        means[key]=average(cohort)
        epochs[key]=int(statistics.median(x["best_epoch"] for x in cohort))
    pool=max(cfg["pools"],key=lambda p:criterion(means[f"{chosen}_{p}"]))
    receipt=dict(at_utc=now(),recipe=chosen,pool=pool,summary=means,rule=initial["rule"],epochs=epochs,holdout_used=False)
    save(root/"results/selection.json",receipt)
    pd.DataFrame(rows).to_csv(root/"results/confirmation.csv",index=False,na_rep="N/A")
    print(dict(chosen=chosen,pool=pool,means={r:{k:m[k] for k in ("AP","f1","FP60_per_day")} for r,m in means.items()}),flush=True)


def calibrate(root,cfg):
    choice=load(root/"results/selection.json")
    r=choice["recipe"]
    specs=[]
    for recipe,pool in [(r,p) for p in cfg["pools"]]+[("B","mean")]:
        epoch=choice["epochs"][f"{recipe}_{pool}"]
        specs.extend(("final",recipe,pool,s,epoch) for s in cfg["seeds"])
    rows=execute(root,cfg,specs)
    pd.DataFrame(rows).to_csv(root/"results/calibration.csv",index=False,na_rep="N/A")
    save(root/"results/calibration_receipt.json",dict(at_utc=now(),rows=rows,calendar=cfg["calibration"],
         data_sha256=sha(root/"prepared/final.npz"),holdout_used=False,choice_sha256=sha(root/"results/selection.json")))
    print(dict(calibrated=len(rows),calendar=cfg["calibration"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("screen","confirm","calibrate"))
    stage=p.parse_args().stage
    globals()[stage](*settings())

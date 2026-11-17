"""Twelve from-scratch fits using the frozen V5 epoch settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import sys

import pandas as pd

from v6_io import V5,load,now,python,record,save,settings,sha
from v6_register import verify_registration
sys.path.insert(0,str(V5))
from v5_train import fit as prior_fit


def configurations(cfg):
    jobs=[("X1",pool,seed) for pool,seed in product(cfg["pools"],cfg["seeds"])]
    jobs.extend(("B","mean",seed) for seed in cfg["seeds"])
    return jobs


def where(root,recipe,pool,seed):
    return root/"results/calibration"/f"{recipe}_{pool}_s{seed}"


def training_budget():
    # V6 registration uses epochs for the fixed checkpoint table.
    # The audited V5 trainer consumes the separate total epoch count.
    epochs=load(V5/"config.json")["epochs"]
    if epochs!=12:
        raise ValueError("The registered full training budget changed")
    return epochs


def fit(root,cfg,recipe,pool,seed):
    if (recipe,pool,seed) not in configurations(cfg):
        raise ValueError("Unregistered configuration")
    if (root/"evidence/freeze.json").exists() or (root/"evidence/unlock.json").exists():
        raise PermissionError("Training is closed once frozen")
    verify_registration(root,cfg)
    opt=dict(fold="final",recipe=recipe,pool=pool,seed=seed,fixed_epoch=cfg["epochs"][f"{recipe}_{pool}"])
    prior_fit(root,{**cfg,"epochs":training_budget()},opt,where(root,recipe,pool,seed))


def calibrate(root,cfg):
    verify_registration(root,cfg)
    if any((root/name).exists() for name in ("evidence/freeze.json","evidence/unlock.json")):
        raise PermissionError("Training is closed once frozen")
    def single(spec):
        recipe,pool,seed=spec
        path=where(root,*spec)
        if not (path/"status.json").is_file():
            record(root,path.name,python("v6_train.py","fit","--recipe",recipe,"--pool",pool,"--seed",seed))
        return load(path/"status.json")
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        rows=list(pool.map(single,configurations(cfg)))
    if len(rows)!=12 or any(r["status"]!="complete" for r in rows):
        raise ValueError("Unexpected fit count")
    pd.DataFrame(rows).to_csv(root/"results/calibration.csv",index=False,na_rep="N/A")
    save(root/"results/calibration_receipt.json",dict(at_utc=now(),rows=rows,calendar=cfg["calibration"],
        data_sha256=sha(root/"prepared/final.npz"),registration_sha256=sha(root/"evidence/registration.json"),
        V5_weights_loaded=False,V5_holdout_used=False,threshold_source="same independent 28-day calibration"))
    print(dict(runs=len(rows),calendar=cfg["calibration"],primary=[cfg["primary_recipe"],cfg["primary_pool"]]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("fit","calibrate"))
    p.add_argument("--recipe")
    p.add_argument("--pool")
    p.add_argument("--seed",type=int)
    args=p.parse_args()
    if args.stage=="fit":
        fit(*settings(),args.recipe,args.pool,args.seed)
    else:
        calibrate(*settings())

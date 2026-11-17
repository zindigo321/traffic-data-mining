"""Three forward development folds and one independent calibration period."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import statistics

from v7_io import load,now,python,record,save,settings
from v7_train import NEURAL,specifications,folder


def execute(root,cfg,folds,epoch=None):
    jobs=[(fold,model,seed) for fold,(model,seed) in product(folds,specifications(cfg))]
    def one(job):
        fold,model,seed=job
        path=folder(root,fold,model,seed)
        if not (path/"status.json").exists():
            cmd=python("v7_train.py","--fold",fold,"--model",model)
            if seed is not None:
                cmd+=["--seed",str(seed)]
            if epoch is not None and model in NEURAL:
                cmd+=["--fixed-epoch",str(epoch[model])]
            record(root,f"{fold}_{path.name}",cmd)
        status=load(path/"status.json")
        if status["status"]!="complete":
            raise ValueError(f"Incomplete job {path}")
        return dict(fold=fold,model=model,seed=seed,job=path.name,best_epoch=status["best_epoch"],AP=status["AP"])
    # Interleave comparators across folds while preserving fixed specs and samples.
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        return list(pool.map(one,jobs))


def development(root,cfg):
    rows=execute(root,cfg,list(cfg["folds"]))
    medians={m:int(statistics.median(x["best_epoch"] for x in rows if x["model"]==m)) for m in NEURAL}
    save(root/"results/development_receipt.json",dict(at_utc=now(),models=rows,epochs=medians,
         rule="Median of three forward folds and three seed checkpoints, AP per fold",heldout_used=False,primary="lead_attention"))
    print(dict(runs=len(rows),epochs=medians),flush=True)


def calibration(root,cfg):
    select=load(root/"results/development_receipt.json")
    rows=execute(root,cfg,["final"],select["epochs"])
    save(root/"results/calibration_receipt.json",dict(at_utc=now(),models=rows,interval=cfg["calibration"],
          warning_rule="Maximum balanced accuracy, tie higher threshold",current_rule="Prior maximum F1",
          heldout_used=False))
    print(dict(runs=len(rows),interval=cfg["calibration"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("development","calibration"))
    globals()[p.parse_args().stage](*settings())

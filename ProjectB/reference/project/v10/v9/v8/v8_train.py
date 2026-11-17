"""Single V8 rank-loss ablation using only V7 lead inputs and training labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from v8_io import V7,load,now,python,record,save,settings,sha
from v8_check import register
sys.path.insert(0,str(V7))
from v7_train import build,predict,read_npz,sample_training,set_seed,study_for
from v7_score import balanced_cutoff
from v4_train import rank_penalty


def arrays(root,cfg,fold,alternate=None):
    if fold not in (*cfg["folds"],"final"):
        raise PermissionError("V8 fitting cannot load a holdout")
    ref=load(root/"evidence/inputs.json")["prepared"][fold]
    path=alternate/"prepared"/f"lead_{fold}.npz" if alternate else Path(ref["path"])
    if sha(path)!=ref["sha256"]:
        raise ValueError("Prepared lead input SHA differs")
    data=read_npz(path)
    day=data["minute"]//1440
    assert day.max()<np.datetime64(cfg["embargo"][0],"D").astype(int)
    period=cfg["calibration"] if fold=="final" else cfg["folds"][fold]
    np.testing.assert_array_equal(np.unique(day[data["split"]==1]).astype("datetime64[D]"),
                                  np.arange(*map(np.datetime64,period),dtype="datetime64[D]"))
    assert data["eligible"].all() and (data["aux"]<0).all()
    return data


def reference(root,fold,model,seed):
    return root/"results/reference"/fold/f"lead_{model}_s{seed}"


def candidate(root,fold,seed):
    return root/"results/candidate"/fold/f"rank_s{seed}"


def fit(root,cfg,fold,seed,fixed):
    register(root,cfg)
    if (root/"evidence/freeze.json").exists():
        raise PermissionError("V8 is frozen")
    path=candidate(root,fold,seed)
    path.mkdir(parents=True,exist_ok=False)
    study=study_for(arrays(root,cfg,fold),"X1")
    train,sampling=sample_training(study,cfg)
    original=reference(root,fold,"attention",seed)
    published=read_npz(original/"sampling_ids.npz")
    np.testing.assert_array_equal(published["sample_id"],study.keys(train))
    np.testing.assert_array_equal(published["y"],study.y[train])
    np.savez_compressed(path/"sampling_ids.npz",sample_id=study.keys(train),y=study.y[train])
    save(path/"sampling.json",sampling)
    prep=study.fit_preprocessor()
    prep.save(path/"preprocessor.npz")
    if sha(path/"preprocessor.npz")!=sha(original/"preprocessor.npz"):
        raise ValueError("Input scaling differs from V7")
    study.configure(prep)
    set_seed(seed,cfg)
    net=build(study.nfeatures*2,cfg,"attention",seed)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg["lr"])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(sampling["positive_weight"],dtype=torch.float32))
    y=torch.from_numpy(study.y.astype(np.float32))
    rng=np.random.default_rng(seed)
    start,process=time.perf_counter(),psutil.Process()
    peak=process.memory_info().rss
    best,epoch,state=-1.,0,None
    curve=[]
    for e in range(1,cfg["epochs"]+1):
        net.train()
        order=rng.permutation(train)
        total=extra=0.
        for off in range(0,len(order),cfg["batch_size"]):
            ids=order[off:off+cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            z=net(*map(torch.from_numpy,study.batch(ids)))
            bce=criterion(z,y[ids])
            tail=rank_penalty(z,y[ids],cfg["v8_rank_tail_fraction"])
            (bce+cfg["v8_rank_weight"]*tail).backward()
            optimizer.step()
            total+=len(ids)*float(bce.detach())
            extra+=len(ids)*float(tail.detach())
        ap=None
        if fixed is None:
            _,values=predict(net,study,study.val,cfg)
            ap=float(average_precision_score(study.y[study.val],values))
        if e==fixed or (ap is not None and ap>best):
            best=ap if ap is not None else best
            epoch=e
            state={k:v.detach().clone() for k,v in net.state_dict().items()}
        peak=max(peak,process.memory_info().rss)
        curve.append(dict(epoch=e,bce=total/len(order),tail=extra/len(order),AP=ap,
                          updates=math.ceil(len(order)/cfg["batch_size"])))
    if state is None:
        raise ValueError("No rank checkpoint selected")
    net.load_state_dict(state)
    torch.save(net.state_dict(),path/"weight.pt")
    pd.DataFrame(curve).to_csv(path/"curve.csv",index=False,na_rep="N/A")
    ids=np.flatnonzero(study.phase==1)
    logits,p=predict(net,study,ids,cfg)
    score=p.astype(np.float64)
    raw=dict(sample_id=study.keys(ids),y=study.y[ids],risk=study.risk[ids],score=score,logit=logits,
             probability=p,current_eligible=study.d["current_eligible"].ravel()[ids])
    v7=read_npz(original/"assessment.npz")
    for key in ("sample_id","y","risk","current_eligible"):
        np.testing.assert_array_equal(raw[key],v7[key])
    known=raw["y"]>=0
    cutoff=balanced_cutoff(raw["y"][known],score[known])
    np.savez_compressed(path/"assessment.npz",**raw,threshold=np.float64(cutoff),pred=score>=cutoff)
    status=dict(status="complete",fold=fold,seed=seed,best_epoch=epoch,fixed_epoch=fixed,
        AP=float(average_precision_score(raw["y"][known],score[known])),fit_and_score_seconds=time.perf_counter()-start,
        peak_RSS_bytes=peak,updates=sum(x["updates"] for x in curve),parameters=sum(v.numel() for v in net.parameters()),
        weight_sha256=sha(path/"weight.pt"),completed_utc=now(),sampling_matches_V7=True,
        routed=False,loss_uses_future_risk=False)
    save(path/"status.json",status)
    print(status,flush=True)


def jobs(root,cfg,folds,fixed=None):
    def single(spec):
        fold,seed=spec
        path=candidate(root,fold,seed)
        command=python("v8_train.py","fit","--fold",fold,"--seed",seed)
        if fixed is not None:
            command+=["--fixed",str(fixed)]
        if not (path/"status.json").is_file():
            record(root,f"rank_{fold}_s{seed}",command)
        return load(path/"status.json")
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        return list(pool.map(single,product(folds,cfg["seeds"])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("fit","matrix"))
    p.add_argument("--fold")
    p.add_argument("--seed",type=int)
    p.add_argument("--fixed",type=int)
    a=p.parse_args()
    root,cfg=settings()
    if a.stage=="fit":
        fit(root,cfg,a.fold,a.seed,a.fixed)
    else:
        dev=jobs(root,cfg,list(cfg["folds"]))
        epoch=int(statistics.median(r["best_epoch"] for r in dev))
        save(root/"results/rank_development.json",dict(rows=dev,median_epoch=epoch,completed_utc=now()))
        cal=jobs(root,cfg,["final"],epoch)
        save(root/"results/rank_calibration.json",dict(rows=cal,fixed_epoch=epoch,completed_utc=now()))
        print(dict(development=len(dev),calibration=len(cal),checkpoint=epoch),flush=True)


if __name__=="__main__":
    main()

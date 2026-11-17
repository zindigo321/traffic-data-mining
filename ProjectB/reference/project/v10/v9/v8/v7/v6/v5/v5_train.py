"""Matched V4 rerun and NYC-only lag/auxiliary Graph Transformer candidates."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from v5_io import load, now, save, settings, sha
from v5_prepare import BASIC_FEATURES, HALF_NAMES, local_dates, read_npz
from v4_train import Study, fit_paired, rank_penalty, matched_indices, typed_score
from v4_metrics import matched_points
from v3_train import epoch_ids, risk_metrics
from experiment import predict,set_seed
from evaluation import measure,threshold
from model import build,primary
from v1.graph import TrainPreprocessor

RECIPES=("B","X1","X2","X3")
HALF=("X1","X3")
AUX=("X2","X3")


def study_for(data,recipe):
    if recipe not in RECIPES or data["x"].shape[-1]!=len(BASIC_FEATURES)+len(HALF_NAMES):
        raise ValueError("Unexpected external feature contract")
    study=Study(data,"R2")
    if recipe in HALF:
        study.nfeatures+=len(HALF_NAMES)
    return study


def pinned_development(root,cfg,fold):
    if fold not in (*cfg["folds"],"final"):
        raise PermissionError("Held-out data cannot be loaded by training")
    path=root/"prepared"/f"{fold}.npz"
    evidence=load(root/"evidence/prepared_development.json")
    if evidence["holdout_parsed"] or evidence["hashes"][fold]!=sha(path):
        raise ValueError("Prepared data changed")
    data=read_npz(path)
    period=cfg["calibration"] if fold=="final" else cfg["folds"][fold]
    days=data["minute"]//1440
    assert set(np.unique(data["split"]))=={0,1}
    for phase,interval in ((0,[cfg["train_start"],period[0]]),(1,period)):
        expected=local_dates([interval]).astype(np.int64)
        np.testing.assert_array_equal(np.unique(days[data["split"]==phase]),expected)
    if period[1]>cfg["holdout"][0]:
        raise ValueError("Illegal development interval")
    return data


def fit_aux(study,ids,budget,stat,opt,cfg):
    seed,pool,epoch_limit=opt["seed"],opt["pool"],opt["fixed_epoch"]
    net=build(study.nfeatures*2,cfg,pool,seed,auxiliary=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg["lr"])
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(stat["pos_weight"],dtype=torch.float32))
    huber=nn.SmoothL1Loss(beta=cfg["aux_beta"])
    y=torch.from_numpy(study.y.astype(np.float32))
    target=torch.from_numpy(study.d["recovery_target"].ravel())
    if not torch.isfinite(target[ids]).all():
        raise ValueError("Unknown auxiliary target selected for training")
    valid=study.val[study.risk[study.val]]
    begin,process=time.perf_counter(),psutil.Process()
    peak,best_ap,best_epoch,weights=process.memory_info().rss,-1.,0,None
    rng=np.random.default_rng(seed)
    curve=[]
    for e in range(1,cfg["epochs"]+1):
        net.train()
        ordered=epoch_ids(ids,budget,rng)
        totals=np.zeros(3)
        for off in range(0,budget,cfg["batch_size"]):
            index=ordered[off:off+cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output=net(*map(torch.from_numpy,study.batch(index)))
            z=primary(output)
            losses=(bce(z,y[index]),rank_penalty(z,y[index],cfg["tail_fraction"]),huber(output[:,1],target[index]))
            (losses[0]+cfg["rank_weight"]*losses[1]+cfg["aux_weight"]*losses[2]).backward()
            optimizer.step()
            totals+=np.array([float(v.detach()) for v in losses])*len(index)
        ap=None
        if epoch_limit is None:
            _,p=predict(net,study,valid,cfg)
            ap=float(average_precision_score(study.y[valid],p))
        if (ap is not None and ap>best_ap) or e==epoch_limit:
            best_ap=ap if ap is not None else best_ap
            best_epoch=e
            weights={k:v.detach().clone() for k,v in net.state_dict().items()}
        peak=max(peak,process.memory_info().rss)
        values=totals/budget
        curve.append(dict(epoch=e,conditional_AP=ap,weighted_BCE=values[0],rank_tail=values[1],aux_huber=values[2],
                          updates=math.ceil(budget/cfg["batch_size"])))
    if weights is None:
        raise ValueError("No epoch selected")
    net.load_state_dict(weights)
    return net,curve,dict(fit_seconds=time.perf_counter()-begin,peak_RSS_bytes=peak,best_epoch=best_epoch,
        updates=sum(row["updates"] for row in curve),parameters=sum(p.numel() for p in net.parameters()))


def fit(root,cfg,opt,folder):
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/"options.json",opt)
    fold,recipe=opt["fold"],opt["recipe"]
    data=pinned_development(root,cfg,fold)
    study=study_for(data,recipe)
    if any(np.unique(study.y[part]).size!=2 for part in (study.train,study.val)):
        raise ValueError("Known assessment/training class is absent")
    set_seed(opt["seed"],cfg)
    prep=study.fit_preprocessor()
    prep.save(folder/"preprocessor.npz")
    study.configure(prep)
    ids,budget,stat=matched_indices(study,cfg)
    save(folder/"sampling.json",{**stat,"seed":cfg["sampling_seed"]})
    np.savez_compressed(folder/"sampling_ids.npz",sample_id=study.keys(ids),label=study.y[ids],risk=study.risk[ids])
    if recipe in AUX:
        net,curve,cost=fit_aux(study,ids,budget,stat,opt,cfg)
    else:
        net,curve,cost=fit_paired(study,ids,budget,stat,opt,cfg,cfg)
    torch.save(net.state_dict(),folder/"weight.pt")
    pd.DataFrame(curve).to_csv(folder/"curve.csv",index=False,na_rep="N/A")
    start=time.perf_counter()
    z=typed_score(study,study.val,*predict(net,study,study.val,cfg))
    predict_seconds=time.perf_counter()-start
    cut=threshold(z["y"],z["score"])
    cov=load(root/"evidence"/f"fold_{fold}.json")["parts"][1]
    metrics=measure(z["y"],z["score"],cut,z["sample_id"],cov)
    risk=risk_metrics(z,cut)
    if not math.isclose(metrics["AP"],risk["risk_AP"],rel_tol=1e-12,abs_tol=1e-14):
        raise ValueError("Routed AP must equal risk AP under this support")
    points,_=matched_points(z["sample_id"],z["y"],z["score"],z["risk"],cfg["recalls"],cov["total_days"])
    save(folder/"matched_recall.json",points)
    np.savez_compressed(folder/"assessment.npz",**z,threshold=np.float64(cut),pred=z["score"]>=cut)
    point=next(r for r in points if r["target_recall"]==cfg["main_recall"])
    payload=dict(status="complete",**opt,**cost,**metrics,**risk,predict_seconds=predict_seconds,
                 FP60_per_day=point["FP_per_day"],FP60=point["FP"],TP60=point["TP"],features=study.nfeatures,
                 logit_dtype=str(z["logit"].dtype),probability_dtype=str(z["probability"].dtype),
                 score_dtype=str(z["score"].dtype),weight_sha256=sha(folder/"weight.pt"),completed_utc=now())
    save(folder/"status.json",payload)
    print(payload,flush=True)


def model_for(cfg,study,opt,path):
    set_seed(opt["seed"],cfg)
    net=build(study.nfeatures*2,cfg,opt["pool"],opt["seed"],opt["recipe"] in AUX)
    net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return net


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fold",required=True,choices=("f1","f2","f3","final"))
    p.add_argument("--recipe",required=True,choices=RECIPES)
    p.add_argument("--pool",required=True,choices=("mean","max","attention"))
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--fixed-epoch",type=int)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    fit(*settings(),{k:getattr(a,k) for k in ("fold","recipe","pool","seed","fixed_epoch")},a.output)

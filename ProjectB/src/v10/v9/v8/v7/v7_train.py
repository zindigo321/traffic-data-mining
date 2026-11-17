"""V7 matched warning fits and inherited current-window comparators."""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from v7_io import V6,load,now,save,settings,sha
from v7_score import cutoff,evaluate,slice_scores
sys.path.insert(0,str(V6/"v5"))
from v5_train import study_for,fit as inherited_fit,read_npz,build,predict,set_seed
from experiment import proportional

NEURAL=("lead_mean","lead_max","lead_attention","lead_center","current_attention")
DETERMINISTIC=("lead_logit","lead_lag","lead_majority","current_rule","current_lag","current_majority")


def specifications(cfg):
    return [(model,seed) for model in NEURAL for seed in cfg["seeds"]]+[(model,None) for model in DETERMINISTIC]


def folder(root,fold,model,seed):
    return root/"results"/("calibration" if fold=="final" else f"development/{fold}")/(f"{model}_s{seed}" if seed is not None else model)


def dataset(root,cfg,fold,lead):
    if fold not in (*cfg["folds"],"final"):
        raise PermissionError("Fitting loader is restricted to development and calibration")
    prefix="lead_" if lead else ""
    file=root/"prepared"/f"{prefix}{fold}.npz"
    proof=load(root/"evidence/prepared_development.json")
    if proof["holdout_parsed"] or sha(file)!=proof["lead_hashes" if lead else "hashes"][fold]:
        raise ValueError("Prepared file digest differs")
    data=read_npz(file)
    period=cfg["calibration"] if fold=="final" else cfg["folds"][fold]
    minutes=data["minute"]
    if (minutes//1440).max()>=np.datetime64(cfg["embargo"][0],"D").astype(int):
        raise ValueError("An old holdout entered the fitted phase")
    assessed=np.unique((minutes[data["split"]==1]//1440).astype("datetime64[D]"))
    np.testing.assert_array_equal(assessed,np.arange(*map(np.datetime64,period),dtype="datetime64[D]"))
    return data


def sample_training(study,cfg):
    # Both subsets contain only train-known y. Actual t-1 risk is not read.
    p=study.train[study.y[study.train]==1]
    n=study.train[study.y[study.train]==0]
    if not len(p) or not len(n):
        raise ValueError("Training requires both labels")
    t,r=np.divmod(n,study.nroad)
    m=study.d["minute"][t]
    strata=((m//1440-(m//1440).min())*2+(m%1440>=720))*study.nroad+r
    other=proportional(n,strata,min(len(n),cfg["negative_budget"]),np.random.default_rng(cfg["sampling_seed"]))
    ids=np.sort(np.r_[p,other])
    return ids,dict(train_positive=len(p),train_negative=len(n),sample_positive=len(p),sample_negative=len(other),
                     positive_weight=len(other)/len(p),seed=cfg["sampling_seed"],strategy="train day, peak, road")


def fit_warning(study,train,sample,opts,cfg):
    central=opts["model"]=="lead_center"
    pool="attention" if central else opts["model"].split("_")[-1]
    seed,locked=opts["seed"],opts["fixed_epoch"]
    set_seed(seed,cfg)
    model=build(study.nfeatures*2,cfg,pool,seed)
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(sample["positive_weight"],dtype=torch.float32))
    y=torch.from_numpy(study.y.astype(np.float32))
    rng=np.random.default_rng(seed)
    start,process=time.perf_counter(),psutil.Process()
    peak=process.memory_info().rss
    best,best_epoch,weights=-1.,0,None
    curve=[]
    for epoch in range(1,cfg["epochs"]+1):
        model.train()
        order=rng.permutation(train)
        total=0.
        for i in range(0,len(order),cfg["batch_size"]):
            index=order[i:i+cfg["batch_size"]]
            opt.zero_grad(set_to_none=True)
            z=model(*map(torch.from_numpy,study.batch(index,central)))
            loss=criterion(z,y[index])
            loss.backward()
            opt.step()
            total+=float(loss.detach())*len(index)
        ap=None
        if locked is None:
            _,pr=predict(model,study,study.val,cfg,central)
            ap=float(average_precision_score(study.y[study.val],pr))
        if (ap is not None and ap>best) or locked==epoch:
            best=ap if ap is not None else best
            best_epoch=epoch
            weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        peak=max(peak,process.memory_info().rss)
        curve.append(dict(epoch=epoch,loss=total/len(order),AP=ap,updates=math.ceil(len(order)/cfg["batch_size"])))
    if weights is None:
        raise ValueError("No checkpoint")
    model.load_state_dict(weights)
    return model,curve,dict(fit_seconds=time.perf_counter()-start,best_epoch=best_epoch,peak_RSS_bytes=peak,
         parameters=sum(x.numel() for x in model.parameters()),updates=sum(x["updates"] for x in curve))


def lag_score(value,fill):
    return np.where(np.isfinite(value),1/(1+np.maximum(value,0)),fill)


def run(root,cfg,fold,model,seed,fixed_epoch):
    if (root/"evidence/calibration_freeze.json").exists():
        raise PermissionError("Calibration is already frozen")
    path=folder(root,fold,model,seed)
    if model=="current_attention":
        inherited_fit(root,cfg,dict(fold=fold,recipe="X1",pool="attention",seed=seed,fixed_epoch=fixed_epoch),path)
        return
    path.mkdir(parents=True,exist_ok=False)
    lead=model.startswith("lead_")
    data=dataset(root,cfg,fold,lead)
    study=study_for(data,"X1")
    ids=np.flatnonzero((study.phase==1)&study.eligible)
    opts=dict(fold=fold,model=model,seed=seed,fixed_epoch=fixed_epoch)
    save(path/"options.json",opts)
    cover=load(root/"evidence"/f"fold_{'lead_' if lead else ''}{fold}.json")["parts"][1]
    model_state={}
    cost=dict(fit_seconds=0.,best_epoch=None,parameters=0,updates=0)
    begin=time.perf_counter()
    if model in NEURAL or model=="lead_logit":
        train,stat=sample_training(study,cfg)
        save(path/"sampling.json",stat)
        np.savez_compressed(path/"sampling_ids.npz",sample_id=study.keys(train),y=study.y[train])
        prep=study.fit_preprocessor()
        prep.save(path/"preprocessor.npz")
        study.configure(prep)
        if model in NEURAL:
            net,curve,cost=fit_warning(study,train,stat,opts,cfg)
            torch.save(net.state_dict(),path/"weight.pt")
            pd.DataFrame(curve).to_csv(path/"curve.csv",index=False,na_rep="N/A")
            z,p=predict(net,study,ids,cfg,model=="lead_center")
            model_state.update(logit=z,probability=p)
            scores=p.astype(np.float64)
        else:
            reg=LogisticRegression(solver="lbfgs",max_iter=600,tol=1e-5)
            begin_fit=time.perf_counter()
            reg.fit(study.flat(train),study.y[train],sample_weight=np.where(study.y[train]==1,stat["positive_weight"],1.))
            cost=dict(fit_seconds=time.perf_counter()-begin_fit,best_epoch=None,
                      parameters=reg.coef_.size+reg.intercept_.size,updates=int(reg.n_iter_[0]),converged=bool(reg.n_iter_[0]<600))
            np.savez_compressed(path/"logit_parameters.npz",coef=reg.coef_,intercept=reg.intercept_)
            scores=reg.predict_proba(study.flat(ids))[:,1].astype(np.float64)
    elif model.endswith("_lag"):
        eligible=study.ratio[study.train]
        observed=eligible[np.isfinite(eligible)]
        if not len(observed):
            raise ValueError("No legal lag in training")
        fill=float(np.median(lag_score(observed,0.)))
        save(path/"rule.json",dict(missing_score=fill,lag_minutes=20 if lead else 10,source="train observed median"))
        scores=lag_score(study.ratio[ids],fill)
    elif model=="current_rule":
        scores=study.risk[ids].astype(np.float64)
    else:
        scores=np.zeros(len(ids),dtype=np.float64)
    z=dict(sample_id=study.keys(ids),y=study.y[ids],risk=study.risk[ids],score=scores,**model_state)
    if lead:
        z["current_eligible"]=data["current_eligible"].ravel()[ids]
    known=z["y"]>=0
    cut=cutoff(model,z["y"][known],scores[known])
    metrics,points=evaluate(slice_scores(z,known),cut,cfg,cover)
    np.savez_compressed(path/"assessment.npz",**z,threshold=np.float64(cut),pred=scores>=cut)
    save(path/"matched_recall.json",points)
    result=dict(status="complete",**opts,**cost,**metrics,elapsed_seconds=time.perf_counter()-begin,
                output_rows=len(ids),routed=False,completed_utc=now())
    save(path/"status.json",result)
    print({k:result[k] for k in ("model","fold","seed","AP","balanced_accuracy","f1","FP60_day","recovery_FP60_day")},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fold",required=True)
    p.add_argument("--model",required=True,choices=NEURAL+DETERMINISTIC)
    p.add_argument("--seed",type=int)
    p.add_argument("--fixed-epoch",type=int)
    args=p.parse_args()
    run(*settings(),args.fold,args.model,args.seed,args.fixed_epoch)

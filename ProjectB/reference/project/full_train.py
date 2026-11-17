"""Actual full-data three-seed graph pooling and center controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import math
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from full_check import register
from full_data import (MODELS,arrays,training_ids,epoch_order,folder,central,pool,groups,study_for,
                       build,predict,set_seed,read_npz)
from full_io import load,now,python,record,save,settings,sha
from v10_train import low_group_rank


def fit(root,cfg,stage,model,seed,fixed):
    register(root,cfg)
    if (root/"evidence/model_threshold_freeze.json").exists():
        raise PermissionError("Evaluation models and thresholds have been frozen")
    if stage=="final":
        select=load(root/"results/development_selection.json")
        assert fixed==select["epochs"][model]
    path=folder(root,stage,model,seed)
    path.mkdir(parents=True,exist_ok=False)
    study=study_for(arrays(root,stage),"X1")
    ids,count=training_ids(study,cfg,model)
    if count==0:
        save(path/"status.json",dict(status="no_known_training_labels",model=model,stage=stage,seed=seed,best_epoch=None))
        return
    y_train=study.y[ids]
    positive=int((y_train==1).sum())
    negative=int((y_train==0).sum())
    group=groups(study.ratio,cfg["ratio"])
    np.savez_compressed(path/"training_ids.npz",sample_id=study.keys(ids),y=y_train,legal_group=group[ids])
    save(path/"training_support.json",dict(full_known=count,unique_training=len(ids),positive=positive,negative=negative,
        uses_all_known=model!="center_sampled_matched",repeated_only_for_sampled_control=model=="center_sampled_matched",
        epoch_exposures=count,epochs=cfg["epochs"],updates_per_epoch=math.ceil(count/cfg["batch_size"])))
    preprocessor=study.fit_preprocessor()
    preprocessor.save(path/"preprocessor.npz")
    study.configure(preprocessor)
    set_seed(seed,cfg)
    net=build(study.nfeatures*2,cfg,pool(model),seed)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg["lr"])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive if positive else 1.,dtype=torch.float32))
    y=torch.from_numpy(study.y.astype(np.float32))
    g=torch.from_numpy(group)
    rng=np.random.default_rng(seed)
    start,tick,process=now(),time.perf_counter(),psutil.Process()
    peak=process.memory_info().rss
    best,epoch,state=-1.,0,None
    curve=[]
    val_y=study.y[study.val]
    val_both=len(val_y)>0 and (val_y==1).any() and (val_y==0).any()
    for e in range(1,cfg["epochs"]+1):
        net.train()
        order=epoch_order(ids,count,rng)
        if model!="center_sampled_matched":
            assert len(order)==len(np.unique(order))==count
        total=extra=0.
        for off in range(0,len(order),cfg["batch_size"]):
            j=order[off:off+cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            z=net(*map(torch.from_numpy,study.batch(j,central(model))))
            bce=criterion(z,y[j])
            loss=bce
            ranking=z.sum()*0
            if model=="center_rank":
                ranking=low_group_rank(z,y[j],g[j],cfg["v10_rank_tail_fraction"])
                loss=loss+cfg["v10_rank_weight"]*ranking
            loss.backward()
            optimizer.step()
            total+=float(bce.detach())*len(j)
            extra+=float(ranking.detach())*len(j)
        ap=None
        if fixed is None and val_both:
            _,p=predict(net,study,study.val,cfg,central(model))
            ap=float(average_precision_score(val_y,p))
        take=e==fixed or (ap is not None and ap>best) or (not val_both and fixed is None and e==cfg["epochs"])
        if take:
            best=ap if ap is not None else best
            epoch=e
            state={k:v.detach().clone() for k,v in net.state_dict().items()}
        peak=max(peak,process.memory_info().rss)
        curve.append(dict(epoch=e,bce=total/count,rank=extra/count,AP=ap,training_exposures=count,
                          unique_seen=len(np.unique(order)),updates=math.ceil(count/cfg["batch_size"])))
    if state is None:
        raise RuntimeError("No valid checkpoint")
    net.load_state_dict(state)
    torch.save(net.state_dict(),path/"weight.pt")
    pd.DataFrame(curve).to_csv(path/"curve.csv",index=False,na_rep="N/A")
    assess=np.flatnonzero(study.phase>0)
    logits,p=predict(net,study,assess,cfg,central(model))
    np.savez_compressed(path/"assessment.npz",sample_id=study.keys(assess),y=study.y[assess],risk=study.risk[assess],
        legal_group=group[assess],phase=study.phase[assess],score=p.astype(np.float64),logit=logits,probability=p)
    result=dict(status="complete",stage=stage,model=model,seed=seed,started_utc=start,completed_utc=now(),
        fit_and_score_seconds=time.perf_counter()-tick,peak_RSS_bytes=peak,parameters=sum(p.numel() for p in net.parameters()),
        best_epoch=epoch,fixed_epoch=fixed,total_training_exposures=count*cfg["epochs"],
        updates=sum(x["updates"] for x in curve),known_train=count,unique_training=len(ids),
        full_known_coverage=len(ids)/count,weight_sha256=sha(path/"weight.pt"),evaluation_predictions=len(assess))
    save(path/"status.json",result)
    print(result,flush=True)


def baseline(root,cfg,stage,model):
    path=folder(root,stage,model,None)
    path.mkdir(parents=True,exist_ok=False)
    study=study_for(arrays(root,stage),"X1")
    train=study.train
    group=groups(study.ratio,cfg["ratio"])
    assess=np.flatnonzero(study.phase>0)
    begin=time.perf_counter()
    if model=="logistic":
        prep=study.fit_preprocessor()
        prep.save(path/"preprocessor.npz")
        study.configure(prep)
        features=study.x.reshape(-1,study.nfeatures*2)
        reg=LogisticRegression(max_iter=400,tol=1e-5,class_weight="balanced",random_state=cfg["seeds"][0])
        reg.fit(features[train],study.y[train])
        score=reg.predict_proba(features[assess])[:,1].astype(np.float64)
        np.savez_compressed(path/"logistic.npz",coef=reg.coef_,intercept=reg.intercept_)
        details=dict(converged=bool(reg.n_iter_[0]<400),iterations=int(reg.n_iter_[0]))
    elif model=="lag20_rule":
        ratio=study.ratio[train]
        observed=ratio[np.isfinite(ratio)]
        fill=float(np.median(1/(1+np.maximum(observed,0)))) if len(observed) else .5
        score=np.where(np.isfinite(study.ratio[assess]),1/(1+np.maximum(study.ratio[assess],0)),fill)
        save(path/"rule.json",dict(missing_fill=fill,lag_minutes=20))
        details={}
    else:
        majority=int((study.y[train]==1).sum()>(study.y[train]==0).sum())
        score=np.full(len(assess),float(majority))
        details=dict(majority=majority)
    np.savez_compressed(path/"assessment.npz",sample_id=study.keys(assess),y=study.y[assess],risk=study.risk[assess],
        legal_group=group[assess],phase=study.phase[assess],score=score)
    result=dict(status="complete",stage=stage,model=model,seed=None,best_epoch=None,known_train=len(train),
        fit_and_score_seconds=time.perf_counter()-begin,**details)
    save(path/"status.json",result)
    print(result,flush=True)


def execute(root,cfg,stage):
    selected=load(root/"results/development_selection.json") if stage=="final" else None
    def one(spec):
        model,seed=spec
        path=folder(root,stage,model,seed)
        if not (path/"status.json").exists():
            cmd=python("full_train.py","fit","--stage",stage,"--model",model,"--seed",seed)
            if selected:
                cmd+=["--fixed",selected["epochs"][model]]
            record(root,f"fit_{stage}_{model}_s{seed}",cmd)
        return load(path/"status.json")
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as executor:
        rows=list(executor.map(one,product(MODELS,cfg["seeds"])))
    for model in ("logistic","lag20_rule","majority"):
        path=folder(root,stage,model,None)
        if not (path/"status.json").exists():
            record(root,f"fit_{stage}_{model}",python("full_train.py","baseline","--stage",stage,"--model",model))
        rows.append(load(path/"status.json"))
    save(root/"results"/f"{stage}_training.json",dict(at_utc=now(),models=rows,stage=stage,
        future_blind_test_claim=False,training_admission_minimum_coverage=0))
    print(dict(stage=stage,neural_runs=18,deterministic_baselines=3),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("fit","matrix","baseline"))
    p.add_argument("--stage",choices=("development","final"),required=True)
    p.add_argument("--model")
    p.add_argument("--seed",type=int)
    p.add_argument("--fixed",type=int)
    a=p.parse_args()
    root,cfg=settings()
    if a.action=="matrix":
        execute(root,cfg,a.stage)
    elif a.action=="baseline":
        baseline(root,cfg,a.stage,a.model)
    else:
        fit(root,cfg,a.stage,a.model,a.seed,a.fixed)

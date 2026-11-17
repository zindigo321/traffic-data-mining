"""Paired V2 reruns and risk-conditional Graph Transformer ablations."""
import argparse
import math
from pathlib import Path
import time

from v3_prepare import FEATURE_NAMES, EXTRA, read_npz
from v3_io import context, load, now, save, sha

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from experiment import (Fold, platt, predict, proportional, sample, set_seed, train_model)
from evaluation import measure, rank_score, threshold
from model import build, primary
from v1.graph import TrainPreprocessor


class Study(Fold):
    def __init__(self, arrays, recipe):
        super().__init__(arrays,"V2")
        self.v3_recipe=recipe
        if recipe in ("R2","R3"):
            self.nfeatures=len(FEATURE_NAMES)+len(EXTRA)


def matched_indices(study,cfg):
    p=study.train[study.y[study.train]==1]
    n=study.train[study.y[study.train]==0]
    if not len(p) or not len(n):
        raise ValueError("Training must contain both known classes")
    if not study.risk[p].all():
        raise ValueError("A confirmed positive lacks previous low")
    budget=len(p)+min(len(n),cfg["negative_budget"])
    hard=n[study.risk[n]]
    if study.v3_recipe=="A":
        main,_,receipt=sample(study,{**cfg,"aux_seed":1971})
        assert len(main)==budget
        return main,budget,{**receipt,"known_risk_positive":len(p),"unique_per_epoch":len(main)}
    if not len(hard):
        raise ValueError("No recovery negative to condition on")
    rng=np.random.default_rng(cfg["sampling_seed"])
    keep=proportional(hard,study.strata(hard),budget-len(p),rng)
    ids=np.sort(np.r_[p,keep])
    receipt=dict(strategy="conditional_prior_low",known_risk_positive=len(p),known_hard=len(hard),known_negative=len(n),
                 sampled_hard=len(keep),main_positive=len(p),unique_per_epoch=len(ids),draws_per_epoch=budget,
                 hard_subsampled=len(keep)<len(hard),oversampling=budget>len(ids),
                 pos_weight=len(hard)/len(p))
    return ids,budget,receipt


def epoch_ids(ids,budget,rng):
    if not 0<len(ids)<=budget:
        raise ValueError("Invalid fixed exposure budget")
    q,r=divmod(budget,len(ids))
    draw=[rng.permutation(ids) for _ in range(q)]
    if r:
        draw.append(rng.permutation(ids)[:r])
    return np.concatenate(draw)


def train_risk(study,idx,budget,stat,opt,cfg):
    seed,pool,epoch_limit=opt["seed"],opt["pool"],opt["fixed_epoch"]
    use_aux=study.v3_recipe=="R3"
    net=build(study.nfeatures*2,cfg,pool,seed,use_aux)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg["lr"])
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(stat["pos_weight"],dtype=torch.float32))
    huber=nn.SmoothL1Loss(beta=cfg["aux_beta"]) if use_aux else None
    y=torch.from_numpy(study.y.astype(np.float32))
    target=None
    if use_aux:
        target=torch.from_numpy(study.d["recovery_target"].ravel())
        if not torch.isfinite(target[idx]).all():
            raise ValueError("Unknown recovery target in selected risk data")
    valid=study.val[study.risk[study.val]]
    if epoch_limit is None and np.unique(study.y[valid]).size!=2:
        raise ValueError("Development risk group does not have both classes")
    rng=np.random.default_rng(seed)
    begin=time.perf_counter()
    proc=psutil.Process()
    peak=proc.memory_info().rss
    best,best_epoch,best_weights=-1.,0,None
    curve=[]
    for e in range(1,cfg["epochs"]+1):
        net.train()
        ordered=epoch_ids(idx,budget,rng)
        total=aux_total=0.
        for off in range(0,budget,cfg["batch_size"]):
            keys=ordered[off:off+cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            z=net(*map(torch.from_numpy,study.batch(keys)))
            primary_loss=bce(primary(z),y[keys])
            loss=primary_loss
            if huber is not None:
                extra=huber(z[:,1],target[keys])
                loss=loss+cfg["aux_weight"]*extra
                aux_total+=float(extra.detach())*len(keys)
            loss.backward()
            optimizer.step()
            total+=float(primary_loss.detach())*len(keys)
        ap=None
        if epoch_limit is None:
            _,p=predict(net,study,valid,cfg)
            ap=float(average_precision_score(study.y[valid],p))
        if (ap is not None and ap>best) or e==epoch_limit:
            if ap is not None:
                best=ap
            best_epoch=e
            best_weights={k:v.detach().clone() for k,v in net.state_dict().items()}
        peak=max(peak,proc.memory_info().rss)
        curve.append(dict(epoch=e,conditional_AP=ap,weighted_BCE=total/budget,
                          recovery_huber=aux_total/budget if use_aux else None,updates=math.ceil(budget/cfg["batch_size"])))
    if best_weights is None:
        raise ValueError("No checkpoint was selected")
    net.load_state_dict(best_weights)
    return net,curve,dict(fit_seconds=time.perf_counter()-begin,peak_RSS_bytes=peak,best_epoch=best_epoch,
                          updates=sum(row["updates"] for row in curve),parameters=sum(p.numel() for p in net.parameters()))


def load_and_score(study,opt,cfg,folder,indices):
    set_seed(opt["seed"],cfg)
    net=build(study.nfeatures*2,cfg,opt["pool"],opt["seed"],study.v3_recipe=="R3")
    net.load_state_dict(torch.load(folder/"weight.pt",map_location="cpu",weights_only=True))
    return predict(net,study,indices,cfg)


def typed_score(study,ids,logits,p):
    return dict(sample_id=study.keys(ids),y=study.y[ids],logit=logits,probability=p,
                risk=study.risk[ids],risk_missing=~np.isfinite(study.ratio[ids]),score=rank_score(p,study.risk[ids],True))


def risk_metrics(z,t):
    take=z["risk"]
    y=z["y"][take]
    p=z["score"][take]
    positive=int(y.sum())
    negative=len(y)-positive
    return dict(
        risk_AP=float(average_precision_score(y,p)) if positive and negative else None,
        risk_positive=positive,risk_negative=negative,
        risk_FP=int(((y==0)&(p>=t)).sum()),risk_TP=int(((y==1)&(p>=t)).sum()),
        ordinary_FP=int(((z["y"]==0)&~z["risk"]&(z["score"]>=t)).sum()),
    )


def job(root,cfg,opt,out):
    out.mkdir(parents=True,exist_ok=False)
    save(out/"options.json",opt)
    fold=opt["fold"]
    data=Study(read_npz(root/"prepared"/f"{fold}.npz"),opt["recipe"])
    if not (len(np.unique(data.y[data.train]))==2 and len(np.unique(data.y[data.val]))==2):
        save(out/"status.json",dict(status="blocked",reason="Missing a class in train or assessment"))
        return
    set_seed(opt["seed"],cfg)
    prep=data.fit_preprocessor()
    prep.save(out/"preprocessor.npz")
    data.configure(prep)
    ids,budget,stat=matched_indices(data,cfg)
    save(out/"sampling.json",dict(**stat,draws_per_epoch=budget,seed=cfg["sampling_seed"])
         if "draws_per_epoch" not in stat else dict(**stat,seed=cfg["sampling_seed"]))
    np.savez_compressed(out/"sampling_ids.npz",sample_id=data.keys(ids),label=data.y[ids],risk=data.risk[ids])
    if opt["recipe"]=="A":
        net,curve,cost=train_model(data,ids,np.empty(0,dtype=np.int64),{**opt,"central":False},
                                   {**cfg,"aux_seed":1971})
    else:
        net,curve,cost=train_risk(data,ids,budget,stat,opt,cfg)
    torch.save(net.state_dict(),out/"weight.pt")
    pd.DataFrame(curve).to_csv(out/"curve.csv",index=False,na_rep="N/A")
    z=typed_score(data,data.val,*predict(net,data,data.val,cfg))
    cutoff=threshold(z["y"],z["score"])
    cover=load(root/"evidence"/f"fold_{fold}.json")["parts"][1]
    measured=measure(z["y"],z["score"],cutoff,z["sample_id"],cover)
    risk=risk_metrics(z,cutoff)
    if risk["risk_AP"] is not None and not math.isclose(measured["AP"],risk["risk_AP"],rel_tol=1e-12,abs_tol=1e-14):
        raise ValueError("Risk ordering is not equivalent to routed AP on complete support")
    prob_cal=platt(z["score"],z["y"]) if fold=="final" else None
    np.savez_compressed(out/"assessment.npz",**z,threshold=np.float64(cutoff),pred=z["score"]>=cutoff)
    payload=dict(status="complete",**opt,**cost,**measured,**risk,platt=prob_cal,
                 logit_dtype=str(z["logit"].dtype),probability_dtype=str(z["probability"].dtype),
                 score_dtype=str(z["score"].dtype),weight_sha256=sha(out/"weight.pt"),completed_utc=now())
    save(out/"status.json",payload)
    print({k:v for k,v in payload.items() if k!="platt"},flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fold",required=True)
    p.add_argument("--recipe",required=True,choices=["A","R1","R2","R3"])
    p.add_argument("--pool",required=True,choices=["mean","max","attention"])
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--fixed-epoch",type=int)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root,cfg=context()
    opt={k:getattr(a,k) for k in ("fold","recipe","pool","seed","fixed_epoch")}
    job(root,cfg,opt,a.output)


if __name__=="__main__":
    main()


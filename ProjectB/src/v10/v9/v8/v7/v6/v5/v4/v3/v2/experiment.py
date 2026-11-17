"""Matched chronological V0–V3 training, sampling, and typed predictions."""
import argparse
import math
from pathlib import Path
import random
import time

from v1 import runtime

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import digest, json_read, json_write, now, settings
from evaluation import measure, rank_score, sigmoid, threshold
from model import build, primary
from prepare_chicago import BASE_FEATURES, FEATURE_NAMES, key_matrix, read_npz
from v1.graph import TrainPreprocessor

RECIPES = ("V0","V1","V2","V3")


def set_seed(seed,cfg):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)


class Fold:
    def __init__(self, arrays, recipe):
        self.d = arrays
        self.recipe = recipe
        self.nroad = len(arrays["road_ids"])
        self.nfeatures = len(BASE_FEATURES) if recipe == "V0" else len(FEATURE_NAMES)
        self.y = arrays["y"].ravel()
        self.aux = arrays["aux"].ravel()
        self.risk = arrays["risk"].ravel()
        self.ratio = arrays["ratio_prior"].ravel()
        self.eligible = arrays["eligible"].ravel()
        self.phase = np.repeat(arrays["split"], self.nroad)
        self.train = np.flatnonzero(self.eligible & (self.y >= 0) & (self.phase == 0))
        self.val = np.flatnonzero(self.eligible & (self.y >= 0) & (self.phase == 1))

    def keys(self, ids):
        t,r = np.divmod(ids,self.nroad)
        return self.d["minute"][t]//10 * 1000 + self.d["road_ids"][r]

    def strata(self, ids):
        t,r = np.divmod(ids,self.nroad)
        m = self.d["minute"][t]
        # Unix civil day zero was Thursday. 0=Monday.
        weekend = ((m//1440+3)%7 >= 5).astype(np.int64)
        slot = (m%1440)//10
        return r * (2*144) + weekend * 144 + slot

    def fit_preprocessor(self):
        raw = self.d["x"][self.d["split"]==0,:,:self.nfeatures]
        return TrainPreprocessor().fit(raw, np.ones(raw.shape[:2],dtype=bool))

    def configure(self, prep):
        raw = self.d["x"][...,:self.nfeatures]
        self.x = prep.transform(raw, np.ones(raw.shape[:2],dtype=bool))

    def batch(self, ids, central=False):
        d = self.d
        t,r = np.divmod(ids,self.nroad)
        if central:
            valid = np.ones((len(ids),1),dtype=bool)
            adj = np.zeros((len(ids),1,1),dtype=bool)
            x = self.x[t,r][:,None,:]
        else:
            valid = d["valid"][r]
            adj = d["adj"][r]
            x = self.x[t[:,None],d["lookup"][r]] * valid[...,None]
        return x, adj, valid, np.zeros(len(ids),dtype=np.int64)

    def flat(self, ids):
        result = []
        for offset in range(0,len(ids),2048):
            x,_,valid,_ = self.batch(ids[offset:offset+2048])
            mean = x.sum(1)/valid.sum(1)[:,None]
            maximum = np.where(valid[...,None],x,-np.inf).max(1)
            result.append(np.concatenate([x[:,0],mean,maximum],axis=-1))
        return np.concatenate(result).astype(np.float32)


def proportional(ids, strata, k, rng):
    if k >= len(ids):
        return ids
    if k == 0:
        return ids[:0]
    perm = rng.permutation(len(ids))
    order = perm[np.argsort(strata[perm], kind="stable")]
    _, starts, counts = np.unique(strata[order],return_index=True,return_counts=True)
    ideal = counts * (k/len(ids))
    quotas = np.floor(ideal).astype(np.int64)
    leftover = int(k-quotas.sum())
    if leftover:
        winners = np.lexsort((rng.random(len(quotas)), -(ideal-quotas)))[:leftover]
        quotas[winners] += 1
    selected = np.concatenate([order[s:s+q] for s,q in zip(starts,quotas,strict=True) if q])
    return ids[selected]


def sample(fold,cfg):
    train = fold.train
    pos = train[fold.y[train]==1]
    neg = train[fold.y[train]==0]
    rng = np.random.default_rng(cfg["sampling_seed"])
    budget = min(cfg["negative_budget"],len(neg))
    hard, ordinary = neg[fold.risk[neg]],neg[~fold.risk[neg]]
    hard_mode = fold.recipe in {"V2","V3"}
    if hard_mode:
        keep_hard = proportional(hard,fold.strata(hard),budget,rng)
        keep_ordinary = proportional(ordinary,fold.strata(ordinary),budget-len(keep_hard),rng)
        neg_chosen = np.concatenate([keep_hard,keep_ordinary])
    else:
        neg_chosen = rng.choice(neg,size=budget,replace=False)
    main = np.sort(np.concatenate([pos,neg_chosen]))
    aux = np.empty(0,dtype=np.int64)
    if fold.recipe == "V3":
        eligible = np.flatnonzero(fold.eligible & (fold.phase == 0) & (fold.aux >= 0))
        aux_pos = eligible[fold.aux[eligible] == 1]
        aux_neg = eligible[fold.aux[eligible] == 0]
        rng = np.random.default_rng(cfg["aux_seed"])
        aux = np.sort(np.concatenate([aux_pos,proportional(aux_neg,fold.strata(aux_neg),4*len(aux_pos),rng)]))
    if len(np.unique(main)) != len(main) or len(main) != len(pos)+budget:
        raise ValueError("Sampler did not preserve fixed unique budget")
    return main,aux,dict(
        strategy="all_prior_low_then_proportional" if hard_mode else "uniform",
        main_positive=len(pos), known_negative=len(neg), sampled_negative=len(neg_chosen),
        known_hard=len(hard), sampled_hard=int(fold.risk[neg_chosen].sum()),
        known_ordinary=len(ordinary), sampled_ordinary=int((~fold.risk[neg_chosen]).sum()),
        aux_total=len(aux), aux_low=int(fold.aux[aux].sum()), aux_main_unknown=int((fold.y[aux]<0).sum()),
    )


def predict(model,fold,ids,cfg,central=False):
    model.eval()
    zs,p = [],[]
    with torch.inference_mode():
        for offset in range(0,len(ids),cfg["eval_batch_size"]):
            j = ids[offset:offset+cfg["eval_batch_size"]]
            logit = primary(model(*map(torch.from_numpy,fold.batch(j,central))))
            zs.append(logit.numpy())
            p.append(logit.sigmoid().numpy())
    return np.concatenate(zs),np.concatenate(p)


def train_model(fold,train,aux,opts,cfg):
    seed,pool,central,fixed_epoch = [opts[k] for k in ("seed","pool","central","fixed_epoch")]
    model = build(fold.nfeatures*2,cfg,pool,seed,fold.recipe=="V3")
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg["lr"])
    y = torch.from_numpy(fold.y.astype(np.float32))
    a = torch.from_numpy(fold.aux.astype(np.float32))
    npos = int(fold.y[train].sum())
    loss_main = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train)-npos)/npos))
    if len(aux):
        low = int(fold.aux[aux].sum())
        loss_aux = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(aux)-low)/low))
    rng = np.random.default_rng(seed)
    rng_aux = np.random.default_rng(cfg["aux_seed"] + seed)
    start = time.perf_counter()
    process = psutil.Process()
    peak = process.memory_info().rss
    best_ap,best_state,best_epoch = -1.,None,0
    curve = []
    for epoch in range(1,cfg["epochs"]+1):
        model.train()
        order = rng.permutation(train)
        main_total = aux_total = 0.
        for offset in range(0,len(order),cfg["batch_size"]):
            ids = order[offset:offset+cfg["batch_size"]]
            if len(aux):
                aid = rng_aux.choice(aux,size=cfg["aux_batch"],replace=len(aux)<cfg["aux_batch"])
                combined = np.r_[ids,aid]
            else:
                combined = ids
            optimizer.zero_grad(set_to_none=True)
            z = model(*map(torch.from_numpy,fold.batch(combined,central)))
            main_loss = loss_main(primary(z)[:len(ids)],y[ids])
            total = main_loss
            if len(aux):
                aux_loss = loss_aux(z[len(ids):,1],a[aid])
                total = total+cfg["aux_weight"]*aux_loss
                aux_total += float(aux_loss.detach())
            total.backward()
            optimizer.step()
            main_total += float(main_loss.detach())*len(ids)
        ap = None
        if fixed_epoch is None:
            _,p = predict(model,fold,fold.val,cfg,central)
            ap = float(average_precision_score(fold.y[fold.val],p))
            chosen = ap > best_ap
        else:
            chosen = epoch == fixed_epoch
        if chosen:
            best_epoch = epoch
            best_ap = ap if ap is not None else best_ap
            best_state = {k:t.detach().clone() for k,t in model.state_dict().items()}
        steps = math.ceil(len(order)/cfg["batch_size"])
        peak = max(peak,process.memory_info().rss)
        curve.append(dict(epoch=epoch,weighted_BCE=main_total/len(train),aux_BCE=aux_total/steps if len(aux) else None,
                          development_AP=ap,updates=steps))
    if best_state is None:
        raise ValueError("No valid checkpoint")
    model.load_state_dict(best_state)
    cost = dict(fit_seconds=time.perf_counter()-start,peak_RSS_bytes=peak,best_epoch=best_epoch,
                updates=sum(r["updates"] for r in curve),parameters=sum(p.numel() for p in model.parameters()))
    return model,curve,cost


def score_arrays(fold,opts,cfg,out,ids=None):
    ids = fold.val if ids is None else ids
    algo = opts["algo"]
    if algo == "rule":
        p = fold.risk[ids].astype(np.float64)
        logit = np.where(p>0,1.,-1.)
    elif algo == "logreg":
        w = read_npz(out/"weight.npz")
        logit = fold.flat(ids).astype(np.float64) @ w["coef"] + w["intercept"][0]
        p = sigmoid(logit)
    else:
        set_seed(opts["seed"],cfg)
        model = build(fold.nfeatures*2,cfg,opts["pool"],opts["seed"],fold.recipe=="V3")
        model.load_state_dict(torch.load(out/"weight.pt",weights_only=True,map_location="cpu"))
        logit,p = predict(model,fold,ids,cfg,opts["central"])
    return dict(sample_id=fold.keys(ids).astype(np.int64), y=fold.y[ids].astype(np.int8),
                logit=logit, probability=p, risk=fold.risk[ids], risk_missing=~np.isfinite(fold.ratio[ids]))


def platt(raw, y):
    if not (np.any(y==0) and np.any(y==1)):
        return None
    # The calibration data alone estimates the slope and intercept. F1 uses the original score.
    score = raw.astype(np.float64)
    m = LogisticRegression(C=1000,solver="lbfgs",max_iter=300).fit(score[:,None],y)
    a,b = float(m.coef_[0,0]),float(m.intercept_[0])
    if a < 0:
        return dict(slope=0.,intercept=float(np.log(y.mean()/(1-y.mean()))),constant_fallback=True)
    return dict(slope=a,intercept=b,constant_fallback=False)


def fit_job(root,cfg,name,opts,destination):
    destination.mkdir(parents=True,exist_ok=False)
    json_write(destination/"options.json",opts)
    f = Fold(read_npz(root/"prepared"/f"{name}.npz"),opts["recipe"])
    tr,va = f.train,f.val
    cover = json_read(root/"evidence"/f"fold_{name}.json")["parts"][1]
    if not (len(np.unique(f.y[tr]))==2 and len(np.unique(f.y[va]))==2):
        json_write(destination/"status.json",dict(status="blocked",reason="Missing both classes in train or assessment"))
        return
    set_seed(opts["seed"],cfg)
    main,aux,sampling = sample(f,cfg)
    start = time.perf_counter()
    weights = None
    if opts["algo"] != "rule":
        pre = f.fit_preprocessor()
        pre.save(destination/"preprocessor.npz")
        f.configure(pre)
        np.savez_compressed(destination/"sampling_ids.npz",main_ids=f.keys(main),main_labels=f.y[main],
                            main_prior_low=f.risk[main],aux_ids=f.keys(aux),aux_labels=f.aux[aux],
                            aux_main_labels=f.y[aux])
        json_write(destination/"sampling.json",dict(**sampling,seed=cfg["sampling_seed"],aux_seed=cfg["aux_seed"]))
    if opts["algo"]=="gt":
        model,curve,cost = train_model(f,main,aux,opts,cfg)
        pd.DataFrame(curve).to_csv(destination/"curve.csv",index=False,na_rep="N/A")
        weights = "weight.pt"
        torch.save(model.state_dict(),destination/weights)
        logits,p = predict(model,f,va,cfg,opts["central"])
        saved = dict(sample_id=f.keys(va),y=f.y[va],logit=logits,probability=p,risk=f.risk[va],
                     risk_missing=~np.isfinite(f.ratio[va]))
    elif opts["algo"]=="logreg":
        xx = f.flat(main)
        lr = LogisticRegression(C=1.,class_weight="balanced",solver="lbfgs",max_iter=600).fit(xx,f.y[main])
        weights = "weight.npz"
        np.savez_compressed(destination/weights,coef=lr.coef_[0],intercept=lr.intercept_)
        saved = score_arrays(f,opts,cfg,destination)
        cost = dict(fit_seconds=time.perf_counter()-start,peak_RSS_bytes=psutil.Process().memory_info().rss,
                    best_epoch=None,updates=int(lr.n_iter_[0]),parameters=xx.shape[-1]+1)
    else:
        saved = score_arrays(f,opts,cfg,destination)
        cost = dict(fit_seconds=0.,peak_RSS_bytes=psutil.Process().memory_info().rss,best_epoch=None,updates=0,parameters=0)
    scored = {}
    variants = [False,True] if opts["algo"]=="gt" and not opts["central"] and opts["recipe"]!="V0" else [False]
    for route in variants:
        score = rank_score(saved["probability"],saved["risk"],route)
        cutoff = .5 if opts["algo"]=="rule" else threshold(saved["y"],score)
        key = "routed" if route else "raw"
        calibration = platt(score,saved["y"]) if name=="final" else None
        np.savez_compressed(destination/f"assessment_{key}.npz",**saved,score=score,
                            threshold=np.float64(cutoff),pred=score>=cutoff)
        scored[key] = dict(**measure(saved["y"],score,cutoff,saved["sample_id"],cover),
                           platt=calibration,score_dtype=str(score.dtype),logit_dtype=str(saved["logit"].dtype),
                           probability_dtype=str(saved["probability"].dtype))
    payload = dict(status="complete",fold=name,**opts,**cost,scored=scored,weight=weights,
                   weight_sha256=digest(destination/weights) if weights else None,completed_utc=now())
    json_write(destination/"status.json",payload)
    print(json_read(destination/"status.json"),flush=True)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--fold",required=True)
    cli.add_argument("--recipe",choices=RECIPES,required=True)
    cli.add_argument("--pool",choices=["mean","max","attention"],default="max")
    cli.add_argument("--seed",required=True,type=int)
    cli.add_argument("--algo",choices=["gt","logreg","rule"],default="gt")
    cli.add_argument("--central",action="store_true")
    cli.add_argument("--fixed-epoch",type=int)
    cli.add_argument("--destination",type=Path,required=True)
    args = cli.parse_args()
    root,cfg = settings()
    opts = {k:getattr(args,k) for k in ("recipe","pool","seed","algo","central","fixed_epoch")}
    fit_job(root,cfg,args.fold,opts,args.destination)


if __name__ == "__main__":
    main()




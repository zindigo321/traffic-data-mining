"""Independent audit of typed predictions, sample budgets, choice and frozen inference."""
import argparse
from datetime import datetime
import math
from pathlib import Path

from v3_io import CODE,context,load,save,sha
from v3_prepare import fingerprint_keys,local_dates,read_npz
from v3_train import Study,load_and_score
from v3_eval import locked

import numpy as np
import pandas as pd
from verify_v2 import independent
from v1.graph import TrainPreprocessor


def same(x,y,label):
    if x is None and y is None:
        return
    if x is None or y is None or not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12):
        raise AssertionError(f"{label}: {x} differs from {y}")


def day(ids):
    return (ids//1000//144).astype("datetime64[D]")


def recompute(z,row,threshold_rule=False):
    p=z["probability"].astype(np.float64)
    score=np.where(z["risk"],.51+.49*p,.49*p)
    np.testing.assert_array_equal(score,z["score"])
    t=float(z["threshold"])
    same(t,row["threshold"],"threshold")
    np.testing.assert_array_equal(score>=t,z["pred"])
    y=z["y"]
    assert z["risk"][y==1].all()
    assert len(np.unique(z["sample_id"]))==len(y)
    result=independent(y,score,t)
    for key,val in result.items():
        if key!="optimal_cutoff":
            same(val,row[key],key)
    if threshold_rule:
        same(result["optimal_cutoff"],t,"calibration F1 selection")
    risk=z["risk"]
    sub=independent(y[risk],score[risk],t)
    same(sub["AP"],row["risk_AP"],"risk AP")
    same(sub["fp"],row["risk_FP"],"risk FP")
    same(sub["tp"],row["risk_TP"],"risk TP")
    assert z["logit"].dtype==np.float32 and z["probability"].dtype==np.float32 and score.dtype==np.float64


def development(artifact,cfg):
    counts=[]
    calendar={}
    same_support={}
    unique_budgets={}
    for directory in (artifact/"results/dev",artifact/"results/calibration"):
        for status in sorted(directory.rglob("status.json")):
            row=load(status)
            if row["status"]!="complete":
                continue
            fold=row["fold"]
            if fold not in calendar:
                intervals=cfg["final_train"] if fold=="final" else cfg["folds"][fold]["train"]
                valid=cfg["calibration"] if fold=="final" else cfg["folds"][fold]["valid"]
                train,evaluation=local_dates(intervals),local_dates([valid])
                if np.intersect1d(train,evaluation).size or np.isin(np.r_[train,evaluation],local_dates(cfg["exclusions"])).any():
                    raise ValueError("Calendar contamination")
                calendar[fold]=(train,evaluation)
            train,evaluation=calendar[fold]
            base=status.parent
            source=read_npz(base/"sampling_ids.npz")
            assess=read_npz(base/"assessment.npz")
            budget=load(base/"sampling.json")["draws_per_epoch"]
            if not np.isin(day(source["sample_id"]),train).all() or not np.isin(day(assess["sample_id"]),evaluation).all():
                raise ValueError("Samples outside declared fold")
            if not np.isin(source["label"],[0,1]).all() or len(np.unique(source["sample_id"]))!=len(source["label"]):
                raise ValueError("Invalid training truth or identity")
            if row["recipe"]!="A" and not source["risk"].all():
                raise ValueError("Conditional fit included a non-risk item")
            expected=cfg["epochs"]*math.ceil(budget/cfg["batch_size"])
            if row["updates"]!=expected or unique_budgets.setdefault(fold,budget)!=budget:
                raise ValueError("Unequal epoch/update budget")
            signature=fingerprint_keys(assess["sample_id"])
            if same_support.setdefault(fold,signature)!=signature:
                raise ValueError("Assessment support differs across compared models")
            recompute(assess,row,threshold_rule=True)
            counts.append(dict(fold=fold,model=base.name,n=len(assess["y"]),train_unique=len(source["label"]),updates=expected))
    return counts


def verify_selection(artifact,cfg):
    scr=load(artifact/"results/screen_selection.json")
    conf=load(artifact/"results/development_selection.json")
    tab=pd.read_csv(artifact/"results/screen.csv")
    criterion=lambda a,b,s:(round(float(a),6),round(float(b),6),-round(float(s),6))
    raw={r:criterion(g.AP.mean(),g.f1.mean(),g.AP.std()) for r,g in tab.groupby("recipe")}
    recipe=max(cfg["recipes"],key=raw.get)
    assert recipe==scr["recipe"]==conf["recipe"]
    tab=pd.read_csv(artifact/"results/confirmation.csv")
    pool={}
    for p,g in tab.groupby("pool"):
        per=g.groupby("fold")[["AP","f1"]].mean()
        assert len(g)==len(cfg["seeds"])*len(cfg["folds"]) and set(per.index)==set(cfg["folds"])
        pool[p]=criterion(per.AP.mean(),per.f1.mean(),per.AP.std())
    selected=max(cfg["pools"],key=pool.get)
    assert selected==conf["pool"] and conf["holdout_used"] is False
    return conf


def event_starts(keys):
    k=set(map(int,keys))
    return sum((key-1000 not in k or (key//1000)%144==0) for key in k)


def replay(artifact,root,cfg,rows):
    path=root/"prepared/holdout.npz"
    if sha(path)!=load(artifact/"evidence/holdout_support.json")["array_sha256"]:
        raise ValueError("Prepared holdout SHA mismatch")
    a=read_npz(path)
    full=np.flatnonzero(a["eligible"].ravel())
    size=cfg["eval_batch_size"]
    choices=np.sort(np.random.default_rng(5711).choice(len(full)//size,3,replace=False))
    draw=np.concatenate([full[i*size:(i+1)*size] for i in choices])
    known=a["y"].ravel()[draw]>=0
    results=[]
    for row in rows:
        name=row["job"]
        folder=artifact/"results/calibration"/name
        opt=load(folder/"options.json")
        d=Study(a,opt["recipe"])
        d.configure(TrainPreprocessor.load(folder/"preprocessor.npz"))
        logits,p=load_and_score(d,opt,cfg,folder,draw)
        z=read_npz(artifact/"results/holdout"/f"{name}.npz")
        keys=d.keys(draw)[known]
        idx=np.searchsorted(z["sample_id"],keys)
        np.testing.assert_array_equal(keys,z["sample_id"][idx])
        np.testing.assert_allclose(p[known],z["probability"][idx],rtol=1e-6,atol=2e-7)
        np.testing.assert_allclose(logits[known],z["logit"][idx],rtol=1e-6,atol=2e-6)
        score=np.where(d.risk[draw][known],.51+.49*p[known].astype(float),.49*p[known].astype(float))
        np.testing.assert_array_equal(score>=row["threshold"],z["pred"][idx])
        results.append(dict(job=name,n=len(keys),max_abs_error=float(np.max(np.abs(p[known]-z["probability"][idx])))))
    return results


def verify(artifact,cfg,external=None):
    checked=development(artifact,cfg)
    if not (artifact/"results/development_selection.json").exists():
        return dict(status="PARTIAL",development_checks=checked,reason="Selection or source unavailable")
    choice=verify_selection(artifact,cfg)
    if not (artifact/"evidence/status.json").exists():
        return dict(status="PARTIAL",development_checks=checked,choice=choice,reason="No completed formal holdout")
    lock=locked(artifact,cfg)
    unlock=load(artifact/"evidence/holdout_unlock.json")
    assert lock["selection"]==choice
    assert sha(artifact/"evidence/freeze.json")==unlock["freeze_sha256"]
    assert datetime.fromisoformat(choice["at_utc"])<datetime.fromisoformat(lock["frozen_utc"])<datetime.fromisoformat(unlock["unlocked_utc"])
    assert not (unlock["prior_value_parse"] or unlock["fitting_allowed"] or unlock["selection_allowed"])
    cal_ids=lock["calibration_known_ids_sha256"]
    for model in lock["models"]:
        path=artifact/"results/calibration"/model["job"]
        row=load(path/"status.json")
        assert datetime.fromisoformat(row["completed_utc"])<=datetime.fromisoformat(lock["frozen_utc"])
        assert fingerprint_keys(read_npz(path/"assessment.npz")["sample_id"])==cal_ids
        same(row["threshold"],model["threshold"],"frozen calibration threshold")
    out=artifact/"results/holdout"
    z=read_npz(out/"support.npz")
    known=z["eligible"]&(z["y"]>=0)
    ids=z["sample_id"][known]
    y=z["y"][known]
    assert fingerprint_keys(z["sample_id"])==lock["expected_keys_sha256"]
    assert fingerprint_keys(ids)==load(artifact/"evidence/holdout_support.json")["known_sha256"]
    assert fingerprint_keys(z["sample_id"][z["eligible"]])==load(artifact/"evidence/holdout_features.json")["eligible_sha256"]
    np.testing.assert_array_equal(ids,z["known_ids"])
    assert np.isin(day(ids),local_dates([cfg["holdout"]])).all()
    assert not np.isin(day(ids),local_dates(cfg["exclusions"])).any()
    events=event_starts(ids[y==1])
    by_id={m["job"]:m for m in lock["models"]}
    rows=load(out/"metrics.json")
    assert len(rows)==len(by_id)==len(cfg["seeds"])*4
    for row in rows:
        scores=read_npz(out/f"{row['job']}.npz")
        np.testing.assert_array_equal(ids,scores["sample_id"])
        np.testing.assert_array_equal(y,scores["y"])
        np.testing.assert_array_equal(z["risk"][known],scores["risk"])
        recompute(scores,row)
        same(row["threshold"],by_id[row["job"]]["threshold"],"locked cutoff")
        assert row["selected"]==by_id[row["job"]]["selected"]
        same(events,row["true_events"],"truth event count")
        same(event_starts(ids[scores["pred"]]),row["alert_events"],"alert event count")
        same(row["fp"]/len(local_dates([cfg["holdout"]])),row["fp_per_day"],"FP/day")
        same(len(ids)/len(known),row["known_coverage"],"coverage")
    outcome=load(artifact/"evidence/status.json")
    selected=[r for r in rows if r["selected"]]
    ap=float(np.mean([r["AP"] for r in selected]))
    f1=float(np.mean([r["f1"] for r in selected]))
    same(ap,outcome["mean_AP"],"goal AP")
    same(f1,outcome["mean_F1"],"goal F1")
    assert bool(ap>=cfg["goal"] and f1>=cfg["goal"])==outcome["reached"]
    loads=replay(artifact,external,cfg,rows) if external else []
    return dict(status="PASS",verified_configurations=len(rows),n=len(ids),positive=int(y.sum()),events=events,
                development_and_calibration_checks=checked,selection=choice,replay=loads)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    root,cfg=context()
    result=verify(a.artifacts or root,cfg,a.replay_root)
    save(a.receipt,result)
    print({k:v for k,v in result.items() if k not in ("development_and_calibration_checks","replay")},flush=True)


if __name__=="__main__":
    main()


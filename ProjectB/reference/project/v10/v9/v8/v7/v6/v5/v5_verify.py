"""Independently verify frozen chronology, scores, matched points and state-dict replay."""
import argparse
from datetime import date
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from v5_io import load,now,save,settings,sha
from v5_lock import verify_lock
from v5_matrix import folder
from v5_prepare import fingerprint_keys,local_dates,read_npz
from v5_train import matched_indices,model_for,study_for
from v4_analysis import bootstrap_scores
from v4_matrix import average,criterion
from v4_verify import check_support,verify_point
from evaluation import event_metrics,rank_score,threshold
from model import primary
from v1.graph import TrainPreprocessor


def equal(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-13)


def validate_metrics(z,stats,points,cfg,cover):
    for name,dtype in (("sample_id",np.int64),("y",np.int8),("score",np.float64),
                       ("probability",np.float32),("logit",np.float32),("pred",np.bool_)):
        assert z[name].dtype==dtype
    ids,y,score=z["sample_id"],z["y"],z["score"]
    assert np.isin(y,[0,1]).all() and np.all(np.diff(ids)>0) and np.all(z["risk"][y==1])
    assert len(y)==stats["n"]==cover["negative"]+cover["positive"]
    assert y.sum()==stats["positive"]==cover["positive"]
    assert np.sum((y==0)&z["risk"])==stats["risk_negative"]==cover["high_risk_recovered"]
    np.testing.assert_array_equal(score,rank_score(z["probability"],z["risk"],True))
    assert z["threshold"].dtype==np.float64
    assert float(z["threshold"])==stats["threshold"]
    decision=score>=z["threshold"]
    np.testing.assert_array_equal(decision,z["pred"])
    assert tuple(confusion_matrix(y,decision,labels=[0,1]).ravel())==tuple(stats[k] for k in ("tn","fp","fn","tp"))
    assert equal(average_precision_score(y,score),stats["AP"]) and equal(roc_auc_score(y,score),stats["ROC_AUC"])
    assert equal(average_precision_score(y[z["risk"]],score[z["risk"]]),stats["risk_AP"])
    assert equal(f1_score(y,decision),stats["f1"])
    true_events=event_metrics(ids,y,decision)
    assert true_events["true_events"]==cover["true_events"]
    assert all(equal(true_events[k],stats[k]) for k in ("event_f1","event_recall","event_precision"))
    assert [a["target_recall"] for a in points]==cfg["recalls"]
    for point in points:
        verify_point(z,point,cfg,cover["total_days"])


def replay_batch(study,z,cfg,options,source,is_heldout):
    eligible=np.flatnonzero(study.eligible) if is_heldout else study.val
    np.testing.assert_array_equal(study.keys(eligible),z["sample_id"])
    np.testing.assert_array_equal(study.y[eligible],z["y"])
    np.testing.assert_array_equal(study.risk[eligible],z["risk"])
    positives=np.flatnonzero(z["y"]==1)
    bs=cfg["eval_batch_size"]
    start=(positives[len(positives)//2]//bs)*bs
    sl=slice(start,start+bs)
    selected=eligible[sl]
    t,r=np.divmod(selected,study.nroad)
    d=study.d
    valid=d["valid"][r]
    raw=d["x"][t[:,None],d["lookup"][r],:study.nfeatures]
    prep=TrainPreprocessor.load(source/"preprocessor.npz")
    x=prep.transform(raw,valid)
    model=model_for(cfg,study,options,source/"weight.pt").eval()
    with torch.inference_mode():
        values=primary(model(*map(torch.from_numpy,(x,d["adj"][r],valid,np.zeros(len(r),dtype=np.int64)))))
        probability=values.sigmoid().numpy()
    np.testing.assert_allclose(values.numpy(),z["logit"][sl],atol=1e-7,rtol=0)
    np.testing.assert_allclose(probability,z["probability"][sl],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(rank_score(probability,z["risk"][sl],True)>=z["threshold"],z["pred"][sl])
    return float(np.max(np.abs(probability-z["probability"][sl])))


def dev_checks(artifact,cfg,lock,replay_root):
    spec=lock["choice"]
    prep=load(artifact/"evidence/prepared_development.json")
    dev=[]
    replay=[]
    periods={**cfg["folds"],"final":cfg["calibration"]}
    for fold,interval in periods.items():
        dir=artifact/("results/calibration" if fold=="final" else f"results/dev/{fold}")
        covered=load(artifact/"evidence"/f"fold_{fold}.json")["parts"][1]
        expected={f"{r}_mean_s{cfg['seeds'][0]}" for r in ("B",*cfg["recipes"])} if fold!="final" else set()
        expected|={f"B_mean_s{s}" for s in cfg["seeds"]}
        expected|={f"{spec['recipe']}_{p}_s{s}" for p in cfg["pools"] for s in cfg["seeds"]}
        assert {item.name for item in dir.iterdir() if item.is_dir()}==expected
        reference,train_reference,data=None,None,None
        if replay_root:
            path=replay_root/"prepared"/f"{fold}.npz"
            assert sha(path)==prep["hashes"][fold]
            data=read_npz(path)
        for path in sorted(dir.iterdir()):
            if not path.is_dir():
                continue
            score=read_npz(path/"assessment.npz")
            metrics=load(path/"status.json")
            opt=load(path/"options.json")
            assert metrics["completed_utc"]<lock["frozen_utc"]
            validate_metrics(score,metrics,load(path/"matched_recall.json"),cfg,covered)
            assert float(score["threshold"])==threshold(score["y"],score["score"])
            check_support(score["sample_id"],[interval],[cfg["holdout"]])
            if reference is None:
                reference={k:score[k] for k in ("sample_id","y","risk","risk_missing")}
            for k,v in reference.items():
                np.testing.assert_array_equal(v,score[k])
            ids=read_npz(path/"sampling_ids.npz")
            check_support(ids["sample_id"],[[cfg["train_start"],interval[0]]],[cfg["holdout"]])
            assert ids["risk"].all() and np.isin(ids["label"],[0,1]).all()
            if train_reference is None:
                train_reference=ids
                if data is not None:
                    study=study_for(data,opt["recipe"])
                    drawn,_,_=matched_indices(study,cfg)
                    np.testing.assert_array_equal(study.keys(drawn),ids["sample_id"])
            for k,v in train_reference.items():
                np.testing.assert_array_equal(v,ids[k])
            assert metrics["weight_sha256"]==sha(path/"weight.pt")
            draw=load(path/"sampling.json")["draws_per_epoch"]
            assert metrics["updates"]==cfg["epochs"]*math.ceil(draw/cfg["batch_size"])
            if fold=="final":
                assert fingerprint_keys(score["sample_id"])==lock["calibration_known_sha256"]
            if data is not None:
                study=study_for(data,opt["recipe"])
                replay.append(replay_batch(study,score,cfg,opt,path,False))
            dev.append(metrics)
    assert len(dev)==54
    screening=load(artifact/"results/screen.json")
    for recipe in ("B",*cfg["recipes"]):
        rows=[r for r in dev if r["fold"]!="final" and (r["recipe"],r["pool"],r["seed"])==(recipe,"mean",cfg["seeds"][0])]
        assert average(rows)==screening["summary"][recipe]
    chosen=max(cfg["recipes"],key=lambda r:criterion(screening["summary"][r]))
    assert chosen==screening["recipe"]==spec["recipe"]
    for key,value in spec["summary"].items():
        r,p=key.split("_")
        rows=[row for row in dev if row["fold"]!="final" and (row["recipe"],row["pool"])==(r,p)]
        assert average(rows)==value
    assert spec["pool"]==max(cfg["pools"],key=lambda p:criterion(spec["summary"][f"{chosen}_{p}"]))
    return dev,replay


def held_checks(artifact,cfg,lock,replay_root):
    support=read_npz(artifact/"results/holdout/support.npz")
    receipt=load(artifact/"evidence/holdout_support.json")
    all_ids=support["sample_id"]
    eligible=support["eligible"]
    valid=eligible&(support["y"]>=0)
    assert np.isin(support["y"],[-1,0,1]).all()
    check_support(all_ids,[cfg["holdout"]],[])
    assert len(all_ids)==lock["expected_windows"]
    assert fingerprint_keys(all_ids)==lock["expected_grid_sha256"]==receipt["all_sha256"]
    assert fingerprint_keys(all_ids[eligible])==receipt["eligible_sha256"]
    assert fingerprint_keys(all_ids[valid])==receipt["known_sha256"]
    np.testing.assert_array_equal(support["known_ids"],all_ids[valid])
    np.testing.assert_array_equal(support["eligible_ids"],all_ids[eligible])
    coverage=load(artifact/"evidence/fold_holdout.json")["parts"][0]
    data=None
    if replay_root:
        path=replay_root/"prepared/holdout.npz"
        assert sha(path)==receipt["prepared_sha256"]
        data=read_npz(path)
        np.testing.assert_array_equal(data["eligible"].ravel(),eligible)
        np.testing.assert_array_equal(data["y"].ravel(),support["y"])
    result=[]
    replays=[]
    values={}
    points={}
    for spec in lock["models"]:
        folder=artifact/"results/holdout"/spec["job"]
        source=artifact/"results/calibration"/spec["job"]
        metrics=load(folder/"status.json")
        raw=read_npz(folder/"all_eligible.npz")
        np.testing.assert_array_equal(raw["sample_id"],all_ids[eligible])
        np.testing.assert_array_equal(raw["y"],support["y"][eligible])
        np.testing.assert_array_equal(raw["risk"],support["risk"][eligible])
        np.testing.assert_array_equal(raw["pred"],raw["score"]>=raw["threshold"])
        assert float(raw["threshold"])==spec["threshold"]==float(read_npz(source/"assessment.npz")["threshold"])
        known=raw["y"]>=0
        z={k:(v if k=="threshold" else v[known]) for k,v in raw.items()}
        np.testing.assert_array_equal(z["sample_id"],all_ids[valid])
        fixed=load(folder/"matched_recall.json")
        validate_metrics(z,metrics,fixed,cfg,coverage)
        if data is not None:
            opt=load(source/"options.json")
            study=study_for(data,opt["recipe"])
            replays.append(replay_batch(study,raw,cfg,opt,source,True))
        key=(spec["recipe"],spec["pool"],spec["seed"])
        values[key]=z
        points[key]=next(p for p in fixed if p["target_recall"]==cfg["main_recall"])
        result.append(metrics)
    return result,values,points,replays


def decision(artifact,cfg,lock,rows,scores,points):
    measured=load(artifact/"evidence/status.json")
    stored=load(artifact/"results/holdout/aggregates.json")
    for name,value in stored.items():
        assert value==average([r for r in rows if f"{r['recipe']}_{r['pool']}"==name])
    recipe,pool=lock["choice"]["recipe"],lock["choice"]["pool"]
    chosen=stored[f"{recipe}_{pool}"]
    assert measured["chosen"]==f"{recipe}_{pool}" and equal(chosen["AP"],measured["AP"]) and equal(chosen["f1"],measured["F1"])
    assert measured["goal_reached"]==(chosen["AP"]>=cfg["goal"] and chosen["f1"]>=cfg["goal"])
    assert measured["Chicago_formal"]=="blocked"
    start,end=map(date.fromisoformat,cfg["holdout"])
    days=(end-start).days
    weights=np.random.default_rng(cfg["bootstrap_seed"]).multinomial(days,np.full(days,1/days),size=cfg["bootstrap_reps"])
    start_index=(start-date(1970,1,1)).days
    rng=cfg["main_recall"]
    deltas=[]
    uncertainty=[]
    for seed in cfg["seeds"]:
        candidate,control=(recipe,pool,seed),("B","mean",seed)
        new,old=points[candidate],points[control]
        assert new["TP"]==old["TP"] and new["other_FP"]==old["other_FP"]==0
        deltas.append((new["recovery_FP"]-old["recovery_FP"])/days)
        a=bootstrap_scores(scores[candidate],weights,start_index,rng)
        b=bootstrap_scores(scores[control],weights,start_index,rng)
        uncertainty.append(a[:,2]-b[:,2])
    all_draws=np.mean(uncertainty,axis=0)
    lo,hi=np.percentile(all_draws[np.isfinite(all_draws)],[2.5,97.5])
    np.testing.assert_allclose([lo,hi],measured["delta_FP_95"],atol=1e-12,rtol=0)
    assert equal(np.mean(deltas),measured["delta_FP_per_day"])
    assert measured["stable_recovery_FP_reduction"]==bool(np.mean(deltas)<0 and sum(x<0 for x in deltas)>=2 and hi<0)


def run(artifact,out,cfg,replay_root):
    lock=verify_lock(artifact,cfg)
    unlock=load(artifact/"evidence/unlock.json")
    features=load(artifact/"evidence/holdout_features.json")
    registered=load(artifact/"evidence/registration.json")
    amend=load(artifact/"evidence/amendment.json")
    dev=load(artifact/"evidence/prepared_development.json")
    cal=load(artifact/"results/calibration_receipt.json")
    finish=load(artifact/"evidence/status.json")
    assert registered["at_utc"]<amend["at_utc"]<dev["prepared_utc"]<lock["choice"]["at_utc"]<cal["at_utc"]
    assert cal["at_utc"]<lock["frozen_utc"]<unlock["unlocked_utc"]<features["at_utc"]<finish["finished_utc"]
    assert unlock["freeze_sha256"]==sha(artifact/"evidence/freeze.json") and unlock["prior_holdout_values_read"] is False
    assert features["target_truth_built"] is False and not lock["choice"]["holdout_used"]
    trained,rep_dev=dev_checks(artifact,cfg,lock,replay_root)
    measured,scores,points,rep_held=held_checks(artifact,cfg,lock,replay_root)
    decision(artifact,cfg,lock,measured,scores,points)
    results=dict(status="PASS",at_utc=now(),development_and_calibration_runs=len(trained),heldout_runs=len(measured),
        heldout_known=finish["n"],full_support_identical=True,cutoffs_recomputed=True,freeze_chronology_verified=True,
        AP_F1_FP_matched_recomputed=True,paired_date_blocks_recomputed=True,dev_replayed=len(rep_dev),heldout_replayed=len(rep_held),
        max_probability_difference=max(rep_dev+rep_held,default=None),replay_root=str(replay_root) if replay_root else None)
    save(out/"evidence"/("verification_replay.json" if replay_root else "verification_offline.json"),results)
    print(results,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    args=p.parse_args()
    root,cfg=settings()
    run(args.artifacts or root,root,cfg,args.replay_root)

"""Independent result and replay verifier for both time protocols."""
import argparse
from datetime import date,datetime
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score
import torch

from v7_io import V6,load,now,save,settings,sha
from v7_analysis import BOOT_METRICS,bootstrap
from v7_freeze import verify_freeze
from v7_score import cutoff,slice_scores
from v7_train import NEURAL,build,folder,lag_score,predict,read_npz,sample_training,set_seed,specifications,study_for
sys.path.insert(0,str(V6/"v5"))
from v5_prepare import fingerprint_keys
from v4_verify import check_support,verify_point
from evaluation import rank_score
from v1.graph import TrainPreprocessor


def close(a,b):
    return math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-11)


def inspect_metrics(z,threshold,row,points,cfg,cover):
    y,p=z["y"],z["score"]
    pred=p>=threshold
    assert len(y)==int(row.n)==cover["positive"]+cover["negative"]
    assert np.isin(y,[0,1]).all() and int(y.sum())==int(row.positive)
    np.testing.assert_array_equal(z["pred"],pred)
    assert close(accuracy_score(y,pred),row.accuracy)
    assert close(balanced_accuracy_score(y,pred),row.balanced_accuracy)
    assert close(average_precision_score(y,p),row.AP)
    assert close(f1_score(y,pred,zero_division=0),row.f1)
    assert tuple(confusion_matrix(y,pred,labels=[0,1]).ravel())==tuple(int(row[k]) for k in ("tn","fp","fn","tp"))
    assert int((pred&z["risk"]&(y==0)).sum())==int(row.recovery_FP)
    assert len(points)==len(cfg["recalls"])
    for point in points.to_dict("records"):
        verify_point(z,point,cfg,cover["total_days"])
        assert close(point["recovery_FP"]/cover["total_days"],point["recovery_FP_per_day"])


def model_replay(study,raw,model,seed,path,cfg):
    lead=model.startswith("lead_")
    ids=study.val if model=="current_attention" else np.flatnonzero(study.eligible&(study.phase==1))
    np.testing.assert_array_equal(study.keys(ids),raw["sample_id"])
    np.testing.assert_array_equal(study.y[ids],raw["y"])
    np.testing.assert_array_equal(study.risk[ids],raw["risk"])
    if lead:
        np.testing.assert_array_equal(study.d["current_eligible"].ravel()[ids],raw["current_eligible"])
    selected=np.flatnonzero(raw["y"]==1)
    bs=cfg["eval_batch_size"]
    begin=(selected[len(selected)//2]//bs)*bs
    block=slice(begin,begin+bs)
    indices=ids[block]
    if model in NEURAL:
        center=model=="lead_center"
        pool="attention" if center else model.split("_")[-1]
        set_seed(seed,cfg)
        net=build(study.nfeatures*2,cfg,pool,seed)
        net.load_state_dict(torch.load(path/"weight.pt",map_location="cpu",weights_only=True))
        z,p=predict(net,study,indices,cfg,center)
        np.testing.assert_allclose(z,raw["logit"][block],rtol=0,atol=1e-7)
        np.testing.assert_allclose(p,raw["probability"][block],rtol=0,atol=1e-7)
        score=rank_score(p,study.risk[indices],not lead)
    elif model=="lead_logit":
        coefs=read_npz(path/"logit_parameters.npz")
        score=expit((study.flat(indices)@coefs["coef"].T).ravel()+coefs["intercept"][0])
    elif model.endswith("_lag"):
        rule=load(path/"rule.json")
        training=study.ratio[study.train]
        fill=float(np.median(lag_score(training[np.isfinite(training)],0.)))
        assert fill==rule["missing_score"]
        score=lag_score(study.ratio[indices],fill)
    elif model=="current_rule":
        score=study.risk[indices].astype(float)
    else:
        score=np.zeros(len(indices))
    err=np.abs(score-raw["score"][block])
    limit=1e-7 if model in NEURAL else 1e-12
    assert float(err.max())<=limit
    np.testing.assert_array_equal(score>=raw["threshold"],raw["pred"][block])
    return float(err.max())


def section(frame,fold,model,seed,support):
    mask=(frame.fold==fold)&(frame.model==model)&(frame.support==support)
    return frame[mask&(frame.seed.isna() if seed is None else frame.seed==seed)]


def check(root,output,cfg,replay_root):
    lock=verify_freeze(root,cfg)
    status=load(root/"evidence/status.json")
    prepared=load(root/"evidence/prepared_development.json")
    prior=load(root/"evidence/prior_prepared.json")
    assert prepared["hashes"]==prior["hashes"] and prepared["A_B_truth_identical"]
    assert status["NYC_formal"]==status["Chicago_formal"]=="blocked" and not status["new_holdout_values_read"]
    assert all(status[k] is None for k in ("balanced_accuracy","AP","F1","matched_recovery_FP"))
    assert not (root/"results/holdout").exists() and not (root/"evidence/unlock.json").exists()
    assert not prepared["holdout_parsed"] and not status["prior_V5_V6_holdout_used"]
    dev=load(root/"results/development_receipt.json")
    cal=load(root/"results/calibration_receipt.json")
    stamp=[load(root/"evidence/prereg.json")["at_utc"],prepared["utc"],dev["at_utc"],cal["at_utc"],lock["frozen_utc"]]
    moment=list(map(datetime.fromisoformat,stamp))
    assert moment==sorted(set(moment))
    metrics=pd.read_csv(root/"results/analysis/phase_metrics.csv",float_precision="round_trip")
    matched=pd.read_csv(root/"results/analysis/matched_recall.csv",float_precision="round_trip")
    periods={**cfg["folds"],"final":cfg["calibration"]}
    diffs=[]
    trials=[]
    for fold,interval in periods.items():
        full=load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]
        common=load(root/"evidence"/f"fold_{fold}.json")["parts"][1]
        studies={}
        if replay_root:
            for role,tag in (("lead","lead_"),("current","")):
                npz=replay_root/"prepared"/f"{tag}{fold}.npz"
                assert sha(npz)==prepared["lead_hashes" if tag else "hashes"][fold]
                studies[role]=study_for(read_npz(npz),"X1")
            np.testing.assert_array_equal(studies["lead"].y,studies["current"].y)
        sample_ref=None
        leading_prep=None
        current_prep=None
        for model,seed in specifications(cfg):
            path=folder(root,fold,model,seed)
            raw=read_npz(path/"assessment.npz")
            lead=model.startswith("lead_")
            is_neural=model in NEURAL
            z=slice_scores(raw,raw["y"]>=0)
            cut=float(raw["threshold"])
            assert raw["threshold"].dtype==np.float64 and raw["score"].dtype==np.float64
            assert raw["sample_id"].dtype==np.int64 and raw["y"].dtype==np.int8
            check_support(raw["sample_id"],[interval],[cfg["embargo"],cfg["holdout"]])
            assert cut==cutoff(model,z["y"],z["score"])
            state=load(path/"status.json")
            if is_neural:
                assert state["updates"]>0 and state["best_epoch"]>=1
                if fold=="final":
                    assert state["best_epoch"]==dev["epochs"][model]
                np.testing.assert_array_equal(raw["score"],rank_score(raw["probability"],raw["risk"],not lead))
            elif model=="lead_logit":
                assert state["converged"]
            if lead:
                assert len(raw["y"])==full["expected"]
                assert int((raw["y"]<0).sum())==full["unknown_eligible"]
                assert fingerprint_keys(z["sample_id"])==full["known_keys_sha256"]
                common_z=slice_scores(z,z["current_eligible"])
            else:
                common_z=z
            assert fingerprint_keys(common_z["sample_id"])==full["common_keys_sha256"]
            for support,values,cover in (("full",z,full if lead else common),("common",common_z,common)):
                row=section(metrics,fold,model,seed,support)
                assert len(row)==1
                points=section(matched,fold,model,seed,support)
                inspect_metrics(values,cut,row.iloc[0],points,cfg,cover)
            if is_neural or model=="lead_logit":
                sample=read_npz(path/"sampling_ids.npz")
                check_support(sample["sample_id"],[[cfg["train_start"],interval[0]]],[cfg["embargo"],cfg["holdout"]])
                if lead:
                    if sample_ref is None:
                        sample_ref=sample
                        if replay_root:
                            ids,_=sample_training(studies["lead"],cfg)
                            np.testing.assert_array_equal(studies["lead"].keys(ids),sample["sample_id"])
                    for name,value in sample_ref.items():
                        np.testing.assert_array_equal(sample[name],value)
                    if is_neural:
                        assert state["updates"]==cfg["epochs"]*math.ceil(len(sample["sample_id"])/cfg["batch_size"])
            if replay_root:
                study=studies["lead" if lead else "current"]
                if (path/"preprocessor.npz").exists():
                    digest=sha(path/"preprocessor.npz")
                    previous=leading_prep if lead else current_prep
                    if previous is None:
                        study.configure(TrainPreprocessor.load(path/"preprocessor.npz"))
                        if lead:
                            leading_prep=digest
                        else:
                            current_prep=digest
                    else:
                        assert digest==previous
                diffs.append(model_replay(study,raw,model,seed,path,cfg))
            if model=="lead_attention" and seed==cfg["seeds"][0]:
                start=(date.fromisoformat(interval[0])-date(1970,1,1)).days
                demo=bootstrap(z,np.ones((1,full["total_days"]),dtype=np.int32),start,cfg["main_recall"])[0]
                row=section(metrics,fold,model,seed,"full").iloc[0]
                np.testing.assert_allclose(demo,[row[k] for k in BOOT_METRICS],rtol=0,atol=1e-12)
            trials.append((fold,model,seed))
    assert len(trials)==84 and len(metrics)==168 and len(matched)==672
    assert len(pd.read_csv(root/"results/analysis/date_blocks.csv"))==30
    result=dict(status="PASS",checked_utc=now(),assessments=len(trials),metrics_rows=len(metrics),matched_rows=len(matched),
        replayed=len(diffs),max_score_difference=max(diffs,default=None),registered_holdout_opened=False,
        known_support_and_cutoffs_verified=True,shared_samples_and_old_holdout_exclusion_verified=True,
        paired_day_bootstrap_weight_unit_check="PASS")
    save(output/"evidence"/("verification_replay.json" if replay_root else "verification_offline.json"),result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root,cfg=settings()
    check(a.artifacts or root,root,cfg,a.replay_root)

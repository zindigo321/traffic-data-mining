"""Independently recompute the full-data results and replay actual weights safely."""
import argparse
from datetime import datetime
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from scipy.special import expit
import torch

from full_io import CODE,V10,load,now,save,settings,sha
from full_check import register
from full_data import (MODELS,arrays,training_ids,folder,central,pool,groups,study_for,TrainPreprocessor,
                        build,predict,set_seed,read_npz)
from full_score import ALL,jobs,choose_thresholds
sys.path.insert(0,str(V10/"v9"))
from v9_verify import independent_cut
from v7_verify import verify_point,check_support


def close(a,b):
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)


def one(table,model,seed,phase=None,policy=None):
    selected=table[(table.model==model)&(table.seed.isna() if seed is None else table.seed==seed)]
    if phase is not None:
        selected=selected[selected.phase==phase]
    if policy is not None:
        selected=selected[selected.policy==policy]
    return selected


def inspect(raw,mask,threshold,row):
    y,p,risk=[raw[k][mask] for k in ("y","score","risk")]
    pred=p>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    both=bool((y==1).any() and (y==0).any())
    pairs=dict(tn=tn,fp=fp,fn=fn,tp=tp,known=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),
               AP=average_precision_score(y,p) if both else np.nan,ROC_AUC=roc_auc_score(y,p) if both else np.nan,
               balanced_accuracy=balanced_accuracy_score(y,pred) if both else np.nan,f1=f1_score(y,pred,zero_division=0) if (y==1).any() else np.nan,
               recall=tp/(tp+fn) if tp+fn else np.nan,FP_day=fp/row.days,recovery_FP=int((pred&(y==0)&risk).sum()))
    for key,value in pairs.items():
        close(row[key],value)
    close(row.recovery_FP_day,pairs["recovery_FP"]/row.days)
    return dict(sample_id=raw["sample_id"][mask],y=y,score=p,risk=risk)


def replay_job(study,raw,model,seed,path,cfg):
    ids=np.flatnonzero(study.phase>0)
    np.testing.assert_array_equal(study.keys(ids),raw["sample_id"])
    np.testing.assert_array_equal(study.y[ids],raw["y"])
    np.testing.assert_array_equal(study.risk[ids],raw["risk"])
    np.testing.assert_array_equal(groups(study.ratio[ids]),raw["legal_group"])
    offsets=[]
    for phase in np.unique(raw["phase"]):
        pos=np.flatnonzero((raw["phase"]==phase)&(raw["y"]==1))
        at=int(pos[len(pos)//2]) if len(pos) else int(np.flatnonzero(raw["phase"]==phase)[0])
        begin=at//cfg["eval_batch_size"]*cfg["eval_batch_size"]
        offsets.append(slice(begin,min(begin+cfg["eval_batch_size"],len(ids))))
    errors=[]
    if model in MODELS:
        set_seed(seed,cfg)
        net=build(study.nfeatures*2,cfg,pool(model),seed)
        net.load_state_dict(torch.load(path/"weight.pt",map_location="cpu",weights_only=True))
        for block in offsets:
            logits,p=predict(net,study,ids[block],cfg,central(model))
            np.testing.assert_allclose(logits,raw["logit"][block],atol=1e-7,rtol=0)
            np.testing.assert_allclose(p,raw["probability"][block],atol=1e-7,rtol=0)
            errors.append(float(np.abs(p.astype(float)-raw["score"][block]).max()))
    elif model=="logistic":
        params=read_npz(path/"logistic.npz")
        features=study.x.reshape(-1,study.nfeatures*2)
        for block in offsets:
            p=expit((features[ids[block]]@params["coef"].T).ravel()+params["intercept"][0])
            err=float(np.abs(p-raw["score"][block]).max())
            assert err<=1e-5
            errors.append(err)
    elif model=="lag20_rule":
        rule=load(path/"rule.json")
        ratio=study.ratio[ids]
        score=np.where(np.isfinite(ratio),1/(1+np.maximum(ratio,0)),rule["missing_fill"])
        np.testing.assert_array_equal(score,raw["score"])
        errors.append(0.)
    else:
        score=float(load(path/"status.json")["majority"])
        assert (raw["score"]==score).all()
        errors.append(0.)
    return errors


def verify(root,out,cfg,replay):
    freeze=load(root/"evidence/model_threshold_freeze.json")
    assert sha(root/"evidence/model_threshold_freeze.json")==load(root/"evidence/model_threshold_freeze_sha256.json")["sha256"]
    pick=load(root/"results/development_selection.json")
    assert sha(root/"results/development_selection.json")==freeze["selection_sha256"]
    assert all(sha(CODE/name)==h for name,h in freeze["code"].items())
    assert freeze["entry_gates_removed"] and freeze["old_V5_V6_dates_used"]
    assert not load(root/"evidence/experiment_status.json")["future_independent_holdout_claim"]
    snapshots=load(root/"evidence/source_snapshot.json")
    assert len({r["file"] for r in snapshots["files"]})==len(snapshots["files"])
    assert sum(r["rows"] for r in snapshots["files"])==load(root/"evidence/prepared.json")["source_funnel"]["rows"]
    op=pd.read_csv(root/"results/operating_points.csv",float_precision="round_trip")
    points=pd.read_csv(root/"results/matched_recall.csv",float_precision="round_trip")
    group_metrics=pd.read_csv(root/"results/groups.csv",float_precision="round_trip")
    assert len(op)==84 and len(points)==336 and len(group_metrics)==252
    source_groups={}
    errors=[]
    neural_count=0
    full_rows=0
    for stage in ("development","final"):
        data=arrays(root,stage,replay) if replay else None
        study=study_for(data,"X1") if replay else None
        reference=None
        support_reference=None
        updates=None
        for model,seed in jobs(cfg):
            path=folder(root,stage,model,seed)
            raw=read_npz(path/"assessment.npz")
            assert raw["score"].dtype==np.float64 and np.isfinite(raw["score"]).all()
            assert np.isin(raw["y"],[-1,0,1]).all()
            if reference is None:
                reference=raw
            for key in ("sample_id","y","risk","legal_group","phase"):
                np.testing.assert_array_equal(raw[key],reference[key])
            if model in MODELS:
                np.testing.assert_array_equal(raw["score"],raw["probability"].astype(float))
                state=load(path/"status.json")
                assert state["weight_sha256"]==sha(path/"weight.pt")
                ids=read_npz(path/"training_ids.npz")
                support=load(path/"training_support.json")
                assert support["epoch_exposures"]==state["known_train"]
                updates=updates or state["updates"]
                assert state["updates"]==updates
                assert not np.intersect1d(ids["sample_id"],raw["sample_id"]).size
                assert np.isin(ids["y"],[0,1]).all()
                if model!="center_sampled_matched":
                    assert len(ids["y"])==state["known_train"] and state["full_known_coverage"]==1.
                    if support_reference is None:
                        support_reference=ids["sample_id"]
                    np.testing.assert_array_equal(ids["sample_id"],support_reference)
                    full_rows+=len(ids["y"])
                curve=pd.read_csv(path/"curve.csv")
                assert len(curve)==cfg["epochs"] and (curve.training_exposures==state["known_train"]).all()
                assert (curve.unique_seen==state["unique_training"]).all()
                if replay:
                    expected,_=training_ids(study,cfg,model)
                    np.testing.assert_array_equal(study.keys(expected),ids["sample_id"])
                    np.testing.assert_array_equal(study.y[expected],ids["y"])
                neural_count+=1
            if replay:
                if (path/"preprocessor.npz").exists():
                    prep=TrainPreprocessor.load(path/"preprocessor.npz")
                    study.configure(prep)
                job_errors=replay_job(study,raw,model,seed,path,cfg)
                errors.append(dict(stage=stage,model=model,seed=seed,batches=len(job_errors),max_error=max(job_errors)))
            if stage=="final":
                entry,=[r for r in freeze["models"] if r["model"]==model and r["seed"]==seed]
                assert {p.name:sha(p) for p in sorted(path.iterdir()) if p.is_file()}==entry["files"]
                cal=(raw["phase"]==1)&(raw["y"]>=0)
                days=28
                for policy,cut in entry["options"].items():
                    if policy=="constrained":
                        expected=independent_cut(raw["y"][cal],raw["score"][cal],raw["risk"][cal],
                                                 days*cfg["FP_budget_day"],days*cfg["recovery_budget_day"])
                        assert expected==cut["threshold"]
                    else:
                        assert choose_thresholds(raw["y"][cal],raw["score"][cal],raw["risk"][cal],days,cfg)[policy]["threshold"]==cut["threshold"]
                    for phase in (1,2):
                        row,=[r for _,r in one(op,model,seed,phase,policy).iterrows()]
                        keep=(raw["phase"]==phase)&(raw["y"]>=0)
                        assert row.scheduled==(raw["phase"]==phase).sum()
                        assert row.unknown==((raw["phase"]==phase)&(raw["y"]<0)).sum()
                        z=inspect(raw,keep,cut["threshold"],row)
                        for group_number,group_name in enumerate(("legal_not_low","legal_low","legal_missing")):
                            grow=one(group_metrics,model,seed,phase,policy)
                            grow=grow[grow.scope==group_name].iloc[0]
                            inspect(raw,keep&(raw["legal_group"]==group_number),cut["threshold"],grow)
                        scopes={"all":np.ones_like(keep),"legal_low":raw["legal_group"]==1}
                        for scope,g in scopes.items():
                            subset=one(points,model,seed,phase)
                            subset=subset[subset.scope==scope]
                            if policy=="constrained":
                                zz={k:raw[k][keep&g] for k in ("sample_id","y","risk","score")}
                                for point in subset.to_dict("records"):
                                    verify_point(zz,point,cfg,row.days)
                roles=[cfg["full_calibration"],cfg["full_test"]]
                check_support(raw["sample_id"],roles,[])
                if model in MODELS:
                    check_support(ids["sample_id"],[cfg["full_train"]],roles)
    assert neural_count==36 and len(errors)==(42 if replay else 0)
    assert all(load(root/"evidence/registration.json")[k] for k in ("prior_V5_V6_dates_authorized_by_latest_user_request",))
    moments=[load(root/"evidence/registration.json")["registered_utc"],load(root/"evidence/prepared.json")["at_utc"],
             pick["at_utc"],load(root/"results/final_training.json")["at_utc"],freeze["frozen_utc"],
             load(root/"evidence/experiment_status.json")["at_utc"]]
    order=list(map(datetime.fromisoformat,moments))
    assert order==sorted(set(order))
    result=dict(status="PASS",checked_utc=now(),neural_models=36,deterministic_baselines=6,
        operating_points=len(op),matched_points=len(points),all_full_models_cover_all_known_training=True,
        neural_replayed=sum(r["model"] in MODELS for r in errors),all_replayed=len(errors),
        maximum_neural_error=max((r["max_error"] for r in errors if r["model"] in MODELS),default=None),
        replay_details=errors,all_configurations_have_matched_step_budgets=True,
        evaluation_type="full-data chronological retrospective",new_independent_holdout_claim=False,
        threshold_and_model_selection_precede_test_metrics=True,unknowns_not_used_as_negative=True)
    save(out/"evidence"/("verification_replay.json" if replay else "verification_offline.json"),result)
    print({k:v for k,v in result.items() if k!="replay_details"},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    out,cfg=settings()
    verify((a.artifacts or out).resolve(),out,cfg,a.replay_root)

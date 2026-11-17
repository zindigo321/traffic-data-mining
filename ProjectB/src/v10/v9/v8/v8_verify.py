"""Recompute every operating point and replay signed scores without a holdout."""
import argparse
from datetime import date,datetime
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score

from v8_io import V7,load,now,save,settings,sha
from v8_analysis import MODELS,location,selection
from v8_freeze import verify_freeze
from v8_thresholds import budgets,cutoffs,transfer_masks
from v8_train import arrays,read_npz,reference,study_for,sample_training
sys.path.insert(0,str(V7))
from v7_verify import model_replay,check_support,verify_point
from v7_analysis import bootstrap,BOOT_METRICS
from v7_score import slice_scores
from v1.graph import TrainPreprocessor


def close(a,b):
    return math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-11)


def only(frame,fold,model,seed):
    return frame[(frame.fold==fold)&(frame.model==model)&(frame.seed==seed)]


def inspect_prediction(raw,mask,threshold,row):
    z=slice_scores(raw,mask&(raw["y"]>=0))
    y,score=z["y"],z["score"]
    pred=score>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    assert (tn,fp,fn,tp)==tuple(int(row[k]) for k in ("tn","fp","fn","tp"))
    assert int((pred&(y==0)&z["risk"]).sum())==row.recovery_FP
    assert len(y)==row.n and int(y.sum())==row.positive and int(mask.sum())-len(y)==row.unknown_eligible
    assert close(average_precision_score(y,score),row.AP)
    assert close(balanced_accuracy_score(y,pred),row.balanced_accuracy)
    assert close(f1_score(y,pred,zero_division=0),row.f1)
    return z


def verify(root,out,cfg,replay):
    lock=verify_freeze(root,cfg)
    state=load(root/"evidence/status.json")
    inputs=load(root/"evidence/inputs.json")
    assert state["NYC_formal"]==state["Chicago_formal"]=="blocked"
    assert all(state[k] is None for k in ("official_AP","official_balanced_accuracy","official_F1","official_recovery_FP"))
    assert not state["holdout_used"] and not state["V5_V6_holdout_used"]
    assert not (root/"results/holdout").exists() and not (root/"evidence/unlock.json").exists()
    assert not load(root/"evidence/date_gate.json")["observation_values_requested"]
    registered=load(root/"evidence/prereg.json")
    rank_done=load(root/"results/rank_calibration.json")
    seq=list(map(datetime.fromisoformat,[registered["at_utc"],rank_done["completed_utc"],lock["chosen"]["at_utc"],lock["at_utc"]]))
    assert seq==sorted(set(seq))
    for job,entries in inputs["reference"].items():
        assert {f.name:sha(f) for f in (root/"results/reference"/job).iterdir() if f.is_file()}==entries
    a=root/"results/analysis"
    ranks=pd.read_csv(a/"ranking.csv",float_precision="round_trip")
    matched=pd.read_csv(a/"matched_recall.csv",float_precision="round_trip")
    policies=pd.read_csv(a/"thresholds.csv",float_precision="round_trip")
    bands=pd.read_csv(a/"paired_date_blocks.csv",float_precision="round_trip")
    diff=[]
    count=0
    for fold,period in {**cfg["folds"],"final":cfg["calibration"]}.items():
        first=None
        study=None
        if replay:
            data=arrays(root,cfg,fold,replay)
            study=study_for(data,"X1")
            prep=reference(root,fold,"attention",cfg["seeds"][0])/"preprocessor.npz"
            study.configure(TrainPreprocessor.load(prep))
            selected,_=sample_training(study,cfg)
        for model in MODELS:
            for seed in cfg["seeds"]:
                folder=location(root,fold,model,seed)
                raw=read_npz(folder/"assessment.npz")
                assert raw["score"].dtype==np.float64 and raw["logit"].dtype==raw["probability"].dtype==np.float32
                assert raw["sample_id"].dtype==np.int64 and raw["y"].dtype==np.int8
                assert np.isin(raw["y"],[-1,0,1]).all()
                np.testing.assert_array_equal(raw["score"],raw["probability"].astype(float))
                np.testing.assert_array_equal(raw["pred"],raw["score"]>=raw["threshold"])
                cols=("sample_id","y","risk","current_eligible")
                if first is None:
                    first={k:raw[k] for k in cols}
                for k in cols:
                    np.testing.assert_array_equal(raw[k],first[k])
                check_support(raw["sample_id"],[period],[cfg["embargo"],cfg["holdout"]])
                early,late,days=transfer_masks(raw["sample_id"],period,cfg["early_days"])
                full=np.ones(len(early),bool)
                one=only(ranks,fold,model,seed)
                assert len(one)==1
                z=inspect_prediction(raw,full,float(raw["threshold"]),one.iloc[0])
                fixed=only(matched,fold,model,seed)
                assert len(fixed)==len(cfg["recalls"])
                for row in fixed.to_dict("records"):
                    verify_point(z,row,cfg,days)
                if model=="rank" and seed==cfg["seeds"][0]:
                    first_day=(date.fromisoformat(period[0])-date(1970,1,1)).days
                    check=bootstrap(z,np.ones((1,days),dtype=np.int32),first_day,cfg["main_recall"])[0]
                    np.testing.assert_allclose(check,[one.iloc[0][k] for k in BOOT_METRICS],atol=1e-12,rtol=0)
                base=full if fold=="final" else early
                assess=full if fold=="final" else late
                fit_days=days if fold=="final" else cfg["early_days"]
                days_assess=days if fold=="final" else days-cfg["early_days"]
                fit=base&(raw["y"]>=0)
                choices=cutoffs(raw["y"][fit],raw["score"][fit],raw["risk"][fit],fit_days,cfg)
                policy=only(policies,fold,model,seed)
                assert len(policy)==3 and policy.AP.nunique()==1
                for rule,chosen in choices.items():
                    row=policy[policy.rule==rule].iloc[0]
                    for k in ("threshold","fit_TP","fit_FP","fit_recovery_FP","fit_BA","fit_F1"):
                        assert close(row[k],chosen[k])
                    inspect_prediction(raw,assess,chosen["threshold"],row)
                    cap1,cap2=budgets(days_assess,cfg)
                    assert bool(row.eval_budget_ok)==(row.fp<=cap1 and row.recovery_FP<=cap2)
                    if fold=="final" and rule=="max_ba":
                        assert float(raw["threshold"])==chosen["threshold"]
                sample=read_npz(folder/"sampling_ids.npz")
                inherited=reference(root,fold,"attention",seed)
                baseline=read_npz(inherited/"sampling_ids.npz")
                for k in baseline:
                    np.testing.assert_array_equal(sample[k],baseline[k])
                assert sha(folder/"preprocessor.npz")==sha(inherited/"preprocessor.npz")
                check_support(sample["sample_id"],[[cfg["train_start"],period[0]]],[cfg["embargo"],cfg["holdout"]])
                if replay:
                    np.testing.assert_array_equal(sample["sample_id"],study.keys(selected))
                    name="lead_center" if model=="center" else "lead_attention"
                    diff.append(model_replay(study,raw,name,seed,folder,cfg))
                count+=1
    assert len(ranks)==count==36 and len(matched)==144 and len(policies)==108 and len(bands)==12
    recompute=selection(cfg,ranks.to_dict("records"),policies.to_dict("records"),load(a/"ranking_summary.json"),load(a/"threshold_summary.json"))
    assert recompute=={k:v for k,v in lock["chosen"].items() if k!="at_utc"}
    for model in lock["models"]:
        row=only(policies,"final",state["frozen_model"],model["seed"])
        assert float(row[row.rule==state["frozen_rule"]].iloc[0].threshold)==model["threshold"]
    result=dict(status="PASS",checked_utc=now(),score_sets=count,threshold_evaluations=len(policies),
        matched_points=len(matched),replayed=len(diff),max_score_difference=max(diff,default=None),
        same_support_and_sampling=True,all_rankings_threshold_invariant=True,chronology_and_freeze=True,
        old_or_new_holdout_used=False,formal_NYC="blocked")
    save(out/"evidence"/("verification_replay.json" if replay else "verification_offline.json"),result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root,cfg=settings()
    verify(a.artifacts or root,root,cfg,a.replay_root)

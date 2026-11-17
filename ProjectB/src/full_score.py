"""All-data model selection, calibration, and complete retrospective evaluation."""
import argparse
from itertools import product
import statistics as stat
import sys

import numpy as np
import pandas as pd

from full_data import MODELS,folder,read_npz
from full_io import CODE,V10,load,now,save,settings,sha
sys.path.insert(0,str(V10))
from v10_score import measure
from v8_thresholds import cutoffs
from v9_policy import write_csv
from v4_metrics import matched_points
from evaluation import event_metrics

BASELINES=("logistic","lag20_rule","majority")
ALL=(*MODELS,*BASELINES)
METRICS=("AP","balanced_accuracy","f1","recall","precision","ROC_AUC","accuracy","tp","fp","fn","tn",
         "FP_day","recovery_FP_day","recovery_FP")
GROUPS=("legal_not_low","legal_low","legal_missing")


def jobs(cfg):
    return [*product(MODELS,cfg["seeds"]),*((b,None) for b in BASELINES)]


def choose_thresholds(y,p,risk,days,cfg):
    if (y==0).any() and (y==1).any():
        return {k:v for k,v in cutoffs(y,p,risk,days,cfg).items() if k in ("constrained","max_f1")}
    return {k:dict(threshold=.5,fit_TP=None,fit_FP=None,fit_recovery_FP=None,fit_days=days,
                   fit_F1=None,fit_BA=None,fit_FP_cap=days*cfg["FP_budget_day"],
                   fit_recovery_cap=days*cfg["recovery_budget_day"],fit_FP_budget_ok=None,
                   reason="single-class calibration, diagnostic default") for k in ("constrained","max_f1")}


def shared(raw,ref):
    for k in ("sample_id","y","risk","legal_group","phase"):
        np.testing.assert_array_equal(raw[k],ref[k])


def macro(rows,metrics=METRICS):
    seeds=list(dict.fromkeys(r["seed"] for r in rows))
    out=dict(runs=len(rows),seeds=len(seeds))
    for metric in metrics:
        by=[]
        for seed in seeds:
            values=[r[metric] for r in rows if r["seed"]==seed and r[metric] is not None and np.isfinite(r[metric])]
            if values:
                by.append(stat.mean(values))
        out[metric]=stat.mean(by) if by else None
        out[metric+"_sd"]=stat.stdev(by) if len(by)>1 else None
        out[metric+"_valid_seeds"]=len(by)
    return out


def phase_metrics(raw,phase,threshold,cfg,model,seed,policy):
    flag=raw["phase"]==phase
    known=flag&(raw["y"]>=0)
    days=len(np.unique(raw["sample_id"][flag]//1000//144))
    y,score,risk=[raw[k][known] for k in ("y","score","risk")]
    result=measure(y,score,risk,threshold,days)
    pred=score>=threshold
    return dict(model=model,seed=seed,phase=phase,policy=policy,days=days,threshold=threshold,
                scheduled=int(flag.sum()),unknown=int((flag&(raw["y"]<0)).sum()),
                coverage=float(known.sum()/flag.sum()),**result,
                total_budget_pass=result["fp"]<=days*cfg["FP_budget_day"],
                recovery_budget_pass=result["recovery_FP"]<=days*cfg["recovery_budget_day"],
                **event_metrics(raw["sample_id"][known],y,pred))


def development(root,cfg):
    target=root/"results/development_selection.json"
    if target.exists():
        raise PermissionError("Development choice already exists")
    rows=[]
    epochs={}
    reference=None
    for model,seed in jobs(cfg):
        path=folder(root,"development",model,seed)
        raw=read_npz(path/"assessment.npz")
        if reference is None:
            reference=raw
        shared(raw,reference)
        k=raw["y"]>=0
        days=len(np.unique(raw["sample_id"]//1000//144))
        options=choose_thresholds(raw["y"][k],raw["score"][k],raw["risk"][k],days,cfg)
        for name,option in options.items():
            rows.append(phase_metrics(raw,1,option["threshold"],cfg,model,seed,name))
    stats=load(root/"results/development_training.json")["models"]
    epochs={m:int(stat.median(r["best_epoch"] or cfg["epochs"] for r in stats if r["model"]==m)) for m in MODELS}
    summary={model:macro([r for r in rows if r["model"]==model and r["policy"]=="constrained"]) for model in ALL}
    rank=sorted(("graph_mean","graph_max","graph_attention"),
                key=lambda m:(-(summary[m]["AP"] if summary[m]["AP"] is not None else -1),
                              -(summary[m]["f1"] if summary[m]["f1"] is not None else -1),
                              MODELS.index(m)))
    write_csv(root/"results/development_metrics.csv",rows)
    choice=dict(primary=rank[0],epochs=epochs,summary=summary,selection_rule="development mean AP then constrained F1",
                at_utc=now(),test_metrics_used=False,models_and_candidates=cfg["models"],
                metrics_sha256=sha(root/"results/development_metrics.csv"))
    save(target,choice)
    print(dict(primary=rank[0],epochs=epochs,AP={m:summary[m]["AP"] for m in rank}),flush=True)


def freeze(root,cfg):
    target=root/"evidence/model_threshold_freeze.json"
    if target.exists():
        raise PermissionError("Thresholds are already frozen")
    pick=load(root/"results/development_selection.json")
    calibration=[]
    thresholds=[]
    reference=None
    for model,seed in jobs(cfg):
        path=folder(root,"final",model,seed)
        raw=read_npz(path/"assessment.npz")
        if reference is None:
            reference=raw
        shared(raw,reference)
        mask=(raw["phase"]==1)&(raw["y"]>=0)
        days=len(np.unique(raw["sample_id"][raw["phase"]==1]//1000//144))
        options=choose_thresholds(raw["y"][mask],raw["score"][mask],raw["risk"][mask],days,cfg)
        for policy,option in options.items():
            calibration.append(phase_metrics(raw,1,option["threshold"],cfg,model,seed,policy))
        thresholds.append(dict(model=model,seed=seed,options=options,
            files={p.name:sha(p) for p in sorted(path.iterdir()) if p.is_file()}))
    write_csv(root/"results/calibration_metrics.csv",calibration)
    receipt=dict(frozen_utc=now(),primary=pick["primary"],primary_policy="constrained",
        selection_sha256=sha(root/"results/development_selection.json"),models=thresholds,
        source_snapshot_sha256=sha(root/"evidence/source_snapshot.json"),prepared_sha256=sha(root/"evidence/prepared.json"),
        test_result_metrics_computed=False,evaluation_type="full available chronological retrospective",
        old_V5_V6_dates_used=True,entry_gates_removed=True,
        code={p.name:sha(p) for p in (CODE/"full_data.py",CODE/"full_train.py",CODE/"full_score.py",CODE/"full_prepare.py")})
    save(target,receipt)
    save(root/"evidence/model_threshold_freeze_sha256.json",dict(sha256=sha(target),at_utc=receipt["frozen_utc"]))
    print(dict(primary=pick["primary"],frozen_models=len(thresholds),calibration_rows=len(calibration)),flush=True)


def evaluate(root,cfg):
    freeze=load(root/"evidence/model_threshold_freeze.json")
    assert sha(root/"evidence/model_threshold_freeze.json")==load(root/"evidence/model_threshold_freeze_sha256.json")["sha256"]
    rows,groups,points=[],[],[]
    for entry in freeze["models"]:
        model,seed=entry["model"],entry["seed"]
        path=folder(root,"final",model,seed)
        assert {p.name:sha(p) for p in sorted(path.iterdir()) if p.is_file()}==entry["files"]
        raw=read_npz(path/"assessment.npz")
        dec=dict(sample_id=raw["sample_id"])
        for policy,option in entry["options"].items():
            threshold=option["threshold"]
            dec["pred_"+policy]=raw["score"]>=threshold
            dec["threshold_"+policy]=np.float64(threshold)
            for phase in (1,2):
                rows.append(phase_metrics(raw,phase,threshold,cfg,model,seed,policy))
                for i,name in enumerate(GROUPS):
                    flag=(raw["phase"]==phase)&(raw["legal_group"]==i)&(raw["y"]>=0)
                    days=len(np.unique(raw["sample_id"][raw["phase"]==phase]//1000//144))
                    groups.append(dict(model=model,seed=seed,phase=phase,policy=policy,scope=name,days=days,
                        **measure(raw["y"][flag],raw["score"][flag],raw["risk"][flag],threshold,days)))
        for phase in (1,2):
            days=len(np.unique(raw["sample_id"][raw["phase"]==phase]//1000//144))
            for scope,cond in (("all",np.ones(len(raw["y"]),dtype=bool)),("legal_low",raw["legal_group"]==1)):
                flag=(raw["phase"]==phase)&cond&(raw["y"]>=0)
                if (raw["y"][flag]==1).any() and (raw["y"][flag]==0).any():
                    fixed,_=matched_points(raw["sample_id"][flag],raw["y"][flag],raw["score"][flag],raw["risk"][flag],
                                           cfg["recalls"],days)
                    points.extend(dict(model=model,seed=seed,phase=phase,scope=scope,**p,
                                       recovery_FP_per_day=p["recovery_FP"]/days) for p in fixed)
        dest=root/"results/decisions"/(model+(f"_s{seed}" if seed is not None else "")+".npz")
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**dec)
    write_csv(root/"results/operating_points.csv",rows)
    write_csv(root/"results/groups.csv",groups)
    write_csv(root/"results/matched_recall.csv",points)
    summary={str(phase):{policy:{model:macro([r for r in rows if r["model"]==model and r["phase"]==phase and r["policy"]==policy])
             for model in ALL} for policy in ("constrained","max_f1")} for phase in (1,2)}
    save(root/"results/summary.json",summary)
    save(root/"evidence/experiment_status.json",dict(status="completed",at_utc=now(),primary=freeze["primary"],
        evaluation_type="chronological retrospective full-data experiment",neural_fits=36,deterministic_fits=6,
        test_known=next(r["known"] for r in rows if r["phase"]==2),used_all_available_dates=True,minimum_admission_coverage=0,
        future_independent_holdout_claim=False,heldout_wait_does_not_block_this_experiment=True,
        no_unknown_imputation=True))
    print(dict(status="completed",operating_points=len(rows),group_rows=len(groups),matched=len(points),
               primary=freeze["primary"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("development","freeze","evaluate"))
    a=p.parse_args()
    root,cfg=settings()
    {"development":development,"freeze":freeze,"evaluate":evaluate}[a.stage](root,cfg)

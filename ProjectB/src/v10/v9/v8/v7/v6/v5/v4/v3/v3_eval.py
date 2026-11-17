"""Evaluate the once-opened V3 holdout using a fully frozen model and calibration receipt."""
from collections import defaultdict
from pathlib import Path
import os
import time

from v3_io import CODE,context,load,now,save,sha
from v3_prepare import fingerprint_keys,local_dates,make_holdout,read_npz
from v3_train import Study,load_and_score,risk_metrics,typed_score

import numpy as np
import pandas as pd
import psutil
from evaluation import measure
from holdout import blocks,calibration_quality,groups
from v1.graph import TrainPreprocessor


def locked(root,cfg):
    e=root/"evidence"
    contract=load(e/"freeze.json")
    if sha(e/"freeze.json")!=load(e/"freeze_sha256.json")["sha256"]:
        raise ValueError("Freeze receipt changed")
    config=Path(os.environ.get("GT3_CONFIG",CODE/"config.json"))
    if (sha(config)!=contract["config_sha256"] or sha(root/"plan.md")!=contract["plan_sha256"]
            or contract["holdout"]!=cfg["holdout"] or contract["holdout_values_read"]):
        raise ValueError("Config, plan or scope has changed")
    for rel,digest in contract["digests"].items():
        if sha(root/rel)!=digest:
            raise ValueError(f"Changed frozen file {rel}")
    for rel,digest in contract["code_hashes"].items():
        if sha(CODE/rel)!=digest:
            raise ValueError(f"Changed frozen code {rel}")
    for row in contract["models"]:
        out=root/"results/calibration"/row["job"]
        for f,digest in row["files"].items():
            if sha(out/f)!=digest:
                raise ValueError(f"Changed model artifact {row['job']}/{f}")
    return contract


def bootstrap(root,cfg,rows):
    directory=root/"results/holdout"
    dates=local_dates([cfg["holdout"]])
    day_start=int(dates[0].astype(np.int64))
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    weights=rng.multinomial(len(dates),np.full(len(dates),1/len(dates)),size=cfg["bootstrap_reps"])
    scores={}
    by_group=defaultdict(list)
    for row in rows:
        name=row["job"]
        z=read_npz(directory/f"{name}.npz")
        day=z["sample_id"]//1000//144-day_start
        scores[name]=blocks(z["y"],z["score"],z["pred"],day,weights)
        by_group[row["comparison"]].append(scores[name])
    np.savez_compressed(directory/"day_bootstrap.npz",dates=dates.astype(np.int64),weights=weights,**scores)
    mean={k:np.mean(v,axis=0) for k,v in by_group.items()}
    selected=next(r["comparison"] for r in rows if r["selected"])
    mean["selected_minus_A"]=mean[selected]-mean["A_mean"]
    intervals=[]
    for k,v in mean.items():
        for i,key in enumerate(("AP","F1")):
            a=v[:,i]
            a=a[np.isfinite(a)]
            l,h=np.quantile(a,[.025,.975]) if len(a) else (None,None)
            intervals.append(dict(comparison=k,metric=key,lower_95=l,upper_95=h,replicates=len(a)))
    pd.DataFrame(intervals).to_csv(directory/"date_block_intervals.csv",index=False,na_rep="N/A")


def main():
    root,cfg=context()
    started=time.perf_counter()
    evidence=root/"evidence"
    if not (evidence/"freeze.json").exists():
        print("Evaluation blocked before freeze",flush=True)
        return
    if (evidence/"holdout_unlock.json").exists():
        raise ValueError("Holdout already opened. Use verification for recomputation.")
    contract=locked(root,cfg)
    out=root/"results/holdout"
    out.mkdir(parents=True,exist_ok=False)
    save(evidence/"holdout_unlock.json",dict(unlocked_utc=now(),freeze_sha256=sha(evidence/"freeze.json"),
         scope=cfg["holdout"],prior_value_parse=False,fitting_allowed=False,selection_allowed=False))
    arrays,grid,cover,digest=make_holdout(root,cfg,contract)
    eligible=arrays["eligible"].ravel()
    labels=arrays["y"].ravel()
    keys=grid.ravel()
    known=eligible&(labels>=0)
    known_keys=keys[known]
    np.savez_compressed(out/"support.npz",sample_id=keys,eligible=eligible,y=labels,known_ids=known_keys,
                        risk=arrays["risk"].ravel())
    save(evidence/"holdout_support.json",dict(at_utc=now(),array_sha256=digest,known_sha256=fingerprint_keys(known_keys),
         n=int(known.sum()),positive=cover["positive"],events=cover["true_events"],expected=cover["expected"]))
    if not (cover["positive"] and cover["negative"]):
        save(evidence/"blocked.json",dict(stage="evaluate",reason="Fixed holdout contains only one class"))
        return
    indices=np.flatnonzero(eligible)
    results,error_rows=[],[]
    proc=psutil.Process()
    for model in contract["models"]:
        tick=time.perf_counter()
        name=model["job"]
        state=root/"results/calibration"/name
        opt=load(state/"options.json")
        data=Study(arrays,opt["recipe"])
        data.configure(TrainPreprocessor.load(state/"preprocessor.npz"))
        all_z=typed_score(data,indices,*load_and_score(data,opt,cfg,state,indices))
        keep=all_z["y"]>=0
        z={k:a[keep] for k,a in all_z.items()}
        np.testing.assert_array_equal(known_keys,z["sample_id"])
        cutoff=model["threshold"]
        pred=z["score"]>=cutoff
        np.savez_compressed(out/f"{name}.npz",**z,threshold=np.float64(cutoff),pred=pred)
        result=dict(job=name,recipe=model["recipe"],pool=model["pool"],seed=model["seed"],selected=model["selected"],
                    comparison=f"{model['recipe']}_{model['pool']}",
                    **measure(z["y"],z["score"],cutoff,z["sample_id"],cover),**risk_metrics(z,cutoff),
                    **calibration_quality(z["y"],z["score"],model["platt"]),
                    logit_dtype=str(z["logit"].dtype),probability_dtype=str(z["probability"].dtype),
                    score_dtype=str(z["score"].dtype),inference_seconds=time.perf_counter()-tick,
                    RSS_bytes=proc.memory_info().rss)
        results.append(result)
        error_rows.extend(dict(job=name,**row) for row in groups(z,pred))
        print({k:result[k] for k in ("job","AP","f1","risk_FP","risk_TP","true_events")},flush=True)
        del data,all_z,z
    save(out/"metrics.json",results)
    table=pd.DataFrame(results)
    table.to_csv(out/"metrics.csv",index=False,na_rep="N/A")
    pd.DataFrame(error_rows).to_csv(out/"errors.csv",index=False)
    num=["AP","f1","ROC_AUC","accuracy","precision","recall","tp","fp","fn","tn","fp_per_day",
         "risk_AP","risk_FP","risk_TP","event_precision","event_recall","event_f1","calibrated_brier"]
    table.groupby("comparison")[num].agg(["mean","std","count"]).to_csv(out/"aggregate.csv",na_rep="N/A")
    chosen=table[table.selected].set_index("seed")
    ref=table[table.comparison=="A_mean"].set_index("seed")
    (chosen[num]-ref[num]).to_csv(out/"paired_vs_v2.csv",na_rep="N/A")
    bootstrap(root,cfg,results)
    ap,f1=float(chosen.AP.mean()),float(chosen.f1.mean())
    save(evidence/"status.json",dict(status="complete",chosen=chosen.comparison.iloc[0],mean_AP=ap,mean_F1=f1,
         target=cfg["goal"],reached=ap>=cfg["goal"] and f1>=cfg["goal"],
         AP_gap=max(0.,cfg["goal"]-ap),F1_gap=max(0.,cfg["goal"]-f1),
         n=len(known_keys),positive=cover["positive"],events=cover["true_events"],positive_days=cover["positive_days"],
         calibration_locked=True,configs=len(results),eval_seconds=time.perf_counter()-started,finished_utc=now()))


if __name__=="__main__":
    main()


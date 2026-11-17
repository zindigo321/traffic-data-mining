"""Predeclared paired comparisons on the identical known NYC holdout."""
from datetime import date
import sys

import numpy as np
import pandas as pd

from v6_io import V5,save
sys.path.insert(0,str(V5))
from v5_train import read_npz
from v4_analysis import bootstrap_scores

NAMES={("X1","mean"):"V5_X1_Mean",("B","mean"):"V4_P1_Mean"}


def stability(deltas,upper,minimum_seeds=2):
    return bool(np.mean(deltas)<0 and sum(d<0 for d in deltas)>=minimum_seeds and upper<0)


def compare(cfg,records,scores,points):
    primary=(cfg["primary_recipe"],cfg["primary_pool"])
    main=cfg["main_recall"]
    a,b=map(date.fromisoformat,cfg["holdout"])
    days=(b-a).days
    anchor=(a-date(1970,1,1)).days
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    weights=rng.multinomial(days,np.full(days,1/days),size=cfg["bootstrap_reps"])
    pre={(*primary,s):bootstrap_scores(scores[(*primary,s)],weights,anchor,main) for s in cfg["seeds"]}
    rows,intervals,summaries=[],[],{}
    for recipe,pool in cfg["controls"]:
        name=NAMES[recipe,pool]
        differences=[]
        for seed in cfg["seeds"]:
            p,c=(*primary,seed),(recipe,pool,seed)
            paired=[]
            for new,old in zip(points[p],points[c],strict=True):
                if any(new[k]!=old[k] for k in ("target_recall","achieved_recall","TP","FN")) or new["other_FP"] or old["other_FP"]:
                    raise ValueError("The two models do not share identical risk-ranking recall support")
                row=dict(control=name,seed=seed,target_recall=new["target_recall"],achieved_recall=new["achieved_recall"],
                    TP=new["TP"],FN=new["FN"],control_recovery_FP=old["recovery_FP"],primary_recovery_FP=new["recovery_FP"],
                    delta_recovery_FP=new["recovery_FP"]-old["recovery_FP"],delta_FP_per_day=new["FP_per_day"]-old["FP_per_day"],
                    delta_AP=records[p]["AP"]-records[c]["AP"],delta_F1=records[p]["f1"]-records[c]["f1"])
                paired.append(row)
            rows.extend(paired)
            boot=bootstrap_scores(scores[c],weights,anchor,main)
            differences.append(pre[p]-boot)
        joint=np.mean(differences,axis=0)
        stats={}
        for i,metric in enumerate(("AP","F1","FP60_per_day")):
            v=joint[:,i]
            v=v[np.isfinite(v)]
            low,high=map(float,np.quantile(v,[.025,.975]))
            intervals.append(dict(control=name,metric=metric,low_95=low,high_95=high,valid_replicates=len(v)))
            stats[metric]=[low,high]
        subset=[r for r in rows if r["control"]==name and r["target_recall"]==main]
        delta=[r["delta_FP_per_day"] for r in subset]
        fp_ci=stats["FP60_per_day"]
        summaries[name]=dict(mean_FP_per_day_delta=float(np.mean(delta)),
            control_FP_per_day=float(np.mean([r["control_recovery_FP"]/days for r in subset])),
            primary_FP_per_day=float(np.mean([r["primary_recovery_FP"]/days for r in subset])),
            seeds_with_fewer_FP=sum(d<0 for d in delta),FP_per_day_95=fp_ci,
            delta_AP=float(np.mean([r["delta_AP"] for r in subset])),
            delta_F1=float(np.mean([r["delta_F1"] for r in subset])),
            stable=stability(delta,fp_ci[1],cfg["minimum_affected_seeds"]))
    return rows,intervals,summaries


def publish(out,comparison):
    rows,ci,summary=comparison
    pd.DataFrame(rows).to_csv(out/"paired.csv",index=False)
    pd.DataFrame(ci).to_csv(out/"date_blocks.csv",index=False)
    save(out/"comparisons.json",summary)


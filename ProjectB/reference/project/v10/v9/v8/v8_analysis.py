"""Compare ranks and threshold transfer, then select by the preregistered rules."""
from datetime import date
import statistics
import sys

import numpy as np
import pandas as pd

from v8_io import V7,load,now,save,settings
from v8_check import register
from v8_train import candidate,reference,read_npz
from v8_thresholds import budgets,cutoffs,transfer_masks
sys.path.insert(0,str(V7))
from v7_score import evaluate,slice_scores
from v7_analysis import bootstrap,BOOT_METRICS

MODELS=("attention","center","rank")
KEYS=("AP","balanced_accuracy","f1","FP60_day","recovery_FP60_day","fp_per_day","recovery_FP_day","precision","recall")


def location(root,fold,model,seed):
    return candidate(root,fold,seed) if model=="rank" else reference(root,fold,model,seed)


def coverage(raw,mask,days):
    y=raw["y"][mask]
    if not len(y):
        raise ValueError("Empty evaluation interval")
    return dict(expected=len(y),eligible=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),
        unknown_eligible=int((y<0).sum()),known_coverage=float(np.mean(y>=0)),total_days=days)


def macro(rows):
    seeds=sorted({r["seed"] for r in rows})
    result=dict(runs=len(rows),seeds=len(seeds))
    for key in KEYS:
        within=[statistics.mean(r[key] for r in rows if r["seed"]==seed and r[key] is not None)
                for seed in seeds if any(r["seed"]==seed and r[key] is not None for r in rows)]
        result[key]=statistics.mean(within) if within else None
        result[key+"_sd"]=statistics.stdev(within) if len(within)>1 else None
    return result


def metrics(raw,mask,threshold,cfg,days):
    z=slice_scores(raw,mask&(raw["y"]>=0))
    out,points=evaluate(z,threshold,cfg,coverage(raw,mask,days))
    out["recovery_FP_day"]=out["recovery_FP"]/days
    a,b=budgets(days,cfg)
    out["eval_budget_ok"]=bool(out["fp"]<=a and out["recovery_FP"]<=b)
    return out,points,z


def selection(cfg,ranks,thresholds,rank_summary,threshold_summary):
    base=rank_summary["development"]["attention"]
    new=rank_summary["development"]["rank"]
    reductions=[]
    for seed in cfg["seeds"]:
        values=[statistics.mean(r["recovery_FP60_day"] for r in ranks if r["stage"]=="development" and
                r["model"]==model and r["seed"]==seed) for model in ("attention","rank")]
        reductions.append(dict(seed=seed,delta=values[1]-values[0]))
    conditions=dict(AP_within_margin=new["AP"]>=base["AP"]-cfg["max_dev_AP_drop"],
        total_not_higher=new["FP60_day"]<=base["FP60_day"],
        recovery_lower=new["recovery_FP60_day"]<base["recovery_FP60_day"],
        two_seed_reductions=sum(r["delta"]<0 for r in reductions)>=2)
    model="rank" if all(conditions.values()) else "attention"
    flags={}
    for rule in cfg["threshold_rules"]:
        d=threshold_summary["development_transfer"][model][rule]
        c=[r for r in thresholds if r["stage"]=="calibration" and r["model"]==model and r["rule"]==rule]
        flags[rule]=dict(development_mean_within=d["fp_per_day"]<=cfg["FP_budget_day"] and
                d["recovery_FP_day"]<=cfg["recovery_budget_day"],calibration_all_seeds_within=all(r["eval_budget_ok"] for r in c))
    ok=[r for r in cfg["threshold_rules"] if all(flags[r].values())]
    if ok:
        def criteria(rule):
            d=threshold_summary["development_transfer"][model][rule]
            c=threshold_summary["calibration"][model][rule]
            return (round(d["f1"],12),round(c["f1"],12),-round(d["recovery_FP_day"],12))
        rule=max(ok,key=criteria)
    else:
        rule="constrained"
    return dict(selected_model=model,selected_rule=rule,rank_conditions=conditions,rank_seed_recovery_changes=reductions,
        threshold_eligibility=flags,transfer_budget_fallback=not ok,official_holdout_used=False)


def intervals(cfg,inputs,periods):
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    rows=[]
    for stage,folds in (("development",list(cfg["folds"])),("calibration",["final"])):
        changes={m:[] for m in ("attention","center")}
        for f in folds:
            start,stop=map(date.fromisoformat,periods[f])
            n=(stop-start).days
            w=rng.multinomial(n,np.full(n,1/n),size=cfg["bootstrap_reps"])
            first=(start-date(1970,1,1)).days
            traces={m:np.mean([bootstrap(inputs[f,m,s],w,first,cfg["main_recall"]) for s in cfg["seeds"]],axis=0)
                    for m in MODELS}
            for m in changes:
                changes[m].append(traces["rank"]-traces[m])
        for model,blocks in changes.items():
            values=np.mean(blocks,axis=0)
            for k in ("AP","FP60_day","recovery_FP60_day"):
                v=values[:,BOOT_METRICS.index(k)]
                low,high=np.quantile(v[np.isfinite(v)],[.025,.975])
                rows.append(dict(stage=stage,contrast="rank_minus_"+model,metric=k,low_95=low,high_95=high,replicates=len(v)))
    return rows


def analyze(root,cfg):
    register(root,cfg)
    out=root/"results/analysis"
    out.mkdir(parents=True,exist_ok=True)
    periods={**cfg["folds"],"final":cfg["calibration"]}
    ranks,points,thresholds=[],[],[]
    records={}
    for f,period in periods.items():
        comparison=None
        for m in MODELS:
            for seed in cfg["seeds"]:
                raw=read_npz(location(root,f,m,seed)/"assessment.npz")
                if comparison is None:
                    comparison={k:raw[k] for k in ("sample_id","y","risk","current_eligible")}
                for k,v in comparison.items():
                    np.testing.assert_array_equal(v,raw[k])
                early,late,days=transfer_masks(raw["sample_id"],period,cfg["early_days"])
                all_keys=np.ones(len(raw["y"]),dtype=bool)
                result,fixed,z=metrics(raw,all_keys,float(raw["threshold"]),cfg,days)
                stage="calibration" if f=="final" else "development"
                fields=dict(stage=stage,fold=f,model=m,seed=seed)
                ranks.append(dict(**fields,**result))
                points.extend(dict(**fields,**point) for point in fixed)
                records[f,m,seed]=z
                if f=="final":
                    tune=evaluate_mask=all_keys
                    nd=vd=days
                    phase="calibration"
                else:
                    tune,evaluate_mask=early,late
                    nd,vd=cfg["early_days"],days-cfg["early_days"]
                    phase="development_transfer"
                known=tune&(raw["y"]>=0)
                choices=cutoffs(raw["y"][known],raw["score"][known],raw["risk"][known],nd,cfg)
                for rule,fit in choices.items():
                    evaluated,_,_=metrics(raw,evaluate_mask,fit["threshold"],cfg,vd)
                    payload={k:v for k,v in fit.items() if k!="threshold"}
                    thresholds.append(dict(stage=phase,fold=f,model=m,seed=seed,rule=rule,**payload,**evaluated))
    rank_summary={stage:{m:macro([r for r in ranks if r["stage"]==stage and r["model"]==m]) for m in MODELS}
                  for stage in ("development","calibration")}
    threshold_summary={stage:{m:{rule:macro([r for r in thresholds if r["stage"]==stage and r["model"]==m and r["rule"]==rule])
                                 for rule in cfg["threshold_rules"]} for m in MODELS}
                       for stage in ("development_transfer","calibration")}
    for name,rows in (("ranking",ranks),("matched_recall",points),("thresholds",thresholds),
                     ("paired_date_blocks",intervals(cfg,records,periods))):
        pd.DataFrame(rows).to_csv(out/f"{name}.csv",index=False,na_rep="N/A")
    save(out/"ranking_summary.json",rank_summary)
    save(out/"threshold_summary.json",threshold_summary)
    choice=selection(cfg,ranks,thresholds,rank_summary,threshold_summary)
    save(out/"selection.json",{**choice,"at_utc":now()})
    print(dict(ranks=len(ranks),thresholds=len(thresholds),**choice),flush=True)


if __name__=="__main__":
    analyze(*settings())

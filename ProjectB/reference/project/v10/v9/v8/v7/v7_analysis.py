"""Common-key evaluation, deterministic equal recall, and calendar paired diagnostics."""
from datetime import date
import statistics

import numpy as np
import pandas as pd

from v7_io import load,save,settings
from v7_score import evaluate,slice_scores
from v7_train import folder,read_npz,specifications

METRICS=("balanced_accuracy","AP","f1","accuracy","ROC_AUC","precision","recall","risk_balanced_accuracy",
         "FP60_day","recovery_FP60_day","fp_per_day")
BOOT_METRICS=("balanced_accuracy","AP","f1","FP60_day","recovery_FP60_day")
BASELINES=("lead_lag","lead_logit","lead_center")


def summarize(rows):
    seeds=list(dict.fromkeys(x["seed"] for x in rows))
    result=dict(runs=len(rows),seeds=len(seeds))
    for key in METRICS:
        individual=[statistics.mean(x[key] for x in rows if x["seed"]==s and x[key] is not None)
                    for s in seeds if any(x["seed"]==s and x[key] is not None for x in rows)]
        result[key]=statistics.mean(individual) if individual else None
        result[f"{key}_sd"]=statistics.stdev(individual) if len(individual)>1 else None
    return result


def bootstrap(z,weights,first_day,recall):
    order=np.lexsort((z["sample_id"],-z["score"]))
    ordered=z["score"][order]
    y=z["y"][order].astype(bool)
    risk=z["risk"][order]&~y
    chosen=z["pred"][order]
    day=z["sample_id"][order]//1000//144-first_day
    days=weights.shape[1]
    if not ((day>=0)&(day<days)).all():
        raise ValueError("Date bootstrap misaligned")
    ends=np.flatnonzero(np.r_[ordered[:-1]!=ordered[1:],True])
    output=[]
    for start in range(0,len(weights),24):
        w=weights[start:start+24,day].astype(np.int32)
        tp=np.cumsum(w*y,axis=1,dtype=np.int32)
        cw=np.cumsum(w,axis=1,dtype=np.int32)
        pos=tp[:,-1]
        neg=cw[:,-1]-pos
        valid=(pos>0)&(neg>0)
        p=np.where(valid,pos,1)
        n=np.where(valid,neg,1)
        dp=(w*(chosen&y)).sum(axis=1)
        dn=(w*(chosen&~y)).sum(axis=1)
        ba=.5*(dp/p+1-dn/n)
        f1=2*dp/(p+dp+dn)
        group_tp=tp[:,ends]
        gain=np.diff(group_tp,axis=1,prepend=0)
        precision=np.divide(group_tp,cw[:,ends],out=np.zeros_like(group_tp,dtype=float),where=cw[:,ends]>0)
        ap=np.sum(precision*gain,axis=1)/p
        stop=np.argmax(tp>=np.ceil(recall*p)[:,None],axis=1)
        row=np.arange(len(stop))
        fp=(cw[row,stop]-tp[row,stop])/days
        recovery=np.cumsum(w*risk,axis=1,dtype=np.int32)[row,stop]/days
        values=np.stack((ba,ap,f1,fp,recovery),axis=-1)
        values[~valid]=np.nan
        output.append(values)
    return np.concatenate(output,axis=0)


def build_summaries(root,cfg):
    out=root/"results/analysis"
    out.mkdir(parents=True,exist_ok=True)
    periods={**cfg["folds"],"final":cfg["calibration"]}
    rows,points=[],[]
    scores={}
    for fold in periods:
        stage="calibration" if fold=="final" else "development"
        common=load(root/"evidence"/f"fold_{fold}.json")["parts"][1]
        lead=load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]
        common_key=None
        full_key=None
        for model,seed in specifications(cfg):
            path=folder(root,fold,model,seed)
            raw=read_npz(path/"assessment.npz")
            status=load(path/"status.json")
            cut=float(raw["threshold"])
            known=raw["y"]>=0
            is_lead=model.startswith("lead_")
            clean=slice_scores(raw,known)
            scores[fold,model,seed]=clean
            if is_lead:
                full_key=clean["sample_id"] if full_key is None else full_key
                np.testing.assert_array_equal(clean["sample_id"],full_key)
                common_mask=known&raw["current_eligible"]
                compare=slice_scores(raw,common_mask)
            else:
                compare=clean
            common_key=compare["sample_id"] if common_key is None else common_key
            np.testing.assert_array_equal(compare["sample_id"],common_key)
            for support,z,cover in (("full",clean,lead if is_lead else common),("common",compare,common)):
                metrics,fixed=evaluate(z,cut,cfg,cover)
                descriptor=dict(stage=stage,fold=fold,model=model,seed=seed,support=support)
                rows.append(dict(**descriptor,**metrics,fit_seconds=status["fit_seconds"],best_epoch=status["best_epoch"]))
                points.extend(dict(**descriptor,**r) for r in fixed)
    pd.DataFrame(rows).to_csv(out/"phase_metrics.csv",index=False,na_rep="N/A")
    pd.DataFrame(points).to_csv(out/"matched_recall.csv",index=False)
    summary={}
    for stage in ("development","calibration"):
        summary[stage]={}
        for support in ("full","common"):
            summary[stage][support]={m:summarize([x for x in rows if x["stage"]==stage and x["support"]==support and x["model"]==m])
                                     for m in dict.fromkeys(x["model"] for x in rows)}
    save(out/"aggregates.json",summary)
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    groups={"development":list(cfg["folds"]),"calibration":["final"]}
    paired=[]
    for stage,folds in groups.items():
        changes={m:[] for m in BASELINES}
        for fold in folds:
            start,stop=map(date.fromisoformat,periods[fold])
            n=(stop-start).days
            weights=rng.multinomial(n,np.full(n,1/n),size=cfg["bootstrap_reps"])
            first=(start-date(1970,1,1)).days
            primary=np.mean([bootstrap(scores[fold,"lead_attention",s],weights,first,cfg["main_recall"])
                             for s in cfg["seeds"]],axis=0)
            for baseline in BASELINES:
                seeds=cfg["seeds"] if baseline=="lead_center" else [None]
                control=np.mean([bootstrap(scores[fold,baseline,s],weights,first,cfg["main_recall"]) for s in seeds],axis=0)
                changes[baseline].append(primary-control)
        for baseline,values in changes.items():
            result=np.mean(values,axis=0)
            a=summary[stage]["full"]["lead_attention"]
            b=summary[stage]["full"][baseline]
            for i,name in enumerate(BOOT_METRICS):
                valid=result[np.isfinite(result[:,i]),i]
                low,high=np.quantile(valid,[.025,.975])
                paired.append(dict(stage=stage,support="full",primary="lead_attention",baseline=baseline,metric=name,
                    difference=a[name]-b[name],low_95=low,high_95=high,replicates=len(valid)))
    pd.DataFrame(paired).to_csv(out/"date_blocks.csv",index=False)
    print(dict(metrics=len(rows),matched=len(points),date_block_intervals=len(paired)),flush=True)


if __name__=="__main__":
    build_summaries(*settings())

"""Paired full-data retrospective analyses, plots, daily errors and audit tables."""
from datetime import date
from itertools import product
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from full_data import MODELS,folder,read_npz
from full_io import V10,load,save,settings
from full_score import ALL,jobs,write_csv,METRICS,macro
sys.path.insert(0,str(V10/"v9"))
from v9_analysis import trace,BOOT

EPOCH=date(1970,1,1)


def read_rows(path):
    frame=pd.read_csv(path,float_precision="round_trip")
    return [{k:None if pd.isna(v) else v for k,v in row.items()} for row in frame.to_dict("records")]


def analyze(root,cfg):
    select=load(root/"results/development_selection.json")
    primary=select["primary"]
    summary=load(root/"results/summary.json")
    freeze=load(root/"evidence/model_threshold_freeze.json")
    lookup={(r["model"],r["seed"]):r for r in freeze["models"]}
    contrasts={
        "attention_minus_mean":("graph_attention","graph_mean"),
        "attention_minus_max":("graph_attention","graph_max"),
        "rank_minus_center_full":("center_rank","center_full"),
        "center_full_minus_sampled_matched":("center_full","center_sampled_matched"),
        "selected_graph_minus_center_full":(primary,"center_full")}
    rows,daily,resources=[],[],[]
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    weights_out={}
    for phase,interval in ((1,cfg["full_calibration"]),(2,cfg["full_test"])):
        first=(date.fromisoformat(interval[0])-EPOCH).days
        n_days=(date.fromisoformat(interval[1])-date.fromisoformat(interval[0])).days
        weights=rng.multinomial(n_days,np.full(n_days,1/n_days),size=cfg["bootstrap_reps"])
        weights_out[str(phase)]=weights
        traces={}
        for model,seed in jobs(cfg):
            path=folder(root,"final",model,seed)
            raw=read_npz(path/"assessment.npz")
            threshold=lookup[model,seed]["options"]["constrained"]["threshold"]
            keep=(raw["phase"]==phase)&(raw["y"]>=0)
            y,score,risk=[raw[k][keep] for k in ("y","score","risk")]
            keys=raw["sample_id"][keep]
            pred=score>=threshold
            if model in MODELS:
                traces[model,seed]=trace(keys,y,score,risk,pred,weights,first,cfg["main_recall"])
            day=raw["sample_id"]//1000//144
            for d in np.unique(day[raw["phase"]==phase]):
                known=(day==d)&(raw["y"]>=0)
                forecast=raw["score"]>=threshold
                positive=raw["y"]==1
                neg=raw["y"]==0
                daily.append(dict(phase=phase,model=model,seed=seed,date=str(np.datetime64(int(d),"D")),
                     scheduled=int((day==d).sum()),known=int(known.sum()),unknown=int(((day==d)&(raw["y"]<0)).sum()),
                     positive=int((known&positive).sum()),TP=int((known&positive&forecast).sum()),
                     FP=int((known&neg&forecast).sum()),FN=int((known&positive&~forecast).sum()),
                     TN=int((known&neg&~forecast).sum()),recovery_FP=int((known&neg&forecast&raw["risk"]).sum())))
        for name,(a,b) in contrasts.items():
            delta=np.mean([traces[a,s]-traces[b,s] for s in cfg["seeds"]],axis=0)
            for i,metric in enumerate(BOOT):
                valid=delta[:,i][np.isfinite(delta[:,i])]
                low,high=np.quantile(valid,[.025,.975])
                rows.append(dict(phase=phase,contrast=name,metric=metric,mean_delta=float(np.nanmean(delta[:,i])),
                                 low95=float(low),high95=float(high),valid_replicates=len(valid)))
    for stage in ("development","final"):
        for model,seed in jobs(cfg):
            state=load(folder(root,stage,model,seed)/"status.json")
            resources.append(dict(stage=stage,model=model,seed=seed,best_epoch=state["best_epoch"],
                fit_and_score_seconds=state["fit_and_score_seconds"],parameters=state.get("parameters"),
                known_train=state.get("known_train"),unique_training=state.get("unique_training"),
                updates=state.get("updates"),peak_RSS_bytes=state.get("peak_RSS_bytes"),
                full_known_coverage=state.get("full_known_coverage")))
    write_csv(root/"results/paired_date_blocks.csv",rows)
    write_csv(root/"results/daily_errors.csv",daily)
    write_csv(root/"results/training_resources.csv",resources)
    np.savez_compressed(root/"results/date_bootstrap_weights.npz",**weights_out)
    groups=read_rows(root/"results/groups.csv")
    matched=read_rows(root/"results/matched_recall.csv")
    summaries={}
    for phase in (1,2):
        summaries[str(phase)]={model:dict(
            groups={scope:macro([r for r in groups if r["phase"]==phase and r["model"]==model
                                and r["policy"]=="constrained" and r["scope"]==scope])
                    for scope in ("legal_low","legal_not_low","legal_missing")},
            matched={scope:{str(rate):macro([r for r in matched if r["phase"]==phase and r["model"]==model
                       and r["scope"]==scope and r["target_recall"]==rate],
                       ("TP","FP","recovery_FP","achieved_recall","FP_per_day","recovery_FP_per_day","F1"))
                       for rate in cfg["recalls"]} for scope in ("all","legal_low")}) for model in ALL}
    save(root/"results/group_matched_summary.json",summaries)
    figures(root,cfg,summary,lookup)
    print(dict(primary=primary,paired_rows=len(rows),resource_rows=len(resources),daily_rows=len(daily)),flush=True)


def figures(root,cfg,summary,lookup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=root/"figures"
    out.mkdir(parents=True,exist_ok=True)
    colors=("#1b6f57","#b86c24","#405eaa")
    fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained")
    for model,color in zip(("graph_mean","graph_max","graph_attention"),colors,strict=True):
        ap=summary["2"]["constrained"][model]["AP"]
        for i,seed in enumerate(cfg["seeds"]):
            raw=read_npz(folder(root,"final",model,seed)/"assessment.npz")
            known=(raw["phase"]==2)&(raw["y"]>=0)
            precision,recall,_=precision_recall_curve(raw["y"][known],raw["score"][known])
            ax.step(recall,precision,where="post",color=color,alpha=.55,linewidth=.9,
                    label=f"{model}, AP={ap:.3f}" if i==0 else None)
    ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1),
           title="Full-data chronological retrospective test")
    ax.legend()
    ax.grid(alpha=.15)
    fig.savefig(out/"test_PR_three_pooling.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    models=list(MODELS)
    display=["Mean","Max","Attention","Center","Center rank","Sample matched"]
    for ax,metrics,title in ((axes[0],("AP","balanced_accuracy","f1","recall"),"Retrospective metrics"),
                             (axes[1],("FP_day","recovery_FP_day"),"Frozen working point")):
        x=np.arange(len(models))
        width=.8/len(metrics)
        for j,key in enumerate(metrics):
            ax.bar(x-.4+(j+.5)*width,[summary["2"]["constrained"][m][key] for m in models],width=width,label=key)
        ax.set(xticks=x,xticklabels=display,title=title)
        ax.tick_params(axis="x",labelrotation=25)
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.15)
    fig.savefig(out/"test_pooling_and_controls.png",dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4),layout="constrained")
    for model in MODELS:
        curve=pd.read_csv(folder(root,"development",model,cfg["seeds"][0])/"curve.csv")
        ax.plot(curve.epoch,curve.AP,label=model,linewidth=1.1)
    ax.set(xlabel="Epoch",ylabel="Development AP",title="Full-data development, registered seed 11")
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
    fig.savefig(out/"development_curves.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    analyze(*settings())

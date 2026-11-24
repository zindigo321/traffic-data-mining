"""Measured V3 development and holdout figures, no fitting."""
import os
from v3_io import context, load
from v3_prepare import read_npz

ROOT,CFG=context()
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"plot_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

COLORS=["#196484","#B65C34","#18795E","#66727D"]
GRID="#DDE5EB"


def save(fig,name):
    dest=ROOT/"figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest/name,dpi=165,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def development():
    path=ROOT/"results/screen.csv"
    if not path.exists():
        return
    t=pd.read_csv(path)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout="constrained")
    for ax,(metric,title) in zip(axes,(("AP","Routed AP"),("f1","Positive F1"))):
        for r,c in zip(["A",*CFG["recipes"]],COLORS):
            s=t[t.recipe==r].set_index("fold").loc[list(CFG["folds"])]
            ax.plot(s.index,s[metric],marker="o",color=c,lw=2,label="A (V2)" if r=="A" else r)
        ax.set(xlabel="Forward development fold",title=title,ylim=(0,max(.65,t[metric].max()+.06)))
        ax.grid(axis="y",color=GRID)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Chicago V3 | development ablations, Mean seed 11",x=.02,ha="left")
    save(fig,"development_ablation.png")


def holdout(t):
    root=ROOT/"results/holdout"
    choice=load(ROOT/"results/development_selection.json")
    status=load(ROOT/"evidence/status.json")
    recipe=choice["recipe"]
    names=[f"{recipe}_{p}" for p in CFG["pools"]]+["A_mean"]
    labels=[p.title()+(" (selected)" if p==choice["pool"] else "") for p in CFG["pools"]]+["V2 Mean refit"]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.2),sharey=True,layout="constrained")
    for ax,metric,heading in zip(axes,("AP","f1"),("Average precision","Positive F1 at frozen cutoff")):
        for i,(k,c) in enumerate(zip(names,COLORS)):
            m=t.loc[t.comparison==k,metric]
            ax.errorbar(m.mean(),i,xerr=m.std(),fmt="o",color=c,capsize=3,markersize=7)
            ax.annotate(f"{m.mean():.3f}",(m.mean(),i),xytext=(8,8),textcoords="offset points",fontsize=8)
        ax.axvline(CFG["goal"],color="#A6595D",ls="--",lw=1.3)
        ax.set_title(heading,loc="left",fontweight="bold",fontsize=11)
        ax.set_yticks(np.arange(4),labels)
        ax.set_ylim(3.5,-.5)
        ax.set_xlim(0,max(.72,t[metric].max()+.11))
        ax.grid(axis="x",color=GRID)
    fig.suptitle(f"Chicago untouched holdout | {status['positive']} positives, {status['events']} events, mean ± seed SD",
                 x=.02,ha="left")
    save(fig,"holdout_comparison.png")
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.35),layout="constrained")
    for ax,risk_only in zip(axes,(False,True)):
        for k,label,c in zip(names,labels,COLORS):
            row=t[(t.comparison==k)&(t.seed==11)].iloc[0]
            z=read_npz(root/f"{row.job}.npz")
            mask=z["risk"] if risk_only else np.ones(len(z["y"]),dtype=bool)
            pre,rec,_=precision_recall_curve(z["y"][mask],z["score"][mask])
            ax.step(rec,pre,where="post",lw=1.7,color=c,label=f"{label}  {row.AP:.3f}")
        ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02),
               title="Within the prior-low risk group" if risk_only else "Complete known support")
        ax.grid(color=GRID,alpha=.65)
    axes[1].legend(frameon=False,fontsize=8,loc="upper right")
    fig.suptitle("Frozen Chicago test PR curves | seed 11",x=.02,ha="left")
    save(fig,"holdout_PR.png")
    fig,ax=plt.subplots(figsize=(8.2,4.1),layout="constrained")
    x=np.arange(len(names))
    bar=ax.bar(x,[t.loc[t.comparison==k,"risk_FP"].mean() for k in names],color=COLORS,width=.60)
    ax.bar_label(bar,fmt="%.1f",padding=3,fontsize=9)
    ax.set_xticks(x,labels)
    ax.set_ylabel("Mean false positives")
    ax.set_title("Previously low, subsequently recovered | frozen alert cutoffs",loc="left")
    ax.grid(axis="y",color=GRID)
    ax.set_axisbelow(True)
    save(fig,"holdout_recovery_false_positives.png")


def main():
    development()
    out=ROOT/"reports"
    out.mkdir(exist_ok=True)
    t=ROOT/"results/holdout/metrics.csv"
    if not t.exists():
        (out/"computed_summary.md").write_text("# V3 Metrics\n\nA complete formal holdout result is unavailable. See evidence/blocked.json.\n",encoding="utf-8")
        return
    table=pd.read_csv(t)
    holdout(table)
    columns=["AP","f1","ROC_AUC","precision","recall","accuracy","risk_FP","fp_per_day","event_f1"]
    p=table.groupby("comparison")[columns].agg(["mean","std"])
    p.to_csv(out/"summary.csv",na_rep="N/A")
    lines=["# Measured Metrics Across Three Seeds","", "| Configuration | AP | Positive F1 | FP/day | Event F1 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for n,g in table.groupby("comparison"):
        values=[f"{g[f].mean():.4f} ± {g[f].std():.4f}" for f in ("AP","f1","fp_per_day","event_f1")]
        lines.append("| "+n+" | "+" | ".join(values)+" |")
    (out/"computed_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()


"""Render only measured development and frozen holdout results."""
import os
from pathlib import Path

from v1 import runtime
from common import json_read, settings
from prepare_chicago import read_npz

ROOT,CFG = settings()
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"plot_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

BLUE = "#176386"
ORANGE = "#B45C31"
GREEN = "#1D795D"
GRAY = "#5F6F7B"
COLORS = [BLUE,ORANGE,GREEN,GRAY,"#665699","#2C858F","#9B7847"]
GRID = "#DDE4EA"


def save(fig,name):
    dst = ROOT/"figures"
    dst.mkdir(exist_ok=True)
    fig.savefig(dst/name,dpi=165,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def style(ax,title,ylab):
    ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
    ax.set_ylabel(ylab)
    ax.grid(axis="y",color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)


def development():
    src = ROOT/"results/screen_metrics.csv"
    if not src.exists():
        return
    table = pd.read_csv(src)
    decision = json_read(ROOT/"results/screen_selection.json")
    styles = [("V0","raw"),("V1","raw"),("V2","raw"),("V3","raw")]
    styles += list(table[table.route=="routed"][["recipe","route"]].drop_duplicates().itertuples(index=False,name=None))
    folds = decision["folds"]
    fig,axs = plt.subplots(1,2,figsize=(10.6,4.2),sharex=True,layout="constrained")
    for a,metric,title in zip(axs,("AP","f1"),("Average precision","Positive F1")):
        for (r,mode),c in zip(styles,COLORS):
            sub = table[(table.recipe==r)&(table.route==mode)].set_index("fold").loc[folds]
            a.plot(folds,sub[metric],marker="o",color=c,lw=2,
                   label=r if mode=="raw" else r+" + prior-low routing",ls="--" if mode=="routed" else "-")
        style(a,title,metric.upper())
        a.set_ylim(bottom=0)
        a.set_xlabel("Chronological development fold")
    axs[1].legend(frameon=False,loc="upper right",fontsize=8)
    fig.suptitle("Chicago development | paired single-seed ablations (Max, seed 11)",x=.02,ha="left",fontsize=13)
    save(fig,"development_ablation.png")


def final_data():
    path = ROOT/"results/holdout/metrics.csv"
    return pd.read_csv(path) if path.exists() else None


def final_holdout(t):
    out = ROOT/"results/holdout"
    status = json_read(ROOT/"evidence/holdout_status.json")
    d = json_read(ROOT/"results/development_selection.json")
    mode,r,pool = [d[k] for k in ("route","recipe","pooling")]
    names = [f"{r}_{p}_{mode}" for p in CFG["pools"]] + ["V0_max_raw",f"{r}_{pool}_center","logreg","rule"]
    labels = ["Mean","Max","Attention","V0 Max","Center only","Logistic","Prior-low rule"]
    group = t.groupby("comparison")
    fig,axs = plt.subplots(1,2,figsize=(12.2,4.8),sharey=True,layout="constrained")
    for a,metric,title in zip(axs,("AP","f1"),("Average precision","Positive F1 at frozen cutoff")):
        for i,(m,n,c) in enumerate(zip(names,labels,COLORS)):
            values = group.get_group(m)[metric]
            mean = float(values.mean())
            sd = float(values.std()) if len(values)>1 else 0.
            a.errorbar(mean,i,xerr=sd,fmt="o",color=c,capsize=3,markersize=7)
            a.annotate(f"{mean:.3f}",(mean,i),xytext=(7,7),textcoords="offset points",fontsize=8)
        a.axvline(.60,color="#B36868",ls="--",lw=1.2,label="0.60 target")
        a.grid(axis="x",color=GRID)
        a.set_title(title,loc="left",fontweight="bold",fontsize=11)
        a.set_yticks(np.arange(len(names)),labels)
        a.set_ylim(len(names)-.5,-.5)
        a.set_xlim(-.014,max(.75,float(t[metric].max())+.10))
    axs[1].legend(frameon=False,fontsize=8,loc="lower right")
    fig.suptitle(f"Chicago untouched holdout | {status['positive']} positive windows, {status['true_events']} proxy events",
                 x=.02,ha="left",fontsize=13)
    save(fig,"holdout_comparison.png")

    fig,ax = plt.subplots(figsize=(8,5.4),layout="constrained")
    main = t[(t.seed==CFG["seeds"][0]) & t.comparison.isin([status["chosen"],"V0_max_raw",
             f"{r}_{pool}_center","logreg","rule"])]
    order = [status["chosen"],"V0_max_raw",f"{r}_{pool}_center","logreg","rule"]
    for ident,c in zip(order,COLORS):
        row = main.loc[main.comparison==ident].iloc[0]
        z = read_npz(out/f"{row.evaluation_id}.npz")
        precision, recall, _ = precision_recall_curve(z["y"],z["score"])
        ax.step(recall,precision,where="post",color=c,lw=1.7,label=f"{ident}   AP {row.AP:.3f}")
    ax.axhline(t.positive_rate.iloc[0],color=GRAY,lw=1.1,ls=":",label="Positive prevalence")
    style(ax,"Chicago frozen test PR curves | seed 11","Precision")
    ax.set_xlabel("Recall")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)
    ax.legend(frameon=False,loc="upper right",fontsize=8)
    save(fig,"holdout_PR_curves.png")

    risk = pd.read_csv(out/"error_groups.csv")
    risk = risk[risk.dimension=="risk"].merge(t[["evaluation_id","comparison"]],on="evaluation_id",validate="many_to_one")
    means = risk[risk.comparison.isin([status["chosen"],"V0_max_raw"])].groupby(["comparison","group"])[["fp","fn"]].mean()
    fig,axs = plt.subplots(1,2,figsize=(10.7,3.9),layout="constrained")
    groups = ["prior_low","ordinary","history_missing"]
    for ax, metric, name in zip(axs,("fp","fn"),("False positive windows","False negative windows")):
        x = np.arange(len(groups))
        for i,(m,c) in enumerate(zip([status["chosen"],"V0_max_raw"],[BLUE,ORANGE])):
            n = means.loc[m][metric].reindex(groups,fill_value=0).to_numpy()
            bars = ax.bar(x+(i-.5)*.32,n,.3,color=c,label=m)
            ax.bar_label(bars,labels=[f"{v:.1f}" if v else "" for v in n],fontsize=8,padding=3)
        ax.set_xticks(x,["Prior low","Ordinary","History missing"])
        style(ax,name,"Mean across seeds")
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Frozen holdout | errors by previous-bin causal risk",x=.02,ha="left",fontsize=13)
    save(fig,"holdout_error_risk_groups.png")


def computed(t):
    out = ROOT/"reports"
    out.mkdir(exist_ok=True)
    lines = ["# Metrics from Run Artifacts", ""]
    sel_path = ROOT/"results/screen_selection.json"
    if sel_path.exists():
        raw = json_read(sel_path)
        lines += ["## Chicago Development", "", "| Candidate | Mean AP across three folds | Mean positive F1 across three folds |",
                  "| --- | ---: | ---: |"]
        for name,row in raw["choices"].items():
            lines.append(f"| {name} | {row['mean_AP']:.4f} | {row['mean_f1']:.4f} |")
        if raw["routed_challenger"]:
            r=raw["routed_challenger"]
            lines.append(f"| {raw['chosen_enhancement']} plus risk routing | {r['mean_AP']:.4f} | {r['mean_f1']:.4f} |")
        lines += ["", "The results above are paired development diagnostics for seed 11 with Max pooling. Each threshold was fitted on its own fold for selection.", ""]
    if t is not None:
        lines += ["## Chicago Frozen Holdout", "", "| Configuration | AP | F1 | ROC-AUC | Precision | Recall | FP/day | Event F1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name,g in t.groupby("comparison",sort=False):
            values = []
            for col in ("AP","f1","ROC_AUC","precision","recall","fp_per_day","event_f1"):
                s = g[col].dropna()
                values.append("N/A" if not len(s) else (f"{s.mean():.4f} ± {s.std():.4f}" if len(s)>1 else f"{s.iloc[0]:.4f}"))
            lines.append("| "+name+" | "+" | ".join(values)+" |")
        lines += ["", "Standard deviations use the fixed seeds. Date-block intervals and paired differences are in results/holdout."]
    else:
        lines += ["## New Chicago Holdout", "", "A complete formal holdout result is unavailable."]
    (out/"computed_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    development()
    t = final_data()
    if t is not None:
        final_holdout(t)
    computed(t)
    print("Computed figures and metric summary",flush=True)


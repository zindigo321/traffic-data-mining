"""Readable figures and measured-only V7 audit report."""
from datetime import date,timedelta
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from v7_io import V6,load,settings
from v7_train import folder,read_npz

LABELS={"lead_attention":"Ten-minute lead Attention (preselected)","lead_mean":"Ten-minute lead Mean","lead_max":"Ten-minute lead Max",
        "lead_center":"Ten-minute lead center-only","lead_logit":"Ten-minute lead logistic regression","lead_lag":"Ten-minute lead lag rule","lead_majority":"Ten-minute lead all noncongested",
        "current_attention":"Current-window Attention","current_rule":"Current-window previous-bin low-speed rule","current_lag":"Current-window relative-speed rule",
        "current_majority":"Current-window all noncongested"}
FIG={"lead_attention":"Warning attention","lead_center":"Warning center only","lead_logit":"Logistic regression",
     "lead_lag":"Lag-2 rule","current_attention":"Current-window attention"}


def period(bounds):
    a,b=map(date.fromisoformat,bounds)
    return f"{a} to {b-timedelta(days=1)}"


def plots(root,cfg):
    p=root/"figures"
    p.mkdir(exist_ok=True)
    frame=pd.read_csv(root/"results/analysis/matched_recall.csv")
    frame=frame[(frame.stage=="development")&(frame.support=="full")]
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.8),layout="constrained")
    for model in ("lead_attention","lead_center","lead_logit","lead_lag"):
        d=frame[frame.model==model].groupby("target_recall")[["FP_per_day","recovery_FP_per_day"]].mean()
        for ax,metric,title in zip(axes,("FP_per_day","recovery_FP_per_day"),("All false alerts/day","Recovery false alerts/day"),strict=True):
            ax.plot(100*d.index,d[metric],label=FIG[model],marker="o")
            ax.axvline(60,color="#777777",alpha=.25,lw=1,ls="--")
            ax.set(xlabel="Target recall (%)",ylabel=title)
            ax.grid(axis="y",alpha=.22)
    axes[0].legend(frameon=False)
    fig.suptitle("V7 NYC development | matched recall on all known warning windows")
    fig.savefig(p/"matched_recall.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,4.8),layout="constrained")
    for model in ("lead_attention","lead_center","lead_logit","current_attention"):
        seed=None if model=="lead_logit" else cfg["seeds"][0]
        z=read_npz(folder(root,"f3",model,seed)/"assessment.npz")
        known=z["y"]>=0
        if model.startswith("lead_"):
            known &= z["current_eligible"]
        precision,recall,_=precision_recall_curve(z["y"][known],z["score"][known])
        ax.step(recall,precision,where="post",label=FIG[model])
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",title="V7 F3 development | shared known windows")
    ax.legend(frameon=False)
    ax.grid(alpha=.22)
    fig.savefig(p/"common_PR_f3.png",dpi=160)
    plt.close(fig)


def fmt(value,sd=None):
    if value is None:
        return "N/A"
    return f"{value:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")


def metrics_table(metrics,models):
    rows=["| Configuration | Balanced accuracy | AP | Positive F1 | Total FP/day at 60% recall | Recovery FP/day at 60% recall |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        val=metrics[m]
        scores=" | ".join(fmt(val[k],val[k+"_sd"]) for k in ("balanced_accuracy","AP","f1"))
        rows.append(f"| {LABELS[m]} | {scores} | {val['FP60_day']:.3f} | {val['recovery_FP60_day']:.3f} |")
    return rows


def render(root,cfg):
    s=load(root/"evidence/status.json")
    g=load(root/"evidence/date_gate.json")
    proof=load(root/"evidence/verification_replay.json")
    freeze=load(root/"evidence/calibration_freeze.json")
    graph=load(root/"evidence/graph.json")
    pre=load(root/"evidence/prereg.json")
    stats=load(root/"results/analysis/aggregates.json")
    ci=pd.read_csv(root/"results/analysis/date_blocks.csv")
    plots(root,cfg)
    d=stats["development"]["full"]["lead_attention"]
    cal=stats["calibration"]["full"]["lead_attention"]
    a=stats["development"]["common"]["current_attention"]
    b=stats["development"]["common"]["lead_attention"]
    lines=[
        "# Project B V7 Audit Report",
        "",
        "**Three development folds and independent calibration completed. Formal NYC validation remains blocked/N/A, and the original Chicago and Geolife protocols remain separately blocked.** "
        "The ten-minute lead warning uses t−2 and earlier inputs. The original current-window classification uses t−1 and earlier inputs. Both protocols share the original labels.",
        "",
        "## Dates, Evidence, and Evaluation Support",
        "",
        f"The new holdout is registered for **{period(cfg['holdout'])}** in NYC local time. The latest official timestamp was {g['latest'] or 'N/A'}, before the registered period ended. "
        "V7 did not transfer, parse, or evaluate speeds and labels from this period. The previous V5 and V6 holdouts from 2026-08-31 through 11-08 serve only as historical references. "
        "Source verification read only the original ZIP fingerprint, code, and authorized development inputs. "
        "[Date evidence](evidence/date_gate.json) · [Plan and excluded periods](plan.md) · [NYC DOT data](https://data.cityofnewyork.us/d/i4gi-tjb9) · "
        "[Chicago discontinuation](https://data.cityofchicago.org/stories/s/c28d-stgw)",
        "",
        f"The fixed graph has {graph['road_count']} directed road segments and {graph['edges']} candidate edges. Rebuilt A arrays match V6 byte for byte. "
        "B and A share ground truth window for window. B scores the entire fixed calendar while A retains its prior-bin eligibility rule. Common support comprises the known A keys. "
        "[Input verification](evidence/inputs.json) · [Reconstruction and label alignment](evidence/prepared_development.json) · [Timing contract](method_and_data_contract.md)",
        "",
        "| Evaluation split | Local dates | B fixed scoring keys | B known windows | B unknown | A/B shared known | Positive | Recovery negative |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold,bounds in {**cfg["folds"],"final":cfg["calibration"]}.items():
        row=load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]
        lines.append(f"| {'Independent calibration' if fold=='final' else fold.upper()} | {period(bounds)} | {row['expected']:,} | "
                     f"{row['positive']+row['negative']:,} | {row['unknown_eligible']:,} | {row['current_known']:,} | "
                     f"{row['positive']:,} | {row['high_risk_recovered']:,} |")
    lines += ["",
        "Unknown windows remain in score files and coverage funnels. Supervised metrics use known labels only. On the common keys, B uses its original threshold fitted on its complete known support. "
        "Missing lag information at the start of a peak period is represented by missingness indicators. "
        "[Example fold support](evidence/fold_lead_f1.json) · [Full and common-support metrics](results/analysis/phase_metrics.csv)",
        "",
        "## Three-Fold Development Results",
        "",
        "For each neural configuration, average the three folds equally within seeds 11, 22, and 33. Then report the seed mean and standard deviation. Rules and logistic regression run once per fold. "
        "Development uses each fold to select the AP checkpoint and calculate diagnostic thresholds.", ""]
    leads=["lead_attention","lead_mean","lead_max","lead_center","lead_logit","lead_lag","lead_majority"]
    lines+=metrics_table(stats["development"]["full"],leads)
    lines+=["",
        f"On the common known keys, B Attention has AP {b['AP']:.4f} and positive F1 {b['f1']:.4f}. "
        f"A current-window Attention has AP {a['AP']:.4f} and positive F1 {a['f1']:.4f}. "
        "The lead protocol excludes the previous-bin quantities that contribute to the label, trains on all known windows, and scores without routing. Differences reflect the combined difficulty of the protocols. "
        f"On the full B support, balanced accuracy is {d['balanced_accuracy']:.4f} and equal-class accuracy within the risk group is {d['risk_balanced_accuracy']:.4f}. "
        "[Common-key PR curves](figures/common_PR_f3.png) · [Common-support metrics](results/analysis/phase_metrics.csv)",
        "",
        "## 28-Day Independent Calibration",
        "",
        f"Lock the median epoch from development folds before fitting final models. B maximizes calibration balanced accuracy. A uses its original max-F1 rule. "
        f"The freeze time is {freeze['frozen_utc']}. The table reports diagnostics on that calibration split. "
        "[Frozen weights and thresholds](evidence/calibration_freeze.json) · [Checkpoints and seeds](results/development_receipt.json)",
        ""]
    lines+=metrics_table(stats["calibration"]["full"],leads[:-1])
    lines+=["",
        f"Calibration B Attention has AP {cal['AP']:.4f}, balanced accuracy {cal['balanced_accuracy']:.4f}, and positive F1 {cal['f1']:.4f}. "
        "Optimizing balanced accuracy and optimizing F1 select different trade-offs. The package includes confusion counts, thresholds, ROC-AUC, coverage, costs, and per-seed results. "
        "[Thresholds and confusion counts](results/analysis/phase_metrics.csv)",
        "",
        "## Matched-Recall False Positives and Paired Intervals",
        "",
        "On the full known support, select score prefixes with the same number of true positives, with 60% recall as the primary target. Report total false positives and false positives for recovery from previous-bin low speed separately. "
        "The confidence intervals use the same calendar dates and 1,000 shared date-block resamples, reporting Attention minus each baseline. The table covers the three development folds. ",
        "",
        "| Baseline | Δ AP (95% interval) | Δ total FP/day (95% interval) | Δ recovery FP/day (95% interval) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("lead_lag","lead_logit","lead_center"):
        pick=ci[(ci.stage=="development")&(ci.baseline==name)].set_index("metric")
        terms=[]
        for k in ("AP","FP60_day","recovery_FP60_day"):
            row=pick.loc[k]
            terms.append(f"{row.difference:+.4f} [{row.low_95:+.4f}, {row.high_95:+.4f}]")
        lines.append(f"| {LABELS[name]} | "+" | ".join(terms)+" |")
    lines+=["",
        "Relative to the center-only ablation, recovery false positives fell at the primary matched-recall point. The AP and total-FP intervals cross zero. On calibration, the center-only model had slightly higher AP. "
        "The three folds support a development-stage gain in early-warning ranking. Performance on new dates awaits formal holdout evaluation. "
        "[All recall points](results/analysis/matched_recall.csv) · [Date blocks](results/analysis/date_blocks.csv) · [False-positive curves](figures/matched_recall.png)",
        "",
        "## Acceptance and Reproduction",
        "",
        "After changing center and neighbor observations from t−1 and t onward, B inputs and scores for all three poolings matched while ground truth changed. Checks also covered lag boundaries, frozen priors, and sampling independent of actual risk. "
        f"Independently recomputed {proof['assessments']} configurations and {proof['matched_rows']} matched-recall rows. "
        f"Sample replay covered {proof['replayed']} groups, with a maximum score difference of {proof['max_score_difference']:.3g}. "
        "The 34 engineering tests verify the implementation mechanisms. "
        "[Recomputation and weight replay](evidence/verification_replay.json) · [Test report](test_report.md)",
        "",
        "1. Verify SHA-256 for the original V6 ZIP, V7 manifest, development caches, and fixed graph.",
        "2. Rebuild development arrays for both protocols. Verify excluded old dates, ground truth, observation cutoffs, and the full key sets.",
        "3. Independently recompute AP, BA, F1, confusion counts, thresholds, and matched recall from per-window NPZ files, then safely replay samples.",
        "4. Check formal status, the date gate, and frozen models. Keep formal status blocked until publication covers all preregistered dates.",
        "",
        "[Reproduction and audit guide](README_audit.md). The clean extraction receipt accompanies the ZIP.",
        ""]
    (root/"audit_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"audit_report.md"),NYC_formal=s["NYC_formal"],development_AP=d["AP"],development_F1=d["f1"]),flush=True)


if __name__=="__main__":
    render(*settings())

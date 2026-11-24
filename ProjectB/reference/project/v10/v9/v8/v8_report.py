"""Measured-only V8 audit of global operating points and equal recall."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from v8_io import load,settings

NAMES={"attention":"V7 Attention","center":"V7 center-only","rank":"V8 tail ranking"}
RULES={"max_ba":"Maximum balanced accuracy","max_f1":"Maximum F1","constrained":"False-positive-constrained F1"}
ENG={"max_ba":"Max BA","max_f1":"Max F1","constrained":"Constrained F1"}


def plots(root,cfg,policies,model):
    out=root/"figures"
    out.mkdir(exist_ok=True)
    order=("max_ba","max_f1","constrained")
    values=policies["calibration"][model]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout="constrained")
    x=np.arange(3)
    for key,off,title,color in (("balanced_accuracy",-.18,"Balanced accuracy","#36688d"),("f1",.18,"Positive F1","#ce7d3e")):
        axes[0].bar(x+off,[values[k][key] for k in order],.35,label=title,color=color)
    axes[0].set(ylim=(0,1),ylabel="Metric",xticks=x,xticklabels=[ENG[k] for k in order])
    for key,off,title,color in (("fp_per_day",-.18,"Total false alerts","#36688d"),
                                ("recovery_FP_day",.18,"Recovery false alerts","#ce7d3e")):
        axes[1].bar(x+off,[values[k][key] for k in order],.35,label=title,color=color)
    axes[1].axhline(cfg["FP_budget_day"],ls="--",color="#36688d",lw=1,alpha=.65)
    axes[1].axhline(cfg["recovery_budget_day"],ls="--",color="#ce7d3e",lw=1,alpha=.65)
    axes[1].set(ylabel="False alerts per calendar day",xticks=x,xticklabels=[ENG[k] for k in order])
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(axis="y",alpha=.22)
    fig.suptitle("NYC V8 | one scoring model, three thresholds | calibration diagnostics")
    fig.savefig(out/"calibration_thresholds.png",dpi=160)
    plt.close(fig)
    points=pd.read_csv(root/"results/analysis/matched_recall.csv")
    points=points[points.stage=="development"]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout="constrained")
    for name,label in (("attention","V7 attention"),("center","V7 center only"),("rank","V8 tail ranking")):
        rows=points[points.model==name].groupby("target_recall")[["FP_per_day","recovery_FP_per_day"]].mean()
        for ax,key,title in zip(axes,("FP_per_day","recovery_FP_per_day"),("All false alerts/day","Recovery false alerts/day"),strict=True):
            ax.plot(100*rows.index,rows[key],label=label,marker="o")
            ax.set(xlabel="Target recall (%)",ylabel=title)
            ax.grid(axis="y",alpha=.23)
    for ax in axes:
        ax.axvline(60,ls="--",color="#777777",lw=1)
        ax.legend(frameon=False)
    fig.suptitle("NYC V8 | full known support | matched recall on development folds")
    fig.savefig(out/"matched_recall.png",dpi=160)
    plt.close(fig)


def operating(rows,model,choice=None):
    output=[]
    for rule in ("max_ba","max_f1","constrained"):
        a=rows[model][rule]
        label=RULES[rule]+(" (frozen)" if rule==choice else "")
        output.append(f"| {label} | {a['AP']:.4f} | {a['balanced_accuracy']:.4f} | {a['f1']:.4f} | "
                      f"{a['recall']:.4f} | {a['fp_per_day']:.3f} | {a['recovery_FP_day']:.3f} |")
    return output


def render(root,cfg):
    base=root/"results/analysis"
    summary=load(base/"threshold_summary.json")
    ranking=load(base/"ranking_summary.json")
    status=load(root/"evidence/status.json")
    gate=load(root/"evidence/date_gate.json")
    lock=load(root/"evidence/freeze.json")
    proof=load(root/"evidence/verification_replay.json")
    bands=pd.read_csv(base/"paired_date_blocks.csv")
    model,rule=status["frozen_model"],status["frozen_rule"]
    plots(root,cfg,summary,model)
    old=summary["calibration"]["attention"]["max_ba"]
    chosen=summary["calibration"][model][rule]
    dev=summary["development_transfer"][model]
    bounds=bands[(bands.stage=="development")&(bands.contrast=="rank_minus_attention")].set_index("metric")
    rank=ranking["development"]
    n=sum(load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]["positive"]+
          load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]["negative"] for fold in cfg["folds"])
    support=load(root/"evidence/fold_lead_final.json")["parts"][1]
    head=["| Threshold | AP | BA | Positive F1 | Recall | Total FP/day | Recovery FP/day |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines=[
        "# Project B V8 Audit Report",
        "",
        f"**Threshold-transfer diagnostics and 28-day independent calibration completed. The frozen choice is {NAMES[model]} plus {RULES[rule]}.** "
        "Formal NYC validation remains blocked/N/A. The original Chicago and Geolife protocols are separately blocked.",
        "",
        "## Timing Contract, Ground Truth, and Samples",
        "",
        "The fixed new holdout is 2026-11-09 through 12-06 in NYC local time. The V5/V6 dates from 08-31 through 11-08 are excluded from training, calibration, and selection. "
        f"The latest official timestamp was {gate['latest'] or 'N/A'}, before the registered period ended. There are no new speeds, labels, or formal results. "
        "The experiment uses V7's verified strict-lead inputs. Speeds from t−1, t, and later are available only for ground truth. "
        "[Date gate](evidence/date_gate.json) · [Registered plan](plan.md) · [NYC DOT data](https://data.cityofnewyork.us/d/i4gi-tjb9) · "
        "[Chicago discontinuation](https://data.cityofchicago.org/stories/s/c28d-stgw)",
        "",
        f"The 3 development folds contain {n:,} known windows. The 28-day calibration set contains {support['positive']+support['negative']:,} known windows, "
        f"{support['positive']:,} positives, {support['high_risk_recovered']:,} recovery negatives, and {support['unknown_eligible']:,} unknown windows. "
        "Within each fold, 3 models and 3 seeds share keys, labels, risk audits, and training samples. "
        "Reused 24 verified sets of V7 Attention and center-only predictions and weights. Trained the new Attention tail-ranking version from scratch 12 times. "
        "[Input evidence](evidence/inputs.json) · [Calibration support](evidence/fold_lead_final.json) · [Method and timing](method_and_data_contract.md)",
        "",
        "## Matched-Recall Ranking Comparison",
        "",
        "Take prefixes containing the same number of true positives on the full known support. The three threshold rules change the operating point while the model's AP and matched-recall ranking stay fixed. "
        "Tables show three-seed means. Development first averages across all three folds within each seed.",
        "",
        "| Stage | Scoring model | AP | Total FP/day at 60% recall | Recovery FP/day at 60% recall |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for stage,label in (("development","Three development folds"),("calibration","Independent calibration")):
        for name in ("attention","center","rank"):
            a=ranking[stage][name]
            lines.append(f"| {label} | {NAMES[name]} | {a['AP']:.4f} | {a['FP60_day']:.3f} | {a['recovery_FP60_day']:.3f} |")
    delta=rank["rank"]["recovery_FP60_day"]-rank["attention"]["recovery_FP60_day"]
    inter=bounds.loc["recovery_FP60_day"]
    lines+=["",
        f"The ranking candidate met the preregistered development admission criteria. At the primary recall target, the recovery FP/day difference was {delta:+.3f}, with reductions in two seeds. "
        f"The 1,000-resample date-block 95% interval relative to V7 Attention was [{inter.low_95:+.3f}, {inter.high_95:+.3f}]. "
        "At matched recall on calibration, recovery false positives were slightly higher than both baselines. The current evidence supports the threshold trade-off. Ranking improvements still require independent dates. "
        "[Paired intervals](results/analysis/paired_date_blocks.csv) · [Full matched recall](results/analysis/matched_recall.csv) · [Curves](figures/matched_recall.png)",
        "",
        "## Development Threshold Transfer and Independent Calibration",
        "",
        "Fit thresholds on the first seven days of each development fold and test transfer on the second seven days, using the model checkpoint selected from the entire fold. "
        "The 28-day independent calibration table gives a same-split threshold diagnostic. The package records all thresholds and per-seed counts. The registered daily limits are 12 total and 6 recovery false positives.",
        "",
        f"### {NAMES[model]} on the Last Seven Days of Each Development Fold",
        "",
        *head,
        *operating(summary["development_transfer"],model),
        "",
        f"### {NAMES[model]} on 28-Day Independent Calibration",
        "",
        *head,
        *operating(summary["calibration"],model,rule),
        "",
        f"The max-F1 rule satisfies the average budgets in development transfer and exceeds them on independent calibration. "
        f"{RULES[rule]} meets the freeze criteria in both splits. The three seed thresholds were locked at {lock['at_utc']}. "
        f"Relative to the original V7 Attention max-BA threshold, calibration positive F1 changes from {old['f1']:.4f} to {chosen['f1']:.4f}, "
        f"recovery FP/day from {old['recovery_FP_day']:.3f} to {chosen['recovery_FP_day']:.3f}, "
        f"and recall from {old['recall']:.4f} to {chosen['recall']:.4f}. "
        "[Thresholds, false positives, and confusion counts](results/analysis/thresholds.csv) · [Freeze receipt](evidence/freeze.json) · [Calibration figure](figures/calibration_thresholds.png)",
        "",
        "| Same calibration false-positive constraints | Positive F1 | BA | Recovery FP/day |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("attention","center","rank"):
        v=summary["calibration"][name]["constrained"]
        lines.append(f"| {NAMES[name]} | {v['f1']:.4f} | {v['balanced_accuracy']:.4f} | {v['recovery_FP_day']:.3f} |")
    lines += ["",
        "Under the same constraints, calibration F1 is similar across all three scoring models, with the two V7 models slightly higher. Most of the positive F1 change comes from moving the operating point. "
        "The budgets are measured on calibration and development transfer. False positives and recall on the formal dates await complete coverage. "
        "[Summary for the three models](results/analysis/threshold_summary.json)",
        "",
        "## Acceptance",
        "",
        f"Independently recomputed {proof['score_sets']} score sets, {proof['threshold_evaluations']} threshold diagnostics, and {proof['matched_points']} matched-recall points. "
        f"Sample replay covered {proof['replayed']} groups, with a maximum score difference of {proof['max_score_difference']:.2g}. "
        "The 40 engineering tests verify the implementation. The package includes frozen hashes, commands, observed exit codes, data mappings, and clean extraction verification. "
        "[Recomputation](evidence/verification_replay.json) · [Tests](test_report.md) · [Reproduction](README_audit.md)",
        ""]
    (root/"audit_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"audit_report.md"),model=model,rule=rule,formal=status["NYC_formal"]),flush=True)


if __name__=="__main__":
    render(*settings())

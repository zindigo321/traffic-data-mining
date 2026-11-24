"""Render the V6 audit, keeping both paired questions visible."""
from datetime import date,timedelta
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from v6_io import V5,load,settings
sys.path.insert(0,str(V5))
from v5_prepare import read_npz

LABELS={"X1_mean":"X1 + Mean (V5 control)","X1_max":"X1 + Max","X1_attention":"X1 + Attention (preselected)",
        "B_mean":"V4 P1 + Mean refit"}
FIGLABELS={"X1_mean":"V5 X1 + Mean refit","X1_max":"X1 + Max","X1_attention":"X1 + Attention (primary)",
           "B_mean":"V4 P1 + Mean refit"}
CONTROLS={"V5_X1_Mean":"V5 X1 + Mean","V4_P1_Mean":"V4 P1 + Mean"}
COLORS=["#525c66","#af7128","#196d91","#765193"]


def images(root,counts,aggregates,paired):
    dest=root/"figures"
    dest.mkdir(exist_ok=True)
    order=list(LABELS)
    fig,ax=plt.subplots(figsize=(8.9,4.8),layout="constrained")
    for name,color in zip(order,COLORS,strict=True):
        r,p=name.split("_")
        view=counts[(counts.recipe==r)&(counts.pool==p)].groupby("target_recall").FP_per_day.mean()
        ax.plot(100*view.index,view.values,marker="o",color=color,label=FIGLABELS[name])
    ax.axvline(60,lw=1,ls="--",color="#666666")
    ax.set(xlabel="Target recall (%)",ylabel="Recovery false alerts per calendar day",
           title="NYC V6 independent holdout | matched recall")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.25)
    fig.savefig(dest/"matched_recall.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.9,4.8),layout="constrained")
    for name,color in zip(order,COLORS,strict=True):
        raw=read_npz(root/"results/holdout"/f"{name}_s11"/"all_eligible.npz")
        include=raw["y"]>=0
        precision,recall,_=precision_recall_curve(raw["y"][include],raw["score"][include])
        ax.step(recall,precision,where="post",label=FIGLABELS[name],color=color)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",title="NYC V6 independent holdout | seed 11")
    ax.grid(alpha=.25)
    ax.legend(frameon=False,loc="lower left")
    fig.savefig(dest/"holdout_PR.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout="constrained")
    names=list(paired)
    change=np.array([paired[k]["mean_FP_per_day_delta"] for k in names])
    edges=np.array([paired[k]["FP_per_day_95"] for k in names])
    ax.errorbar(change,np.arange(len(names)),xerr=np.abs(edges.T-change),fmt="o",capsize=5,color="#196d91")
    ax.axvline(0,color="#777777",lw=1,ls="--")
    ax.set(yticks=np.arange(len(names)),yticklabels=["vs V5 X1 + Mean","vs V4 P1 + Mean"],
           xlabel="Attention minus control, recovery false alerts/day",title="NYC V6 | matched 60% recall, paired 95% date intervals")
    ax.grid(axis="x",alpha=.25)
    fig.savefig(dest/"paired_date_intervals.png",dpi=160)
    plt.close(fig)


def render(root,cfg):
    state=load(root/"evidence/status.json")
    proof=load(root/"evidence/verification_replay.json")
    gate=load(root/"evidence/date_gate.json")
    integrity=load(root/"evidence/v5_integrity.json")
    pre=load(root/"evidence/registration.json")
    if state["status"]!="complete":
        (root/"audit_report.md").write_text(
            "# Project B V6 Audit Report\n\n**Formal New York validation is blocked/N/A. The original Chicago protocol remains blocked.**\n\n"
            +state["reason"]+"\n\nThe package includes 12 refits and their calibration, date evidence, tests, and logs.\n",encoding="utf-8")
        return
    out=root/"results/holdout"
    lock=load(root/"evidence/freeze.json")
    unlock=load(root/"evidence/unlock.json")
    prior=load(root/"evidence/v5_historical_only.json")
    graph=load(root/"evidence/graph.json")
    data=load(root/"evidence/fold_holdout.json")
    support=data["parts"][0]
    receipt=load(root/"evidence/holdout_support.json")
    cal=load(root/"evidence/fold_final.json")["parts"]
    aggregates=load(out/"aggregates.json")
    comparisons=load(out/"comparisons.json")
    points=pd.read_csv(out/"matched_recall.csv",float_precision="round_trip")
    pairs=pd.read_csv(out/"paired.csv",float_precision="round_trip")
    current=pairs[pairs.target_recall==cfg["main_recall"]]
    images(root,points,aggregates,comparisons)
    a,b=map(date.fromisoformat,cfg["holdout"])
    inclusive=b-timedelta(days=1)
    actual=current.iloc[0].achieved_recall
    success="met" if state["goal_reached"] else "did not meet"
    both="met" if state["stable_vs_both"] else "did not meet"
    pin=", ".join(f"{k}={v}" for k,v in cfg["epochs"].items())
    lines=[
        "# Project B V6 Audit Report",
        "",
        f"**On the new New York holdout, preselected X1 + Attention {success} the dual 0.60 target and {both} the stability criterion for reducing recovery false positives against both controls.** "
        f"The three-seed mean AP was {state['AP']:.4f} and positive-class F1 was {state['F1']:.4f}. The original Chicago protocol remains **blocked/N/A**. Geolife is reported separately as blocked.",
        "",
        "## Dates, Samples, and Evidence",
        "",
        f"The V5 ZIP passed checks for {integrity['manifest_rows']} manifest entries and {integrity['hashes']} SHA hashes. "
        "The entire previous New York holdout, 2026-08-31 through 2026-10-11, was excluded from V6 training, statistics, calibration, and scoring. Only published aggregate results are cited. "
        f"The fresh dates were set using official counts without speeds and registered at {pre['at_utc']}. "
        "[V5 verification](evidence/v5_integrity.json) · [Historical reference](evidence/v5_historical_only.json) · "
        "[Date gate](evidence/date_gate.json) · [Preregistration](registered_holdout.md)",
        "",
        f"The fresh New York holdout ran from **{a} through {inclusive}** over {gate['calendar_days']} days, of which {gate['substantial_days']} met the release criterion. "
        "October 12 had no observations, and the October 13 morning peak was unreported. The original calendar and road denominators were retained. "
        f"New shards contained {data['funnel']['rows']:,} raw rows, including {data['funnel']['rows']-data['funnel']['outside_pinned_roster']:,} within the original roster. "
        f"The comparison count from the timestamp query was {receipt['raw_roster_count_in_metadata_gate']:,} rows. "
        f"The fixed network has {graph['road_count']} directed road segments and {graph['edges']} connection edges. Known formal holdout windows come from {support['known_roads']} segments. "
        "[Official NYC DOT source](https://data.cityofnewyork.us/d/i4gi-tjb9) · [Shards and hashes](evidence/download_holdout_sealed.json) · [Graph](evidence/graph.json)",
        "",
        "| Period | Eligible windows | Known windows | Positive | Recovery negatives | Proxy events | Known coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label,c in (("Final training",cal[0]),("Independent calibration",cal[1]),("Fresh New York holdout",support)):
        lines.append(f"| {label} | {c['eligible']:,} | {c['positive']+c['negative']:,} | {c['positive']:,} | "
                     f"{c['high_risk_recovered']:,} | {c['true_events']:,} | {c['known_coverage']:.2%} |")
    lines+=["",
        f"The formal grid has {support['expected']:,} expected windows. The {support['unknown_eligible']:,} eligible unknown graph windows receive scores and are excluded from supervised evaluation. "
        f"The known positive rate is {support['positive']/state['n']:.2%}. New York combines two half-bins into each ten-minute speed estimate. "
        "Chicago uses bus probes. Geometry-based connections, missing reports, repeated estimates from the provider, and seasonality limit cross-city interpretation. "
        "[Holdout funnel](evidence/fold_holdout.json) · [Original protocol status](evidence/chicago_status.json) · [Official Chicago notice](https://data.cityofchicago.org/stories/s/c28d-stgw)",
        "",
        "## Training, Calibration, and One-Time Unsealing",
        "",
        "Four model groups were refitted from scratch with three seeds each. X1 + Mean is both a pooling variant and the V5 control. "
        "The three earlier development folds, final training, fixed topology, and history were rebuilt and verified within the permitted V5 scope. The previous holdout was excluded. "
        f"Checkpoints follow the V5 development settings from before the previous unsealing ({pin}). Each run completed 12 epochs. "
        f"Thresholds maximize positive-class F1 on the same independent 28-day calibration period. Freeze time {lock['frozen_utc']}. Unseal time {unlock['unlocked_utc']}. "
        "[Development arrays](evidence/prepared_development.json) · [Calibration](results/calibration.csv) · [Freeze](evidence/freeze.json) · [Unseal](evidence/unlock.json)",
        "",
        "The first adapter passed the fixed checkpoint table as the total epoch count, causing a failure at fit startup. The adapter was corrected before unsealing. "
        "Registered dates, the 12-epoch budget, fixed checkpoints, and all controls were retained. Failure logs and partial artifacts were kept. "
        "[Adapter fix receipt](evidence/trainer_adapter_fix.json)",
        "",
        "## Full Fresh-Holdout Results",
        "",
        "| Recipe | AP mean ± SD | Positive-class F1 mean ± SD | ROC-AUC | Precision | Recall | Frozen-threshold FP/day | Matched-recall recovery FP/day |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key,name in LABELS.items():
        s=aggregates[key]
        lines.append(f"| {name} | {s['AP']:.4f} ± {s['AP_sd']:.4f} | {s['f1']:.4f} ± {s['f1_sd']:.4f} | "
                     f"{s['ROC_AUC']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | "
                     f"{s['fp_per_day']:.3f} | {s['FP60_per_day']:.3f} |")
    lines+=["",
        "AP uses average_precision_score. F1 uses the calibration thresholds frozen before unsealing. See the metrics table for seeds, accuracy, TP/FP/FN/TN, event results, and resource use. "
        "[Metrics by seed](results/holdout/metrics.csv) · [PR](figures/holdout_PR.png)",
        "",
        "## Two Paired Comparisons at Equal Recall",
        "",
        f"The primary point achieved **{actual:.3%}** recall, detecting **{int(current.iloc[0].TP)}** positives for each seed and model. "
        "All false positives at this point are known negatives with low speed in the previous bin and recovery in the current bin. The 1,000 date-block draws use shared sampled dates. The 95% difference interval is Attention minus control. "
        "Matched recall uses observed labels to choose a rank cutoff. Frozen thresholds are reported separately. The V5 control holds X1 features fixed and mainly compares pooling. The V4 control compares the complete recipes. ",
        "",
        "| Control | Control FP/day | Attention FP/day | 95% interval for daily difference | Seeds with decreases | Stability criterion met |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key,s in comparisons.items():
        lo,hi=s["FP_per_day_95"]
        stable="Yes" if s["stable"] else "No"
        lines.append(f"| {CONTROLS[key]} | {s['control_FP_per_day']:.3f} | {s['primary_FP_per_day']:.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {s['seeds_with_fewer_FP']}/3 | {stable} |")
    lines+=["",
        "Recovery false positives and positive-class false negatives on fixed support are broken down by low-speed streak length, peak period, segment, and date. The date intervals reflect this month. Generalization to sparsely observed months and other roads remains unverified. "
        "[Pairs by seed](results/holdout/paired.csv) · [Date blocks](results/holdout/date_blocks.csv) · "
        "[Matched-recall curve](figures/matched_recall.png) · [Error groups](results/holdout/errors_grouped.csv)",
        "",
        "## Verification",
        "",
        f"Independent recomputation covered {proof['calibrations']} calibration runs and {proof['heldout_runs']} formal holdout results. "
        f"Replay sampled {proof['calibration_replayed']+proof['heldout_replayed']} prediction sets, with a maximum probability difference of {proof['max_probability_difference']:.3g}. "
        "The 28 engineering tests check the implementation. "
        "[Recomputation](evidence/verification_replay.json) · [Test report](test_report.md)",
        "",
        "1. Verify signatures for the original V5 ZIP, the package manifest, and external raw inputs. Confirm the original V5 holdout was excluded from the rerun.",
        "2. Rebuild the graph, history, and prepared arrays from the earlier development data and newly sealed shards. Check registration, dates, and denominators that include unknowns.",
        "3. Verify the model and threshold freeze, the single unsealing timestamp, and per-window scores on common support.",
        "4. Run verify to independently recompute thresholds, full metrics, matched-recall pairs, and intervals. Use SHA-verified arrays for safe replay.",
        "",
        "[Reproduction and navigation](README_audit.md). The clean extraction receipt is next to the ZIP.",
        ""]
    (root/"audit_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"audit_report.md"),primary=state["primary"],goal=state["goal_reached"],
          stable_vs_both=state["stable_vs_both"]),flush=True)


if __name__=="__main__":
    render(*settings())

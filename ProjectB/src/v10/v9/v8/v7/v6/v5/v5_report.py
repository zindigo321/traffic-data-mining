"""Generate the audit from real external measurements and frozen receipts."""
from datetime import date,timedelta

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from v5_io import load,settings
from v5_prepare import read_npz


def readable_dates(interval):
    a,b=map(date.fromisoformat,interval)
    return f"{a} through {b-timedelta(days=1)}"


def plots(root,metrics,points,choice,groups):
    out=root/"figures"
    out.mkdir(exist_ok=True)
    colors=["#166e89","#9c5c17","#674f9b","#414a54"]
    labels=["V5 Mean","V5 Max","V5 Attention","V4 P1 + Mean refit"]
    fig,ax=plt.subplots(figsize=(8.7,4.7),layout="constrained")
    for name,label,col in zip(groups,labels,colors,strict=True):
        recipe,pool=name.split("_")
        view=points[(points.recipe==recipe)&(points.pool==pool)].groupby("target_recall").FP_per_day.mean()
        ax.plot(100*view.index,view.to_numpy(),label=label,color=col,marker="o")
    ax.axvline(60,color="#777777",lw=1,ls="--")
    ax.set(xlabel="Target recall (%)",ylabel="Recovery false alerts per calendar day",
           title="NYC independent holdout | identical known support")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.25)
    fig.savefig(out/"matched_recall.png",dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.7,4.7),layout="constrained")
    for name,label,col in zip(groups,labels,colors,strict=True):
        recipe,pool=name.split("_")
        row=metrics[(metrics.recipe==recipe)&(metrics.pool==pool)&(metrics.seed==11)].iloc[0]
        z=read_npz(root/"results/holdout"/row.job/"all_eligible.npz")
        known=z["y"]>=0
        precision,recall,_=precision_recall_curve(z["y"][known],z["score"][known])
        ax.step(recall,precision,where="post",color=col,label=f"{label} (AP {row.AP:.3f})")
    ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02),
           title="NYC independent holdout | PR curves, seed 11")
    ax.legend(frameon=False,loc="lower left")
    ax.grid(alpha=.25)
    fig.savefig(out/"holdout_PR.png",dpi=160)
    plt.close(fig)

    data=load(root/"results/holdout/aggregates.json")
    fig,ax=plt.subplots(figsize=(9.4,4.6),layout="constrained")
    x=np.arange(len(groups))
    for metric,delta,color,text in (("AP",-.18,"#166e89","AP"),("f1",.18,"#a5682a","Positive F1")):
        ax.bar(x+delta,[data[g][metric] for g in groups],.34,yerr=[data[g][metric+"_sd"] for g in groups],
               capsize=4,color=color,label=text)
    ax.axhline(.6,lw=1,ls="--",color="#333333",label="Registered 0.60")
    ax.set(ylim=(0,1.05),ylabel="Three-seed mean ± SD",xticks=x,xticklabels=labels,
           title="NYC independent holdout | frozen selection and threshold")
    ax.legend(frameon=False,ncols=3,loc="lower center")
    ax.grid(axis="y",alpha=.22)
    fig.savefig(out/"holdout_comparison.png",dpi=160)
    plt.close(fig)


def render(root,cfg):
    out=root/"results/holdout"
    state=load(root/"evidence/status.json")
    graph=load(root/"evidence/graph.json")
    lock=load(root/"evidence/freeze.json")
    opened=load(root/"evidence/unlock.json")
    dev=load(root/"evidence/prepared_development.json")
    verification=load(root/"evidence/verification_replay.json")
    recipe,pool=lock["choice"]["recipe"],lock["choice"]["pool"]
    selected=f"{recipe}_{pool}"
    groups=[f"{recipe}_{p}" for p in cfg["pools"]]+["B_mean"]
    labels=[f"{recipe} + Mean",f"{recipe} + Max",f"{recipe} + Attention","V4 P1 + Mean common-support refit"]
    names=dict(zip(groups,labels,strict=True))
    aggregate=load(out/"aggregates.json")
    screen=load(root/"results/screen.json")["summary"]
    metrics=pd.read_csv(out/"metrics.csv",float_precision="round_trip")
    points=pd.read_csv(out/"matched_recall.csv",float_precision="round_trip")
    pairs=pd.read_csv(out/"paired.csv",float_precision="round_trip")
    bands=pd.read_csv(out/"date_blocks.csv",float_precision="round_trip")
    plots(root,metrics,points,selected,groups)
    ref=aggregate["B_mean"]
    best=aggregate[selected]
    p_main=pairs[(pairs.recipe==recipe)&(pairs.pool==pool)&(pairs.target_recall==cfg["main_recall"])]
    held=load(root/"evidence/fold_holdout.json")["parts"][0]
    day=readable_dates(cfg["holdout"])
    raw_ratio=dev["source_funnel"]["implied_miles_over_polyline_miles_p10_median_p90"]
    stable="met" if state["stable_recovery_FP_reduction"] else "did not meet"
    goal="met" if state["goal_reached"] else "did not meet"
    lines=[
        "# Project B V5 Audit Report",
        "",
        f"**The New York external holdout {goal} the dual 0.60 target and {stable} the preregistered criterion for a stable decrease in recovery false positives.** "
        f"The development-selected {names[selected]} scored AP {state['AP']:.4f} and positive-class F1 {state['F1']:.4f}. Formal Chicago validation under the original protocol remains **blocked/N/A**.",
        "",
        "## 1. Data Boundaries, Protocol, and Evidence",
        "",
        "Chicago table 4g9f-3jbs still ends on 2026-04-30. The query from May 1 onward returns no records for the same roads. Daily traffic volumes published after discontinuation lack comparable segment speeds. "
        "V4 Chicago development results and earlier holdouts are historical references. Geolife remains blocked and is reported separately. "
        "[Query receipt](evidence/feasibility.json) · [Chicago status](evidence/chicago_status.json) · "
        "[Official discontinuation notice](https://data.cityofchicago.org/stories/s/c28d-stgw) · [Geolife](evidence/geolife_prior.json)",
        "",
        f"New York uses NYC DOT table i4gi-tjb9. Dates were selected using metadata without speeds and registered at {load(root/'evidence/registration.json')['at_utc']}. "
        "The official data provide point-to-point speed estimates on arterial roads. The study uses New York's local calendar, two half-bin aggregates, its own directed graph, and newly fitted weights. "
        "The two cities' results are reported separately. The field dictionary lists status as a legacy field, and the amendment was locked before speed parsing. "
        f"The development snapshot's median ratio of miles implied by speed and travel time to geometric length is {raw_ratio[1]:.3f}. "
        "[Preregistration](protocol_nyc.md) · [Field amendment](amendment_nyc.md) · "
        "[Official NYC DOT data](https://data.cityofnewyork.us/d/i4gi-tjb9) · "
        "[NYC DOT field descriptions](https://www1.nyc.gov/html/dot/downloads/pdf/metadata-trafficspeeds.pdf)",
        "",
        f"The anchor snapshot has 125 IDs. The fixed graph retains {graph['road_count']} roads, {graph['edges']} directed edges, and {graph['isolated']} isolated nodes. "
        f"The largest observed subgraph has {graph['max_subgraph_nodes']} nodes. For the {graph['metadata_link_conflicts']} IDs with multiple metadata variants, the anchor mode is used. "
        f"Known formal holdout windows come from {held['known_roads']} segments. The expected grid for all fixed segments counts toward coverage. "
        "Proximity-based connections can miss grade separation and physical turns. "
        "[Nodes and edges](evidence/graph.json) · [Availability and labels](method_and_data_contract.md)",
        "",
        "| Period | New York local dates | All known windows | Positives | Recovery negatives | Proxy events | Known coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    periods={**cfg["folds"],"final":cfg["calibration"],"holdout":cfg["holdout"]}
    for key,interval in periods.items():
        cov=load(root/"evidence"/f"fold_{key}.json")["parts"][0 if key=="holdout" else 1]
        label={"final":"Calibration","holdout":"Formal holdout"}.get(key,key.upper())
        lines.append(f"| {label} | {readable_dates(interval)} | {cov['positive']+cov['negative']:,} | "
                     f"{cov['positive']:,} | {cov['high_risk_recovered']:,} | {cov['true_events']:,} | {cov['known_coverage']:.2%} |")
    lines += ["",
        f"The formal holdout covers {held['total_days']} days with {held['expected']:,} fixed windows and {held['eligible']:,} eligible windows. "
        f"The {held['unknown_eligible']:,} eligible unknowns receive scores and are excluded from supervised evaluation. The positive rate is {held['positive']/state['n']:.2%}. "
        f"The positive rate among known risk-group windows with low speed in the previous bin is {held['positive']/(held['positive']+held['high_risk_recovered']):.2%}. "
        "Risk-group prevalence differs substantially from Chicago. Both half-bins can reflect an estimate carried forward by the same provider, so the proxy label does not prove actual congestion throughout five continuous minutes. "
        "[Funnel and labels](evidence/fold_holdout.json) · [Full-support summary](evidence/holdout_support.json)",
        "",
        "## 2. Development Selection and Formal Freeze",
        "",
        "| Initial screening, three folds and one seed | AP | Within-fold diagnostic F1 | FP/day at 60% recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name,value in screen.items():
        lines.append(f"| {name} | {value['AP']:.4f} | {value['f1']:.4f} | {value['FP60_per_day']:.3f} |")
    lines += ["",
        f"Initial screening selected {recipe}. After confirming the three pooling variants, development AP selected {names[selected]}. "
        "B refits the V4 P1 recipe in New York using the same graph, samples, number of training steps, and threshold rule. "
        "After selection, the 12 models chose maximum-F1 thresholds on the same independent 28-day calibration period. "
        f"Freeze time {lock['frozen_utc']}. Unseal time {opened['unlocked_utc']}. "
        "Calendar, graph, code, preprocessing, weights, and threshold hashes were recorded before unsealing. "
        "[Confirmation](results/confirmation.csv) · [Freeze](evidence/freeze.json) · [Unseal](evidence/unlock.json)",
        "",
        "## 3. Full New York Holdout Results",
        "",
        "| Recipe | AP mean ± SD | Positive-class F1 mean ± SD | ROC-AUC | Precision | Recall | Frozen-threshold FP/day | Event F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in groups:
        a=aggregate[g]
        name=f"**{names[g]} (preselected)**" if g==selected else names[g]
        lines.append(f"| {name} | {a['AP']:.4f} ± {a['AP_sd']:.4f} | {a['f1']:.4f} ± {a['f1_sd']:.4f} | "
                     f"{a['ROC_AUC']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['fp_per_day']:.3f} | {a['event_f1']:.4f} |")
    lines += ["",
        f"Against the V4 common-support refit, the preselected group scored ΔAP {best['AP']-ref['AP']:+.4f} and ΔF1 {best['f1']-ref['f1']:+.4f}. "
        "The V4 refit also exceeded both 0.60 targets. See the metrics table for frozen-threshold confusion counts, accuracy, TP/FP/FN/TN by seed, and event statistics. "
        "The formal decision uses the preselected pooling method. Results for all three methods are retained. "
        "[Metrics by configuration](results/holdout/metrics.csv) · [Comparison plot](figures/holdout_comparison.png) · [PR](figures/holdout_PR.png)",
        "",
        "## 4. Recovery False Positives at Matched Recall",
        "",
        f"The primary comparison achieved {p_main.iloc[0].actual_recall:.3%} recall, with {int(p_main.iloc[0].TP):,} positives detected per model. "
        "Every false positive at this point had low speed in the previous bin and recovery in the current bin. Date blocks are shared across all seeds and paired models.",
        "",
        "| Seed | V4 recovery FP | Preselected V5 recovery FP | V5−V4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in p_main.itertuples():
        lines.append(f"| {row.seed} | {row.V4_recovery_FP} | {row.V5_recovery_FP} | {row.delta_recovery_FP:+d} |")
    lo,hi=state["delta_FP_95"]
    decrease=(ref["FP60_per_day"]-best["FP60_per_day"])/ref["FP60_per_day"]
    lines += ["",
        f"Mean recovery FP/day across three seeds declined from {ref['FP60_per_day']:.3f} to {best['FP60_per_day']:.3f} ({decrease:.1%}). "
        f"The 95% difference interval from 1000 paired date-block draws is [{lo:+.3f}, {hi:+.3f}] per day. "
        f"Recovery false positives decreased for {state['seeds_with_fewer_FP']} seeds. The preregistration requires decreases for at least two seeds and an upper bound below zero. The improvement {stable} the stability criterion. "
        "Matched recall uses observed labels to choose a rank cutoff. The deployment threshold comes from independent calibration. "
        "[Full matched recall](results/holdout/matched_recall.csv) · [Date-block intervals](results/holdout/date_blocks.csv) · "
        "[Grouped false positives and negatives](results/holdout/errors_grouped.csv) · [False-positive curve](figures/matched_recall.png)",
        "",
        "## 5. Verification and Limitations",
        "",
        f"Independent checks recomputed {verification['development_and_calibration_runs']} development and calibration runs and {verification['heldout_runs']} holdout runs. "
        f"Safe replay sampled {verification['dev_replayed']+verification['heldout_replayed']} runs. "
        f"The maximum probability difference was {verification['max_probability_difference']:.2g}. "
        "There are 23 engineering cases. Synthetic cases check the implementation. Commands, exit codes, durations, hardware, and seeds are recorded in logs. "
        "The fixed roads and season, geometry-based network connections, unreported segments, different probes, and label enrichment constrain cross-city interpretation. "
        "[Independent recomputation](evidence/verification_replay.json) · [Test report](test_report.md)",
        "",
        "1. Verify the package manifest, SHA256SUMS, original V4 ZIP, and external New York shard hashes.",
        "2. Rebuild the four prepared arrays and frozen history from anchor geometry and the development CSV files. Check chronological partitions, unknown labels, common fold support, and sampling.",
        "3. Verify registration, amendment, development selection, thresholds, weights, and the order of freeze and unseal.",
        "4. Filter all eligible per-window scores to known labels and recompute AP, F1, confusion counts, matched recall, and paired date intervals. Replay a sample using the fixed weights.",
        "",
        "[Run commands and navigation](README_audit.md). The clean extraction receipt is next to the ZIP.",
        ""]
    (root/"audit_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"audit_report.md"),figures=3,selected=selected,external_goal=state["goal_reached"],
               stable_recovery_FP_reduction=state["stable_recovery_FP_reduction"]),flush=True)


if __name__=="__main__":
    render(*settings())

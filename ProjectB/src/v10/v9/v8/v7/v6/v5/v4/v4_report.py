"""Build audit narrative and figures exclusively from measured development artifacts."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from v4_io import load, settings
from v4_matrix import location
from v4_train import read_npz

NAME = dict(B_mean="V3 R2 + Mean refit", P1="Tail ranking", P2="Directional evidence", P3="Tail ranking and directional evidence")


def figures(root, cfg, selected, model_names, points):
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.1, 4.6), layout="constrained")
    for model in model_names:
        values = points[points.model == model].sort_values("target_recall")
        ax.plot(100 * values.target_recall, values.mean_FP_per_day, marker="o", label=NAME.get(model, model.replace("_", " + ")))
    ax.axvline(60, color="#777777", lw=1, ls="--")
    ax.set(xlabel="Target recall (%)", ylabel="False alerts per calendar day",
           title="Chicago development | matched recall over all known windows")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(out / "matched_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.1, 4.7), layout="constrained")
    fold, seed = cfg["folds"][-1], cfg["seeds"][0]
    for model in model_names:
        recipe, pool = model.split("_")
        z = read_npz(location(root, fold, recipe, pool, seed) / "assessment.npz")
        precision, recall, _ = precision_recall_curve(z["y"], z["score"])
        ax.step(recall, precision, where="post", label=NAME.get(model, model.replace("_", " + ")))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
           title=f"Chicago development PR | {fold.upper()} · seed {seed}")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.savefig(out / "development_PR.png", dpi=160)
    plt.close(fig)

    errs = pd.read_csv(root / "results/paired_error_review/groups.csv")
    order = ["1", "2", "3+"]
    ref = errs[(errs.model == "B_mean") & (errs.dimension == "depth_bin")].set_index("category").loc[order]
    alt = errs[(errs.model == selected) & (errs.dimension == "depth_bin")].set_index("category").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), layout="constrained", sharey=True)
    x = np.arange(len(order))
    for ax, col, title in zip(axes, ("FP_rate", "FN_rate"), ("Recovery-negative false alarm rate", "Positive miss rate"), strict=True):
        ax.bar(x - .18, 100 * ref[col], .35, label="V3 R2 + Mean refit")
        ax.bar(x + .18, 100 * alt[col], .35, label=selected.replace("_", " + "))
        ax.set(xticks=x, xticklabels=order, ylim=(0, 100), title=title, xlabel="Prior consecutive low windows")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Percent of stratum")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Chicago development | target recall 60%")
    fig.savefig(out / "risk_errors_by_streak.png", dpi=160)
    plt.close(fig)


def render(root, cfg):
    select = load(root / "results/selection.json")
    chosen = select["recipe"] + "_" + select["pool"]
    screen = load(root / "results/screen.json")["summary"]
    confirm = load(root / "results/confirmation.json")["summary"]
    calibration = load(root / "results/calibration_summary.json")["summary"]
    proof = load(root / "evidence/verification_replay.json")
    gate = load(root / "evidence/holdout_gate.json")
    points = pd.read_csv(root / "results/comparison/matched_aggregates.csv")
    bands = pd.read_csv(root / "results/comparison/paired_date_blocks.csv")
    errors = pd.read_csv(root / "results/paired_error_review/groups.csv")
    latest = gate["queries"]["latest"].get("rows", [{}])[0].get("time", "N/A")
    models = [f"{select['recipe']}_{p}" for p in cfg["pools"]] + ["B_mean"]
    figures(root, cfg, chosen, models, points)
    lines = [
        "# Project B V4 Audit Report",
        "",
        "**Formal validation is blocked, and the dual 0.60 target for AP and positive-class F1 is N/A.** Round four completed three Chicago development folds and the original independent calibration. "
        "A V3 R2 + Mean refit is the common-support control. All V4 scores come from development or calibration dates.",
        "",
        "## Data Boundaries and Evaluation Support",
        "",
        f"The latest official API timestamp is {latest} in Chicago local time. Queries from 2026-05-01 onward for the same segment IDs return HTTP 200 with an empty array. "
        "The official discontinuation notice and catalog search revealed no verifiable successor with the same definition. The suggested alternative daily traffic volumes lack the original task's ten-minute speeds on directed road segments. "
        "Earlier years cannot provide forward validation after V3. [Query receipt](evidence/holdout_gate.json) · "
        "[Publisher discontinuation notice](https://data.cityofchicago.org/stories/s/c28d-stgw) · [Dataset](https://data.cityofchicago.org/d/4g9f-3jbs)",
        "",
        "For the V3 holdout from 2026-02-02 through 04-30, only the published R2 + Mean aggregate AP of 0.4356 and F1 of 0.4459 are cited. "
        "V1, V2, and V3 holdout observations were excluded from training, historical baselines, false-positive review, selection, and calibration. Holdout files in the V3 archive were used only to verify byte integrity. "
        "Geolife retains its original audited status of blocked/N/A. Chicago remains a supplemental proxy task. [Original input verification](evidence/inputs.json) · "
        "[V3 historical summary](evidence/v3_historical_only.json) · [Geolife](evidence/geolife_status.json)",
        "",
        "The protocol retains 231 directed segments, 220 edges, the original peaks, and ten-minute bins. A positive requires two consecutive speeds below 40% of their respective historical baselines. "
        "Unknown labels and prediction eligibility follow V3. Supervised evaluation covers every known graph window in each fold. Keys, labels, risk groups, and training samples match exactly within a fold.",
        "",
        "| Purpose | Chicago local dates (start inclusive, end exclusive) | Known windows | Positive | Recovery negatives | Positive episodes | Unknown graph windows | Known coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in cfg["folds"] + ["final"]:
        a, b = cfg["fold_limits"][fold]
        row = load(root / "evidence" / f"fold_{fold}.json")["parts"][1]
        lines.append(f"| {fold.upper() if fold != 'final' else '84-day calibration'} | {a} to {b} | {row['positive']+row['negative']:,} "
                     f"| {row['positive']:,} | {row['high_risk_recovered']:,} | {row['true_events']:,} "
                     f"| {row['unknown_eligible']:,} | {row['known_coverage']:.2%} |")
    lines += ["", "Known coverage uses the complete expected grid on the fixed network as the denominator. Episodes are adjacent positive windows on the same road and day. "
              "[Data funnel](results/comparison/support.csv) · [Availability contract](method_and_data_contract.md)",
              "", "## Paired Development Ablation", "",
              "The runs share sampling, Graph Transformer, routing, and update budget. Initial screening uses Mean, seed 11, and three folds. Development F1 uses the best threshold in the same fold for diagnosis.", "",
              "| Group | Change | AP | Diagnostic F1 | FP/day at 60% recall |",
              "| --- | --- | ---: | ---: | ---: |"]
    for model, value in screen.items():
        lines.append(f"| {model} | {NAME.get(model, 'V3 R2 + Mean refit')} | {value['AP']:.4f} | {value['f1']:.4f} | {value['FP60_per_day']:.3f} |")
    lines += ["", f"Preregistration rules shortlisted {select['recipe']}. Each pooling method ran three seeds on each of three folds, and development selected **{chosen}**. The table first averages the three folds equally within each seed, then calculates the standard deviation across the three seeds. "
              "[Initial screening](results/screen.json) · [Confirmation](results/confirmation.json) · [Selection receipt](results/selection.json)",
              "", "| Recipe | AP | Diagnostic F1 | FP/day at 60% recall |",
              "| --- | ---: | ---: | ---: |"]
    for model in models:
        value = confirm[model]
        label = f"**{model}**" if model == chosen else model
        lines.append(f"| {label} | {value['AP']:.4f} ± {value['AP_sd']:.4f} | {value['f1']:.4f} ± {value['f1_sd']:.4f} "
                     f"| {value['FP60_per_day']:.3f} ± {value['FP60_per_day_sd']:.3f} |")
    lines += ["", "## False Positives and Negatives at Matched Recall", "",
              "For each fold, take the shortest prefix of all ranked known scores that reaches the required TP count. All positives are in the high-risk branch. "
              "At these four recall points, every false positive is a known negative with low speed in the previous bin and recovery in the current one. Ties are ordered by sample key. The table also retains the count when a single threshold includes the entire tied group. "
              "These comparisons use observed labels in the fold and measure the ranking cost at equal recall. [Matched recall by fold](results/comparison/matched_points.csv) · "
              "[Paired differences](results/comparison/paired_matched_recall.csv) · [Matched-recall curve](figures/matched_recall.png)",
              "", "| Target recall | Actual mean recall across folds | V3 refit FP/day | V4 Mean | V4 Max | V4 Attention |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for rate in cfg["recalls"]:
        view = points[points.target_recall == rate].set_index("model")
        lines.append(f"| {rate:.0%} | {view.loc['B_mean', 'mean_recall']:.2%} | "
                     + " | ".join(f"{view.loc[model, 'mean_FP_per_day']:.3f}" for model in ["B_mean", *models[:3]]) + " |")
    ref, alt = confirm["B_mean"], confirm[chosen]
    ci = bands[bands.model == chosen].set_index("metric")
    lines += ["", f"{chosen} differed from the V3 refit in development by ΔAP {alt['AP']-ref['AP']:+.4f}, "
              f"ΔF1 {alt['f1']-ref['f1']:+.4f}, and matched-recall ΔFP/day {alt['FP60_per_day']-ref['FP60_per_day']:+.3f}. "
              f"The 95% intervals from 250 paired date-block draws were [{ci.loc['AP', 'lower_95']:+.4f}, {ci.loc['AP', 'upper_95']:+.4f}], "
              f"[{ci.loc['diagnostic_F1', 'lower_95']:+.4f}, {ci.loc['diagnostic_F1', 'upper_95']:+.4f}], and "
              f"[{ci.loc['FP60_per_day', 'lower_95']:+.3f}, {ci.loc['FP60_per_day', 'upper_95']:+.3f}], respectively. "
              "All three seeds and models share date blocks within a fold. F1 resampling keeps the original diagnostic thresholds fixed. Reuse of development dates and sparse events limit inference. "
              "[Paired intervals](results/comparison/paired_date_blocks.csv) · [Development PR](figures/development_PR.png)",
              "", "Grouping uses the length of the low-speed streak. Denominators and counts are seed-windows, so a road window can count once in each of three seeds.",
              "", "| Low-speed streak from the preceding bin | Recovery negative support | V3 false positives | V4 false positives | Positive support | V3 false negatives | V4 false negatives |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cat in ("1", "2", "3+"):
        rows = errors[(errors.dimension == "depth_bin") & (errors.category == cat)].set_index("model")
        before, after = rows.loc["B_mean"], rows.loc[chosen]
        lines.append(f"| {cat} | {before.recovery_support:,} | {before.FP:,} | {after.FP:,} "
                     f"| {before.positive_support:,} | {before.FN:,} | {after.FN:,} |")
    lines += ["", "The full grouping also includes previous-bin speed ratio, bus probe count, connectivity, peak period, day type, and road. "
              "Bus counts within the bin and low-speed streak length can reflect both observation quality and traffic conditions. Grouped errors do not support causal inference. "
              "[Risk by window key](results/paired_error_review/risk_windows.csv) · [Groups](results/paired_error_review/groups.csv) · "
              "[Error plot](figures/risk_errors_by_streak.png)",
              "", "## Existing Independent Calibration and Verification", "",
              "After development selection and checkpoint locking, diagnostic thresholds were fitted on the previously independent 84-day calibration period. Three seeds are retained for each group. "
              "V3 used this period for calibration. The scores below describe calibration fit in this round and are excluded from the dual 0.60 decision.", "",
              "| Group | AP | Maximum calibration F1 | Fixed epoch |",
              "| --- | ---: | ---: | ---: |"]
    for model in models:
        value = calibration[model]
        lines.append(f"| {model} | {value['AP']:.4f} ± {value['AP_sd']:.4f} | "
                     f"{value['f1']:.4f} ± {value['f1_sd']:.4f} | {select['fixed_epochs'][model]} |")
    lines += ["", f"Independent verification recomputed metrics, thresholds, and matched-recall counts for {proof['runs']} models. "
              f"The nine V3 development refits matched in samples, decisions, AP, F1, and matched-recall counts, with {proof['exact_V3_dev_runs']} bit-identical runs. "
              f"The maximum probability difference across the other runs was {proof['max_V3_probability_difference']:.3g}. The original failed bitwise-check logs are retained. "
              f"One inference batch was replayed for each of {proof['replays']} sets of weights. "
              f"The maximum probability difference was {proof['replay_max_probability_difference']:.2g}. "
              "Sealing receipts preserve hashes of weights, thresholds, sampled keys, preprocessing, configuration, and diagnostic results. "
              "[Calibration receipt](results/calibration_freeze.json) · [Metrics and replay](evidence/verification_replay.json) · [Test report](test_report.md)",
              "", "1. Verify the original V3 ZIP, raw-cache fingerprints, and this package manifest.",
              "2. Use the plan to check development and calibration dates, the sealed interval, and excluded intervals. Inspect labels, sample keys, unknown coverage, and samples in each fold's NPZ.",
              "3. Run verify for offline recomputation, or supply the fixed V3 arrays and original V3 archive to replay weights and development controls.",
              "4. Verify the formal status JSON, original label contract, and API receipts. A new preregistered holdout and formal validation require compatible future speed data.",
              "", "See [audit entry points](README_audit.md) for commands, environments, and data paths. The clean extraction receipt is next to the ZIP.",
              ""]
    (root / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(dict(report=str(root / "audit_report.md"), figures=3, chosen=chosen, formal="blocked"), flush=True)


if __name__ == "__main__":
    render(*settings())


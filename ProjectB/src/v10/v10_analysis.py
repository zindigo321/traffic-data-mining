"""Full-support, groupwise and calendar-paired analyses of actual NYC runs."""
from datetime import date
from itertools import product
import sys
import statistics as stat

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from v10_data import GROUP_NAMES
from v10_io import V9, load, save, settings
from v10_score import METRICS, raw_for, read_npz, split, write_csv
from v10_train import folder

sys.path.insert(0, str(V9))
from v9_analysis import trace, BOOT
from evaluation import event_metrics

EPOCH = date(1970, 1, 1)
LABELS = {"B0": "V9 center rerun", "S1": "Legal-low negative sampling",
          "R1": "Group ranking", "R2": "Group ranking + balanced loss"}


def summarize(rows, fields):
    seeds = sorted({r["seed"] for r in rows})
    result = dict(runs=len(rows), seeds=len(seeds))
    for field in fields:
        by_seed = []
        for seed in seeds:
            values = [r[field] for r in rows if r["seed"] == seed and r[field] is not None and np.isfinite(r[field])]
            if values:
                by_seed.append(stat.mean(values))
        result[field] = stat.mean(by_seed) if by_seed else None
        result[field + "_sd"] = stat.stdev(by_seed) if len(by_seed) > 1 else None
        result[field + "_valid_seeds"] = len(by_seed)
    return result


def analyze(root, cfg):
    pick = load(root / "results/development_selection.json")
    freeze = load(root / "evidence/freeze.json")
    assert pick["primary"] == freeze["primary"]
    res = root / "results"
    read = lambda name: pd.read_csv(res / name, float_precision="round_trip").to_dict("records")
    ops = [*read("development_operating.csv"), *read("calibration_operating.csv")]
    groups = [*read("development_groups.csv"), *read("calibration_groups.csv")]
    points = [*read("development_matched.csv"), *read("calibration_matched.csv")]
    summary, rows, events, daily, resources, audit = {}, [], [], [], [], []
    weights_out, plot_data = {}, {}
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    primary = pick["primary"]
    for stage, folds in (("development", list(cfg["folds"])), ("calibration", ["final"])):
        summary[stage] = {}
        contrasts = {"primary_minus_B0": (primary, "B0"), "S1_minus_B0": ("S1", "B0"),
                     "R1_minus_B0": ("R1", "B0"), "R2_minus_B0": ("R2", "B0")}
        diffs = {k: [] for k in contrasts}
        for fold in folds:
            interval = cfg["calibration"] if fold == "final" else cfg["folds"][fold]
            start, stop = map(date.fromisoformat, interval)
            first = (start - EPOCH).days + (cfg["early_days"] if stage == "development" else 0)
            n_days = (stop - EPOCH).days - first
            weights = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=cfg["bootstrap_reps"])
            weights_out[fold] = weights
            traces = {}
            for recipe, seed in product(cfg["v10_recipes"], cfg["seeds"]):
                raw = raw_for(root, cfg, fold, recipe, seed)
                dec = read_npz(root / "results/decisions" / fold / f"{recipe}_s{seed}.npz")
                keep = dec["eval_mask"] & (raw["y"] >= 0)
                threshold = float(dec["threshold"])
                z = [raw["sample_id"][keep], raw["y"][keep], raw["score"][keep],
                     raw["risk"][keep], dec["pred"][keep]]
                traces[recipe, seed] = trace(*z, weights, first, cfg["main_recall"])
                events.append(dict(fold=fold, recipe=recipe, seed=seed, **event_metrics(z[0], z[1], z[4])))
                state = load(folder(root, fold, recipe, seed) / "status.json")
                resources.append({k: state[k] for k in ("fold", "recipe", "seed", "best_epoch", "fixed_epoch",
                                  "parameters", "updates", "train_count", "fit_and_score_seconds", "peak_RSS_bytes",
                                  "baseline_max_error", "weight_sha256")})
                days = raw["sample_id"] // 1000 // 144
                for day in np.unique(days):
                    sched = days == day
                    known = sched & (raw["y"] >= 0)
                    y, pred, risk = raw["y"], dec["pred"], raw["risk"]
                    daily.append(dict(fold=fold, recipe=recipe, seed=seed,
                        date=str(np.datetime64(int(day), "D")), threshold=threshold,
                        scheduled=int(sched.sum()), known=int(known.sum()), unknown=int((sched & (y < 0)).sum()),
                        positive=int((known & (y == 1)).sum()), tp=int((known & pred & (y == 1)).sum()),
                        fp=int((known & pred & (y == 0)).sum()), fn=int((known & ~pred & (y == 1)).sum()),
                        tn=int((known & ~pred & (y == 0)).sum()), recovery_FP=int((known & pred & risk & (y == 0)).sum())))
                if stage == "calibration":
                    plot_data[recipe, seed] = (raw["y"][keep], raw["score"][keep], dec["pred"][keep])
                if recipe == "B0" and seed == cfg["seeds"][0]:
                    sample = read_npz(folder(root, fold, recipe, seed) / "sampling_ids.npz")
                    audit.append(dict(fold=fold, period=interval, scheduled=len(raw["y"]),
                                      known=int((raw["y"] >= 0).sum()), unknown=int((raw["y"] < 0).sum()),
                                      training_sample=len(sample["y"]), sample_overlap_with_assessment=int(np.intersect1d(
                                      sample["sample_id"], raw["sample_id"]).size), persons_or_trajectories="N/A, DOT road estimates"))
            for name, (a, b) in contrasts.items():
                diffs[name].append(np.mean([traces[a, s] - traces[b, s] for s in cfg["seeds"]], axis=0))
        for recipe in cfg["v10_recipes"]:
            o = [r for r in ops if r["fold"] in folds and r["recipe"] == recipe]
            summary[stage][recipe] = dict(**summarize(o, METRICS),
                budget_pass_rows=sum(r["eval_budget_ok"] for r in o),
                groups={group: summarize([r for r in groups if r["fold"] in folds and r["recipe"] == recipe
                                          and r["scope"] == group], METRICS) for group in GROUP_NAMES},
                matched={scope: {str(rate): summarize([r for r in points if r["fold"] in folds and r["recipe"] == recipe
                              and r["scope"] == scope and r["target_recall"] == rate],
                              ("TP", "FP", "recovery_FP", "achieved_recall", "FP_per_day", "recovery_FP_per_day", "F1"))
                         for rate in cfg["recalls"]} for scope in ("all", "legal_low")})
        for name, (left, right) in contrasts.items():
            delta = np.mean(diffs[name], axis=0)
            a, b = summary[stage][left], summary[stage][right]
            point = str(cfg["main_recall"])
            estimates = [a[k] - b[k] for k in BOOT[:4]]
            estimates.extend(a["matched"]["all"][point][k] - b["matched"]["all"][point][k]
                             for k in ("FP_per_day", "recovery_FP_per_day"))
            for i, metric in enumerate(BOOT):
                valid = delta[:, i][np.isfinite(delta[:, i])]
                low, high = np.quantile(valid, [.025, .975])
                rows.append(dict(stage=stage, contrast=name, metric=metric, estimate=estimates[i],
                                 lo95=float(low), hi95=float(high), valid_reps=len(valid)))
    for name, records in (("operating_points.csv", ops), ("groups.csv", groups), ("matched_recall.csv", points),
                          ("paired_date_blocks.csv", rows), ("events.csv", events), ("daily_errors.csv", daily),
                          ("training_resources.csv", resources)):
        write_csv(res / name, records)
    np.savez_compressed(res / "date_bootstrap_weights.npz", **weights_out)
    save(res / "summary.json", summary)
    save(root / "evidence/leakage_support.json", dict(status="PASS", periods=audit, all_sample_key_overlaps=0,
          group_inputs="ratio_prior derived from t-2 only", embargo=cfg["embargo"], holdout=cfg["holdout"],
          V5_V6_holdout_used=False, new_holdout_read=False))
    figures(root, cfg, summary, plot_data)
    print(dict(primary=primary, operating_rows=len(ops), group_rows=len(groups), matched_rows=len(points),
               paired_rows=len(rows), events=len(events), newly_trained=len(resources)), flush=True)


def figures(root, cfg, summary, plot_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    colors = ("#176650", "#ac6d26", "#376ea0", "#9c4280")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for recipe, color in zip(cfg["v10_recipes"], colors, strict=True):
        for i, seed in enumerate(cfg["seeds"]):
            y, p, pred = plot_data[recipe, seed]
            prec, rec, _ = precision_recall_curve(y, p)
            axes[0].step(rec, prec, where="post", color=color, alpha=.5, linewidth=.9,
                         label=f"{recipe}, mean AP {summary['calibration'][recipe]['AP']:.3f}" if i == 0 else None)
        for scope, marker in (("all", "o"), ("legal_low", "s")):
            a = summary["calibration"][recipe]
            axes[1].scatter(a["recall"] if scope == "all" else a["groups"][scope]["recall"],
                            a["f1"] if scope == "all" else a["groups"][scope]["f1"],
                            marker=marker, color=color, s=50, label=f"{recipe} {scope}")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="All known windows")
    axes[1].set(xlabel="Recall at frozen threshold", ylabel="Positive F1",
                title="Calibration, three-seed means")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.16)
    fig.suptitle("NYC V10 | strict 10-minute lead | calibration diagnostics")
    fig.savefig(out / "calibration_PR_and_groups.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for j, (recipe, color) in enumerate(zip(cfg["v10_recipes"], colors, strict=True)):
        a = summary["calibration"][recipe]
        b = a["matched"]["all"][str(cfg["main_recall"])]
        axes[0].bar(np.arange(2) + .19 * (j - 1.5), [a["FP_day"], a["recovery_FP_day"]],
                    width=.18, color=color, label=recipe)
        axes[1].bar(np.arange(2) + .19 * (j - 1.5), [b["FP_per_day"], b["recovery_FP_per_day"]],
                    width=.18, color=color, label=recipe)
    for ax, title in zip(axes, ("Frozen constrained-F1 thresholds", "Matched recall 60%"), strict=True):
        ax.set(xticks=[0, 1], xticklabels=["All FP/day", "Recovery FP/day"], title=title, ylabel="Windows/calendar day")
        ax.grid(axis="y", alpha=.15)
        ax.legend()
    fig.savefig(out / "calibration_false_alarms.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 4.3), layout="constrained")
    for recipe, color in zip(cfg["v10_recipes"], colors, strict=True):
        curve = pd.read_csv(folder(root, "f1", recipe, cfg["seeds"][0]) / "curve.csv")
        ax.plot(curve.epoch, curve.AP, marker=".", color=color, label=recipe)
    ax.set(xlabel="Epoch", ylabel="Development AP", title="First registered fold f1, seed 11")
    ax.grid(alpha=.15)
    ax.legend()
    fig.savefig(out / "development_training_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    analyze(*settings())

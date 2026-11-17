"""Paired, same-support operating and ranking analyses on actual NYC development data."""
from datetime import date
from itertools import product
from pathlib import Path
import statistics as stat
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from v9_check import MODELS, register
from v9_io import V8, load, save, settings
from v9_policy import EPOCH, METRICS, POLICIES, period_for, raw_scores, read_arrays, write_csv

sys.path.insert(0, str(V8 / "v7"))
from v7_train import study_for  # initializes the signed V7 dependency paths
from evaluation import event_metrics, spans

LABELS = {"center": "V7 Center", "attention": "V7 Attention", "rank": "V8 Tail ranking"}
BOOT = ("f1", "recall", "FP_day", "recovery_FP_day", "FP60_day", "recovery_FP60_day")
POINT = ("achieved_recall", "TP", "FP", "FN", "TN", "recovery_FP", "FP_per_day", "recovery_FP_per_day", "F1", "precision")


def summarize(rows, fields):
    seeds = sorted({r["seed"] for r in rows})
    result = dict(runs=len(rows), seeds=len(seeds))
    for field in fields:
        by_seed = [stat.mean([r[field] for r in rows if r["seed"] == s and r[field] is not None]) for s in seeds]
        result[field] = stat.mean(by_seed)
        result[field + "_sd"] = stat.stdev(by_seed) if len(by_seed) > 1 else None
    return result


def trace(keys, y, score, risk, pred, weights, first, rate):
    """Bootstrap frozen operating points and matched ranking using daily count prefixes."""
    n_days = weights.shape[1]
    day = keys // 1000 // 144 - first
    if not ((day >= 0) & (day < n_days)).all():
        raise ValueError("Date weights and support do not align")
    pos = y == 1
    rec = risk & ~pos

    def counts(mask):
        return np.bincount(day[mask], minlength=n_days)

    tp = weights @ counts(pred & pos)
    fp = weights @ counts(pred & ~pos)
    hard = weights @ counts(pred & rec)
    npos = weights @ counts(pos)
    neg = weights @ counts(~pos)
    valid = (npos > 0) & (neg > 0)

    order = np.lexsort((keys, -score))
    ranks = np.flatnonzero(pos[order])
    others = np.flatnonzero(~pos[order])
    loc = np.searchsorted(ranks, others)
    period = day[order[others]]
    rec_others = rec[order[others]]
    all_steps = np.zeros((len(ranks) + 1, n_days), dtype=np.int32)
    rec_steps = np.zeros_like(all_steps)
    np.add.at(all_steps, (loc, period), 1)
    np.add.at(rec_steps, (loc[rec_others], period[rec_others]), 1)
    all_prefix = np.cumsum(all_steps, axis=0)
    rec_prefix = np.cumsum(rec_steps, axis=0)

    cum_positive = np.cumsum(weights[:, day[order[ranks]]], axis=1)
    stop = np.argmax(cum_positive >= np.ceil(rate * npos)[:, None], axis=1)
    matched_fp = np.sum(weights * all_prefix[stop], axis=1)
    matched_rec = np.sum(weights * rec_prefix[stop], axis=1)
    values = np.stack((2 * tp / np.maximum(npos + tp + fp, 1), tp / np.maximum(npos, 1),
                       fp / n_days, hard / n_days, matched_fp / n_days, matched_rec / n_days), axis=1)
    values[~valid] = np.nan
    return values


def variant(raw, decision, policy, keep):
    return (raw["sample_id"][keep], raw["y"][keep], decision[f"score_{policy}"][keep],
            raw["risk"][keep], decision[f"pred_{policy}"][keep])


def error_rows(raw, decision, groups, model, seed, fold, policy):
    keys, y, risk = (raw[k] for k in ("sample_id", "y", "risk"))
    days = len(np.unique(keys[decision["eval_mask"]] // 1000 // 144))
    keep = decision["eval_mask"] & (y >= 0)
    positive = y == 1
    pred = decision[f"pred_{policy}"]
    low, missing = groups["group"], groups["missing"]
    morning = ((keys // 1000) % 144) < 9 * 6
    slices = {"all": np.ones_like(low), "legal_low_t_minus_2": low,
              "legal_known_other_t_minus_2": ~low & ~missing, "legal_missing_t_minus_2": missing,
              "AM": morning, "PM": ~morning}
    for name, mask in slices.items():
        s = keep & mask
        recovery = s & risk & ~positive
        fp = s & pred & ~positive
        yield dict(fold=fold, model=model, seed=seed, policy=policy, slice=name, days=days,
                   known=int(s.sum()), positives=int((s & positive).sum()), recovery_negatives=int(recovery.sum()),
                   tp=int((s & pred & positive).sum()), fn=int((s & ~pred & positive).sum()),
                   fp=int(fp.sum()), recovery_fp=int((fp & risk).sum()), other_fp=int((fp & ~risk).sum()))


def render_figures(root, cfg, summary, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    colors = ("#176650", "#a55b1f", "#5466aa")
    fig, ax = plt.subplots(figsize=(8.2, 5.1), layout="constrained")
    for (model, label), color in zip(LABELS.items(), colors, strict=True):
        avg = summary["calibration"][model]["selected"]["AP"]
        for i, seed in enumerate(cfg["seeds"]):
            keys, y, score, risk, pred = data[model, seed, "selected"]
            precision, recall, _ = precision_recall_curve(y, score)
            ax.step(recall, precision, color=color, linewidth=1, alpha=.55, where="post",
                    label=f"{label} (3 seeds, mean AP={avg:.3f})" if i == 0 else None)
            tp = (pred & (y == 1)).sum()
            ax.scatter(tp / y.sum(), tp / pred.sum(), color=color, s=22)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="NYC calibration | strict 10-minute lead")
    ax.grid(alpha=.18)
    ax.legend(loc="lower left")
    fig.savefig(out / "calibration_PR.png", dpi=160)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    cols = ["FP_day", "recovery_FP_day"]
    anchors = ["All false positives", "Recovery false positives"]
    for j, (model, color) in enumerate(zip(LABELS, colors, strict=True)):
        rates = summary["calibration"][model]["selected"]
        matched = summary["calibration"][model]["selected_points"][str(cfg["main_recall"])]
        left.bar(np.arange(2) + .23 * (j - 1), [rates[x] for x in cols], width=.22, color=color, label=LABELS[model])
        right.bar(np.arange(2) + .23 * (j - 1), [matched["FP_per_day"], matched["recovery_FP_per_day"]],
                  width=.22, color=color, label=LABELS[model])
    for i, cap in enumerate((cfg["FP_budget_day"], cfg["recovery_budget_day"])):
        left.hlines(cap, i - .38, i + .38, linestyle="--", color="#272727", linewidth=1.1)
    for ax, title in ((left, "Frozen thresholds"), (right, "Matched recall at 60%")):
        ax.set(xticks=[0, 1], xticklabels=anchors, ylabel="Windows per calendar day", title=title, ylim=(0, 14))
        ax.grid(axis="y", alpha=.16)
    right.legend(loc="upper right", fontsize=8)
    fig.suptitle("NYC calibration | same known windows, three-seed means")
    fig.savefig(out / "calibration_false_alarms.png", dpi=160)
    plt.close(fig)


def analyze(root, cfg):
    register(root, cfg)
    lock = load(root / "evidence/freeze.json")
    assert lock["holdout_opened_utc"] is None
    result = root / "results"
    oper = pd.read_csv(result / "operating_points.csv", float_precision="round_trip").to_dict("records")
    matched = pd.read_csv(result / "matched_recall.csv", float_precision="round_trip").to_dict("records")
    rows, events, slice_list, support = [], [], [], []
    summary, cal = {}, {}
    weight_data = {}
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    targets = (("center_global", "center", "global"), ("V7_Attention_global", "attention", "global"),
               ("V8_rank_global", "rank", "global"), ("V7_Attention_selected", "attention", "selected"))
    for stage, folds in (("development", list(cfg["folds"])), ("calibration", ["final"])):
        summary[stage] = {}
        blocks = {name: [] for name, _, _ in targets}
        for fold in folds:
            period = period_for(cfg, fold)
            a, b = map(date.fromisoformat, period)
            left = (a - EPOCH).days + (cfg["early_days"] if stage == "development" else 0)
            n_days = (b - EPOCH).days - left
            weights = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=cfg["bootstrap_reps"])
            weight_data[fold] = weights
            traces = {}
            groups = read_arrays(result / "groups" / f"{fold}.npz")
            for model, seed in product(MODELS, cfg["seeds"]):
                raw = raw_scores(root, cfg, fold, model, seed)
                dec = read_arrays(result / "policies" / fold / f"{model}_s{seed}.npz")
                mask = dec["eval_mask"]
                keep = mask & (raw["y"] >= 0)
                for policy in POLICIES:
                    z = variant(raw, dec, policy, keep)
                    traces[model, seed, policy] = (
                        traces[model, seed, "global"] if policy == "selected" and np.array_equal(dec["score_global"], dec["score_selected"])
                        else trace(*z, weights, left, cfg["main_recall"]))
                    events.append(dict(fold=fold, model=model, seed=seed, policy=policy,
                                       **event_metrics(z[0], z[1], z[4])))
                    slice_list.extend(error_rows(raw, dec, groups, model, seed, fold, policy))
                    if stage == "calibration":
                        cal[model, seed, policy] = z
                if model == "center" and seed == cfg["seeds"][0]:
                    for part, flag in (("full", np.ones_like(mask)), ("fit", dec["fit_mask"]), ("assess", mask)):
                        y = raw["y"]
                        pos = flag & (y == 1)
                        known = flag & (y >= 0)
                        support.append(dict(stage=stage, fold=fold, scope=part, days=len(np.unique(raw["sample_id"][flag]//1000//144)),
                                            scheduled=int(flag.sum()), known=int(known.sum()), positives=int(pos.sum()),
                                            negatives=int((flag & (y == 0)).sum()), unknown=int((flag & (y < 0)).sum()),
                                            recovery_negatives=int((flag & raw["risk"] & (y == 0)).sum()),
                                            event_segments=sum(map(len, spans(raw["sample_id"][pos]).values())),
                                            known_roads=len(np.unique(raw["sample_id"][known] % 1000)),
                                            legal_low=int((known & groups["group"]).sum()), legal_missing=int((known & groups["missing"]).sum())))
            for name, model, policy in targets:
                diff = np.mean([traces["center", s, "selected"] - traces[model, s, policy] for s in cfg["seeds"]], axis=0)
                blocks[name].append(diff)
        for model in MODELS:
            summary[stage][model] = {}
            for policy in POLICIES:
                block = [r for r in oper if r["fold"] in folds and r["model"] == model and r["policy"] == policy]
                summary[stage][model][policy] = dict(**summarize(block, METRICS),
                    budget_pass_rows=sum(r["eval_budget_ok"] for r in block), delta=block[0]["delta"])
                summary[stage][model][policy + "_points"] = {
                    str(rate): summarize([r for r in matched if r["fold"] in folds and r["model"] == model
                                         and r["policy"] == policy and r["target_recall"] == rate], POINT)
                    for rate in cfg["recalls"]}
        for name, model, policy in targets:
            diff = np.mean(blocks[name], axis=0)
            left_metrics = summary[stage]["center"]["selected"]
            right_metrics = summary[stage][model][policy]
            rate = str(cfg["main_recall"])
            left_point = summary[stage]["center"]["selected_points"][rate]
            right_point = summary[stage][model][policy + "_points"][rate]
            estimate = {k: left_metrics[k] - right_metrics[k] for k in BOOT[:4]}
            estimate["FP60_day"] = left_point["FP_per_day"] - right_point["FP_per_day"]
            estimate["recovery_FP60_day"] = left_point["recovery_FP_per_day"] - right_point["recovery_FP_per_day"]
            for i, metric in enumerate(BOOT):
                z = diff[:, i]
                valid = z[np.isfinite(z)]
                low, high = np.quantile(valid, [.025, .975])
                rows.append(dict(stage=stage, contrast="center_selected_minus_" + name, metric=metric,
                                 estimate=estimate[metric], lo95=float(low), hi95=float(high), valid_reps=len(valid)))
    write_csv(result / "paired_date_blocks.csv", rows)
    write_csv(result / "error_slices.csv", slice_list)
    write_csv(result / "events.csv", events)
    write_csv(result / "support.csv", support)
    np.savez_compressed(result / "date_bootstrap_weights.npz", **weight_data)
    save(result / "summary.json", summary)
    render_figures(root, cfg, summary, cal)
    print(dict(contrasts=len(rows), slices=len(slice_list), event_rows=len(events), replicas=cfg["bootstrap_reps"]), flush=True)


if __name__ == "__main__":
    analyze(*settings())

"""Summarize completed development comparisons and paired calendar-block uncertainty."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from v4_io import load, save, settings
from v4_matrix import location
from v4_train import read_npz


def cases(root, cfg):
    selected = load(root / "results/selection.json")
    recipe = selected["recipe"]
    return [("B", "mean")] + [(recipe, pool) for pool in cfg["pools"]]


def bootstrap_scores(z, date_weights, first_date, rate):
    # Risk scores are strictly above all ordinary scores, and all positives are risk.
    # Thus full-support AP, FP at a target recall and F1 at the saved diagnostic
    # threshold can be evaluated on risk arrays, while preserving day sampling.
    risk = z["risk"]
    if (np.any((z["y"] == 1) & ~risk)
            or z["score"][risk].min() <= z["score"][~risk].max()
            or np.any(z["pred"] & ~risk)):
        raise ValueError("Invalid risk prefix for efficient bootstrap")
    idx = np.flatnonzero(risk)
    order = idx[np.lexsort((z["sample_id"][idx], -z["score"][idx]))]
    keys, y, score, pred = (z[k][order] for k in ("sample_id", "y", "score", "pred"))
    relative_days = keys // 1000 // 144 - first_date
    if relative_days.min() < 0 or relative_days.max() >= date_weights.shape[1]:
        raise ValueError("Bootstrap calendar differs")
    weights = date_weights[:, relative_days]
    tp = (weights * (y == 1)).cumsum(axis=1)
    fp = (weights * (y == 0)).cumsum(axis=1)
    npos = tp[:, -1]
    limit = np.ceil(rate * npos).astype(int)
    boundary = np.argmax(tp >= limit[:, None], axis=1)
    fp_at_recall = fp[np.arange(len(weights)), boundary] / date_weights.shape[1]
    ends = np.r_[score[1:] != score[:-1], True]
    true, false = tp[:, ends], fp[:, ends]
    denom = true + false
    increments = np.diff(true, axis=1, prepend=0)
    weighted_precision = np.divide(true, denom, out=np.zeros_like(true, dtype=float), where=denom > 0)
    ap = np.divide((weighted_precision * increments).sum(axis=1), npos,
                   out=np.full(len(weights), np.nan), where=npos > 0)
    fixed_tp = (weights * ((y == 1) & pred)).sum(axis=1)
    fixed_total = (weights * pred).sum(axis=1) + npos
    f1 = np.divide(2 * fixed_tp, fixed_total, out=np.full(len(weights), np.nan), where=npos > 0)
    fp_at_recall[npos == 0] = np.nan
    return np.stack((ap, f1, fp_at_recall), axis=1)


def analyze(root, cfg):
    output = root / "results/comparison"
    output.mkdir(parents=True, exist_ok=True)
    groups = cases(root, cfg)
    count = cfg["bootstrap_reps"]
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    metrics = ("AP", "diagnostic_F1", "FP60_per_day")
    points, pairs = [], []
    draws = {f"{r}_{p}": [] for r, p in groups}
    for fold in cfg["folds"]:
        start, stop = map(date.fromisoformat, cfg["fold_limits"][fold])
        first = (start - date(1970, 1, 1)).days
        n_days = (stop - start).days
        weights = rng.multinomial(n_days, np.full(n_days, 1/n_days), size=count)
        for seed in cfg["seeds"]:
            reference = None
            for recipe, pool in groups:
                folder = location(root, fold, recipe, pool, seed)
                name = f"{recipe}_{pool}"
                current = load(folder / "matched_recall.json")
                z = read_npz(folder / "assessment.npz")
                draws[name].append(bootstrap_scores(z, weights, first, cfg["main_recall"]))
                points.extend(dict(fold=fold, seed=seed, model=name, **p) for p in current)
                if reference is None:
                    reference = current
                    continue
                for original, after in zip(reference, current, strict=True):
                    if any(original[k] != after[k] for k in ("TP", "FN", "target_recall", "achieved_recall")):
                        raise ValueError("Unmatched recall or support")
                    pairs.append(dict(fold=fold, seed=seed, model=name, target_recall=after["target_recall"],
                                      achieved_recall=after["achieved_recall"], TP=after["TP"],
                                      baseline_FP=original["FP"], candidate_FP=after["FP"],
                                      delta_FP=after["FP"]-original["FP"],
                                      baseline_recovery_FP=original["recovery_FP"], candidate_recovery_FP=after["recovery_FP"],
                                      delta_FP_per_day=after["FP_per_day"]-original["FP_per_day"],
                                      delta_precision=after["precision"]-original["precision"],
                                      delta_F1=after["F1"]-original["F1"]))
    points = pd.DataFrame(points)
    points.to_csv(output / "matched_points.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_matched_recall.csv", index=False)
    counts = points.groupby(["model", "target_recall"], sort=False).agg(
        mean_FP=("FP", "mean"), mean_recovery_FP=("recovery_FP", "mean"),
        mean_TP=("TP", "mean"), mean_recall=("achieved_recall", "mean"),
        mean_FP_per_day=("FP_per_day", "mean"), mean_precision=("precision", "mean"), mean_F1=("F1", "mean"))
    counts.reset_index().to_csv(output / "matched_aggregates.csv", index=False)
    for row in draws:
        draws[row] = np.mean(draws[row], axis=0)
    boot = []
    for model, scores in draws.items():
        if model == "B_mean":
            continue
        change = scores - draws["B_mean"]
        for i, metric in enumerate(metrics):
            valid = change[np.isfinite(change[:, i]), i]
            low, high = np.percentile(valid, (2.5, 97.5))
            boot.append(dict(model=model, comparison="candidate minus B_mean", metric=metric,
                             lower_95=low, upper_95=high, valid_replicates=len(valid), requested=count))
    pd.DataFrame(boot).to_csv(output / "paired_date_blocks.csv", index=False)
    coverage = []
    for fold in cfg["folds"] + ["final"]:
        payload = load(root / "evidence" / f"fold_{fold}.json")
        for part in payload["parts"]:
            coverage.append({**{k: v for k, v in part.items() if not isinstance(v, (dict, list))}, "fold": fold})
    pd.DataFrame(coverage).to_csv(output / "support.csv", index=False)
    gate = load(root / "evidence/holdout_gate.json")
    if gate["status"] not in ("blocked", "needs_review"):
        raise ValueError("Unregistered formal evaluation")
    save(root / "evidence/status.json", dict(status="development_complete",
        formal_status="blocked", reason=gate["reason"], selected=load(root / "results/selection.json"),
        current_heldout_dates=None, formal_AP=None, formal_F1=None, goal=cfg["goal"], target_reached=None,
        heldout_models_run=0, old_holdout_used=False, development_runs=42, calibration_runs=12,
        development_fixed_recall_only=True))
    print(dict(points=len(points), comparisons=len(pairs), intervals=len(boot),
               formal="blocked", scope="development and calibration"), flush=True)


if __name__ == "__main__":
    analyze(*settings())

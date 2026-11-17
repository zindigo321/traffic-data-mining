"""V9 scoring policy with a separated developer fit and independent calibration."""
import argparse
import csv
from datetime import date
from itertools import product
from pathlib import Path
import statistics as stat
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from v9_check import MODELS, job, register
from v9_io import V8, load, now, save, settings, sha

sys.path.insert(0, str(V8))
from v8_thresholds import budgets, cutoffs, transfer_masks

V4 = V8 / "v7/v6/v5/v4"
sys.path.insert(0, str(V4))
from v4_metrics import matched_points

EPOCH = date(1970, 1, 1)
POLICIES = ("global", "selected")
METRICS = ("AP", "ROC_AUC", "balanced_accuracy", "f1", "recall", "precision", "accuracy",
           "tp", "fp", "fn", "tn", "recovery_FP", "other_FP", "FP_day", "recovery_FP_day")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def period_for(cfg, fold):
    if fold == "final":
        return cfg["calibration"]
    if fold not in cfg["folds"]:
        raise PermissionError("Evaluation data outside development and calibration are forbidden")
    return cfg["folds"][fold]


def authorized(root, cfg, fold):
    period_for(cfg, fold)
    if fold == "final" and not (root / "evidence/development_selection.json").is_file():
        raise PermissionError("Freeze development selection before opening calibration arrays")


def raw_scores(root, cfg, fold, model, seed):
    authorized(root, cfg, fold)
    path = job(root, fold, model, seed) / "assessment.npz"
    signed = load(root / "evidence/inputs.json")["signed_jobs"][f"{fold}/{model}/{seed}"]["files"]
    if sha(path) != signed[path.name]:
        raise ValueError(f"Altered signed predictions: {path}")
    raw = read_arrays(path)
    assert raw["score"].dtype == np.float64 and raw["logit"].dtype == raw["probability"].dtype == np.float32
    assert raw["sample_id"].dtype == np.int64 and raw["y"].dtype == np.int8
    np.testing.assert_array_equal(raw["score"], raw["probability"].astype(np.float64))
    return raw


def group_mask(ratio, limit):
    return np.isfinite(ratio) & (ratio < limit)


def adjust(score, group, delta):
    """Apply a frozen adjustment. The function has no truth or realized risk input."""
    if delta == 0:
        return score
    result = score.copy()
    p = score[group]
    raised = p * np.exp(delta)
    result[group] = raised / (1 - p + raised)
    return result


def legal_groups(root, cfg, fold, raw, alternate=None):
    authorized(root, cfg, fold)
    ref = load(root / "evidence/inputs.json")["prepared"][fold]
    path = (alternate / "prepared" / f"lead_{fold}.npz") if alternate else Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("Unexpected legal input SHA")
    with np.load(path, allow_pickle=False) as arrays:
        phase, minute, roads = (arrays[k] for k in ("split", "minute", "road_ids"))
        train, assess = [phase == k for k in (0, 1)]
        assert (train | assess).all() and minute[train].max() < minute[assess].min()
        assert minute.max() // 1440 < (date.fromisoformat(cfg["embargo"][0]) - EPOCH).days
        assert arrays["eligible"].all()
        keys = (minute[assess, None] // 10 * 1000 + roads[None, :]).ravel()
        np.testing.assert_array_equal(keys, raw["sample_id"])
        for name in ("y", "risk", "current_eligible"):
            np.testing.assert_array_equal(arrays[name][assess].ravel(), raw[name])
        ratio = arrays["ratio_prior"][assess].ravel()
    return dict(sample_id=keys, group=group_mask(ratio, cfg["v9_legal_ratio"]), missing=~np.isfinite(ratio))


def split_masks(raw, cfg, fold):
    early, late, days = transfer_masks(raw["sample_id"], period_for(cfg, fold), cfg["early_days"])
    y = raw["y"]
    calendar = raw["sample_id"] // 1000 // 144
    _, counts = np.unique(calendar, return_counts=True)
    assert len(counts) == days and (counts == 121 * 24).all()
    assert (np.diff(raw["sample_id"]) > 0).all() and np.isin(y, [-1, 0, 1]).all()
    if fold == "final":
        return early | late, early | late, days, days
    return early, late, cfg["early_days"], days - cfg["early_days"]


def operating(y, score, risk, threshold, days, cfg):
    pred = score >= threshold
    positive = y == 1
    tp, fp = int((pred & positive).sum()), int((pred & ~positive).sum())
    fn, tn = int(positive.sum()) - tp, int((~positive).sum()) - fp
    rec_fp = int((pred & ~positive & risk).sum())
    total_cap, rec_cap = budgets(days, cfg)
    return dict(AP=float(average_precision_score(y, score)), ROC_AUC=float(roc_auc_score(y, score)),
                balanced_accuracy=.5 * (tp / (tp + fn) + tn / (tn + fp)),
                f1=2 * tp / (2 * tp + fp + fn), recall=tp / (tp + fn),
                precision=tp / (tp + fp) if tp + fp else None, accuracy=(tp + tn) / len(y),
                tp=tp, fp=fp, fn=fn, tn=tn, recovery_FP=rec_fp, other_FP=fp - rec_fp,
                FP_day=fp / days, recovery_FP_day=rec_fp / days,
                positive=tp + fn, negative=fp + tn, known=len(y), prevalence=(tp + fn) / len(y),
                recovery_negatives=int((risk & ~positive).sum()), FP_cap=total_cap, recovery_cap=rec_cap,
                eval_budget_ok=fp <= total_cap and rec_fp <= rec_cap)


def evaluate_job(raw, group, cfg, fold, model, seed, delta):
    base, assess, fit_days, days = split_masks(raw, cfg, fold)
    y, risk = raw["y"], raw["risk"]
    fit = base & (y >= 0)
    keep = assess & (y >= 0)
    score = adjust(raw["score"], group["group"], delta)
    cut = cutoffs(y[fit], score[fit], risk[fit], fit_days, cfg)["constrained"]
    result = operating(y[keep], score[keep], risk[keep], cut["threshold"], days, cfg)
    row = dict(fold=fold, model=model, seed=seed, delta=delta, days=days, scheduled=int(assess.sum()),
               unknown=int((assess & (y < 0)).sum()), coverage=float(keep.sum() / assess.sum()),
               fit_known=int(fit.sum()), fit_positive=int((fit & (y == 1)).sum()),
               fit_recovery_negatives=int((fit & risk & (y == 0)).sum()),
               legal_low_known=int((keep & group["group"]).sum()),
               legal_low_positive=int((keep & group["group"] & (y == 1)).sum()),
               legal_low_recovery=int((keep & group["group"] & risk & (y == 0)).sum()),
               legal_missing_known=int((keep & group["missing"]).sum()), **cut, **result)
    return row, score, base, assess


def selection(rows, cfg):
    chosen = {}
    grid = cfg["v9_offsets"]
    for model in MODELS:
        relevant = [r for r in rows if r["model"] == model]
        baseline = [r for r in relevant if r["delta"] == 0]
        assert len(baseline) == len(cfg["seeds"]) * len(cfg["folds"])
        f0, r0 = [stat.mean(r[k] for r in baseline) for k in ("f1", "recall")]
        evaluations = []
        for i, delta in enumerate(grid):
            block = [r for r in relevant if r["delta"] == delta]
            assert [(r["fold"], r["seed"]) for r in block] == [(r["fold"], r["seed"]) for r in baseline]
            result = {k: stat.mean(r[k] for r in block) for k in ("f1", "recall", "FP_day", "recovery_FP_day")}
            improved = []
            for seed in cfg["seeds"]:
                new = [r for r in block if r["seed"] == seed]
                orig = [r for r in baseline if r["seed"] == seed]
                if all(stat.mean(r[k] for r in new) > stat.mean(r[k] for r in orig) for k in ("f1", "recall")):
                    improved.append(seed)
            eligible = (result["f1"] > f0 and result["recall"] > r0
                        and result["FP_day"] <= cfg["FP_budget_day"] and result["recovery_FP_day"] <= cfg["recovery_budget_day"]
                        and len(improved) >= cfg["v9_min_seeds_improved"])
            evaluations.append(dict(delta=delta, order=i, **result, improved_seeds=improved, eligible=eligible))
        best = min((r for r in evaluations if r["eligible"]),
                   key=lambda r: (-r["f1"], -r["recall"], r["recovery_FP_day"], abs(r["delta"]), r["order"]),
                   default=None)
        chosen[model] = dict(delta=best["delta"] if best is not None else 0,
                             passed=best is not None, grid=evaluations)
    return chosen


def shared_support(raw, first):
    for k in ("sample_id", "y", "risk", "current_eligible"):
        np.testing.assert_array_equal(raw[k], first[k])


def develop(root, cfg):
    lock = root / "evidence/development_selection.json"
    if lock.exists():
        raise PermissionError("Development selection is already frozen")
    rows = []
    groups = root / "results/groups"
    groups.mkdir(parents=True, exist_ok=True)
    for fold in cfg["folds"]:
        first = raw_scores(root, cfg, fold, "center", cfg["seeds"][0])
        group = legal_groups(root, cfg, fold, first)
        np.savez_compressed(groups / f"{fold}.npz", **group)
        for model, seed in product(MODELS, cfg["seeds"]):
            raw = raw_scores(root, cfg, fold, model, seed)
            shared_support(raw, first)
            for delta in cfg["v9_offsets"]:
                rows.append(evaluate_job(raw, group, cfg, fold, model, seed, delta)[0])
        print(dict(fold=fold, candidate_rows=len(rows)), flush=True)
    results = root / "results/development_grid.csv"
    write_csv(results, rows)
    picked = selection(rows, cfg)
    save(lock, dict(at_utc=now(), chosen=picked, grid_sha256=sha(results), calibration_assessments_parsed=False,
                    primary=cfg["v9_primary"], holdout_read=False))
    print({k: (v["delta"], v["passed"]) for k, v in picked.items()}, flush=True)


def policy_rows(root, cfg, folds, picked):
    rows, points = [], []
    for fold in folds:
        first = raw_scores(root, cfg, fold, "center", cfg["seeds"][0])
        group = (legal_groups(root, cfg, fold, first) if fold == "final"
                 else read_arrays(root / "results/groups" / f"{fold}.npz"))
        if fold == "final":
            np.savez_compressed(root / "results/groups/final.npz", **group)
        for model, seed in product(MODELS, cfg["seeds"]):
            raw = raw_scores(root, cfg, fold, model, seed)
            shared_support(raw, first)
            y = raw["y"]
            stored = dict(sample_id=raw["sample_id"])
            for rule, delta in (("global", 0), ("selected", picked[model]["delta"])):
                row, score, base, assess = evaluate_job(raw, group, cfg, fold, model, seed, delta)
                rows.append(dict(policy=rule, **row))
                known = assess & (y >= 0)
                candidates, _ = matched_points(raw["sample_id"][known], y[known], score[known],
                                                raw["risk"][known], cfg["recalls"], row["days"])
                points.extend(dict(fold=fold, model=model, seed=seed, policy=rule, delta=delta,
                    **p, recovery_FP_per_day=p["recovery_FP"] / row["days"]) for p in candidates)
                stored.update({"fit_mask": base, "eval_mask": assess, f"threshold_{rule}": np.float64(row["threshold"]),
                               f"score_{rule}": score, f"pred_{rule}": score >= row["threshold"]})
            dest = root / "results/policies" / fold / f"{model}_s{seed}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **stored)
    return rows, points


def calibrate(root, cfg):
    if (root / "evidence/freeze.json").exists():
        raise PermissionError("Calibration is already frozen")
    lock_path = root / "evidence/development_selection.json"
    lock = load(lock_path)
    assert not lock["calibration_assessments_parsed"]
    if lock["grid_sha256"] != sha(root / "results/development_grid.csv"):
        raise ValueError("Development table changed")
    start = now()
    rows, points = policy_rows(root, cfg, (*cfg["folds"], "final"), lock["chosen"])
    p = root / "results"
    write_csv(p / "operating_points.csv", rows)
    write_csv(p / "matched_recall.csv", points)
    fit = [r for r in rows if r["fold"] == "final"]
    assert len(fit) == 18 and all(r["eval_budget_ok"] for r in fit)
    final = read_arrays(p / "groups/final.npz")
    frozen = []
    for row in fit:
        fold, model, seed = (row[k] for k in ("fold", "model", "seed"))
        folder = job(root, fold, model, seed)
        frozen.append({k: row[k] for k in ("policy", "model", "seed", "delta", "threshold", "fit_FP", "fit_recovery_FP")}
                      | dict(weights_sha256=sha(folder / "weight.pt"), preprocessor_sha256=sha(folder / "preprocessor.npz"),
                             source_scores_sha256=sha(folder / "assessment.npz"),
                             decisions_sha256=sha(p / "policies/final" / f"{model}_s{seed}.npz")))
    gate = load(root / "evidence/date_gate.json")
    assert not gate["observation_values_requested"] and gate["NYC_status"] == "blocked"
    save(root / "evidence/freeze.json", dict(at_utc=now(), calibration_opened_utc=start,
        primary=cfg["v9_primary"], primary_policy="selected", objective="max_positive_F1_given_FP_caps",
        registered=register(root, cfg), development_selection_sha256=sha(lock_path),
        policy_code_sha256=sha(Path(__file__)), calibration_date=cfg["calibration"], calendar_days=28,
        calendar_FP_caps=dict(total_per_day=cfg["FP_budget_day"], recovery_per_day=cfg["recovery_budget_day"]),
        calibration_key_sha256=__import__("hashlib").sha256(final["sample_id"].astype("<i8").tobytes()).hexdigest(),
        calibration_key_count=len(final["sample_id"]), calibration_known=fit[0]["known"],
        holdout=cfg["holdout"], holdout_opened_utc=None, holdout_key_digest=None, holdout_gate_sha256=sha(root / "evidence/date_gate.json"),
        models=frozen, score_type="float64", decision="adjusted_score >= per_seed_threshold"))
    save(root / "evidence/status.json", dict(status="development_and_calibration_complete", NYC_formal="blocked",
        Chicago_formal="blocked", Geolife_formal="blocked", reason=gate["NYC_reason"], V5_V6_holdout_used=False,
        new_holdout_values_read=False, formal_metrics={k: None for k in METRICS},
        primary=cfg["v9_primary"], primary_delta=lock["chosen"][cfg["v9_primary"]]["delta"], calibrated_jobs=9,
        score_reuse_only=True, goal_status="not_evaluated", recorded_utc=now()))
    print(dict(rows=len(rows), matched=len(points), frozen=len(frozen), calibration_all_budgets=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("develop", "calibrate"))
    args = parser.parse_args()
    root, cfg = settings()
    register(root, cfg)
    {"develop": develop, "calibrate": calibrate}[args.stage](root, cfg)

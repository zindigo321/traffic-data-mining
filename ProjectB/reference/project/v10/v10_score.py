"""Fixed-support V10 selection and constrained-F1 thresholds."""
import argparse
from itertools import product
import statistics as stat
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from v10_check import register
from v10_data import GROUP_NAMES, read_npz
from v10_io import CODE, V9, load, now, save, settings, sha
from v10_train import folder

sys.path.insert(0, str(V9))
from v9_policy import write_csv
from v8_thresholds import cutoffs, transfer_masks
from v4_metrics import matched_points

METRICS = ("AP", "balanced_accuracy", "f1", "recall", "precision", "ROC_AUC", "accuracy",
           "tp", "fp", "fn", "tn", "FP_day", "recovery_FP_day", "recovery_FP")


def raw_for(root, cfg, fold, recipe, seed):
    if fold not in (*cfg["folds"], "final"):
        raise PermissionError("Unregistered evaluation period")
    if fold == "final" and not (root / "results/development_selection.json").exists():
        raise PermissionError("Freeze selection before parsing calibration scores")
    raw = read_npz(folder(root, fold, recipe, seed) / "assessment.npz")
    assert raw["score"].dtype == np.float64 and raw["logit"].dtype == raw["probability"].dtype == np.float32
    np.testing.assert_array_equal(raw["score"], raw["probability"].astype(float))
    assert np.isin(raw["y"], [-1, 0, 1]).all() and np.isfinite(raw["score"]).all()
    assert (np.diff(raw["sample_id"]) > 0).all()
    return raw


def same_support(raw, ref):
    for k in ("sample_id", "y", "risk", "current_eligible", "legal_group"):
        np.testing.assert_array_equal(raw[k], ref[k])


def split(raw, cfg, fold):
    interval = cfg["calibration"] if fold == "final" else cfg["folds"][fold]
    early, late, days = transfer_masks(raw["sample_id"], interval, cfg["early_days"])
    assert len(np.unique(raw["sample_id"] // 1000 // 144)) == days
    if fold == "final":
        return early | late, early | late, days, days
    return early, late, cfg["early_days"], days - cfg["early_days"]


def measure(y, score, risk, threshold, days):
    positive = y == 1
    pred = score >= threshold
    tp, fp = int((positive & pred).sum()), int((~positive & pred).sum())
    fn, tn = int((positive & ~pred).sum()), int((~positive & ~pred).sum())
    pos, neg = tp + fn, fp + tn
    hard = int((~positive & pred & risk).sum())
    both = pos > 0 and neg > 0
    return dict(known=len(y), positive=pos, negative=neg, prevalence=pos / len(y) if len(y) else None,
                AP=float(average_precision_score(y, score)) if both else None,
                ROC_AUC=float(roc_auc_score(y, score)) if both else None,
                balanced_accuracy=.5 * (tp / pos + tn / neg) if both else None,
                f1=2 * tp / (pos + tp + fp) if pos else None, recall=tp / pos if pos else None,
                precision=tp / (tp + fp) if tp + fp else None,
                accuracy=(tp + tn) / len(y) if len(y) else None, tp=tp, fp=fp, fn=fn, tn=tn,
                recovery_FP=hard, recovery_negatives=int((~positive & risk).sum()),
                FP_day=fp / days, recovery_FP_day=hard / days)


def phase_rows(root, cfg, folds):
    rows, points, group_rows = [], [], []
    for fold in folds:
        first = raw_for(root, cfg, fold, "B0", cfg["seeds"][0])
        for recipe, seed in product(cfg["v10_recipes"], cfg["seeds"]):
            raw = raw_for(root, cfg, fold, recipe, seed)
            same_support(raw, first)
            before, after, fit_days, days = split(raw, cfg, fold)
            y, score, risk = (raw[k] for k in ("y", "score", "risk"))
            fit, keep = before & (y >= 0), after & (y >= 0)
            cut = cutoffs(y[fit], score[fit], risk[fit], fit_days, cfg)["constrained"]
            threshold = cut["threshold"]
            row = dict(fold=fold, recipe=recipe, seed=seed, days=days, scheduled=int(after.sum()),
                       unknown=int((after & (y < 0)).sum()), coverage=float(keep.sum() / after.sum()),
                       fit_known=int(fit.sum()), fit_positive=int((fit & (y == 1)).sum()), **cut,
                       **measure(y[keep], score[keep], risk[keep], threshold, days))
            row.update(FP_cap=days * cfg["FP_budget_day"], recovery_cap=days * cfg["recovery_budget_day"])
            row["eval_budget_ok"] = row["fp"] <= row["FP_cap"] and row["recovery_FP"] <= row["recovery_cap"]
            rows.append(row)
            decisions = dict(sample_id=raw["sample_id"], fit_mask=before, eval_mask=after,
                             threshold=np.float64(threshold), pred=score >= threshold)
            dest = root / "results/decisions" / fold / f"{recipe}_s{seed}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **decisions)
            scopes = [("all", np.ones(len(y), dtype=bool))]
            scopes.extend((name, raw["legal_group"] == i) for i, name in enumerate(GROUP_NAMES))
            for scope, group in scopes:
                known = keep & group
                metric = measure(y[known], score[known], risk[known], threshold, days)
                group_rows.append(dict(fold=fold, recipe=recipe, seed=seed, scope=scope, days=days,
                                  scheduled=int((after & group).sum()), unknown=int((after & group & (y < 0)).sum()), **metric))
                if scope in ("all", "legal_low") and metric["positive"] and metric["negative"]:
                    matched, _ = matched_points(raw["sample_id"][known], y[known], score[known],
                                                risk[known], cfg["recalls"], days)
                    points.extend(dict(fold=fold, recipe=recipe, seed=seed, scope=scope, **p,
                                       recovery_FP_per_day=p["recovery_FP"] / days) for p in matched)
    return rows, points, group_rows


def average(rows, metric):
    seeds = sorted({r["seed"] for r in rows})
    per_seed = []
    for seed in seeds:
        values = [r[metric] for r in rows if r["seed"] == seed and r[metric] is not None]
        if values:
            per_seed.append(stat.mean(values))
    return stat.mean(per_seed) if per_seed else None


def select_recipe(rows, points, group_rows, cfg):
    def r(recipe):
        return [x for x in rows if x["recipe"] == recipe]

    def p(recipe):
        return [x for x in points if x["recipe"] == recipe and x["scope"] == "all"
                and x["target_recall"] == cfg["main_recall"]]

    def g(recipe, group):
        return [x for x in group_rows if x["recipe"] == recipe and x["scope"] == group]

    base = r("B0")
    base_points = p("B0")
    candidates = []
    for order, recipe in enumerate(cfg["v10_recipes"][1:]):
        block = r(recipe)
        match = p(recipe)
        metrics = {k: average(block, k) for k in ("AP", "f1", "recall", "FP_day", "recovery_FP_day")}
        low_AP = average(g(recipe, "legal_low"), "AP")
        recovery = average(match, "recovery_FP_per_day")
        improved = [s for s in cfg["seeds"]
                    if average([x for x in match if x["seed"] == s], "recovery_FP_per_day")
                    < average([x for x in base_points if x["seed"] == s], "recovery_FP_per_day")]
        tests = dict(AP_guard=metrics["AP"] >= average(base, "AP") - cfg["v10_max_AP_drop"],
                     F1_guard=metrics["f1"] >= average(base, "f1"),
                     recall_guard=metrics["recall"] >= average(base, "recall"),
                     any_gain=metrics["f1"] > average(base, "f1") or metrics["recall"] > average(base, "recall"),
                     low_AP_gain=low_AP > average(g("B0", "legal_low"), "AP"),
                     matched_recovery_gain=recovery < average(base_points, "recovery_FP_per_day"),
                     seeds_guard=len(improved) >= cfg["v10_min_seeds_recovery_improved"],
                     FP_cap=metrics["FP_day"] <= cfg["FP_budget_day"],
                     recovery_cap=metrics["recovery_FP_day"] <= cfg["recovery_budget_day"])
        for group in ("legal_not_low", "legal_missing"):
            tests[group + "_guard"] = (average(g(recipe, group), "recall") >=
                                      average(g("B0", group), "recall") - cfg["v10_max_group_recall_drop"])
        candidates.append(dict(recipe=recipe, order=order, **metrics, low_AP=low_AP,
                               matched_recovery_FP_day=recovery, improved_seeds=improved, checks=tests, eligible=all(tests.values())))
    best = min((r for r in candidates if r["eligible"]),
               key=lambda r: (-r["f1"], -r["recall"], r["matched_recovery_FP_day"], -r["AP"], r["order"]), default=None)
    return dict(primary=best["recipe"] if best else "B0", candidates=candidates,
                fallback=best is None, baseline="B0")


def development(root, cfg):
    path = root / "results/development_selection.json"
    if path.exists():
        raise PermissionError("V10 recipe selection is already frozen")
    rows, points, group_rows = phase_rows(root, cfg, cfg["folds"])
    files = {"development_operating.csv": rows, "development_matched.csv": points, "development_groups.csv": group_rows}
    for name, records in files.items():
        write_csv(root / "results" / name, records)
    decision = select_recipe(rows, points, group_rows, cfg)
    training = load(root / "results/development_training.json")
    save(path, dict(**decision, at_utc=now(), epochs=training["epochs"],
         calibration_scores_parsed=False, new_holdout_read=False,
         source_hashes={name: sha(root / "results" / name) for name in files},
         code={name: sha(CODE / name) for name in ("v10_data.py", "v10_train.py", "v10_score.py")}))
    print(decision, flush=True)


def calibration(root, cfg):
    if (root / "evidence/freeze.json").exists():
        raise PermissionError("V10 thresholds are already frozen")
    pick = load(root / "results/development_selection.json")
    for name, digest in pick["code"].items():
        assert sha(CODE / name) == digest
    opened = now()
    rows, points, group_rows = phase_rows(root, cfg, ["final"])
    for name, records in {"calibration_operating.csv": rows, "calibration_matched.csv": points,
                          "calibration_groups.csv": group_rows}.items():
        write_csv(root / "results" / name, records)
    assert len(rows) == 12 and all(r["eval_budget_ok"] for r in rows)
    signed = []
    for row in rows:
        job = folder(root, "final", row["recipe"], row["seed"])
        signed.append(dict(recipe=row["recipe"], seed=row["seed"], threshold=row["threshold"],
                           fit_FP=row["fit_FP"], fit_recovery_FP=row["fit_recovery_FP"],
                           hashes={p.name: sha(p) for p in sorted(job.iterdir()) if p.is_file()},
                           decisions_sha256=sha(root / "results/decisions/final" / f"{row['recipe']}_s{row['seed']}.npz")))
    gate = load(root / "evidence/date_gate.json")
    assert gate["NYC_status"] == "blocked" and not gate["observation_values_requested"]
    keys = raw_for(root, cfg, "final", "B0", cfg["seeds"][0])["sample_id"]
    from prepare_chicago import fingerprint_keys
    freeze = dict(at_utc=now(), calibration_scores_opened_utc=opened, primary=pick["primary"], threshold_rule="constrained_F1",
                  preregistered=register(root, cfg), selection_sha256=sha(root / "results/development_selection.json"),
                  calibration=cfg["calibration"], calibration_known=rows[0]["known"], calibration_windows=len(keys),
                  calibration_key_sha256=fingerprint_keys(keys), holdout=cfg["holdout"], holdout_read=False,
                  holdout_opened_utc=None, holdout_key_digest=None, gate_sha256=sha(root / "evidence/date_gate.json"),
                  models=signed, code=pick["code"], score_dtype="float64", probability_and_logit_dtype="float32")
    save(root / "evidence/freeze.json", freeze)
    save(root / "evidence/freeze_sha256.json", dict(sha256=sha(root / "evidence/freeze.json"), at_utc=freeze["at_utc"]))
    save(root / "evidence/status.json", dict(status="development_calibration_and_freeze_complete", NYC_formal="blocked",
         Chicago_formal="blocked", Geolife_formal="blocked", reason=gate["NYC_reason"],
         new_holdout_values_read=False, V5_V6_holdout_used=False, primary=pick["primary"],
         formal_metrics={k: None for k in METRICS}, newly_trained_models=48, recorded_utc=now()))
    print(dict(primary=pick["primary"], frozen_models=len(signed), calibration_all_budgets=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "calibration"))
    args = parser.parse_args()
    root, cfg = settings()
    register(root, cfg)
    {"development": development, "calibration": calibration}[args.stage](root, cfg)

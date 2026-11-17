"""Independently verify cutoffs, indicators, and signed score replay."""
import argparse
from datetime import datetime
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from v9_check import MODELS, SOURCE_EVIDENCE, job, register, snapshot
from v9_io import V8, load, now, save, settings, sha
from v9_policy import (POLICIES, adjust, legal_groups, period_for, raw_scores, read_arrays,
                       selection, shared_support, split_masks)
from v8_train import arrays, sample_training, study_for
from v7_verify import check_support, model_replay, verify_point
from v1.graph import TrainPreprocessor
from prepare_chicago import fingerprint_keys


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def independent_cut(y, score, risk, total, recovery):
    values, inv = np.unique(score, return_inverse=True)
    positives = np.bincount(inv[y == 1], minlength=len(values))[::-1].cumsum()
    negatives = np.bincount(inv[y == 0], minlength=len(values))[::-1].cumsum()
    recovered = np.bincount(inv[(y == 0) & risk], minlength=len(values))[::-1].cumsum()
    ok = (negatives <= total) & (recovered <= recovery)
    candidates = np.r_[np.nextafter(values[-1], np.inf), values[::-1][ok]]
    f1 = np.r_[0, 2 * positives[ok] / ((y == 1).sum() + positives[ok] + negatives[ok])]
    return candidates[f1.argmax()]


def verify_stats(y, score, risk, threshold, row):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recovered = int((pred & (y == 0) & risk).sum())
    actual = dict(tn=tn, fp=fp, fn=fn, tp=tp, recovery_FP=recovered, other_FP=fp-recovered, known=len(y),
                  positive=int(y.sum()), negative=int((y == 0).sum()), AP=average_precision_score(y, score),
                  ROC_AUC=roc_auc_score(y, score), balanced_accuracy=balanced_accuracy_score(y, pred),
                  f1=f1_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                  precision=precision_score(y, pred, zero_division=np.nan), accuracy=accuracy_score(y, pred),
                  FP_day=fp / row.days, recovery_FP_day=recovered / row.days)
    for k, v in actual.items():
        same(row[k], v)
    assert row.eval_budget_ok == (fp <= row.FP_cap and recovered <= row.recovery_cap)


def rows_for(table, fold, model, seed):
    return table[(table.fold == fold) & (table.model == model) & (table.seed == seed)]


def verify_policy(row, raw, group, cfg, masks):
    fit, assess, fdays, days = masks
    y, risk = raw["y"], raw["risk"]
    score = adjust(raw["score"], group["group"], row.delta)
    fit = fit & (y >= 0)
    known = assess & (y >= 0)
    caps = fdays * cfg["FP_budget_day"], fdays * cfg["recovery_budget_day"]
    cut = independent_cut(y[fit], score[fit], risk[fit], *caps)
    assert row.threshold == cut and row.fit_days == fdays and row.days == days
    assert row.scheduled == assess.sum() and row.unknown == (assess & (y < 0)).sum()
    assert row.fit_known == fit.sum() and row.fit_positive == (fit & (y == 1)).sum()
    fit_pred = fit & (score >= cut)
    np.testing.assert_array_equal((row.fit_TP, row.fit_FP, row.fit_recovery_FP),
                                 ((fit_pred & (y == 1)).sum(), (fit_pred & (y == 0)).sum(),
                                  (fit_pred & (y == 0) & risk).sum()))
    assert row.fit_FP <= caps[0] and row.fit_recovery_FP <= caps[1]
    verify_stats(y[known], score[known], risk[known], cut, row)
    return score, known


def verify_lock(root, cfg):
    lock = load(root / "evidence/freeze.json")
    selection_path = root / "evidence/development_selection.json"
    picked = load(selection_path)
    gate_path = root / "evidence/date_gate.json"
    gate = load(gate_path)
    st = load(root / "evidence/status.json")
    assert lock["registered"] == register(root, cfg)
    assert sha(selection_path) == lock["development_selection_sha256"]
    assert sha(root / "results/development_grid.csv") == picked["grid_sha256"]
    assert sha(root / "project/v9_policy.py") == lock["policy_code_sha256"]
    assert sha(gate_path) == lock["holdout_gate_sha256"] and not gate["observation_values_requested"]
    stamps = [lock["registered"]["at_utc"], picked["at_utc"], lock["calibration_opened_utc"], lock["at_utc"]]
    order = list(map(datetime.fromisoformat, stamps))
    assert order == sorted(set(order))
    assert picked["primary"] == lock["primary"] == cfg["v9_primary"]
    assert picked["calibration_assessments_parsed"] is False and picked["holdout_read"] is False
    assert st["NYC_formal"] == st["Chicago_formal"] == st["Geolife_formal"] == "blocked"
    assert not st["new_holdout_values_read"] and not st["V5_V6_holdout_used"]
    assert all(v is None for v in st["formal_metrics"].values())
    assert lock["holdout"] == cfg["holdout"] and lock["holdout_opened_utc"] is None and lock["holdout_key_digest"] is None
    assert not (root / "results/holdout").exists() and not (root / "evidence/unlock.json").exists()
    return lock, picked


def verify(root, out, cfg, replay):
    lock, picked = verify_lock(root, cfg)
    src = load(root / "evidence/inputs.json")
    assert src["v8_code"] == snapshot(root / "project/v8")
    assert src["source_evidence"] == {f: sha(root / "evidence" / f) for f in SOURCE_EVIDENCE}
    p = root / "results"
    grid = pd.read_csv(p / "development_grid.csv", float_precision="round_trip")
    op = pd.read_csv(p / "operating_points.csv", float_precision="round_trip")
    points = pd.read_csv(p / "matched_recall.csv", float_precision="round_trip")
    assert selection(grid.to_dict("records"), cfg) == picked["chosen"]
    errs = []
    n_grid = n_oper = n_points = 0
    for fold in (*cfg["folds"], "final"):
        period = period_for(cfg, fold)
        group = read_arrays(p / "groups" / f"{fold}.npz")
        assert group["group"].dtype == group["missing"].dtype == np.bool_ and not (group["group"] & group["missing"]).any()
        first = raw_scores(root, cfg, fold, "center", cfg["seeds"][0])
        np.testing.assert_array_equal(group["sample_id"], first["sample_id"])
        check_support(first["sample_id"], [period], [cfg["embargo"], cfg["holdout"]])
        keep = first["y"] >= 0
        proof = load(root / "evidence" / f"fold_lead_{fold}.json")["parts"][1]
        assert fingerprint_keys(first["sample_id"][keep]) == proof["known_keys_sha256"]
        assert keep.sum() == proof["positive"] + proof["negative"]
        assert len(keep) == proof["expected"]
        if fold == "final":
            assert sha256(group["sample_id"].astype("<i8").tobytes()).hexdigest() == lock["calibration_key_sha256"]
            assert len(keep) == lock["calibration_key_count"] and keep.sum() == lock["calibration_known"]
        if replay:
            regenerated = legal_groups(root, cfg, fold, first, alternate=replay)
            for name in group:
                np.testing.assert_array_equal(group[name], regenerated[name])
            study = study_for(arrays(root, cfg, fold, replay), "X1")
            prep = job(root, fold, "center", cfg["seeds"][0]) / "preprocessor.npz"
            study.configure(TrainPreprocessor.load(prep))
            ids, _ = sample_training(study, cfg)
        expected_fit = read_arrays(job(root, fold, "center", cfg["seeds"][0]) / "sampling_ids.npz")
        expected_prep = sha(job(root, fold, "center", cfg["seeds"][0]) / "preprocessor.npz")
        for model, seed in product(MODELS, cfg["seeds"]):
            folder = job(root, fold, model, seed)
            assert snapshot(folder) == src["signed_jobs"][f"{fold}/{model}/{seed}"]["files"]
            assert sha(folder / "preprocessor.npz") == expected_prep
            sample = read_arrays(folder / "sampling_ids.npz")
            for name in sample:
                np.testing.assert_array_equal(sample[name], expected_fit[name])
            check_support(sample["sample_id"], [[cfg["train_start"], period[0]]], [cfg["embargo"], cfg["holdout"]])
            assert not np.intersect1d(sample["sample_id"], first["sample_id"]).size
            raw = raw_scores(root, cfg, fold, model, seed)
            shared_support(raw, first)
            if replay:
                np.testing.assert_array_equal(sample["sample_id"], study.keys(ids))
                model_name = "lead_center" if model == "center" else "lead_attention"
                errs.append(model_replay(study, raw, model_name, seed, folder, cfg))
            masks = split_masks(raw, cfg, fold)
            grid_job = rows_for(grid, fold, model, seed)
            assert list(grid_job.delta) == ([] if fold == "final" else cfg["v9_offsets"])
            for _, row in grid_job.iterrows():
                verify_policy(row, raw, group, cfg, masks)
                n_grid += 1
            saved = p / "policies" / fold / f"{model}_s{seed}.npz"
            dec = read_arrays(saved)
            np.testing.assert_array_equal(dec["sample_id"], raw["sample_id"])
            np.testing.assert_array_equal(dec["fit_mask"], masks[0])
            np.testing.assert_array_equal(dec["eval_mask"], masks[1])
            pair = rows_for(op, fold, model, seed)
            assert list(pair.policy) == list(POLICIES)
            for _, row in pair.iterrows():
                rule = row.policy
                assert row.delta == (0 if rule == "global" else picked["chosen"][model]["delta"])
                score, keep = verify_policy(row, raw, group, cfg, masks)
                assert dec[f"score_{rule}"].dtype == np.float64 and dec[f"threshold_{rule}"].dtype == np.float64
                assert row.threshold == dec[f"threshold_{rule}"]
                np.testing.assert_array_equal(dec[f"score_{rule}"], score)
                np.testing.assert_array_equal(dec[f"pred_{rule}"], score >= row.threshold)
                z = dict(sample_id=raw["sample_id"][keep], y=raw["y"][keep], risk=raw["risk"][keep], score=score[keep])
                matched = rows_for(points, fold, model, seed)
                subset = matched[matched.policy == rule]
                assert list(subset.target_recall) == cfg["recalls"]
                for point in subset.to_dict("records"):
                    verify_point(z, point, cfg, row.days)
                    same(point["recovery_FP"] / row.days, point["recovery_FP_per_day"])
                    n_points += 1
                if fold == "final":
                    single, = [item for item in lock["models"] if (item["model"], item["seed"], item["policy"]) == (model, seed, rule)]
                    assert all(row[k] == single[k] for k in ("threshold", "delta", "fit_FP", "fit_recovery_FP"))
                    assert single["weights_sha256"] == sha(folder / "weight.pt")
                    assert single["source_scores_sha256"] == sha(folder / "assessment.npz")
                    assert single["preprocessor_sha256"] == expected_prep
                    assert single["decisions_sha256"] == sha(saved)
                    assert row.eval_budget_ok
                n_oper += 1
    assert len(grid) == n_grid == 297 and len(op) == n_oper == 72 and len(points) == n_points == 288
    pairs = pd.read_csv(p / "paired_date_blocks.csv", float_precision="round_trip")
    assert len(pairs) == 48 and (pairs.valid_reps == cfg["bootstrap_reps"]).all()
    assert len(errs) == (36 if replay else 0)
    result = dict(status="PASS", checked_utc=now(), grid_recomputed=n_grid, frozen_and_transfer_recomputed=n_oper,
                  matched_recall_recomputed=n_points, models_replayed=len(errs), max_score_difference=max(errs, default=None),
                  same_support=True, same_training_samples_and_preprocessor=True, past_only_group_rebuilt=bool(replay),
                  threshold_freeze_and_chronology=True, V5_V6_or_new_holdout_used=False, NYC_formal="blocked")
    save(out / "evidence" / ("verification_replay.json" if replay else "verification_offline.json"), result)
    print(result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path)
    ap.add_argument("--replay-root", type=Path)
    args = ap.parse_args()
    output, cfg = settings()
    verify((args.artifacts or output).resolve(), output, cfg, args.replay_root)

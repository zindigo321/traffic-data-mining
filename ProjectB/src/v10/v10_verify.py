"""Independent metrics, sample provenance, and safe model replay for V10."""
import argparse
from datetime import date, datetime
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from v10_check import EVIDENCE, original, register, snapshot
from v10_data import GROUP_NAMES, TrainPreprocessor, data_for, groups, read_npz, study_for, training_sample
from v10_io import CODE, V9, load, now, save, settings, sha
from v10_score import raw_for, same_support, select_recipe, split
from v10_train import folder

sys.path.insert(0, str(V9))
from v9_verify import independent_cut
from v7_verify import check_support, model_replay, verify_point
from prepare_chicago import fingerprint_keys


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)


def records(frame):
    return [{k: None if pd.isna(v) else v for k, v in row.items()} for row in frame.to_dict("records")]


def subset(table, fold, recipe, seed):
    return table[(table.fold == fold) & (table.recipe == recipe) & (table.seed == seed)]


def inspect(y, score, risk, threshold, row):
    pred = score >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = bool(y.sum() and (y == 0).sum())
    pairs = dict(tn=tn, fp=fp, fn=fn, tp=tp, known=len(y), positive=int(y.sum()), negative=int((y == 0).sum()),
                 recovery_FP=int((pred & (y == 0) & risk).sum()), FP_day=fp / row.days,
                 recall=recall_score(y, pred, zero_division=np.nan),
                 precision=precision_score(y, pred, zero_division=np.nan),
                 f1=f1_score(y, pred, zero_division=np.nan) if y.sum() else np.nan,
                 AP=average_precision_score(y, score) if both else np.nan,
                 ROC_AUC=roc_auc_score(y, score) if both else np.nan,
                 balanced_accuracy=balanced_accuracy_score(y, pred) if both else np.nan)
    for k, value in pairs.items():
        close(row[k], value)
    close(row.recovery_FP_day, pairs["recovery_FP"] / row.days)


def verify_lock(root, cfg):
    register(root, cfg)
    freeze = load(root / "evidence/freeze.json")
    assert load(root / "evidence/freeze_sha256.json")["sha256"] == sha(root / "evidence/freeze.json")
    pick = load(root / "results/development_selection.json")
    assert sha(root / "results/development_selection.json") == freeze["selection_sha256"]
    assert pick["code"] == freeze["code"]
    assert all(sha(CODE / k) == v for k, v in pick["code"].items())
    assert all(sha(root / "results" / k) == v for k, v in pick["source_hashes"].items())
    gate = load(root / "evidence/date_gate.json")
    assert sha(root / "evidence/date_gate.json") == freeze["gate_sha256"]
    assert not gate["observation_values_requested"] and gate["NYC_status"] == "blocked"
    assert freeze["holdout"] == cfg["holdout"] and not freeze["holdout_read"]
    assert freeze["holdout_opened_utc"] is None and freeze["holdout_key_digest"] is None
    assert not pick["calibration_scores_parsed"] and not pick["new_holdout_read"]
    status = load(root / "evidence/status.json")
    assert status["NYC_formal"] == status["Chicago_formal"] == status["Geolife_formal"] == "blocked"
    assert not status["new_holdout_values_read"] and not status["V5_V6_holdout_used"]
    assert all(v is None for v in status["formal_metrics"].values())
    assert not (root / "results/holdout").exists()
    stamps = [freeze["preregistered"]["at_utc"], load(root / "results/development_training.json")["at_utc"],
              pick["at_utc"], load(root / "results/calibration_training.json")["at_utc"],
              freeze["calibration_scores_opened_utc"], freeze["at_utc"]]
    moments = list(map(datetime.fromisoformat, stamps))
    assert moments == sorted(set(moments))
    return freeze, pick


def verify(root, output, cfg, replay):
    freeze, pick = verify_lock(root, cfg)
    inputs = load(root / "evidence/inputs.json")
    assert snapshot(CODE / "v9") == inputs["embedded_v9"]
    assert {f: sha(root / "evidence" / f) for f in EVIDENCE} == inputs["source_evidence"]
    res = root / "results"
    read = lambda name: pd.read_csv(res / name, float_precision="round_trip")
    ops, matched, group_rows = [read(n) for n in ("operating_points.csv", "matched_recall.csv", "groups.csv")]
    recomputed = select_recipe(records(read("development_operating.csv")), records(read("development_matched.csv")),
                               records(read("development_groups.csv")), cfg)
    assert recomputed == {k: pick[k] for k in recomputed}
    counts = dict(operating_points=0, groups=0, matched=0, models=0)
    errors = []
    for fold in (*cfg["folds"], "final"):
        interval = cfg["calibration"] if fold == "final" else cfg["folds"][fold]
        first = raw_for(root, cfg, fold, "B0", cfg["seeds"][0])
        check_support(first["sample_id"], [interval], [cfg["embargo"], cfg["holdout"]])
        proof = load(root / "evidence" / f"fold_lead_{fold}.json")["parts"][1]
        assert fingerprint_keys(first["sample_id"][first["y"] >= 0]) == proof["known_keys_sha256"]
        assert len(first["y"]) == proof["expected"] and (first["y"] < 0).sum() == proof["unknown_eligible"]
        if replay:
            data = data_for(root, cfg, fold, alternate=replay)
            study = study_for(data, "X1")
            prep = TrainPreprocessor.load(original(root, fold, cfg["seeds"][0]) / "preprocessor.npz")
            study.configure(prep)
            ids = np.flatnonzero(study.phase == 1)
            np.testing.assert_array_equal(first["sample_id"], study.keys(ids))
            np.testing.assert_array_equal(first["legal_group"], groups(study.ratio[ids], cfg["ratio"]))
        expected_steps = None
        for recipe, seed in product(cfg["v10_recipes"], cfg["seeds"]):
            path = folder(root, fold, recipe, seed)
            raw = raw_for(root, cfg, fold, recipe, seed)
            same_support(raw, first)
            assert raw["legal_group"].dtype == np.int8 and raw["sample_id"].dtype == np.int64 and raw["y"].dtype == np.int8
            before, after, fit_days, days = split(raw, cfg, fold)
            known_fit = before & (raw["y"] >= 0)
            keep = after & (raw["y"] >= 0)
            threshold = independent_cut(raw["y"][known_fit], raw["score"][known_fit], raw["risk"][known_fit],
                                        fit_days * cfg["FP_budget_day"], fit_days * cfg["recovery_budget_day"])
            dec = read_npz(root / "results/decisions" / fold / f"{recipe}_s{seed}.npz")
            assert float(dec["threshold"]) == threshold and dec["threshold"].dtype == np.float64
            np.testing.assert_array_equal(dec["sample_id"], raw["sample_id"])
            np.testing.assert_array_equal(dec["pred"], raw["score"] >= threshold)
            np.testing.assert_array_equal(dec["fit_mask"], before)
            np.testing.assert_array_equal(dec["eval_mask"], after)
            row, = [r for _, r in subset(ops, fold, recipe, seed).iterrows()]
            assert row.threshold == threshold and row.scheduled == after.sum() and row.unknown == (after & (raw["y"] < 0)).sum()
            inspect(raw["y"][keep], raw["score"][keep], raw["risk"][keep], threshold, row)
            assert row.eval_budget_ok == (row.fp <= row.FP_cap and row.recovery_FP <= row.recovery_cap)
            fit_pred = known_fit & dec["pred"]
            assert (fit_pred & (raw["y"] == 0)).sum() == row.fit_FP <= fit_days * cfg["FP_budget_day"]
            assert (fit_pred & (raw["y"] == 0) & raw["risk"]).sum() == row.fit_recovery_FP <= fit_days * cfg["recovery_budget_day"]
            scopes = [("all", np.ones(len(keep), dtype=bool)), *((name, raw["legal_group"] == i) for i, name in enumerate(GROUP_NAMES))]
            for scope, group in scopes:
                selected = keep & group
                gr = subset(group_rows, fold, recipe, seed)
                single = gr[gr.scope == scope].iloc[0]
                assert single.scheduled == (after & group).sum()
                inspect(raw["y"][selected], raw["score"][selected], raw["risk"][selected], threshold, single)
                counts["groups"] += 1
                points = subset(matched, fold, recipe, seed)
                points = points[points.scope == scope]
                assert len(points) == (4 if scope in ("all", "legal_low") else 0)
                z = {k: raw[k][selected] for k in ("sample_id", "y", "score", "risk")}
                for point in points.to_dict("records"):
                    verify_point(z, point, cfg, days)
                    close(point["recovery_FP_per_day"], point["recovery_FP"] / days)
                    counts["matched"] += 1
            state = load(path / "status.json")
            assert state["parameters"] == 22594 and not state["loss_uses_realized_risk"]
            assert state["weight_sha256"] == sha(path / "weight.pt")
            expected_steps = expected_steps or state["updates"]
            assert state["updates"] == expected_steps
            assert sha(path / "preprocessor.npz") == sha(original(root, fold, seed) / "preprocessor.npz")
            sample = read_npz(path / "sampling_ids.npz")
            check_support(sample["sample_id"], [[cfg["train_start"], interval[0]]], [cfg["embargo"], cfg["holdout"]])
            assert not np.intersect1d(sample["sample_id"], raw["sample_id"]).size
            assert np.isin(sample["y"], [0, 1]).all()
            weights = read_npz(path / "group_weights.npz")
            actual_counts = np.bincount(sample["legal_group"] * 2 + sample["y"], minlength=6).reshape(3, 2)
            np.testing.assert_array_equal(weights["counts"], actual_counts)
            assert weights["weights"].dtype == np.float32
            np.testing.assert_allclose(weights["weights"][sample["legal_group"], sample["y"]].mean(), 1., atol=1e-6, rtol=0)
            if recipe == "B0":
                ref = original(root, fold, seed)
                assert snapshot(ref) == inputs["reference"][f"{fold}/{seed}"]
                old = read_npz(ref / "assessment.npz")
                np.testing.assert_array_equal(raw["score"], old["score"])
                assert state["baseline_max_error"] == 0
                if fold == "final":
                    oldcut, = [m["threshold"] for m in load(root / "evidence/v9_freeze.json")["models"]
                               if m["model"] == "center" and m["policy"] == "selected" and m["seed"] == seed]
                    assert oldcut == threshold
            if recipe in ("R1", "R2"):
                np.testing.assert_array_equal(sample["sample_id"], read_npz(folder(root, fold, "S1", seed) / "sampling_ids.npz")["sample_id"])
            if replay:
                ids, stat = training_sample(study, cfg, recipe)
                np.testing.assert_array_equal(study.keys(ids), sample["sample_id"])
                np.testing.assert_array_equal(study.y[ids], sample["y"])
                np.testing.assert_array_equal(groups(study.ratio[ids]), sample["legal_group"])
                adapted = dict(**raw, threshold=np.float64(threshold), pred=dec["pred"])
                errors.append(model_replay(study, adapted, "lead_center", seed, path, cfg))
            if fold == "final":
                assert row.eval_budget_ok and state["best_epoch"] == state["fixed_epoch"] == pick["epochs"][recipe]
                item, = [m for m in freeze["models"] if m["recipe"] == recipe and m["seed"] == seed]
                assert item["threshold"] == threshold and item["hashes"] == snapshot(path)
                assert item["decisions_sha256"] == sha(root / "results/decisions/final" / f"{recipe}_s{seed}.npz")
            counts["models"] += 1
            counts["operating_points"] += 1
    assert counts == dict(operating_points=48, groups=192, matched=384, models=48)
    assert len(errors) == (48 if replay else 0)
    assert load(root / "evidence/leakage_support.json")["all_sample_key_overlaps"] == 0
    result = dict(status="PASS", checked_utc=now(), **counts, models_replayed=len(errors),
        max_score_difference=max(errors, default=None), baseline_all_scores_identical_to_V9=True,
        same_eval_support_and_step_budget=True, legal_groups_and_sampling_rebuilt=bool(replay),
        independent_thresholds_and_metrics=True, development_selection_reproduced=True,
        freeze_and_chronology=True, V5_V6_or_new_holdout_used=False, NYC_formal="blocked")
    save(output / "evidence" / ("verification_replay.json" if replay else "verification_offline.json"), result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path)
    p.add_argument("--replay-root", type=Path)
    a = p.parse_args()
    output, cfg = settings()
    verify((a.artifacts or output).resolve(), output, cfg, a.replay_root)

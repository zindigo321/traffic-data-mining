"""Independently recompute typed metrics and optionally replay V4 state dicts."""
import argparse
from datetime import date
import math
from pathlib import Path

from v4_data import V3_NAME
from v4_io import legacy_config, load, now, save, settings, sha
from v4_matrix import average, criterion, location
from v4_metrics import matched_points
from v4_train import make_study, matched_indices, read_npz
from evaluation import event_metrics, rank_score, threshold
from experiment import set_seed
from model import build
from v1.graph import TrainPreprocessor

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def dates_for(ids):
    return (ids // 1000 // 144).astype("datetime64[D]")


def check_support(keys, limits, banned):
    dates = dates_for(keys)
    for start, stop in banned:
        if np.any((dates >= np.datetime64(start)) & (dates < np.datetime64(stop))):
            raise ValueError("Embargo intersection")
    allowed = np.zeros(len(keys), dtype=bool)
    for start, stop in limits:
        allowed |= (dates >= np.datetime64(start)) & (dates < np.datetime64(stop))
    if not allowed.all() or len(np.unique(keys)) != len(keys) or not len(keys):
        raise ValueError("Invalid or duplicate support")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)


def verify_point(z, reported, cfg, days):
    y, p, keys, risk = (z[k] for k in ("y", "score", "sample_id", "risk"))
    boundary, key = reported["boundary_score"], reported["boundary_key"]
    pred = (p > boundary) | ((p == boundary) & (keys <= key))
    all_ties = p >= boundary
    boundary_row = np.flatnonzero(keys == key)
    assert len(boundary_row) == 1 and y[boundary_row[0]] == 1 and p[boundary_row[0]] == boundary
    assert reported["target_recall"] in cfg["recalls"]
    tp, fp = int(y[pred].sum()), int((pred & (y == 0)).sum())
    assert tp == math.ceil(reported["target_recall"] * int(y.sum())) == reported["TP"]
    assert fp == reported["FP"] and reported["recovery_FP"] == int((pred & risk & (y == 0)).sum())
    assert reported["FN"] == int(y.sum()) - tp and reported["TN"] == len(y) - int(y.sum()) - fp
    assert close(reported["achieved_recall"], tp / y.sum()) and close(reported["FP_per_day"], fp / days)
    assert (int(y[all_ties].sum()), int((all_ties & (y == 0)).sum())) == (reported["score_only_TP"], reported["score_only_FP"])
    assert close(f1_score(y, pred), reported["F1"])


def verify_job(folder, cfg, cover, train_intervals):
    z = read_npz(folder / "assessment.npz")
    stat = load(folder / "status.json")
    opt = load(folder / "options.json")
    if stat["status"] != "complete" or any(stat[k] != v for k, v in opt.items()):
        raise ValueError("Missing fit status")
    assert z["logit"].dtype == z["probability"].dtype == np.float32 and z["score"].dtype == np.float64
    assert z["threshold"].dtype == np.float64 and z["y"].dtype == np.int8 and z["sample_id"].dtype == np.int64
    y, p, ids = z["y"], z["score"], z["sample_id"]
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all()
    assert np.all(z["risk"][y == 1]) and np.all(np.diff(ids) > 0)
    np.testing.assert_array_equal(p, rank_score(z["probability"], z["risk"], True))
    check_support(ids, [cfg["fold_limits"][opt["fold"]]], cfg["all_exclusions"])
    assert len(y) == stat["n"] == cover["positive"] + cover["negative"]
    assert int(y.sum()) == stat["positive"] == cover["positive"]
    assert int((z["risk"] & (y == 0)).sum()) == cover["high_risk_recovered"]
    assert close(stat["known_coverage"], len(y) / cover["expected"])
    t = float(z["threshold"])
    assert close(t, threshold(y, p)) and close(t, stat["threshold"])
    pred = p >= t
    np.testing.assert_array_equal(pred, z["pred"])
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    assert (tn, fp, fn, tp) == tuple(stat[key] for key in ("tn", "fp", "fn", "tp"))
    assert close(average_precision_score(y, p), stat["AP"])
    assert close(average_precision_score(y[z["risk"]], p[z["risk"]]), stat["risk_AP"])
    assert close(roc_auc_score(y, p), stat["ROC_AUC"]) and close(f1_score(y, pred), stat["f1"])
    events = event_metrics(ids, y, pred)
    assert events["true_events"] == cover["true_events"] and close(events["event_f1"], stat["event_f1"])
    points = load(folder / "matched_recall.json")
    assert [x["target_recall"] for x in points] == cfg["recalls"]
    for point in points:
        verify_point(z, point, cfg, cover["total_days"])
    sm = read_npz(folder / "sampling_ids.npz")
    check_support(sm["sample_id"], train_intervals, cfg["all_exclusions"])
    assert np.isin(sm["label"], [0, 1]).all() and sm["risk"].all()
    original = legacy_config()
    sampling = load(folder / "sampling.json")
    assert len(sm["label"]) == sampling["unique_per_epoch"] and int(sm["label"].sum()) == sampling["known_risk_positive"]
    assert stat["updates"] == original["epochs"] * math.ceil(sampling["draws_per_epoch"] / original["batch_size"])
    assert stat["weight_sha256"] == sha(folder / "weight.pt")
    return z, sm


def replay(study, z, folder, opt, cfg):
    set_seed(opt["seed"], cfg)
    prep = TrainPreprocessor.load(folder / "preprocessor.npz")
    model = build(study.nfeatures * 2, cfg, opt["pool"], opt["seed"])
    model.load_state_dict(torch.load(folder / "weight.pt", map_location="cpu", weights_only=True))
    model.eval()
    positive = np.flatnonzero(z["y"] == 1)[len(np.flatnonzero(z["y"] == 1)) // 2]
    size = cfg["eval_batch_size"]
    offset = (positive // size) * size
    cut = slice(offset, offset + size)
    selected = study.val[cut]
    t, r = np.divmod(selected, study.nroad)
    d = study.d
    valid = d["valid"][r]
    data = d["x"][t[:, None], d["lookup"][r], :study.nfeatures]
    x = prep.transform(data, valid)
    inputs = (x, d["adj"][r], valid, np.zeros(len(selected), dtype=np.int64))
    with torch.inference_mode():
        logits = model(*map(torch.from_numpy, inputs))
        probability = logits.sigmoid().numpy()
    np.testing.assert_array_equal(study.keys(selected), z["sample_id"][cut])
    np.testing.assert_array_equal(study.y[selected], z["y"][cut])
    np.testing.assert_allclose(logits.numpy(), z["logit"][cut], atol=1e-7, rtol=0)
    np.testing.assert_allclose(probability, z["probability"][cut], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(rank_score(probability, z["risk"][cut], True) >= z["threshold"], z["pred"][cut])
    return len(selected), float(np.abs(probability - z["probability"][cut]).max())


def compare_v3(folder, old_folder, opt, z, sample, cover, cfg):
    old = read_npz(old_folder / "assessment.npz")
    earlier_sample = read_npz(old_folder / "sampling_ids.npz")
    for key in ("sample_id", "y", "risk", "risk_missing", "pred"):
        np.testing.assert_array_equal(old[key], z[key])
    for key in ("sample_id", "label", "risk"):
        np.testing.assert_array_equal(earlier_sample[key], sample[key])
    # Same float32 training recipe can exhibit small run-to-run CPU drift.
    # Decision, AP, F1 and the matched recall error counts must still agree.
    tolerance = {"logit": 1e-5, "probability": 2e-6, "score": 1e-6, "threshold": 1e-6}
    diffs = {}
    for key, limit in tolerance.items():
        np.testing.assert_allclose(old[key], z[key], atol=limit, rtol=0)
        diffs[key] = float(np.abs(old[key] - z[key]).max())
    past, current = (load(f / "status.json") for f in (old_folder, folder))
    assert past["best_epoch"] == current["best_epoch"]
    assert all(close(past[key], current[key]) for key in ("AP", "f1", "ROC_AUC"))
    old_points, _ = matched_points(old["sample_id"], old["y"], old["score"], old["risk"], cfg["recalls"], cover["total_days"])
    for a, b in zip(old_points, load(folder / "matched_recall.json"), strict=True):
        assert all(a[key] == b[key] for key in ("TP", "FP", "TN", "FN", "recovery_FP"))
    return dict(fold=opt["fold"], seed=opt["seed"], n=len(z["y"]), exact=not any(diffs.values()),
                maximum_absolute_differences=diffs, tolerances=tolerance, decisions_match=True, AP_F1_ROC_match=True,
                matched_recall_error_counts_match=True)


def run(artifact, out, cfg, external):
    gate, state = (load(artifact / f"evidence/{p}.json") for p in ("holdout_gate", "status"))
    assert gate["status"] == "blocked" and gate["requested_speeds_or_labels"] is False
    assert state["formal_status"] == "blocked" and state["old_holdout_used"] is False
    assert all(state[k] is None for k in ("formal_AP", "formal_F1", "current_heldout_dates", "target_reached"))
    assert not (artifact / "results/holdout").exists()
    original = legacy_config()
    selection = load(artifact / "results/selection.json")
    screen = load(artifact / "results/screen.json")
    conf = load(artifact / "results/confirmation.json")
    lock = load(artifact / "results/calibration_freeze.json")
    assert lock["test_unlocked"] is False and lock["selection_sha256"] == sha(artifact / "results/selection.json")
    assert date.fromisoformat(cfg["fold_limits"]["final"][1]) <= date.fromisoformat(cfg["earliest_new_date"])
    screens = ["B", *cfg["variants"]]
    reference = Path(cfg["v3_release"]) / V3_NAME / "results/dev"
    checks, old_checks, scores_replayed = [], [], []
    for fold in cfg["folds"] + ["final"]:
        train_ranges = original["final_train"] if fold == "final" else original["folds"][fold]["train"]
        cover = load(artifact / "evidence" / f"fold_{fold}.json")["parts"][1]
        dirs = (artifact / ("results/calibration" if fold == "final" else f"results/dev/{fold}"))
        expected = {f"{r}_mean_s{cfg['seeds'][0]}" for r in screens} if fold != "final" else set()
        expected |= {f"B_mean_s{seed}" for seed in cfg["seeds"]}
        expected |= {f"{selection['recipe']}_{pool}_s{seed}" for pool in cfg["pools"] for seed in cfg["seeds"]}
        assert {d.name for d in dirs.iterdir() if d.is_dir()} == expected
        arrays, sample = None, None
        v4_studies = {}
        for folder in sorted(dirs.iterdir()):
            z, sm = verify_job(folder, cfg, cover, train_ranges)
            if arrays is None:
                arrays = {k: z[k] for k in ("sample_id", "y", "risk", "risk_missing")}
                sample = sm
            for key, values in arrays.items():
                np.testing.assert_array_equal(z[key], values)
            for key in ("sample_id", "label", "risk"):
                np.testing.assert_array_equal(sm[key], sample[key])
            opt = load(folder / "options.json")
            if fold == "final":
                for name, digest in lock["hashes"][folder.name].items():
                    assert sha(folder / name) == digest
                assert close(float(z["threshold"]), lock["thresholds"][folder.name])
            if external:
                extra = opt["recipe"] in ("P2", "P3")
                if extra not in v4_studies:
                    study = make_study(artifact, cfg, fold, opt["recipe"])
                    np.testing.assert_array_equal(study.keys(study.val), z["sample_id"])
                    np.testing.assert_array_equal(study.y[study.val], z["y"])
                    np.testing.assert_array_equal(study.risk[study.val], z["risk"])
                    drawn, _, _ = matched_indices(study, original)
                    np.testing.assert_array_equal(study.keys(drawn), sm["sample_id"])
                    v4_studies[extra] = study
                n, diff = replay(v4_studies[extra], z, folder, opt, original)
                scores_replayed.append(dict(fold=fold, model=folder.name, n=n, max_probability_diff=diff))
                if fold != "final" and opt["recipe"] == "B":
                    old_path = reference / fold / f"R2_mean_s{opt['seed']}"
                    old_checks.append(compare_v3(folder, old_path, opt, z, sm, cover, cfg))
            checks.append(dict(fold=fold, model=folder.name, known=len(z["y"]), positive=int(z["y"].sum())))
        del v4_studies
    for rows, recorded, groups, column in (
        (screen["jobs"], screen["summary"], screens, "recipe"),
        (conf["rows"], conf["summary"], list(conf["summary"]), "combined")):
        for name in groups:
            subset = [r for r in rows if (r["recipe"] if column == "recipe" else f"{r['recipe']}_{r['pool']}") == name]
            for row in subset:
                folder = location(artifact, row["fold"], row["recipe"], row["pool"], row["seed"])
                stat = load(folder / "status.json")
                assert all(row[k] == v for k, v in stat.items())
                match = next(x for x in load(folder / "matched_recall.json") if x["target_recall"] == cfg["main_recall"])
                assert close(match["FP_per_day"], row["FP60_per_day"])
            assert average(subset) == recorded[name]
    assert screen["recipe"] == max(cfg["variants"], key=lambda r: criterion(screen["summary"][r]))
    assert selection["recipe"] == screen["recipe"]
    assert selection["pool"] == max(cfg["pools"], key=lambda p: criterion(conf["summary"][f"{selection['recipe']}_{p}"]))
    assert len(checks) == state["development_runs"] + state["calibration_runs"] == 54
    payload = dict(status="PASS", utc=now(), runs=len(checks), eligible_support_identical_per_fold=True,
                   known_and_sampling_embargo_passed=True, thresholds_recomputed=True, matched_recall_recomputed=True,
                   no_formal_holdout=True, external_replay=external, verified_V3_dev_runs=len(old_checks),
                   exact_V3_dev_runs=sum(row["exact"] for row in old_checks),
                   max_V3_probability_difference=max((row["maximum_absolute_differences"]["probability"]
                                                      for row in old_checks), default=None),
                   replays=len(scores_replayed), replay_max_probability_difference=max(
                       (r["max_probability_diff"] for r in scores_replayed), default=None),
                   comparison=old_checks, replay_details=scores_replayed, checks=checks)
    save(out / "evidence" / ("verification_replay.json" if external else "verification_offline.json"), payload)
    print({k: v for k, v in payload.items() if k not in ("comparison", "replay_details", "checks")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    root, cfg = settings()
    run(args.artifacts or root, root, cfg, args.replay)

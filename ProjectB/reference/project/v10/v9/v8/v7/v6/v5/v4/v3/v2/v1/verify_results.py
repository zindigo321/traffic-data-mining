"""Independent checks of public labels, timing, topology, predictions and validation selection."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiment import SPLITS, StudyData, dump, read_npz, set_seed, test_probabilities
from fetch_chicago import digest
from graph import TrainPreprocessor
from settings import CONFIG, WORK_ROOT


def require(test, message):
    if not test:
        raise AssertionError(message)


def verify_data(root):
    base = root / "prepared"
    prepared = read_npz(base / "chicago.npz")
    audit = read_npz(base / "chicago_audit_local.npz")
    speed, history, observed, source_time = [audit[k] for k in ("speed", "history", "observed", "source_time")]
    x, y, split, slots, ids = [prepared[k] for k in ("x", "y", "split", "window_minute", "road_ids")]
    cfg = CONFIG["public"]
    step = cfg["bin_minutes"]
    metadata = json.loads((root / "public_data" / "download.json").read_text(encoding="utf-8"))
    require(metadata["status"] == "complete" and digest(root / "public_data" / metadata["file"]) == metadata["sha256"],
            "Original downloaded CSV checksum")
    times = slots.astype("datetime64[m]")
    days = times.astype("datetime64[D]")
    dates = np.unique(days)
    require(np.all(days[(y >= 0).any(axis=1)] > dates[0]),
            "A labeled window precedes the end of first-day topology discovery")
    require(np.all(np.diff(slots) > 0) and np.all(np.diff(split) >= 0), "Time grid/split is monotonic")
    for before, after in zip(SPLITS, SPLITS[1:]):
        a, b = SPLITS.index(before), SPLITS.index(after)
        require(slots[split == a].max() + step < slots[split == b].min() - 3*step, "Split scopes overlap")
    require(np.array_equal(observed, ~np.isnat(source_time)), "Presence mask disagrees with source time")
    require(np.all(source_time[observed] >= np.broadcast_to(times[:, None], y.shape)[observed]), "Source before bin")
    require(np.all(source_time[observed] < np.broadcast_to((times + np.timedelta64(step, "m"))[:, None],
                                                          y.shape)[observed]), "Source after end of bin")

    # Sequential, per-day replay. Freeze the sums upon exiting the training split.
    totals, counts = {}, {}
    expected = np.empty_like(history)
    n = np.empty_like(audit["history_days"])
    for day in dates:
        row_idx = np.flatnonzero(days == day)
        weekday = pd.Timestamp(day).weekday() >= 5
        key = weekday
        if key not in totals:
            totals[key] = np.zeros_like(speed[row_idx])
            counts[key] = np.zeros_like(n[row_idx])
        count = counts[key]
        n[row_idx] = count
        expected[row_idx] = np.divide(totals[key], count, out=np.full_like(totals[key], np.nan),
                                     where=count >= CONFIG["minimum_history_days"])
        if split[row_idx[0]] == 0:
            current = speed[row_idx]
            totals[key] += np.nan_to_num(current, nan=0)
            counts[key] += np.isfinite(current)
    np.testing.assert_allclose(history, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(n, audit["history_days"])

    valid_history = (expected > 0) & np.isfinite(expected)
    low = valid_history & (speed < CONFIG["congestion_ratio"] * expected)
    known = valid_history & np.isfinite(speed) & observed
    labels = np.where(known & ~low, 0, -1)
    time_to_idx = {int(v): i for i, v in enumerate(slots)}
    lag_values = np.full_like(x[..., :5], np.nan)
    for i, start in enumerate(slots):
        for j in range(3):
            p = time_to_idx.get(int(start) - (j + 1)*step)
            if p is None or split[i] != split[p]:
                continue
            lag_values[i, :, j] = speed[p]
            require(np.all(source_time[p, observed[p]] < times[i]), "Lag uses an observation unavailable at input cutoff")
            if j == 0:
                labels[i, low[i] & low[p]] = 1
                lag_values[i, :, 3] = audit["bus"][p]
                lag_values[i, :, 4] = audit["message"][p]
    np.testing.assert_array_equal(labels, y)
    np.testing.assert_allclose(lag_values, x[..., :5], rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(np.isfinite(lag_values[..., 0]), prepared["eligible"])
    require(np.isfinite(x[..., 7:]).all(), "Missing static/time features")

    topo = json.loads((root / "data_audit" / "chicago" / "topology.json").read_text(encoding="utf-8"))
    nodes, raw_edges = topo["nodes"], topo["edges"]
    pairs = {(e["src"], e["dst"]) for e in raw_edges}
    require(len(pairs) == len(raw_edges), "Duplicate directed edges")
    require(set(map(int, nodes)) == set(ids.tolist()), "Road node index mismatch")
    decimals = cfg["endpoint_round_decimals"]
    radius = cfg["endpoint_named_radius_m"]
    for e in raw_edges:
        a, b = nodes[str(e["src"])], nodes[str(e["dst"])]
        d = lambda u, v: 2 * 6371000 * math.asin(math.sqrt(
            math.sin(math.radians((u[0]-v[0])/2))**2
            + math.cos(math.radians(u[0]))*math.cos(math.radians(v[0]))
            * math.sin(math.radians((u[1]-v[1])/2))**2))
        require(e["src"] != e["dst"] and d(a["entry"], b["exit"]) > radius, "Immediate reversal or self-road edge")
        if e["evidence"] == "rounded_endpoint":
            require([round(c, decimals) for c in a["exit"]] == [round(c, decimals) for c in b["entry"]], "Endpoint mismatch")
        else:
            require(a["exit_crossing"] and a["exit_crossing"] == b["entry_crossing"] and
                    d(a["exit"], b["entry"]) <= radius, "Named-intersection edge lacks evidence")
    for i, ident in enumerate(ids):
        valid = prepared["valid"][i]
        pos = prepared["lookup"][i, valid]
        selected = ids[pos].tolist()
        require(int(ident) == selected[0] and len(set(selected)) == len(selected), "Center/target mismatch")
        local = prepared["adj"][i, :len(pos), :len(pos)]
        expected_edges = np.array([[(a, b) in pairs for b in selected] for a in selected])
        np.testing.assert_array_equal(expected_edges, local)
        require(not prepared["adj"][i, ~valid].any() and not prepared["adj"][i, :, ~valid].any(), "Padding edge")
    return dict(status="PASS", expected_windows=int(y.size), source_observation_times=int(observed.sum()),
                causal_histories_checked=int(y.size), labels_checked=int(y.size), lag_feature_values_checked=int(lag_values.size),
                directed_edges=len(pairs), target_center_maps=len(ids), source_sha256=metadata["sha256"],
                leakage_checks="Source time, feature availability, strict prior-day history, frozen training cutoff, whole-date gaps")


def metrics_from_csv(y, p, cutoff):
    pred = p >= cutoff
    n = len(y)
    pos = int(y.sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tp = int(((y == 1) & pred).sum())
    ap = None
    if 0 < pos < n:
        order = np.argsort(-p, kind="stable")
        scores = p[order]
        ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), n-1]
        true_so_far = np.cumsum(y[order])[ends]
        increments = np.diff(np.r_[0, true_so_far])
        ap = float((increments * true_so_far / (ends + 1)).sum() / pos)
    return dict(n=n, positive=pos, negative=n-pos, tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tn+tp)/n if n else None, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/pos if pos else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None, pr_auc=ap)


def scores(frame, row):
    dtype = np.float32 if row["model"] in {"GraphTransformer", "MLP"} else np.float64
    return frame["probability"].to_numpy(dtype=dtype).astype(np.float64)


def check_equal(actual, expected, key):
    require((actual is None and expected is None) or
            (actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)),
            f"Metric mismatch for {key}: {actual} != {expected}")


def independently_select_threshold(y, p):
    order = np.argsort(-p)
    ranked = p[order]
    end = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(y)-1]
    values = ranked[end]
    tp = np.cumsum(y[order])[end]
    f1 = 2 * tp / (int(y.sum()) + end + 1)
    candidates = values[np.isclose(f1, f1.max(), atol=1e-12, rtol=0)]
    return min(candidates, key=lambda t: (abs(t-.5), t))


def verify_results(root, artifact, replay):
    geolife = artifact / "results" / "geolife" / "status.json"
    status = dict(geolife=json.loads(geolife.read_text(encoding="utf-8")) if geolife.exists() else None)
    out = artifact / "results" / "chicago"
    if not (out / "status.json").exists():
        status["chicago"] = dict(status="SKIP", reason="Chicago model matrix unavailable")
        return status
    recorded = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if recorded["status"] != "complete":
        status["chicago"] = dict(status="SKIP", reason=recorded.get("reason", recorded["status"]))
        return status
    start = time.perf_counter()
    config = json.loads((out / "config_frozen.json").read_text(encoding="utf-8"))
    frozen = json.loads((out / "selection_before_test.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    pool_order = config["pooling"]
    require(len(rows) == len(config["seeds"]) * (len(pool_order) + 3), "Incomplete model matrix")
    require(max(pool_order, key=frozen["validation_ap_means"].get) == frozen["selected_pooling"], "Wrong validation pool")
    truth = pd.read_csv(out / "test_truth.csv.gz")
    require(truth.sample_id.is_unique and truth.label.isin([0, 1]).all(), "Unknown or duplicate test targets")
    if replay:
        data = StudyData(root, "chicago")
        set_seed(config["seeds"][0])
        training_ids = read_npz(out / "train_index.npz")["sample_id"]
        tr = data.parts["train"]
        require(np.isin(training_ids, tr).all() and len(np.unique(training_ids)) == len(training_ids), "Train index leakage")
        require(set(tr[data.y[tr] == 1]) <= set(training_ids), "Not all positives retained")
        expected_prep = data.fit_preprocessor()
        prep = TrainPreprocessor.load(out / "preprocessor.npz")
        for attr in ("mean", "median", "scale"):
            np.testing.assert_array_equal(getattr(expected_prep, attr), getattr(prep, attr))
        data.configure(prep)
        test_ids = data.parts["test"]
        pd.testing.assert_frame_equal(pd.DataFrame(data.keys(test_ids)), truth, check_dtype=False)
        flat = data.flat(test_ids) if len(test_ids) else np.empty((0, 3*data.features), dtype=np.float32)
    comparison = []
    for row in rows:
        stem = f"seed_{row['seed']}/{row['name']}"
        val = pd.read_csv(out / f"{stem}_validation_predictions.csv.gz", float_precision="round_trip")
        y_val = val.label.to_numpy()
        p_val = scores(val, row)
        require(val.sample_id.is_unique and val.label.isin([0, 1]).all(), "Invalid validation targets")
        if replay:
            pd.testing.assert_frame_equal(pd.DataFrame(data.keys(data.parts["validation"])),
                                          val[truth.columns], check_dtype=False)
        threshold = .5 if row["model"] == "Majority" else independently_select_threshold(y_val, p_val)
        check_equal(threshold, row["threshold"], "validation threshold")
        calculated_val = metrics_from_csv(y_val, p_val, threshold)
        check_equal(calculated_val["pr_auc"], row["validation_ap"], "validation AP")
        check_equal(calculated_val["f1"], row["validation_f1"], "validation F1")
        if row["model"] == "GraphTransformer":
            curve = pd.read_csv(out / f"{stem}_curve.csv")
            require(int(curve.loc[curve.validation_ap.idxmax(), "epoch"]) == row["best_epoch"], "Checkpoint selection")
        test = pd.read_csv(out / f"{stem}_test_predictions.csv.gz", float_precision="round_trip")
        pd.testing.assert_frame_equal(truth, test[truth.columns], check_dtype=False)
        p = scores(test, row)
        np.testing.assert_array_equal(p >= row["threshold"], test.predicted.to_numpy(dtype=bool))
        actual = metrics_from_csv(test.label.to_numpy(), p, row["threshold"])
        for key in actual:
            check_equal(actual[key], row.get(key), key)
        deviation = None
        if replay:
            reloaded = test_probabilities(data, out, config, row, test_ids, flat)
            np.testing.assert_allclose(reloaded, p, rtol=1e-7, atol=1e-8)
            deviation = float(np.max(np.abs(reloaded - p))) if len(p) else 0.
        comparison.append(dict(model=stem, validation_rows=len(val), test_rows=len(test),
                               max_abs_reload_error=deviation, metrics="PASS"))
    for pool in pool_order:
        mean = float(np.mean([r["validation_ap"] for r in rows if r["pooling"] == pool]))
        check_equal(mean, frozen["validation_ap_means"][pool], "validation mean")
    status["chicago"] = dict(status="PASS", models=len(comparison), test_rows_per_model=len(truth),
                             reload_all_models=replay, model_checks=comparison,
                             seconds=time.perf_counter() - start)
    return status


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stage", required=True, choices=["data", "metrics", "reload"])
    cli.add_argument("--artifacts", type=Path, default=WORK_ROOT)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.stage == "data":
        result = verify_data(WORK_ROOT)
    else:
        result = verify_results(WORK_ROOT, args.artifacts, args.stage == "reload")
    dump(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()



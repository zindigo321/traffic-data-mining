"""Quantify high-risk recovery false alarms and missed positives without held-out data."""
import argparse
from pathlib import Path

from v4_io import load, save, settings
from v4_data import V3_NAME
from v4_metrics import matched_points
from v4_train import make_study, read_npz

import numpy as np
import pandas as pd

SELECTED_COLUMNS = ("fold", "model", "seed", "sample_id", "label", "score", "alert_60", "outcome", "road",
                    "day", "peak", "daytype", "prior_speed_ratio", "prior_ratio_bin", "prior_low_depth",
                    "depth_bin", "prior_buses", "bus_bin", "neighbors", "connected")


def error_rows(study, z, fold, model, seed, days, rate):
    ids = study.val
    keys = study.keys(ids)
    np.testing.assert_array_equal(z["sample_id"], keys)
    np.testing.assert_array_equal(z["y"], study.y[ids])
    np.testing.assert_array_equal(z["risk"], study.risk[ids])
    points, selected = matched_points(keys, z["y"], z["score"], z["risk"], [rate], days)
    if points[0]["other_FP"]:
        raise ValueError("Routing should place known positives above all ordinary windows")
    alert = np.zeros(len(keys), dtype=bool)
    alert[selected[rate]] = True
    take = np.flatnonzero(z["risk"])
    labels, alarms = z["y"][take], alert[take]
    t, r = np.divmod(ids[take], study.nroad)
    prior_ratio = study.ratio[ids[take]]
    buses = study.d["x"][t, r, 3]
    depth = study.d["x"][t, r, 31]
    connections = study.d["valid"][r].sum(axis=1) - 1
    days = study.d["minute"][t] // 1440
    peak = np.where(study.d["minute"][t] % 1440 < 12 * 60, "AM", "PM")
    return pd.DataFrame(dict(fold=fold, model=model, seed=seed, sample_id=keys[take], label=labels,
        score=z["score"][take], alert_60=alarms, outcome=np.select(
            [(labels == 1) & alarms, labels == 1, alarms], ["TP", "FN", "FP"], default="TN"),
        road=study.d["road_ids"][r], day=pd.to_datetime(days, unit="D").strftime("%Y-%m-%d"),
        peak=peak, daytype=np.where((days + 3) % 7 >= 5, "weekend", "weekday"),
        prior_speed_ratio=prior_ratio, prior_ratio_bin=np.select(
            [prior_ratio < .2, prior_ratio < .3], ["<0.20", "[0.20,0.30)"], default="[0.30,0.40)"),
        prior_low_depth=depth, depth_bin=np.select([depth == 1, depth == 2], ["1", "2"], default="3+"),
        prior_buses=buses, bus_bin=np.select([~np.isfinite(buses), buses <= 1, buses <= 3],
            ["missing", "0-1", "2-3"], default="4+"),
        neighbors=connections, connected=np.where(connections > 0, "has_neighbor", "isolated")))


def summarize(output, events):
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "risk_windows.csv", index=False, columns=SELECTED_COLUMNS, float_format="%.17g")
    groups = []
    for dimension in ("prior_ratio_bin", "depth_bin", "bus_bin", "connected", "peak", "daytype", "road"):
        for (model, category), g in events.groupby(["model", dimension], sort=True):
            counts = g["outcome"].value_counts()
            n = {k: int(counts.get(k, 0)) for k in ("FP", "FN", "TP", "TN")}
            rec, pos = n["FP"] + n["TN"], n["FN"] + n["TP"]
            groups.append(dict(model=model, dimension=dimension, category=str(category), **n, recovery_support=rec,
                               positive_support=pos, FP_rate=n["FP"]/rec if rec else None,
                               FN_rate=n["FN"]/pos if pos else None))
    pd.DataFrame(groups).to_csv(output / "groups.csv", index=False, na_rep="N/A")
    save(output / "summary.json", dict(total_rows=len(events), models=sorted(events.model.unique().tolist()),
                                      folds=sorted(events.fold.unique().tolist()), dimensions=7,
                                      rates="Known risky recoveries or positive windows, seed-window units",
                                      retrospective_only=True))


def run(root, cfg, previous=False):
    parts = []
    selected = load(root / "results/selection.json")["recipe"] if not previous else None
    old = Path(cfg["v3_release"]) / V3_NAME / "results/dev"
    for fold in cfg["folds"]:
        study = make_study(root, cfg, fold, "B")
        cover = load(root / "evidence" / f"fold_{fold}.json")["parts"][1]
        for seed in cfg["seeds"]:
            if previous:
                spec = [("V3_published_dev_mean", old / fold / f"R2_mean_s{seed}")]
            else:
                base = root / "results/dev" / fold
                spec = [("B_mean", base / f"B_mean_s{seed}")]
                spec += [(f"{selected}_{pool}", base / f"{selected}_{pool}_s{seed}") for pool in cfg["pools"]]
            for model, folder in spec:
                parts.append(error_rows(study, read_npz(folder / "assessment.npz"), fold,
                                        model, seed, cover["total_days"], cfg["main_recall"]))
    dest = root / "results" / ("initial_error_review" if previous else "paired_error_review")
    events = pd.concat(parts, ignore_index=True)
    summarize(dest, events)
    print(dict(stage=dest.name, known_risk_seed_windows=len(events), outcomes=events["outcome"].value_counts().to_dict()),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", action="store_true")
    run(*settings(), parser.parse_args().previous)

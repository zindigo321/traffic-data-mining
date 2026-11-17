"""Chicago wall-clock folds, frozen topology, lag-only inputs and separately derived truth."""
import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import CODE, digest, json_read, json_write, settings
sys.path.append(str(CODE / "v1"))
from v1.prepare_chicago import (FEATURES as BASE_FEATURES, META, aligned_lag, build_features, causal_history,
                             make_labels, take_lag, topology)

FEATURE_NAMES = BASE_FEATURES + [
    f"lag{lag}_{feature}" for lag in (10, 20, 30)
    for feature in ("relative_speed", "speed_minus_threshold_mph", "history_days")
] + ["lag20_bus_count", "lag20_message_count", "lag30_bus_count", "lag30_message_count"]
REASONS = {
    0: "not_low_current_estimate", 1: "two_consecutive_low_estimates", 2: "unconfirmed_low_estimate",
    3: "insufficient_history", 4: "invalid_speed_or_probe_support", 5: "absent_or_changed_metadata",
}
PREPARED = "prepared"


def read_npz(path):
    with np.load(path, allow_pickle=False) as arr:
        return {key: arr[key] for key in arr.files}


def local_dates(intervals):
    dates = [d for start, end in intervals for d in pd.date_range(start, end, inclusive="left").to_numpy(dtype="datetime64[D]")]
    if len(set(dates)) != len(dates):
        raise ValueError("Overlapping calendar intervals")
    return np.asarray(sorted(dates), dtype="datetime64[D]")


def times_for(dates, cfg):
    slots = np.array([60 * h + m for h in cfg["source"]["hours"] for m in (0,10,20,30,40,50)])
    return pd.DatetimeIndex((dates[:, None] + slots[None, :].astype("timedelta64[m]")).ravel())


def minute_values(times):
    return times.to_numpy(dtype="datetime64[m]").astype(np.int64)


def check_days(cfg, train, evaluation):
    forbidden = local_dates(cfg["exclusions"])
    if np.isin(np.concatenate([train,evaluation]), forbidden).any() or not len(evaluation):
        raise ValueError("Forbidden old test date or empty assessment segment")
    if len(train) and (train[-1] >= evaluation[0] or np.intersect1d(train,evaluation).size):
        raise ValueError("Forward fold overlaps or touches a past calendar date")


def parse_legacy(root, cfg):
    entry = json_read(root / "evidence/v1_inputs.json")
    raw = Path(entry["cache"]) / "public_data/chicago_fixed.csv"
    if digest(raw) != entry["verified_cache"]["public_data/chicago_fixed.csv"]:
        raise ValueError("v1 raw checksum changed")
    data = pd.read_csv(raw)
    data["time"] = pd.to_datetime(data["time"], format="ISO8601")
    first, old_stop = [pd.Timestamp(x) for x in ("2024-07-01", cfg["exclusions"][0][0])]
    use = data.loc[(data["time"] >= first) & (data["time"] < old_stop)].copy()
    return use, int(len(data) - len(use))


def normalize_meta(raw):
    for name in ("street", "direction", "from_street", "to_street"):
        raw[name] = raw[name].fillna("").str.upper().str.split().str.join(" ")
    return raw


def load_roads(root, cfg, legacy=None):
    meta_path = root / "evidence/roads_v1.csv"
    adj_path = root / PREPARED / "topology.npz"
    if meta_path.is_file() and adj_path.is_file():
        return pd.read_csv(meta_path, float_precision="round_trip", keep_default_na=False).set_index("segment_id"), read_npz(adj_path)
    if legacy is None:
        raise ValueError("Prepare original topology before using it")
    entry = json_read(root / "evidence/v1_inputs.json")
    pinned = read_npz(Path(entry["cache"]) / "prepared/chicago.npz")
    day = legacy[legacy["time"].dt.normalize() == pd.Timestamp("2024-07-01")]
    roads = normalize_meta(day[["segment_id", *META]].drop_duplicates().copy()).drop_duplicates()
    roads = roads.set_index("segment_id", verify_integrity=True).sort_index()
    np.testing.assert_array_equal(roads.index.to_numpy(), pinned["road_ids"])
    if len(roads) != cfg["roster_count"]:
        raise ValueError("Roster changed")
    original_config = json_read(Path(entry["extracted"]) / "project/config.json")
    nodes, edges, lookup, valid, adj = topology(roads, original_config["public"], original_config)
    for key, a in (("lookup", lookup), ("valid", valid), ("adj", adj)):
        np.testing.assert_array_equal(a, pinned[key])
    graph = {key: pinned[key] for key in ("road_ids", "lookup", "valid", "adj")}
    roads.to_csv(meta_path, float_format="%.17g")
    out = root / PREPARED
    out.mkdir(exist_ok=True)
    np.savez_compressed(adj_path, **graph)
    json_write(root / "evidence/road_identity.json", dict(count=len(roads), edges=len(edges), road_ids=list(nodes),
               road_sha256=digest(meta_path), topology_sha256=digest(adj_path), exactly_matches_v1=True))
    return roads, graph


def read_extension(root, name):
    if name == "holdout_sealed" and not (root / "evidence/holdout_unlock.json").is_file():
        raise PermissionError("Holdout observations may only be parsed after freeze and unlock")
    receipt = json_read(root / "evidence" / f"download_{name}.json")
    if receipt["status"] != "complete":
        raise ValueError(f"Extension {name} is incomplete")
    chunks = []
    for block in receipt["shards"]:
        path = Path(receipt["directory"]) / block["file"]
        if digest(path) != block["sha256"]:
            raise ValueError(f"Extension changed: {path}")
        raw = pd.read_csv(path)
        if len(raw) != block["rows"]:
            raise ValueError("CSV row count differs from transport receipt")
        raw["time"] = pd.to_datetime(raw["time"], format="ISO8601")
        dates = raw["time"].dt.normalize()
        if ((dates < pd.Timestamp(block["start"])) | (dates >= pd.Timestamp(block["stop_exclusive"]))).any():
            raise ValueError("Observation outside its frozen shard")
        chunks.append(raw)
    return pd.concat(chunks, ignore_index=True)


def grid_from_raw(raw, dates, roads, cfg):
    times = times_for(dates, cfg)
    raw = raw.loc[raw["time"].dt.normalize().isin(pd.to_datetime(dates))].copy()
    if not raw["time"].dt.hour.isin(cfg["source"]["hours"]).all():
        raise ValueError("Non-peak records in official export")
    normalize_meta(raw)
    present_roster = raw["segment_id"].isin(roads.index)
    # Match the v1 metadata definition exactly. Numeric values are parsed by the same pandas parser.
    stable = raw.merge(roads.reset_index(), on=["segment_id", *META], validate="many_to_one")
    stable_count = len(stable)
    stable = stable.drop_duplicates()
    if stable.duplicated(["segment_id", "time"]).any():
        raise ValueError("Conflicting observations at identical source timestamps")
    stable["window"] = stable["time"].dt.floor("10min")
    stable = stable.sort_values("time").drop_duplicates(["window","segment_id"], keep="last")
    index = pd.MultiIndex.from_product([times, roads.index], names=["window","segment_id"])
    frame = stable.set_index(["window","segment_id"]).reindex(index)
    shape = (len(times), len(roads))
    observed = frame["time"].notna().to_numpy().reshape(shape)
    actual_time = frame["time"].to_numpy(dtype="datetime64[us]").reshape(shape)
    t = times.to_numpy(dtype="datetime64[m]")[:,None]
    if not (np.all((actual_time >= t)[observed]) and np.all((actual_time < t + np.timedelta64(10, "m"))[observed])):
        raise ValueError("Observation outside its declared time bin")
    valid = (frame["speed"].between(0,cfg["source"]["max_speed_mph"]) & frame["bus_count"].gt(0)
             & frame["message_count"].gt(0))
    speed, bus, message = [frame[c].where(valid).to_numpy(dtype=np.float64).reshape(shape)
                           for c in ("speed","bus_count","message_count")]
    funnel = dict(rows=len(raw), outside_pinned_roster=int((~present_roster).sum()),
                  metadata_conflicts=int(present_roster.sum())-stable_count,
                  duplicates_or_superseded=stable_count-len(stable), observed=int(observed.sum()),
                  usable=int(valid.sum()), expected=int(observed.size), observed_days=int(frame["time"].dt.normalize().nunique()))
    return dict(speed=speed, bus=bus, message=message, observed=observed, times=times), funnel


def frozen_history(speed, weekdays, day_phase, cfg):
    days, slots, roads = speed.shape
    sums = np.zeros((2,slots,roads),dtype=np.float64)
    n = np.zeros((2,slots,roads),dtype=np.int32)
    for weekend in (0, 1):
        part = speed[(weekdays == weekend) & (day_phase == 0)]
        sums[weekend] = np.nansum(part, axis=0)
        n[weekend] = np.isfinite(part).sum(axis=0)
    mean = np.divide(sums, n, out=np.full_like(sums, np.nan), where=n >= cfg["minimum_history_days"])
    return dict(reference=mean, days=n)


def inputs_only(grid, history, support, phase, roads, cfg):
    """Current row is used as a timestamp only. All traffic values are indexed via prior bins."""
    times = grid["times"]
    lag = [aligned_lag(times, phase, 10*j) for j in (1,2,3)]
    base, eligible = build_features(grid["speed"], grid["bus"], grid["message"], times, lag, roads, 10)
    extension = []
    prior_ratio = []
    for j in lag:
        speed, baseline = take_lag(grid["speed"],j), take_lag(history,j)
        relative = np.divide(speed, baseline, out=np.full_like(speed,np.nan), where=baseline > 0)
        deviation = np.where(baseline > 0, speed-cfg["ratio"]*baseline, np.nan)
        extension.extend((relative, deviation, take_lag(support.astype(np.float64),j)))
        prior_ratio.append(relative)
    for j in lag[1:]:
        extension.extend((take_lag(grid["bus"],j), take_lag(grid["message"],j)))
    ext = np.stack(extension, axis=-1).astype(np.float32)
    return dict(x=np.concatenate([base,ext],axis=-1), eligible=eligible, ratio_prior=prior_ratio[0],
                risk=prior_ratio[0] < cfg["ratio"], lag=lag[0])


def truth_only(grid, hist, lag, cfg):
    y, reasons = make_labels(grid["speed"], hist, grid["observed"], lag, cfg["ratio"])
    quality = grid["observed"] & np.isfinite(grid["speed"]) & np.isfinite(hist) & (hist > 0)
    aux = np.where(quality, (grid["speed"] < cfg["ratio"] * hist).astype(np.int8), -1).astype(np.int8)
    return y, reasons, aux


def fingerprint_keys(keys):
    import hashlib
    return hashlib.sha256(np.asarray(keys, dtype="<i8").tobytes()).hexdigest()


def key_matrix(minutes, roads):
    return (np.asarray(minutes, dtype=np.int64)[:, None] // 10) * 1000 + np.asarray(roads)[None, :]


def summarize(out, arr, reasons, phase_name, funnel):
    rows, errors = [], []
    split, y, eligible, risk, aux = [arr[k] for k in ("split","y","eligible","risk","aux")]
    minutes = arr["minute"]
    rows_days = pd.to_datetime(minutes, unit="m").normalize().to_numpy(dtype="datetime64[D]")
    positive = eligible & (y == 1)
    contiguous = (np.diff(minutes,prepend=0) == 10) & (np.diff(split,prepend=-1) == 0)
    prior = np.vstack([np.zeros(y.shape[1], dtype=bool), positive[:-1]]) & contiguous[:,None]
    events = positive & ~prior
    for p in np.unique(split):
        mask = split == p
        part = phase_name[p]
        ys, es, rs, az = y[mask], eligible[mask], risk[mask], aux[mask]
        known = es & (ys >= 0)
        if ((ys == 1) & es & ~rs).any():
            raise ValueError("Confirmed positive outside the mathematically required prior-low group")
        rows.append(dict(part=part, expected=int(ys.size), eligible=int(es.sum()),
                         positive=int((es & (ys==1)).sum()), negative=int((es & (ys==0)).sum()),
                         unknown_eligible=int((es & (ys==-1)).sum()), excluded_no_lag=int((~es).sum()),
                         true_events=int(events[mask].sum()),
                         positive_days=len(np.unique(rows_days[mask][(ys==1).any(axis=1)])),
                         known_roads=int(known.any(0).sum()), total_days=len(np.unique(rows_days[mask])),
                         high_risk_recovered=int((es & (ys==0) & rs).sum()),
                         risk_missing_eligible=int((es & ~np.isfinite(arr["ratio_prior"][mask])).sum()),
                         aux_low=int((es & (az==1)).sum()), aux_normal=int((es & (az==0)).sum()),
                         known_coverage=float(known.sum()/ys.size),
                         feature_missing={col:float((~np.isfinite(arr["x"][mask,:,i])).mean())
                                          for i,col in enumerate(FEATURE_NAMES)}))
        for key, reason in REASONS.items():
            reason_mask = reasons[mask] == key
            errors.append(dict(part=part, reason=reason, expected=int(reason_mask.sum()), eligible=int((reason_mask & es).sum())))
    json_write(out, dict(parts=rows, funnel=funnel, reasons=errors,
               overlaps=dict(train_eval_days=0, same_public_road_roster=True, users_and_trajectory_ids="N/A"),
               window_minutes=10, source_value_availability="source time < target start for lags",
               sample_key="(Chicago civil minute since 1970 // 10) * 1000 + directed segment_id"))
    print(out.name, [{k:v for k,v in p.items() if k!="feature_missing"} for p in rows], flush=True)


def make_fold(root, cfg, name, grid, road, graph, train, assess, funnel):
    check_days(cfg,train,assess)
    dates = np.concatenate([train,assess])
    times = times_for(dates,cfg)
    index = grid["times"].get_indexer(times)
    if (index < 0).any():
        raise ValueError(f"Data grid does not cover {name}")
    raw = {key:grid[key][index] for key in ("speed","bus","message","observed")}
    raw["times"] = times
    count_slots = len(cfg["source"]["hours"])*6
    stage = np.r_[np.zeros(len(train),dtype=np.int8), np.ones(len(assess),dtype=np.int8)]
    phase = np.repeat(stage,count_slots)
    weekends = np.asarray(pd.to_datetime(dates).dayofweek >= 5)
    cube = raw["speed"].reshape(len(dates),count_slots,len(road))
    hist, days = causal_history(cube, weekends, stage, cfg["minimum_history_days"])
    x = inputs_only(raw,hist,days,phase,road,cfg)
    y, reasons, aux = truth_only(raw,hist,x.pop("lag"),cfg)
    arr = dict(**x, y=y, aux=aux, split=phase, minute=minute_values(times), **graph)
    prep = root/PREPARED
    if name == "f1":
        cache = Path(json_read(root/"evidence/v1_inputs.json")["cache"])
        old = read_npz(cache/"prepared/chicago.npz")
        for key,val in (("x",x["x"][...,:len(BASE_FEATURES)]),("eligible",x["eligible"]),("y",y),("split",phase)):
            np.testing.assert_array_equal(val,old[key][:len(phase)])
        json_write(root/"evidence/f1_exact_support.json",dict(status="PASS", bins=int(y.size), v1_features_identical=True,
                       label_and_eligibility_identical=True, checked_at_local_windows=len(phase)))
    np.savez_compressed(prep/f"{name}.npz",**arr)
    names = {0:"train",1:"calibration" if name=="final" else "development"}
    summarize(root/"evidence"/f"fold_{name}.json",arr,reasons,names,funnel)
    if name == "final":
        np.savez_compressed(prep/"frozen_history.npz",**frozen_history(cube,weekends,stage,cfg))
    return arr


def make_holdout(root, cfg, freeze):
    check_days(cfg,np.empty(0,dtype="datetime64[D]"),local_dates([cfg["holdout"]]))
    road, graph = load_roads(root,cfg)
    date_index = local_dates([cfg["holdout"]])
    raw, funnel = grid_from_raw(read_extension(root,"holdout_sealed"),date_index,road,cfg)
    shape = raw["speed"].shape
    phase = np.ones(shape[0],dtype=np.int8)
    hist_state = read_npz(root/PREPARED/"frozen_history.npz")
    weekends = (pd.to_datetime(date_index).dayofweek >= 5).astype(int)
    history = hist_state["reference"][weekends].reshape(shape)
    days = hist_state["days"][weekends].reshape(shape)
    x = inputs_only(raw,history,days,phase,road,cfg)
    minutes = minute_values(raw["times"])
    keys = key_matrix(minutes,graph["road_ids"])
    if fingerprint_keys(keys) != freeze["expected_grid_key_sha256"]:
        raise ValueError("Actual holdout grid differs from preregistration")
    elig = keys[x["eligible"]]
    json_write(root/"evidence/holdout_features_receipt.json",dict(eligible_count=len(elig), eligible_ids_sha256=fingerprint_keys(elig),
               all_grid_count=keys.size, expected_grid_key_sha256=fingerprint_keys(keys), truth_built=False))
    y, reasons, aux = truth_only(raw,history,x.pop("lag"),cfg)
    data = dict(**x,y=y,aux=aux,split=phase,minute=minutes,**graph)
    summarize(root/"evidence/fold_holdout.json",data,reasons,{1:"holdout"},funnel)
    np.savez_compressed(root/PREPARED/"holdout.npz", **data)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scope", choices=["local","extended"])
    scope = p.parse_args().scope
    root,cfg = settings()
    prior, excluded = parse_legacy(root,cfg)
    roads,graph = load_roads(root,cfg,prior)
    intervals = [["2024-07-01",cfg["exclusions"][0][0]]]
    if scope == "extended":
        prior = pd.concat([prior,read_extension(root,"development")],ignore_index=True)
        intervals.append(cfg["development_download"])
        names = ["f2","f3","final"]
    else:
        names = ["f1", "local_a", "local_b"]
    dates = local_dates(intervals)
    grid,funnel = grid_from_raw(prior,dates,roads,cfg)
    funnel["old_test_rows_excluded"] = excluded
    for name in names:
        f = cfg["folds"][name] if name != "final" else {"train":cfg["final_train"],"valid":cfg["calibration"]}
        make_fold(root,cfg,name,grid,roads,graph,local_dates(f["train"]),local_dates([f["valid"]]),funnel)


if __name__ == "__main__":
    main()



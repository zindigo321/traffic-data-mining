"""Prepare Chicago roadway graphs and causal proxy labels from the pinned API slice."""
from collections import defaultdict
from datetime import date, datetime
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_chicago import digest, write_json
from graph import select_nodes
from settings import CONFIG, WORK_ROOT

SPLITS = ("train", "validation", "test")
DIRECTION = {key: i * math.pi / 4 for i, key in enumerate(("NB", "NE", "EB", "SE", "SB", "SW", "WB", "NW"))}
GEO = ["start_latitude", "start_longitude", "end_latitude", "end_longitude"]
META = GEO + ["direction", "length", "street", "from_street", "to_street"]
FEATURES = [
    "lag10_speed_mph", "lag20_speed_mph", "lag30_speed_mph", "lag10_bus_count", "lag10_message_count",
    "lagged_speed_std_mph", "lag10_minus_lag20_mph", "length_miles", "direction_east", "direction_north",
    "decision_time_sin", "decision_time_cos", "weekend",
]
REASON = {
    0: "not_low_current_estimate", 1: "two_consecutive_low_estimates",
    2: "unconfirmed_low_estimate", 3: "insufficient_history",
    4: "invalid_speed_or_probe_support", 5: "absent_or_changed_metadata",
}


def expected_grid(cfg):
    start, end = cfg["start_date"], cfg["end_date_exclusive"]
    dates = pd.date_range(start, end, inclusive="left")
    slots = [60 * h + m for h in sorted(cfg["peak_hours"]) for m in range(0, 60, cfg["bin_minutes"])]
    times = pd.DatetimeIndex((dates.to_numpy()[:, None] + pd.to_timedelta(slots, unit="min").to_numpy()).ravel())
    train = int(len(dates) * cfg["train_fraction"])
    validation = int(len(dates) * cfg["validation_end_fraction"])
    if not 0 < train < validation < len(dates):
        raise ValueError("Need at least one calendar day in each split")
    day_split = np.searchsorted([train, validation], np.arange(len(dates)), side="right")
    return dates, slots, times, day_split


def causal_history(speed, is_weekend, day_split, minimum_days):
    """Input shape [calendar_day, time_of_day, road]; NaN indicates unusable speed."""
    mean = np.full(speed.shape, np.nan, dtype=np.float64)
    counts = np.zeros(speed.shape, dtype=np.int32)
    for weekend in (False, True):
        mask = is_weekend == weekend
        selected = speed[mask]
        train = (day_split[mask] == 0)[:, None, None]
        available = np.isfinite(selected) & train
        prior_sum = np.cumsum(np.where(available, selected, 0), axis=0)
        prior_n = np.cumsum(available, axis=0)
        prior_sum = np.concatenate([np.zeros_like(prior_sum[:1]), prior_sum[:-1]], axis=0)
        prior_n = np.concatenate([np.zeros_like(prior_n[:1]), prior_n[:-1]], axis=0)
        mean[mask] = np.divide(prior_sum, prior_n, out=np.full_like(prior_sum, np.nan),
                               where=prior_n >= minimum_days)
        counts[mask] = prior_n
    return mean.reshape(-1, speed.shape[-1]), counts.reshape(-1, speed.shape[-1])


def aligned_lag(times, split, minutes):
    index = times.get_indexer(times - pd.Timedelta(minutes=minutes))
    valid = (index >= 0) & (split[index] == split)
    return np.where(valid, index, -1)


def take_lag(values, index):
    result = values[index].copy()
    result[index == -1] = np.nan
    return result


def make_labels(speed, history, observed, lag, ratio):
    has_history = np.isfinite(history) & (history > 0)
    usable = np.isfinite(speed)
    low = usable & has_history & (speed < ratio * history)
    prev_low = low[lag] & (lag >= 0)[:, None]
    known = observed & usable & has_history
    y = np.full(speed.shape, -1, dtype=np.int8)
    y[known & ~low] = 0
    y[known & low & prev_low] = 1
    reasons = np.select(
        [~observed, ~usable, ~has_history, low & ~prev_low, y == 1],
        [5, 4, 3, 2, 1], default=0,
    ).astype(np.int8)
    return y, reasons


def endpoint_distance(a, b):
    """Local equirectangular distance in metres, suitable for a 20 m test."""
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1]) * math.cos(math.radians((a[0] + b[0]) / 2))
    return 6371000 * math.hypot(dlat, dlon)


def crossing(street, other):
    return tuple(sorted((street, other))) if street and other else ()


def valid_direction(row):
    heading = DIRECTION.get(row.direction)
    if heading is None or not all(np.isfinite(getattr(row, name)) for name in GEO):
        return False
    north = row.end_latitude - row.start_latitude
    east = (row.end_longitude - row.start_longitude) * math.cos(math.radians(row.start_latitude))
    return math.hypot(north, east) > 0 and (north * math.cos(heading) + east * math.sin(heading) > 0)


def topology(roads, cfg, graph_cfg):
    nodes = {}
    for row in roads.itertuples():
        nodes[int(row.Index)] = dict(
            entry=[row.start_latitude, row.start_longitude], exit=[row.end_latitude, row.end_longitude],
            entry_crossing=crossing(row.street, row.from_street), exit_crossing=crossing(row.street, row.to_street),
            direction=row.direction,
        )
    radius = cfg["endpoint_named_radius_m"]
    decimals = cfg["endpoint_round_decimals"]

    def rounded(p):
        return tuple(round(c, decimals) for c in p)

    neighbors = defaultdict(set)
    successors = defaultdict(list)
    edges = []
    for a, src in nodes.items():
        for b, dst in nodes.items():
            if a == b:
                continue
            reverse = endpoint_distance(src["entry"], dst["exit"]) <= radius
            if reverse:
                continue
            distance = endpoint_distance(src["exit"], dst["entry"])
            if rounded(src["exit"]) == rounded(dst["entry"]):
                evidence = "rounded_endpoint"
            elif src["exit_crossing"] and src["exit_crossing"] == dst["entry_crossing"] and distance <= radius:
                evidence = "near_endpoint_same_named_intersection"
            else:
                continue
            edges.append(dict(src=a, dst=b, distance_m=distance, evidence=evidence))
            neighbors[a].add(b)
            neighbors[b].add(a)
            successors[a].append(b)
    order = {k: i for i, k in enumerate(nodes)}
    groups = [select_nodes(k, neighbors, graph_cfg["graph_hops"], graph_cfg["max_nodes"]) for k in nodes]
    k_max = max(map(len, groups))
    lookup = np.zeros((len(nodes), k_max), dtype=np.int32)
    valid = np.zeros_like(lookup, dtype=bool)
    adj = np.zeros((len(nodes), k_max, k_max), dtype=bool)
    for i, ids in enumerate(groups):
        idx = {k: j for j, k in enumerate(ids)}
        lookup[i, :len(ids)] = [order[k] for k in ids]
        valid[i, :len(ids)] = True
        for a, ix in idx.items():
            for b in successors[a]:
                if b in idx:
                    adj[i, ix, idx[b]] = True
    if not np.array_equal(lookup[:, 0], np.arange(len(nodes))):
        raise ValueError("Target mapping failed")
    return nodes, edges, lookup, valid, adj


def canonicalize(raw, dates, cfg):
    if raw.empty:
        raise ValueError("Published query returned no rows")
    data = raw.copy()
    data["time"] = pd.to_datetime(data["time"], format="ISO8601", errors="raise")
    if (not data["time"].dt.hour.isin(cfg["peak_hours"]).all()
            or not data["time"].dt.normalize().isin(dates).all()
            or not data["segment_id"].between(1, cfg["segment_max_id"]).all()):
        raise ValueError("Input is outside the frozen public data scope")
    for col in ("street", "from_street", "to_street", "direction"):
        data[col] = data[col].fillna("").str.upper().str.split().str.join(" ")
    first = data.loc[data["time"].dt.normalize() == dates[0], ["segment_id", *META]].drop_duplicates()
    ambiguous = first.loc[first["segment_id"].duplicated(keep=False), "segment_id"].nunique()
    first = first.drop_duplicates("segment_id", keep=False).set_index("segment_id").sort_index()
    direction = [valid_direction(row) and np.isfinite(row.length) and row.length > 0 for row in first.itertuples()]
    roads = first.loc[direction]
    if roads.empty:
        raise ValueError("No valid unambiguous direction/geometry metadata on the first configured date")
    in_roster = data["segment_id"].isin(roads.index)
    stable = data.merge(roads.reset_index(), on=["segment_id", *META], how="inner", validate="many_to_one")
    stable["window"] = stable["time"].dt.floor(f"{cfg['bin_minutes']}min")
    matched_count = len(stable)
    stable = stable.drop_duplicates()
    tied = stable.duplicated(["segment_id", "time"], keep=False)
    if tied.any():
        raise ValueError("Conflicting observations at identical source timestamps")
    use = stable.sort_values(["window", "segment_id", "time"])
    use = use.drop_duplicates(["window", "segment_id"], keep="last")
    funnel = dict(
        raw_rows=len(raw), raw_distinct_roads=int(data["segment_id"].nunique()), first_date_roads=len(first) + ambiguous,
        first_date_ambiguous_roads=int(ambiguous), first_date_invalid_heading_or_length=len(first) - len(roads),
        rows_outside_frozen_roster=int((~in_roster).sum()),
        rows_changed_metadata=int(in_roster.sum()) - matched_count,
        duplicate_or_superseded_rows=matched_count - len(use), chosen_observations=len(use),
    )
    return use, roads, funnel


def build_features(speed, bus, message, times, lag, roads, step):
    a, b, c = [take_lag(speed, ix) for ix in lag]
    triplet = np.stack([a, b, c], axis=-1)
    count = np.isfinite(triplet).sum(axis=-1)
    total = np.nansum(triplet, axis=-1)
    avg = np.divide(total, count, where=count > 0, out=np.zeros_like(total))
    ss = np.nansum((triplet - avg[..., None]) ** 2, axis=-1)
    std = np.sqrt(np.divide(ss, count, where=count >= 2, out=np.full_like(ss, np.nan)))
    theta = roads["direction"].map(DIRECTION).to_numpy()
    decision = times + pd.Timedelta(minutes=step)
    phase = (decision.hour * 60 + decision.minute) * 2 * math.pi / 1440
    time = np.asarray([np.sin(phase), np.cos(phase), times.dayofweek >= 5], dtype=np.float64).T
    static = np.asarray([roads["length"], np.sin(theta), np.cos(theta)]).T
    features = np.concatenate([
        np.stack([a, b, c, take_lag(bus, lag[0]), take_lag(message, lag[0]), std, a - b], axis=-1),
        np.broadcast_to(static, (*a.shape, static.shape[1])),
        np.broadcast_to(time[:, None, :], (*a.shape, time.shape[-1])),
    ], axis=-1).astype(np.float32)
    return features, np.isfinite(a)


def summarize_arrays(path, arrays, reasons, counts, speed, history, funnel, times, dates, nodes, edges, cfg):
    y, eligible, split = [arrays[k] for k in ("y", "eligible", "split")]
    road_ids = arrays["road_ids"]
    details = []
    graph_counts = []
    overlaps = []
    parts = []
    step = cfg["bin_minutes"]
    for i, name in enumerate(SPLITS):
        mask = split == i
        sub = y[mask]
        take = eligible[mask]
        has_supervision = take & (sub >= 0)
        parts.append(set(road_ids[has_supervision.any(axis=0)]))
        for value, reason in REASON.items():
            chosen = reasons[mask] == value
            details.append(dict(split=name, reason=reason, expected_bins=int(chosen.sum()),
                                graphs=int((chosen & take).sum()), roads=int(chosen.any(0).sum())))
        row = dict(split=name, first_date=str(times[mask][0].date()), last_date=str(times[mask][-1].date()),
                   expected_bins=int(sub.size), graphs=int(take.sum()), graph_roads=int(take.any(0).sum()),
                   supervised_roads=int(has_supervision.any(0).sum()), unknown_all=int((sub == -1).sum()),
                   negative_all=int((sub == 0).sum()), positive_all=int((sub == 1).sum()),
                   unknown_graph=int((take & (sub == -1)).sum()), negative_graph=int((take & (sub == 0)).sum()),
                   positive_graph=int((take & (sub == 1)).sum()),
                   positive_dates=int(((take & (sub == 1)).any(1).reshape(-1, np.sum(times.normalize() == dates[0]))).any(1).sum()),
                   history_missing_fraction=float((~np.isfinite(history[mask])).mean()),
                   speed_missing_fraction=float((~np.isfinite(speed[mask])).mean()),
                   history_days_max=int(counts[mask].max()))
        graph_counts.append(row)
    for a, b in combinations(range(len(SPLITS)), 2):
        overlaps.append(dict(left=SPLITS[a], right=SPLITS[b], supervised_road_overlap=len(parts[a] & parts[b]),
                             time_bin_overlap=0, user_overlap=None, trajectory_overlap=None,
                             na_reason="Public segment aggregates contain no individual or trajectory identifiers"))
    observed = arrays["valid"].sum(1)
    distribution = pd.DataFrame(graph_counts)
    distribution.to_csv(path / "split_summary.csv", index=False)
    pd.DataFrame(details).to_csv(path / "label_funnel.csv", index=False)
    pd.DataFrame(overlaps).to_csv(path / "overlap.csv", index=False, na_rep="N/A")
    pd.DataFrame([dict(split=name, feature=f, missing_fraction=float((~np.isfinite(arrays["x"][split == i, :, j])).mean()))
                  for i, name in enumerate(SPLITS) for j, f in enumerate(FEATURES)]).to_csv(path / "missingness.csv", index=False)
    write_json(path / "topology.json", dict(nodes=nodes, edges=edges))
    write_json(path / "summary.json", dict(
        data_source="City of Chicago Traffic Tracker", dataset_id=cfg["dataset_id"],
        **funnel, usable_speed_bins=int(np.isfinite(speed).sum()), eligible_graphs=int(eligible.sum()),
        expected_bins=int(y.size), excluded_graphs_without_lag=int((~eligible).sum()),
        nodes=len(nodes), edges=len(edges), max_local_nodes=int(observed.max()), min_local_nodes=int(observed.min()),
        mean_local_nodes=float(observed.mean()), isolated_nodes=int((observed == 1).sum()),
        edges_named_near=sum(e["evidence"] == "near_endpoint_same_named_intersection" for e in edges),
        static_topology_as_of=str(dates[0].date()), zone="America/Chicago (CDT during this slice)",
        feature_names=FEATURES, lag_minutes=[step * j for j in (1, 2, 3)],
        time_of_day_reference="Decision at the end of a 10-minute source bin; raw time is source timestamp, feed delivery latency unavailable",
        ratio=CONFIG["congestion_ratio"], sustained_proxy="Two adjacent low-speed estimates; continuous support unavailable",
        course_equivalent=False,
    ))
    print(distribution.to_string(index=False), flush=True)
    print(json.dumps(dict(**funnel, roads=len(nodes), edges=len(edges), isolated=int((observed == 1).sum()))), flush=True)


def main():
    cfg = CONFIG["public"]
    if (cfg["bin_minutes"], cfg["lag_windows"], cfg["consecutive_low_estimates"]) != (10, 3, 2):
        raise ValueError("Public protocol currently supports 10-minute bins, 3 lags and 2-estimate confirmation")
    source = WORK_ROOT / "public_data"
    record = json.loads((source / "download.json").read_text(encoding="utf-8"))
    if record.get("file") != "chicago_fixed.csv":
        raise ValueError("Download record is incomplete or refers to an unexpected source filename")
    raw_path = source / record["file"]
    if record["status"] != "complete" or digest(raw_path) != record["sha256"]:
        raise ValueError("Public source is missing, incomplete or different from the pinned checksum")
    raw = pd.read_csv(raw_path)
    if len(raw) != record["rows"]:
        raise ValueError("Unexpected number of CSV observations")
    dates, slots, times, day_split = expected_grid(cfg)
    split = np.repeat(day_split, len(slots))
    use, roads, funnel = canonicalize(raw, dates, cfg)
    nodes, edges, lookup, valid, adj = topology(roads, cfg, CONFIG)
    index = pd.MultiIndex.from_product([times, roads.index], names=["window", "segment_id"])
    grid = use.set_index(["window", "segment_id"]).reindex(index)
    shape = (len(times), len(roads))
    observed = grid["time"].notna().to_numpy().reshape(shape)
    good = grid["speed"].between(0, cfg["max_speed_mph"]) & grid["bus_count"].gt(0) & grid["message_count"].gt(0)
    speed, bus, message = [grid[name].where(good).to_numpy(dtype=np.float64).reshape(shape)
                           for name in ("speed", "bus_count", "message_count")]
    history, days = causal_history(speed.reshape(len(dates), len(slots), len(roads)),
                                   np.asarray(dates.dayofweek >= 5), day_split, CONFIG["minimum_history_days"])
    lag = [aligned_lag(times, split, cfg["bin_minutes"] * j) for j in (1, 2, 3)]
    y, reasons = make_labels(speed, history, observed, lag[0], CONFIG["congestion_ratio"])
    x, eligible = build_features(speed, bus, message, times, lag, roads, cfg["bin_minutes"])
    data = dict(
        x=x, y=y, eligible=eligible, adj=adj, valid=valid, lookup=lookup, split=split.astype(np.int8),
        road_ids=roads.index.to_numpy(dtype=np.int64), window_minute=times.to_numpy(dtype="datetime64[m]").astype(np.int64),
    )
    prepared = WORK_ROOT / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared / "chicago.npz", **data)
    out = WORK_ROOT / "data_audit" / "chicago"
    out.mkdir(parents=True, exist_ok=True)
    summarize_arrays(out, data, reasons, days, speed, history, funnel, times, dates, nodes, edges, cfg)
    # A local sidecar allows independent checking of every label and feature without repeating the HTTP query.
    minute = grid["time"].to_numpy(dtype="datetime64[us]").reshape(shape)
    np.savez_compressed(prepared / "chicago_audit_local.npz", speed=speed, bus=bus, message=message,
                        observed=observed, history=history, history_days=days, reasons=reasons, source_time=minute)
    write_json(out / "prepared_digests.json", {name: digest(prepared / name)
              for name in ("chicago.npz", "chicago_audit_local.npz")})


if __name__ == "__main__":
    main()



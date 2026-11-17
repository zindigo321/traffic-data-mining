"""Single frozen holdout evaluation. No fitting, threshold changes, or selection occurs here."""
from collections import defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from v1 import runtime

import numpy as np
import pandas as pd
import psutil

from common import CODE, digest, json_read, json_write, now, settings
from evaluation import counts, measure, rank_score, sigmoid
from experiment import Fold, score_arrays
from prepare_chicago import fingerprint_keys, key_matrix, local_dates, make_holdout, read_npz
from v1.graph import TrainPreprocessor


def checked_contract(root,cfg):
    file = root/"evidence/freeze.json"
    lock = json_read(root/"evidence/freeze_digest.json")
    if digest(file)!=lock["sha256"]:
        raise ValueError("Freeze receipt changed")
    freeze = json_read(file)
    if (freeze["holdout_range"]!=cfg["holdout"] or freeze["holdout_observation_values_read"] or
            digest(Path(os.environ.get("GT2_CONFIG",CODE/"config.json")))!=freeze["cfg_sha256"]):
        raise ValueError("Config or holdout provenance differs")
    for filename,sha in freeze["digests"].items():
        if digest(root/filename)!=sha:
            raise ValueError(f"Frozen artifact changed: {filename}")
    for filename,sha in freeze["code_digests"].items():
        if digest(CODE/filename)!=sha:
            raise ValueError(f"Frozen code changed: {filename}")
    for row in freeze["models"]:
        for filename,sha in row["files"].items():
            if digest(root/"results/calibration"/row["job"]/filename)!=sha:
                raise ValueError(f"Frozen model changed: {row['job']}/{filename}")
    return freeze


def calibration_quality(y, score, fit):
    raw_brier = float(np.mean((score-y)**2))
    if fit is None:
        return dict(raw_score_brier=raw_brier,calibrated_brier=None,calibrated_ECE_10=None)
    p = sigmoid(fit["slope"]*score+fit["intercept"])
    buckets = np.minimum((p*10).astype(int),9)
    error = 0.
    for b in np.unique(buckets):
        group = buckets==b
        error += group.mean() * abs(float(p[group].mean()-y[group].mean()))
    return dict(raw_score_brier=raw_brier,calibrated_brier=float(np.mean((p-y)**2)),calibrated_ECE_10=error)


def groups(z,pred):
    dt,road = np.divmod(z["sample_id"],1000)
    day = dt//144
    hour = dt%144//6
    cohort = np.where(z["risk_missing"],"history_missing",np.where(z["risk"],"prior_low","ordinary"))
    y = z["y"]
    rows = []
    for dim, values in (("risk",cohort),("peak",np.where(hour<12,"AM","PM")),("road",road),("day",day)):
        # One linear grouping pass, also usable for sparse mistake groups.
        frame = pd.DataFrame(dict(group=values,n=1,positive=y,tp=(y==1)&pred,fp=(y==0)&pred,fn=(y==1)&~pred,tn=(y==0)&~pred))
        for name, row in frame.groupby("group",sort=True).sum().iterrows():
            value = str(np.datetime64(int(name),"D")) if dim=="day" else str(name)
            rows.append(dict(dimension=dim,group=value,**{k:int(v) for k,v in row.items()}))
    return rows


def blocks(y,score,pred,days,weights):
    """Exact tie-aware AP under integer day weights; dates are shared by all runs."""
    order = np.argsort(-score,kind="stable")
    ranked = score[order]
    end = np.r_[np.flatnonzero(ranked[1:]!=ranked[:-1]),len(y)-1]
    day = days[order]
    positives = y[order]
    day_stats = np.stack([np.bincount(days,weights=m,minlength=weights.shape[1])
                          for m in ((y==1)&pred,(y==0)&pred,(y==1)&~pred)])
    tp,fp,fn = (weights @ day_stats.T).T
    den = 2*tp+fp+fn
    f1 = np.divide(2*tp,den,out=np.full(len(den),np.nan),where=den>0)
    ap = np.full(len(weights),np.nan)
    for i, block in enumerate(weights):
        w = block[day]
        t = np.cumsum(w*positives,dtype=np.int64)[end]
        n = np.cumsum(w,dtype=np.int64)[end]
        if not 0 < t[-1] < n[-1]:
            continue
        precision = np.divide(t,n,out=np.zeros(len(t)),where=n>0)
        ap[i] = float(np.dot(np.diff(np.r_[0,t]),precision)/t[-1])
    return np.column_stack([ap,f1])


def bootstrap(root,cfg,records):
    out = root/"results/holdout"
    dates = local_dates([cfg["holdout"]])
    base = dates[0].astype(np.int64)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    weights = rng.multinomial(len(dates),np.full(len(dates),1/len(dates)),size=cfg["bootstrap_reps"])
    rep = {}
    for row in records:
        name = row["evaluation_id"]
        z = read_npz(out/f"{name}.npz")
        day = z["sample_id"]//1000//144-base
        rep[name] = blocks(z["y"],z["score"],z["pred"],day,weights)
    np.savez_compressed(out/"bootstrap_replicates.npz", dates=dates.astype(np.int64), weights=weights,**rep)
    by_group = defaultdict(list)
    for row in records:
        by_group[row["comparison"]].append(rep[row["evaluation_id"]])
    means = {group:np.mean(vals,axis=0) for group,vals in by_group.items()}
    result = []

    def add(group,replicates):
        for j, metric in enumerate(("AP","F1")):
            col = replicates[:,j]
            finite = col[np.isfinite(col)]
            ci = np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None]
            result.append(dict(comparison=group,metric=metric,lower_95=ci[0],upper_95=ci[1],
                               valid_reps=len(finite),total_reps=len(weights)))

    for group,replicates in means.items():
        add(group,replicates)
    chosen = next(row["comparison"] for row in records if row["selected"])
    add("selected_minus_V0_max", means[chosen]-means["V0_max_raw"])
    pd.DataFrame(result).to_csv(out/"date_block_intervals.csv",index=False,na_rep="N/A")


def name_group(row):
    if row["algo"]!="gt":
        return row["algo"]
    if row["central"]:
        return f"{row['recipe']}_{row['pool']}_center"
    return f"{row['recipe']}_{row['pool']}_{row['mode']}"


def main():
    root,cfg = settings()
    start = time.perf_counter()
    evidence = root/"evidence"
    if not (evidence/"freeze.json").is_file():
        print("Holdout blocked; inspect evidence/holdout_status.json",flush=True)
        return
    if (evidence/"holdout_unlock.json").exists():
        raise RuntimeError("The formal holdout was already opened. Use independent verification to recompute.")
    freeze = checked_contract(root,cfg)
    out = root/"results/holdout"
    out.mkdir(parents=True,exist_ok=False)
    json_write(evidence/"holdout_unlock.json",dict(
        unlocked_utc=now(),freeze_sha256=digest(evidence/"freeze.json"), scope=freeze["holdout_range"],
        parsed_values_before_this_receipt=False, fitting_allowed=False, selection_allowed=False,
    ))
    arr = make_holdout(root,cfg,freeze)
    cover = json_read(evidence/"fold_holdout.json")["parts"][0]
    grid = key_matrix(arr["minute"],arr["road_ids"])
    known = arr["eligible"] & (arr["y"]>=0)
    known_keys = grid[known]
    np.savez_compressed(out/"support.npz",sample_id=grid.ravel(),eligible=arr["eligible"].ravel(),y=arr["y"].ravel(),
                        known_ids=known_keys)
    json_write(evidence/"holdout_support_receipt.json",dict(
        recorded_utc=now(),prepared_sha256=digest(root/"prepared/holdout.npz"),
        expected_grid_key_sha256=fingerprint_keys(grid),known_ids_sha256=fingerprint_keys(known_keys),support=len(known_keys),
        positive=cover["positive"],negative=cover["negative"],true_events=cover["true_events"],calendar_days=cover["total_days"],
    ))
    if not (cover["positive"] and cover["negative"]):
        json_write(evidence/"holdout_status.json",dict(status="insufficient_test_classes",score_goal="not_evaluable"))
        return
    process = psutil.Process()
    rows, error_rows = [], []
    previous = None
    for model in freeze["models"]:
        job = model["job"]
        directory = root/"results/calibration"/job
        t = time.perf_counter()
        opts = json_read(directory/"options.json")
        if previous != job:
            fold = Fold(arr,opts["recipe"])
            if opts["algo"]!="rule":
                fold.configure(TrainPreprocessor.load(directory/"preprocessor.npz"))
            predict_ids = np.flatnonzero(fold.eligible & (fold.phase==1))
            all_eligible = score_arrays(fold,opts,cfg,directory,ids=predict_ids)
            keep = all_eligible["y"] >= 0
            z = {key:vals[keep] for key,vals in all_eligible.items()}
        previous = job
        np.testing.assert_array_equal(z["sample_id"],known_keys)
        p = rank_score(z["probability"],z["risk"],model["mode"]=="routed")
        cutoff = model["threshold"]
        pred = p >= cutoff
        ident = f"{job}_{model['mode']}"
        info = dict(evaluation_id=ident,comparison=name_group(model),**{k:v for k,v in model.items() if k not in ("files","threshold","platt")})
        measure_row = measure(z["y"],p,cutoff,z["sample_id"],cover)
        quality = calibration_quality(z["y"],p,model["platt"])
        rows.append(dict(**info,**measure_row,**quality,score_dtype=str(p.dtype),logit_dtype=str(z["logit"].dtype),
                         probability_dtype=str(z["probability"].dtype),seconds=time.perf_counter()-t,
                         process_RSS_bytes=process.memory_info().rss))
        np.savez_compressed(out/f"{ident}.npz",**z,score=p,threshold=np.float64(cutoff),pred=pred)
        error_rows.extend(dict(evaluation_id=ident,**g) for g in groups(z,pred))
        print({k:rows[-1][k] for k in ("evaluation_id","AP","f1","fp","tp","true_events")},flush=True)
    json_write(out/"metrics.json",rows)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False,na_rep="N/A")
    pd.DataFrame(error_rows).to_csv(out/"error_groups.csv",index=False)
    table = pd.DataFrame(rows)
    numeric = ["AP","f1","ROC_AUC","precision","recall","accuracy","tp","fp","fn","tn","fp_per_day",
               "event_precision","event_recall","event_f1","raw_score_brier","calibrated_brier","calibrated_ECE_10"]
    table.groupby("comparison")[numeric].agg(["mean","std","count"]).to_csv(out/"aggregate.csv",na_rep="N/A")
    chosen = table[table.selected].set_index("seed")
    ref = table[table.comparison=="V0_max_raw"].set_index("seed")
    paired = chosen[numeric] - ref[numeric]
    paired.to_csv(out/"paired_vs_v0.csv",na_rep="N/A")
    bootstrap(root,cfg,rows)
    ap,f1 = float(chosen.AP.mean()),float(chosen.f1.mean())
    goal = dict(mean_AP=ap,mean_positive_F1=f1)
    for g in cfg["goals"]:
        goal[f"target_{g:.2f}"] = dict(reached=ap>=g and f1>=g,AP_gap=max(0.,g-ap),F1_gap=max(0.,g-f1))
    json_write(evidence/"holdout_status.json",dict(status="complete",chosen=chosen.comparison.iloc[0],**goal,
                formal_variant_count=len(rows),common_support=len(known_keys),calibration_and_selection_frozen=True,
                positive=cover["positive"],negative=cover["negative"],true_events=cover["true_events"],
                seconds=time.perf_counter()-start,completed_utc=now()))
    print(json_read(evidence/"holdout_status.json"),flush=True)


if __name__ == "__main__":
    main()




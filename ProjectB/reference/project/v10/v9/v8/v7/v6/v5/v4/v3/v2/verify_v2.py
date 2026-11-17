"""Independently recompute archived scores, time splits, frozen choices, and sampled inference."""
import argparse
from datetime import datetime
from pathlib import Path
import math
import os

from v1 import runtime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CODE, digest, json_read, json_write, settings
from experiment import Fold, score_arrays
from prepare_chicago import fingerprint_keys, local_dates, read_npz
from v1.graph import TrainPreprocessor


def same(actual, saved, label):
    if actual is None and saved is None:
        return
    if actual is None or saved is None or not math.isclose(actual,saved,rel_tol=1e-10,abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {saved}")


def independent(y,p,t):
    if not np.isin(y,[0,1]).all():
        raise ValueError("Unknown truth in scored support")
    selected = p >= t
    tp, fp = int(y[selected].sum()), int(selected.sum()-y[selected].sum())
    pos = int(y.sum())
    neg = len(y)-pos
    fn,tn = pos-tp,neg-fp
    ap = cutoff = None
    if pos and neg:
        order = np.argsort(-p,kind="stable")
        ranked = p[order]
        end = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]),len(ranked)-1]
        true = np.cumsum(y[order])[end]
        ap = float((np.diff(np.r_[0,true])*true/(end+1)).sum()/pos)
        f = 2*true/(pos+end+1)
        winners = ranked[end][np.isclose(f,f.max(),rtol=0,atol=1e-12)]
        cutoff = float(min(winners,key=lambda z:(abs(z-.5),z)))
    return dict(AP=ap, ROC_AUC=float(roc_auc_score(y,p)) if pos and neg else None, tp=tp,fp=fp,fn=fn,tn=tn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/pos if pos else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,accuracy=(tp+tn)/len(y),optimal_cutoff=cutoff)


def verify_row(z, row, label, optimize=False):
    t = float(z["threshold"])
    same(t,row["threshold"],label+" cutoff")
    np.testing.assert_array_equal(z["pred"],z["score"]>=t)
    calc = independent(z["y"],z["score"],t)
    for k,v in calc.items():
        if k!="optimal_cutoff":
            same(v,row[k],label+" "+k)
    if optimize:
        same(calc["optimal_cutoff"],t,label+" validation rule")
    if len(np.unique(z["sample_id"])) != len(z["sample_id"]):
        raise ValueError("Duplicate sample key")
    if str(z["logit"].dtype)!=row["logit_dtype"] or str(z["probability"].dtype)!=row["probability_dtype"]:
        raise ValueError("Stored dtype is inconsistent")


def dates_for_ids(keys):
    return (keys//1000//144).astype("datetime64[D]")


def check_sampling(artifact,cfg):
    rows = []
    signatures = {}
    datasets = [("final",artifact/"results/calibration")]
    datasets += [(f.name,f) for f in (artifact/"results/dev").iterdir() if f.is_dir()]
    for fold,parent in datasets:
        if not parent.is_dir():
            continue
        ranges = cfg["final_train"] if fold=="final" else cfg["folds"][fold]["train"]
        permitted = local_dates(ranges)
        for f in sorted(parent.iterdir()):
            if not (f/"sampling_ids.npz").is_file():
                continue
            stat = json_read(f/"sampling.json")
            z = read_npz(f/"sampling_ids.npz")
            for k in ("main_ids","aux_ids"):
                if not np.isin(dates_for_ids(z[k]),permitted).all() or len(np.unique(z[k]))!=len(z[k]):
                    raise AssertionError("Sampling outside training dates or duplicate ID")
            if not np.isin(z["main_labels"],[0,1]).all() or not np.isin(z["aux_labels"],[0,1]).all():
                raise AssertionError("Unknown label entered either target loss")
            if (len(z["main_ids"]) != stat["main_positive"]+stat["sampled_negative"] or
                    int(z["main_prior_low"][z["main_labels"]==0].sum()) != stat["sampled_hard"]):
                raise AssertionError("Wrong negative sample budget or hard counts")
            key = (fold,stat["strategy"])
            sha = fingerprint_keys(z["main_ids"])
            if key in signatures and sha!=signatures[key]:
                raise AssertionError("Matched models used different main sample IDs")
            signatures[key] = sha
            rows.append(dict(fold=fold,job=f.name,main=len(z["main_ids"]),aux=len(z["aux_ids"])))
    return rows


def check_development(artifact,cfg):
    rows = []
    for top in (artifact/"results/dev",artifact/"results/calibration"):
        for path in sorted(top.rglob("status.json")):
            state = json_read(path)
            if state["status"]!="complete":
                continue
            assessment = (cfg["calibration"] if state["fold"]=="final" else cfg["folds"][state["fold"]]["valid"])
            for mode,metrics in state["scored"].items():
                z = read_npz(path.parent/f"assessment_{mode}.npz")
                assert np.isin(dates_for_ids(z["sample_id"]),local_dates([assessment])).all()
                verify_row(z,metrics,str(path.parent.name)+mode,optimize=state["algo"]!="rule")
                rows.append(dict(fold=state["fold"],job=path.parent.name,mode=mode,n=len(z["y"])))
    return rows


def verify_selection(artifact,cfg,freeze):
    sc = json_read(artifact/"results/screen_selection.json")
    confirm = json_read(artifact/"results/development_selection.json")
    candidates = pd.read_csv(artifact/"results/screen_metrics.csv")
    folds = sc["folds"]
    def key(ap, f1, sd):
        return round(float(ap),6),round(float(f1),6),-round(float(sd),6)
    raw = candidates[candidates.route=="raw"].groupby("recipe")[["AP","f1"]]
    score = {r:key(g.AP.mean(),g.f1.mean(),g.AP.std(ddof=1)) for r,g in raw}
    recipe = max(["V1","V2","V3"],key=score.get)
    assert recipe == sc["chosen_enhancement"] == confirm["recipe"]
    routed = candidates[(candidates.recipe==recipe)&(candidates.route=="routed")]
    mode = "routed" if len(routed) and key(routed.AP.mean(),routed.f1.mean(),routed.AP.std(ddof=1))>score[recipe] else "raw"
    assert mode == sc["route"] == confirm["route"]
    table = pd.read_csv(artifact/"results/confirmation_metrics.csv")
    gt = table[(table.algo=="gt")&(~table.central)]
    key_pool = {}
    for pool in cfg["pools"]:
        scores = gt[(gt.pool==pool)&(gt.recipe==recipe)]
        by_fold = scores.groupby("fold")[["AP","f1"]].mean()
        assert len(scores)==len(folds)*len(cfg["seeds"]) and set(by_fold.index)==set(folds)
        key_pool[pool] = key(by_fold.AP.mean(),by_fold.f1.mean(),by_fold.AP.std(ddof=1))
    assert max(cfg["pools"],key=key_pool.get)==confirm["pooling"]
    if freeze is not None:
        assert confirm == freeze["selection"] and confirm["holdout_labels_used"] is False
        assert datetime.fromisoformat(confirm["completed_utc"]) < datetime.fromisoformat(freeze["frozen_utc"])
        for item in freeze["models"]:
            path = artifact/"results/calibration"/item["job"]
            state = json_read(path/"status.json")
            assert datetime.fromisoformat(state["completed_utc"]) <= datetime.fromisoformat(freeze["frozen_utc"])
            for fname,sha in item["files"].items():
                assert digest(path/fname)==sha
            same(item["threshold"],state["scored"][item["mode"]]["threshold"],"freeze threshold")
        selected = [r for r in freeze["models"] if r["selected"]]
        assert sorted(r["seed"] for r in selected)==cfg["seeds"]
        assert all(r["recipe"]==recipe and r["pool"]==confirm["pooling"] and r["mode"]==mode for r in selected)
    return dict(recipe=recipe,pooling=confirm["pooling"],route=mode,folds=folds)


def replay_artifact(artifact,root,cfg,freeze,rows):
    prep = root/"prepared/holdout.npz"
    expected = json_read(artifact/"evidence/holdout_support_receipt.json")["prepared_sha256"]
    if digest(prep)!=expected:
        raise ValueError("External holdout array is not the verified prepared input")
    arr = read_npz(prep)
    rng = np.random.default_rng(42011)
    all_ids = np.flatnonzero(arr["eligible"].ravel())
    size = cfg["eval_batch_size"]
    chosen = np.sort(rng.choice(len(all_ids)//size,3,replace=False))
    batches = np.concatenate([all_ids[k*size:(k+1)*size] for k in chosen])
    replayed = {}
    out = artifact/"results/holdout"
    for row in rows:
        name = row["job"]
        path = artifact/"results/calibration"/name
        if name not in replayed:
            opts = json_read(path/"options.json")
            fold = Fold(arr,opts["recipe"])
            if opts["algo"]!="rule":
                fold.configure(TrainPreprocessor.load(path/"preprocessor.npz"))
            zz = score_arrays(fold,opts,cfg,path,ids=batches)
            keep = zz["y"]>=0
            replayed[name] = {k:v[keep] for k,v in zz.items()}
        sample = replayed[name]
        z = read_npz(out/f"{row['evaluation_id']}.npz")
        keys = sample["sample_id"]
        idx = np.searchsorted(z["sample_id"],keys)
        np.testing.assert_array_equal(z["sample_id"][idx],keys)
        np.testing.assert_array_equal(z["y"][idx],sample["y"])
        np.testing.assert_allclose(z["probability"][idx],sample["probability"],rtol=1e-6,atol=2e-7)
        np.testing.assert_allclose(z["logit"][idx],sample["logit"],rtol=1e-6,atol=2e-6)
        a = sample["probability"].astype(np.float64)
        if row["mode"]=="routed":
            a = np.where(sample["risk"],.51+.49*a,.49*a)
        np.testing.assert_array_equal(a>=row["threshold"],z["pred"][idx])
        row["replayed_rows"] = len(keys)
        row["replay_max_abs_probability_error"] = float(np.max(np.abs(a-z["score"][idx])))
    return len(replayed)


def verify(artifact,cfg,replay_root=None):
    sampling = check_sampling(artifact,cfg)
    folds = check_development(artifact,cfg)
    freeze_file = artifact/"evidence/freeze.json"
    if not freeze_file.is_file():
        selection = verify_selection(artifact,cfg,None)
        return dict(status="PASS",scope="development_only",sampling=sampling,fold_checks=folds,selection=selection)
    freeze = json_read(freeze_file)
    unlock = json_read(artifact/"evidence/holdout_unlock.json")
    for name,expected in freeze["digests"].items():
        assert digest(artifact/name)==expected
    for name,expected in freeze["code_digests"].items():
        assert digest(CODE/name)==expected
    sha = digest(freeze_file)
    assert sha==unlock["freeze_sha256"]==json_read(artifact/"evidence/freeze_digest.json")["sha256"]
    assert digest(Path(os.environ.get("GT2_CONFIG",CODE/"config.json")))==freeze["cfg_sha256"]
    assert datetime.fromisoformat(freeze["frozen_utc"]) <= datetime.fromisoformat(unlock["unlocked_utc"])
    assert not any(unlock[k] for k in ("parsed_values_before_this_receipt","fitting_allowed","selection_allowed"))
    selection = verify_selection(artifact,cfg,freeze)
    out = artifact/"results/holdout"
    support = read_npz(out/"support.npz")
    eligible = support["eligible"]
    known = eligible & (support["y"]>=0)
    keys = support["sample_id"][known]
    np.testing.assert_array_equal(keys,support["known_ids"])
    assert fingerprint_keys(support["sample_id"])==freeze["expected_grid_key_sha256"]
    assert np.isin(dates_for_ids(keys),local_dates([cfg["holdout"]])).all()
    assert fingerprint_keys(keys)==json_read(artifact/"evidence/holdout_support_receipt.json")["known_ids_sha256"]
    assert fingerprint_keys(support["sample_id"][eligible])==json_read(
        artifact/"evidence/holdout_features_receipt.json")["eligible_ids_sha256"]
    rows = json_read(out/"metrics.json")
    frozen_models = {(r["job"],r["mode"]):r for r in freeze["models"]}
    assert len(rows)==len(frozen_models)
    for row in rows:
        z = read_npz(out/f"{row['evaluation_id']}.npz")
        np.testing.assert_array_equal(z["sample_id"],keys)
        np.testing.assert_array_equal(z["y"],support["y"][known])
        same(row["threshold"],frozen_models[row["job"],row["mode"]]["threshold"],"frozen holdout threshold")
        verify_row(z,row,row["evaluation_id"])
    replayed = replay_artifact(artifact,replay_root,cfg,freeze,rows) if replay_root else 0
    return dict(status="PASS",scope="formal_holdout",sampling=sampling,fold_checks=folds,selection=selection,
                verified_scores=len(rows),test_support=len(keys),verified_model_weights=replayed,
                sampled=[{k:v for k,v in r.items() if k in ("evaluation_id","replayed_rows","replay_max_abs_probability_error")}
                         for r in rows] if replayed else [],
                chronology="development choice and calibration frozen before opening holdout observation values")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    root,cfg = settings()
    result = verify(a.artifacts or root,cfg,a.replay_root)
    json_write(a.output,result)
    print({k:v for k,v in result.items() if k not in ("sampling","fold_checks","sampled")},flush=True)


if __name__=="__main__":
    main()




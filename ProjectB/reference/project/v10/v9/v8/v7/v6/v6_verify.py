"""Audit fixed cohorts, calibration thresholds, and the two matched comparisons."""
import argparse
from datetime import datetime
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from v6_io import V5,load,now,save,settings,sha
from v6_lock import verify_lock
from v6_register import verify_registration
from v6_train import configurations,training_budget,where
from v6_stats import compare
sys.path.insert(0,str(V5))
from v5_prepare import fingerprint_keys,read_npz
from v5_train import matched_indices,study_for
from v5_verify import replay_batch,validate_metrics,check_support,equal
from v4_matrix import average
from evaluation import threshold


def calibrated(root,cfg,lock,replay_root):
    dev=load(root/"evidence/prepared_development.json")
    source=load(root/"evidence/v5_prepared_development.json")
    assert dev["hashes"]==source["hashes"] and dev["history_sha256"]==source["history_sha256"]
    cover=load(root/"evidence/fold_final.json")["parts"][1]
    embargo=[cfg["prior_holdout"],cfg["holdout"]]
    first_sample=first_eval=None
    cal=load(root/"results/calibration_receipt.json")
    assert not cal["V5_holdout_used"] and not cal["V5_weights_loaded"]
    assert cal["registration_sha256"]==sha(root/"evidence/registration.json")
    statuses={(x["recipe"],x["pool"],x["seed"]):x for x in cal["rows"]}
    expected=configurations(cfg)
    assert len(statuses)==len(expected)==12
    data=None
    if replay_root:
        path=replay_root/"prepared/final.npz"
        assert sha(path)==dev["hashes"]["final"]
        data=read_npz(path)
    results=[]
    diffs=[]
    for job in expected:
        recipe,pool,seed=job
        folder=where(root,*job)
        opt=load(folder/"options.json")
        stats=load(folder/"status.json")
        z=read_npz(folder/"assessment.npz")
        sample=read_npz(folder/"sampling_ids.npz")
        assert stats==statuses[job]
        assert opt==dict(fold="final",recipe=recipe,pool=pool,seed=seed,fixed_epoch=cfg["epochs"][f"{recipe}_{pool}"])
        if lock is not None:
            assert stats["completed_utc"]<lock["frozen_utc"]
            assert fingerprint_keys(z["sample_id"])==lock["calibration_known_sha256"]
        validate_metrics(z,stats,load(folder/"matched_recall.json"),cfg,cover)
        assert float(z["threshold"])==threshold(z["y"],z["score"])
        assert equal(stats["fp_per_day"],stats["fp"]/cover["total_days"])
        check_support(z["sample_id"],[cfg["calibration"]],embargo)
        check_support(sample["sample_id"],[[cfg["train_start"],cfg["calibration"][0]]],embargo)
        assert sample["risk"].all() and np.isin(sample["label"],[0,1]).all()
        if first_sample is None:
            first_sample=sample
            first_eval={k:z[k] for k in ("sample_id","y","risk","risk_missing")}
            if data is not None:
                study=study_for(data,recipe)
                drawn,_,_=matched_indices(study,cfg)
                np.testing.assert_array_equal(study.keys(drawn),sample["sample_id"])
        for key,val in first_sample.items():
            np.testing.assert_array_equal(sample[key],val)
        for key,val in first_eval.items():
            np.testing.assert_array_equal(z[key],val)
        receipt=load(folder/"sampling.json")
        assert stats["updates"]==training_budget()*math.ceil(receipt["draws_per_epoch"]/cfg["batch_size"])
        assert stats["weight_sha256"]==sha(folder/"weight.pt")
        if data is not None:
            diffs.append(replay_batch(study_for(data,recipe),z,cfg,opt,folder,False))
        results.append(stats)
    return results,diffs


def heldout(root,cfg,lock,replay_root):
    out=root/"results/holdout"
    support=read_npz(out/"support.npz")
    receipt=load(root/"evidence/holdout_support.json")
    ids=support["sample_id"]
    eligible=support["eligible"]
    known=eligible&(support["y"]>=0)
    check_support(ids,[cfg["holdout"]],[cfg["prior_holdout"]])
    assert np.isin(support["y"],[-1,0,1]).all()
    assert len(ids)==lock["expected_windows"]==receipt["expected"]
    assert fingerprint_keys(ids)==lock["expected_grid_sha256"]==receipt["all_sha256"]
    assert fingerprint_keys(ids[eligible])==receipt["eligible_sha256"]
    assert fingerprint_keys(ids[known])==receipt["known_sha256"]
    np.testing.assert_array_equal(ids[eligible],support["eligible_ids"])
    np.testing.assert_array_equal(ids[known],support["known_ids"])
    cover=load(root/"evidence/fold_holdout.json")["parts"][0]
    assert int((eligible & (support["y"]<0)).sum())==cover["unknown_eligible"]
    assert int(eligible.sum())==cover["eligible"]
    data=None
    if replay_root:
        file=replay_root/"prepared/holdout.npz"
        assert sha(file)==receipt["prepared_sha256"]
        data=read_npz(file)
        for k in ("y","eligible","risk"):
            np.testing.assert_array_equal(data[k].ravel(),support[k])
    results,scores,points,diffs={},{},{},[]
    for spec in lock["models"]:
        key=tuple(spec[k] for k in ("recipe","pool","seed"))
        file=out/spec["job"]
        raw=read_npz(file/"all_eligible.npz")
        for col in ("sample_id","y","risk"):
            np.testing.assert_array_equal(raw[col],support[col][eligible])
        np.testing.assert_array_equal(raw["pred"],raw["score"]>=raw["threshold"])
        valid=raw["y"]>=0
        values={k:(v if k=="threshold" else v[valid]) for k,v in raw.items()}
        np.testing.assert_array_equal(values["sample_id"],support["known_ids"])
        stats=load(file/"status.json")
        pts=load(file/"matched_recall.json")
        calib=root/"results/calibration"/spec["job"]
        assert spec["threshold"]==float(raw["threshold"])==float(read_npz(calib/"assessment.npz")["threshold"])
        validate_metrics(values,stats,pts,cfg,cover)
        assert equal(stats["fp_per_day"],stats["fp"]/cover["total_days"])
        if data is not None:
            diffs.append(replay_batch(study_for(data,key[0]),raw,cfg,load(calib/"options.json"),calib,True))
        results[key],scores[key],points[key]=stats,values,pts
    result=load(root/"evidence/status.json")
    expected=load(out/"aggregates.json")
    for group,stat in expected.items():
        assert stat==average([row for (r,p,s),row in results.items() if f"{r}_{p}"==group])
    primary=expected["_".join(lock["primary"])]
    assert result["AP"]==primary["AP"] and result["F1"]==primary["f1"]
    assert result["goal_reached"]==(primary["AP"]>=cfg["goal"] and primary["f1"]>=cfg["goal"])
    rows,intervals,comp=compare(cfg,results,scores,points)
    for name,source in (("paired",rows),("date_blocks",intervals)):
        observed=pd.read_csv(out/f"{name}.csv",float_precision="round_trip")
        pd.testing.assert_frame_equal(observed,pd.DataFrame(source),check_exact=False,atol=1e-13,rtol=1e-12)
    assert comp==load(out/"comparisons.json")==result["comparisons"]
    assert result["stable_vs_both"]==all(x["stable"] for x in comp.values())
    return results,diffs


def run(root,output,cfg,replay_root):
    registered=verify_registration(root,cfg)
    state=load(root/"evidence/status.json")
    gate=load(root/"evidence/date_gate.json")
    assert state["Chicago_formal"]=="blocked" and gate["observation_values_read"] is False
    lock=verify_lock(root,cfg) if (root/"evidence/freeze.json").is_file() else None
    fits,replays=calibrated(root,cfg,lock,replay_root)
    results,actual={},[]
    if state["status"]=="complete":
        opened=load(root/"evidence/unlock.json")
        features=load(root/"evidence/holdout_features.json")
        times=[gate["at_utc"],registered["at_utc"],load(root/"evidence/prepared_development.json")["prepared_utc"],
               load(root/"results/calibration_receipt.json")["at_utc"],lock["frozen_utc"],opened["unlocked_utc"],
               features["at_utc"],state["finished_utc"]]
        chronology=list(map(datetime.fromisoformat,times))
        assert chronology==sorted(set(chronology))
        assert opened["freeze_sha256"]==sha(root/"evidence/freeze.json") and not opened["old_V5_holdout_parsed"]
        assert features["target_truth_built"] is False and state["V5_holdout_used"] is False
        results,actual=heldout(root,cfg,lock,replay_root)
    else:
        assert state["AP"] is None and state["F1"] is None and not (root/"results/holdout").exists()
    summary=dict(status="PASS",NYC_formal=state["NYC_formal"],calibrations=len(fits),heldout_runs=len(results),
        calibration_replayed=len(replays),heldout_replayed=len(actual),paired_FP_and_stability_recomputed=bool(results),
        checked_identical_support=True,checked_old_test_exclusion=True,checked_frozen_thresholds=True,
        max_probability_difference=max(replays+actual,default=None),at_utc=now())
    save(output/"evidence"/("verification_replay.json" if replay_root else "verification_offline.json"),summary)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root,cfg=settings()
    run(a.artifacts or root,root,cfg,a.replay_root)

"""Finalize calibration contract before sealed NYC observations can be read."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from v5_io import CODE,CONFIG,V4,load,now,save,settings,sha
from v5_download import verify_registration
from v5_prepare import fingerprint_keys,key_matrix,local_dates,minute_values,read_npz,times_for,verify_amendment
from v5_matrix import folder

LOCK_FILES=("evidence/registration.json","evidence/amendment.json","evidence/dictionary_receipt.json",
            "evidence/prior_inputs.json","evidence/download_geometry.json","evidence/download_development.json",
            "evidence/download_holdout_sealed.json","evidence/graph.json","evidence/nyc_roads.csv","evidence/nyc_edges.csv",
            "evidence/prepared_development.json","prepared/topology.npz","prepared/frozen_history.npz",
            "results/screen.json","results/screen.csv","results/selection.json","results/confirmation.csv",
            "results/calibration_receipt.json","synthetic/status.json")
MODEL_FILES=("weight.pt","preprocessor.npz","sampling_ids.npz","sampling.json","options.json",
             "assessment.npz","matched_recall.json","curve.csv","status.json")
CORE=("v5_io.py","v5_download.py","v5_graph.py","v5_prepare.py","v5_train.py","v5_matrix.py","v5_lock.py","v5_eval.py")


def transport(root,cfg):
    held=load(root/"evidence/download_holdout_sealed.json")
    if (held["interval"]!=cfg["holdout"] or held["status"]!="complete" or not held["transport_only"]
            or held["observation_values_parsed"] is not False or held["rows"]<=0):
        raise ValueError("Transport-only scope invalid")
    for row in held["files"]:
        path=Path(held["directory"])/row["file"]
        if path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]:
            raise ValueError("Sealed source changed")
    return held


def freeze(root,cfg):
    ev=root/"evidence"
    if any((root/name).exists() for name in ("evidence/freeze.json","evidence/unlock.json","prepared/holdout.npz","results/holdout")):
        raise PermissionError("External run is already frozen or opened")
    reg=verify_registration(root,cfg)
    verify_amendment(root)
    held=transport(root,cfg)
    assert reg["at_utc"] <= min(r["first_requested_utc"] for r in held["files"])
    assert load(root/"synthetic/status.json")["status"]=="PASS"
    choice=load(root/"results/selection.json")
    cal=load(root/"results/calibration_receipt.json")
    if cal["calendar"]!=cfg["calibration"] or cal["holdout_used"] or choice["holdout_used"]:
        raise ValueError("External test leaked into selection/calibration")
    if cal["choice_sha256"]!=sha(root/"results/selection.json") or cal["data_sha256"]!=sha(root/"prepared/final.npz"):
        raise ValueError("Calibration data/choice changed")
    assert choice["at_utc"]<cal["at_utc"]
    recipe=choice["recipe"]
    specs=[(recipe,p,s) for p,s in product(cfg["pools"],cfg["seeds"])]+[("B","mean",s) for s in cfg["seeds"]]
    roster=read_npz(root/"prepared/topology.npz")["road_ids"]
    minutes=minute_values(times_for(local_dates([cfg["holdout"]]),cfg))
    keys=key_matrix(minutes,roster)
    models=[]
    signatures=set()
    for r,p,s in specs:
        path=folder(root,"final",r,p,s)
        result=load(path/"status.json")
        predictions=read_npz(path/"assessment.npz")
        signatures.add(fingerprint_keys(predictions["sample_id"]))
        assert result["fixed_epoch"]==choice["epochs"][f"{r}_{p}"]
        assert float(predictions["threshold"])==result["threshold"] and np.isfinite(result["threshold"])
        models.append(dict(recipe=r,pool=p,seed=s,job=path.name,threshold=result["threshold"],
                           selected=(r,p)==(recipe,choice["pool"]),hashes={k:sha(path/k) for k in MODEL_FILES}))
    if len(signatures)!=1 or len(models)!=12:
        raise ValueError("Model count or calibration support mismatch")
    core=[CODE/f for f in CORE]+sorted(V4.rglob("*.py"))
    contract=dict(frozen_utc=now(),config_sha256=sha(CONFIG),holdout=cfg["holdout"],calibration=cfg["calibration"],
                  choice=choice,expected_grid_sha256=fingerprint_keys(keys),expected_windows=int(keys.size),
                  calibration_known_sha256=signatures.pop(),code_hashes={p.relative_to(CODE).as_posix():sha(p) for p in core},
                  files={f:sha(root/f) for f in LOCK_FILES},models=models,holdout_values_read=False,
                  threshold_rule="Positive F1 on complete calibration only; tie nearest 0.5, then lower",
                  support_contract="All eligible prior-speed 10-minute bins; metrics use all known with same fixed grid")
    save(ev/"freeze.json",contract)
    save(ev/"freeze_sha256.json",dict(sha256=sha(ev/"freeze.json"),frozen_utc=contract["frozen_utc"]))
    print(dict(frozen_utc=contract["frozen_utc"],weights=len(models),expected=len(keys)*len(roster),test_parsed=False),flush=True)


def verify_lock(root,cfg):
    ev=root/"evidence"
    receipt=load(ev/"freeze_sha256.json")
    contract=load(ev/"freeze.json")
    if (sha(ev/"freeze.json")!=receipt["sha256"] or receipt["frozen_utc"]!=contract["frozen_utc"]
            or contract["holdout"]!=cfg["holdout"] or contract["config_sha256"]!=sha(CONFIG)
            or contract["holdout_values_read"] is not False):
        raise ValueError("Frozen contract changed")
    verify_registration(root,cfg)
    verify_amendment(root)
    for file,h in contract["files"].items():
        if sha(root/file)!=h:
            raise ValueError(f"Changed frozen input {file}")
    for file,h in contract["code_hashes"].items():
        if sha(CODE/file)!=h:
            raise ValueError(f"Changed frozen code {file}")
    for model in contract["models"]:
        for file,h in model["hashes"].items():
            if sha(root/"results/calibration"/model["job"]/file)!=h:
                raise ValueError(f"Changed model {model['job']}/{file}")
    return contract


if __name__=="__main__":
    freeze(*settings())


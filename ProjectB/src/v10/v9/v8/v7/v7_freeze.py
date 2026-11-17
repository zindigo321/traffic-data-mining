"""Freeze all calibration artifacts and keep the formal cohort blocked."""
from pathlib import Path

import numpy as np

from v7_io import CODE,CONFIG,V6,load,now,save,settings,sha
from v7_check import registered
from v7_prepare import fingerprint_keys,key_matrix,local_dates,minute_values,times_for
from v7_train import folder,read_npz,specifications

FILES=("plan.md","重规划.md","evidence/prereg.json","evidence/inputs.json","evidence/date_gate.json",
       "evidence/prepared_development.json","evidence/graph.json","prepared/topology.npz",
       "prepared/frozen_history.npz","results/development_receipt.json","results/calibration_receipt.json",
       "synthetic/status.json")
CORE=("v7_io.py","v7_check.py","v7_probe.py","v7_prepare.py","v7_score.py","v7_train.py","v7_matrix.py","v7_analysis.py","v7_freeze.py")


def freeze(root,cfg):
    registered(root)
    target=root/"evidence/calibration_freeze.json"
    if target.exists() or (root/"results/holdout").exists():
        raise PermissionError("This run has already been frozen")
    dev=load(root/"results/development_receipt.json")
    cal=load(root/"results/calibration_receipt.json")
    gate=load(root/"evidence/date_gate.json")
    prep=load(root/"evidence/prepared_development.json")
    assert dev["at_utc"]<cal["at_utc"] and not dev["heldout_used"] and not cal["heldout_used"]
    assert gate["observation_values_requested"] is False and prep["holdout_parsed"] is False
    assert load(root/"synthetic/status.json")["status"]=="PASS"
    signed=[]
    for model,seed in specifications(cfg):
        path=folder(root,"final",model,seed)
        raw=read_npz(path/"assessment.npz")
        known=raw["y"]>=0
        ids=raw["sample_id"][known]
        signed.append(dict(model=model,seed=seed,job=path.name,threshold=float(raw["threshold"]),
            known_sha256=fingerprint_keys(ids),hashes={f.name:sha(f) for f in sorted(path.iterdir()) if f.is_file()}))
    roster=read_npz(root/"prepared/topology.npz")["road_ids"]
    keys=key_matrix(minute_values(times_for(local_dates([cfg["holdout"]]),cfg)),roster)
    files=[CODE/p for p in CORE]+sorted(V6.rglob("*.py"))
    receipt=dict(frozen_utc=now(),scope="Calibration only; no formal holdout opened",holdout=cfg["holdout"],
        primary="lead_attention",lead_minutes=cfg["lead_minutes"],expected_keys_sha256=fingerprint_keys(keys),
        expected_windows=int(keys.size),calibration=cfg["calibration"],config_sha256=sha(CONFIG),models=signed,
        hashes={f:sha(root/f) for f in FILES},code={f.relative_to(CODE).as_posix():sha(f) for f in files},heldout_read=False)
    save(target,receipt)
    save(root/"evidence/calibration_freeze_sha256.json",dict(sha256=sha(target),frozen_utc=receipt["frozen_utc"]))
    state=dict(status="development_and_calibration_complete",NYC_formal="blocked",
        Chicago_formal="blocked",Geolife_formal="blocked",NYC_reason=gate["NYC_reason"],
        preregistered_holdout=cfg["holdout"],AP=None,F1=None,balanced_accuracy=None,matched_recovery_FP=None,
        prior_V5_V6_holdout_used=False,new_holdout_values_read=False,development_runs=len(dev["models"]),
        calibration_runs=len(cal["models"]),recorded_utc=now())
    save(root/"evidence/status.json",state)
    print(state,flush=True)


def verify_freeze(root,cfg):
    registered(root)
    contract=load(root/"evidence/calibration_freeze.json")
    expected=load(root/"evidence/calibration_freeze_sha256.json")
    if (expected["sha256"]!=sha(root/"evidence/calibration_freeze.json") or contract["heldout_read"]
            or contract["holdout"]!=cfg["holdout"] or contract["config_sha256"]!=sha(CONFIG)):
        raise ValueError("Frozen cutoff contract changed")
    for key,dig in contract["hashes"].items():
        assert sha(root/key)==dig
    for key,dig in contract["code"].items():
        assert sha(CODE/key)==dig
    for model in contract["models"]:
        path=root/"results/calibration"/model["job"]
        for key,dig in model["hashes"].items():
            assert sha(path/key)==dig
    return contract


if __name__=="__main__":
    freeze(*settings())

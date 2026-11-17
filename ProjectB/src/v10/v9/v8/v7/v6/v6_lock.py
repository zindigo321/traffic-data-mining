"""Immutable model and new-calendar contract before one permitted label read."""
from datetime import datetime
from pathlib import Path
import sys

from v6_io import CODE,CONFIG,V5,load,now,save,settings,sha
from v6_register import verify_registration
from v6_train import configurations,where
sys.path.insert(0,str(V5))
from v5_prepare import fingerprint_keys,key_matrix,local_dates,minute_values,read_npz,times_for

MODEL_FILES=("weight.pt","preprocessor.npz","assessment.npz","sampling_ids.npz","sampling.json",
             "options.json","curve.csv","status.json","matched_recall.json")
INPUTS=("evidence/date_gate.json","evidence/registration.json","evidence/v5_integrity.json",
        "evidence/v5_historical_only.json","evidence/download_geometry.json","evidence/download_development.json",
        "evidence/download_holdout_sealed.json","evidence/chicago_status.json","evidence/prepared_development.json",
        "evidence/graph.json","evidence/nyc_roads.csv","evidence/nyc_edges.csv","prepared/topology.npz",
        "prepared/frozen_history.npz","results/calibration.csv","results/calibration_receipt.json","synthetic/status.json")
CORE=("v6_io.py","v6_register.py","v6_download.py","v6_prepare.py","v6_train.py","v6_lock.py","v6_stats.py","v6_eval.py")


def check_transport(root,cfg,registration):
    held=load(root/"evidence/download_holdout_sealed.json")
    if (not cfg["formal_eligible"] or held["interval"]!=cfg["holdout"] or held["status"]!="complete"
            or held["observation_values_parsed"] is not False or not held["transport_only"] or held["rows"]<1):
        raise ValueError("New holdout transport incomplete")
    for row in held["files"]:
        file=Path(held["directory"])/row["file"]
        if sha(file)!=row["sha256"] or file.stat().st_size!=row["bytes"]:
            raise ValueError("Sealed file changed")
        if datetime.fromisoformat(row["first_requested_utc"])<datetime.fromisoformat(registration["at_utc"]):
            raise ValueError("New values fetched before exact preregistration")
    return held


def freeze(root,cfg):
    if any((root/name).exists() for name in ("evidence/freeze.json","evidence/unlock.json","prepared/holdout.npz","results/holdout")):
        raise PermissionError("The run is already frozen/open")
    registration=verify_registration(root,cfg)
    if not cfg["formal_eligible"]:
        save(root/"evidence/status.json",dict(status="blocked",NYC_formal="blocked",Chicago_formal="blocked",
             AP=None,F1=None,goal_reached=None,reason="Date-only coverage gate did not pass"))
        return
    check_transport(root,cfg,registration)
    cal=load(root/"results/calibration_receipt.json")
    if cal["V5_weights_loaded"] or cal["V5_holdout_used"] or cal["calendar"]!=cfg["calibration"]:
        raise ValueError("V5 holdout or weight leakage")
    if (cal["registration_sha256"]!=sha(root/"evidence/registration.json")
            or cal["data_sha256"]!=sha(root/"prepared/final.npz") or load(root/"synthetic/status.json")["status"]!="PASS"):
        raise ValueError("Calibration or tests are not pinned")
    models=[]
    supports=set()
    for r,p,s in configurations(cfg):
        folder=where(root,r,p,s)
        row=load(folder/"status.json")
        values=read_npz(folder/"assessment.npz")
        if float(values["threshold"])!=row["threshold"] or row["fixed_epoch"]!=cfg["epochs"][f"{r}_{p}"]:
            raise ValueError("Cutoff or checkpoint differs")
        supports.add(fingerprint_keys(values["sample_id"]))
        models.append(dict(recipe=r,pool=p,seed=s,job=folder.name,threshold=row["threshold"],
            primary=[r,p]==registration["primary"],hashes={f:sha(folder/f) for f in MODEL_FILES}))
    if len(supports)!=1 or len(models)!=12:
        raise ValueError("Calibrated model support differs")
    roads=read_npz(root/"prepared/topology.npz")["road_ids"]
    keys=key_matrix(minute_values(times_for(local_dates([cfg["holdout"]]),cfg)),roads)
    deps=[CODE/name for name in CORE]+sorted(V5.rglob("*.py"))
    row=dict(frozen_utc=now(),config_sha256=sha(CONFIG),holdout=cfg["holdout"],prior_holdout=cfg["prior_holdout"],
        calibration=cfg["calibration"],primary=registration["primary"],controls=cfg["controls"],
        expected_grid_sha256=fingerprint_keys(keys),expected_windows=int(keys.size),calibration_known_sha256=supports.pop(),
        files={f:sha(root/f) for f in INPUTS},code={p.relative_to(CODE).as_posix():sha(p) for p in deps},models=models,
        threshold_rule="Max F1 on 28-day known calibration only, tie nearest .5 then lower",
        holdout_values_read=False,V5_holdout_used=False)
    save(root/"evidence/freeze.json",row)
    save(root/"evidence/freeze_sha256.json",dict(sha256=sha(root/"evidence/freeze.json"),frozen_utc=row["frozen_utc"]))
    print(dict(locked=len(models),primary=row["primary"],expected_windows=keys.size,interval=cfg["holdout"],frozen_utc=row["frozen_utc"]),flush=True)


def verify_lock(root,cfg):
    reg=verify_registration(root,cfg)
    lock=load(root/"evidence/freeze.json")
    seal=load(root/"evidence/freeze_sha256.json")
    if (sha(root/"evidence/freeze.json")!=seal["sha256"] or lock["config_sha256"]!=sha(CONFIG)
            or lock["holdout"]!=cfg["holdout"] or lock["primary"]!=reg["primary"] or lock["holdout_values_read"]
            or lock["V5_holdout_used"]):
        raise ValueError("The formal contract was altered")
    for rel,h in lock["files"].items():
        if sha(root/rel)!=h:
            raise ValueError(f"Changed pinned input {rel}")
    for rel,h in lock["code"].items():
        if sha(CODE/rel)!=h:
            raise ValueError(f"Changed code {rel}")
    for spec in lock["models"]:
        for rel,h in spec["hashes"].items():
            if sha(root/"results/calibration"/spec["job"]/rel)!=h:
                raise ValueError(f"Changed frozen model {spec['job']}/{rel}")
    return lock


if __name__=="__main__":
    freeze(*settings())

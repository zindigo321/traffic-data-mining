"""Freeze one scoring recipe and three calibrated cutoffs with zero holdout reads."""
from pathlib import Path

import pandas as pd

from v8_io import CODE,CONFIG,V7,load,now,save,settings,sha
from v8_check import register
from v8_analysis import location

CORE=("v8_io.py","v8_check.py","v8_probe.py","v8_train.py","v8_thresholds.py","v8_analysis.py","v8_freeze.py")
EVIDENCE=("plan.md","evidence/prereg.json","evidence/inputs.json","evidence/date_gate.json",
    "evidence/prepared_development.json","results/rank_development.json","results/rank_calibration.json",
    "results/analysis/selection.json","results/analysis/thresholds.csv","results/analysis/ranking.csv",
    "results/analysis/matched_recall.csv","results/analysis/ranking_summary.json","results/analysis/threshold_summary.json",
    "results/analysis/paired_date_blocks.csv","synthetic/status.json")


def freeze(root,cfg):
    register(root,cfg)
    receipt=root/"evidence/freeze.json"
    if receipt.exists() or (root/"results/holdout").exists():
        raise PermissionError("V8 already frozen/opened")
    gate=load(root/"evidence/date_gate.json")
    choice=load(root/"results/analysis/selection.json")
    assert gate["observation_values_requested"] is False and not choice["official_holdout_used"]
    assert load(root/"synthetic/status.json")["status"]=="PASS"
    frame=pd.read_csv(root/"results/analysis/thresholds.csv",float_precision="round_trip")
    model,rule=choice["selected_model"],choice["selected_rule"]
    models=[]
    for seed in cfg["seeds"]:
        row=frame[(frame.model==model)&(frame.rule==rule)&(frame.stage=="calibration")&(frame.seed==seed)]
        assert len(row)==1 and bool(row.iloc[0]["eval_budget_ok"])
        path=location(root,"final",model,seed)
        models.append(dict(seed=seed,threshold=float(row.iloc[0]["threshold"]),path=path.relative_to(root).as_posix(),
            files={f.name:sha(f) for f in sorted(path.iterdir()) if f.is_file()}))
    files=[CODE/f for f in CORE]+sorted(V7.rglob("*.py"))
    signed=dict(at_utc=now(),scope="Calibrated policy only; official holdout blocked",config_sha256=sha(CONFIG),
        holdout=cfg["holdout"],embargo=cfg["embargo"],calibration=cfg["calibration"],chosen=choice,models=models,
        lead_minutes=10,observation_values_read=False,
        hashes={f:sha(root/f) for f in EVIDENCE},code={f.relative_to(CODE).as_posix():sha(f) for f in files})
    save(receipt,signed)
    save(root/"evidence/freeze_sha256.json",dict(sha256=sha(receipt),at_utc=signed["at_utc"]))
    save(root/"evidence/status.json",dict(status="development_and_calibration_complete",NYC_formal="blocked",
        Chicago_formal="blocked",Geolife_formal="blocked",reason=gate["NYC_reason"],official_AP=None,
        official_balanced_accuracy=None,official_F1=None,official_recovery_FP=None,holdout_used=False,
        V5_V6_holdout_used=False,frozen_model=model,frozen_rule=rule,trained_rank_models=12,
        reused_V7_reference_models=24,recorded_utc=now()))
    print(dict(frozen_model=model,frozen_rule=rule,cutoffs=[x["threshold"] for x in models],formal="blocked"),flush=True)


def verify_freeze(root,cfg):
    register(root,cfg)
    f=root/"evidence/freeze.json"
    lock=load(f)
    seal=load(root/"evidence/freeze_sha256.json")
    assert sha(f)==seal["sha256"] and lock["config_sha256"]==sha(CONFIG) and not lock["observation_values_read"]
    assert lock["holdout"]==cfg["holdout"] and lock["lead_minutes"]==cfg["lead_minutes"]
    for file,h in lock["hashes"].items():
        assert sha(root/file)==h
    for file,h in lock["code"].items():
        assert sha(CODE/file)==h
    for spec in lock["models"]:
        for file,h in spec["files"].items():
            assert sha(root/spec["path"]/file)==h
    return lock


if __name__=="__main__":
    freeze(*settings())

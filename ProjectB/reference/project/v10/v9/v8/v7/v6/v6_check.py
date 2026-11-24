"""Verify the pinned V5 release and attach only non-holdout source roles."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil

from v6_io import V5,V5_NAME,load,now,save,sha

PINNED="49c4a1af96c4663189654fb23e9d9f78cde8c5b1ea8080e1518e03da3f615bc3"
RELEASE="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V5"
WORK="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/work/gt_v5/20261110_v5"
EPOCHS=dict(X1_mean=4,X1_max=2,X1_attention=3,B_mean=2)
OLD=["2026-08-31","2026-10-12"]


def verify_directory(directory):
    manifest=directory/"manifest.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    wanted={r["path"]:r["sha256"] for r in rows}
    actual={p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if len(wanted)!=len(rows) or actual!=wanted.keys()|{"manifest.csv","SHA256SUMS"}:
        raise ValueError("Manifest file list differs")
    for row in rows:
        file=directory/row["path"]
        if file.stat().st_size!=int(row["size_bytes"]) or sha(file)!=row["sha256"]:
            raise ValueError(f"Manifest mismatch: {row['path']}")
    wanted["manifest.csv"]=sha(manifest)
    signatures=[line.split("  ",1) for line in (directory/"SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    if len(signatures)!=len(wanted) or {key:h for h,key in signatures}!=wanted:
        raise ValueError("SHA256SUMS mismatch")
    return dict(manifest_rows=len(rows),hashes=len(signatures))


def check(root):
    release=Path(os.environ.get("GT6_V5_RELEASE",RELEASE))
    work=Path(os.environ.get("GT6_V5_WORK",WORK))
    bundle=release/V5_NAME
    archive=release/(V5_NAME+".zip")
    if sha(archive)!=PINNED:
        raise ValueError("V5 archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.read(f"{V5_NAME}/manifest.csv")!=(bundle/"manifest.csv").read_bytes():
            raise ValueError("V5 bundle does not correspond to the archive")
    integrity=verify_directory(bundle)
    old_config=load(bundle/"project/config.json")
    old_lock=load(bundle/"evidence/freeze.json")
    old_unlock=load(bundle/"evidence/unlock.json")
    if old_unlock["freeze_sha256"]!=sha(bundle/"evidence/freeze.json") or old_unlock["unlocked_utc"]<=old_lock["frozen_utc"]:
        raise ValueError("Prior epoch configuration was not frozen before old test")
    if old_lock["choice"]["recipe"]!="X1" or old_lock["choice"]["epochs"]!=EPOCHS or old_config["holdout"]!=OLD:
        raise ValueError("V5 pretest recipe or embargo changed")
    if old_config["calibration"][1]>OLD[0] or old_lock["choice"]["holdout_used"]:
        raise ValueError("Prior calibration/date policy differs")
    code=bundle/"project"
    if not V5.exists():
        shutil.copytree(code,V5)
    wanted={p.relative_to(code).as_posix() for p in code.rglob("*") if p.is_file()}
    actual={p.relative_to(V5).as_posix() for p in V5.rglob("*") if p.is_file()}
    if actual!=wanted or any(sha(code/name)!=sha(V5/name) for name in wanted):
        raise ValueError("Embedded V5 source differs")
    ev=root/"evidence"
    ev.mkdir(exist_ok=True)
    checks={}
    for role in ("geometry","development"):
        name=f"download_{role}.json"
        transport=load(bundle/"evidence"/name)
        if transport["status"]!="complete" or (role=="development" and transport["interval"]!=[
                old_config["train_start"],old_config["calibration"][1]]):
            raise ValueError("V5 allowed raw development scope differs")
        directory=work/"raw_external"/role
        for row in transport["files"]:
            file=directory/row["file"]
            if file.stat().st_size!=row["bytes"] or sha(file)!=row["sha256"]:
                raise ValueError(f"V5 prior development raw file changed: {file}")
        save(ev/name,{**transport,"directory":str(directory.resolve()),"bound_from_v5_utc":now()})
        checks[role]=len(transport["files"])
    for src,dst in (("graph.json","v5_graph.json"),("prepared_development.json","v5_prepared_development.json"),
                     ("chicago_status.json","v5_chicago_status.json"),("geolife_prior.json","geolife_prior.json")):
        shutil.copy2(bundle/"evidence"/src,ev/dst)
    old_report=(bundle/"\u5ba1\u8ba1\u62a5\u544a.md").read_text(encoding="utf-8")
    historical=[row for row in old_report.splitlines() if row.startswith(("| **X1＋Mean","| X1＋Max","| X1＋Attention","| V4 P1＋Mean \u540c\u652f\u6301\u91cd\u8dd1"))]
    save(ev/"v5_historical_only.json",dict(interval=OLD,source=str(bundle/"\u5ba1\u8ba1\u62a5\u544a.md"),aggregate_rows=historical,
        source_sha256=sha(bundle/"\u5ba1\u8ba1\u62a5\u544a.md"),observation_files_used=False,labels_or_scores_parsed=False))
    receipt=dict(status="PASS",sha256=PINNED,**integrity,code_files=len(wanted),archive=str(archive),
        raw_prior_roles=checks,v5_holdout=OLD,v5_epoch_settings=EPOCHS,prior_epoch_freeze_sha256=old_unlock["freeze_sha256"],
        old_holdout_read="Archive integrity bytes and published aggregate narrative only",verified_utc=now())
    save(ev/"v5_integrity.json",receipt)
    names=("torch","numpy","pandas","scikit-learn","scipy","matplotlib","requests","psutil")
    cpu=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0]
    save(root/"logs/environment.json",dict(cpu=cpu,platform=platform.platform(),executable=sys.executable,python=sys.version,
        physical_cores=psutil.cpu_count(False),logical_cores=psutil.cpu_count(),device="CPU",threads=2,workers=4,
        seeds=old_config["seeds"],packages={key:importlib.metadata.version(key) for key in names}))
    print(receipt,flush=True)


if __name__=="__main__":
    check(Path(os.environ["GT6_ROOT"]))

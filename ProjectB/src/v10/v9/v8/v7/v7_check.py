"""Read-only input verification for the strict lead experiment."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil
from v7_io import CONFIG,V6,V6_NAME,load,now,save,sha

RELEASE="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V6"
WORK="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/work/gt_v5/20261110_v5"
PIN="d04d0c5017f0042abfc666f7af0a64a1495368ac0382a302698d5ecf78ea499e"


def directory(root):
    manifest=root/"manifest.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    files={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    entries={r["path"]:r["sha256"] for r in rows}
    assert len(entries)==len(rows) and files==entries.keys()|{"manifest.csv","SHA256SUMS"}
    for row in rows:
        p=root/row["path"]
        if sha(p)!=row["sha256"] or p.stat().st_size!=int(row["size_bytes"]):
            raise ValueError(f"Manifest mismatch: {row['path']}")
    entries["manifest.csv"]=sha(manifest)
    sums=[x.split("  ",1) for x in (root/"SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    if len(sums)!=len(entries) or {rel:h for h,rel in sums}!=entries:
        raise ValueError("SHA256SUMS mismatch")
    return dict(manifest_rows=len(rows),hashes=len(sums))


def registered(root):
    receipt=load(root/"evidence/prereg.json")
    required={root/"plan.md":"plan_sha256",root/"重规划.md":"replan_sha256",CONFIG:"config_sha256"}
    if receipt["observation_values_read"] is not False or any(sha(f)!=receipt[k] for f,k in required.items()):
        raise ValueError("V7 preregistration changed")
    cfg=load(CONFIG)
    if cfg["lead_minutes"]!=10 or cfg["holdout"][0]!=cfg["embargo"][1] or cfg["calibration"][1]>cfg["embargo"][0]:
        raise ValueError("Protocol or embargo mismatch")
    return cfg


def verify(root):
    cfg=registered(root)
    release=Path(os.environ.get("GT7_V6_RELEASE",RELEASE))
    work=Path(os.environ.get("GT7_V5_WORK",WORK))
    bundle=release/V6_NAME
    zip_path=release/(V6_NAME+".zip")
    if sha(zip_path)!=PIN:
        raise ValueError("Unexpected V6 archive")
    with zipfile.ZipFile(zip_path) as f:
        if f.read(f"{V6_NAME}/manifest.csv")!=(bundle/"manifest.csv").read_bytes():
            raise ValueError("Source release is not the archive")
    found=directory(bundle)
    prior=load(bundle/"project/config.json")
    assert prior["holdout"][1]==cfg["embargo"][1] and prior["prior_holdout"][0]==cfg["embargo"][0]
    for key in ("nyc_dataset","zone","calibration","folds","train_start","ratio","seeds","source","pools"):
        if cfg[key]!=prior[key]:
            raise ValueError(f"Changed inherited contract {key}")
    source=bundle/"project"
    if not V6.exists():
        shutil.copytree(source,V6)
    records={f.relative_to(source) for f in source.rglob("*") if f.is_file()}
    assert records=={p.relative_to(V6) for p in V6.rglob("*") if p.is_file()}
    assert all(sha(source/f)==sha(V6/f) for f in records)
    ev=root/"evidence"
    roles={}
    for role in ("geometry","development"):
        name=f"download_{role}.json"
        receipt=load(bundle/"evidence"/name)
        if role=="development" and receipt["interval"]!=[cfg["train_start"],cfg["calibration"][1]]:
            raise ValueError("Previous data interval not allowed")
        location=work/"raw_external"/role
        for entry in receipt["files"]:
            f=location/entry["file"]
            if sha(f)!=entry["sha256"] or f.stat().st_size!=entry["bytes"]:
                raise ValueError(f"Changed raw development source: {f}")
        roles[role]=len(receipt["files"])
        save(ev/name,{**receipt,"directory":str(location.resolve()),"verified_by_v7":now()})
    for src,dst in (("graph.json","prior_graph.json"),("prepared_development.json","prior_prepared.json"),
        ("chicago_status.json","chicago_prior_status.json"),("geolife_prior.json","geolife_prior.json")):
        shutil.copy2(bundle/"evidence"/src,ev/dst)
    report=bundle/"审计报告.md"
    result=dict(status="PASS",zip_sha256=PIN,zip=str(zip_path),**found,source_files=len(records),
        raw_roles=roles,excluded=cfg["embargo"],read_prior_holdout="Byte-level integrity only",
        prior_published_report_sha256=sha(report),checked_utc=now())
    save(ev/"inputs.json",result)
    cpu=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu=winreg.QueryValueEx(k,"ProcessorNameString")[0]
    pkgs=("torch","numpy","pandas","scikit-learn","scipy","matplotlib","requests","psutil")
    save(root/"logs/environment.json",dict(cpu=cpu,python=sys.version,executable=sys.executable,platform=platform.platform(),
        physical_cores=psutil.cpu_count(False),logical_cores=psutil.cpu_count(),memory=psutil.virtual_memory().total,
        device="CPU",threads=cfg["cpu_threads"],workers=cfg["workers"],seeds=cfg["seeds"],
        packages={name:importlib.metadata.version(name) for name in pkgs}))
    print(result,flush=True)


if __name__=="__main__":
    verify(Path(os.environ["GT7_ROOT"]))

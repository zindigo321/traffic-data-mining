"""Verify prior artifacts without using any V2 holdout values in V3."""
import csv
from datetime import date
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil
from v3_io import CODE, context, load, now, save, sha


def verified_directory(root):
    with (root/"manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        records=list(csv.DictReader(f))
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    wanted={r["path"] for r in records}
    if len(wanted)!=len(records) or actual!=wanted|{"manifest.csv","SHA256SUMS"}:
        raise ValueError("Manifest file set differs")
    for r in records:
        p=root/r["path"]
        if p.stat().st_size!=int(r["size_bytes"]) or sha(p)!=r["sha256"]:
            raise ValueError(f"Invalid manifest row {r['path']}")
    rows=[line.split("  ",1) for line in (root/"SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    seen={p:digest for digest,p in rows}
    target={r["path"]:r["sha256"] for r in records}|{"manifest.csv":sha(root/"manifest.csv")}
    if len(seen)!=len(rows) or seen!=target:
        raise ValueError("SHA256SUMS differs")
    return dict(manifest_rows=len(records),hashed_files=len(rows))


def main():
    root,cfg=context()
    bundle,archive=Path(cfg["v2_bundle"]),Path(cfg["v2_zip"])
    archive_sha=sha(archive)
    if archive_sha!=cfg["v2_zip_sha256"]:
        raise ValueError("Unexpected V2 archive")
    with zipfile.ZipFile(archive) as z:
        if z.read(f"{bundle.name}/manifest.csv") != (bundle/"manifest.csv").read_bytes():
            raise ValueError("V2 directory is not the pinned archive")
    integrity=verified_directory(bundle)
    selection=load(bundle/"results/development_selection.json")
    if [selection[k] for k in ("recipe","pooling","route")]!=["V2","mean","routed"]:
        raise ValueError("V2 comparator identity differs")
    evidence=root/"evidence"
    evidence.mkdir(exist_ok=True)
    old=load(bundle/"evidence/v1_inputs.json")
    raw=Path(old["cache"])/"public_data/chicago_fixed.csv"
    if sha(raw)!=old["verified_cache"]["public_data/chicago_fixed.csv"]:
        raise ValueError("Original Chicago cache SHA mismatch")
    receipts=load(bundle/"evidence/download_development.json")
    begin,end=cfg["history_intervals"][1]
    if receipts["status"]!="complete" or receipts["calendar_interval"]!=[begin,end]:
        raise ValueError("Historical V2 development date range changed")
    previous=begin
    source_parts=[]
    for receipt in receipts["shards"]:
        path=Path(receipts["directory"])/receipt["file"]
        if (receipt["status"]!="complete" or receipt["start"]!=previous or receipt["stop_exclusive"]>end or
                sha(path)!=receipt["sha256"]):
            raise ValueError("Incomplete or modified historical V2 development shard")
        source_parts.append(dict(path=str(path),start=receipt["start"],stop=receipt["stop_exclusive"],
                                 sha256=receipt["sha256"],rows=receipt["rows"],query=receipt["query"]))
        previous=receipt["stop_exclusive"]
    if previous!=end:
        raise ValueError("Historical date gap")
    prior=CODE/"v2"
    if prior.exists():
        for p in (bundle/"project").rglob("*"):
            if p.is_file() and sha(prior/p.relative_to(bundle/"project"))!=sha(p):
                raise ValueError("Embedded V2 code is not the audited code")
    else:
        shutil.copytree(bundle/"project",prior,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copy2(bundle/"evidence/roads_v1.csv",evidence/"roads.csv")
    prepared=root/"prepared"
    prepared.mkdir(exist_ok=True)
    shutil.copy2(bundle/"prepared/topology.npz",prepared/"topology.npz")
    for f in ("geolife_status.json","geolife_split_counts.json"):
        shutil.copy2(bundle/"evidence"/f,evidence/f)
    shutil.copy2(bundle/"evidence/chicago_publisher.json",evidence/"v2_publisher.json")
    shutil.copy2(bundle/"evidence/holdout_status.json",evidence/"v2_historical_result.json")
    info=dict(status="PASS",at_utc=now(),archive=str(archive),archive_sha256=archive_sha,**integrity,
              legacy_csv=str(raw),legacy_sha256=sha(raw),historical_development=source_parts,
              reused_roster_sha256=sha(evidence/"roads.csv"),reused_topology_sha256=sha(prepared/"topology.npz"),
              v2_test_read_for_v3=False,excluded_calendar=cfg["exclusions"])
    save(evidence/"prior_inputs.json",info)
    pkgs=["torch","numpy","pandas","scikit-learn","scipy","matplotlib","requests","psutil"]
    cpu=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0]
    save(root/"logs/environment.json",dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
         cpu=cpu,cores=psutil.cpu_count(False),logical=psutil.cpu_count(True),ram_bytes=psutil.virtual_memory().total,
         device="CPU",workers=cfg["workers"],threads_per_worker=cfg["cpu_threads"],seeds=cfg["seeds"],
         packages={name:importlib.metadata.version(name) for name in pkgs}))
    print({k:v for k,v in info.items() if k!="historical_development"},flush=True)


if __name__=="__main__":
    main()


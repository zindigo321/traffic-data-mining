"""Verify V7 release and attach only signed, legal development artifacts."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil

from v8_io import CONFIG,V7,V7_NAME,load,now,save,settings,sha

RELEASE="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V7"
WORK="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/work/gt_v7/20261110_v7"
EXPECTED="0eeb6dcd7b8bd6f9e89433df08f02a5dcda89b6df45e83ed21c666694f9e8560"
BASELINES=("lead_attention","lead_center")


def directory(root):
    manifest=root/"manifest.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    files={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    entries={r["path"]:r["sha256"] for r in rows}
    if len(entries)!=len(rows) or files!=entries.keys()|{"manifest.csv","SHA256SUMS"}:
        raise ValueError("Incorrect manifest file list")
    for row in rows:
        p=root/row["path"]
        if sha(p)!=row["sha256"] or p.stat().st_size!=int(row["size_bytes"]):
            raise ValueError(f"Changed file {row['path']}")
    entries["manifest.csv"]=sha(manifest)
    sums=[x.split("  ",1) for x in (root/"SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    if len(sums)!=len(entries) or {rel:h for h,rel in sums}!=entries:
        raise ValueError("SHA256SUMS mismatch")
    return dict(manifest_rows=len(rows),hashes=len(sums))


def register(root,cfg):
    r=load(root/"evidence/prereg.json")
    assert sha(root/"plan.md")==r["plan_sha256"] and sha(CONFIG)==r["config_sha256"] and r["holdout_read"] is False
    assert cfg["lead_minutes"]==10 and cfg["holdout"][0]==cfg["embargo"][1] and cfg["calibration"][1]<=cfg["embargo"][0]
    return r


def check(root,cfg):
    registration=register(root,cfg)
    rel=Path(os.environ.get("GT8_V7_RELEASE",RELEASE))
    old=Path(os.environ.get("GT8_V7_WORK",WORK))
    bundle=rel/V7_NAME
    archive=rel/(V7_NAME+".zip")
    if sha(archive)!=EXPECTED:
        raise ValueError("Unpinned V7 ZIP")
    with zipfile.ZipFile(archive) as f:
        assert f.read(f"{V7_NAME}/manifest.csv")== (bundle/"manifest.csv").read_bytes()
    sizes=directory(bundle)
    source=bundle/"project"
    prior=load(source/"config.json")
    assert all(cfg[k]==v for k,v in prior.items())
    st=load(bundle/"evidence/status.json")
    assert st["NYC_formal"]=="blocked" and not st["new_holdout_values_read"] and not st["prior_V5_V6_holdout_used"]
    if not V7.exists():
        shutil.copytree(source,V7)
    files={p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    assert files=={p.relative_to(V7) for p in V7.rglob("*") if p.is_file()}
    assert all(sha(source/f)==sha(V7/f) for f in files)
    ev=root/"evidence"
    allowed=[*cfg["folds"],"final"]
    for f in ("prepared_development.json","graph.json","download_geometry.json","download_development.json"):
        shutil.copy2(bundle/"evidence"/f,ev/f)
    for fold in allowed:
        shutil.copy2(bundle/"evidence"/f"fold_lead_{fold}.json",ev/f"fold_lead_{fold}.json")
    saved=load(ev/"prepared_development.json")
    pinned={}
    for fold in allowed:
        path=old/"prepared"/f"lead_{fold}.npz"
        h=sha(path)
        if h!=saved["lead_hashes"][fold]:
            raise ValueError("Pinned legal input differs")
        pinned[fold]=dict(path=str(path.resolve()),sha256=h,bytes=path.stat().st_size)
    snap={}
    for fold in allowed:
        for model in BASELINES:
            for seed in cfg["seeds"]:
                job=f"{model}_s{seed}"
                base=bundle/"results"/("calibration" if fold=="final" else f"development/{fold}")/job
                dest=root/"results/reference"/fold/job
                if not dest.exists():
                    shutil.copytree(base,dest)
                snap[f"{fold}/{job}"]={p.name:sha(p) for p in sorted(base.iterdir()) if p.is_file()}
                assert snap[f"{fold}/{job}"]=={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
    orig_freeze=bundle/"evidence/calibration_freeze.json"
    info=dict(status="PASS",checked_utc=now(),V7_zip_sha256=EXPECTED,**sizes,code_files=len(files),
        source_archive=str(archive.resolve()),V7_report_sha256=sha(bundle/"\u5ba1\u8ba1\u62a5\u544a.md"),
        original_V7_freeze_sha256=sha(orig_freeze),prepared=pinned,reference=snap,reference_runs=len(snap),
        V5_V6_holdout_used=False,new_holdout_read=False,inheritance="V7 frozen legal data, dev/cal prediction and weight only",
        prereg_utc=registration["at_utc"])
    save(ev/"inputs.json",info)
    cpu=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu=winreg.QueryValueEx(k,"ProcessorNameString")[0]
    names=("torch","numpy","pandas","scikit-learn","scipy","matplotlib","requests","psutil")
    save(root/"logs/environment.json",dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
        cpu=cpu,cores=psutil.cpu_count(False),logical=psutil.cpu_count(True),memory=psutil.virtual_memory().total,
        device="CPU",threads=cfg["cpu_threads"],workers=cfg["workers"],seeds=cfg["seeds"],
        packages={k:importlib.metadata.version(k) for k in names}))
    print({k:v for k,v in info.items() if k not in ("reference","prepared")},flush=True)


if __name__=="__main__":
    check(*settings())

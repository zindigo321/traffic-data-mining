"""Pin V4 code, archive, and published Chicago scope before external fitting."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil

from v5_io import V4, V4_NAME, load, now, save, sha

PINNED_SHA = "7d5a944e08b5c970c26a1711c92c9b3796d4142daaebaddd17dc35d68a90f2b4"
DEFAULT_RELEASE = "G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V4"


def verify_directory(directory):
    manifest = directory / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    signed = {r["path"]: r["sha256"] for r in rows}
    if len(signed) != len(rows) or files != set(signed) | {"manifest.csv","SHA256SUMS"}:
        raise ValueError("Manifest file list differs")
    for row in rows:
        p = directory / row["path"]
        if p.stat().st_size != int(row["size_bytes"]) or sha(p) != row["sha256"]:
            raise ValueError(f"Manifest content mismatch: {row['path']}")
    signed["manifest.csv"] = sha(manifest)
    sums = [s.split("  ",1) for s in (directory/"SHA256SUMS").read_text(encoding="utf-8-sig").splitlines()]
    if len(sums) != len(signed) or signed != {name: h for h,name in sums}:
        raise ValueError("SHA256SUMS differs")
    return dict(manifest_rows=len(rows),hashes=len(sums))


def check(root):
    release = Path(os.environ.get("GT5_V4_RELEASE", DEFAULT_RELEASE))
    archive = release / (V4_NAME + ".zip")
    bundle = release / V4_NAME
    if sha(archive) != PINNED_SHA:
        raise ValueError("Unexpected V4 archive SHA-256")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.read(f"{V4_NAME}/manifest.csv") != (bundle/"manifest.csv").read_bytes():
            raise ValueError("Archive and directory differ")
    checked = verify_directory(bundle)
    result = load(bundle/"results/selection.json")
    if (result["recipe"],result["pool"],result["scope"]) != ("P1","mean","Development only"):
        raise ValueError("V4 start configuration differs")
    code = bundle/"project"
    if not V4.exists():
        shutil.copytree(code,V4)
    actual_code = {f.relative_to(V4).as_posix() for f in V4.rglob("*") if f.is_file()}
    required_code = {f.relative_to(code).as_posix() for f in code.rglob("*") if f.is_file()}
    if actual_code != required_code or any(sha(V4 / rel)!=sha(code/rel) for rel in required_code):
        raise ValueError("Embedded V4 project differs")
    ev = root/"evidence"
    for src, dst in (("holdout_gate.json","chicago_v4_gate.json"),("status.json","chicago_v4_status.json"),
                     ("geolife_status.json","geolife_prior.json")):
        shutil.copy2(bundle/"evidence"/src,ev/dst)
    payload = dict(status="PASS", v4_zip=str(archive), sha256=PINNED_SHA, **checked, source_files=len(required_code),
                   start="P1_mean", v4_chicago_scope="Prior development only",
                   read_from_prior="Archive integrity, V4 source, published selection and status. No prior scores, weights or holdout parsed.",
                   verified_utc=now())
    save(ev/"prior_inputs.json",payload)
    names=("torch","numpy","pandas","scikit-learn","scipy","matplotlib","requests","psutil")
    cpu=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0]
    save(root/"logs/environment.json",dict(cpu=cpu,platform=platform.platform(),executable=sys.executable,
        python=sys.version,physical_cores=psutil.cpu_count(False),logical_cores=psutil.cpu_count(),
        memory_bytes=psutil.virtual_memory().total,device="CPU",workers=4,threads_per_worker=2,seeds=[11,22,33],
        packages={k:importlib.metadata.version(k) for k in names}))
    print(payload,flush=True)


if __name__=="__main__":
    check(Path(os.environ["GT5_ROOT"]))

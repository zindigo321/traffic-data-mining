"""Verify the immutable prior artifact, the independent review and candidate caches."""
import csv
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import zipfile

import psutil
from common import digest, json_read, json_write, now, settings


def verify_manifest(root):
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected = {m["path"] for m in manifest}
    if len(manifest) != len(expected) or paths != expected | {"manifest.csv", "SHA256SUMS"}:
        raise ValueError("Mismatch in inventory")
    for row in manifest:
        p = root / row["path"]
        if p.stat().st_size != int(row["size_bytes"]) or digest(p) != row["sha256"]:
            raise ValueError(f"Manifest mismatch {row['path']}")
    sums = [line.split("  ", 1) for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    recorded = {p: sha for sha, p in sums}
    compare = {m["path"]: m["sha256"] for m in manifest}
    compare["manifest.csv"] = digest(root / "manifest.csv")
    if len(recorded) != len(sums) or recorded != compare:
        raise ValueError("SHA256SUMS mismatch")
    return dict(manifest_rows=len(manifest), hashed_files=len(sums), inventory_files=len(paths))


def main():
    root, cfg = settings()
    source = Path(cfg["v1_zip"])
    if digest(source) != cfg["v1_zip_sha256"]:
        raise ValueError("v1 archive digest differs")
    review = Path(cfg["review_file"])
    if digest(review) != cfg["review_sha256"]:
        raise ValueError("Review attachment digest differs")
    temp = root / "v1_extracted"
    temp.mkdir(exist_ok=False)
    archive_name = "ProjectB_GraphTransformer_审计包"
    with zipfile.ZipFile(source) as z:
        names = set()
        for member in z.infolist():
            p = PurePosixPath(member.filename)
            if (p.is_absolute() or len(p.parts) < 2 or p.parts[0] != archive_name or ".." in p.parts
                    or any(":" in s or "\\" in s for s in p.parts)
                    or member.filename.casefold() in names or member.is_dir()):
                raise ValueError("Unsafe or duplicate ZIP member")
            names.add(member.filename.casefold())
        if z.testzip():
            raise ValueError("CRC mismatch")
        z.extractall(temp)
    original = temp / archive_name
    integrity = verify_manifest(original)
    cache = Path(cfg["v1_cache"])
    download = json_read(original / "data_audit/chicago/download.json")
    prepared = json_read(original / "data_audit/chicago/prepared_digests.json")
    known = {"public_data/chicago_fixed.csv": download["sha256"], **{f"prepared/{name}": sha for name, sha in prepared.items()}}
    for name, sha in known.items():
        if digest(cache / name) != sha:
            raise ValueError(f"Untrusted cache {name}")
    info = dict(status="PASS", at_utc=now(), extracted=str(original), archive_sha256=digest(source),
                archive_path=str(source), review_sha256=digest(review), review_path=str(review),
                cache=str(cache), verified_cache=known, **integrity)
    json_write(root / "evidence/v1_inputs.json", info)
    python_pkgs = ["numpy", "pandas", "torch", "scipy", "scikit-learn", "matplotlib", "requests", "psutil"]
    gpu_cmd = shutil.which("nvidia-smi")
    gpu = subprocess.run([gpu_cmd, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=15).stdout.strip() if gpu_cmd else None
    cpu = platform.processor()
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    json_write(root / "logs/environment.json", dict(python=sys.version, executable=sys.executable, os=platform.platform(),
               cpu=cpu, cores=psutil.cpu_count(logical=False), threads=psutil.cpu_count(), gpu_installed=gpu,
               ram_bytes=psutil.virtual_memory().total, device="cpu", seeds=cfg["seeds"], cpu_threads=cfg["cpu_threads"],
               workers=cfg["workers"], packages={p: importlib.metadata.version(p) for p in python_pkgs}))
    for f in ("download.json", "metadata.json", "prepared_digests.json"):
        shutil.copy2(original / "data_audit" / "chicago" / f, root / "evidence" / ("v1_" + f))
    for f in ("status.json", "split_counts.json"):
        shutil.copy2(original / "results" / "geolife" / f, root / "evidence" / ("geolife_" + f))
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


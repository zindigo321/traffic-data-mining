"""Verify V9 and attach signed NYC development inputs without opening a holdout."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil

from v10_io import CONFIG, V9, V9_NAME, load, now, save, settings, sha

RELEASE = "G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V9"
PREPARED = "G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/work/gt_v7/20261110_v7/prepared"
PIN = "e61242dc9b14d3fc2ce765313f1d7e9ed1da92d61e8c4af03f73473373920a53"
EVIDENCE = ("prepared_development.json", "graph.json", "download_geometry.json", "download_development.json",
            "fold_lead_f1.json", "fold_lead_f2.json", "fold_lead_f3.json", "fold_lead_final.json")


def directory(root):
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    entries = {r["path"]: r["sha256"] for r in rows}
    if len(entries) != len(rows) or files != entries.keys() | {"manifest.csv", "SHA256SUMS"}:
        raise ValueError("Manifest file list differs")
    for row in rows:
        p = root / row["path"]
        if sha(p) != row["sha256"] or p.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Changed file {row['path']}")
    entries["manifest.csv"] = sha(root / "manifest.csv")
    sums = [r.split("  ", 1) for r in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    if len(sums) != len(entries) or {f: h for h, f in sums} != entries:
        raise ValueError("SHA256SUMS differs")
    return dict(manifest_rows=len(rows), hashes=len(sums))


def snapshot(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def register(root, cfg):
    prereg = load(root / "evidence/prereg.json")
    assert sha(root / "plan.md") == prereg["plan_sha256"] and sha(CONFIG) == prereg["config_sha256"]
    assert prereg["new_holdout_read"] is False and prereg["v10_calibration_parsed"] is False
    assert cfg["lead_minutes"] == 10 and cfg["v9_latest_legal_lag_minutes"] == 20
    assert cfg["holdout"][0] == cfg["embargo"][1] and cfg["calibration"][1] <= cfg["embargo"][0]
    return prereg


def original(root, fold, seed):
    return root / "results/reference" / fold / f"center_s{seed}"


def check(root, cfg):
    register(root, cfg)
    release = Path(os.environ.get("GT10_V9_RELEASE", RELEASE)).resolve()
    prepared = Path(os.environ.get("GT10_PREPARED", PREPARED)).resolve()
    bundle, archive = release / V9_NAME, release / (V9_NAME + ".zip")
    if sha(archive) != PIN:
        raise ValueError("V9 ZIP signature mismatch")
    with zipfile.ZipFile(archive) as z:
        for name in ("manifest.csv", "SHA256SUMS"):
            assert z.read(f"{V9_NAME}/{name}") == (bundle / name).read_bytes()
    sizes = directory(bundle)
    oldcfg = load(bundle / "project/config.json")
    assert all(cfg[k] == v for k, v in oldcfg.items())
    oldstate = load(bundle / "evidence/status.json")
    assert oldstate["NYC_formal"] == "blocked" and not oldstate["new_holdout_values_read"] and not oldstate["V5_V6_holdout_used"]
    if not V9.exists():
        shutil.copytree(bundle / "project", V9)
    assert snapshot(V9) == snapshot(bundle / "project")
    for name in EVIDENCE:
        shutil.copy2(bundle / "evidence" / name, root / "evidence" / name)
    for name in ("freeze.json", "status.json", "development_selection.json"):
        shutil.copy2(bundle / "evidence" / name, root / "evidence" / ("v9_" + name))
    proof = load(root / "evidence/prepared_development.json")
    assert not proof["holdout_parsed"] and proof["latest_dynamic_lag_minutes"] == 20
    mapping, jobs = {}, {}
    for fold in (*cfg["folds"], "final"):
        file = prepared / f"lead_{fold}.npz"
        if sha(file) != proof["lead_hashes"][fold]:
            raise ValueError(f"Prepared data hash differs in {fold}")
        mapping[fold] = dict(path=str(file), sha256=sha(file), bytes=file.stat().st_size)
        for seed in cfg["seeds"]:
            source = bundle / "results/signed" / fold / f"lead_center_s{seed}"
            dest = original(root, fold, seed)
            if not dest.exists():
                shutil.copytree(source, dest)
            assert snapshot(source) == snapshot(dest)
            jobs[f"{fold}/{seed}"] = snapshot(dest)
    save(root / "evidence/inputs.json", dict(status="PASS", checked_utc=now(), v9_archive=str(archive),
         v9_zip_sha256=PIN, **sizes, prepared=mapping, reference=jobs, embedded_v9=snapshot(V9),
         source_evidence={name: sha(root / "evidence" / name) for name in EVIDENCE},
         new_holdout_values_read=False, V5_V6_holdout_used=False))
    packages = ("torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "requests", "psutil",
                "tzdata", "geopandas", "shapely", "pyproj", "pyogrio", "joblib")
    versions = {k: importlib.metadata.version(k) for k in packages}
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    save(root / "logs/environment.json", dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
         cpu=cpu, physical_cores=psutil.cpu_count(False), logical_cores=psutil.cpu_count(True),
         total_RAM_bytes=psutil.virtual_memory().total, device="CPU", threads=cfg["cpu_threads"],
         workers=cfg["workers"], seeds=cfg["seeds"], packages=versions))
    (root / "project/requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in versions.items()), encoding="utf-8")
    print(dict(status="PASS", source="V9", reference_jobs=len(jobs), **sizes), flush=True)


if __name__ == "__main__":
    check(*settings())

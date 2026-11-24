"""Verify and attach only the signed V8 development and calibration inputs."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import psutil

from v9_io import CONFIG, V8, V8_NAME, load, now, save, settings, sha

RELEASE = "G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V8"
PREPARED = "G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/work/gt_v7/20261110_v7/prepared"
V8_SHA = "9fbb55f2bba5494afb5ae36b30f47753bc8cf1ad881dd98f43447c1afb4c275a"
MODELS = {"center": "lead_center", "attention": "lead_attention", "rank": "rank"}
SOURCE_EVIDENCE = (
    "prepared_development.json", "graph.json", "download_geometry.json", "download_development.json",
    "fold_lead_f1.json", "fold_lead_f2.json", "fold_lead_f3.json", "fold_lead_final.json"
)


def directory(root):
    manifest = root / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    entries = {row["path"]: row["sha256"] for row in rows}
    if len(entries) != len(rows) or files != entries.keys() | {"manifest.csv", "SHA256SUMS"}:
        raise ValueError("Incorrect manifest file list")
    for row in rows:
        p = root / row["path"]
        if sha(p) != row["sha256"] or p.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Changed file {row['path']}")
    entries["manifest.csv"] = sha(manifest)
    sums = [x.split("  ", 1) for x in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    if len(sums) != len(entries) or {rel: digest for digest, rel in sums} != entries:
        raise ValueError("SHA256SUMS mismatch")
    return dict(manifest_rows=len(rows), hashes=len(sums))


def register(root, cfg):
    prereg = load(root / "evidence/prereg.json")
    if (sha(root / "plan.md") != prereg["plan_sha256"] or sha(CONFIG) != prereg["config_sha256"]
            or prereg["new_values_read"] is not False):
        raise ValueError("V9 preregistration changed")
    assert cfg["lead_minutes"] == 10 and cfg["v9_latest_legal_lag_minutes"] == 20
    assert cfg["holdout"][0] == cfg["embargo"][1] and cfg["calibration"][1] <= cfg["embargo"][0]
    return prereg


def job(root, fold, model, seed):
    return root / "results/signed" / fold / f"{MODELS[model]}_s{seed}"


def source_job(bundle, fold, model, seed):
    return bundle / "results" / ("candidate" if model == "rank" else "reference") / fold / f"{MODELS[model]}_s{seed}"


def snapshot(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def evidence(root, cfg):
    register(root, cfg)
    release = Path(os.environ.get("GT9_V8_RELEASE", RELEASE)).resolve()
    prep = Path(os.environ.get("GT9_PREPARED", PREPARED)).resolve()
    bundle, archive = release / V8_NAME, release / (V8_NAME + ".zip")
    if sha(archive) != V8_SHA:
        raise ValueError("Unpinned V8 ZIP")
    with zipfile.ZipFile(archive) as zipped:
        for f in ("manifest.csv", "SHA256SUMS"):
            assert zipped.read(f"{V8_NAME}/{f}") == (bundle / f).read_bytes()
    sizes = directory(bundle)
    old = load(bundle / "project/config.json")
    assert all(cfg[key] == value for key, value in old.items())
    status = load(bundle / "evidence/status.json")
    assert status["NYC_formal"] == "blocked" and not status["holdout_used"] and not status["V5_V6_holdout_used"]
    if not V8.exists():
        shutil.copytree(bundle / "project", V8)
    code = snapshot(V8)
    assert code == snapshot(bundle / "project")
    ev = root / "evidence"
    sources = {}
    for name in SOURCE_EVIDENCE:
        source = bundle / "evidence" / name
        shutil.copy2(source, ev / name)
        sources[name] = sha(source)
    prepared = load(ev / "prepared_development.json")
    assert prepared["latest_dynamic_lag_minutes"] == cfg["v9_latest_legal_lag_minutes"]
    assert prepared["embargo"] == cfg["embargo"] and not prepared["holdout_parsed"]
    mapping = {}
    jobs = {}
    for fold in (*cfg["folds"], "final"):
        p = prep / f"lead_{fold}.npz"
        digest = sha(p)
        if digest != prepared["lead_hashes"][fold]:
            raise ValueError(f"Wrong prepared SHA for {fold}")
        mapping[fold] = dict(path=str(p), sha256=digest, bytes=p.stat().st_size)
        for model in MODELS:
            for seed in cfg["seeds"]:
                src = source_job(bundle, fold, model, seed)
                dest = job(root, fold, model, seed)
                if not dest.exists():
                    shutil.copytree(src, dest)
                files = snapshot(src)
                assert files == snapshot(dest)
                jobs[f"{fold}/{model}/{seed}"] = dict(source=src.relative_to(bundle).as_posix(), files=files)
    info = dict(status="PASS", checked_utc=now(), v8_zip_sha256=V8_SHA, source_archive=str(archive),
                source_manifest_sha256=sha(bundle / "manifest.csv"), prior_report_sha256=sha(bundle / "\u5ba1\u8ba1\u62a5\u544a.md"),
                **sizes, v8_code=code, source_evidence=sources, prepared=mapping, signed_jobs=jobs,
                V5_V6_holdout_used=False, new_holdout_values_read=False)
    save(ev / "inputs.json", info)
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    names = ("torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "requests", "psutil", "tzdata", "geopandas", "shapely", "pyproj", "pyogrio", "joblib")
    packages = {k: importlib.metadata.version(k) for k in names}
    save(root / "logs/environment.json", dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
         cpu=cpu, cores=psutil.cpu_count(False), logical=psutil.cpu_count(True), memory_bytes=psutil.virtual_memory().total,
         device="CPU", replay_threads=cfg["cpu_threads"], analysis_workers=1, seeds=cfg["seeds"], packages=packages))
    (root / "project/requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in packages.items()), encoding="utf-8")
    print(dict(status=info["status"], signed_jobs=len(jobs), code_files=len(code), **sizes), flush=True)


if __name__ == "__main__":
    evidence(*settings())

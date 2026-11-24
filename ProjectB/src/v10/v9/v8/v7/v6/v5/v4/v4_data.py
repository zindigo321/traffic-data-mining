"""Verify V3 source lineage and bind only allowable development arrays."""
import csv
from datetime import date
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys
import zipfile

from v4_io import CODE, LEGACY, load, now, save, settings, sha

V3_NAME = "ProjectB_GraphTransformer_V3_\u5ba1\u8ba1\u5305"
EVIDENCE_FILES = ("prior_inputs.json", "prepared_development.json", "download_development.json",
                  "roads.csv", "geolife_status.json", "geolife_split_counts.json")
FOLD_FILES = ("f1", "f2", "f3", "final")


def verify_directory(bundle):
    manifest = bundle / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    expected = {item["path"]: item["sha256"] for item in records}
    actual = {item.relative_to(bundle).as_posix() for item in bundle.rglob("*") if item.is_file()}
    if len(expected) != len(records) or actual != expected.keys() | {"manifest.csv", "SHA256SUMS"}:
        raise ValueError("Manifest file set mismatch")
    for row in records:
        path = bundle / row["path"]
        if path.stat().st_size != int(row["size_bytes"]) or sha(path) != row["sha256"]:
            raise ValueError(f"Manifest mismatch: {row['path']}")
    expected["manifest.csv"] = sha(manifest)
    raw = [row.split("  ", 1) for row in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    if len(raw) != len(expected) or {name: digest for digest, name in raw} != expected:
        raise ValueError("SHA256SUMS mismatch")
    return dict(manifest_rows=len(records), verified_hashes=len(raw))


def interval_ok(interval, banned):
    return all(interval[1] <= start or interval[0] >= stop for start, stop in banned)


def bind(root, cfg):
    release = Path(cfg["v3_release"])
    bundle, archive = release / V3_NAME, release / (V3_NAME + ".zip")
    archive_sha = sha(archive)
    if archive_sha != cfg["v3_archive_sha256"]:
        raise ValueError("V3 archive SHA differs")
    with zipfile.ZipFile(archive) as z:
        if z.read(f"{V3_NAME}/manifest.csv") != (bundle / "manifest.csv").read_bytes():
            raise ValueError("V3 directory and archive manifests differ")
    integrity = verify_directory(bundle)
    original = load(bundle / "project/config.json")
    if (original["holdout"] != cfg["v3_known_unavailable"] or original["holdout"][1] != cfg["earliest_new_date"]
            or original["exclusions"] + [original["holdout"]] != cfg["all_exclusions"]
            or original["seeds"] != cfg["seeds"] or original["pools"] != cfg["pools"]):
        raise ValueError("V3 calendar or model protocol differs")
    for fold, phase in original["folds"].items():
        intervals = phase["train"] + [phase["valid"]]
        if phase["valid"] != cfg["fold_limits"][fold] or not all(interval_ok(x, cfg["all_exclusions"]) for x in intervals):
            raise ValueError("V3 fold intersects an embargo")
    if original["calibration"] != cfg["fold_limits"]["final"]:
        raise ValueError("V3 calibration changed")
    if not all(interval_ok(x, cfg["all_exclusions"]) for x in original["final_train"] + [original["calibration"]]):
        raise ValueError("V3 final dates intersect an embargo")
    if not LEGACY.exists():
        shutil.copytree(bundle / "project", LEGACY)
    for item in (bundle / "project").rglob("*"):
        if item.is_file() and sha(LEGACY / item.relative_to(bundle / "project")) != sha(item):
            raise ValueError(f"Embedded V3 differs from audited source: {item.name}")
    work = Path(cfg["v3_work"])
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for file in (*EVIDENCE_FILES, *(f"fold_{fold}.json" for fold in FOLD_FILES)):
        shutil.copy2(bundle / "evidence" / file, evidence / file)
    for name in ("topology.npz", "frozen_history.npz"):
        dest = root / "prepared" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / "prepared" / name, dest)
    original_hashes = load(evidence / "prepared_development.json")["hashes"]
    prepared = {}
    for fold in FOLD_FILES:
        path = work / "prepared" / f"{fold}.npz"
        if sha(path) != original_hashes[fold]:
            raise ValueError(f"V3 prepared array differs: {fold}")
        prepared[fold] = dict(path=str(path.resolve()), sha256=original_hashes[fold], bytes=path.stat().st_size)
    raw = load(evidence / "download_development.json")
    raw_files = []
    previous = raw["calendar_interval"][0]
    for row in raw["shards"]:
        path = work / "raw_external" / "development" / row["file"]
        if row["start"] != previous or row["status"] != "complete" or sha(path) != row["sha256"]:
            raise ValueError("V3 development source mismatch")
        if not interval_ok([row["start"], row["stop_exclusive"]], cfg["all_exclusions"]):
            raise ValueError("V3 shard intersects an embargo")
        raw_files.append(dict(path=str(path), sha256=row["sha256"], rows=row["rows"]))
        previous = row["stop_exclusive"]
    if previous != raw["calendar_interval"][1]:
        raise ValueError("V3 raw development shards do not reach calibration end")
    earlier = load(evidence / "prior_inputs.json")
    legacy_raw = [dict(path=earlier["legacy_csv"], sha256=earlier["legacy_sha256"])]
    legacy_raw += earlier["historical_development"]
    if not all(sha(item["path"]) == item["sha256"] for item in legacy_raw):
        raise ValueError("Earlier external history changed")
    old_line = next(line for line in (bundle / "reports/computed_summary.md").read_text(encoding="utf-8").splitlines()
                    if line.startswith("| R2_mean |"))
    save(evidence / "v3_historical_only.json", dict(
        source=str(bundle / "reports/computed_summary.md"), report_sha256=sha(bundle / "reports/computed_summary.md"),
        embargo=cfg["v3_known_unavailable"], published_R2_mean_row=old_line,
        use="Published aggregate only. All old holdout arrays and predictions excluded from parsing and fitting."))
    prereg = evidence / "prereg_hashes.json"
    if not prereg.exists() and (CODE.parent / "plan.md").is_file():
        shutil.copy2(CODE.parent / "evidence/prereg_hashes.json", prereg)
        shutil.copy2(CODE.parent / "evidence/prereg_time_utc.txt", evidence / "prereg_time_utc.txt")
    plan_source = root / "plan.md"
    if not plan_source.is_file():
        shutil.copy2(CODE.parent / "plan.md", plan_source)
    for document in ("README_audit.md", "method_and_data_contract.md", "test_report.md"):
        source = CODE.parent / document
        if source.is_file() and not (root / document).exists():
            shutil.copy2(source, root / document)
    pinned = load(prereg)
    if [x["sha256"] for x in pinned] != [sha(plan_source), sha(CODE / "v4_config.json")]:
        raise ValueError("V4 plan or config was changed")
    receipt = dict(status="PASS", utc=now(), archive=str(archive), bundle=str(bundle),
                   archive_sha256=archive_sha, **integrity, prepared=prepared,
                   v3_raw_dev_checked=len(raw_files), v3_raw_dev_rows=sum(r["rows"] for r in raw_files),
                   earlier_raw_checked=len(legacy_raw), exclusions=cfg["all_exclusions"],
                   scope="Hash-only archive integrity; no old holdout arrays, scores, speed or labels parsed",
                   v3_raw=raw_files, previous_sources=legacy_raw)
    save(evidence / "inputs.json", receipt)
    import psutil
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    names = ("torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "requests", "psutil")
    save(root / "logs/environment.json", dict(executable=sys.executable, python=sys.version,
        platform=platform.platform(), cpu=cpu, cores=psutil.cpu_count(False), logical=psutil.cpu_count(True),
        memory_bytes=psutil.virtual_memory().total, device="CPU", workers=original["workers"],
        threads=original["cpu_threads"], seeds=cfg["seeds"],
        packages={n: importlib.metadata.version(n) for n in names}))
    print({k: v for k, v in receipt.items() if k not in ("v3_raw", "previous_sources")}, flush=True)


def pinned_array(receipt, fold, work=None):
    if fold not in FOLD_FILES:
        raise PermissionError(f"Unauthorized assessment array {fold}")
    entry = receipt["prepared"][fold]
    path = Path(work) / "prepared" / f"{fold}.npz" if work else Path(entry["path"])
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Prepared array changed: {fold}")
    return path


if __name__ == "__main__":
    bind(*settings())

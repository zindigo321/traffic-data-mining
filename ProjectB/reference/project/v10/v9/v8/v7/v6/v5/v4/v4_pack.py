"""Archive only derived V4 artifacts and verify a clean extraction."""
import argparse
import csv
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from v4_data import verify_directory
from v4_io import load, now, save, sha

NAME = "ProjectB_GraphTransformer_V4_审计包"
DOCS = ("README_审计入口.md", "plan.md", "方法与数据合同.md", "审计报告.md", "测试报告.md")
FOLDERS = ("project", "evidence", "results", "reports", "figures", "logs", "synthetic", "prepared")


def entries(root):
    return sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*") if p.is_file())


def purpose(path):
    head = path.split("/", 1)[0]
    descriptions = dict(project="Runnable code, contract, dependencies, or tests",
                        evidence="Calendar, source hash, status or verification",
                        results="Real development/calibration scores, diagnostics, or model state",
                        reports="Method change record", figures="Measured development plot",
                        logs="Executed command, timings, environment, exit status",
                        prepared="Small hashed topology or training history",
                        synthetic="Engineering fixture result")
    return descriptions.get(head, "Human audit document")


def build(run, output):
    if load(run / "evidence/verification_replay.json")["status"] != "PASS":
        raise ValueError("No verified V4 run")
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / NAME
    bundle.mkdir()
    for path in DOCS:
        shutil.copy2(run / path, bundle / path)
    for folder in FOLDERS:
        shutil.copytree(run / folder, bundle / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.part"))
    blocked = {".plt", ".pkl", ".zip", ".gpkg", ".part"}
    records = []
    for rel, path in entries(bundle):
        if path.suffix.lower() in blocked or any(s in rel for s in ("raw_external/", "results/holdout/", "holdout.npz")):
            raise ValueError(f"Out-of-scope file {rel}")
        records.append((rel, path.stat().st_size, sha(path), purpose(rel)))
    manifest = bundle / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "size_bytes", "sha256", "purpose"))
        writer.writerows(records)
    with (bundle / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        for rel, _, digest, _ in records:
            handle.write(f"{digest}  {rel}\n")
        handle.write(f"{sha(manifest)}  manifest.csv\n")
    archive = output / (NAME + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for rel, path in entries(bundle):
            z.write(path, f"{NAME}/{rel}")
    print(dict(path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size, files=len(records)), flush=True)


def extract_check(archive, output, receipt):
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = set()
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            if (len(path.parts) < 2 or path.parts[0] != NAME or ".." in path.parts
                    or any(":" in segment or "\\" in segment for segment in path.parts)
                    or item.filename.casefold() in names or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f"Unsafe archive member {item.filename}")
            names.add(item.filename.casefold())
        if z.testzip() is not None:
            raise ValueError("ZIP CRC mismatch")
        z.extractall(output)
    bundle = output / NAME
    info = dict(status="PASS", checked_utc=now(), sha256=sha(archive), extracted=str(bundle),
                **verify_directory(bundle))
    save(receipt, info)
    print(info, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "extract-check", "verify-dir"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        build(args.run, args.output)
    elif args.stage == "extract-check":
        extract_check(args.archive, args.output, args.receipt)
    else:
        print(verify_directory(args.output), flush=True)


if __name__ == "__main__":
    main()

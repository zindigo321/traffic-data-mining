"""Build the V2 archive and verify a safe fresh extraction."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from common import digest, json_write
from check_inputs import verify_manifest

NAME = "ProjectB_GraphTransformer_V2_审计包"
META = {"manifest.csv","SHA256SUMS"}
DOCS = ("README_审计入口.md","plan.md","变更说明.md","方法与数据合同.md","审计报告.md","测试报告.md")


def files(root):
    return {p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file()}


def purpose(path):
    if path.startswith("project/"):
        return "Reproducible code, tests, fixed config or version lock"
    if path.startswith("evidence/"):
        return "Source, scope, labels, or freeze/verification receipt"
    if path.startswith("logs/"):
        return "Command, runtime and error evidence"
    if path.endswith(("weight.pt","weight.npz","preprocessor.npz")):
        return "Real trained numerical weight or preprocessing artifact"
    if path.endswith("sampling_ids.npz"):
        return "Training sampling IDs and supervision masks"
    if path.endswith(".npz"):
        return "Typed predictions, observed support, or safe numerical result"
    if path.startswith("results/"):
        return "Per-configuration real result or development selection"
    if path.startswith("figures/"):
        return "Figure generated from measured data"
    if path.startswith("synthetic/"):
        return "Engineering-only synthetic fixtures result"
    return "Audit or explanatory document"


def index(root):
    entries = [(name,p) for name,p in sorted(files(root).items()) if name not in META]
    manifest = root/"manifest.csv"
    rows = [(name,p.stat().st_size,digest(p),purpose(name)) for name,p in entries]
    with manifest.open("w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","size_bytes","sha256","purpose"])
        w.writerows(rows)
    with (root/"SHA256SUMS").open("w",encoding="utf-8",newline="\n") as f:
        for name,_,sha,_ in rows:
            f.write(f"{sha}  {name}\n")
        f.write(f"{digest(manifest)}  manifest.csv\n")
    return len(rows)


def build(run,output):
    output.mkdir(parents=True,exist_ok=False)
    root = output/NAME
    root.mkdir()
    for name in DOCS:
        shutil.copy2(run/name,root/name)
    for name in ("project","results","evidence","logs","figures","reports","synthetic"):
        src = run/name
        if src.is_dir():
            shutil.copytree(src,root/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.part","v0_first_attempt_partial"))
    prep = root/"prepared"
    prep.mkdir(exist_ok=True)
    for name in ("topology.npz","frozen_history.npz"):
        source = run/"prepared"/name
        if source.is_file():
            shutil.copy2(source,prep/name)
    for name in files(root):
        if name.endswith((".zip",".pkl",".plt",".gpkg",".part")) or "raw_external/" in name:
            raise ValueError(f"Raw/private data selected: {name}")
    rows = index(root)
    p = output/(NAME+".zip")
    with zipfile.ZipFile(p,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for name,f in sorted(files(root).items()):
            z.write(f,f"{NAME}/{name}")
    print(json.dumps(dict(zip=str(p),bytes=p.stat().st_size,sha256=digest(p),manifest_rows=rows),ensure_ascii=False))


def extract_check(archive,output,receipt):
    output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = set()
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if (p.is_absolute() or len(p.parts)<2 or p.parts[0]!=NAME or ".." in p.parts
                    or any(":" in s or "\\" in s for s in p.parts) or info.filename.casefold() in names):
                raise ValueError(f"Unsafe zip member: {info.filename}")
            names.add(info.filename.casefold())
        if z.testzip():
            raise ValueError("ZIP CRC check failed")
        z.extractall(output)
    root = output/NAME
    result = dict(status="PASS",archive_sha256=digest(archive),extract_path=str(root),
                  checked_utc=datetime.now(timezone.utc).isoformat(),**verify_manifest(root))
    if receipt:
        json_write(receipt,result)
    print(json.dumps(result,ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["build","extract-check","verify-dir"])
    p.add_argument("--run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--zip",type=Path)
    p.add_argument("--receipt",type=Path)
    a = p.parse_args()
    if a.action=="build":
        build(a.run,a.output)
    elif a.action=="extract-check":
        extract_check(a.zip,a.output,a.receipt)
    else:
        print(verify_manifest(a.output))


if __name__=="__main__":
    main()


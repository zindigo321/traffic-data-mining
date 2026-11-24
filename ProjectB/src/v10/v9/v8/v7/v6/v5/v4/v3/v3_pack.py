"""Package auditable V3 artifacts and validate a clean extraction."""
import argparse
import csv
from pathlib import Path,PurePosixPath
import shutil
import zipfile

from check_prior import verified_directory
from v3_io import now,save,sha

NAME="ProjectB_GraphTransformer_V3_audit_bundle"
DOCUMENTS=("README_audit.md","plan.md","method_and_data_contract.md","audit_report.md","test_report.md")


def files(root):
    return sorted((p.relative_to(root).as_posix(),p) for p in root.rglob("*") if p.is_file())


def purpose(rel):
    if rel.startswith("project/"):
        return "Runnable source/config/dependencies/tests"
    if rel.startswith("evidence/"):
        return "Input, split, source, freeze or verification evidence"
    if rel.startswith("logs/"):
        return "Actual command, return code, runtime or failure"
    if rel.endswith("weight.pt"):
        return "CPU Graph Transformer state_dict, weights_only"
    if rel.endswith("sampling_ids.npz"):
        return "Training IDs and observed labels"
    if rel.endswith(".npz"):
        return "Typed results, preprocessing or numeric support"
    if rel.startswith("synthetic/"):
        return "Engineering fixtures only"
    if rel.startswith("results/"):
        return "Measured model result, uncertainty or choice"
    return "Audit narrative or measured figure"


def build(run,out):
    out.mkdir(parents=True,exist_ok=False)
    root=out/NAME
    root.mkdir()
    for name in DOCUMENTS:
        shutil.copy2(run/name,root/name)
    for name in ("project","results","evidence","logs","figures","reports","synthetic"):
        source=run/name
        if source.is_dir():
            shutil.copytree(source,root/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.part"))
    (root/"prepared").mkdir()
    for name in ("topology.npz","frozen_history.npz"):
        p=run/"prepared"/name
        if p.is_file():
            shutil.copy2(p,root/"prepared"/name)
    data=[]
    for rel,p in files(root):
        if p.suffix.lower() in {".pkl",".plt",".gpkg",".zip",".part"} or "raw_external/" in rel:
            raise ValueError(f"Excluded source selected for archive: {rel}")
        data.append((rel,p.stat().st_size,sha(p),purpose(rel)))
    manifest=root/"manifest.csv"
    with manifest.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["path","size_bytes","sha256","purpose"])
        w.writerows(data)
    with (root/"SHA256SUMS").open("w",encoding="utf-8",newline="\n") as f:
        for rel,_,h,_ in data:
            f.write(f"{h}  {rel}\n")
        f.write(f"{sha(manifest)}  manifest.csv\n")
    path=out/(NAME+".zip")
    with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for rel,p in files(root):
            z.write(p,f"{NAME}/{rel}")
    print(dict(archive=str(path),size_bytes=path.stat().st_size,sha256=sha(path),manifest_rows=len(data)),flush=True)


def extract(archive,output,receipt):
    output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        seen=set()
        for m in z.infolist():
            p=PurePosixPath(m.filename)
            if (p.is_absolute() or len(p.parts)<2 or p.parts[0]!=NAME or ".." in p.parts
                    or any("\\" in x or ":" in x for x in p.parts) or m.filename.casefold() in seen):
                raise ValueError(f"Unsafe member {m.filename}")
            seen.add(m.filename.casefold())
        if z.testzip():
            raise ValueError("Invalid zip CRC")
        z.extractall(output)
    root=output/NAME
    result=dict(status="PASS",archive_sha256=sha(archive),checked_utc=now(),extracted=str(root),**verified_directory(root))
    if receipt:
        save(receipt,result)
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["build","extract-check","verify-dir"])
    p.add_argument("--run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--zip",type=Path)
    p.add_argument("--receipt",type=Path)
    a=p.parse_args()
    if a.stage=="build":
        build(a.run,a.output)
    elif a.stage=="extract-check":
        extract(a.zip,a.output,a.receipt)
    else:
        print(verified_directory(a.output))


if __name__=="__main__":
    main()


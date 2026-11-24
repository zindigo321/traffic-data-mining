"""Archive derived V5 artifacts and validate clean extraction."""
import argparse
import csv
from pathlib import Path,PurePosixPath
import shutil
import stat
import zipfile

from v5_check import verify_directory
from v5_io import load,now,save,sha

NAME="ProjectB_GraphTransformer_V5_audit_bundle"
DOCS=("README_audit.md","plan.md","protocol_nyc.md","amendment_nyc.md","method_and_data_contract.md","audit_report.md","test_report.md")
FOLDERS=("project","evidence","results","reports","figures","logs","synthetic")


def members(root):
    return sorted((p.relative_to(root).as_posix(),p) for p in root.rglob("*") if p.is_file())


def purpose(name):
    section=name.split("/",1)[0]
    groups=dict(project="Runnable V5 and pinned prior source, configuration, tests",
                evidence="Publication, input digest, data, freeze or verification receipt",
                results="Measured fit, calibration or formal external result",
                logs="Actual command, timing, exit or diagnostics",
                prepared="Pinned topology or frozen historical aggregates",
                synthetic="Engineering tests only",figures="True NYC holdout figure",reports="Change record")
    return groups.get(section,"Audit narrative or plan")


def build(run,release):
    assert load(run/"evidence/verification_replay.json")["status"]=="PASS"
    release.mkdir(parents=True,exist_ok=False)
    bundle=release/NAME
    bundle.mkdir()
    for f in DOCS:
        shutil.copy2(run/f,bundle/f)
    for d in FOLDERS:
        shutil.copytree(run/d,bundle/d,ignore=shutil.ignore_patterns(
            "__pycache__","*.pyc","*.part","nyc_dictionary.pdf","dictionary_local.txt"))
    (bundle/"prepared").mkdir()
    for name in ("topology.npz","frozen_history.npz"):
        shutil.copy2(run/"prepared"/name,bundle/"prepared"/name)
    data=[]
    for rel,path in members(bundle):
        if path.suffix.lower() in {".plt",".pkl",".zip",".part",".gpkg"} or "raw_external/" in rel:
            raise ValueError(f"Out-of-scope file {rel}")
        data.append((rel,path.stat().st_size,sha(path),purpose(rel)))
    manifest=bundle/"manifest.csv"
    with manifest.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(("path","size_bytes","sha256","purpose"))
        writer.writerows(data)
    with (bundle/"SHA256SUMS").open("w",encoding="utf-8",newline="\n") as handle:
        for rel,_,digest,_ in data:
            handle.write(f"{digest}  {rel}\n")
        handle.write(f"{sha(manifest)}  manifest.csv\n")
    file=release/(NAME+".zip")
    with zipfile.ZipFile(file,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for rel,path in members(bundle):
            z.write(path,f"{NAME}/{rel}")
    print(dict(path=str(file),sha256=sha(file),bytes=file.stat().st_size,files=len(data)),flush=True)


def extract(archive,dest,receipt):
    dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=set()
        for item in z.infolist():
            path=PurePosixPath(item.filename)
            if (len(path.parts)<2 or path.parts[0]!=NAME or ".." in path.parts or
                any(":" in x or "\\" in x for x in path.parts) or item.filename.casefold() in names
                or stat.S_ISLNK(item.external_attr>>16)):
                raise ValueError(f"Unsafe archive member: {item.filename}")
            names.add(item.filename.casefold())
        if z.testzip() is not None:
            raise ValueError("CRC mismatch")
        z.extractall(dest)
    bundle=dest/NAME
    result=dict(status="PASS",sha256=sha(archive),checked_utc=now(),extract=str(bundle),**verify_directory(bundle))
    save(receipt,result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("build","extract","verify-dir"))
    p.add_argument("--run",type=Path)
    p.add_argument("--archive",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--receipt",type=Path)
    a=p.parse_args()
    if a.action=="build":
        build(a.run,a.output)
    elif a.action=="extract":
        extract(a.archive,a.output,a.receipt)
    else:
        print(verify_directory(a.output),flush=True)

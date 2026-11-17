"""Archive V7 artifacts and verify a clean extraction."""
import argparse
import csv
from pathlib import Path,PurePosixPath
import shutil
import stat
import zipfile

from v7_check import directory as verify_directory
from v7_io import load,now,save,sha

NAME="ProjectB_GraphTransformer_V7_审计包"
DOCS=("README_审计入口.md","plan.md","重规划.md","方法与数据合同.md","审计报告.md","测试报告.md")
DIRS=("project","evidence","results","reports","figures","logs","synthetic")


def enumerate_files(root):
    return sorted((p.relative_to(root).as_posix(),p) for p in root.rglob("*") if p.is_file())


def purpose(rel):
    if rel.startswith("results/failed_calibration"):
        return "Preserved partial files from failed adapter attempt"
    groups=dict(project="Runnable source, frozen configuration and tests", evidence="Source, calendar, fit or metric receipt",
                results="Auditable development and calibration", reports="Change record",figures="Measured result plot",
                logs="Executed command, resource, time or failure",synthetic="Engineering only",prepared="Topology or frozen history")
    return groups.get(rel.split("/",1)[0],"Audit narrative or calendar plan")


def build(run,output):
    if load(run/"evidence/verification_replay.json")["status"]!="PASS":
        raise ValueError("V7 model and score verification is missing")
    output.mkdir(parents=True,exist_ok=False)
    bundle=output/NAME
    bundle.mkdir()
    for f in DOCS:
        shutil.copy2(run/f,bundle/f)
    for section in DIRS:
        shutil.copytree(run/section,bundle/section,ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.part"))
    (bundle/"prepared").mkdir()
    for f in ("topology.npz","frozen_history.npz"):
        shutil.copy2(run/"prepared"/f,bundle/"prepared"/f)
    rows=[]
    for rel,p in enumerate_files(bundle):
        if "raw_external/" in rel or p.suffix.lower() in {".pkl",".plt",".gpkg",".zip",".part"}:
            raise ValueError(f"Out-of-scope archive member {rel}")
        rows.append((rel,p.stat().st_size,sha(p),purpose(rel)))
    manifest=bundle/"manifest.csv"
    with manifest.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("path","size_bytes","sha256","purpose"))
        writer.writerows(rows)
    with (bundle/"SHA256SUMS").open("w",encoding="utf-8",newline="\n") as f:
        for rel,_,h,_ in rows:
            f.write(f"{h}  {rel}\n")
        f.write(f"{sha(manifest)}  manifest.csv\n")
    archive=output/(NAME+".zip")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for rel,p in enumerate_files(bundle):
            z.write(p,f"{NAME}/{rel}")
    print(dict(zip=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(rows)),flush=True)


def extract(archive,destination,receipt):
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        seen=set()
        for item in z.infolist():
            path=PurePosixPath(item.filename)
            if (len(path.parts)<2 or path.parts[0]!=NAME or ".." in path.parts
                    or any(":" in s or "\\" in s for s in path.parts)
                    or item.filename.casefold() in seen or stat.S_ISLNK(item.external_attr>>16)):
                raise ValueError(f"Unsafe member: {item.filename}")
            seen.add(item.filename.casefold())
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        z.extractall(destination)
    root=destination/NAME
    checked=dict(status="PASS",sha256=sha(archive),checked_utc=now(),extracted=str(root),**verify_directory(root))
    save(receipt,checked)
    print(checked,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("build","extract","verify-dir"))
    p.add_argument("--run",type=Path)
    p.add_argument("--archive",type=Path)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.action=="build":
        build(a.run,a.output)
    elif a.action=="extract":
        extract(a.archive,a.output,a.receipt)
    else:
        print(verify_directory(a.output),flush=True)


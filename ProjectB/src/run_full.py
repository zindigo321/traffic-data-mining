"""Reproduce the all-available NYC chronological experiment in an independent directory."""
import argparse
from pathlib import Path
import re
import shutil

from full_io import CODE,V10,python,record,save

PHASES=("check","fetch","prepare","tests","development","select","final","freeze","evaluate","analysis","verify","report")


def initialize(out,source):
    out.mkdir(parents=True,exist_ok=False)
    (out/"evidence").mkdir()
    (out/"raw_extension").mkdir()
    (out/"prepared").mkdir()
    (out/"figures").mkdir()
    (out/"reports").mkdir()
    shutil.copytree(CODE,out/"project",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copy2(source/"plan.md",out/"plan.md")
    shutil.copy2(source/"evidence/registration.json",out/"evidence/registration.json")


def engineering(out):
    paths=(CODE,V10,V10/"v9",V10/"v9/v8",V10/"v9/v8/v7",V10/"v9/v8/v7/v6",V10/"v9/v8/v7/v6/v5",
           V10/"v9/v8/v7/v6/v5/v4",V10/"v9/v8/v7/v6/v5/v4/v3")
    receipts=[]
    counts={}
    for label,path in zip(("full","v10","v9","v8","v7","v6","v5","v4","v3"),paths,strict=True):
        receipt=record(out,"engineering_"+label,python("-m","unittest","discover","-s","tests","-v"),cwd=path)
        txt=(out/"logs"/receipt["log"]).read_text(encoding="utf-8")
        counts[label]=int(re.search(r"Ran (\d+) tests?",txt).group(1))
        receipts.append(receipt)
    save(out/"synthetic/status.json",dict(status="PASS",engineering_only=True,counts=counts,total=sum(counts.values()),commands=receipts))


def stage(out,phase,args):
    if phase=="tests":
        engineering(out)
        return
    extra={}
    if args.source_snapshot:
        extra["GTFULL_SOURCE_SNAPSHOT"]=str(args.source_snapshot.resolve())
    opts=["--artifacts",args.artifacts] if args.artifacts else []
    replay=args.replay_root or (out if args.phase=="all" else None)
    if phase=="verify" and replay:
        opts.extend(["--replay-root",replay])
    commands=dict(check=["full_check.py"],fetch=["full_source.py"],prepare=["full_prepare.py"],
         development=["full_train.py","matrix","--stage","development"],select=["full_score.py","development"],
         final=["full_train.py","matrix","--stage","final"],freeze=["full_score.py","freeze"],
         evaluate=["full_score.py","evaluate"],analysis=["full_analysis.py"],verify=["full_verify.py",*opts],
         report=["full_report.py"])
    record(out,phase,python(*commands[phase]),extra=extra)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("all","initialize",*PHASES),default="all")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--protocol-root",type=Path,default=CODE.parent)
    p.add_argument("--source-snapshot",type=Path)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    out=a.output.resolve()
    sources=[CODE]
    sources.extend(x for x in (a.artifacts,a.replay_root) if x is not None)
    if any(out==s.resolve() or s.resolve() in out.parents for s in sources):
        raise ValueError("Write outside read-only source trees")
    if a.source_snapshot is None:
        candidate=a.protocol_root/"evidence/source_snapshot.json"
        a.source_snapshot=candidate if candidate.is_file() else None
    if a.phase in ("all","initialize"):
        initialize(out,a.protocol_root.resolve())
    else:
        out.mkdir(parents=True,exist_ok=True)
    if a.phase!="initialize":
        for phase in PHASES if a.phase=="all" else (a.phase,):
            stage(out,phase,a)


if __name__=="__main__":
    main()

"""Reproduce V8 development, calibration and freeze in an independent workspace."""
import argparse
import os
from pathlib import Path
import shutil

from v8_io import CODE,python,record
from v8_check import RELEASE,WORK

PHASES=("check","probe","test","train","analysis","freeze","verify","report")
DOCS=("plan.md","README_audit.md","method_and_data_contract.md","test_report.md")


def initialize(root,source):
    root.mkdir(parents=True,exist_ok=False)
    (root/"evidence").mkdir()
    for name in DOCS:
        path=source/name
        if path.is_file():
            shutil.copy2(path,root/name)
    shutil.copy2(source/"evidence/prereg.json",root/"evidence/prereg.json")


def stage(root,name,artifact,replay):
    cmd={
        "check":["v8_check.py"],"probe":["v8_probe.py"],"test":["v8_tests.py"],
        "train":["v8_train.py","matrix"],"analysis":["v8_analysis.py"],"freeze":["v8_freeze.py"],
        "verify":["v8_verify.py","--artifacts",artifact or root]+(["--replay-root",replay] if replay else []),
        "report":["v8_report.py"],
    }[name]
    record(root,name,python(*cmd))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("all","initialize",*PHASES),default="all")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol-root",type=Path,default=CODE.parent)
    p.add_argument("--v7-release",type=Path)
    p.add_argument("--v7-work",type=Path)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root=a.output.resolve()
    sources=[(a.v7_release or Path(RELEASE)).resolve(),(a.v7_work or Path(WORK)).resolve(),CODE]
    sources.extend(path.resolve() for path in (a.artifacts,a.replay_root) if path)
    if any(root==s or s in root.parents for s in sources):
        raise ValueError("Audit outputs must be separate from read-only inputs")
    for k,v in (("GT8_V7_RELEASE",a.v7_release),("GT8_V7_WORK",a.v7_work)):
        if v:
            os.environ[k]=str(v.resolve())
    if a.stage in ("all","initialize"):
        initialize(root,a.protocol_root.resolve())
    else:
        root.mkdir(parents=True,exist_ok=True)
    if a.stage!="initialize":
        for phase in PHASES if a.stage=="all" else (a.stage,):
            replay=(a.v7_work or Path(WORK)) if a.stage=="all" else a.replay_root
            stage(root,phase,a.artifacts,replay)


if __name__=="__main__":
    main()

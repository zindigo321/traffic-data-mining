"""Reproducible V5 one-shot experiment and audit entry."""
import argparse
import os
from pathlib import Path
import shutil

from v5_io import CODE,python,record

STAGES=("check","download","prepare","test","screen","confirm","calibrate","freeze","evaluate","verify","report")
EVIDENCE=("registration.json","amendment.json","dictionary_receipt.json","feasibility.json",
          "coverage_only.json","chicago_status.json")


def start_reproduction(root,source):
    (root/"evidence").mkdir(parents=True,exist_ok=True)
    for doc in ("plan.md","protocol_nyc.md","amendment_nyc.md","README_审计入口.md","方法与数据合同.md","测试报告.md"):
        origin=source/doc
        if origin.is_file():
            shutil.copy2(origin,root/doc)
    for name in EVIDENCE:
        shutil.copy2(source/"evidence"/name,root/"evidence"/name)


def phase(root,stage,artifact,replay):
    mapping={
        "check":["v5_check.py"],"probe":["v5_probe.py"],"download":["v5_download.py"],
        "prepare":["v5_prepare.py"],"test":["v5_tests.py"],"screen":["v5_matrix.py","screen"],
        "confirm":["v5_matrix.py","confirm"],"calibrate":["v5_matrix.py","calibrate"],
        "freeze":["v5_lock.py"],"evaluate":["v5_eval.py"],"report":["v5_report.py"],
        "verify":["v5_verify.py","--artifacts",artifact or root]+(["--replay-root",replay] if replay else []),
    }
    record(root,stage,python(*mapping[stage]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("all","initialize","probe",*STAGES),default="all")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol-root",type=Path,default=CODE.parent)
    p.add_argument("--raw-cache",type=Path)
    p.add_argument("--v4-release",type=Path)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root=a.output.resolve()
    if a.artifacts and (root==a.artifacts.resolve() or a.artifacts.resolve() in root.parents):
        raise ValueError("Verification must write outside the artifacts")
    if a.stage in ("all","initialize"):
        if root.exists():
            raise FileExistsError("Use a fresh reproduction directory")
        start_reproduction(root,a.protocol_root.resolve())
    else:
        root.mkdir(parents=True,exist_ok=True)
    for k,v in (("GT5_RAW",a.raw_cache),("GT5_V4_RELEASE",a.v4_release)):
        if v:
            os.environ[k]=str(v.resolve())
    if a.stage=="initialize":
        return
    for stage in STAGES if a.stage=="all" else (a.stage,):
        replay=a.replay_root if a.stage!="all" else root
        phase(root,stage,a.artifacts,replay)


if __name__=="__main__":
    main()

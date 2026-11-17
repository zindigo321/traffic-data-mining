"""V6 reproducible experiment and verification entry point."""
import argparse
import os
from pathlib import Path
import shutil

from v6_check import RELEASE,WORK
from v6_io import CODE,python,record

STAGES=("check","download","prepare","test","calibrate","freeze","evaluate","verify","report")
DOCS=("plan.md","registered_holdout.md","README_审计入口.md","方法与数据合同.md","测试报告.md")
EVIDENCE=("plan_receipt.json","date_gate.json","registration.json","chicago_status.json")


def initialize(root,source):
    root.mkdir(parents=True,exist_ok=False)
    (root/"evidence").mkdir()
    for f in DOCS:
        if (source/f).is_file():
            shutil.copy2(source/f,root/f)
    for f in EVIDENCE:
        shutil.copy2(source/"evidence"/f,root/"evidence"/f)


def stage(root,phase,artifact,replay):
    commands={
        "check":["v6_check.py"],"download":["v6_download.py"],"prepare":["v6_prepare.py"],
        "test":["v6_tests.py"],"calibrate":["v6_train.py","calibrate"],"freeze":["v6_lock.py"],
        "evaluate":["v6_eval.py"],"report":["v6_report.py"],
        "verify":["v6_verify.py","--artifacts",artifact or root]+(["--replay-root",replay] if replay else []),
    }
    record(root,phase,python(*commands[phase]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("all","initialize",*STAGES),default="all")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol-root",type=Path,default=CODE.parent)
    p.add_argument("--v5-release",type=Path)
    p.add_argument("--v5-work",type=Path)
    p.add_argument("--held-cache",type=Path)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root=a.output.resolve()
    sources=[(a.v5_release or Path(RELEASE)).resolve(),(a.v5_work or Path(WORK)).resolve(),CODE]
    if a.artifacts:
        sources.append(a.artifacts.resolve())
    if a.held_cache:
        sources.append(a.held_cache.resolve())
    if any(root==s or s in root.parents for s in sources):
        raise ValueError("Results must stay outside read-only inputs")
    for env,path in (("GT6_V5_RELEASE",a.v5_release),("GT6_V5_WORK",a.v5_work),("GT6_HELD_SOURCE",a.held_cache)):
        if path:
            os.environ[env]=str(path.resolve())
    if a.stage in ("all","initialize"):
        initialize(root,a.protocol_root.resolve())
    else:
        root.mkdir(parents=True,exist_ok=True)
    if a.stage!="initialize":
        for phase in STAGES if a.stage=="all" else (a.stage,):
            replay=root if a.stage=="all" else a.replay_root
            stage(root,phase,a.artifacts,replay)


if __name__=="__main__":
    main()

"""One-click reproducibility of V7 development and calibration."""
import argparse
import os
from pathlib import Path
import shutil

from v7_check import RELEASE,WORK
from v7_io import CODE,python,record

PHASES=("check","probe","prepare","test","development","calibration","freeze","analysis","verify","report")
DOCS=("plan.md","重规划.md","README_审计入口.md","方法与数据合同.md","测试报告.md")


def initialize(out,source):
    out.mkdir(parents=True,exist_ok=False)
    (out/"evidence").mkdir()
    for name in DOCS:
        if (source/name).is_file():
            shutil.copy2(source/name,out/name)
    shutil.copy2(source/"evidence/prereg.json",out/"evidence/prereg.json")


def stage(out,phase,artifacts,replay):
    commands={
        "check":["v7_check.py"],"probe":["v7_probe.py"],"prepare":["v7_prepare.py"],"test":["v7_tests.py"],
        "development":["v7_matrix.py","development"],"calibration":["v7_matrix.py","calibration"],
        "freeze":["v7_freeze.py"],"analysis":["v7_analysis.py"],"report":["v7_report.py"],
        "verify":["v7_verify.py","--artifacts",artifacts or out]+(["--replay-root",replay] if replay else []),
    }
    record(out,phase,python(*commands[phase]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("all","initialize",*PHASES),default="all")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol-root",type=Path,default=CODE.parent)
    p.add_argument("--v6-release",type=Path)
    p.add_argument("--v5-work",type=Path)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    out=a.output.resolve()
    protected=[(a.v6_release or Path(RELEASE)).resolve(),(a.v5_work or Path(WORK)).resolve(),CODE]
    if a.artifacts:
        protected.append(a.artifacts.resolve())
    if a.replay_root:
        protected.append(a.replay_root.resolve())
    if any(out==item or item in out.parents for item in protected):
        raise ValueError("Write results outside read-only inputs")
    for k,v in (("GT7_V6_RELEASE",a.v6_release),("GT7_V5_WORK",a.v5_work)):
        if v:
            os.environ[k]=str(v.resolve())
    if a.stage in ("all","initialize"):
        initialize(out,a.protocol_root.resolve())
    else:
        out.mkdir(parents=True,exist_ok=True)
    if a.stage!="initialize":
        for phase in PHASES if a.stage=="all" else (a.stage,):
            stage(out,phase,a.artifacts,out if a.stage=="all" else a.replay_root)


if __name__=="__main__":
    main()

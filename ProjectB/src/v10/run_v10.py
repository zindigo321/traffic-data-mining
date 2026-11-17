"""One-click V10 training, independent calibration, verification, and report."""
import argparse
import os
from pathlib import Path
import re
import shutil

from v10_check import PREPARED, RELEASE
from v10_io import CODE, V9, python, record, save

PHASES = ("check", "probe", "test", "development", "select", "calibration", "freeze", "analysis", "verify", "report")
V7_PROTOCOL = Path("G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V7/ProjectB_GraphTransformer_V7_审计包")


def initialize(output, source):
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence").mkdir()
    shutil.copytree(CODE, output / "project", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source / "plan.md", output / "plan.md")
    shutil.copy2(source / "evidence/prereg.json", output / "evidence/prereg.json")


def tests(output):
    paths = (CODE, V9, V9 / "v8", V9 / "v8/v7", V9 / "v8/v7/v6", V9 / "v8/v7/v6/v5",
             V9 / "v8/v7/v6/v5/v4", V9 / "v8/v7/v6/v5/v4/v3")
    counts, receipts = {}, []
    for version, path in zip(range(10, 2, -1), paths, strict=True):
        name = f"v{version}"
        receipt = record(output, f"engineering_{name}", python("-m", "unittest", "discover", "-s", "tests", "-v"), cwd=path)
        text = (output / "logs" / receipt["log"]).read_text(encoding="utf-8")
        counts[name] = int(re.search(r"Ran (\d+) tests?", text).group(1))
        receipts.append(receipt)
    save(output / "synthetic/status.json", dict(status="PASS", counts=counts, total=sum(counts.values()),
                                               engineering_only=True, commands=receipts))


def stage(output, phase, args):
    if phase == "test":
        tests(output)
        return
    if phase == "prepare":
        for sub in ("initialize", "check", "prepare"):
            record(output, f"rebuild_v7_{sub}", python(V9 / "v8/v7/run_v7.py", "--stage", sub,
                      "--output", output / "rebuilt", "--protocol-root", args.v7_protocol))
        return
    if phase == "probe" and (output / "evidence/freeze.json").exists():
        raise PermissionError("Keep frozen date evidence and write a new probe to an independent scratch directory")
    extra = ["--artifacts", args.artifacts] if args.artifacts else []
    replay = args.replay_root or (args.prepared.parent if args.stage == "all" else None)
    if phase == "verify" and replay:
        extra.extend(["--replay-root", replay])
    cmd = {"check": ("v10_check.py",), "probe": ("v10_probe.py",), "development": ("v10_train.py", "development"),
           "select": ("v10_score.py", "development"), "calibration": ("v10_train.py", "calibration"),
           "freeze": ("v10_score.py", "calibration"), "analysis": ("v10_analysis.py",),
           "verify": ("v10_verify.py", *extra), "report": ("v10_report.py",)}[phase]
    record(output, phase, python(*cmd), extra={"GT10_V9_RELEASE": str(args.v9_release.resolve()),
                                               "GT10_PREPARED": str(args.prepared.resolve())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "initialize", "prepare", *PHASES), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, default=CODE.parent)
    parser.add_argument("--v9-release", type=Path, default=Path(RELEASE))
    parser.add_argument("--prepared", type=Path, default=Path(PREPARED))
    parser.add_argument("--v7-protocol", type=Path, default=V7_PROTOCOL)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--replay-root", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    protected = [args.v9_release, args.prepared.parent, args.v7_protocol, CODE]
    protected.extend(p for p in (args.artifacts, args.replay_root) if p)
    if any(output == p.resolve() or p.resolve() in output.parents for p in protected):
        raise ValueError("Use an independent output outside input sources")
    if args.artifacts and args.stage != "verify":
        raise ValueError("Artifact inputs are limited to read-only verification")
    if args.stage in ("all", "initialize"):
        initialize(output, args.protocol_root.resolve())
    else:
        output.mkdir(parents=True, exist_ok=True)
    if args.stage != "initialize":
        for phase in PHASES if args.stage == "all" else (args.stage,):
            stage(output, phase, args)


if __name__ == "__main__":
    main()

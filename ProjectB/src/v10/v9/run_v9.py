"""Reproduce V9 policy fitting or verify a read-only audit bundle."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys

from v9_check import PREPARED, RELEASE
from v9_io import CODE, V8, python, record, save

PHASES = ("check", "probe", "test", "develop", "calibrate", "analysis", "verify", "report")
V7_PROTOCOL = Path("G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V7/ProjectB_GraphTransformer_V7_审计包")


def initialize(out, source):
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    shutil.copytree(CODE, out / "project", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source / "plan.md", out / "plan.md")
    shutil.copy2(source / "evidence/prereg.json", out / "evidence/prereg.json")


def tests(out):
    command = python("-m", "unittest", "discover", "-s", "tests", "-v")
    paths = [CODE, V8, V8 / "v7", V8 / "v7/v6", V8 / "v7/v6/v5",
             V8 / "v7/v6/v5/v4", V8 / "v7/v6/v5/v4/v3"]
    counts, receipts = {}, []
    for number, path in zip(range(9, 2, -1), paths, strict=True):
        name = f"v{number}"
        receipt = record(out, f"engineering_{name}", command, cwd=path)
        report = (out / "logs" / receipt["log"]).read_text(encoding="utf-8")
        counts[name] = int(re.search(r"Ran (\d+) tests?", report).group(1))
        receipts.append(receipt)
    save(out / "synthetic/status.json", dict(status="PASS", engineering_only=True, counts=counts, total=sum(counts.values()),
                                             commands=receipts))


def stage(out, phase, args, prepared):
    if phase == "test":
        tests(out)
        return
    if phase == "prepare":
        command = V8 / "v7/run_v7.py"
        for sub in ("initialize", "check", "prepare"):
            record(out, f"rebuild_v7_{sub}", python(command, "--stage", sub, "--protocol-root", args.v7_protocol,
                                                   "--output", out / "rebuilt"))
        return
    if phase == "probe" and (out / "evidence/freeze.json").exists():
        raise PermissionError("Write the time-only recheck to a separate scratch directory")
    opts = ["--artifacts", args.artifacts] if args.artifacts else []
    replay = args.replay_root or (prepared.parent if args.stage == "all" else None)
    if phase == "verify" and replay:
        opts.extend(["--replay-root", replay])
    commands = {"check": ("v9_check.py",), "probe": ("v9_probe.py",), "develop": ("v9_policy.py", "develop"),
                "calibrate": ("v9_policy.py", "calibrate"), "analysis": ("v9_analysis.py",),
                "verify": ("v9_verify.py", *opts), "report": ("v9_report.py",)}
    record(out, phase, python(*commands[phase]), extra={"GT9_V8_RELEASE": str(args.v8_release.resolve()),
                                                      "GT9_PREPARED": str(prepared.resolve())})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("all", "initialize", "prepare", *PHASES), default="all")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--protocol-root", type=Path, default=CODE.parent)
    ap.add_argument("--v8-release", type=Path, default=Path(RELEASE))
    ap.add_argument("--prepared", type=Path, default=Path(PREPARED))
    ap.add_argument("--v7-protocol", type=Path, default=V7_PROTOCOL)
    ap.add_argument("--artifacts", type=Path)
    ap.add_argument("--replay-root", type=Path)
    args = ap.parse_args()
    out = args.output.resolve()
    sources = [args.v8_release, args.prepared.parent, args.v7_protocol, CODE]
    sources.extend(x for x in (args.artifacts, args.replay_root) if x is not None)
    if any(p.resolve() == out or p.resolve() in out.parents for p in sources):
        raise ValueError("Write only to an independent run directory")
    if args.artifacts and args.stage != "verify":
        raise ValueError("Artifacts are an input only to read-only verification")
    if args.stage in ("all", "initialize"):
        initialize(out, args.protocol_root.resolve())
    else:
        out.mkdir(parents=True, exist_ok=True)
    if args.stage != "initialize":
        for phase in PHASES if args.stage == "all" else (args.stage,):
            stage(out, phase, args, args.prepared)


if __name__ == "__main__":
    main()

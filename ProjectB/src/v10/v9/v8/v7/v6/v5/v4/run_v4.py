"""Reproducible V4 stages with a development-only formal holdout gate."""
import argparse
import os
from pathlib import Path

from v4_io import CODE, LEGACY, load, python, record, save

STAGES = ("check", "probe", "test", "review", "screen", "confirm", "calibrate", "analyze", "verify", "report")


def run_stage(root, stage, artifacts, replay):
    if stage == "test":
        v4 = record(root, "engineering_v4", python("-m", "unittest", "discover", "-s", "tests", "-v"))
        v3 = record(root, "engineering_v3", python("-m", "unittest", "discover", "-s", "tests", "-v"), cwd=LEGACY)
        save(root / "synthetic/status.json", dict(status="PASS", nature="synthetic engineering fixtures only",
            logs=[v4["log"], v3["log"]], v4_tests=8, inherited_v3_tests=10))
        return
    commands = {
        "check": [("source_check", ["v4_data.py"])],
        "probe": [("holdout_gate", ["v4_probe.py"])],
        "review": [("initial_error_review", ["v4_errors.py", "--previous"])],
        "screen": [("screen", ["v4_matrix.py", "screen"])],
        "confirm": [("confirm", ["v4_matrix.py", "confirm"])],
        "calibrate": [("calibrate", ["v4_matrix.py", "calibrate"])],
        "analyze": [("paired_error_review", ["v4_errors.py"]), ("analyze", ["v4_analysis.py"])],
        "verify": [("independent_verification", ["v4_verify.py", "--artifacts", artifacts or root]
                    + (["--replay"] if replay else []))],
        "report": [("report", ["v4_report.py"])],
    }
    for name, cmd in commands[stage]:
        record(root, name, python(*cmd))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--v3-work", type=Path)
    parser.add_argument("--v3-release", type=Path)
    parser.add_argument("--artifacts", type=Path, help="Read result bundle for the verify stage")
    parser.add_argument("--replay", action="store_true", help="Add hash-checked external weight replay")
    args = parser.parse_args()
    cfg = load(CODE / "v4_config.json")
    old_work = (args.v3_work or Path(cfg["v3_work"])).resolve()
    old_release = (args.v3_release or Path(cfg["v3_release"])).resolve()
    root = args.output.resolve()
    # Writes always go to a new V4 workspace or an existing V4 run.
    for source in (old_work, old_release, CODE):
        if root == source or source in root.parents:
            raise ValueError("V4 output cannot be inside a read-only source or code directory")
    if args.artifacts and (root == args.artifacts.resolve() or args.artifacts.resolve() in root.parents):
        raise ValueError("Verification receipts must stay outside the checked artifacts")
    if args.stage == "all" and root.exists():
        raise FileExistsError("A fresh output directory is required for all")
    if root.exists() and not ((root / "plan.md").is_file() or (root / "logs").is_dir()):
        raise ValueError("Existing output is not a V4 run")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GT4_V3_WORK"], os.environ["GT4_V3_RELEASE"] = str(old_work), str(old_release)
    for stage in STAGES if args.stage == "all" else (args.stage,):
        run_stage(root, stage, args.artifacts, args.replay or args.stage == "all")


if __name__ == "__main__":
    main()

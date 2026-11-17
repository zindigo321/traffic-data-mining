import argparse
import os
from pathlib import Path
import sys
from common import record

cli = argparse.ArgumentParser()
cli.add_argument("--root", type=Path, required=True)
cli.add_argument("--name", required=True)
cli.add_argument("command", nargs=argparse.REMAINDER)
a = cli.parse_args()
env = {"GT2_ROOT": str(a.root.resolve())}
for key in ("GT2_CONFIG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    if key in os.environ:
        env[key] = os.environ[key]
sys.exit(record(a.root, a.name, a.command[1:] if a.command[0] == "--" else a.command, env))


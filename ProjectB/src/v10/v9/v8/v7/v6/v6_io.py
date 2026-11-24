"""Independent V6 paths, integrity primitives and command receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
CONFIG = CODE / "config.json"
V5 = CODE / "v5"
V5_NAME = "ProjectB_GraphTransformer_V5_\u5ba1\u8ba1\u5305"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def settings():
    return Path(os.environ["GT6_ROOT"]).resolve(), load(CONFIG)


def python(*args):
    return [sys.executable, "-B", "-X", "utf8", *map(str, args)]


def record(root, stage, cmd, cwd=CODE, extra=None):
    logdir = root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = logdir / f"{stamp}_{stage}.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "GT6_ROOT": str(root.resolve()), **(extra or {})}
    dll = Path(sys.prefix) / "Library/bin"
    if sys.platform == "win32" and dll.is_dir():
        env["PATH"] = str(dll) + os.pathsep + env.get("PATH", "")
    started, t0 = now(), time.perf_counter()
    print(f"Running {stage} -> {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = dict(stage=stage, command=cmd, cwd=str(cwd), started_utc=started, ended_utc=now(),
                   elapsed_seconds=round(time.perf_counter()-t0, 4), exit_code=proc.returncode, log=log.name,
                   paths={k: env[k] for k in ("GT6_ROOT", "GT6_HELD_SOURCE", "GT6_V5_RELEASE", "GT6_V5_WORK") if k in env})
    save(log.with_suffix(".json"), receipt)
    if proc.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-5000:], flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return receipt



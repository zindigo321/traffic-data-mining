"""Local paths, digests, and per-command audit receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def settings():
    root = Path(os.environ["GT2_ROOT"]).resolve()
    config_path = Path(os.environ.get("GT2_CONFIG", CODE / "config.json")).resolve()
    return root, json_read(config_path)


def record(root, name, command, extra=None, cwd=CODE):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = logs / f"{stamp}_{name}.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", **(extra or {})}
    dll = Path(sys.prefix) / "Library" / "bin"
    if os.name == "nt" and dll.is_dir():
        env["PATH"] = str(dll) + os.pathsep + env.get("PATH", "")
    started = now()
    t = time.perf_counter()
    print(f"Running {name} -> {log}", flush=True)
    with log.open("w", encoding="utf-8") as file:
        rc = subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=file, stderr=subprocess.STDOUT).returncode
    receipt = dict(stage=name, command=list(map(str, command)), cwd=str(cwd), started_utc=started,
                   completed_utc=now(), elapsed_seconds=round(time.perf_counter() - t, 4),
                   exit_code=rc, environment=extra or {}, log=log.name)
    json_write(log.with_suffix(".json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return rc



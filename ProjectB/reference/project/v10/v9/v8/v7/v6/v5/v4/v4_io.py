"""Local V4 paths, stable evidence, and subprocess receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
LEGACY = CODE / "v3"
CONFIG = CODE / "v4_config.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def settings():
    root = Path(os.environ["GT4_ROOT"]).resolve()
    cfg = load(CONFIG)
    for key in ("v3_work", "v3_release"):
        cfg[key] = os.environ.get(f"GT4_{key.upper()}", cfg[key])
    return root, cfg


def legacy_config():
    return load(LEGACY / "config.json")


def python(*args):
    return [sys.executable, "-B", "-X", "utf8", *map(str, args)]


def record(root, name, command, cwd=CODE):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = logs / f"{stamp}_{name}.log"
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "GT4_ROOT": str(root.resolve())}
    dll = Path(sys.prefix) / "Library/bin"
    if sys.platform == "win32" and dll.is_dir():
        environment["PATH"] = str(dll) + os.pathsep + environment.get("PATH", "")
    started, begin = now(), time.perf_counter()
    print(f"Running {name} -> {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    receipt = dict(stage=name, command=command, cwd=str(cwd), started_utc=started, ended_utc=now(),
                   elapsed_seconds=round(time.perf_counter() - begin, 4), exit_code=result.returncode,
                   log=log.name, environment={key: environment[key] for key in
                                              ("GT4_ROOT", "GT4_V3_WORK", "GT4_V3_RELEASE") if key in environment})
    save(log.with_suffix(".json"), receipt)
    if result.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-5000:], flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    return receipt

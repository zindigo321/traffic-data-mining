"""Paths and durable per-command receipts."""
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


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path,obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def context():
    return Path(os.environ["GT3_ROOT"]).resolve(), load(os.environ.get("GT3_CONFIG",CODE/"config.json"))


def record(root, name, command, cwd=CODE, extra=None):
    logdir = root/"logs"
    logdir.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = logdir/f"{stamp}_{name}.log"
    environment = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONUTF8":"1",**(extra or {})}
    dll = Path(sys.prefix)/"Library/bin"
    if os.name=="nt" and dll.is_dir():
        environment["PATH"] = str(dll)+os.pathsep+environment.get("PATH","")
    begin, started = time.perf_counter(),now()
    cmd = list(map(str,command))
    print(f"Running {name} -> {log}",flush=True)
    with log.open("w",encoding="utf-8") as out:
        rc = subprocess.run(cmd,cwd=cwd,env=environment,stdout=out,stderr=subprocess.STDOUT).returncode
    receipt = dict(stage=name,command=cmd,cwd=str(cwd),started_utc=started,ended_utc=now(),
                   elapsed_seconds=round(time.perf_counter()-begin,4),exit_code=rc,log=log.name,
                   run_environment={key:environment[key] for key in ("GT3_ROOT","GT3_CONFIG","GT2_ROOT","GT2_CONFIG","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS") if key in environment})
    save(log.with_suffix(".json"),receipt)
    print(receipt,flush=True)
    return rc


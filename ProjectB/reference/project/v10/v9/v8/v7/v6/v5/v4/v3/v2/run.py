"""One-command or staged V2 run. All writes are confined to the specified run directory."""
import argparse
import os
from pathlib import Path
import shutil
import sys

from common import CODE, digest, json_read, json_write, now, record

STAGES = ("check","download","prepare","test","screen","confirm","freeze","evaluate","verify","report")


def attach_download_cache(source, root):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for name in ("probe.json","chicago_publisher.json","download_development.json","download_holdout_sealed.json"):
        src = source/"evidence"/name
        if not src.is_file():
            continue
        data = json_read(src)
        if name.startswith("download_") and data["status"] == "complete":
            for item in data["shards"]:
                path = Path(data["directory"])/item["file"]
                if digest(path)!=item["sha256"]:
                    raise ValueError(f"Copied cache receipt differs from file: {path}")
        dest = root/"evidence"/name
        if dest.exists():
            if digest(dest)!=digest(src):
                raise ValueError(f"Run already has a different download receipt: {name}")
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest)
    json_write(root/"evidence/attached_cache.json",dict(read_only_source=str(source),at_utc=now()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["all",*STAGES],default="all")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--config",type=Path,default=CODE/"config.json")
    p.add_argument("--cache-run",type=Path,help="Use SHA-verified Chicago shards from another run without copying raw data")
    p.add_argument("--artifacts",type=Path,help="Verify saved audit package instead of this run folder")
    p.add_argument("--replay-root",type=Path,help="Optional external root with SHA-verified holdout prepared arrays")
    a = p.parse_args()
    conf = a.config.resolve()
    root = a.output.resolve()
    cfg = json_read(conf)
    for prohibited in (Path(cfg["v1_zip"]).resolve().parent, Path(cfg["v1_cache"]).resolve(), CODE):
        if root.is_relative_to(prohibited):
            raise ValueError("Output must be separate from v1 audit/cache and the executable code directory")
    root.mkdir(parents=True,exist_ok=a.stage!="all")
    env = {"GT2_ROOT":str(root),"GT2_CONFIG":str(conf)}
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        env[k]=str(cfg["cpu_threads"])
    stamp = now().replace(":","").replace("+","_")
    if (CODE.parent/"plan.md").exists() and not (root/"plan.md").exists():
        shutil.copy2(CODE.parent/"plan.md",root/"plan.md")
    hashes = {f.relative_to(CODE).as_posix():digest(f) for f in CODE.rglob("*.py")}
    json_write(root/"logs"/f"invocation_{stamp}.json",dict(
        command=[sys.executable,*sys.argv],at_utc=now(),config=str(conf),config_sha256=digest(conf),
        prereg_plan_sha256=digest(root/"plan.md") if (root/"plan.md").exists() else None,
        code_sha256=hashes,env=env))
    def stage(name,script,*args,optional=False):
        target = script if script.startswith("-") else CODE/script
        rc = record(root,name,[sys.executable,"-B","-X","utf8",target,*args],env)
        if rc and not optional:
            raise SystemExit(rc)
        return rc

    for step in STAGES if a.stage=="all" else (a.stage,):
        if step=="check":
            if not (root/"evidence/v1_inputs.json").exists():
                stage("verify_inputs","check_inputs.py")
            if not (root/"results/v0_regression/status.json").exists():
                stage("v0_regression","v0_regression.py")
        elif step=="download":
            if a.cache_run:
                attach_download_cache(a.cache_run.resolve(),root)
            else:
                if not (root/"evidence/probe.json").exists():
                    stage("probe","download_chicago.py","probe")
                for scope in ("development","holdout_sealed"):
                    receipt = root/"evidence"/f"download_{scope}.json"
                    if not receipt.is_file() or json_read(receipt)["status"]!="complete":
                        stage(f"download_{scope}","download_chicago.py",scope,optional=True)
        elif step=="prepare":
            if not (root/"evidence/f1_exact_support.json").exists():
                stage("prepare_local","prepare_chicago.py","local")
            receipt = root/"evidence/download_development.json"
            if receipt.is_file() and json_read(receipt)["status"]=="complete" and not (root/"prepared/final.npz").exists():
                stage("prepare_extended","prepare_chicago.py","extended")
        elif step=="test":
            stage("unit_tests","-m","unittest","discover","-s",CODE/"tests","-v")
            json_write(root/"synthetic/test_scope.json",dict(
                status="PASS",data_source="SYNTHETIC_TEST_FIXTURES_ONLY",
                scope="Causal mutation, graph/auxiliary gradients, sampling, partitions, metric and artifact tests",
                traffic_performance=False))
        elif step in ("screen","confirm","freeze"):
            stage(step,"matrix.py",step)
        elif step=="evaluate":
            stage("evaluate_holdout","holdout.py")
        elif step=="verify":
            args = ["--artifacts",(a.artifacts or root).resolve(),"--output",root/"evidence"/f"verification_{stamp}.json"]
            if a.replay_root or ((root/"prepared/holdout.npz").exists() and a.artifacts is None):
                args += ["--replay-root",(a.replay_root or root).resolve()]
            stage("verify_v2","verify_v2.py",*args)
        else:
            stage("report","report.py")


if __name__=="__main__":
    main()


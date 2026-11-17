"""One command or staged V3 run. It keeps prior archives and raw caches read-only."""
import argparse
from pathlib import Path
import shutil
import sys

from v3_io import CODE, load, now, record, save, sha

PHASES=("check","download","prepare","test","screen","confirm","freeze","evaluate","verify","report")


def use_cache(src,dst):
    d=dst/"evidence"
    d.mkdir(exist_ok=True)
    for filename in ("probe.json","chicago_publisher.json","download_development.json","download_holdout_sealed.json"):
        file=src/"evidence"/filename
        if not file.exists():
            raise ValueError(f"Missing cache receipt {filename}")
        data=load(file)
        if filename.startswith("download_"):
            if data["status"]!="complete":
                raise ValueError("Cached downloads are incomplete")
            for chunk in data["shards"]:
                if sha(Path(data["directory"])/chunk["file"])!=chunk["sha256"]:
                    raise ValueError(f"Cache changed: {chunk['file']}")
        target=d/filename
        if target.exists() and sha(target)!=sha(file):
            raise ValueError("Different receipt already present")
        if not target.exists():
            shutil.copy2(file,target)
    save(d/"cache_attached.json",dict(source=str(src),at_utc=now(),values_parsed=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["all",*PHASES],default="all")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,default=CODE/"config.json")
    p.add_argument("--cache-run",type=Path)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--replay-root",type=Path)
    a=p.parse_args()
    root,config=a.output.resolve(),a.config.resolve()
    cfg=load(config)
    for path in (CODE,Path(cfg["v2_bundle"]).resolve().parent,Path(cfg["v2_cache"]).resolve()):
        if root.is_relative_to(path):
            raise ValueError("Output must be separate from prior results, cache and code")
    root.mkdir(parents=True,exist_ok=a.stage!="all")
    if not (root/"plan.md").exists():
        shutil.copy2(CODE.parent/"plan.md",root/"plan.md")
    env=dict(GT3_ROOT=str(root),GT3_CONFIG=str(config),GT2_ROOT=str(root),GT2_CONFIG=str(config))
    env.update({n:str(cfg["cpu_threads"]) for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    stamp=now().replace(":","").replace("+","_")
    save(root/"logs"/f"invocation_{stamp}.json",dict(command=[sys.executable,*sys.argv],
         config_sha256=sha(config),plan_sha256=sha(root/"plan.md"),env=env))
    def step(name,script,*args,nonfatal=False):
        path=script if script.startswith("-") else CODE/script
        rc=record(root,name,[sys.executable,"-B","-X","utf8",path,*args],extra=env)
        if rc and not nonfatal:
            raise SystemExit(rc)
        return rc

    for s in PHASES if a.stage=="all" else (a.stage,):
        if s=="check":
            if not (root/"evidence/prior_inputs.json").exists():
                step("check_prior","check_prior.py")
        elif s=="download":
            if a.cache_run:
                use_cache(a.cache_run.resolve(),root)
            else:
                if not (root/"evidence/probe.json").exists():
                    step("probe","v2/download_chicago.py","probe")
                for scope in ("development","holdout_sealed"):
                    p=root/"evidence"/f"download_{scope}.json"
                    if not p.exists() or load(p)["status"]!="complete":
                        step(f"download_{scope}","v2/download_chicago.py",scope,nonfatal=True)
        elif s=="prepare":
            p=root/"evidence/download_development.json"
            if not p.is_file() or load(p)["status"]!="complete":
                save(root/"evidence/blocked.json",dict(stage="prepare",reason="Fixed development data unavailable"))
            elif not (root/"evidence/prepared_development.json").exists():
                step("prepare_v3","v3_prepare.py")
        elif s=="test":
            step("test_v3","-m","unittest","discover","-s",CODE/"tests","-v")
            save(root/"synthetic/status.json",dict(status="PASS",usage="Engineering fixtures only",traffic_performance=False))
        elif s in ("screen","confirm","freeze"):
            step(s,"v3_matrix.py",s)
        elif s=="evaluate":
            step("holdout","v3_eval.py")
        elif s=="verify":
            args=["--artifacts",(a.artifacts or root).resolve(),"--receipt",root/"evidence"/f"verification_{stamp}.json"]
            if a.replay_root or ((root/"prepared/holdout.npz").exists() and not a.artifacts):
                args+=["--replay-root",(a.replay_root or root).resolve()]
            step("verify","v3_verify.py",*args)
        else:
            step("report","v3_report.py")


if __name__=="__main__":
    main()


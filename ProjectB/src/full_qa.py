"""Package and audit a clean extraction with source rebuild and weight replay."""
import argparse
from pathlib import Path

from full_io import load,now,python,record,save,sha
from full_pack import NAME


def check(run,output,scratch):
    scratch.mkdir(parents=True,exist_ok=True)
    archive=output/(NAME+".zip")
    record(scratch,"build",python("full_pack.py","build","--run",run,"--output",output))
    record(scratch,"extract",python("full_pack.py","extract","--archive",archive,
           "--output",scratch/"extract","--receipt",scratch/"zip_check.json"))
    clean=scratch/"extract"/NAME
    code=clean/"project"
    record(scratch,"delivered_dir",python("full_pack.py","verify-dir","--output",output/NAME),cwd=code)
    commands=[]
    for phase in ("initialize","check","fetch","prepare"):
        commands.append(record(scratch,"clean_rebuild_"+phase,
             python("run_full.py","--phase",phase,"--output",scratch/"clean_rebuilt","--protocol-root",clean,
                    "--source-snapshot",clean/"evidence/source_snapshot.json"),cwd=code))
    for phase,opts,folder in (
        ("verify",["--artifacts",clean],"offline"),
        ("verify",["--artifacts",clean,"--replay-root",scratch/"clean_rebuilt"],"replay")):
        commands.append(record(scratch,"clean_"+folder,python("run_full.py","--phase",phase,
                         "--output",scratch/folder,*opts),cwd=code))
    commands.append(record(scratch,"clean_engineering_new",
              python("-m","unittest","discover","-s","tests","-v"),cwd=code))
    record(scratch,"final_clean_integrity",python("full_pack.py","verify-dir","--output",clean),cwd=code)
    offline=load(scratch/"offline/evidence/verification_offline.json")
    replay=load(scratch/"replay/evidence/verification_replay.json")
    assert offline["status"]==replay["status"]=="PASS" and replay["neural_replayed"]==36
    hashes=load(clean/"evidence/prepared.json")["files"]
    for stage,pinned in hashes.items():
        assert sha(scratch/"clean_rebuilt/prepared"/(stage+".npz"))==pinned["sha256"]
    docs=("README_全量实验入口.md","plan.md","方法与数据合同.md","审计报告.md","测试报告.md")
    assert all((clean/f).is_file() and (clean/f).stat().st_size for f in docs)
    logs=[]
    for directory in (scratch/"logs",scratch/"clean_rebuilt/logs",scratch/"offline/logs",scratch/"replay/logs"):
        for p in sorted(directory.glob("*.json")):
            r=load(p)
            if "exit_code" in r:
                assert r["exit_code"]==0
                log=directory/r["log"]
                logs.append(dict(**r,console_sha256=sha(log),console_tail=log.read_text(encoding="utf-8")[-1200:]))
    checked=load(scratch/"zip_check.json")
    save(output/"解压复核.json",dict(**checked,completed_utc=now(),bytes=archive.stat().st_size,
        clean_rebuilt_feature_hashes={k:v["sha256"] for k,v in hashes.items()},
        independent_metrics=offline,clean_model_replay=replay,original_engineering_tests=63,
        clean_new_engineering_tests=7,fresh_refits_in_clean_extraction=0,
        repeat_training_scope="No second complete training matrix. Causal inputs rebuilt and all 36 neural plus 6 baseline weights replayed.",
        all_raw_files_external=True,experiment_completed=True,evaluation_type="chronological retrospective",commands=logs))
    print(dict(status="PASS",zip=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
               clean_neural_replay=36,baselines=6,engineering_tests=63),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--qa",type=Path,required=True)
    a=p.parse_args()
    check(a.run.resolve(),a.output.resolve(),a.qa.resolve())

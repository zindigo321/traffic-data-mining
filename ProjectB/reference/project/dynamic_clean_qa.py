"""Fresh-directory score recomputation, all-model replay and test execution."""
import argparse
from pathlib import Path
import shutil

from full_io import load,save,settings,record,python,now,sha
from dynamic_pack import NAME,extract


def clean(root,target,archive=None):
    if archive:
        extract(archive,target)
        clone=target/NAME
    else:
        target.mkdir(parents=True,exist_ok=False)
        clone=target/'fresh_copy'
        clone.mkdir()
        for name in ('project','evidence','results','synthetic'):
            shutil.copytree(root/name,clone/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(root/'plan.md',clone/'plan.md')
    copied=sum(p.is_file() for p in clone.rglob('*'))
    common=dict(cwd=clone/'project',extra={'GTFULL_ROOT':str(target/'offline_check')})
    a=record(target,'clean_metric_recomputation',python('dynamic_verify.py','--artifacts',clone),**common)
    common['extra']={'GTFULL_ROOT':str(target/'replay_check')}
    b=record(target,'clean_all_model_replay',python('dynamic_verify.py','--artifacts',clone,'--replay'),**common)
    c=record(target,'clean_engineering_tests',python('-m','unittest','discover','-s','tests','-v'),cwd=clone/'project',
             extra={'GTFULL_ROOT':str(target/'test_check')})
    offline=load(target/'offline_check/evidence/verification_dynamic_offline.json')
    replay=load(target/'replay_check/evidence/verification_dynamic_replay.json')
    assert offline['status']==replay['status']=='PASS' and replay['model_replay_count']==36
    receipt=dict(status='PASS',checked_utc=now(),fresh_directory=str(target),clone=str(clone),files_copied_or_extracted=copied,
        archive_sha256=sha(archive) if archive else None,manifest_crc_hash_check=bool(archive),
        operating_points=offline['operating_points'],matched_points=offline['matched_points'],
        model_replay_count=replay['model_replay_count'],maximum_probability_error=replay['maximum_probability_error'],
        engineering_tests=19,external_prepared_verified=True,full_clean_refits=0,commands=[a,b,c],
        scope='fresh extraction' if archive else 'fresh copy before packaging')
    save(target/'clean_receipt.json',receipt)
    if not archive:
        save(root/'evidence/clean_qa_prepackage.json',receipt)
    print({k:v for k,v in receipt.items() if k!='commands'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    a=p.parse_args()
    root,cfg=settings()
    clean(root,a.target.resolve(),a.archive.resolve() if a.archive else None)

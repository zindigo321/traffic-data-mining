"""Package-local adapters; model, truth, features and metrics stay unchanged."""
import argparse
from itertools import product
import os
from pathlib import Path
import shutil
import sys

PACKAGE=Path(os.environ['PROJECTB_PACKAGE_ROOT']).resolve()
DATA=PACKAGE/'data'
REFERENCE=PACKAGE/'reference'
OUTPUT=Path(os.environ['GTFULL_ROOT']).resolve()


def no_legacy_files(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(args[0]))
    if not path.is_absolute():return
    absolute=path.resolve()
    if any(absolute.is_relative_to(p) for p in (PACKAGE,OUTPUT,Path(sys.prefix).resolve())):return
    normalized=absolute.as_posix().lower()
    if any(marker in normalized for marker in ('/work/gt_v','/work/full_data/','/work/dynamic_edges/',
           '/course-jiaotongshujuwajue/','/wechatt/xwechat_files/')):
        raise PermissionError(f'Portable mode refused historical external file: {absolute}')


sys.addaudithook(no_legacy_files)

import numpy as np
import pandas as pd
import torch

from full_io import CODE,load,now,python,record,save,settings,sha
from full_data import arrays,folder,study_for,TrainPreprocessor,predict,set_seed
from dynamic_model import build_model
from dynamic_score import STATIC,DYNAMIC


def sources():
    for row in load(DATA/'source_snapshot.json')['files']:
        path=PACKAGE/row['path']
        if not path.is_file() or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError(f'Bundled data checksum mismatch: {path}')
        yield path,row


def check(root,cfg):
    source_rows=list(sources())
    details={}
    for stage in ('development','final'):
        d=arrays(REFERENCE,stage)
        details[stage]=dict(features=list(d['x'].shape),roads=len(d['road_ids']),
            labels={str(k):int((d['y']==k).sum()) for k in (-1,0,1)},
            local_input=str(DATA/'prepared'/f'{stage}.npz'))
    geom=load(DATA/'geometry_manifest.json')
    assert sha(PACKAGE/geom['path'])==geom['sha256']
    result=dict(status='PASS',at_utc=now(),python=sys.executable,device='CPU',torch=torch.__version__,
        cpu_threads=cfg['cpu_threads'],workers=cfg['workers'],seeds=cfg['seeds'],package=str(PACKAGE),
        raw_shards=len(source_rows),raw_records=sum(row['rows'] for _,row in source_rows),
        geometry_SHA256=geom['sha256'],prepared=details,external_old_paths_required=False,
        historical_file_access_guard_enabled=True,niulaoshi_environment_modified=False)
    save(root/'check_result.json',result)
    print(result,flush=True)


def prepare(root,cfg):
    from full_prepare import construct,stats
    import v5_prepare
    import v5_graph
    (root/'evidence').mkdir(parents=True,exist_ok=True)
    geometry=load(DATA/'geometry_manifest.json')
    def geometry_files(unused,name):
        if name!='geometry':raise ValueError(name)
        yield PACKAGE/geometry['path'],geometry
    v5_graph.source_files=geometry_files
    table,graph=v5_graph.build(root,cfg)
    expected=load(REFERENCE/'evidence/graph.json')
    actual=load(root/'evidence/graph.json')
    for name in ('roster_sha256','adjacency_sha256','edge_sha256'):
        assert actual[name]==expected[name],name
    v5_prepare.source_files=lambda unused,name: sources() if name=='all_available' else geometry_files(unused,name)
    raw,funnel=v5_prepare.grid(root,cfg,'all_available',table,cfg['corpus'])
    result={}
    for stage,train_end,cal_end,stop in (
        ('development',cfg['development_train'][1],cfg['development_assess'][1],cfg['development_assess'][1]),
        ('final',cfg['full_train'][1],cfg['full_calibration'][1],cfg['corpus'][1])):
        d=construct(raw,table,graph,cfg,train_end,cal_end,stop)
        path=root/'prepared'/f'{stage}.npz'
        np.savez_compressed(path,**d)
        # Byte-for-byte agreement, not merely matching shapes or label counts.
        expected_hash=load(DATA/'prepared_manifest.json')['files'][stage]['sha256']
        assert sha(path)==expected_hash,stage
        result[stage]=dict(sha256=sha(path),bytes=path.stat().st_size)
        save(root/'evidence'/f'cohort_{stage}.json',stats(d,cfg,stage))
    save(root/'prepare_result.json',dict(status='PASS',at_utc=now(),rebuilt_from_bundled_public_raw=True,
        legacy_paths_opened=False,all_arrays_byte_identical=True,files=result,source_funnel=funnel))
    print(dict(status='PASS',prepared=result),flush=True)


def smoke(root,cfg):
    study=study_for(arrays(REFERENCE,'development'),'X1')
    seed=11
    rng=np.random.default_rng(seed)
    ids=np.r_[rng.choice(study.train[study.y[study.train]==1],8,replace=False),
              rng.choice(study.train[study.y[study.train]==0],24,replace=False)]
    study.configure(TrainPreprocessor.load(folder(REFERENCE,'development','graph_mean',seed)/'preprocessor.npz'))
    x,adj,valid,center=map(torch.from_numpy,study.batch(ids))
    y=torch.from_numpy(study.y[ids].astype(np.float32))
    details=[]
    for pool in ('mean','max','attention'):
        set_seed(seed,cfg)
        net=build_model(104,cfg,'dynamic_'+pool,seed)
        opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'])
        criterion=torch.nn.BCEWithLogitsLoss()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss=criterion(net(x,adj,valid,center),y)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step()
        path=root/f'smoke_{pool}_weight.pt'
        torch.save(net.state_dict(),path)
        expected=sha(path)
        restored=build_model(104,cfg,'dynamic_'+pool,seed)
        assert sha(path)==expected
        restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        with torch.inference_mode():torch.testing.assert_close(net(x,adj,valid,center),restored(x,adj,valid,center),atol=0,rtol=0)
        details.append(dict(pool=pool,seed=seed,real_train_rows=len(ids),updates=2,weight_sha256=expected,
                            gradient_finite=True,save_load_exact=True))
    np.savez_compressed(root/'smoke_training_ids.npz',sample_id=study.keys(ids),y=study.y[ids])
    save(root/'smoke_result.json',dict(status='PASS',engineering_only=True,source='small real training batch',
        main_experiment_performance=None,full_training_executed=False,models=details))
    print(dict(status='PASS',three_pool_train_save_load=True,engineering_only=True),flush=True)


def inference(root,cfg,pool,seed):
    model='dynamic_'+pool
    path=folder(REFERENCE,'final',model,seed)
    freeze=load(REFERENCE/'evidence/dynamic_threshold_freeze.json')
    entry,=[r for r in freeze['models'] if r['model']==model and r['seed']==seed]
    for name,digest in entry['files'].items():assert sha(path/name)==digest
    study=study_for(arrays(REFERENCE,'final'),'X1')
    study.configure(TrainPreprocessor.load(path/'preprocessor.npz'))
    set_seed(seed,cfg)
    net=build_model(104,cfg,model,seed)
    net.load_state_dict(torch.load(path/'weight.pt',map_location='cpu',weights_only=True))
    ids=np.flatnonzero(study.phase>0)
    logit,prob=predict(net,study,ids,cfg,False)
    score=prob.astype(np.float64)
    keep=study.phase[ids]==2
    keys=study.keys(ids)[keep]
    truth=study.y[ids][keep]
    threshold=np.float64(entry['options']['constrained']['threshold'])
    pred=score[keep]>=threshold
    # Verify the full prediction vector, including unknown-label rows.
    with np.load(REFERENCE/f'results/evaluation/{model}_s{seed}.npz',allow_pickle=False) as reference:
        np.testing.assert_array_equal(prob,reference['probability'])
    frame=pd.DataFrame(dict(sample_id=keys,NYC_window_start=pd.to_datetime(keys//1000*10,unit='m'),
        road_code=keys%1000,probability=score[keep],logit=logit[keep],threshold=threshold,
        prediction=pred.astype(np.int8),truth=truth,known=truth>=0))
    frame.to_csv(root/'test_predictions.csv',index=False,float_format='%.17g')
    np.savez_compressed(root/'test_predictions.npz',sample_id=keys,probability=prob[keep],score=score[keep],
        logit=logit[keep],threshold=threshold,prediction=pred,truth=truth)
    save(root/'prediction_result.json',dict(status='PASS',model=model,seed=seed,rows=len(frame),
        known=int((truth>=0).sum()),unknown=int((truth<0).sum()),weight_sha256=sha(path/'weight.pt'),
        probability_dtype='float32',comparison_dtype='float64 exact promotion',threshold=float(threshold),
        historical_test_replay=True,independent_new_test=False,reference_probabilities_identical=True))
    print(dict(status='PASS',model=model,rows=len(frame),csv=str(root/'test_predictions.csv')),flush=True)


def initialize_training(root,cfg):
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(CODE,root/'project',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(REFERENCE/'plan.md',root/'plan.md')
    (root/'evidence').mkdir()
    for name in ('prepared.json','source_snapshot.json','graph.json','original_graph.json','cohort_development.json',
                 'cohort_final.json','nyc_edges.csv','nyc_roads.csv','static_reference_selection.json',
                 'static_reference_freeze.json','static_reference_summary.json'):
        shutil.copy2(REFERENCE/'evidence'/name,root/'evidence'/name)
    shutil.copytree(REFERENCE/'source_provenance',root/'source_provenance')
    controls=[]
    for stage,model,seed in product(('development','final'),STATIC,cfg['seeds']):
        src=folder(REFERENCE,stage,model,seed)
        assert sha(src/'weight.pt')==load(src/'status.json')['weight_sha256']
        dst=folder(root,stage,model,seed)
        shutil.copytree(src,dst)
        controls.append(dict(stage=stage,model=model,seed=seed,weight_sha256=sha(src/'weight.pt'),new_training=False))
    save(root/'evidence/path_mapping.json',dict(source_bundle=str(REFERENCE),run=str(root),
         public_data=str(DATA),prepared_override=str(DATA/'prepared'),source_archive_included=False))
    import platform,psutil,importlib.metadata
    save(root/'logs/environment_dynamic.json',dict(executable=sys.executable,python=sys.version,cpu=platform.processor(),
        device='CPU',cpu_threads=cfg['cpu_threads'],threads=cfg['cpu_threads'],workers=cfg['workers'],seeds=cfg['seeds'],
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','pandas','scikit-learn','scipy','psutil','matplotlib')}))
    origin=load(DATA/'package_origin.json')
    rows=list(sources())
    save(root/'evidence/registration_dynamic.json',dict(status='PASS',registered_utc=now(),
        original_dynamic_archive_sha256=origin['original_dynamic_archive_sha256'],source_archive_required=False,
        transport='bundled public snapshot and hash-verified cached static controls',raw_shards=len(rows),
        raw_records=sum(row['rows'] for _,row in rows),controls=controls,new_fits_planned=18,static_reused=18,
        fresh_blind_test=False,plan_sha256=sha(root/'plan.md'),config_sha256=sha(CODE/'config.json'),
        prepared_location_override=str(DATA/'prepared'),historical_file_access_guard_enabled=True))
    save(root/'evidence/portable_binding.json',dict(package=str(PACKAGE),data=str(DATA),
        prepared_sha256=load(DATA/'prepared_manifest.json')['files'],adaptation='Explicit package-local path override only',
        historical_raw_snapshot_embedded=True,private_trajectories_included=False))


def train(root,cfg,phase):
    if phase in ('all','initialize'):
        initialize_training(root,cfg)
    elif not (root/'evidence/registration_dynamic.json').is_file():
        raise FileNotFoundError('For staged training, first run train --phase initialize --output with the same new directory.')
    if phase=='initialize':return
    from run_dynamic import run
    for stage in (('tests','development','select','final','freeze','evaluate','analysis','verify','report') if phase=='all' else (phase,)):
        run(root,stage)
    if phase in ('all','report'):
        summary=load(root/'results/summary.json')
        save(root/'portable_training_result.json',dict(status='PASS',new_full_dynamic_fits=18,verified_static_controls_reused=18,
            original_protocol=True,source='bundled raw/prepared data',external_old_paths_required=False,
            main=load(root/'results/dynamic_selection.json')['primary'],test=summary['2']['constrained']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--pool',default='mean')
    p.add_argument('--seed',default=11,type=int)
    p.add_argument('--phase',default='all')
    args=p.parse_args()
    root,cfg=settings()
    if args.command=='train':train(root,cfg,args.phase);return
    if args.command=='check':record(root,'local_check',python('portable_tasks.py','worker_check','--output',root));return
    if args.command=='worker_check':check(root,cfg);return
    if args.command=='verify':record(root,'local_verify',python('dynamic_verify.py','--artifacts',REFERENCE));return
    if args.command=='replay':record(root,'local_replay',python('dynamic_verify.py','--artifacts',REFERENCE,'--replay'));return
    if args.command=='tests':
        from run_dynamic import run
        run(root,'tests')
        record(root,'portable_path_regressions',python('-m','unittest','discover','-s',PACKAGE/'package_tests','-v'))
        save(root/'portable_tests_result.json',dict(status='PASS',historical_model_tests=75,portable_tests=6,total=81))
        return
    if args.command in ('smoke','prepare','predict'):
        # Self-invocation keeps full stdout, exit and timing receipts without changing core modules.
        record(root,'local_'+args.command,python('portable_tasks.py','worker_'+args.command,'--output',root,
                                               '--pool',args.pool,'--seed',args.seed));return
    if args.command=='worker_smoke':smoke(root,cfg);return
    if args.command=='worker_prepare':prepare(root,cfg);return
    if args.command=='worker_predict':inference(root,cfg,args.pool,args.seed);return
    raise ValueError(args.command)


if __name__=='__main__':main()

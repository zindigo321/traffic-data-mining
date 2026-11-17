"""Verify real source artifacts, immutable cached arrays and reused static fits."""
import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

from full_io import CODE,V10,load,now,save,settings,sha
from full_data import arrays,read_npz
sys.path.insert(0,str(V10))
from v10_check import directory

EXPECTED='0436558b017bb6c5d52d0ab2103ae52f2315b970e4a9b9cf615c7b4a9f0c1c47'
STATIC=('graph_mean','graph_max','graph_attention')


def check(root,cfg):
    target=root/'evidence/registration_dynamic.json'
    if target.exists():
        raise FileExistsError('Do not overwrite registration')
    mapping=load(root/'evidence/path_mapping.json')
    source=Path(mapping['source_bundle'])
    archive=Path(mapping['source_archive'])
    assert sha(archive)==EXPECTED
    payload=directory(source)
    raw=[]
    for row in load(root/'evidence/source_snapshot.json')['files']:
        p=Path(row['path'])
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        raw.append(dict(path=str(p),sha256=row['sha256'],rows=row['rows']))
    data_receipts={}
    for stage in ('development','final'):
        data=arrays(root,stage)
        data_receipts[stage]=dict(shape=list(data['x'].shape),known=int((data['y']>=0).sum()),
                                  roads=len(data['road_ids']))
    controls=[]
    for stage in ('development','final'):
        for model in STATIC:
            for seed in cfg['seeds']:
                rel=Path('results/models')/stage/f'{model}_s{seed}'
                src=source/rel
                weight=src/'weight.pt'
                assert sha(weight)==load(src/'status.json')['weight_sha256']
                shutil.copytree(src,root/rel)
                controls.append(dict(stage=stage,model=model,seed=seed,weight_sha256=sha(weight),
                                     source=str(src),new_training=False))
    save(root/'evidence/static_reference_selection.json',load(source/'results/development_selection.json'))
    save(root/'evidence/static_reference_freeze.json',load(source/'evidence/model_threshold_freeze.json'))
    save(root/'evidence/static_reference_summary.json',load(source/'results/summary.json'))
    save(root/'logs/environment_dynamic.json',dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
        cpu=platform.processor(),physical_cores=psutil.cpu_count(logical=False),logical_cores=psutil.cpu_count(),
        RAM_bytes=psutil.virtual_memory().total,device='CPU',threads=cfg['cpu_threads'],workers=cfg['workers'],
        seeds=cfg['seeds'],packages={name:importlib.metadata.version(name) for name in
            ('torch','numpy','pandas','scikit-learn','scipy','psutil','matplotlib')}))
    receipt=dict(status='PASS',registered_utc=now(),source_archive_sha256=EXPECTED,source_payload=payload,
        raw_shards=len(raw),raw_records=sum(r['rows'] for r in raw),raw_files=raw,arrays=data_receipts,
        controls=controls,new_fits_planned=18,static_reused=18,protocol='chronological retrospective',
        fresh_blind_test=False,plan_sha256=sha(root/'plan.md'),config_sha256=sha(CODE/'config.json'),
        implementation={p.name:sha(p) for p in CODE.glob('dynamic_*.py')})
    save(target,receipt)
    print(dict(status='PASS',raw_shards=len(raw),controls=len(controls),arrays=data_receipts),flush=True)


if __name__=='__main__':
    check(*settings())

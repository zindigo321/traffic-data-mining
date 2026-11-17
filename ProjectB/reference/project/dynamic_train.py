"""New dynamic fits only: matched all-known training and unchanged budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import math
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from full_io import load,now,python,record,save,settings,sha
from full_data import arrays,epoch_order,folder,groups,study_for,predict,set_seed
from dynamic_model import build_model


def fit(root,cfg,stage,model,seed):
    if (root/'evidence/dynamic_threshold_freeze.json').exists():
        raise PermissionError('Models and thresholds are frozen')
    lock=load(root/'results/dynamic_selection.json')['epochs'][model] if stage=='final' else None
    path=folder(root,stage,model,seed)
    path.mkdir(parents=True,exist_ok=False)
    study=study_for(arrays(root,stage),'X1')
    ids=study.train.copy()
    count=len(ids)
    y_train=study.y[ids]
    positive=int((y_train==1).sum())
    negative=int((y_train==0).sum())
    assert positive and negative
    group=groups(study.ratio,cfg['ratio'])
    np.savez_compressed(path/'training_ids.npz',sample_id=study.keys(ids),y=y_train,legal_group=group[ids])
    save(path/'training_support.json',dict(full_known=count,unique_training=count,positive=positive,negative=negative,
        uses_all_known=True,epoch_exposures=count,epochs=cfg['epochs'],updates_per_epoch=math.ceil(count/cfg['batch_size'])))
    preprocessor=study.fit_preprocessor()
    preprocessor.save(path/'preprocessor.npz')
    # Exact paired fit, not a second normalization fitted on extra data.
    paired=folder(root,stage,'graph_'+model.split('_')[-1],seed)
    with np.load(paired/'preprocessor.npz',allow_pickle=False) as pp:
        for key in pp.files:
            np.testing.assert_array_equal(pp[key],getattr(preprocessor,key))
    with np.load(paired/'training_ids.npz',allow_pickle=False) as training:
        np.testing.assert_array_equal(study.keys(ids),training['sample_id'])
    study.configure(preprocessor)
    set_seed(seed,cfg)
    net=build_model(study.nfeatures*2,cfg,model,seed)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['lr'])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,dtype=torch.float32))
    y=torch.from_numpy(study.y.astype(np.float32))
    rng=np.random.default_rng(seed)
    started,tick,process=now(),time.perf_counter(),psutil.Process()
    peak=process.memory_info().rss
    best,best_epoch,state=-1.,0,None
    curve=[]
    for e in range(1,cfg['epochs']+1):
        net.train()
        order=epoch_order(ids,count,rng)
        total=0.
        for offset in range(0,len(order),cfg['batch_size']):
            j=order[offset:offset+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            z=net(*map(torch.from_numpy,study.batch(j)))
            loss=criterion(z,y[j])
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(j)
        ap=None
        if lock is None:
            _,prob=predict(net,study,study.val,cfg,False)
            ap=float(average_precision_score(study.y[study.val],prob))
        if e==lock or (ap is not None and ap>best):
            best=ap if ap is not None else best
            best_epoch=e
            state={k:v.detach().clone() for k,v in net.state_dict().items()}
        peak=max(peak,process.memory_info().rss)
        curve.append(dict(epoch=e,bce=total/count,AP=ap,training_exposures=count,unique_seen=len(np.unique(order)),
                          updates=math.ceil(count/cfg['batch_size'])))
        print(dict(stage=stage,model=model,seed=seed,epoch=e,AP=ap,bce=total/count,
                   elapsed=round(time.perf_counter()-tick,1)),flush=True)
    if state is None:
        raise RuntimeError('No checkpoint')
    net.load_state_dict(state)
    torch.save(net.state_dict(),path/'weight.pt')
    pd.DataFrame(curve).to_csv(path/'curve.csv',index=False,na_rep='N/A')
    # Before freezing final thresholds, only calibration is scored. No test labels/scores here.
    assess=np.flatnonzero(study.phase==1)
    logits,prob=predict(net,study,assess,cfg,False)
    np.savez_compressed(path/'assessment.npz',sample_id=study.keys(assess),y=study.y[assess],risk=study.risk[assess],
        legal_group=group[assess],phase=study.phase[assess],score=prob.astype(np.float64),logit=logits,probability=prob)
    result=dict(status='complete',stage=stage,model=model,seed=seed,started_utc=started,completed_utc=now(),
        fit_and_score_seconds=time.perf_counter()-tick,peak_RSS_bytes=peak,parameters=sum(p.numel() for p in net.parameters()),
        best_epoch=best_epoch,fixed_epoch=lock,total_training_exposures=count*cfg['epochs'],
        updates=sum(r['updates'] for r in curve),known_train=count,unique_training=count,full_known_coverage=1.,
        weight_sha256=sha(path/'weight.pt'),assessment_rows=len(assess),test_scored_before_freeze=False)
    save(path/'status.json',result)
    print(result,flush=True)


def execute(root,cfg,stage):
    def job(spec):
        model,seed=spec
        dest=folder(root,stage,model,seed)
        if dest.exists():
            if not (dest/'status.json').exists():
                raise RuntimeError(f'Partial run requires review: {dest}')
        else:
            record(root,f'dynamic_fit_{stage}_{model}_s{seed}',python('dynamic_train.py','fit',
                '--stage',stage,'--model',model,'--seed',seed))
        return load(dest/'status.json')
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        rows=list(pool.map(job,product(cfg['dynamic_models'],cfg['seeds'])))
    save(root/f'results/dynamic_{stage}_training.json',dict(completed_utc=now(),new_fits=len(rows),models=rows))
    print(dict(stage=stage,new_fits=len(rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('fit','matrix'))
    p.add_argument('--stage',required=True,choices=('development','final'))
    p.add_argument('--model',choices=('dynamic_mean','dynamic_max','dynamic_attention'))
    p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.action=='fit': fit(*settings(),a.stage,a.model,a.seed)
    else: execute(*settings(),a.stage)

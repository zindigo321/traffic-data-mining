"""Development selection, calibration freeze, then one retrospective evaluation."""
import argparse
from itertools import product
import statistics
import numpy as np
import torch

from full_io import CODE,load,now,save,settings,sha
from full_data import arrays,folder,read_npz,study_for,TrainPreprocessor,predict,set_seed,groups
from full_score import macro,phase_metrics,choose_thresholds,shared,write_csv,GROUPS
from v4_metrics import matched_points
from dynamic_model import build_model

STATIC=('graph_mean','graph_max','graph_attention')
DYNAMIC=('dynamic_mean','dynamic_max','dynamic_attention')
ALL=(*STATIC,*DYNAMIC)


def jobs(cfg):
    return product(ALL,cfg['seeds'])


def read_assessment(root,stage,model,seed,evaluated=False):
    if evaluated and stage=='final' and model in DYNAMIC:
        return read_npz(root/f'results/evaluation/{model}_s{seed}.npz')
    return read_npz(folder(root,stage,model,seed)/'assessment.npz')


def selection(root,cfg):
    dest=root/'results/dynamic_selection.json'
    if dest.exists():raise FileExistsError('Selection already frozen')
    rows=[]
    ref=None
    for model,seed in jobs(cfg):
        raw=read_assessment(root,'development',model,seed)
        ref=raw if ref is None else ref
        shared(raw,ref)
        k=raw['y']>=0
        days=len(np.unique(raw['sample_id']//1000//144))
        cut=choose_thresholds(raw['y'][k],raw['score'][k],raw['risk'][k],days,cfg)
        for policy,option in cut.items():
            rows.append(phase_metrics(raw,1,option['threshold'],cfg,model,seed,policy))
    summary={m:macro([r for r in rows if r['model']==m and r['policy']=='constrained']) for m in ALL}
    epochs={m:int(statistics.median(load(folder(root,'development',m,s)/'status.json')['best_epoch']
                for s in cfg['seeds'])) for m in DYNAMIC}
    ranked=sorted(DYNAMIC,key=lambda m:(-summary[m]['AP'],-summary[m]['f1'],DYNAMIC.index(m)))
    write_csv(root/'results/development_metrics.csv',rows)
    save(dest,dict(primary=ranked[0],static_primary='graph_max',epochs=epochs,summary=summary,
        selection_rule='development three-seed mean AP then constrained F1 then Mean/Max/Attention order',
        frozen_utc=now(),candidates=1,retrospective_already_public=True,test_used_for_current_selection=False,
        config_sha256=sha(CODE/'config.json'),metrics_sha256=sha(root/'results/development_metrics.csv')))
    print(dict(primary=ranked[0],epochs=epochs,AP={m:summary[m]['AP'] for m in ALL}),flush=True)


def freeze(root,cfg):
    dest=root/'evidence/dynamic_threshold_freeze.json'
    if dest.exists():raise FileExistsError('Thresholds already frozen')
    selected=load(root/'results/dynamic_selection.json')
    assert sha(CODE/'config.json')==selected['config_sha256']
    entries,rows=[],[]
    for model,seed in jobs(cfg):
        path=folder(root,'final',model,seed)
        raw=read_assessment(root,'final',model,seed)
        k=(raw['phase']==1)&(raw['y']>=0)
        options=choose_thresholds(raw['y'][k],raw['score'][k],raw['risk'][k],28,cfg)
        for policy,cut in options.items():
            rows.append(phase_metrics(raw,1,cut['threshold'],cfg,model,seed,policy))
        entries.append(dict(model=model,seed=seed,options=options,
            files={p.name:sha(p) for p in sorted(path.iterdir()) if p.is_file()}))
    write_csv(root/'results/calibration_metrics.csv',rows)
    receipt=dict(frozen_utc=now(),primary=selected['primary'],static_primary='graph_max',policy='constrained',
        selection_sha256=sha(root/'results/dynamic_selection.json'),config_sha256=sha(CODE/'config.json'),
        prepared_sha256=sha(root/'evidence/prepared.json'),source_snapshot_sha256=sha(root/'evidence/source_snapshot.json'),
        code={p.name:sha(p) for p in CODE.glob('dynamic_*.py')},models=entries,
        current_round_test_metrics_computed=False,new_dynamic_test_scores_computed=False,
        historical_test_already_public=True,fresh_blind_validation=False,
        test_sample_summary=load(root/'evidence/cohort_final.json')['parts'][2])
    save(dest,receipt)
    save(root/'evidence/dynamic_threshold_freeze_sha256.json',dict(sha256=sha(dest),frozen_utc=receipt['frozen_utc']))
    print(dict(frozen_models=len(entries),primary=selected['primary']),flush=True)


def evaluate(root,cfg):
    dest=root/'results/summary.json'
    if dest.exists():raise FileExistsError('Do not overwrite retrospective results')
    freeze=load(root/'evidence/dynamic_threshold_freeze.json')
    assert sha(root/'evidence/dynamic_threshold_freeze.json')==load(root/'evidence/dynamic_threshold_freeze_sha256.json')['sha256']
    assert sha(CODE/'config.json')==freeze['config_sha256']
    data=arrays(root,'final')
    study=study_for(data,'X1')
    assess=np.flatnonzero(study.phase>0)
    group=groups(study.ratio,cfg['ratio'])
    rows,group_rows,points=[],[],[]
    reference=None
    evaluation_started=now()
    for entry in freeze['models']:
        model,seed=entry['model'],entry['seed']
        path=folder(root,'final',model,seed)
        for name,digest in entry['files'].items():assert sha(path/name)==digest
        if model in DYNAMIC:
            set_seed(seed,cfg)
            study.configure(TrainPreprocessor.load(path/'preprocessor.npz'))
            net=build_model(104,cfg,model,seed)
            net.load_state_dict(torch.load(path/'weight.pt',map_location='cpu',weights_only=True))
            logits,prob=predict(net,study,assess,cfg,False)
            raw=dict(sample_id=study.keys(assess),y=study.y[assess],risk=study.risk[assess],
                legal_group=group[assess],phase=study.phase[assess],score=prob.astype(np.float64),logit=logits,probability=prob)
            out=root/f'results/evaluation/{model}_s{seed}.npz'
            out.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(out,**raw)
        else:raw=read_assessment(root,'final',model,seed)
        reference=raw if reference is None else reference
        shared(raw,reference)
        decisions=dict(sample_id=raw['sample_id'])
        for policy,option in entry['options'].items():
            cut=option['threshold']
            decisions['pred_'+policy]=raw['score']>=cut
            decisions['threshold_'+policy]=np.float64(cut)
            for phase in (1,2):
                rows.append(phase_metrics(raw,phase,cut,cfg,model,seed,policy))
                for number,name in enumerate(GROUPS):
                    k=(raw['phase']==phase)&(raw['legal_group']==number)&(raw['y']>=0)
                    from full_score import measure
                    group_rows.append(dict(model=model,seed=seed,phase=phase,policy=policy,scope=name,
                        **measure(raw['y'][k],raw['score'][k],raw['risk'][k],cut,28 if phase==1 else 45)))
        for phase in (1,2):
            for scope,condition in (('all',np.ones(len(raw['y']),bool)),('legal_low',raw['legal_group']==1)):
                k=(raw['phase']==phase)&condition&(raw['y']>=0)
                matched,_=matched_points(raw['sample_id'][k],raw['y'][k],raw['score'][k],raw['risk'][k],
                    cfg['recalls'],28 if phase==1 else 45)
                points.extend(dict(model=model,seed=seed,phase=phase,scope=scope,**p,
                    recovery_FP_per_day=p['recovery_FP']/(28 if phase==1 else 45)) for p in matched)
        out=root/f'results/decisions/{model}_s{seed}.npz'
        out.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(out,**decisions)
    write_csv(root/'results/operating_points.csv',rows)
    write_csv(root/'results/groups.csv',group_rows)
    write_csv(root/'results/matched_recall.csv',points)
    summary={str(phase):{policy:{model:macro([r for r in rows if r['model']==model and r['phase']==phase and r['policy']==policy])
             for model in ALL} for policy in ('constrained','max_f1')} for phase in (1,2)}
    save(dest,summary)
    save(root/'evidence/dynamic_experiment_status.json',dict(status='completed',evaluation_started_utc=evaluation_started,
        completed_utc=now(),new_dynamic_fits=18,reused_static_fits=18,test_known=rows[1]['known'] if rows[1]['phase']==2 else 80536,
        evaluation_type='chronological retrospective full available data',new_independent_validation=False,
        primary=freeze['primary'],new_test_data_downloaded=False,Chicago='blocked',Geolife='blocked'))
    print(dict(status='completed',operating_points=len(rows),matched_points=len(points)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('selection','freeze','evaluate'))
    a=p.parse_args()
    {'selection':selection,'freeze':freeze,'evaluate':evaluate}[a.stage](*settings())

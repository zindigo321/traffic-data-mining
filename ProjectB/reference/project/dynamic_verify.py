"""Independent metric/threshold recomputation and weights-only causal replay."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score

from full_io import CODE,load,now,save,settings,sha
from full_data import arrays,folder,read_npz,study_for,TrainPreprocessor,predict,set_seed,epoch_order
from full_score import shared
from dynamic_score import ALL,DYNAMIC,STATIC,jobs,read_assessment
from dynamic_model import build_model
from v9_verify import independent_cut
from v7_verify import verify_point,check_support


def close(actual,expected):
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12,equal_nan=True)


def recompute(raw,k,threshold,days,row):
    y,p,risk=[raw[name][k] for name in ('y','score','risk')]
    pred=p>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    both=bool((y==1).any() and (y==0).any())
    values=dict(tn=tn,fp=fp,fn=fn,tp=tp,known=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),
        AP=float(average_precision_score(y,p)) if both else np.nan,
        ROC_AUC=float(roc_auc_score(y,p)) if both else np.nan,
        balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else np.nan,
        accuracy=float((pred==y).mean()) if len(y) else np.nan,
        f1=float(f1_score(y,pred,zero_division=0)) if (y==1).any() else np.nan,
        recall=tp/(tp+fn) if tp+fn else np.nan,FP_day=fp/days,
        recovery_FP=int((pred&(y==0)&risk).sum()))
    values['recovery_FP_day']=values['recovery_FP']/days
    for name,value in values.items():close(row[name],value)


def replay_one(root,cfg,stage,model,seed,study,raw):
    path=folder(root,stage,model,seed)
    set_seed(seed,cfg)
    study.configure(TrainPreprocessor.load(path/'preprocessor.npz'))
    net=build_model(104,cfg,model,seed)
    net.load_state_dict(torch.load(path/'weight.pt',map_location='cpu',weights_only=True))
    ids=np.flatnonzero(study.phase>0) if stage=='final' else np.flatnonzero(study.phase==1)
    np.testing.assert_array_equal(study.keys(ids),raw['sample_id'])
    np.testing.assert_array_equal(study.y[ids],raw['y'])
    np.testing.assert_array_equal(study.risk[ids],raw['risk'])
    errors=[]
    for phase in np.unique(raw['phase']):
        spots=np.flatnonzero((raw['phase']==phase)&(raw['y']==1))
        at=int(spots[len(spots)//2]) if len(spots) else int(np.flatnonzero(raw['phase']==phase)[0])
        start=at//cfg['eval_batch_size']*cfg['eval_batch_size']
        sl=slice(start,min(start+cfg['eval_batch_size'],len(ids)))
        logits,prob=predict(net,study,ids[sl],cfg,False)
        np.testing.assert_allclose(logits,raw['logit'][sl],atol=1e-7,rtol=0)
        np.testing.assert_allclose(prob,raw['probability'][sl],atol=1e-7,rtol=0)
        errors.append(float(np.max(np.abs(prob.astype(float)-raw['score'][sl]))))
    return dict(stage=stage,model=model,seed=seed,batches=len(errors),max_probability_error=max(errors))


def verify(root,out,cfg,replay=False):
    freeze=load(root/'evidence/dynamic_threshold_freeze.json')
    assert sha(root/'evidence/dynamic_threshold_freeze.json')==load(root/'evidence/dynamic_threshold_freeze_sha256.json')['sha256']
    assert sha(root/'results/dynamic_selection.json')==freeze['selection_sha256']
    assert sha(CODE/'config.json')==freeze['config_sha256']
    for name,digest in freeze['code'].items():assert sha(CODE/name)==digest
    selection=load(root/'results/dynamic_selection.json')
    candidates=sorted(DYNAMIC,key=lambda m:(-selection['summary'][m]['AP'],-selection['summary'][m]['f1'],DYNAMIC.index(m)))
    assert selection['primary']==candidates[0] and not selection['test_used_for_current_selection']
    state=load(root/'evidence/dynamic_experiment_status.json')
    assert not state['new_independent_validation']
    stamp=lambda v:datetime.fromisoformat(v)
    assert stamp(selection['frozen_utc'])<stamp(freeze['frozen_utc'])<stamp(state['evaluation_started_utc'])
    for model in DYNAMIC:
        expected=int(np.median([load(folder(root,'development',model,s)/'status.json')['best_epoch'] for s in cfg['seeds']]))
        assert expected==selection['epochs'][model]
        for seed in cfg['seeds']:
            status=load(folder(root,'final',model,seed)/'status.json')
            assert status['best_epoch']==expected and not status['test_scored_before_freeze']
            assert stamp(selection['frozen_utc'])<stamp(status['started_utc'])<stamp(status['completed_utc'])<stamp(freeze['frozen_utc'])
    op=pd.read_csv(root/'results/operating_points.csv',float_precision='round_trip')
    matched=pd.read_csv(root/'results/matched_recall.csv',float_precision='round_trip')
    group_rows=pd.read_csv(root/'results/groups.csv',float_precision='round_trip')
    assert len(op)==72 and len(matched)==288 and len(group_rows)==216
    support={}
    replay_rows=[]
    for stage in ('development','final'):
        study=study_for(arrays(root,stage),'X1') if replay else None
        ref=None
        reference_ids=None
        updates=None
        for model,seed in jobs(cfg):
            path=folder(root,stage,model,seed)
            raw=read_assessment(root,stage,model,seed,True)
            ref=raw if ref is None else ref
            shared(raw,ref)
            assert raw['score'].dtype==np.float64 and raw['probability'].dtype==np.float32 and raw['logit'].dtype==np.float32
            np.testing.assert_array_equal(raw['score'],raw['probability'].astype(np.float64))
            assert np.isfinite(raw['score']).all()
            status=load(path/'status.json')
            assert status['weight_sha256']==sha(path/'weight.pt')
            ids=read_npz(path/'training_ids.npz')
            assert np.isin(ids['y'],[0,1]).all()
            assert len(ids['y'])==status['known_train']==status['unique_training']
            reference_ids=ids['sample_id'] if reference_ids is None else reference_ids
            np.testing.assert_array_equal(reference_ids,ids['sample_id'])
            assert not np.intersect1d(ids['sample_id'],raw['sample_id']).size
            updates=status['updates'] if updates is None else updates
            assert status['updates']==updates
            curve=pd.read_csv(path/'curve.csv')
            assert len(curve)==cfg['epochs'] and (curve.unique_seen==len(ids['y'])).all()
            if replay:
                np.testing.assert_array_equal(study.keys(study.train),ids['sample_id'])
                replay_rows.append(replay_one(root,cfg,stage,model,seed,study,raw))
            if stage=='final':
                entry,=[e for e in freeze['models'] if e['model']==model and e['seed']==seed]
                for name,digest in entry['files'].items():assert sha(path/name)==digest
                cal=(raw['phase']==1)&(raw['y']>=0)
                cut=independent_cut(raw['y'][cal],raw['score'][cal],raw['risk'][cal],28*cfg['FP_budget_day'],28*cfg['recovery_budget_day'])
                close(cut,entry['options']['constrained']['threshold'])
                for phase,days in ((1,28),(2,45)):
                    for policy,option in entry['options'].items():
                        row=op[(op.model==model)&(op.seed==seed)&(op.phase==phase)&(op.policy==policy)].iloc[0]
                        k=(raw['phase']==phase)&(raw['y']>=0)
                        recompute(raw,k,option['threshold'],days,row)
                        assert row.scheduled==(raw['phase']==phase).sum()
                        assert row.unknown==((raw['phase']==phase)&(raw['y']<0)).sum()
                        for number,name in enumerate(('legal_not_low','legal_low','legal_missing')):
                            row=group_rows[(group_rows.model==model)&(group_rows.seed==seed)&(group_rows.phase==phase)
                                &(group_rows.policy==policy)&(group_rows.scope==name)].iloc[0]
                            recompute(raw,k&(raw['legal_group']==number),option['threshold'],days,row)
                    for name,condition in (('all',np.ones(len(k),bool)),('legal_low',raw['legal_group']==1)):
                        table=matched[(matched.model==model)&(matched.seed==seed)&(matched.phase==phase)&(matched.scope==name)]
                        z={key:raw[key][k&condition] for key in ('sample_id','y','score','risk')}
                        for point in table.to_dict('records'):verify_point(z,point,cfg,days)
                check_support(raw['sample_id'],[cfg['full_calibration'],cfg['full_test']],[])
                check_support(ids['sample_id'],[cfg['full_train']],[cfg['full_calibration'],cfg['full_test']])
                # Reused static metrics must remain exactly the original release values.
                if model in STATIC:
                    original=load(root/'evidence/static_reference_summary.json')['2']['constrained'][model]
                    current=load(root/'results/summary.json')['2']['constrained'][model]
                    assert original==current
    if replay:assert len(replay_rows)==36
    receipt=dict(status='PASS',checked_utc=now(),operating_points=len(op),matched_points=len(matched),
        group_metrics=len(group_rows),new_dynamic_fits=18,reused_static_fits=18,
        model_replay_count=len(replay_rows),maximum_probability_error=max((r['max_probability_error'] for r in replay_rows),default=None),
        replay=replay_rows,threshold_fit_only_calibration=True,all_known_train_used=True,
        unknown_excluded=True,static_metrics_identical=True,fresh_blind_validation=False,
        threshold_and_weights_frozen_before_current_round_test_scoring=True,full_clean_retraining_performed=False)
    save(out/('evidence/verification_dynamic_replay.json' if replay else 'evidence/verification_dynamic_offline.json'),receipt)
    print({k:v for k,v in receipt.items() if k!='replay'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',type=Path)
    p.add_argument('--replay',action='store_true')
    a=p.parse_args()
    out,cfg=settings()
    verify((a.artifacts or out).resolve(),out,cfg,a.replay)

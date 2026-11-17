"""Paired real-data edge ablations, date intervals, error groups, and figures."""
from datetime import date
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from full_io import load,save,settings,now
from full_data import arrays,folder,study_for,TrainPreprocessor,predict,set_seed
from full_score import macro,measure,write_csv
from dynamic_score import ALL,STATIC,DYNAMIC,jobs,read_assessment
from dynamic_model import build_model
from v9_analysis import trace,BOOT


def analyze(root,cfg):
    freeze=load(root/'evidence/dynamic_threshold_freeze.json')
    cuts={(r['model'],r['seed']):r['options']['constrained']['threshold'] for r in freeze['models']}
    data=arrays(root,'final')
    study=study_for(data,'X1')
    road_connected=data['adj'].any(axis=(1,2))
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    intervals,daily,topology,resource=[],[],[],[]
    weight_files={}
    for phase,period in ((1,cfg['full_calibration']),(2,cfg['full_test'])):
        first=(date.fromisoformat(period[0])-date(1970,1,1)).days
        days=(date.fromisoformat(period[1])-date.fromisoformat(period[0])).days
        weights=rng.multinomial(days,np.full(days,1/days),size=cfg['bootstrap_reps'])
        weight_files[str(phase)]=weights
        traces={}
        for model,seed in jobs(cfg):
            raw=read_assessment(root,'final',model,seed,True)
            k=(raw['phase']==phase)&(raw['y']>=0)
            threshold=cuts[model,seed]
            y,p,risk=[raw[key][k] for key in ('y','score','risk')]
            pred=p>=threshold
            keys=raw['sample_id'][k]
            traces[model,seed]=trace(keys,y,p,risk,pred,weights,first,cfg['main_recall'])
            road=raw['sample_id']%1000
            ix=np.searchsorted(data['road_ids'],road)
            assert np.array_equal(data['road_ids'][ix],road)
            connected=road_connected[ix]
            for scope,condition in (('connected',connected),('isolated',~connected)):
                keep=(raw['phase']==phase)&(raw['y']>=0)&condition
                topology.append(dict(model=model,seed=seed,phase=phase,scope=scope,days=days,
                    **measure(raw['y'][keep],raw['score'][keep],raw['risk'][keep],threshold,days)))
            day=raw['sample_id']//1000//144
            for d in np.unique(day[raw['phase']==phase]):
                keep=(day==d)&(raw['y']>=0)
                forecast=raw['score']>=threshold
                pos=raw['y']==1
                neg=raw['y']==0
                daily.append(dict(model=model,seed=seed,phase=phase,date=str(np.datetime64(int(d),'D')),
                    known=int(keep.sum()),positive=int((keep&pos).sum()),
                    TP=int((keep&pos&forecast).sum()),FP=int((keep&neg&forecast).sum()),
                    FN=int((keep&pos&~forecast).sum()),TN=int((keep&neg&~forecast).sum()),
                    recovery_FP=int((keep&neg&forecast&raw['risk']).sum())))
        for pool in ('mean','max','attention'):
            a,b='dynamic_'+pool,'graph_'+pool
            delta=np.mean([traces[a,s]-traces[b,s] for s in cfg['seeds']],axis=0)
            for j,metric in enumerate(BOOT):
                values=delta[:,j][np.isfinite(delta[:,j])]
                lo,hi=np.quantile(values,[.025,.975])
                intervals.append(dict(phase=phase,contrast=a+'_minus_'+b,metric=metric,
                    mean_delta=float(np.mean(values)),low95=float(lo),high95=float(hi),replicates=len(values)))
    for stage in ('development','final'):
        for model,seed in jobs(cfg):
            r=load(folder(root,stage,model,seed)/'status.json')
            resource.append(dict(stage=stage,model=model,seed=seed,new_training=model in DYNAMIC,
                **{k:r[k] for k in ('parameters','best_epoch','fit_and_score_seconds','peak_RSS_bytes','updates','known_train','unique_training')}))
    write_csv(root/'results/paired_date_blocks.csv',intervals)
    write_csv(root/'results/daily_errors.csv',daily)
    write_csv(root/'results/topology_groups.csv',topology)
    write_csv(root/'results/training_resources.csv',resource)
    np.savez_compressed(root/'results/date_bootstrap_weights.npz',**weight_files)
    matched=pd.read_csv(root/'results/matched_recall.csv',float_precision='round_trip').to_dict('records')
    summary={str(phase):{model:dict(matched60=macro([r for r in matched if r['phase']==phase and r['model']==model
         and r['scope']=='all' and r['target_recall']==.6],('FP_per_day','recovery_FP_per_day','F1','achieved_recall')),
         topology={scope:macro([r for r in topology if r['phase']==phase and r['model']==model and r['scope']==scope])
             for scope in ('connected','isolated')}) for model in ALL} for phase in (1,2)}
    save(root/'results/paired_summary.json',summary)
    edge_diagnostics(root,cfg,data,study,cuts)
    figures(root,cfg)
    print(dict(status='PASS',paired_rows=len(intervals),topology_rows=len(topology),daily_rows=len(daily)),flush=True)


def edge_diagnostics(root,cfg,data,study,cuts):
    assess=np.flatnonzero(study.phase>0)
    center_road=assess%study.nroad
    connected=data['adj'].any(axis=(1,2))
    eligible=assess[connected[center_road]]
    selection=np.sort(np.random.default_rng(1811).choice(eligible,min(1024,len(eligible)),replace=False))
    summaries,ablation=[],[]
    out=root/'results/edge_weights'
    out.mkdir(parents=True,exist_ok=True)
    for model in DYNAMIC:
        for seed in cfg['seeds']:
            path=folder(root,'final',model,seed)
            study.configure(TrainPreprocessor.load(path/'preprocessor.npz'))
            set_seed(seed,cfg)
            net=build_model(104,cfg,model,seed)
            net.load_state_dict(torch.load(path/'weight.pt',map_location='cpu',weights_only=True))
            net.eval()
            x,adj,valid,center=map(torch.from_numpy,study.batch(selection))
            with torch.inference_mode():
                bias=torch.stack([gate(x,adj,valid) for gate in net.gates],1)
                multiplier=bias.exp().numpy()
            active=adj.transpose(1,2)&valid[:,:,None]&valid[:,None,:]
            mask=active.numpy()[:,None,None,:,:]
            expanded=np.broadcast_to(mask,multiplier.shape)
            real=multiplier[expanded]
            summaries.append(dict(model=model,seed=seed,graphs=len(selection),physical_head_layer_edges=len(real),
                min=float(real.min()),p10=float(np.quantile(real,.1)),median=float(np.median(real)),
                p90=float(np.quantile(real,.9)),max=float(real.max()),neutral_fraction=float((real==1).mean()),
                below_one_fraction=float((real<1).mean()),above_one_fraction=float((real>1).mean())))
            np.savez_compressed(out/f'{model}_s{seed}.npz',sample_id=study.keys(selection),
                physical_multiplier=multiplier,adjacency=adj.numpy(),valid=valid.numpy(),
                road_ids=data['road_ids'][data['lookup'][selection%study.nroad]])
            net.edge_enabled=False
            logits,prob=predict(net,study,assess,cfg,False)
            np.savez_compressed(out/f'{model}_s{seed}_neutral_scores.npz',sample_id=study.keys(assess),
                probability=prob,logit=logits)
            # This is a post-fit mechanism diagnostic; no retraining or threshold refit.
            raw=read_assessment(root,'final',model,seed,True)
            for phase,days in ((1,28),(2,45)):
                k=(study.phase[assess]==phase)&(study.y[assess]>=0)
                ablation.append(dict(model=model,seed=seed,phase=phase,threshold=cuts[model,seed],
                    scope='same trained backbone, dynamic edge priors neutralized only at inference',
                    **measure(study.y[assess][k],prob[k].astype(float),study.risk[assess][k],cuts[model,seed],days),
                    maximum_probability_change=float(np.abs(prob[k].astype(float)-raw['score'][k]).max())))
    write_csv(root/'results/edge_multiplier_summary.csv',summaries)
    write_csv(root/'results/edge_neutralization_diagnostic.csv',ablation)
    save(out/'scope.json',dict(at_utc=now(),engineering=False,selection='fixed seed 1811, connected assessment graphs, no labels used',
        graphs_per_fit=len(selection),orientation='[graph,layer,head,target,source]',self_multiplier=1,
        note='Only physical directed edges enter summaries; neutral scores are diagnostic, not a new selected model.'))


def figures(root,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=root/'figures'
    output.mkdir(exist_ok=True)
    stats=load(root/'results/summary.json')['2']['constrained']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,pool in zip(axes,('mean','max','attention'),strict=True):
        for prefix,color in (('graph_','#7b8492'),('dynamic_','#1d7768')):
            for i,seed in enumerate(cfg['seeds']):
                model=prefix+pool
                raw=read_assessment(root,'final',model,seed,True)
                k=(raw['phase']==2)&(raw['y']>=0)
                p,r,_=precision_recall_curve(raw['y'][k],raw['score'][k])
                label=f"{prefix[:-1]}, AP={stats[model]['AP']:.3f}" if i==0 else None
                ax.step(r,p,where='post',lw=1,alpha=.65,color=color,label=label)
        ax.set(title=pool.title(),xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('NYC strict 10-minute warning | retrospective, same support')
    fig.savefig(output/'paired_PR.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,('AP','f1','recall'),strict=True):
        for j,prefix in enumerate(('graph_','dynamic_')):
            models=[prefix+p for p in ('mean','max','attention')]
            ax.bar(np.arange(3)+(j-.5)*.36,[stats[m][key] for m in models],.36,
                yerr=[stats[m][key+'_sd'] for m in models],label=prefix[:-1],capsize=3)
        ax.set(xticks=np.arange(3),xticklabels=['Mean','Max','Attention'],ylabel=key,ylim=(0,1))
        ax.axhline(.6,color='#ba6f26',ls='--',lw=.8)
        ax.grid(axis='y',alpha=.15)
        ax.legend(fontsize=8)
    fig.savefig(output/'paired_metrics.png',dpi=160)
    plt.close(fig)
    paired=load(root/'results/paired_summary.json')['2']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,label in zip(axes,('FP_per_day','recovery_FP_per_day'),('Total FP/day','Recovery FP/day'),strict=True):
        for j,prefix in enumerate(('graph_','dynamic_')):
            values=[paired[prefix+p]['matched60'][key] for p in ('mean','max','attention')]
            ax.bar(np.arange(3)+(j-.5)*.36,values,.36,label=prefix[:-1])
        ax.set(xticks=np.arange(3),xticklabels=['Mean','Max','Attention'],ylabel=label,title='60% recall diagnostic')
        ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(output/'matched_recall_FP.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,pool in zip(axes,('mean','max','attention'),strict=True):
        for prefix in ('graph_','dynamic_'):
            curve=pd.read_csv(folder(root,'development',prefix+pool,11)/'curve.csv')
            ax.plot(curve.epoch,curve.AP,label=prefix[:-1])
        ax.set(title=pool.title(),xlabel='Epoch',ylabel='Development AP')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(output/'development_curves.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':analyze(*settings())

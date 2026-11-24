"""Project A teacher submission: default aggregate verification, explicit full rerun."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
_dll_handle=None
if os.name=='nt' and (Path(sys.prefix)/'Library/bin').is_dir():
    _dll_handle=os.add_dll_directory(str(Path(sys.prefix)/'Library/bin'))
    os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,hashlib,json,subprocess,time
from datetime import datetime
import numpy as np
import pandas as pd

def verify(out):
    result=ROOT/'results'
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((result/'analysis_summary.json').read_text(encoding='utf-8'))
    db=pd.read_csv(result/'dbscan_experiments.csv')
    km=pd.read_csv(result/'kmeans_experiments.csv')
    h=pd.read_csv(result/'hotspots.csv')
    k=pd.read_csv(result/'kmeans_clusters.csv')
    common=pd.read_csv(result/'common_sample_comparison.csv')
    grid=pd.read_csv(result/'density_bias_grid.csv')
    temporal=pd.read_csv(result/'time_period_hotspots.csv')
    assert len(db)==len(cfg['eps_m'])*len(cfg['min_samples'])==15
    assert db.n.eq(summary['points']).all()
    assert np.allclose(db.coverage+db.noise_fraction,1)
    main=db[(db.eps_m==cfg['main_eps_m'])&(db.min_samples==cfg['main_min_samples'])].iloc[0]
    assert main.clusters==len(h)==174
    assert int(h.points.sum())==30040
    assert abs(h.points.sum()/summary['points']-main.coverage)<1e-12
    assert len(k)==summary['kmeans_k']==5 and int(k.points.sum())==summary['points']==53786
    ks=km.k.to_numpy();inertia=km.inertia.to_numpy()
    xn=(ks-ks.min())/(ks.max()-ks.min());yn=(inertia-inertia.min())/(inertia.max()-inertia.min())
    assert int(ks[np.argmax(1-xn-yn)])==summary['kmeans_k']
    assert common.common_n.eq(30040).all()
    assert abs(common[common.model=='DBSCAN'].silhouette.iloc[0]-main.silhouette)<1e-12
    assert abs(.5*np.abs(grid.raw_share-grid.thinned_share).sum()-summary['grid_bias_total_variation'])<1e-12
    assert np.isclose(grid.raw_share.sum(),1) and np.isclose(grid.thinned_share.sum(),1)
    assert int(temporal.points.sum())==summary['points']
    assert np.allclose(temporal.groupby('period').within_period_share.sum(),1)
    for name in ('hotspots.csv','kmeans_clusters.csv','user_summary_anonymous.csv','candidate_histogram.csv'):
        cols=pd.read_csv(result/name,nrows=0).columns
        assert not set(cols)&{'user_id','trajectory_id','time','time_start','time_end','lon','lat','x','y','x_min','x_max','y_min','y_max'}
    figs=sorted((result/'figures').glob('*.png'))
    assert len(figs)==5 and all(f.stat().st_size>10000 for f in figs)
    report=(result/'experiment_report.md').read_text(encoding='utf-8')
    for f in figs:assert 'figures/'+f.name in report
    receipt=dict(status='PASS',mode='offline aggregate consistency verification',raw_data_required=False,
        points=summary['points'],dbscan_clusters=174,dbscan_assigned=30040,kmeans_k=5,figures=5,
        candidate_count=summary['candidate_count'],
        scope='aggregate arithmetic, elbow selection, privacy column exclusion and figure presence',
        not_recomputed_here=['point-level clustering','silhouette sampling','stability ARI'],
        note='independent original-point refit evidence is separately provided; no private point tables in submission')
    (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description='Project A: safe teacher submission')
    p.add_argument('command',nargs='?',default='verify',choices=('verify','full','tests'))
    p.add_argument('--data',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    out=(args.out or ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out==ROOT or out.is_relative_to(ROOT/'src') or out.is_relative_to(ROOT/'results'):raise ValueError('Use a separate output directory')
    if out.exists():raise FileExistsError('Output exists; choose a fresh --out')
    if args.command=='full':
        if args.data is None or not args.data.is_file():p.error('full requires --data pointing to your authorized Geolife ZIP')
        command=[sys.executable,'-B','-X','utf8',str(ROOT/'run_original.py'),'--data',str(args.data.resolve()),'--out',str(out),'--stage','all','--config',str(ROOT/'config.json')]
    elif args.command=='tests':command=[sys.executable,'-B','-X','utf8','-m','unittest','discover','-s',str(ROOT/'tests'),'-v']
    else:command=None
    out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    if command:
        with (out/'console.log').open('w',encoding='utf-8') as stream:
            proc=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT)
        code=proc.returncode
    else:verify(out);code=0
    receipt=dict(command=command or [sys.executable,str(ROOT/'run.py'),'verify'],exit_code=code,elapsed_seconds=time.perf_counter()-tick,
        executable=sys.executable,raw_data_included=False,output=str(out))
    (out/'command_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Exit {code}: {out}',flush=True)
    return code
if __name__=='__main__':sys.exit(main())

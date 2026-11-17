"""Single reproducible entry point. Raw ZIP is always read-only."""
import os
import sys
# Direct python.exe invocation does not activate Conda's native DLL search path.
_dll_handle=None
if os.name=='nt' and os.path.isdir(os.path.join(sys.prefix,'Library','bin')):
    _bin=os.path.join(sys.prefix,'Library','bin')
    _dll_handle=os.add_dll_directory(_bin)
    os.environ['PATH']=_bin+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import pandas as pd
from src.data import inventory,prepare

def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        '--data',
        type=Path,
        default=Path(r'G:\course-jiaotongshujuwajue\dataset\Geolife Trajectories 1.3.zip'),
        help='Path to original Geolife 1.3 ZIP'
    )
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('results'))
    p.add_argument('--stage',choices=['all','inventory','prepare','analyze','report','verify'],default='all')
    args=p.parse_args(); cfg=json.loads(args.config.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    if args.stage in ['all','inventory','prepare'] and (not args.data or not args.data.is_file()):
        p.error('--data must point to an existing ZIP')
    if args.stage in ['all','inventory']:
        inv=inventory(args.data,args.out,cfg)
    if args.stage in ['all','prepare']:
        inv=pd.read_csv(args.out/'inventory.csv',dtype={'user_id':str})
        prepare(args.data,args.out,cfg,inv)
    if args.stage in ['all','analyze']:
        from src.analysis import analyze
        analyze(args.out,cfg)
    meta_path=args.out/'run_metadata.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() and args.stage!='all' else {}
    meta.update(config=cfg,python=platform.python_version(),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','matplotlib','scipy','pyproj','psutil']})
    if args.stage in ['all','inventory','prepare','analyze']:
        meta.setdefault('stage_seconds',{})[args.stage]=time.perf_counter()-start
    if args.data:
        meta['input_zip_name']=args.data.name
        meta['input_zip_bytes']=args.data.stat().st_size
        with args.data.open('rb') as f: meta['input_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.stage in ['all','report']:
        from src.report import report
        report(args.out,cfg)
    if args.stage in ['all','verify']:
        from src.verify import verify
        verify(args.out,cfg)
    print(f'Completed {args.stage}: {args.out}',flush=True)

if __name__=='__main__': main()

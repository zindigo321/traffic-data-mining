"""Portable Project B launcher. Uses the active niulaoshi interpreter, never installs."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
COMMANDS=('check','verify','replay','tests','smoke','prepare','predict','train')
PACKAGES={'torch':'torch','numpy':'numpy','pandas':'pandas','scikit-learn':'sklearn','scipy':'scipy',
    'psutil':'psutil','matplotlib':'matplotlib','requests':'requests','tzdata':'tzdata',
    'geopandas':'geopandas','shapely':'shapely','pyproj':'pyproj','pyogrio':'pyogrio','joblib':'joblib'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_integrity():
    manifest=ROOT/'manifest.csv'
    sums=ROOT/'SHA256SUMS'
    if not manifest.is_file() or not sums.is_file():
        raise FileNotFoundError('manifest.csv or SHA256SUMS is missing. Extract the entire runtime package.')
    expected={}
    for line in sums.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        expected[name]=digest
    if expected.get('manifest.csv')!=sha(manifest):raise ValueError('manifest SHA mismatch')
    count=0
    with manifest.open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            rel=PurePosixPath(row['path'])
            if rel.is_absolute() or '..' in rel.parts or any(':' in p or '\\' in p for p in rel.parts):
                raise ValueError('Unsafe manifest path')
            path=ROOT.joinpath(*rel.parts)
            if not path.resolve().is_relative_to(ROOT) or not path.is_file():raise ValueError(f'Missing {rel}')
            digest=sha(path)
            if path.stat().st_size!=int(row['size_bytes']) or digest!=row['sha256'] or expected.get(rel.as_posix())!=digest:
                raise ValueError(f'Hash mismatch: {rel}')
            count+=1
    if len(expected)!=count+1:raise ValueError('Checksum count differs from manifest')
    return count


def environment():
    if sys.version_info<(3,11):raise RuntimeError('Use Python 3.11 or later in the niulaoshi environment.')
    if Path(sys.prefix).name.lower()!='niulaoshi':
        raise RuntimeError(f'The active environment is not niulaoshi: {sys.executable}. Run conda activate niulaoshi first.')
    missing=[name for name,module in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:raise RuntimeError('Missing dependencies: '+', '.join(missing)+'. The launcher does not modify your Conda environment.')
    return dict(executable=sys.executable,python=sys.version,prefix=sys.prefix,
        packages={name:importlib.metadata.version(name) for name in PACKAGES})


def main():
    p=argparse.ArgumentParser(description='Project B dynamic edge weights: local niulaoshi launcher')
    p.add_argument('command',choices=COMMANDS,nargs='?',default='verify')
    p.add_argument('--output',type=Path,help='New output directory, defaulting to runs/<timestamp>_<command> in the package')
    p.add_argument('--pool',choices=('mean','max','attention'),default='mean')
    p.add_argument('--seed',type=int,choices=(11,22,33),default=11)
    p.add_argument('--phase',choices=('all','initialize','tests','development','select','final','freeze','evaluate','analysis','verify','report'),default='all')
    args=p.parse_args()
    info=environment()
    checked=verify_integrity()
    output=(args.output or ROOT/'runs'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.command)).resolve()
    if output==ROOT or output.is_relative_to(ROOT/'reference') or output.is_relative_to(ROOT/'data') or output.is_relative_to(ROOT/'src'):
        raise ValueError('Write to a separate directory or runs/. The source, data, and reference experiment directories are protected.')
    if output.exists() and not (args.command=='train' and args.phase not in ('all','initialize')):
        raise FileExistsError(f'The output directory already exists. Choose a different path: {output}')
    if args.command!='train':output.mkdir(parents=True,exist_ok=False)
    os.environ['PROJECTB_PACKAGE_ROOT']=str(ROOT)
    os.environ['PROJECTB_PREPARED_DIR']=str(ROOT/'data/prepared')
    env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','GTFULL_ROOT':str(output)}
    env['PYTHONPATH']=str(ROOT/'src')
    dll=Path(sys.prefix)/'Library/bin'
    if dll.is_dir():env['PATH']=str(dll)+os.pathsep+env.get('PATH','')
    command=[sys.executable,'-B','-X','utf8',str(ROOT/'src/portable_tasks.py'),args.command,'--output',str(output),
             '--pool',args.pool,'--seed',str(args.seed),'--phase',args.phase]
    print(f'Python: {sys.executable}\nPayload hashes: {checked}\nCommand: {args.command}\nOutput: {output}',flush=True)
    tick=time.perf_counter()
    started=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(command,env=env,cwd=ROOT/'src')
    receipt=dict(command=command,cwd=str(ROOT/'src'),started_utc=started,elapsed_seconds=time.perf_counter()-tick,
        exit_code=result.returncode,environment=info,package_root=str(ROOT),data_root=str(ROOT/'data'),
        dependency_installation=False,package_hashes_checked=checked,external_task_paths_required=False)
    save(output/'launcher_receipt.json',receipt)
    print(f'Exit: {result.returncode}; receipts: {output}',flush=True)
    return result.returncode


if __name__=='__main__':
    try:sys.exit(main())
    except (RuntimeError,ValueError,FileNotFoundError,FileExistsError) as error:
        print(str(error),file=sys.stderr)
        sys.exit(2)

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
        raise FileNotFoundError('缺少 manifest.csv / SHA256SUMS，请完整解压运行包。')
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
    if sys.version_info<(3,11):raise RuntimeError('请使用 Python 3.11 或更高版本的 niulaoshi 环境。')
    if Path(sys.prefix).name.lower()!='niulaoshi':
        raise RuntimeError(f'当前环境不是 niulaoshi：{sys.executable}。请先 conda activate niulaoshi。')
    missing=[name for name,module in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:raise RuntimeError('缺少依赖：'+', '.join(missing)+'。不会自动修改您的 Conda 环境。')
    return dict(executable=sys.executable,python=sys.version,prefix=sys.prefix,
        packages={name:importlib.metadata.version(name) for name in PACKAGES})


def main():
    p=argparse.ArgumentParser(description='Project B 动态边权：niulaoshi 完整本地运行入口')
    p.add_argument('command',choices=COMMANDS,nargs='?',default='verify')
    p.add_argument('--output',type=Path,help='新输出目录；默认包内 runs/时间戳_命令')
    p.add_argument('--pool',choices=('mean','max','attention'),default='mean')
    p.add_argument('--seed',type=int,choices=(11,22,33),default=11)
    p.add_argument('--phase',choices=('all','initialize','tests','development','select','final','freeze','evaluate','analysis','verify','report'),default='all')
    args=p.parse_args()
    info=environment()
    checked=verify_integrity()
    output=(args.output or ROOT/'runs'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.command)).resolve()
    if output==ROOT or output.is_relative_to(ROOT/'reference') or output.is_relative_to(ROOT/'data') or output.is_relative_to(ROOT/'src'):
        raise ValueError('输出必须在独立目录或 runs/，不能覆盖代码、数据或参考实验。')
    if output.exists() and not (args.command=='train' and args.phase not in ('all','initialize')):
        raise FileExistsError(f'输出目录已存在，请另选名称：{output}')
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

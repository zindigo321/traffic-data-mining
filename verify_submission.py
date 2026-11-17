"""Check the submission hashes; optionally run both offline project verifiers."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse, csv, hashlib, json, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(root):
    sums={}
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if name in sums:raise ValueError('duplicate checksum')
        sums[name]=digest
    if sha(root/'manifest.csv')!=sums.pop('manifest.csv'):raise ValueError('manifest SHA mismatch')
    with (root/'manifest.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if len(sums)!=len(rows):raise ValueError('checksum count mismatch')
    for row in rows:
        rel=PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or any(':' in p or '\\' in p for p in rel.parts):raise ValueError('unsafe path')
        p=root.joinpath(*rel.parts)
        if not p.resolve().is_relative_to(root) or not p.is_file():raise ValueError('missing '+str(rel))
        if p.stat().st_size!=int(row['size_bytes']) or sha(p)!=row['sha256'] or sums.get(str(rel))!=row['sha256']:
            raise ValueError('hash/size mismatch '+str(rel))
    return len(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();tick=time.perf_counter();count=verify(ROOT)
    receipt=dict(status='PASS',manifest_entries=count,checksums_self_excluded=True,
      started_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,commands=[])
    out=(a.output or ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    if a.run:
        env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
        for project,arg in [('ProjectA','--out'),('ProjectB','--output')]:
            command=[sys.executable,'-B','-X','utf8',str(ROOT/project/'run.py'),arg,str(out/project)]
            t=time.perf_counter()
            with (out/(project+'_console.log')).open('w',encoding='utf-8') as stream:
                proc=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            receipt['commands'].append(dict(command=command,exit_code=proc.returncode,elapsed_seconds=time.perf_counter()-t))
            if proc.returncode:receipt['status']='FAIL'
    receipt['elapsed_seconds']=time.perf_counter()-tick
    (out/'验收.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
    return int(receipt['status']!='PASS')

if __name__=='__main__':sys.exit(main())

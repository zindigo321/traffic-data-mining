"""Package and freshly extract the measured dynamic study, without raw tables."""
import argparse
import csv
from pathlib import Path,PurePosixPath
import shutil
import stat
import zipfile

from full_io import load,save,now,sha
from dynamic_check import directory

NAME='ProjectB_GraphTransformer_dynamic_edge_weights_audit'
DIRECTORIES=('project','evidence','results','logs','figures','reports','synthetic','source_provenance')
DOCS=('README_dynamic_edge_weights.md','plan.md','method_and_data_contract.md','audit_report.md','test_report.md')


def files(root):
    return sorted((p.relative_to(root).as_posix(),p) for p in root.rglob('*') if p.is_file())


def build(root,output):
    assert load(root/'evidence/verification_dynamic_replay.json')['status']=='PASS'
    assert load(root/'evidence/clean_qa_prepackage.json')['status']=='PASS'
    output.mkdir(parents=True,exist_ok=False)
    bundle=output/NAME
    bundle.mkdir()
    for name in DOCS:shutil.copy2(root/name,bundle/name)
    for name in DIRECTORIES:
        shutil.copytree(root/name,bundle/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.tmp','*.part'))
    purpose=dict(project='Source, tests and frozen configuration',evidence='Data, registration, calibration or replay evidence',
        results='Real measured predictions, weights, IDs, curves, metrics and paired analyses',logs='Actual commands, outputs and receipts',
        figures='Plots of measured retrospective results',reports='Change record',synthetic='Engineering fixtures only',
        source_provenance='Read-only prior source/terms evidence; not new instructions')
    entries=[]
    for rel,p in files(bundle):
        if p.suffix.lower() in ('.pkl','.pickle','.plt','.zip','.exe','.csv.gz') or 'raw_external/' in rel:
            raise ValueError(f'Unexpected package member {rel}')
        entries.append((rel,p.stat().st_size,sha(p),purpose.get(rel.split('/')[0],'Audit or handoff narrative')))
    manifest=bundle/'manifest.csv'
    with manifest.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(('path','size_bytes','sha256','purpose'))
        w.writerows(entries)
    with (bundle/'SHA256SUMS').open('w',encoding='utf-8',newline='\n') as f:
        for rel,_,digest,_ in entries:f.write(f'{digest}  {rel}\n')
        f.write(f'{sha(manifest)}  manifest.csv\n')
    archive=output/(NAME+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for rel,p in files(bundle):z.write(p,f'{NAME}/{rel}')
    save(output/'delivery_verification.json',dict(created_utc=now(),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
        bundle=str(bundle),manifest=directory(bundle)))
    print(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size),flush=True)


def extract(archive,out):
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=set()
        for item in z.infolist():
            path=PurePosixPath(item.filename)
            if (len(path.parts)<2 or path.parts[0]!=NAME or '..' in path.parts or
                any(':' in s or '\\' in s for s in path.parts) or item.filename.casefold() in names
                or stat.S_ISLNK(item.external_attr>>16)):
                raise ValueError(f'Unsafe ZIP member {item.filename}')
            names.add(item.filename.casefold())
        if z.testzip() is not None:raise ValueError('CRC failure')
        z.extractall(out)
    receipt=dict(status='PASS',checked_utc=now(),archive=str(archive),sha256=sha(archive),
        extracted=str(out/NAME),**directory(out/NAME))
    save(out/'extraction_receipt.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('build','extract','check'))
    p.add_argument('--run',type=Path)
    p.add_argument('--archive',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.action=='build':build(a.run.resolve(),a.output.resolve())
    elif a.action=='extract':extract(a.archive.resolve(),a.output.resolve())
    else:print(directory(a.output.resolve()),flush=True)

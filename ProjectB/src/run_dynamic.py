"""One command runs the registered causal dynamic-edge study, with receipts."""
import argparse
from pathlib import Path
import os
import shutil

from full_io import CODE,python,record,load,save,now

PHASES=('check','tests','development','select','final','freeze','evaluate','analysis','verify','report')


def run(root,phase,artifacts=None,no_replay=False):
    if phase=='tests':
        paths=(CODE,CODE/'v10',CODE/'v10/v9',CODE/'v10/v9/v8',CODE/'v10/v9/v8/v7',
            CODE/'v10/v9/v8/v7/v6',CODE/'v10/v9/v8/v7/v6/v5',CODE/'v10/v9/v8/v7/v6/v5/v4',
            CODE/'v10/v9/v8/v7/v6/v5/v4/v3')
        import re
        counts=[]
        for i,p in enumerate(paths):
            r=record(root,f'engineering_dynamic_{i}',python('-m','unittest','discover','-s','tests','-v'),cwd=p)
            txt=(root/'logs'/r['log']).read_text(encoding='utf-8')
            counts.append(int(re.search(r'Ran (\d+) tests?',txt).group(1)))
        save(root/'synthetic/engineering_status.json',dict(status='PASS',engineering_only=True,counts=counts,total=sum(counts),at_utc=now()))
        return
    commands=dict(check=['dynamic_check.py'],development=['dynamic_train.py','matrix','--stage','development'],
        select=['dynamic_score.py','selection'],final=['dynamic_train.py','matrix','--stage','final'],
        freeze=['dynamic_score.py','freeze'],evaluate=['dynamic_score.py','evaluate'],analysis=['dynamic_analysis.py'],
        verify=['dynamic_verify.py',*(['--artifacts',artifacts] if artifacts else []),
                *([] if no_replay else ['--replay'])],report=['dynamic_report.py'])
    record(root,'dynamic_'+phase,python(*commands[phase]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('all','continue',*PHASES),default='all')
    p.add_argument('--output',type=Path,default=CODE.parent)
    p.add_argument('--source-run',type=Path,default=CODE.parent)
    p.add_argument('--artifacts',type=Path)
    p.add_argument('--no-replay',action='store_true')
    a=p.parse_args()
    root=a.output.resolve()
    if a.phase=='all' and root!=CODE.parent:
        root.mkdir(parents=True,exist_ok=False)
        shutil.copytree(CODE,root/'project',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(a.source_run/'plan.md',root/'plan.md')
        for name in ('path_mapping.json','prepared.json','source_snapshot.json','graph.json','original_graph.json',
                     'cohort_development.json','cohort_final.json','nyc_edges.csv','nyc_roads.csv'):
            (root/'evidence').mkdir(exist_ok=True)
            shutil.copy2(a.source_run/'evidence'/name,root/'evidence'/name)
        mapping=load(root/'evidence/path_mapping.json')
        mapping['run']=str(root)
        save(root/'evidence/path_mapping.json',mapping)
    os.environ['GTFULL_ROOT']=str(root)
    root.mkdir(exist_ok=True,parents=True)
    phases=PHASES if a.phase=='all' else PHASES[2:] if a.phase=='continue' else (a.phase,)
    for phase in phases:run(root,phase,a.artifacts,a.no_replay)


if __name__=='__main__':main()

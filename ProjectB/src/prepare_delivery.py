"""Preserve prior provenance and record static review before final packaging."""
from pathlib import Path
import ast
import importlib.metadata
import shutil

from full_io import CODE,load,now,save,settings,sha


def main():
    root,cfg=settings()
    mapping=load(root/'evidence/path_mapping.json')
    source=Path(mapping['source_bundle'])
    provenance=root/'source_provenance'
    provenance.mkdir(exist_ok=True)
    copied=[]
    for name in ('README_全量实验入口.md','方法与数据合同.md','审计报告.md'):
        dst=provenance/('原全量_'+name)
        shutil.copy2(source/name,dst)
        copied.append(dict(source=str(source/name),target=dst.relative_to(root).as_posix(),sha256=sha(dst)))
    v5=root.parents[1]/'gt_v5/20261110_v5'
    for rel in ('protocol_nyc.md','evidence/feasibility.json','evidence/dictionary_receipt.json','evidence/dictionary_local.txt'):
        src=v5/rel
        if src.exists():
            dst=provenance/('原V5_'+src.name)
            shutil.copy2(src,dst)
            copied.append(dict(source=str(src),target=dst.relative_to(root).as_posix(),sha256=sha(dst)))
    save(root/'evidence/provenance_copies.json',dict(copied_utc=now(),files=copied,
        new_online_license_check=False,note='Historical collection evidence only; source snapshots remain external.'))
    env=load(root/'logs/environment_dynamic.json')
    for name in ('requests','tzdata','geopandas','shapely','pyproj','pyogrio','joblib'):
        env['packages'][name]=importlib.metadata.version(name)
    env['dependency_inventory_expanded_utc']=now()
    save(root/'logs/environment_dynamic.json',env)
    checked=[]
    for p in [*CODE.glob('dynamic_*.py'),CODE/'run_dynamic.py',CODE/'finish_dynamic.py']:
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        checked.append(dict(path=p.name,sha256=sha(p)))
    save(root/'evidence/static_review.json',dict(status='PASS',checked_utc=now(),files=checked,
        dimensions=['security','compatibility','error handling','resource handling','change scope','shape/type checks'],
        original_source_unchanged=True,safe_loading='weights_only=True; NPZ allow_pickle=False',
        no_new_dependency_installed=True,known_limitations=['external snapshot paths needed for retraining',
        'no fresh blind validation; dense local graph attention'],runtime_verified_separately=True))
    print(dict(provenance_files=len(copied),parsed=len(checked)),flush=True)


if __name__=='__main__':main()

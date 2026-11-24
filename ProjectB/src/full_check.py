"""Validate signed ancestors and enumerate every verified raw source shard."""
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys

import psutil

from full_io import CODE, CONFIG, V10, V10_NAME, load, now, save, settings, sha

BASE="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer"
RELEASE=BASE+"/outputs/ProjectB_GraphTransformer_V10"
PIN="08ec34785c6e231555b04ba9302c38f1a8a8c1bc6377ad0a08a7fd0daf2ddd6d"


def register(root,cfg):
    receipt=load(root/"evidence/registration.json")
    assert sha(root/"plan.md")==receipt["plan_sha256"] and sha(CONFIG)==receipt["config_sha256"]
    assert cfg["training_use_all_known"] and all(v in (0,None) for v in cfg["experiment_admission"].values())
    return receipt


def check(root,cfg):
    register(root,cfg)
    release=Path(os.environ.get("GTFULL_V10_RELEASE",RELEASE))
    source=release/V10_NAME
    archive=release/(V10_NAME+".zip")
    assert sha(archive)==PIN
    if not V10.exists():
        shutil.copytree(source/"project",V10)
    sys.path.insert(0,str(V10))
    from v10_check import directory,snapshot
    source_checks={"V10":dict(zip_sha256=PIN,**directory(source))}
    assert snapshot(V10)==snapshot(source/"project")
    raw_entries=[]
    saved_receipts=[]
    specs=[("V10","download_development.json"),("V5","download_holdout_sealed.json"),("V6","download_holdout_sealed.json")]
    for version,filename in specs:
        prior=Path(BASE)/"outputs"/("ProjectB_GraphTransformer"+("" if version=="V1" else "_"+version))
        name="ProjectB_GraphTransformer_"+version+"_\u5ba1\u8ba1\u5305"
        bundle=source if version=="V10" else prior/name
        if version!="V10":
            source_checks[version]=dict(zip_sha256=sha(prior/(name+".zip")),**directory(bundle))
        path=bundle/"evidence"/filename
        receipt=load(path)
        assert receipt["status"]=="complete"
        dest=root/"evidence"/f"{version}_{filename}"
        shutil.copy2(path,dest)
        saved_receipts.append(dict(path=dest.name,sha256=sha(dest),interval=receipt["interval"]))
        directory_path=Path(receipt["directory"])
        for row in receipt["files"]:
            p=directory_path/row["file"]
            assert p.stat().st_size==row["bytes"] and sha(p)==row["sha256"]
            raw_entries.append(dict(path=str(p.resolve()),origin=version,**row))
    intervals=[(e["file"],e["sha256"]) for e in raw_entries]
    assert len({name for name,_ in intervals})==len(intervals)
    geometry=load(source/"evidence/download_geometry.json")
    shutil.copy2(source/"evidence/download_geometry.json",root/"evidence/download_geometry.json")
    for row in geometry["files"]:
        p=Path(geometry["directory"])/row["file"]
        assert sha(p)==row["sha256"]
    shutil.copy2(source/"evidence/graph.json",root/"evidence/original_graph.json")
    save(root/"evidence/sources_cached.json",dict(status="PASS",checked_utc=now(),sources=source_checks,
         receipts=saved_receipts,files=raw_entries,files_count=len(raw_entries),
         rows=sum(e["rows"] for e in raw_entries),private_trajectory_sources=False,
         prior_holdout_dates_used_under_new_user_authorization=True))
    packages=("torch","numpy","pandas","scikit-learn","scipy","matplotlib","requests","psutil","tzdata",
              "geopandas","shapely","pyproj","pyogrio","joblib")
    versions={name:importlib.metadata.version(name) for name in packages}
    cpu=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu=winreg.QueryValueEx(k,"ProcessorNameString")[0]
    save(root/"logs/environment.json",dict(executable=sys.executable,python=sys.version,platform=platform.platform(),cpu=cpu,
         cores=psutil.cpu_count(False),logical=psutil.cpu_count(True),RAM_bytes=psutil.virtual_memory().total,
         threads=cfg["cpu_threads"],workers=cfg["workers"],device="CPU",seeds=cfg["seeds"],packages=versions))
    (root/"project/requirements.txt").write_text("".join(f"{k}=={v}\n" for k,v in versions.items()),encoding="utf-8")
    print(dict(status="PASS",cached_shards=len(raw_entries),rows=sum(e["rows"] for e in raw_entries),
               signed_releases=list(source_checks)),flush=True)


if __name__=="__main__":
    check(*settings())

"""Acquire the available continuation without date or coverage admission gates."""
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from pathlib import Path

from full_io import V10,load,now,save,settings,sha
from full_check import register

sys.path.insert(0,str(V10))
from v10_probe import query
from v10_data import study_for
from v5_download import OBSERVATIONS,run_shard,split_specs


def fetch(root,cfg):
    register(root,cfg)
    pinned=os.environ.get("GTFULL_SOURCE_SNAPSHOT")
    if pinned:
        pinned=Path(pinned)
        ledger=load(pinned)
        for row in ledger["files"]:
            path=Path(row["path"])
            assert sha(path)==row["sha256"] and path.stat().st_size==row["bytes"]
        save(root/"evidence/source_snapshot.json",ledger)
        save(root/"evidence/source_attachment.json",dict(at_utc=now(),source=str(pinned.resolve()),
             sha256=sha(pinned),files=len(ledger["files"])))
        print(dict(attached_snapshot=str(pinned),files=len(ledger["files"])),flush=True)
        return
    _,probe=query(("NYC",f"https://data.cityofnewyork.us/resource/{cfg['nyc_dataset']}.json",
                   {"$select":"data_as_of","$where":"data_as_of >= '2026-11-09T00:00:00'",
                    "$order":"data_as_of DESC","$limit":1}))
    latest=probe["result"]
    stamp=latest[0]["data_as_of"] if latest else None
    records=[]
    if stamp:
        specs=[]
        for name,params in split_specs("2026-11-09",cfg["corpus"][1],cfg):
            params={**params,"$where":f"({params['$where']}) AND data_as_of <= '{stamp}'"}
            specs.append((name,params))
        with ThreadPoolExecutor(max_workers=2) as pool:
            fetched=list(pool.map(lambda spec:run_shard(root/"raw_extension",spec,OBSERVATIONS,cfg),specs))
        for entry in fetched:
            if entry["status"]=="complete":
                records.append(dict(path=str((root/"raw_extension"/entry["file"]).resolve()),origin="live_extension",**entry))
    else:
        fetched=[]
    cached=load(root/"evidence/sources_cached.json")["files"]
    joined=[*cached,*records]
    failed=[x for x in fetched if x["status"]!="complete"]
    save(root/"evidence/source_snapshot.json",dict(status="available_snapshot",pinned_utc=now(),latest_source_time=stamp,
         official_time_probe=probe,corpus_requested=cfg["corpus"],files=joined,
         cached_rows=sum(x["rows"] for x in cached),new_rows=sum(x["rows"] for x in records),
         failed_extension_shards=failed,dates_or_coverage_do_not_block_experiment=True,
         unknown_retained_on_missing_dates=True,label_or_score_performance_used_for_admission=False))
    print(dict(cached_shards=len(cached),new_shards=len(records),new_rows=sum(x["rows"] for x in records),
               latest=stamp,failed=len(failed)),flush=True)


if __name__=="__main__":
    fetch(*settings())

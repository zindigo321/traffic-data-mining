"""Transport-only acquisition of the registered V6 holdout."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

from v6_io import V5,load,now,save,settings,sha
from v6_register import verify_registration
sys.path.insert(0,str(V5))
from v5_download import OBSERVATIONS,run_shard,split_specs


def transport(root,cfg):
    reg=verify_registration(root,cfg)
    if not cfg["formal_eligible"]:
        print("Blocked by date gate",flush=True)
        return
    if (root/"evidence/unlock.json").exists():
        raise PermissionError("Held-out snapshot already unlocked")
    specs=list(split_specs(*cfg["holdout"],cfg))
    cached=os.environ.get("GT6_HELD_SOURCE")
    if cached:
        origin=Path(cached)
        signed=load(origin/"evidence/download_holdout_sealed.json")
        if (signed["interval"]!=cfg["holdout"] or signed["status"]!="complete"
                or signed["observation_values_parsed"] is not False
                or [(r["file"],r["query"]) for r in signed["files"]]!=specs):
            raise ValueError("Incorrect pinned external snapshot")
        directory=origin/"raw_external/holdout_sealed"
        files=signed["files"]
        for row in files:
            path=directory/row["file"]
            if path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]:
                raise ValueError("Pinned V6 transport bytes differ")
    else:
        directory=root/"raw_external/holdout_sealed"
        directory.mkdir(parents=True,exist_ok=True)
        with ThreadPoolExecutor(max_workers=cfg["source"]["workers"]) as pool:
            files=list(pool.map(lambda spec:run_shard(directory,spec,OBSERVATIONS,cfg),specs))
    complete=[r for r in files if r["status"]=="complete"]
    row=dict(status="complete" if len(files)==len(complete) else "blocked",name="holdout_sealed",
             dataset=cfg["nyc_dataset"],interval=cfg["holdout"],directory=str(directory.resolve()),files=files,
             attempted=len(files),complete=len(complete),rows=sum(f["rows"] for f in complete),
             transport_only=True,observation_values_parsed=False,preregistered_utc=reg["at_utc"],checked_utc=now())
    if cached:
        row["attached_from"]=str(origin.resolve())
    save(root/"evidence/download_holdout_sealed.json",row)
    print({k:v for k,v in row.items() if k!="files"},flush=True)
    if row["status"]!="complete":
        raise ValueError("A registered new shard could not be acquired")


if __name__=="__main__":
    transport(*settings())

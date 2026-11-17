"""Fetch and pin immutable NYC transport shards; do not parse sealed values."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import os
from pathlib import Path
import time

import requests

from v5_io import CONFIG, load, now, save, settings, sha

GEOMETRY = ("link_id", "link_name", "borough", "encoded_poly_line", "link_points")
OBSERVATIONS = ("data_as_of", "link_id", "speed", "travel_time", "status")


def date_range(start, end):
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    return [start + timedelta(days=i) for i in range((end-start).days)]


def verify_registration(root, cfg):
    receipt = load(root/"evidence/registration.json")
    if receipt["observation_values_read"] is not False or receipt["holdout"] != cfg["holdout"]:
        raise ValueError("Calendar not preregistered")
    files = [root/"plan.md", root/"protocol_nyc.md", CONFIG]
    if {f.name:sha(f) for f in files} != receipt["documents"]:
        raise ValueError("Preregistration changed")
    return receipt


def split_specs(start, stop, cfg):
    all_days = date_range(start,stop)
    days_per = cfg["source"]["week_days"]
    for left in range(0,len(all_days),days_per):
        dates = all_days[left:left+days_per]
        end = dates[-1] + timedelta(days=1)
        for label, hours in (("am",(7,9)),("pm",(17,19))):
            active = " OR ".join(
                f"(data_as_of >= '{d}T{hours[0]:02}:00:00' AND data_as_of < '{d}T{hours[1]:02}:00:00')" for d in dates)
            where = f"data_as_of >= '{dates[0]}T00:00:00' AND data_as_of < '{end}T00:00:00' AND ({active})"
            yield f"{dates[0]}_{end}_{label}.csv", dict(
                **{"$select":",".join(OBSERVATIONS),"$where":where,"$order":"data_as_of,link_id,:id","$limit":50000})


def run_shard(directory, spec, header, cfg):
    name, params = spec
    target = directory/name
    receipt = target.with_suffix(".json")
    if target.is_file() and receipt.is_file():
        entry = load(receipt)
        if sha(target) != entry["sha256"] or entry["query"] != params:
            raise ValueError(f"Changed cached shard {name}")
        return entry
    endpoint=f"https://data.cityofnewyork.us/resource/{cfg['nyc_dataset']}.csv"
    started, attempts = now(), []
    for attempt in range(1,cfg["source"]["http_retries"]+1):
        try:
            response = requests.get(endpoint,params=params,timeout=(8,55))
            attempts.append(dict(attempt=attempt, at_utc=now(),status=response.status_code,
                                 elapsed_seconds=round(response.elapsed.total_seconds(),4),url=response.url))
            if not response.ok:
                if response.status_code in {429,500,502,503,504}:
                    time.sleep(attempt*2)
                    continue
                attempts[-1]["error_preview"]=response.text[:300]
                break
            content=response.content
            first, _, tail = content.partition(b"\n")
            actual_header = first.decode("utf-8-sig").rstrip("\r").replace('"',"").split(",")
            if tuple(actual_header) != tuple(header):
                raise ValueError(f"CSV schema mismatch: {actual_header}")
            rows=tail.count(b"\n")+bool(tail and not tail.endswith(b"\n"))
            if rows >= params["$limit"]:
                raise ValueError("SODA limit reached; refusing truncated source")
            tmp = target.with_suffix(".part")
            tmp.write_bytes(content)
            tmp.replace(target)
            entry=dict(status="complete",file=name,query=params,rows=rows,bytes=len(content),sha256=sha(target),
                       first_requested_utc=started,completed_utc=now(),attempts=attempts)
            save(receipt,entry)
            return entry
        except (requests.RequestException, ValueError) as error:
            attempts.append(dict(attempt=attempt,at_utc=now(),error=str(error)[:350]))
            time.sleep(attempt*2)
    return dict(status="failed",file=name,query=params,attempts=attempts)


def download(root,cfg):
    lock=verify_registration(root,cfg)
    raw=root/"raw_external"
    geometry=dict(**{"$select":",".join(GEOMETRY),"$where":
        f"data_as_of >= '{cfg['geometry_window'][0]}' AND data_as_of < '{cfg['geometry_window'][1]}'",
        "$order":"link_id,data_as_of,:id","$limit":50000})
    dev=[cfg["train_start"],cfg["calibration"][1]]
    packages=[
        ("geometry", [("roads.csv",geometry)], GEOMETRY, cfg["geometry_window"]),
        ("development",list(split_specs(*dev,cfg)),OBSERVATIONS,dev),
        ("holdout_sealed",list(split_specs(*cfg["holdout"],cfg)),OBSERVATIONS,cfg["holdout"]),
    ]
    if os.environ.get("GT5_RAW"):
        attach(root,cfg,Path(os.environ["GT5_RAW"]),packages)
        return
    commands=[]
    for name,specs,header,interval in packages:
        directory=raw/name
        directory.mkdir(parents=True,exist_ok=True)
        commands.extend((name,directory,spec,header) for spec in specs)
    with ThreadPoolExecutor(max_workers=cfg["source"]["workers"]) as pool:
        responses=list(pool.map(lambda item:(item[0],run_shard(*item[1:],cfg)),commands))
    failed=[]
    for name,specs,_,interval in packages:
        rows=[receipt for group,receipt in responses if group==name]
        complete=[row for row in rows if row["status"]=="complete"]
        payload=dict(status="complete" if len(complete)==len(rows) else "blocked", name=name,
                     interval=interval,directory=str((raw/name).resolve()),files=rows,attempted=len(rows),
                     complete=len(complete),rows=sum(r["rows"] for r in complete),preregistered_utc=lock["at_utc"],
                     transport_only=True,observation_values_parsed=False,dataset=cfg["nyc_dataset"],reported_utc=now())
        save(root/"evidence"/f"download_{name}.json",payload)
        failed.extend(r["file"] for r in rows if r["status"]!="complete")
        print({k:v for k,v in payload.items() if k!="files"},flush=True)
    if failed:
        raise RuntimeError(f"Missing registered shards: {failed}")


def attach(root,cfg,prior_run,packages):
    for name,specs,_,interval in packages:
        receipt=load(prior_run/"evidence"/f"download_{name}.json")
        if receipt["status"]!="complete" or receipt["interval"]!=interval or receipt["dataset"]!=cfg["nyc_dataset"]:
            raise ValueError("Snapshot scope mismatch")
        rows=receipt["files"]
        if [(r["file"],r["query"]) for r in rows]!=specs:
            raise ValueError("Snapshot query mismatch")
        directory=prior_run/"raw_external"/name
        if not directory.is_dir():
            directory=Path(receipt["directory"])
        for row in rows:
            p=directory/row["file"]
            if p.stat().st_size!=row["bytes"] or sha(p)!=row["sha256"]:
                raise ValueError(f"Snapshot bytes differ: {p}")
        save(root/"evidence"/f"download_{name}.json",{**receipt,"directory":str(directory.resolve()),
             "attached_cache":str(prior_run.resolve()),"attached_utc":now()})
        print(dict(attached=name,files=len(rows),rows=receipt["rows"]),flush=True)


def source_files(root,name):
    if name=="holdout_sealed" and not (root/"evidence/unlock.json").is_file():
        raise PermissionError("Held-out speed parsing requires verified unlock")
    if name not in ("geometry","development","holdout_sealed"):
        raise ValueError("Unexpected data role")
    receipt=load(root/"evidence"/f"download_{name}.json")
    if receipt["status"]!="complete":
        raise ValueError(f"Incomplete source {name}")
    for row in receipt["files"]:
        path=Path(receipt["directory"])/row["file"]
        if sha(path)!=row["sha256"]:
            raise ValueError(f"Source bytes differ: {path}")
        yield path,row


if __name__=="__main__":
    download(*settings())

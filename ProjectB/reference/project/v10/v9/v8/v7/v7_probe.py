"""Time-only availability query for the preregistered future interval."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import time

import requests

from v7_io import now,save,settings
from v7_check import registered


def request(spec):
    name,url,params=spec
    attempts=[]
    for _ in range(2):
        begin,t0=now(),time.perf_counter()
        try:
            r=requests.get(url,params=params,timeout=(8,22))
            attempts.append(dict(utc=begin,status=r.status_code,url=r.url,seconds=round(time.perf_counter()-t0,4)))
            if r.ok:
                return name,dict(result=r.json(),attempts=attempts,params=params)
            if r.status_code not in (429,500,502,503,504):
                break
        except (ValueError,requests.RequestException) as e:
            attempts.append(dict(utc=begin,error=type(e).__name__))
    return name,dict(result=None,attempts=attempts,params=params)


def probe(root,cfg):
    registered(root)
    nyc=f"https://data.cityofnewyork.us/resource/{cfg['nyc_dataset']}.json"
    chicago="https://data.cityofchicago.org/resource/4g9f-3jbs.json"
    targets=[
        ("nyc_latest",nyc,{"$select":"data_as_of","$where":f"data_as_of >= '{cfg['holdout'][0]}T00:00:00'",
                           "$order":"data_as_of DESC","$limit":1}),
        ("chicago_latest",chicago,{"$select":"time","$order":"time DESC","$limit":1}),
        ("chicago_after_stop",chicago,{"$select":"time","$where":"time >= '2026-05-01T00:00:00' AND segment_id between 1 and 240","$limit":1}),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        raw=dict(pool.map(request,targets))
    latest=raw["nyc_latest"]["result"]
    if latest and (len(latest)!=1 or set(latest[0])!={"data_as_of"}):
        raise ValueError("NYC probe must be time only")
    local=datetime.now(ZoneInfo(cfg["zone"]))
    mature=local.date()>=datetime.fromisoformat(cfg["holdout"][1]).date()
    covered=bool(latest and datetime.fromisoformat(latest[0]["data_as_of"])>=datetime.fromisoformat(cfg["holdout"][1]))
    status="needs_coverage_gate" if mature and covered else "blocked"
    reason=("Registered 28-day interval has not ended in NYC." if not mature else
            "Latest verified NYC timestamp does not reach registered stop." if not covered else
            "Date bound reached; verify per-peak road coverage before any held-out speed read.")
    evidence=dict(at_utc=now(),NYC_now=local.isoformat(),NYC_status=status,NYC_reason=reason,holdout=cfg["holdout"],
        embargo=cfg["embargo"],latest=latest[0]["data_as_of"] if latest else None,dates_mature=mature,
        time_extent_reaches_stop=covered,observation_values_requested=False,queries=raw,Chicago_status="blocked",
        Chicago_new_time=raw["chicago_after_stop"]["result"],Geolife_status="blocked")
    save(root/"evidence/date_gate.json",evidence)
    print({k:v for k,v in evidence.items() if k!="queries"},flush=True)


if __name__=="__main__":
    probe(*settings())

"""Time-only recheck without downloading partial registered observations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import time

import requests

from v8_io import now,save,settings
from v8_check import register


def query(spec):
    name,endpoint,args=spec
    attempts=[]
    for _ in range(2):
        started,t0=now(),time.perf_counter()
        try:
            reply=requests.get(endpoint,params=args,timeout=(8,20))
            attempts.append(dict(utc=started,status=reply.status_code,url=reply.url,seconds=round(time.perf_counter()-t0,4)))
            if reply.ok:
                return name,dict(result=reply.json(),attempts=attempts)
            if reply.status_code not in {429,500,502,503,504}:
                break
        except (ValueError,requests.RequestException) as e:
            attempts.append(dict(utc=started,error=type(e).__name__))
    return name,dict(result=None,attempts=attempts)


def probe(root,cfg):
    register(root,cfg)
    specs=[
        ("NYC",f"https://data.cityofnewyork.us/resource/{cfg['nyc_dataset']}.json",
         {"$select":"data_as_of","$where":f"data_as_of >= '{cfg['holdout'][0]}T00:00:00'","$order":"data_as_of DESC","$limit":1}),
        ("Chicago","https://data.cityofchicago.org/resource/4g9f-3jbs.json",
         {"$select":"time","$where":"time >= '2026-05-01T00:00:00' AND segment_id between 1 and 240","$limit":1})
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=dict(pool.map(query,specs))
    latest=result["NYC"]["result"]
    if latest and (len(latest)!=1 or set(latest[0])!={"data_as_of"}):
        raise ValueError("Only time fields are allowed")
    local=datetime.now(ZoneInfo(cfg["zone"]))
    stop=datetime.fromisoformat(cfg["holdout"][1])
    mature=local.date()>=stop.date()
    reached=bool(latest and datetime.fromisoformat(latest[0]["data_as_of"])>=stop)
    reason="预登记纽约区间尚未结束" if not mature else "官方时间未覆盖登记终点" if not reached else "须先复核双高峰逐日覆盖"
    state=dict(at_utc=now(),NYC_now=local.isoformat(),NYC_status="blocked",NYC_reason=reason,
               latest=latest[0]["data_as_of"] if latest else None,holdout=cfg["holdout"],
               dates_mature=mature,time_extent_reaches_stop=reached,Chicago_status="blocked",Geolife_status="blocked",
               observation_values_requested=False,queries=result)
    save(root/"evidence/date_gate.json",state)
    print({k:v for k,v in state.items() if k!="queries"},flush=True)


if __name__=="__main__":
    probe(*settings())

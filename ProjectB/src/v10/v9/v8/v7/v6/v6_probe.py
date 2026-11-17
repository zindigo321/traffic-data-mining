"""Bounded NYC coverage probe without accessing speeds or labels."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import time

import requests

from v6_io import load,now,save,sha

NY="https://data.cityofnewyork.us/resource/i4gi-tjb9.json"
CHI="https://data.cityofchicago.org/resource/4g9f-3jbs.json"
ANCHOR="2026-10-12"
UPPER="2026-11-09"
MIN_DAYS=14
MIN_ROWS=1000
MIN_ROADS=70
RELEASE="G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/outputs/ProjectB_GraphTransformer_V5"
V5="ProjectB_GraphTransformer_V5_审计包"
ROAD_SHA="f4ca63c60fbff1bcbef7d4807250e904011e649e4ad45c801e658bd78661cc9d"


def query(name,url,params):
    attempts=[]
    for attempt in range(2):
        started,t0=now(),time.perf_counter()
        try:
            reply=requests.get(url,params=params,timeout=(8,22))
            attempts.append(dict(utc=started,status=reply.status_code,seconds=round(time.perf_counter()-t0,4),url=reply.url))
            if reply.ok:
                data=reply.json()
                return dict(name=name,params=params,attempts=attempts,result=data)
            if reply.status_code not in {429,500,502,503,504}:
                attempts[-1]["error"]=reply.text[:280]
                break
        except (requests.RequestException,ValueError) as ex:
            attempts.append(dict(utc=started,error=type(ex).__name__,seconds=round(time.perf_counter()-t0,4)))
    return dict(name=name,params=params,attempts=attempts,result=None)


def daily_request(days,label,roster,group):
    start,stop=(7,9) if label=="am" else (17,19)
    where=" OR ".join(f"(data_as_of >= '{d}T{start:02d}:00:00' AND data_as_of < '{d}T{stop:02d}:00:00')" for d in days)
    args={"$select":"count(*) as rows,count(distinct link_id) as links","$where":f"link_id in ({roster}) AND ({where})"}
    if group:
        args["$select"]="date_trunc_ymd(data_as_of) as local_day,"+args["$select"]
        args["$group"]="date_trunc_ymd(data_as_of)"
    return (f"{days[0]}_{days[-1]}_{label}",NY,args)


def run(root):
    plan=load(root/"evidence/plan_receipt.json")
    if sha(root/"plan.md")!=plan["sha256"] or plan["candidate_observation_values_read"] is not False:
        raise ValueError("V6 plan missing or changed")
    metadata=[
        ("ny_latest",NY,{"$select":"data_as_of","$where":f"data_as_of >= '{ANCHOR}T00:00:00'","$order":"data_as_of DESC","$limit":1}),
        ("chicago_latest",CHI,{"$select":"time","$order":"time DESC","$limit":1}),
        ("chicago_post",CHI,{"$select":"time","$where":"time >= '2026-05-01T00:00:00' AND segment_id between 1 and 240","$limit":1}),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        initial={q["name"]:q for q in pool.map(lambda x:query(*x),metadata)}
    result=initial["ny_latest"]["result"]
    start=date.fromisoformat(ANCHOR)
    end=start
    latest=None
    if result:
        if len(result)!=1 or set(result[0])!={"data_as_of"}:
            raise ValueError("Latest probe requested observations instead of metadata")
        latest=datetime.fromisoformat(result[0]["data_as_of"])
        end=min(date.fromisoformat(UPPER),latest.date()+timedelta(days=int(latest.hour>=19)))
    days=[start+timedelta(days=i) for i in range(max(0,(end-start).days))]
    path=Path(os.environ.get("GT6_V5_RELEASE",RELEASE))/V5/"evidence/nyc_roads.csv"
    if sha(path)!=ROAD_SHA:
        raise ValueError("NYC roster identity differs")
    with path.open(encoding="utf-8-sig",newline="") as f:
        links=[r["link_id"] for r in csv.DictReader(f)]
    if len(links)!=121 or not all(k.isdigit() for k in links):
        raise ValueError("Invalid pinned roster")
    roster=",".join(f"'{i}'" for i in links)
    counts={str(day):{"am":dict(rows=0,links=0),"pm":dict(rows=0,links=0)} for day in days}
    requests_log=[]
    complete=True
    with ThreadPoolExecutor(max_workers=6) as pool:
        grouped=[(days[i:i+7],part) for i in range(0,len(days),7) for part in ("am","pm")]
        for (period,part),response in zip(grouped,pool.map(lambda x:query(*daily_request(*x,roster,True)),grouped),strict=True):
            requests_log.append(response)
            if response["result"] is None:
                for day,r in zip(period,pool.map(lambda d:query(*daily_request([d],part,roster,False)),period),strict=True):
                    requests_log.append(r)
                    if r["result"] is None:
                        complete=False
                        continue
                    row=r["result"][0] if r["result"] else {}
                    counts[str(day)][part]={k:int(row.get(k,0)) for k in ("rows","links")}
                continue
            for row in response["result"]:
                day=row["local_day"][:10]
                if day not in counts:
                    raise ValueError("API date coverage outside candidate")
                counts[day][part]={k:int(row[k]) for k in ("rows","links")}
    sufficient=[day for day,parts in counts.items()
                if all(p["rows"]>=MIN_ROWS and p["links"]>=MIN_ROADS for p in parts.values())]
    status="eligible" if complete and len(sufficient)>=MIN_DAYS else "blocked"
    payload=dict(at_utc=now(),status=status,reason=None if status=="eligible" else "Not enough date-only peak coverage",
        candidate=[ANCHOR,UPPER],candidate_stop_exclusive=str(end),ny_latest=latest.isoformat() if latest else None,
        interval=[ANCHOR,str(end)] if status=="eligible" else None,calendar_days=len(days),substantial_days=len(sufficient),
        sufficient=sufficient,daily_counts=counts,roster_sha256=ROAD_SHA,requests=requests_log,initial=initial,
        observation_values_read=False,preregistration_plan_sha256=plan["sha256"])
    save(root/"evidence/date_gate.json",payload)
    print({k:v for k,v in payload.items() if k not in ("requests","initial","daily_counts")},flush=True)
    chi=initial["chicago_post"]["result"]
    save(root/"evidence/chicago_status.json",dict(status="blocked",latest=initial["chicago_latest"]["result"],
         post_2026_04_30_query=chi,dataset="4g9f-3jbs",AP=None,F1=None,goal_reached=None,
         prior_same_protocol_blocked=True,independent_NYC_report_separate=True,at_utc=now()))


if __name__=="__main__":
    run(Path(os.environ["GT6_ROOT"]))

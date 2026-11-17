"""Feasibility queries confined to schema, time and counts."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import requests

from v5_io import now, save

NYC = "https://data.cityofnewyork.us"
CHICAGO = "https://data.cityofchicago.org"
NY_ID = "i4gi-tjb9"
CHI_ID = "4g9f-3jbs"
NY_ENDPOINT = f"{NYC}/resource/{NY_ID}.json"


def query(spec):
    name, url, args = spec
    attempts = []
    for _ in range(2):
        start, tic = now(), time.perf_counter()
        try:
            response = requests.get(url, params=args, timeout=(8, 21))
            attempts.append(dict(utc=start, seconds=round(time.perf_counter()-tic, 4), status=response.status_code, url=response.url))
            if response.ok:
                if name == "ny_meta":
                    raw = response.json()
                    result = dict(id=raw["id"], name=raw["name"], updated=raw["rowsUpdatedAt"],
                                  metadata=raw.get("metadata"), fields=[(c["fieldName"],c["dataTypeName"]) for c in raw["columns"]])
                elif name == "stop":
                    result = dict(stop_confirmed="4/30/2026" in response.text, daily_replacement="Average Daily Traffic Counts" in response.text)
                else:
                    result = response.json()
                return name, dict(query=args, attempts=attempts, result=result)
        except requests.RequestException as e:
            attempts.append(dict(utc=start, seconds=round(time.perf_counter()-tic,4), error=type(e).__name__))
    return name, dict(query=args, attempts=attempts, result=None)


def probe(root):
    dates = ("2026-05-12", "2026-06-23", "2026-08-04", "2026-09-01", "2026-10-13")
    specs = [
        ("chi_latest", f"{CHICAGO}/resource/{CHI_ID}.json", {"$select":"time","$order":"time DESC","$limit":1}),
        ("chi_post_v3", f"{CHICAGO}/resource/{CHI_ID}.json", {"$select":"time","$where":"time >= '2026-05-01T00:00:00' AND segment_id between 1 and 240","$limit":1}),
        ("stop", f"{CHICAGO}/stories/s/c28d-stgw", None),
        ("ny_meta", f"{NYC}/api/views/{NY_ID}.json", None),
        ("ny_cadence_training", NY_ENDPOINT, {"$select":"link_id,data_as_of,status", "$where":
            "data_as_of >= '2026-05-12T07:00:00' AND data_as_of < '2026-05-12T07:12:00'",
            "$order":"data_as_of,link_id","$limit":5000}),
    ]
    for day in dates:
        specs.append((f"ny_count_{day}",NY_ENDPOINT,{"$select":"count(*) as n,count(distinct link_id) as roads",
            "$where":f"data_as_of >= '{day}T07:00:00' AND data_as_of < '{day}T09:00:00'"}))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(pool.map(query,specs))
    import collections
    cadence = results["ny_cadence_training"]
    rows = cadence.get("result") or []
    cadence["result"] = dict(row_count=len(rows), links=len({x.get("link_id") for x in rows}),
                            statuses=dict(collections.Counter(x.get("status","") for x in rows)),
                            timestamps=[(str(k),v) for k,v in collections.Counter(x.get("data_as_of") for x in rows).most_common(30)])
    save(root / "evidence/feasibility.json", dict(at_utc=now(), speed_values_requested=False, results=results,
         ny_publisher="https://www.nyc.gov/html/dot/html/about/datafeeds.shtml",
         ny_terms="https://www.nyc.gov/html/datamine/html/data/terms.html?dataSetJs=raw",
         ny_faq="https://www.nyc.gov/opendata/get-started/FAQs"))
    print({k:v["result"] if k!="ny_meta" else {x:v["result"].get(x) for x in ("name","updated","fields","metadata")} if v["result"] else None for k,v in results.items()},flush=True)


if __name__=="__main__":
    probe(Path(os.environ["GT5_ROOT"]))

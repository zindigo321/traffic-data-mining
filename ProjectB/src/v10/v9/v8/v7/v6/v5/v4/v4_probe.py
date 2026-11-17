"""Read public availability evidence without requesting any observation values."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import requests

from v4_io import now, save, settings

BASE = "https://data.cityofchicago.org"
DATASET = "4g9f-3jbs"
DEPRECATION = BASE + "/stories/s/c28d-stgw"
REPLACEMENT = BASE + "/d/pfsx-4n4m"
STOP = "2026-04-30"
FIELDS = ("name", "id", "rowsUpdatedAt", "publicationDate")


def query(name, endpoint, params=None):
    attempts = []
    for _ in range(2):
        started, begin = now(), time.perf_counter()
        try:
            response = requests.get(endpoint, params=params, timeout=(8, 22))
            status = response.status_code
            attempts.append(dict(utc=started, seconds=round(time.perf_counter() - begin, 3),
                                 status_code=status, url=response.url))
            if status == 200:
                return name, response, attempts
            if status not in {429, 500, 502, 503, 504}:
                break
        except requests.RequestException as error:
            attempts.append(dict(utc=started, seconds=round(time.perf_counter() - begin, 3),
                                 error=type(error).__name__, detail=str(error)))
    return name, None, attempts


def main():
    root, cfg = settings()
    endpoint = f"{BASE}/resource/{DATASET}.json"
    # Time only, including a same-roster interval where the V3 protocol starts.
    query_specs = [
        ("latest", endpoint, {"$select": "time", "$order": "time DESC", "$limit": 1}),
        ("post_v3", endpoint, {"$select": "time", "$where":
             f"time >= '{cfg['earliest_new_date']}T00:00:00' AND segment_id between 1 and 240",
             "$order": "time", "$limit": 1}),
        ("metadata", f"{BASE}/api/views/{DATASET}.json", None),
        ("publisher", DEPRECATION, None),
        ("catalog", "https://api.us.socrata.com/api/catalog/v1",
         {"search_context": "data.cityofchicago.org", "q": "Traffic Tracker", "limit": 40}),
    ]
    result = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for name, response, attempts in executor.map(lambda args: query(*args), query_specs):
            entry = dict(attempts=attempts, params=next(q[2] for q in query_specs if q[0] == name))
            if response is not None:
                if name == "publisher":
                    body = response.text
                    entry.update(stop_date_mentioned=any(s in body for s in ("4/30/2026", "April 30, 2026")),
                                 replacement_mentioned="Average Daily Traffic Counts" in body)
                elif name == "metadata":
                    data = response.json()
                    entry.update(metadata={k: data.get(k) for k in FIELDS},
                                 fields=[c["fieldName"] for c in data.get("columns", ())])
                elif name == "catalog":
                    entry["results"] = [{k: row["resource"].get(k) for k in ("id", "name", "type")}
                                        for row in response.json().get("results", [])]
                else:
                    data = response.json()
                    if not isinstance(data, list) or any(set(row) != {"time"} for row in data):
                        raise ValueError("Availability query returned a non-time field")
                    entry["rows"] = data
            result[name] = entry
    if result["post_v3"].get("rows"):
        status, reason = "needs_review", "Post-V3 timestamp found. Validate its coverage and lineage and preregister a separate new holdout before formal evaluation."
    else:
        status = "blocked"
        reason = ("Publisher ended comparable directed 10-minute bus-GPS estimates on 2026-04-30. "
                  "V3 used the last forward interval. No verified compatible successor was found.")
    payload = dict(status=status, reason=reason, dataset=DATASET, publisher=DEPRECATION, stop_date=STOP,
                   independent_from=cfg["earliest_new_date"], retrieved_utc=now(), queries=result,
                   successor_review=[
                       dict(source="Traffic Tracker - Congestion Estimates by Segments", id="n4j6-wkkf",
                            decision="Deprecated; same series and last date"),
                       dict(source="Traffic Tracker - Historical Congestion Estimates by Region", id="kf7e-cur8",
                            decision="Region-level, same end date, lacks same directed segment support"),
                       dict(source="Average Daily Traffic Counts", decision="Daily volumes; lacks 10-minute directed segment speeds"),
                       dict(source="Pre-2024 archived Traffic Tracker", decision="Dates precede the development and calibration cutoff")],
                   requested_speeds_or_labels=False, new_holdout=None, V3_holdout_history_only=True,
                   formal_AP=None, formal_F1=None, goal=cfg["goal"], target_reached=None)
    save(root / "evidence/holdout_gate.json", payload)
    print({key: value for key, value in payload.items() if key != "queries"}, flush=True)


if __name__ == "__main__":
    main()

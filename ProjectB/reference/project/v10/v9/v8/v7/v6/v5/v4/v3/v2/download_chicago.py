"""Bounded official Chicago queries. The holdout transport does not parse observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, timedelta
from pathlib import Path
import sys
import time
import traceback

import requests

from common import digest, json_read, json_write, now, settings

BASE = "https://data.cityofchicago.org"
COLUMNS = ("time,segment_id,speed,street,direction,from_street,to_street,length,bus_count,"
           "message_count,start_latitude,start_longitude,end_latitude,end_longitude")


def metadata(root, cfg):
    url = f"{BASE}/api/views/{cfg['source']['dataset']}.json"
    with requests.get(url, timeout=(15, 35)) as response:
        response.raise_for_status()
        obj = response.json()
    result = dict(retrieved_utc=now(), url=url, **{k: obj.get(k) for k in (
        "id", "name", "viewLastModified", "rowsUpdatedAt", "attribution", "licenseId", "license")})
    result["terms_url"] = "https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html"
    result["official_deprecation"] = f"{BASE}/stories/s/c28d-stgw"
    if result["id"] != cfg["source"]["dataset"] or not result["license"]:
        raise ValueError("Unexpected publisher metadata")
    json_write(root / "evidence/chicago_publisher.json", result)
    return result


def probe(root, cfg):
    try:
        info = metadata(root, cfg)
    except (requests.RequestException, ValueError):
        traceback.print_exc()
        print("Publisher metadata endpoint failed; pinned v1 attribution and terms remain available.", flush=True)
        info = None
    trials = []
    scopes = {"development": cfg["development_download"], "holdout_transport": cfg["holdout"]}
    url = f"{BASE}/resource/{cfg['source']['dataset']}.csv"
    for scope, (start, stop) in scopes.items():
        start = date.fromisoformat(start)
        params = {"$select": "time,segment_id", "$where": (
            f"time >= '{start}T07:00:00' AND time < '{start + timedelta(days=2)}T09:00:00' "
            "AND segment_id between 1 and 40 AND hour in (7,8)"), "$limit": 6}
        for attempt in range(1, 3):
            t = time.perf_counter()
            try:
                with requests.get(url, params=params, timeout=(15, 26)) as response:
                    response.raise_for_status()
                    rows = list(csv.reader(response.text.splitlines()))
                    if not rows or rows[0] != ["time", "segment_id"]:
                        raise ValueError("Unexpected probe header")
                    trials.append(dict(scope=scope, params=params, url=response.url, returned_rows=len(rows) - 1,
                                       values=rows[1:], status="PASS", at_utc=now(), seconds=time.perf_counter()-t))
                    break
            except (requests.RequestException, ValueError):
                failure = traceback.format_exc()
                trials.append(dict(scope=scope, params=params, status="failed", attempt=attempt,
                                   error=failure, at_utc=now(), seconds=time.perf_counter()-t))
                print(failure, flush=True)
    result = dict(status="probe_finished", no_speed_or_label_read=True, scopes=scopes, version=info, attempts=trials)
    json_write(root / "evidence/probe.json", result)
    print(json_read(root / "evidence/probe.json")["status"], flush=True)


def block(folder, start, stop, cfg):
    src = cfg["source"]
    tag = f"{start}_{stop}"
    path = folder / f"{tag}.csv"
    receipt = folder / f"{tag}.json"
    if path.is_file() and receipt.is_file():
        old = json_read(receipt)
        if old.get("status") == "complete" and digest(path) == old["sha256"]:
            return old
        raise ValueError(f"Existing unverified shard {tag}")
    params = {"$select": COLUMNS,
              "$where": (f"time >= '{start}T00:00:00' AND time < '{stop}T00:00:00' "
                         f"AND segment_id between 1 and {src['max_segment_id']} "
                         f"AND hour in ({','.join(map(str, src['hours']))})"),
              "$limit": 85000, "$order": "time,segment_id,:id"}
    url = f"{BASE}/resource/{src['dataset']}.csv"
    attempts = []
    for i in range(1, src["http_retries"] + 1):
        tmp = path.with_suffix(".part")
        t = time.perf_counter()
        started = now()
        try:
            with requests.get(url, params=params, stream=True, timeout=(15, 55)) as response:
                response.raise_for_status()
                bytes_out = 0
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(1 << 18):
                        bytes_out += len(chunk)
                        if bytes_out > 22_000_000:
                            raise ValueError("Shard exceeds byte limit")
                        out.write(chunk)
                final_url = response.url
                version = {k: response.headers.get(k) for k in ("Last-Modified", "ETag")}
            with tmp.open("rb") as f:
                if f.readline().strip().decode("ascii").replace('"', "") != COLUMNS:
                    raise ValueError("Unexpected CSV header")
                rows = sum(1 for _ in f)
            if rows >= params["$limit"]:
                raise ValueError("Shard hit limit")
            tmp.replace(path)
            result = dict(status="complete", start=str(start), stop_exclusive=str(stop), file=path.name,
                          rows=rows, bytes=bytes_out, sha256=digest(path), query=params, url=final_url,
                          response_version=version, retrieved_utc=started, attempt=i,
                          elapsed_seconds=time.perf_counter()-t, earlier_attempts=attempts)
            json_write(receipt, result)
            print(json_read(receipt), flush=True)
            return result
        except (requests.RequestException, ValueError, OSError):
            attempts.append(dict(attempt=i, utc=started, elapsed_seconds=time.perf_counter()-t,
                                 error=traceback.format_exc()))
            print(tag, attempts[-1], flush=True)
            tmp.unlink(missing_ok=True)
            if i < src["http_retries"]:
                time.sleep(2*i)
    result = dict(status="failed", start=str(start), stop_exclusive=str(stop), query=params, attempts=attempts)
    json_write(receipt, result)
    return result


def fetch_scope(root, cfg, name):
    if not (root / "evidence/probe.json").is_file():
        raise ValueError("Run the metadata and small-query probe first")
    key = "development_download" if name == "development" else "holdout"
    start, stop = map(date.fromisoformat, cfg[key])
    folder = root / "raw_external" / name
    folder.mkdir(parents=True, exist_ok=True)
    windows = [(d, min(stop, d + timedelta(days=cfg["source"]["week_days"])))
               for d in (start + timedelta(days=i) for i in range(0, (stop-start).days, cfg["source"]["week_days"]))]
    with ThreadPoolExecutor(max_workers=cfg["source"]["workers"]) as pool:
        results = list(pool.map(lambda pair: block(folder, *pair, cfg), windows))
    passed = [r for r in results if r["status"] == "complete"]
    status = "complete" if len(passed) == len(results) else "blocked"
    value = dict(status=status, name=name, calendar_interval=cfg[key], row_count=sum(r["rows"] for r in passed),
                 directory=str(folder), complete=len(passed), attempted=len(results), shards=results,
                 observation_values_parsed=False, zone=cfg["source"]["zone"], dataset=cfg["source"]["dataset"])
    json_write(root / "evidence" / f"download_{name}.json", value)
    print({k: v for k, v in value.items() if k != "shards"}, flush=True)
    return 0 if status == "complete" else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["probe", "development", "holdout_sealed"])
    stage = p.parse_args().stage
    root, cfg = settings()
    if stage == "probe":
        probe(root, cfg)
        return 0
    return fetch_scope(root, cfg, stage)


if __name__ == "__main__":
    sys.exit(main())


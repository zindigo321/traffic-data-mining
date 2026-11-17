"""Fetch fixed historical Chicago segment estimates from the City's public API."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import requests

from settings import CONFIG, WORK_ROOT

COLUMNS = (
    "time,segment_id,speed,street,direction,from_street,to_street,length,bus_count,"
    "message_count,start_latitude,start_longitude,end_latitude,end_longitude"
)
URL_ROOT = "https://data.cityofchicago.org"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fetch_block(block):
    base, dataset_id, start, end, max_id, hours, attempts = block
    url = f"{URL_ROOT}/resource/{dataset_id}.csv"
    params = {
        "$select": COLUMNS,
        "$where": (
            f"time >= '{start}T00:00:00' AND time < '{end}T00:00:00' "
            f"AND segment_id between 1 and {max_id} AND hour in ({','.join(map(str, hours))})"
        ),
        "$limit": 85000,
        "$order": "time,segment_id,:id",
    }
    path = base / f"{start}_{end}.csv"
    temp = path.with_suffix(".part")
    errors = []
    for attempt in range(1, attempts + 1):
        begun = time.perf_counter()
        retrieved = datetime.now(timezone.utc).isoformat()
        try:
            with requests.get(url, params=params, stream=True, timeout=(20, 95)) as response:
                response.raise_for_status()
                final_url = response.url
                size = 0
                with temp.open("wb") as f:
                    for chunk in response.iter_content(1 << 18):
                        size += len(chunk)
                        if size > 24_000_000:
                            raise ValueError("Response larger than the requested bounded block")
                        f.write(chunk)
            with temp.open("rb") as f:
                if f.readline().strip().decode("ascii").replace('"', "") != COLUMNS:
                    raise ValueError("Unexpected CSV header")
                rows = sum(1 for _ in f)
            if rows >= params["$limit"]:
                raise ValueError("Limit reached; decrease slice_days to avoid truncation")
            temp.replace(path)
            result = dict(status="complete", query=params, url=final_url, start=str(start),
                          end_exclusive=str(end), rows=rows, bytes=size, sha256=digest(path),
                          path=path.name, retrieved_utc=retrieved, attempt=attempt,
                          seconds=time.perf_counter() - begun, earlier_errors=errors)
            print(json.dumps(result), flush=True)
            return result
        except (requests.RequestException, OSError, ValueError) as exc:
            detail = dict(attempt=attempt, error=str(exc), seconds=time.perf_counter() - begun,
                          at_utc=retrieved, traceback=traceback.format_exc())
            errors.append(detail)
            print(json.dumps(dict(block=str(start), **detail)), flush=True)
            temp.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 * attempt)
    return dict(status="blocked", query=params, url=url, start=str(start), errors=errors)


def main():
    cfg = CONFIG["public"]
    base = WORK_ROOT / "public_data"
    base.mkdir(parents=True, exist_ok=False)
    start, end = [date.fromisoformat(cfg[key]) for key in ("start_date", "end_date_exclusive")]
    days = cfg["slice_days"]
    if not (0 < days <= 7 and 1 <= cfg["segment_max_id"] <= 350 and start < end):
        raise ValueError("Use 1–7-day query slices and at most 350 segments")
    meta_url = f"{URL_ROOT}/api/views/{cfg['dataset_id']}.json"
    try:
        r = requests.get(meta_url, timeout=(15, 30))
        r.raise_for_status()
        meta = r.json()
        write_json(base / "metadata.json", {
            **{k: meta.get(k) for k in ("id", "name", "description", "rowsUpdatedAt", "viewLastModified", "attribution", "attributionLink", "licenseId", "license")},
            "metadata_url": meta_url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "license_page": "https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html",
            "fields": [{k: f.get(k) for k in ("fieldName", "name", "description", "dataTypeName")}
                       for f in meta.get("columns", []) if f.get("fieldName") in COLUMNS.split(",")],
        })
    except (requests.RequestException, ValueError):
        print("Metadata fetch failed; see traceback and independent publisher references.", flush=True)
        traceback.print_exc()
    blocks = []
    for offset in range(0, (end - start).days, days):
        first = start + timedelta(days=offset)
        blocks.append((base, cfg["dataset_id"], first, min(first + timedelta(days=days), end),
                       cfg["segment_max_id"], cfg["peak_hours"], 3))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch_block, blocks))
    record = dict(dataset_id=cfg["dataset_id"], source=f"{URL_ROOT}/d/{cfg['dataset_id']}", blocks=results)
    success = all(x["status"] == "complete" for x in results)
    record["status"] = "complete" if success else "blocked"
    if success:
        path = base / "chicago_fixed.csv"
        with path.open("wb") as out:
            out.write((COLUMNS + "\n").encode("ascii"))
            for item in results:
                with (base / item["path"]).open("rb") as f:
                    f.readline()
                    shutil.copyfileobj(f, out)
        record.update(rows=sum(x["rows"] for x in results), bytes=path.stat().st_size, sha256=digest(path),
                      file=path.name)
    write_json(base / "download.json", record)
    print({k: v for k, v in record.items() if k != "blocks"}, flush=True)
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())


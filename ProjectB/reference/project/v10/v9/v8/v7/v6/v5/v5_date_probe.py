from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from v5_io import now,save
from v5_probe import NY_ENDPOINT,query

root=Path(os.environ["GT5_ROOT"])
days=("2026-06-09","2026-07-14","2026-07-28","2026-08-18","2026-08-25","2026-09-08","2026-09-15","2026-09-22","2026-09-27","2026-10-06","2026-10-20")
specs=[(d,NY_ENDPOINT,{"$select":"count(*) as n,count(distinct link_id) as roads",
                       "$where":f"data_as_of >= '{d}T07:00:00' AND data_as_of < '{d}T09:00:00'"}) for d in days]
with ThreadPoolExecutor(max_workers=6) as p:
    results=dict(p.map(query,specs))
save(root/"evidence/coverage_only.json",dict(results=results,speed_values_requested=False,at_utc=now()))
print({k:v["result"] for k,v in results.items()},flush=True)

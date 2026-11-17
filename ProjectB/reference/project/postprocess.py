import time
from full_io import load,record,python,settings

root,cfg=settings()
start=time.monotonic()
while True:
    records=[load(p) for p in (root/"logs").glob("*completion_pipeline.json")]
    if records:
        if records[-1]["exit_code"]!=0:
            raise RuntimeError("Completion pipeline failed. Read preserved logs.")
        break
    if time.monotonic()-start>7200:
        raise TimeoutError("Completion pipeline did not return")
    time.sleep(5)
record(root,"verify_full_replay",python("full_verify.py","--replay-root",r"G:/gpt-file/Codex/2026-11-10/project-b-graph-transformer-graph-transformer/work/full_data/qa_20261112/rebuilt"))
record(root,"render_report",python("full_report.py"))

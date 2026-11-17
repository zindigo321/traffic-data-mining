import time
from full_io import load,record,python,settings

root,cfg=settings()
started=time.monotonic()
while not (root/"results/development_training.json").exists():
    errors=[load(p) for p in (root/"logs").glob("*development_training.json")]
    if any(p["exit_code"]!=0 for p in errors):
        raise RuntimeError("Development training failed. Read preserved logs before retrying.")
    if time.monotonic()-started>7200:
        raise TimeoutError("Development receipt did not arrive")
    time.sleep(5)
for stage,args in (
    ("development_selection_retry",("full_score.py","development")),
    ("final_training",("full_train.py","matrix","--stage","final")),
    ("threshold_freeze",("full_score.py","freeze")),
    ("evaluate",("full_score.py","evaluate")),
    ("analysis",("full_analysis.py",))):
    record(root,stage,python(*args))

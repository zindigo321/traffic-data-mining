from pathlib import Path
from v4_io import load, settings, save, sha, now
from v4_data import V3_NAME
from v4_train import read_npz
import numpy as np

root, cfg = settings()
old = Path(cfg["v3_release"]) / V3_NAME / "results/dev"
compare = []
for fold in cfg["folds"]:
    for seed in cfg["seeds"]:
        dirs = [old / fold / f"R2_mean_s{seed}", root / f"results/dev/{fold}/B_mean_s{seed}"]
        a, b = [read_npz(x/"assessment.npz") for x in dirs]
        inputs = [read_npz(x/"sampling_ids.npz") for x in dirs]
        for field in ("sample_id", "y", "risk", "risk_missing"):
            np.testing.assert_array_equal(a[field], b[field])
        for field in inputs[0]:
            np.testing.assert_array_equal(inputs[0][field], inputs[1][field])
        old_m, new_m = [load(x/"status.json") for x in dirs]
        row = dict(fold=fold, seed=seed, max_logit_diff=float(np.max(np.abs(a["logit"]-b["logit"]))),
                   max_probability_diff=float(np.max(np.abs(a["probability"]-b["probability"]))),
                   max_score_diff=float(np.max(np.abs(a["score"]-b["score"]))),
                   decision_mismatches=int((a["pred"]!=b["pred"]).sum()), delta_AP=new_m["AP"]-old_m["AP"],
                   delta_F1=new_m["f1"]-old_m["f1"],
                   delta_threshold=float(b["threshold"]-a["threshold"]),
                   matching_epoch=old_m["best_epoch"]==new_m["best_epoch"], same_weight=sha(dirs[0]/"weight.pt")==sha(dirs[1]/"weight.pt"))
        compare.append(row)
        print(row, flush=True)
save(root/"evidence/v3_numeric_regression.json", dict(at_utc=now(), rows=compare))

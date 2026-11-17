"""V0-only regression over the externally pinned v1 artifact."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CODE, json_read, json_write, settings
sys.path.insert(0, str(CODE / "v1"))
from experiment import StudyData, graph_model, metrics, read_npz, select_threshold
from graph import TrainPreprocessor
from verify_results import metrics_from_csv, scores, check_equal


def main():
    root, cfg = settings()

    entry = json_read(root / "evidence/v1_inputs.json")
    source = Path(entry["extracted"]) / "results/chicago"
    cache = Path(entry["cache"])
    legacy = json_read(source / "config_frozen.json")
    torch.set_num_threads(legacy["torch_threads"])
    choices = json_read(source / "selection_before_test.json")
    old_rows = json_read(source / "metrics.json")
    if choices["selected_pooling"] != cfg["reference_pool"] or len(old_rows) != 18:
        raise ValueError("Unexpected v1 model matrix")
    if not json_read(source / "status.json")["formal_training_executed"]:
        raise ValueError("Expected completed v1 Chicago matrix")
    out = root / "results/v0_regression"
    out.mkdir(parents=True, exist_ok=False)
    data = StudyData(cache, "chicago")
    data.configure(TrainPreprocessor.load(source / "preprocessor.npz"))
    arr = data.data
    y = arr["y"]
    aux_src = read_npz(cache / "prepared/chicago_audit_local.npz")
    hist, speed = aux_src["history"], aux_src["speed"]
    slow = (hist > 0) & np.isfinite(speed) & (speed < .4 * hist)
    same_step = (np.diff(arr["window_minute"], prepend=-100) == 10) & (
        np.diff(arr["split"], prepend=-1) == 0)
    prev_slow = np.vstack([np.zeros(y.shape[1], dtype=bool), slow[:-1]]) & same_step[:, None]
    positive = (y == 1) & arr["eligible"]
    previous_positive = np.vstack([np.zeros(y.shape[1], dtype=bool), positive[:-1]]) & same_step[:, None]
    event_start = positive & ~previous_positive
    old_idx = read_npz(source / "train_index.npz")["sample_id"]
    hard = prev_slow & (y == 0) & arr["eligible"]
    training = arr["split"] == 0
    stats = dict(
        events={k: int(event_start[arr["split"] == i].sum()) for i, k in enumerate(("train","validation","test"))},
        high_risk_recovered_train=int(hard[training].sum()),
        high_risk_recovered_sampled=int(hard.ravel()[old_idx].sum()),
        aux_low_eligible_train=int((slow & arr["eligible"])[training].sum()),
        total_eligible_train_negative=int(((y==0) & arr["eligible"])[training].sum()),
    )
    ids = data.parts["test"]
    rng = np.random.default_rng(9071)
    batch_size = legacy["eval_batch_size"]
    batches = np.sort(rng.choice(len(ids)//batch_size, 5, replace=False))
    positions = np.concatenate([np.arange(b*batch_size, (b+1)*batch_size) for b in batches])
    torch_types = {"GraphTransformer", "MLP"}
    results = []
    for row in old_rows:
        stem = source / f"seed_{row['seed']}"
        name = row["name"]
        label_results = dict(model=row["model"], pooling=row["pooling"], seed=row["seed"], name=name)
        test_p = None
        for part in ("validation", "test"):
            frame = pd.read_csv(stem / f"{name}_{part}_predictions.csv.gz", float_precision="round_trip")
            expected = pd.DataFrame(data.keys(data.parts[part]))
            pd.testing.assert_frame_equal(frame[list(expected.columns)], expected, check_dtype=False)
            label = frame["label"].to_numpy(dtype=np.int8)
            p = scores(frame, row)
            if row["model"] in torch_types:
                p = p.astype(np.float32)
            cutoff = row["threshold"]
            if part == "validation":
                chosen = select_threshold(label, p) if row["model"] != "Majority" else .5
                check_equal(chosen, cutoff, "v1 validation threshold")
                check_equal(float(average_precision_score(label, p)), row["validation_ap"], "v1 validation AP")
            else:
                test_p = p
                calculated = metrics_from_csv(label, p.astype(np.float64), cutoff)
                for k, value in calculated.items():
                    check_equal(value, row.get(k), k)
                np.testing.assert_array_equal(p >= cutoff, frame["predicted"].to_numpy(dtype=bool))
                calculated["AP"] = calculated.pop("pr_auc")
                calculated["ROC_AUC"] = float(roc_auc_score(label, p))
                ystar = select_threshold(label, p)
                calculated["historical_optimistic_F1"] = metrics(label, p, ystar)["f1"]
                label_results.update(**calculated, threshold=cutoff, probability_dtype=str(p.dtype))
            if row["model"] == "GraphTransformer":
                np.savez_compressed(out / f"{name}_s{row['seed']}_{part}.npz",
                                    v1_sample_id=frame["sample_id"].to_numpy(dtype=np.int64),
                                    label=label, probability=p, threshold=np.float64(cutoff))
        if row["model"] == "GraphTransformer":
            model = graph_model(data, legacy, row["pooling"])
            model.load_state_dict(torch.load(stem / row["weight"], map_location="cpu", weights_only=True))
            model.eval()
            with torch.inference_mode():
                logits = torch.cat([model(*[torch.from_numpy(a) for a in data.batch(ids[pos])])
                    for pos in np.array_split(positions,5)])
                p = logits.sigmoid().numpy()
            np.testing.assert_allclose(p, test_p[positions], atol=2e-7, rtol=1e-6)
            np.testing.assert_array_equal(p >= cutoff, test_p[positions] >= cutoff)
            delta = float(np.max(np.abs(p - test_p[positions])))
            np.savez_compressed(out / f"{name}_s{row['seed']}_replay.npz",
                                v1_sample_id=ids[positions], logit=logits.numpy(), probability=p)
            label_results.update(replayed=len(p), max_abs_error=delta, cutoff_decisions_identical=True)
        results.append(label_results)
        print(name, row["seed"], label_results["AP"], label_results["f1"], flush=True)
    result = pd.DataFrame(results)
    result.to_csv(out / "metrics.csv", index=False, na_rep="N/A")
    result.groupby(["model","pooling"])[["AP","ROC_AUC","f1","fp"]].agg(["mean","std"]).to_csv(out/"aggregate.csv")
    json_write(out / "status.json", dict(status="PASS", datasets="V1 Chicago historical regression only",
                                        old_test_used_for_v2_selection=False, checked_runs=len(results),
                                        checked_predictions=2*len(results), **stats))


if __name__ == "__main__":
    main()



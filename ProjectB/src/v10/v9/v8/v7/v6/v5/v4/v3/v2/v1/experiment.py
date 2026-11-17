"""Train, select on validation, then evaluate directed-road identification models."""
import argparse
from itertools import combinations, product
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from graph import TrainPreprocessor, check_temporal, load_graphs
from model import GraphTransformer
from settings import CONFIG, WORK_ROOT

SPLITS = ("train", "validation", "test")


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return dict(n=0, status="N/A", na_reasons={"all": "No known labels"})
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected known labels and finite probabilities in [0, 1]")
    positive = y == 1
    pred = p >= threshold
    tn, fp, fn, tp = [int(mask.sum()) for mask in
                      (~positive & ~pred, ~positive & pred, positive & ~pred, positive & pred)]
    na = {}

    def divide(a, b, name):
        if b == 0:
            na[name] = "zero denominator"
            return None
        return a / b

    ap = None
    if tp + fn and tn + fp:
        ap = float(average_precision_score(y, p))
    else:
        na["pr_auc"] = "only one true class"
    return dict(n=len(y), negative=tn+fp, positive=tp+fn, accuracy=(tn+tp)/len(y),
                precision=divide(tp, tp+fp, "precision"), recall=divide(tp, tp+fn, "recall"),
                f1=divide(2*tp, 2*tp+fn+fp, "f1"), pr_auc=ap, tn=tn, fp=fp, fn=fn, tp=tp,
                threshold=float(threshold), na_reasons=na)


def select_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    candidates = thresholds[best]
    return float(candidates[np.lexsort((candidates, np.abs(candidates - .5)))[0]])


class MLP(nn.Module):
    """Two hidden layers matching the original Project B baseline sizes."""
    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(.15),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class StudyData:
    def __init__(self, root, source):
        self.source = source
        if source == "chicago":
            self.data = read_npz(root / "prepared" / "chicago.npz")
            self.y = self.data["y"].ravel()
            self.n_roads = self.data["y"].shape[1]
            self.split = np.repeat(self.data["split"], self.n_roads)
            self.eligible = self.data["eligible"].ravel()
        else:
            self.data, self.meta = load_graphs(root / "prepared" / "geolife")
            check_temporal(self.meta)
            self.y = self.data["y"]
            self.split = np.asarray([SPLITS.index(row["split"]) for row in self.meta])
            self.eligible = np.ones(len(self.y), dtype=bool)
        self.features = self.data["x"].shape[-1] * 2
        self.parts = {name: np.flatnonzero(self.eligible & (self.y >= 0) & (self.split == i))
                      for i, name in enumerate(SPLITS)}

    def keys(self, ids):
        if self.source == "chicago":
            ti, ri = np.divmod(ids, self.n_roads)
            t = self.data["window_minute"][ti].astype("datetime64[m]") + np.timedelta64(10, "m")
            return dict(sample_id=ids, decision_time=np.datetime_as_string(t, unit="m"),
                        target_road=self.data["road_ids"][ri], label=self.y[ids])
        return dict(sample_id=ids, decision_time=[self.meta[i]["time"] for i in ids],
                    target_road=[self.meta[i]["target"] for i in ids], label=self.y[ids])

    def fit_preprocessor(self):
        if self.source == "chicago":
            x = self.data["x"][self.data["split"] == 0]
            return TrainPreprocessor().fit(x, np.ones(x.shape[:-1], dtype=bool))
        ids = self.parts["train"]
        return TrainPreprocessor().fit(self.data["x"][ids], self.data["valid"][ids], [self.meta[i] for i in ids])

    def configure(self, preprocessor):
        d = self.data
        valid = np.ones(d["x"].shape[:-1], dtype=bool) if self.source == "chicago" else d["valid"]
        self.x = preprocessor.transform(d["x"], valid)

    def batch(self, ids):
        d = self.data
        if self.source == "chicago":
            t, r = np.divmod(ids, self.n_roads)
            valid = d["valid"][r]
            x = self.x[t[:, None], d["lookup"][r]]
            x *= valid[..., None]
            center = np.zeros(len(ids), dtype=np.int64)
        else:
            r = ids
            valid = d["valid"][r]
            x = self.x[r]
            center = d["center"][r]
        return x, d["adj"][r], valid, center

    def flat(self, ids):
        result = []
        for start in range(0, len(ids), 1024):
            x, _, valid, center = self.batch(ids[start:start+1024])
            mean = (x * valid[..., None]).sum(1) / valid.sum(1)[:, None]
            maximum = np.where(valid[..., None], x, -np.inf).max(1)
            result.append(np.concatenate([x[np.arange(len(x)), center], mean, maximum], axis=-1))
        return np.concatenate(result).astype(np.float32)


def training_ids(data, budget, seed):
    full = data.parts["train"]
    neg = full[data.y[full] == 0]
    pos = full[data.y[full] == 1]
    chosen = np.random.default_rng(seed).choice(neg, size=min(len(neg), budget), replace=False)
    return np.sort(np.concatenate([pos, chosen]))


def initialize(data, out, config):
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for i, name in enumerate(SPLITS):
        k = data.parts[name]
        counts[name] = dict(negative=int((data.y[k] == 0).sum()), positive=int((data.y[k] == 1).sum()),
                            unknown_graphs=int((data.eligible & (data.split == i) & (data.y == -1)).sum()))
    dump(out / "split_counts.json", counts)
    # The test class balance is reported but never gates fitting or selects a configuration.
    missing = [name for name in SPLITS[:2] if not (counts[name]["negative"] and counts[name]["positive"])]
    dump(out / "config_frozen.json", config)
    if missing:
        reason = "Training or validation has no two-class supervision: " + ", ".join(missing)
        rows = [dict(model="GraphTransformer", pooling=pool, seed=seed)
                for pool, seed in product(config["pooling"], config["seeds"])]
        rows += [dict(model=name, pooling="N/A", seed=seed)
                 for name, seed in product(("MLP", "RandomForest", "Majority"), config["seeds"])]
        for row in rows:
            row.update(status="NOT_RUN", reason=reason, accuracy=None, precision=None, recall=None, f1=None, pr_auc=None)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, na_rep="N/A")
        dump(out / "metrics.json", rows)
        dump(out / "status.json", dict(data_source=data.source, status="blocked", reason=reason, formal_training_executed=False))
        print(reason, flush=True)
        return
    ids = training_ids(data, config["train_negative_budget"], config["train_sampling_seed"])
    np.savez_compressed(out / "train_index.npz", sample_id=ids)
    data.fit_preprocessor().save(out / "preprocessor.npz")
    dump(out / "train_sample_counts.json", dict(negative=int((data.y[ids] == 0).sum()), positive=int((data.y[ids] == 1).sum()),
         negative_budget=config["train_negative_budget"], sampling_seed=config["train_sampling_seed"]))
    dump(out / "status.json", dict(data_source=data.source, status="ready", formal_training_executed=False))


def graph_model(data, config, pool):
    return GraphTransformer(data.features, config["width"], config["heads"], config["layers"], pool)


def torch_predict(model, make_input, ids, batch):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for offset in range(0, len(ids), batch):
            ix = ids[offset:offset + batch]
            probabilities.append(torch.sigmoid(model(*make_input(ix))).numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=float)


def fit_torch(model, make_input, labels, train, val, seed, config, lr, batch):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((labels[train] == 0).sum() / (labels[train] == 1).sum())))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(labels, dtype=torch.float32)
    start = time.perf_counter()
    process = psutil.Process()
    peak = process.memory_info().rss
    best_ap, best_state, best_epoch, curve = -1.0, None, 0, []
    for epoch in range(1, config["epochs"] + 1):
        total = 0.
        order = rng.permutation(train)
        model.train()
        for offset in range(0, len(order), batch):
            ids = order[offset:offset+batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(*make_input(ids)), y[ids])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        model.eval()
        probabilities = []
        validation_loss = 0.
        with torch.inference_mode():
            for offset in range(0, len(val), config["eval_batch_size"]):
                ids = val[offset:offset+config["eval_batch_size"]]
                z = model(*make_input(ids))
                validation_loss += float(loss_fn(z, y[ids])) * len(ids)
                probabilities.append(torch.sigmoid(z).numpy())
        p = np.concatenate(probabilities)
        ap = float(average_precision_score(labels[val], p))
        curve.append(dict(epoch=epoch, training_weighted_bce=total/len(train),
                          validation_weighted_bce=validation_loss/len(val), validation_ap=ap))
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        peak = max(peak, process.memory_info().rss)
    model.load_state_dict(best_state)
    cost = dict(fit_seconds=time.perf_counter()-start, peak_rss_sampled_bytes=peak,
                parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch, validation_ap=best_ap)
    return curve, cost


def export_forest(path, model):
    trees = [m.tree_ for m in model.estimators_]
    bounds = np.concatenate([[0], np.cumsum([len(t.feature) for t in trees])])
    arrays = {name: np.concatenate([getattr(t, name) for t in trees])
              for name in ("children_left", "children_right", "feature", "threshold", "value")}
    np.savez_compressed(path, bounds=bounds, **arrays)


def predict_forest(path, x):
    if not len(x):
        return np.empty(0, dtype=float)
    a = read_npz(path)
    x = x.astype(np.float64)
    rows = np.arange(len(x))
    result = np.zeros(len(x))
    for begin, end in zip(a["bounds"][:-1], a["bounds"][1:]):
        left, right, feature, threshold, value = [a[k][begin:end] for k in
            ("children_left", "children_right", "feature", "threshold", "value")]
        cursor = np.zeros(len(x), dtype=np.int64)
        while True:
            parent = left[cursor] != right[cursor]
            if not parent.any():
                break
            node = cursor[parent]
            current = x[rows[parent], feature[node]]
            cursor[parent] = np.where(current <= threshold[node], left[node], right[node])
        distribution = value[cursor, 0]
        result += distribution[:, 1] / distribution.sum(1)
    return result / (len(a["bounds"]) - 1)


def fit_seed(data, out, config, seed):
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if status["status"] == "blocked":
        return
    run = out / f"seed_{seed}"
    run.mkdir(exist_ok=False)
    data.configure(TrainPreprocessor.load(out / "preprocessor.npz"))
    train = read_npz(out / "train_index.npz")["sample_id"]
    val = data.parts["validation"]

    def graph_batch(ids):
        return tuple(torch.from_numpy(a) for a in data.batch(ids))

    flat = data.flat(np.concatenate([train, val]))
    flat_positions = np.full(len(data.y), -1, dtype=np.int64)
    flat_positions[np.concatenate([train, val])] = np.arange(len(flat))

    def flat_batch(ids):
        positions = flat_positions[ids]
        if (positions < 0).any():
            raise ValueError("Fit touched a sample outside training and validation")
        return (torch.from_numpy(flat[positions]),)

    choices = []

    def save_choice(name, model, pool, p, cost, weight, fixed_threshold=None):
        threshold = select_threshold(data.y[val], p) if fixed_threshold is None else fixed_threshold
        row = dict(name=name, model=model, pooling=pool, seed=seed, weight=weight, **cost)
        row.update(validation_f1=metrics(data.y[val], p, threshold)["f1"], threshold=threshold)
        choices.append(row)
        pd.DataFrame(dict(**data.keys(val), probability=p, threshold=threshold)).to_csv(
            run / f"{name}_validation_predictions.csv.gz", index=False)
        dump(run / f"{name}_selection.json", row)
        print(json.dumps(row), flush=True)

    for pool in config["pooling"]:
        name = f"gt_{pool}"
        set_seed(seed)
        net = graph_model(data, config, pool)
        curve, cost = fit_torch(net, graph_batch, data.y, train, val, seed, config,
                               config["learning_rate"], config["batch_size"])
        weight = f"{name}.pt"
        torch.save(net.state_dict(), run / weight)
        pd.DataFrame(curve).to_csv(run / f"{name}_curve.csv", index=False)
        p = torch_predict(net, graph_batch, val, config["eval_batch_size"])
        save_choice(name, "GraphTransformer", pool, p, cost, weight)
    mlp_grid, selected = [], None
    for lr, batch in product((.001, .0001), (32, 64)):
        set_seed(seed)
        net = MLP(flat.shape[1])
        curve, cost = fit_torch(net, flat_batch, data.y, train, val, seed, config, lr, batch)
        choice = dict(learning_rate=lr, batch_size=batch, **cost)
        mlp_grid.append(choice)
        pd.DataFrame(curve).to_csv(run / f"mlp_lr{lr}_b{batch}_curve.csv", index=False)
        key = (cost["validation_ap"], -lr, -batch)
        if selected is None or key > selected[0]:
            selected = key, net, choice
    _, net, cost = selected
    pd.DataFrame(mlp_grid).to_csv(run / "mlp_validation_grid.csv", index=False)
    torch.save(net.state_dict(), run / "mlp.pt")
    p = torch_predict(net, flat_batch, val, config["eval_batch_size"])
    save_choice("mlp", "MLP", "N/A", p, cost, "mlp.pt")

    start = time.perf_counter()
    process = psutil.Process()
    memory = process.memory_info().rss
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                random_state=seed, n_jobs=2).fit(flat[flat_positions[train]], data.y[train])
    seconds = time.perf_counter() - start
    p = rf.predict_proba(flat[flat_positions[val]])[:, 1]
    export_forest(run / "rf.npz", rf)
    np.testing.assert_allclose(predict_forest(run / "rf.npz", flat[flat_positions[val]]), p, rtol=1e-12, atol=1e-12)
    save_choice("rf", "RandomForest", "N/A", p,
                dict(fit_seconds=seconds, peak_rss_sampled_bytes=max(memory, process.memory_info().rss),
                     parameters=None, validation_ap=metrics(data.y[val], p)["pr_auc"]), "rf.npz")

    majority = int(np.argmax(np.bincount(data.y[train], minlength=2)))
    dump(run / "majority.json", dict(predicted_class=majority))
    p = np.full(len(val), float(majority))
    save_choice("majority", "Majority", "N/A", p, dict(fit_seconds=0., peak_rss_sampled_bytes=process.memory_info().rss,
                parameters=0, validation_ap=metrics(data.y[val], p)["pr_auc"]), "majority.json", .5)
    dump(run / "choices.json", choices)


def freeze(out, config):
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if status["status"] == "blocked":
        return
    choices = [row for seed in config["seeds"] for row in
               json.loads((out / f"seed_{seed}" / "choices.json").read_text(encoding="utf-8"))]
    ap = {pool: float(np.mean([row["validation_ap"] for row in choices if row["pooling"] == pool]))
          for pool in config["pooling"]}
    selected = max(config["pooling"], key=ap.get)
    dump(out / "selection_before_test.json", dict(
        selected_pooling=selected, validation_ap_means=ap, choices=choices,
        checkpoint="Highest validation average precision; earliest epoch wins ties",
        threshold="Highest validation F1 over unique validation scores; tie nearest .5 then lower, majority .5",
        pooling_choice="Mean validation AP across paired seeds; fixed configured order breaks ties",
        test_used_for_selection=False,
    ))


def test_probabilities(data, out, config, row, ids, flat):
    run = out / f"seed_{row['seed']}"
    weight = run / row["weight"]
    if row["model"] == "GraphTransformer":
        net = graph_model(data, config, row["pooling"])
        net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        return torch_predict(net, lambda j: tuple(torch.from_numpy(a) for a in data.batch(j)), ids, config["eval_batch_size"])
    if row["model"] == "MLP":
        net = MLP(flat.shape[1])
        net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        local = np.arange(len(flat))
        return torch_predict(net, lambda j: (torch.from_numpy(flat[j]),), local, config["eval_batch_size"])
    if row["model"] == "RandomForest":
        return predict_forest(weight, flat)
    return np.full(len(ids), float(json.loads(weight.read_text(encoding="utf-8"))["predicted_class"]))


def evaluate(data, out, config):
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if status["status"] == "blocked":
        return
    set_seed(config["seeds"][0])
    selection = json.loads((out / "selection_before_test.json").read_text(encoding="utf-8"))
    data.configure(TrainPreprocessor.load(out / "preprocessor.npz"))
    ids = data.parts["test"]
    y = data.y[ids]
    flat = data.flat(ids) if len(ids) else np.empty((0, 3 * data.features), dtype=np.float32)
    pd.DataFrame(data.keys(ids)).to_csv(out / "test_truth.csv.gz", index=False)
    results = []
    for row in selection["choices"]:
        start = time.perf_counter()
        p = test_probabilities(data, out, config, row, ids, flat)
        elapsed = time.perf_counter() - start
        test = metrics(y, p, row["threshold"])
        result = dict(**row, eval_seconds=elapsed, **{key: val for key, val in test.items() if key != "threshold"})
        results.append(result)
        table = pd.DataFrame(dict(**data.keys(ids), probability=p, threshold=row["threshold"],
                                  predicted=(p >= row["threshold"]).astype(np.int8)))
        table.to_csv(out / f"seed_{row['seed']}" / f"{row['name']}_test_predictions.csv.gz", index=False)
        print(json.dumps(result), flush=True)
    table = pd.DataFrame(results)
    table.to_csv(out / "metrics.csv", index=False, na_rep="N/A")
    dump(out / "metrics.json", results)
    numeric = ["accuracy", "precision", "recall", "f1", "pr_auc", "fit_seconds", "eval_seconds", "peak_rss_sampled_bytes"]
    table.groupby(["model", "pooling"])[numeric].agg(["mean", "std", "count"]).to_csv(out / "aggregate.csv", na_rep="N/A")
    paired = []
    for a, b in combinations(config["pooling"], 2):
        for seed in config["seeds"]:
            left = next(r for r in results if r["pooling"] == a and r["seed"] == seed)
            right = next(r for r in results if r["pooling"] == b and r["seed"] == seed)
            diff = {f"{key}_diff": (left[key] - right[key] if left.get(key) is not None and right.get(key) is not None else None)
                    for key in ("f1", "pr_auc", "fit_seconds")}
            paired.append(dict(pool_a=a, pool_b=b, seed=seed, **diff))
    paired_table = pd.DataFrame(paired)
    paired_table.to_csv(out / "paired_differences.csv", index=False, na_rep="N/A")
    paired_table.groupby(["pool_a", "pool_b"])[["f1_diff", "pr_auc_diff", "fit_seconds_diff"]].agg(
        ["mean", "std", "count"]).to_csv(out / "paired_summary.csv", na_rep="N/A")
    dump(out / "status.json", dict(data_source=data.source, status="complete", formal_training_executed=True,
                                   test_used_for_selection=False, n_test=len(ids), n_models=len(results)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["init", "fit", "freeze", "evaluate"])
    cli.add_argument("--dataset", required=True, choices=["geolife", "chicago"])
    cli.add_argument("--seed", type=int)
    args = cli.parse_args()
    if CONFIG["device"] != "cpu":
        raise ValueError("This deterministic benchmark is CPU-only")
    out = WORK_ROOT / "results" / args.dataset
    if args.stage == "freeze":
        freeze(out, CONFIG)
        return
    data = StudyData(WORK_ROOT, args.dataset)
    if args.stage == "init":
        initialize(data, out, CONFIG)
    elif args.stage == "fit":
        if args.seed not in CONFIG["seeds"]:
            raise ValueError("Provide a configured seed")
        fit_seed(data, out, CONFIG, args.seed)
    else:
        evaluate(data, out, CONFIG)


if __name__ == "__main__":
    main()


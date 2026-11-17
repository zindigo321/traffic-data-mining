"""Paired center-model sampling, group ranking, and balanced-group loss ablations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import math
import statistics
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from v10_check import original, register
from v10_data import (balanced_group_weights, build, data_for, groups, predict, read_npz, set_seed,
                      study_for, training_sample)
from v10_io import load, now, python, record, save, settings, sha


def folder(root, fold, recipe, seed):
    return root / "results/models" / fold / f"{recipe}_s{seed}"


def low_group_rank(logit, y, group, tail):
    positives = logit[(y == 1) & (group == 1)]
    negatives = logit[(y == 0) & (group == 1)]
    if not len(positives) or not len(negatives):
        return logit.sum() * 0
    k = max(1, math.ceil(tail * len(negatives)))
    hard = negatives.topk(k).values
    return F.softplus(hard[None, :] - positives[:, None]).mean()


def fit(root, cfg, fold, recipe, seed, fixed):
    register(root, cfg)
    if (root / "evidence/freeze.json").exists():
        raise PermissionError("Model and threshold freeze already exists")
    if recipe not in cfg["v10_recipes"] or seed not in cfg["seeds"]:
        raise ValueError("Unregistered training configuration")
    if fold == "final":
        picked = load(root / "results/development_selection.json")
        assert fixed == picked["epochs"][recipe]
    path = folder(root, fold, recipe, seed)
    path.mkdir(parents=True, exist_ok=False)
    data = data_for(root, cfg, fold)
    study = study_for(data, "X1")
    group = groups(study.ratio, cfg["ratio"])
    train, sampling = training_sample(study, cfg, recipe)
    reference = original(root, fold, seed)
    published = read_npz(reference / "sampling_ids.npz")
    if recipe == "B0":
        np.testing.assert_array_equal(published["sample_id"], study.keys(train))
        np.testing.assert_array_equal(published["y"], study.y[train])
    np.savez_compressed(path / "sampling_ids.npz", sample_id=study.keys(train), y=study.y[train], legal_group=group[train])
    weights, counts = balanced_group_weights(group[train], study.y[train])
    save(path / "sampling.json", dict(**sampling, group_class_weights=weights.tolist(),
         sampled_true_recovery_negatives=int(((study.y[train] == 0) & study.risk[train]).sum())))
    np.savez_compressed(path / "group_weights.npz", counts=counts, weights=weights)
    prep = study.fit_preprocessor()
    prep.save(path / "preprocessor.npz")
    assert sha(path / "preprocessor.npz") == sha(reference / "preprocessor.npz")
    study.configure(prep)
    set_seed(seed, cfg)
    net = build(study.nfeatures * 2, cfg, "attention", seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(sampling["positive_weight"], dtype=torch.float32))
    y = torch.from_numpy(study.y.astype(np.float32))
    g = torch.from_numpy(group)
    w = torch.from_numpy(weights)
    rng = np.random.default_rng(seed)
    started, tick, process = now(), time.perf_counter(), psutil.Process()
    peak = process.memory_info().rss
    best, epoch, state = -1., 0, None
    curve = []
    for e in range(1, cfg["epochs"] + 1):
        net.train()
        order = rng.permutation(train)
        total = extra = macro = 0.
        rank_batches = 0
        for off in range(0, len(order), cfg["batch_size"]):
            ids = order[off:off + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            z = net(*map(torch.from_numpy, study.batch(ids, True)))
            bce = criterion(z, y[ids])
            loss = bce
            ranking = balancing = z.sum() * 0
            if recipe in ("R1", "R2"):
                ranking = low_group_rank(z, y[ids], g[ids], cfg["v10_rank_tail_fraction"])
                loss = loss + cfg["v10_rank_weight"] * ranking
                rank_batches += int(((g[ids] == 1) & (y[ids] == 1)).any() and ((g[ids] == 1) & (y[ids] == 0)).any())
            if recipe == "R2":
                balancing = (F.binary_cross_entropy_with_logits(z, y[ids], reduction="none")
                             * w[g[ids].long(), y[ids].long()]).mean()
                loss = loss + cfg["v10_macro_weight"] * balancing
            loss.backward()
            optimizer.step()
            total += float(bce.detach()) * len(ids)
            extra += float(ranking.detach()) * len(ids)
            macro += float(balancing.detach()) * len(ids)
        ap = None
        if fixed is None:
            _, score = predict(net, study, study.val, cfg, True)
            ap = float(average_precision_score(study.y[study.val], score))
        if e == fixed or (ap is not None and ap > best):
            best = ap if ap is not None else best
            epoch = e
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        peak = max(peak, process.memory_info().rss)
        curve.append(dict(epoch=e, bce=total / len(order), rank=extra / len(order),
                          balanced=macro / len(order), AP=ap, rank_batches=rank_batches,
                          updates=math.ceil(len(order) / cfg["batch_size"])))
    if state is None:
        raise ValueError("No checkpoint selected")
    net.load_state_dict(state)
    torch.save(net.state_dict(), path / "weight.pt")
    pd.DataFrame(curve).to_csv(path / "curve.csv", index=False, na_rep="N/A")
    ids = np.flatnonzero(study.phase == 1)
    logits, probability = predict(net, study, ids, cfg, True)
    scores = probability.astype(np.float64)
    raw = dict(sample_id=study.keys(ids), y=study.y[ids], risk=study.risk[ids], score=scores,
               logit=logits, probability=probability, legal_group=group[ids],
               current_eligible=data["current_eligible"].ravel()[ids])
    old = read_npz(reference / "assessment.npz")
    for key in ("sample_id", "y", "risk", "current_eligible"):
        np.testing.assert_array_equal(raw[key], old[key])
    baseline_error = float(np.abs(scores - old["score"]).max()) if recipe == "B0" else None
    if recipe == "B0":
        np.testing.assert_array_equal(scores, old["score"])
    np.savez_compressed(path / "assessment.npz", **raw)
    known = raw["y"] >= 0
    result = dict(status="complete", fold=fold, recipe=recipe, seed=seed, fixed_epoch=fixed,
        best_epoch=epoch, AP=float(average_precision_score(raw["y"][known], scores[known])),
        fit_and_score_seconds=time.perf_counter() - tick, peak_RSS_bytes=peak,
        parameters=sum(v.numel() for v in net.parameters()), updates=sum(c["updates"] for c in curve),
        train_count=len(train), training_started_utc=started, completed_utc=now(),
        baseline_max_error=baseline_error, weight_sha256=sha(path / "weight.pt"),
        loss_uses_realized_risk=False, graph_scope="center node", lag_minutes=20)
    save(path / "status.json", result)
    print(result, flush=True)


def execute(root, cfg, folds, epochs=None):
    def one(spec):
        fold, recipe, seed = spec
        path = folder(root, fold, recipe, seed)
        if not (path / "status.json").is_file():
            cmd = python("v10_train.py", "fit", "--fold", fold, "--recipe", recipe, "--seed", seed)
            if epochs:
                cmd.extend(["--fixed", epochs[recipe]])
            record(root, f"train_{fold}_{recipe}_s{seed}", cmd)
        state = load(path / "status.json")
        assert state["status"] == "complete"
        return state
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        return list(pool.map(one, product(folds, cfg["v10_recipes"], cfg["seeds"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("fit", "development", "calibration"))
    p.add_argument("--fold")
    p.add_argument("--recipe")
    p.add_argument("--seed", type=int)
    p.add_argument("--fixed", type=int)
    args = p.parse_args()
    root, cfg = settings()
    register(root, cfg)
    if args.stage == "fit":
        fit(root, cfg, args.fold, args.recipe, args.seed, args.fixed)
    elif args.stage == "development":
        rows = execute(root, cfg, cfg["folds"])
        epochs = {r: int(statistics.median(x["best_epoch"] for x in rows if x["recipe"] == r)) for r in cfg["v10_recipes"]}
        save(root / "results/development_training.json", dict(at_utc=now(), epochs=epochs, runs=rows, holdout_read=False))
        print(dict(runs=len(rows), epochs=epochs), flush=True)
    else:
        pick = load(root / "results/development_selection.json")
        rows = execute(root, cfg, ["final"], pick["epochs"])
        save(root / "results/calibration_training.json", dict(at_utc=now(), runs=rows, fixed_epochs=pick["epochs"],
                                                             holdout_read=False))
        print(dict(runs=len(rows), fixed_epochs=pick["epochs"]), flush=True)

"""V4 matched training with optional causal neighbor features and tail loss."""
import argparse
import math
from pathlib import Path
import sys
import time

from v4_io import LEGACY, legacy_config, load, now, save, settings, sha
from v4_data import pinned_array
from v4_metrics import matched_points

sys.path.insert(0, str(LEGACY))
from v3_prepare import local_dates, read_npz
from v3_train import Study, epoch_ids, matched_indices, risk_metrics, train_risk, typed_score
from experiment import predict, set_seed
from evaluation import measure, threshold
from model import build

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score


NEIGHBOR_NAMES = [f"prior10_{side}_{stat}" for side in ("upstream", "downstream")
                  for stat in ("usable", "low_fraction", "min_ratio")]
PAIRED = ("P1", "P3")
NEIGHBOR = ("P2", "P3")
RECIPES = ("B", "P1", "P2", "P3")


def neighbor_features(ratio, lookup, valid, adj, low_ratio):
    if not np.array_equal(lookup[:, 0], np.arange(ratio.shape[1])):
        raise ValueError("Unexpected graph center mapping")
    usable = valid.copy()
    usable[:, 0] = False
    links = [usable & adj[:, :, 0], usable & adj[:, 0, :]]
    features = np.empty((*ratio.shape, len(NEIGHBOR_NAMES)), dtype=np.float32)
    for lo in range(0, len(ratio), 512):
        hi = min(lo + 512, len(ratio))
        candidates = ratio[lo:hi, lookup]
        observed = np.isfinite(candidates)
        for j, linked in enumerate(links):
            support = observed & linked[None, :, :]
            count = support.sum(axis=-1)
            lows = (support & (candidates < low_ratio)).sum(axis=-1)
            fraction = np.divide(lows, count, where=count > 0, out=np.full(count.shape, np.nan, dtype=np.float32))
            minimum = np.where(support, candidates, np.inf).min(axis=-1)
            features[lo:hi, :, j * 3:j * 3 + 3] = np.stack((count, fraction, np.where(count > 0, minimum, np.nan)), axis=-1)
    return features


def check_dates(data, fold, opt, legacy):
    used = data["minute"] // 1440
    train_ranges = legacy["final_train"] if fold == "final" else legacy["folds"][fold]["train"]
    for role, dates in ((0, train_ranges), (1, [opt["fold_limits"][fold]])):
        wanted = local_dates(dates).astype(np.int64)
        np.testing.assert_array_equal(np.unique(used[data["split"] == role]), wanted)
    excluded = local_dates(opt["all_exclusions"]).astype(np.int64)
    if np.isin(used, excluded).any() or (data["split"] == 0).sum() == 0:
        raise ValueError("Invalid embargo or empty training dates")
    if used[data["split"] == 0].max() >= used[data["split"] == 1].min():
        raise ValueError("Evaluation precedes training")
    if len(data["road_ids"]) != legacy["roster_count"]:
        raise ValueError("Unexpected road roster")


def make_study(root, cfg, fold, recipe, receipt=None):
    if recipe not in RECIPES:
        raise ValueError("Unknown recipe")
    data = read_npz(pinned_array(receipt or load(root / "evidence/inputs.json"), fold, cfg["v3_work"]))
    legacy = legacy_config()
    check_dates(data, fold, cfg, legacy)
    study = Study(data, "R2")
    if recipe in NEIGHBOR:
        more = neighbor_features(data["ratio_prior"], data["lookup"], data["valid"], data["adj"], legacy["ratio"])
        data["x"] = np.concatenate([data["x"], more], axis=-1)
        study.nfeatures += more.shape[-1]
    return study


def rank_penalty(logits, y, fraction):
    positive = logits[y == 1]
    negative = logits[y == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0
    hard = torch.topk(negative, math.ceil(fraction * negative.numel()), sorted=False).values
    return F.softplus(hard[:, None] - positive[None, :]).mean()


def fit_paired(study, ids, budget, sampling, opt, cfg, v4):
    # Same model, optimizer, order, risk AP checkpoint and update budget as V3.
    net = build(study.nfeatures * 2, cfg, opt["pool"], opt["seed"])
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(sampling["pos_weight"], dtype=torch.float32))
    y = torch.from_numpy(study.y.astype(np.float32))
    valid = study.val[study.risk[study.val]]
    generator = np.random.default_rng(opt["seed"])
    begin, process = time.perf_counter(), psutil.Process()
    best_ap, best_epoch, state = -1.0, 0, None
    peak, curve = process.memory_info().rss, []
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        ordered = epoch_ids(ids, budget, generator)
        total = tail_total = 0.
        for start in range(0, budget, cfg["batch_size"]):
            batch = ordered[start:start + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = net(*map(torch.from_numpy, study.batch(batch)))
            primary = bce(logits, y[batch])
            penalty = rank_penalty(logits, y[batch], v4["tail_fraction"])
            (primary + v4["rank_weight"] * penalty).backward()
            optimizer.step()
            total += float(primary.detach()) * len(batch)
            tail_total += float(penalty.detach()) * len(batch)
        ap = None
        if opt["fixed_epoch"] is None:
            _, probabilities = predict(net, study, valid, cfg)
            ap = float(average_precision_score(study.y[valid], probabilities))
        if (ap is not None and ap > best_ap) or epoch == opt["fixed_epoch"]:
            best_ap = ap if ap is not None else best_ap
            best_epoch = epoch
            state = {name: weight.detach().clone() for name, weight in net.state_dict().items()}
        peak = max(peak, process.memory_info().rss)
        curve.append(dict(epoch=epoch, conditional_AP=ap, weighted_BCE=total / budget,
                          rank_tail=tail_total / budget, updates=math.ceil(budget / cfg["batch_size"])))
    if state is None:
        raise ValueError("No epoch checkpoint selected")
    net.load_state_dict(state)
    cost = dict(fit_seconds=time.perf_counter() - begin, peak_RSS_bytes=peak, best_epoch=best_epoch,
                updates=sum(row["updates"] for row in curve), parameters=sum(p.numel() for p in net.parameters()))
    return net, curve, cost


def job(root, v4, opt, output):
    output.mkdir(parents=True, exist_ok=False)
    save(output / "options.json", opt)
    cfg = legacy_config()
    fold, recipe = opt["fold"], opt["recipe"]
    study = make_study(root, v4, fold, recipe)
    if any(np.unique(study.y[index]).size != 2 for index in (study.train, study.val)):
        raise ValueError("A known phase lacks a class")
    set_seed(opt["seed"], cfg)
    preprocessor = study.fit_preprocessor()
    preprocessor.save(output / "preprocessor.npz")
    study.configure(preprocessor)
    ids, budget, sampling = matched_indices(study, cfg)
    save(output / "sampling.json", {**sampling, "sampling_seed": cfg["sampling_seed"]})
    np.savez_compressed(output / "sampling_ids.npz", sample_id=study.keys(ids), label=study.y[ids], risk=study.risk[ids])
    if recipe in PAIRED:
        net, curve, cost = fit_paired(study, ids, budget, sampling, opt, cfg, v4)
    else:
        net, curve, cost = train_risk(study, ids, budget, sampling, opt, cfg)
    torch.save(net.state_dict(), output / "weight.pt")
    pd.DataFrame(curve).to_csv(output / "curve.csv", index=False, na_rep="N/A")
    scores = typed_score(study, study.val, *predict(net, study, study.val, cfg))
    y, p, keys, risk = (scores[k] for k in ("y", "score", "sample_id", "risk"))
    cutoff = threshold(y, p)
    coverage = load(root / "evidence" / f"fold_{fold}.json")["parts"][1]
    metrics = measure(y, p, cutoff, keys, coverage)
    by_risk = risk_metrics(scores, cutoff)
    if not math.isclose(metrics["AP"], by_risk["risk_AP"], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("All positives should be high risk and ordered above ordinary windows")
    points, _ = matched_points(keys, y, p, risk, v4["recalls"], coverage["total_days"])
    save(output / "matched_recall.json", points)
    np.savez_compressed(output / "assessment.npz", **scores, threshold=np.float64(cutoff), pred=p >= cutoff)
    state = dict(status="complete", **opt, **cost, **metrics, **by_risk,
                 rank_weight=v4["rank_weight"] if recipe in PAIRED else 0., features=study.nfeatures,
                 logit_dtype=str(scores["logit"].dtype), score_dtype=str(p.dtype),
                 probability_dtype=str(scores["probability"].dtype),
                 weight_sha256=sha(output / "weight.pt"), completed_utc=now())
    save(output / "status.json", state)
    print(state, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, choices=("f1", "f2", "f3", "final"))
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--pool", required=True, choices=("mean", "max", "attention"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fixed-epoch", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, cfg = settings()
    job(root, cfg, {key: getattr(args, key) for key in ("fold", "recipe", "pool", "seed", "fixed_epoch")}, args.output)


if __name__ == "__main__":
    main()

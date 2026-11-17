"""Pinned causal NYC arrays and legal lag20 groups."""
from pathlib import Path
import sys

import numpy as np

from v10_io import V9, load, sha

sys.path.insert(0, str(V9 / "v8/v7"))
from v7_train import build, predict, read_npz, sample_training, set_seed, study_for
from experiment import proportional
from v1.graph import TrainPreprocessor

GROUP_NAMES = ("legal_not_low", "legal_low", "legal_missing")


def groups(ratio, limit=.4):
    group = np.full(ratio.shape, 2, dtype=np.int8)
    finite = np.isfinite(ratio)
    group[finite] = 0
    group[finite & (ratio < limit)] = 1
    return group


def data_for(root, cfg, fold, alternate=None):
    if fold not in (*cfg["folds"], "final"):
        raise PermissionError("Only registered development and calibration arrays may be read")
    if fold == "final" and not (root / "results/development_selection.json").is_file():
        raise PermissionError("Freeze development recipe before opening calibration arrays")
    ref = load(root / "evidence/inputs.json")["prepared"][fold]
    path = alternate / "prepared" / f"lead_{fold}.npz" if alternate else Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError(f"Pinned causal input SHA differs in {fold}")
    data = read_npz(path)
    period = cfg["calibration"] if fold == "final" else cfg["folds"][fold]
    day = data["minute"] // 1440
    np.testing.assert_array_equal(np.unique(day[data["split"] == 1]).astype("datetime64[D]"),
                                  np.arange(*map(np.datetime64, period), dtype="datetime64[D]"))
    assert day.max() < np.datetime64(cfg["embargo"][0], "D").astype(int)
    assert data["eligible"].all() and (data["aux"] < 0).all()
    return data


def stratum(study, ids):
    t, road = np.divmod(ids, study.nroad)
    minute = study.d["minute"][t]
    return ((minute // 1440 - (minute // 1440).min()) * 2 + (minute % 1440 >= 720)) * study.nroad + road


def training_sample(study, cfg, recipe):
    base, legacy = sample_training(study, cfg)
    group = groups(study.ratio, cfg["ratio"])
    positives = study.train[study.y[study.train] == 1]
    negatives = study.train[study.y[study.train] == 0]
    hard = negatives[group[negatives] == 1]
    budget = min(cfg["negative_budget"], len(negatives))
    ids = base
    if recipe != "B0":
        rng = np.random.default_rng(cfg["sampling_seed"])
        reserved = proportional(hard, stratum(study, hard), min(budget, len(hard)), rng)
        other = negatives[group[negatives] != 1]
        fill = proportional(other, stratum(study, other), budget - len(reserved), rng)
        ids = np.sort(np.r_[positives, reserved, fill])
    assert len(ids) == len(base) and len(np.unique(ids)) == len(ids)
    assert np.isin(ids, study.train).all() and (study.y[ids] >= 0).all()
    sampled_low = int(((study.y[ids] == 0) & (group[ids] == 1)).sum())
    count = np.zeros((3, 2), dtype=np.int64)
    np.add.at(count, (group[ids], study.y[ids]), 1)
    stats = dict(train_positive=len(positives), train_negative=len(negatives), sample_positive=len(positives),
                 sample_negative=budget, positive_weight=budget / len(positives), seed=cfg["sampling_seed"],
                 eligible_legal_low_negatives=len(hard), sampled_legal_low_negatives=sampled_low,
                 legal_low_negative_inclusion_rate=sampled_low / len(hard) if len(hard) else None,
                 sampled_group_class_counts=count.tolist(), uses_realized_t_minus_1_risk=False,
                 strategy="legacy date/peak/road" if recipe == "B0" else "reserve legal-low negatives then date/peak/road")
    return ids, stats


def balanced_group_weights(group, y):
    count = np.zeros((3, 2), dtype=np.int64)
    np.add.at(count, (group, y), 1)
    weights = np.zeros((3, 2), dtype=np.float32)
    active = count > 0
    weights[active] = len(y) / (int(active.sum()) * count[active])
    return weights, count

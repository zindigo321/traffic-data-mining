"""Registered screen, three-seed rolling confirmation and diagnostic calibration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import statistics

from v4_io import legacy_config, load, now, python, record, save, settings, sha


def location(root, fold, recipe, pool, seed):
    results = root / "results"
    parent = results / "calibration" if fold == "final" else results / "dev" / fold
    return parent / f"{recipe}_{pool}_s{seed}"


def fit(root, spec):
    fold, recipe, pool, seed, epoch = spec
    output = location(root, fold, recipe, pool, seed)
    status = output / "status.json"
    if not status.exists():
        cmd = python("v4_train.py", "--fold", fold, "--recipe", recipe, "--pool", pool,
                     "--seed", seed, "--output", output)
        if epoch is not None:
            cmd += ["--fixed-epoch", str(epoch)]
        record(root, f"{fold}_{output.name}", cmd)
    result = load(status)
    if result["status"] != "complete" or tuple(result[k] for k in ("fold", "recipe", "pool", "seed", "fixed_epoch")) != spec:
        raise ValueError(f"Wrong or incomplete job: {output}")
    point = next(p for p in load(output / "matched_recall.json") if p["target_recall"] == .60)
    return {**result, "FP60_per_day": point["FP_per_day"], "FP60": point["FP"], "TP60": point["TP"]}


def execute(root, jobs):
    jobs = list(jobs)
    with ThreadPoolExecutor(max_workers=legacy_config()["workers"]) as pool:
        return list(pool.map(lambda spec: fit(root, spec), jobs))


def average(rows):
    if not rows:
        raise ValueError("Empty experiment group")
    seeds = sorted({r["seed"] for r in rows})
    measures = ("AP", "f1", "ROC_AUC", "recall", "precision", "fp_per_day", "FP60_per_day", "FP60",
                "TP60", "event_f1", "fit_seconds", "peak_RSS_bytes")
    values = {k: [statistics.mean(r[k] for r in rows if r["seed"] == seed) for seed in seeds] for k in measures}
    return {**{k: statistics.mean(v) for k, v in values.items()},
            **{f"{k}_sd": statistics.stdev(v) if len(v) > 1 else None for k, v in values.items()},
            "runs": len(rows), "seed_count": len(seeds)}


def criterion(row):
    return (round(row["AP"], 6), -round(row["FP60_per_day"], 6), round(row["f1"], 6))


def screen(root, cfg):
    variants = ["B", *cfg["variants"]]
    jobs = [(f, r, "mean", cfg["seeds"][0], None) for f, r in product(cfg["folds"], variants)]
    rows = execute(root, jobs)
    summaries = {r: average([row for row in rows if row["recipe"] == r]) for r in variants}
    winner = max(cfg["variants"], key=lambda r: criterion(summaries[r]))
    payload = dict(recipe=winner, summary=summaries, jobs=rows, at_utc=now(), stage="development_screen",
                   rule="Macro AP to six decimals, then fewer FP/day at recall 60%, then larger diagnostic F1, fixed order",
                   old_holdout_used=False)
    save(root / "results/screen.json", payload)
    print(dict(recipe=winner, AP={k: v["AP"] for k, v in summaries.items()}), flush=True)


def confirm(root, cfg):
    spec = load(root / "results/screen.json")
    winner = spec["recipe"]
    jobs = [(f, winner, p, s, None) for f, p, s in product(cfg["folds"], cfg["pools"], cfg["seeds"])]
    jobs += [(f, "B", "mean", s, None) for f, s in product(cfg["folds"], cfg["seeds"])]
    rows = execute(root, jobs)
    groups = [(winner, pool) for pool in cfg["pools"]] + [("B", "mean")]
    summaries, epochs = {}, {}
    for recipe, pool in groups:
        key = f"{recipe}_{pool}"
        selected = [r for r in rows if (r["recipe"], r["pool"]) == (recipe, pool)]
        summaries[key] = average(selected)
        epochs[key] = int(statistics.median(r["best_epoch"] for r in selected))
    choice = max(cfg["pools"], key=lambda p: criterion(summaries[f"{winner}_{p}"]))
    save(root / "results/confirmation.json", dict(rows=rows, summary=summaries))
    payload = dict(recipe=winner, pool=choice, fixed_epochs=epochs, rule=spec["rule"], old_holdout_used=False,
                   scope="Development only", at_utc=now())
    save(root / "results/selection.json", payload)
    print(dict(selection=payload, AP={k: v["AP"] for k, v in summaries.items()}), flush=True)


def calibrate(root, cfg):
    selected = load(root / "results/selection.json")
    winner = selected["recipe"]
    jobs = []
    for recipe, pool in [(winner, p) for p in cfg["pools"]] + [("B", "mean")]:
        epoch = selected["fixed_epochs"][f"{recipe}_{pool}"]
        jobs.extend(("final", recipe, pool, seed, epoch) for seed in cfg["seeds"])
    rows = execute(root, jobs)
    artifacts = {}
    for fold, recipe, pool, seed, _ in jobs:
        path = location(root, fold, recipe, pool, seed)
        artifacts[path.name] = {file: sha(path / file) for file in ("weight.pt", "preprocessor.npz", "assessment.npz",
            "matched_recall.json", "options.json", "sampling_ids.npz", "curve.csv", "status.json")}
    save(root / "results/calibration_freeze.json", dict(at_utc=now(), thresholds={location(root, r["fold"],
        r["recipe"], r["pool"], r["seed"]).name:r["threshold"] for r in rows},
        hashes=artifacts, selection_sha256=sha(root / "results/selection.json"),
        date_interval=cfg["fold_limits"]["final"], result_scope="Calibration fit only; formal release blocked",
        test_unlocked=False))
    save(root / "results/calibration_summary.json", dict(rows=rows, summary={name: average(
        [r for r in rows if f"{r['recipe']}_{r['pool']}" == name]) for name in selected["fixed_epochs"]}))
    print(dict(calibration="complete", models=len(rows), formal_heldout=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "confirm", "calibrate"))
    args = parser.parse_args()
    globals()[args.stage](*settings())

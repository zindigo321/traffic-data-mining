"""New extraction, causal source rebuild, independent replay, and full V10 retraining QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from v10_io import CODE, load, now, python, record, save, sha
from v10_pack import DOCS, NAME


def qa(run, output, scratch):
    scratch.mkdir(parents=True, exist_ok=False)
    archive = output / f"{NAME}.zip"
    bundle = output / NAME
    clean = scratch / "extract" / NAME
    record(scratch, "build", python("v10_pack.py", "build", "--run", run, "--output", output))
    record(scratch, "extract", python("v10_pack.py", "extract", "--archive", archive,
           "--output", clean.parent, "--receipt", scratch / "zip.json"))
    code = clean / "project"
    record(scratch, "delivered_directory", python("v10_pack.py", "verify-dir", "--output", bundle), cwd=code)

    def invoke(phase, dest, *args):
        return record(scratch, f"clean_{dest}", python("run_v10.py", "--stage", phase,
                      "--output", scratch / dest, *args), cwd=code)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(invoke, "verify", "offline", "--artifacts", clean),
                 pool.submit(invoke, "prepare", "prepare")]
        for task in tasks:
            task.result()
    rebuilt = scratch / "prepare/rebuilt"
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(invoke, "verify", "replay", "--artifacts", clean, "--replay-root", rebuilt),
                 pool.submit(invoke, "all", "retrain", "--prepared", rebuilt / "prepared")]
        for task in tasks:
            task.result()

    offline = load(scratch / "offline/evidence/verification_offline.json")
    replay = load(scratch / "replay/evidence/verification_replay.json")
    retrain = scratch / "retrain"
    rerun = load(retrain / "evidence/verification_replay.json")
    tests = load(retrain / "synthetic/status.json")
    assert offline["status"] == replay["status"] == rerun["status"] == tests["status"] == "PASS"
    assert replay["models_replayed"] == rerun["models_replayed"] == 48
    signed = load(clean / "evidence/prepared_development.json")["lead_hashes"]
    for fold, digest in signed.items():
        assert sha(rebuilt / "prepared" / f"lead_{fold}.npz") == digest
    table_names = ("development_operating.csv", "development_matched.csv", "development_groups.csv",
                   "calibration_operating.csv", "calibration_matched.csv", "calibration_groups.csv",
                   "operating_points.csv", "groups.csv", "matched_recall.csv", "paired_date_blocks.csv",
                   "events.csv", "daily_errors.csv", "summary.json")
    tables = {}
    for name in table_names:
        digest = sha(retrain / "results" / name)
        assert digest == sha(clean / "results" / name), f"Rerun table differs: {name}"
        tables[name] = digest
    old, new = [load(r / "evidence/freeze.json") for r in (clean, retrain)]
    assert old["primary"] == new["primary"] == "B0"
    weight_rows = []
    for left, right in zip(old["models"], new["models"], strict=True):
        assert all(left[k] == right[k] for k in ("recipe", "seed", "threshold", "fit_FP", "fit_recovery_FP"))
        assert left["decisions_sha256"] == right["decisions_sha256"]
        for k in ("assessment.npz", "sampling_ids.npz", "preprocessor.npz", "group_weights.npz", "curve.csv"):
            assert left["hashes"][k] == right["hashes"][k]
        weight_rows.append(dict(recipe=left["recipe"], seed=left["seed"],
                               delivered=left["hashes"]["weight.pt"], retrained=right["hashes"]["weight.pt"],
                               bytes_identical=left["hashes"]["weight.pt"] == right["hashes"]["weight.pt"]))
    all_logs = (scratch / "logs", scratch / "offline/logs", scratch / "replay/logs",
                scratch / "prepare/logs", rebuilt / "logs", retrain / "logs")
    detail = []
    for folder in all_logs:
        for path in sorted(folder.glob("*.json")):
            receipt = load(path)
            if "exit_code" in receipt:
                assert receipt["exit_code"] == 0
                log = folder / receipt["log"]
                detail.append(dict(**receipt, console_sha256=sha(log),
                                   console_tail=log.read_text(encoding="utf-8", errors="replace")[-1400:]))
    initial = load(scratch / "zip.json")
    save(output / "解压复核.json", dict(**initial, completed_utc=now(), bytes=archive.stat().st_size,
         documents=len(DOCS), independent_recompute=offline, clean_model_replay=replay,
         clean_full_retrain=rerun, engineering_tests=tests["total"], reconstructed_lead_arrays=signed,
         reproduced_tables=tables, thresholds_identical=True, calibration_weight_digests=weight_rows,
         NYC_formal="blocked", Chicago_formal="blocked", Geolife_formal="blocked", holdout_used=False,
         raw_sources_in_zip=False, commands=detail))
    print(dict(status="PASS", archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
               clean_replayed=48, retrained_models=48, engineering_tests=tests["total"], tables=len(tables)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qa-dir", required=True, type=Path)
    args = parser.parse_args()
    qa(args.run.resolve(), args.output.resolve(), args.qa_dir.resolve())

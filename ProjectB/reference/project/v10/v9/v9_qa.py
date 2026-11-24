"""Build and verify V9 from a new extraction and read-only public source caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v9_io import CODE, load, now, python, record, save, sha
from v9_pack import DOCS, NAME


def qa(run, output, scratch):
    scratch.mkdir(parents=True, exist_ok=False)
    archive = output / f"{NAME}.zip"
    bundle = output / NAME
    extracted = scratch / "extract" / NAME
    commands = []
    commands.append(record(scratch, "build", python("v9_pack.py", "build", "--run", run, "--output", output)))
    commands.append(record(scratch, "extract", python("v9_pack.py", "extract", "--archive", archive,
                                                   "--output", extracted.parent, "--receipt", scratch / "zip.json")))
    clean_code = extracted / "project"
    commands.append(record(scratch, "delivered_directory", python("v9_pack.py", "verify-dir", "--output", bundle),
                           cwd=clean_code))

    def clean(phase, destination, *options):
        return record(scratch, f"clean_{destination}", python("run_v9.py", "--stage", phase,
                      "--output", scratch / destination, *options), cwd=clean_code)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = (pool.submit(clean, "verify", "offline", "--artifacts", extracted),
                pool.submit(clean, "prepare", "prepare"))
        for job in jobs:
            commands.append(job.result())

    rebuilt = scratch / "prepare/rebuilt"
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = (pool.submit(clean, "verify", "replay", "--artifacts", extracted, "--replay-root", rebuilt),
                pool.submit(clean, "all", "refit", "--prepared", rebuilt / "prepared"))
        for job in jobs:
            commands.append(job.result())

    offline = load(scratch / "offline/evidence/verification_offline.json")
    replay = load(scratch / "replay/evidence/verification_replay.json")
    refit = scratch / "refit"
    proof = load(refit / "evidence/verification_replay.json")
    tests = load(refit / "synthetic/status.json")
    assert offline["status"] == replay["status"] == proof["status"] == tests["status"] == "PASS"
    assert replay["models_replayed"] == proof["models_replayed"] == 36
    signed = load(extracted / "evidence/prepared_development.json")["lead_hashes"]
    for fold, digest in signed.items():
        assert sha(rebuilt / "prepared" / f"lead_{fold}.npz") == digest
    compare = ("development_grid.csv", "operating_points.csv", "matched_recall.csv", "paired_date_blocks.csv",
               "support.csv", "error_slices.csv", "events.csv", "summary.json")
    hashes = {}
    for name in compare:
        digest = sha(refit / "results" / name)
        assert digest == sha(extracted / "results" / name)
        hashes[name] = digest
    assert load(refit / "evidence/freeze.json")["models"] == load(extracted / "evidence/freeze.json")["models"]
    assert load(refit / "evidence/status.json")["NYC_formal"] == "blocked"
    all_logs = [scratch / "logs", scratch / "offline/logs", scratch / "replay/logs",
                scratch / "prepare/logs", rebuilt / "logs", refit / "logs"]
    detail = []
    for folder in all_logs:
        for path in sorted(folder.glob("*.json")):
            receipt = load(path)
            if "exit_code" in receipt:
                assert receipt["exit_code"] == 0
                log = folder / receipt["log"]
                detail.append(dict(**receipt, console_sha256=sha(log), console_tail=log.read_text(encoding="utf-8")[-1600:]))
    checks = load(scratch / "zip.json")
    save(output / "extraction_verification.json", dict(**checks, completed_utc=now(), bytes=archive.stat().st_size,
         checked_documents=len(DOCS), independent_recompute=offline, clean_model_replay=replay,
         clean_one_click_refit=proof, engineering_tests=tests["total"], reconstructed_lead_arrays=signed,
         reproduced_tables=hashes, freeze_thresholds_and_weight_digests_identical=True,
         NYC_formal="blocked", Chicago_formal="blocked", Geolife_formal="blocked", holdout_used=False,
         raw_sources_in_zip=False, commands=detail))
    print(dict(status="PASS", archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
               clean_replayed=replay["models_replayed"], engineering_tests=tests["total"], tables=len(hashes)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--qa-dir", required=True, type=Path)
    args = ap.parse_args()
    qa(args.run.resolve(), args.output.resolve(), args.qa_dir.resolve())

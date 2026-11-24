# Project B Full-Data Experiment Entry Points

This translated provenance record describes the historical full-run archive. Relative references to its evidence, results, and figures use that archive layout.

The latest authorization on 2026-11-12 relaxed the experiment's entry conditions, and the experiment used all obtainable New York project data. The original scope covers 121 directed road segments, two peak periods (07:00–09:00 and 17:00–19:00), and dates from 2026-05-11 to 11-11. The 54 shards contain 1019891 records, including 15353 new records. The experiment can begin without waiting for a complete 7 or 28 days, 40% coverage, or a minimum event count.

The full time grid contains 537240 windows, with 299988 known labels and 237252 unknowns. Unknown windows receive scores and count toward coverage. Each full-data supervised recipe visits every known sample in its training partition each epoch. There is no negative sample cap. The experiment includes three pooling methods with three seeds each, center-only controls, within-group ranking, a step-matched sampling control, and deterministic baselines. [Audit report](original_full_audit_report.md)

| Navigation | Files |
|---|---|
| Registration and definitions | Plan (`plan.md`), [method and data contract](original_full_method_and_data_contract.md) |
| Data and dates | Snapshot and per-file hashes (`evidence/source_snapshot.json`), full partitions (`evidence/cohort_final.json`), raw data filtering counts (`evidence/prepared.json`) |
| Completion | Experiment status (`evidence/experiment_status.json`), model selection (`results/development_selection.json`), threshold freezing (`evidence/model_threshold_freeze.json`) |
| Per-sample results and models | results/models, results/decisions |
| Metrics | Operating points by seed (`results/operating_points.csv`), groups (`results/groups.csv`), matched recall (`results/matched_recall.csv`), daily false positives (`results/daily_errors.csv`) |
| Audit and resources | Test report (`test_report.md`), verification (`evidence/verification_replay.json`), resources (`results/training_resources.csv`), changes (`reports/change_notes.md`) |
| Integrity | manifest.csv (`manifest.csv`), SHA256SUMS (`SHA256SUMS`) |

This round uses previously public dates and November 9–11 records, so the result is a full-data temporal backtest. Model evaluation uses chronological training, calibration, and test periods. New independent unseen dates need a separate protocol.

Use the verified read-only CPU environment at `D:/conda_envs/niulaoshi/python.exe`. Dependency versions are pinned in `project/requirements.txt`, and the actual environment is recorded in `logs/environment.json`. All command outputs are outside the package. Source data and existing audit packages are read-only.

```powershell
$fpy = 'D:/conda_envs/niulaoshi/python.exe'
$fbundle = (Get-Location).Path
$fwork = 'G:/gpt-file/Codex/work/ProjectB_full_recomputation'

# Recompute bundled per-sample scores and thresholds without downloading large data.
& $fpy -B -X utf8 ./project/full_pack.py verify-dir --output $fbundle
& $fpy -B -X utf8 ./project/run_full.py --phase verify --artifacts $fbundle --output "$fwork/offline"
& $fpy -B -X utf8 ./project/run_full.py --phase tests --output "$fwork/tests"

# Rebuild, train, calibrate, and evaluate with the hash-verified external raw shards.
& $fpy -B -X utf8 ./project/run_full.py --phase all --output "$fwork/rerun"

# Run stages separately.
& $fpy -B -X utf8 ./project/run_full.py --phase initialize --output "$fwork/staged"
foreach ($fphase in 'check','fetch','prepare','tests','development','select','final','freeze','evaluate','analysis','verify','report') {
    & $fpy -B -X utf8 ./project/run_full.py --phase $fphase --output "$fwork/staged"
    if ($LASTEXITCODE -ne 0) { throw "Full experiment phase $fphase failed" }
}
```

By default, `fetch` attaches the frozen `source_snapshot` in the protocol directory and verifies each file. External raw shard paths are in the snapshot. Use `--source-snapshot` to specify a snapshot with remapped paths. The official API is used for a current continuation only when there is no snapshot. A new snapshot should use a new run directory. The package excludes large raw CSVs and complete feature arrays. Record the specific gaps if files are missing.

The `prepare` stage rebuilds the development and final feature arrays. After rebuilding, the saved models can be replayed.

```powershell
& $fpy -B -X utf8 ./project/run_full.py --phase verify --artifacts $fbundle --replay-root "$fwork/rerun" --output "$fwork/replay"
```

Local work checked source code, ran tests, rebuilt the data, and replayed models. The archive's adjacent extraction receipt records checks performed in a clean directory and identifies any retraining runs that were not repeated. Per-model training IDs, unique visits per epoch, update counts, weights, and preprocessing artifacts are retained to verify full-data coverage.

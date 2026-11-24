# Project B Dynamic Edge Weight Audit

The full NYC historical backtest completed 18 new dynamic training runs and reused 18 verified static controls. The primary model was `dynamic_mean`. Its AP was 0.6664 ± 0.0243 and positive-class F1 was 0.6501 ± 0.0195. Both met the backtest metric targets. Total false positives were 14.600 per day and recovery false positives were 7.504 per day. See the [audit report](audit_report.md) for the false-positive budget checks.

The original test dates were already public. This ablation reuses the historical NYC snapshot, labels, and full test support. A new independent blind test remains outstanding. The original Geolife and Chicago protocols remain blocked.

| Configuration | AP, mean ± sample SD | Positive F1, mean ± sample SD | Balanced accuracy | Recall | Total FP/day | Recovery FP/day |
| --- | --- | --- | --- | --- | --- | --- |
| graph_mean | 0.6553 ± 0.0348 | 0.6455 ± 0.0114 | 0.7892 | 0.5870 | 14.756 | 7.578 |
| graph_max | 0.6591 ± 0.0262 | 0.6551 ± 0.0158 | 0.7955 | 0.5994 | 14.652 | 7.630 |
| graph_attention | 0.6638 ± 0.0332 | 0.6388 ± 0.0196 | 0.7832 | 0.5747 | 14.289 | 7.459 |
| dynamic_mean | 0.6664 ± 0.0243 | 0.6501 ± 0.0195 | 0.7919 | 0.5922 | 14.600 | 7.504 |
| dynamic_max | 0.6591 ± 0.0144 | 0.6440 ± 0.0053 | 0.7833 | 0.5743 | 13.341 | 6.852 |
| dynamic_attention | 0.6687 ± 0.0134 | 0.6541 ± 0.0079 | 0.7941 | 0.5965 | 14.489 | 7.348 |

Start with the [audit report](audit_report.md), [method and data contract](method_and_data_contract.md), and [plan](plan.md). The operating points, matched recall, and paired date-block results are in `results/operating_points.csv`, `results/matched_recall.csv`, and `results/paired_date_blocks.csv`. Then see `evidence/dynamic_threshold_freeze.json`, `logs/`, and the [test report](test_report.md).

`project/dynamic_model.py` implements bounded edge weights. `dynamic_train.py` trains on the full support. `dynamic_score.py` handles development selection, calibration freeze, and historical scoring. `dynamic_verify.py` recomputes metrics independently and handles safe replay. `run_dynamic.py` is the unified entry point. The configuration is in `project/config.json`. The original preprocessing modules are in `project/full_prepare.py` and its inherited code.

```powershell
$python = 'D:/conda_envs/niulaoshi/python.exe'
# Recompute metrics and thresholds from the bundled predictions in the extracted audit directory.
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_verify_new --no-replay
# Add the verified external prepared arrays and sample replay inputs for all 36 models.
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_replay_new
# Run all 18 dynamic training fits in a new directory and reuse the 18 verified static controls.
& $python -B -X utf8 project/run_dynamic.py --phase all --output G:/gpt-file/Codex/work/ProjectB_dynamic_refit_new
```

Choose a fresh output directory. For a phased run, replace `--phase all` with `check`, `tests`, `development`, `select`, `final`, `freeze`, `evaluate`, `analysis`, `verify`, or `report`. The full entry point copies initialization files. Keep existing registration receipts when running `check`.

Offline metric recomputation needs the bundled predictions and the original Python dependencies. Replay also needs the two read-only arrays in `evidence/prepared.json`. Full retraining from this historical audit directory requires the original full audit package and its raw snapshot path. Verify hashes first. The original large public tables, private Geolife trajectories, and large preceding archives were outside this audit directory. See `logs/environment_dynamic.json` and `project/requirements_actual.txt` for dependency versions.

`manifest.csv` lists the package members except the manifest and `SHA256SUMS`. `SHA256SUMS` includes the manifest and excludes itself. Keep the extraction receipt and the complete ZIP hash outside the archive.

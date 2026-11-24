# Project B Dynamic Edge Weight Experiment Audit

## Results and Acceptance

The NYC public-data experiment completed 18 dynamic training runs across three pooling methods, three seeds, and development and final stages. It reused and verified 18 static models. Development AP selected **dynamic_mean** as primary. Across three historical test seeds, AP was 0.6664 ± 0.0243, positive-class F1 was 0.6501 ± 0.0195, balanced accuracy was 0.7919, and recall was 0.5922. **Both metric targets were met.** Total false positives were 14.600 per day and recovery false positives were 7.504 per day. The table records the separate false-positive budget checks. ROC-AUC was 0.9696, precision was 0.7208, and accuracy was 0.9773. The positive rate of about 3.56% makes accuracy insufficient as a measure of positive-class performance.

The original backtest dates are public. This experiment compares ablations on the same historical support. A new independent blind test remains outstanding. The original Geolife coursework and Chicago protocol remain blocked or N/A. The public NYC proxy cannot establish the original Geolife user count, years, Beijing road coverage, or at-least-five-minute congestion requirement.

| Pooling | Historical AP ≥ .60 and F1 ≥ .60, three-seed means | Total FP ≤ 12/day | Recovery FP ≤ 6/day |
| --- | --- | --- | --- |
| dynamic_mean | Yes | No | No |
| dynamic_max | Yes | No | No |
| dynamic_attention | Yes | No | No |

## Data, Labels, and Full Support

The frozen i4gi-tjb9 snapshot has 54 shards and 1,019,891 source rows. Latest `data_as_of` was `2026-11-11T17:12:06`. See `evidence/source_snapshot.json` and `evidence/registration_dynamic.json` for paths, queries, and hashes. There are 121 fixed roads, 56 candidate edges, and 54 isolated roads. Dates use NYC local time. All fixed windows from the incomplete final source day remain in the sample. Labels without evidence are unknown.

| Split | Date range, end exclusive | Days | Known windows | Positive | Negative | Unknown | Positive events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2026-05-11 → 2026-08-31 | 112 | 171672 | 4649 | 167023 | 153576 | 1503 |
| calibration | 2026-08-31 → 2026-09-28 | 28 | 47780 | 1476 | 46304 | 33532 | 433 |
| retrospective_test | 2026-09-28 → 2026-11-12 | 45 | 80536 | 2866 | 77670 | 50144 | 796 |

Supervised training uses every known window once per epoch. Unknown labels are excluded from training and supervised metrics. The three pooling methods and seeds share training IDs, normalization, evaluation keys, and labels within each development or final phase, with a common 12-epoch budget. The test set has 80,536 known windows and 2,866 positives, a positive rate of 3.5587%. Coverage is 80,536/130,680 or 61.6284%. The 50,144 unknown test windows receive scores and appear in prediction files.

The time splits and window boundaries follow the preceding full-data protocol. Roads overlap across splits, so the experiment measures temporal generalization on known roads. The public speed estimates contain no person or trajectory identifiers, so separation by person could not be checked. Arrival delays remain an offline assumption.

## Implementation and Paired Results

See [the method and data contract](method_and_data_contract.md) for the edge weight formula, tensor shape, direction, missingness and self-edge rules, and complexity. The core implementation is `project/dynamic_model.py`. Training and full-support sampling follow the original controls. The two edge-weight MLPs add 1,704 parameters. Testing confirmed that zero-initialized gates exactly matched static outputs.

The following backtest results use **the threshold frozen on calibration under false-positive constraints**. Seeds are 11, 22, and 33. SD is the sample standard deviation. AP is `sklearn.average_precision_score`.

| Configuration | AP, mean ± sample SD | Positive F1, mean ± sample SD | Balanced accuracy | Recall | Total FP/day | Recovery FP/day |
| --- | --- | --- | --- | --- | --- | --- |
| graph_mean | 0.6553 ± 0.0348 | 0.6455 ± 0.0114 | 0.7892 | 0.5870 | 14.756 | 7.578 |
| graph_max | 0.6591 ± 0.0262 | 0.6551 ± 0.0158 | 0.7955 | 0.5994 | 14.652 | 7.630 |
| graph_attention | 0.6638 ± 0.0332 | 0.6388 ± 0.0196 | 0.7832 | 0.5747 | 14.289 | 7.459 |
| dynamic_mean | 0.6664 ± 0.0243 | 0.6501 ± 0.0195 | 0.7919 | 0.5922 | 14.600 | 7.504 |
| dynamic_max | 0.6591 ± 0.0144 | 0.6440 ± 0.0053 | 0.7833 | 0.5743 | 13.341 | 6.852 |
| dynamic_attention | 0.6687 ± 0.0134 | 0.6541 ± 0.0079 | 0.7941 | 0.5965 | 14.489 | 7.348 |

| Same-pooling paired difference | Δ AP | Δ F1 | Δ total FP/day at matched recall | Δ recovery FP/day at matched recall |
| --- | --- | --- | --- | --- |
| dynamic_mean − graph_mean | +0.0112 | +0.0045 | -0.630 | -0.244 |
| dynamic_max − graph_max | -0.0000 | -0.0112 | +0.422 | -0.185 |
| dynamic_attention − graph_attention | +0.0048 | +0.0153 | -0.859 | -0.437 |

The table reports both gains and regressions. The test set never selected a pooling method, seed, or threshold. The selected dynamic model increased AP by +0.0112 and F1 by +0.0045 relative to its same-pooling static control. The original preselected static Max is reported separately. Comparisons across different pooling methods cannot isolate the effect of edge weights.

## Matched Recall and Statistical Limits

The primary diagnostic holds the number of true positives fixed at 60% recall over all known test windows. Ties are broken by `sample_id`. These cutoffs use retrospective ranking and do not establish deployable thresholds. Results at 50%, 70%, and 80% recall and for the t−2 low-speed group appear in `matched_recall.csv`.

| Configuration | Achieved recall | Total FP/day | Recovery FP/day |
| --- | --- | --- | --- |
| graph_mean | 0.6001 | 15.378 | 7.815 |
| graph_max | 0.6001 | 14.444 | 7.541 |
| graph_attention | 0.6001 | 15.541 | 7.822 |
| dynamic_mean | 0.6001 | 14.748 | 7.570 |
| dynamic_max | 0.6001 | 14.867 | 7.356 |
| dynamic_attention | 0.6001 | 14.681 | 7.385 |

Mean reduced total and recovery false positives for 3/3 seeds, with both decreasing for 3/3. Max reduced total false positives for 1/3 and recovery false positives for 2/3, with both decreasing for 1/3. Attention reduced total false positives for 2/3 and recovery false positives for 3/3, with both decreasing for 2/3.

| Pair | Metric | Conditional 95% lower bound | Conditional 95% upper bound |
| --- | --- | --- | --- |
| dynamic_mean_minus_graph_mean | f1 | -0.0040 | +0.0120 |
| dynamic_mean_minus_graph_mean | FP60_day | -1.2004 | -0.0148 |
| dynamic_mean_minus_graph_mean | recovery_FP60_day | -0.5852 | +0.0891 |
| dynamic_max_minus_graph_max | f1 | -0.0181 | -0.0051 |
| dynamic_max_minus_graph_max | FP60_day | -0.2080 | +1.1556 |
| dynamic_max_minus_graph_max | recovery_FP60_day | -0.4815 | +0.1630 |
| dynamic_attention_minus_graph_attention | f1 | +0.0049 | +0.0251 |
| dynamic_attention_minus_graph_attention | FP60_day | -1.8741 | -0.0146 |
| dynamic_attention_minus_graph_attention | recovery_FP60_day | -0.9333 | -0.0739 |

The bootstrap uses 1,000 shared date-block draws conditional on the fitted models, fixed thresholds, and observed dates. The intervals do not measure variation from retraining or sampling new roads. No multiple-comparison correction was applied. An interval that crosses zero offers insufficient evidence of a stable advantage. The main table gives across-seed AP standard deviations, separate from these intervals.

## False Positives, Topology, and Edge Weight Evidence

`results/groups.csv` contains AP, F1, confusion counts, and recovery false positives for the t−2 low-speed, regular, and missing groups. `topology_groups.csv` separates connected and isolated roads. `daily_errors.csv` contains daily counts. Event-level results group contiguous ten-minute windows on the same road and appear separately from window metrics in `operating_points.csv`.

Using seed 1811, randomly sample up to 1,024 connected assessment graphs without using labels and save edge weights under `results/edge_weights/`. `edge_multiplier_summary.csv` reports the median, quantiles, bounds, and share of weights reverting to 1 on physical edges. `edge_neutralization_diagnostic.csv` disables the edge prior on the same trained backbone while holding thresholds fixed. It tests the mechanism and contributes no new paired fit or model selection.

The 54 isolated roads and the small set of geometry-based candidate edges limit where edge weights can act. An isolated graph produces identical outputs when the edge prior is toggled on the same fixed backbone. Retraining a backbone can still change its shared parameters, so cross-model differences on isolated roads cannot establish direct edge propagation. Geometry based on endpoint proximity, conflicting metadata, estimated speeds, and missingness continue to limit interpretation.

## Freeze, Execution, and Test Evidence

The development recipe selected mean AP across the three seeds, then constrained F1, then Mean, Max, Attention in registered order. The final checkpoint is epoch 11 for `dynamic_mean`, `dynamic_max`, and `dynamic_attention`. The weights and thresholds were frozen at `2026-11-12T07:20:30.453969+00:00` UTC. The freeze receipt SHA-256 is `e716e42b73f4275e8ce0ffb06fd2212b817e1dfbafc034671894d970b56e02de`. Thresholds used only the 28-day calibration period. Final training executed all 12 epochs and saved calibration scores before producing dynamic test scores after the freeze. Static test scores came from the earlier public controls.

All 75 engineering tests passed, including 12 new dynamic tests for forward and backward passes, overfitting, graphs and pooling, batch and center mapping, direction, missingness fallback, weight bounds, safe saving, and invariance to changes in t−1, t, and future observations. Synthetic scores are engineering evidence. Independent verification recomputed 72 operating points, 288 matched-recall points, and 216 subgroup metrics. It sampled all 36 models, including 18 static controls and 18 new dynamic models, and found a maximum replay probability error of 0.0. Static metrics matched the preceding bundle.

The CPU was Intel64 Family 6 Model 183 Stepping 1, GenuineIntel. Python was `3.11.17 | packaged by conda-forge | (main, Oct  6 2026, 23:51:21) [MSC v.1944 64 bit (AMD64)]`, with `torch=2.14.1+cpu`, 2 threads per process, and at most 4 concurrent jobs. Fit-and-score time summed to 3,191.5 seconds across the new runs. See `training_resources.csv` for per-model resources and `logs/` for commands, exit codes, timings, and errors. The original Conda environment was used directly. Model loading used `weights_only=True` and NPZ loading used `allow_pickle=False`.

The historical clean extraction audit checked hashes, recalculated metrics, replayed weights, and ran the tests. It did not repeat all 18 training fits. Its root extraction receipt records that scope. Pre-packaging recomputation and replay in a clean directory passed. See `evidence/clean_qa_prepackage.json`.

## Verification and Delivery

1. Check the external ZIP digest and then `manifest.csv` and `SHA256SUMS`. The checksum list includes the manifest and excludes itself.
2. Locate the read-only prior audit bundle and prepared arrays using `evidence/path_mapping.json`. Verify each shard's query, row count, and SHA-256 against `source_snapshot.json`. Source collection and usage records are in `source_provenance/`.
3. Verify hashes for the plan, configuration, source, prepared arrays, and frozen weights. Verify the order of model selection, final training, threshold freeze, and test scoring. Independent evaluation on future dates remains outstanding.
4. Use `assessment.npz`, `training_ids.npz`, and `curve.csv` in `results/models/` and the dynamic final predictions in `results/evaluation/` to recompute thresholds and full metrics. Check coverage, unknown labels, event groupings, and the day denominator on the full support.
5. Run `project/run_dynamic.py --phase verify` in a new output directory and sample replay with verified preprocessors and weights-only model loading.

```powershell
$python = 'D:/conda_envs/niulaoshi/python.exe'
# Recompute metrics and thresholds from bundled predictions in the extracted audit root.
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_verify_new --no-replay
# Add the verified external prepared arrays to sample replay inputs for all 36 models.
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_replay_new
# Run the 18 dynamic fits in a fresh directory and reuse the 18 verified static controls.
& $python -B -X utf8 project/run_dynamic.py --phase all --output G:/gpt-file/Codex/work/ProjectB_dynamic_refit_new
```

Choose a fresh output directory. For staged execution, replace `--phase all` with `check`, `tests`, `development`, `select`, `final`, `freeze`, `evaluate`, `analysis`, `verify`, or `report`. The full entry point copies initialization files. Keep existing registration receipts when running `check`.

The defense can report a measured directed Graph Transformer edge-prior ablation with three pooling methods and three seeds on the full NYC proxy backtest, with a strict ten-minute prediction lead. Report gains, regressions, and false-positive budgets separately. A new blind test and the original Geolife and Chicago tasks require their own evidence.

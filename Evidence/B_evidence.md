# Project B Read-Only Evidence Review

Review date: November 17, 2026

## Version and Provenance

`G:/course-jiaotongshujuwajue/ProjectB` is the original Geolife MLP/RF project. Its `results/training_status.json` reports `blocked` and `formal_training_executed=false`. `pipeline_summary.json` reports that all 578 road-minute windows are unknown due to `insufficient_history`. The value 0.9916535624 is the nearest-road match rate (8,198 of 8,267). The trained Graph Transformer comes from a separate package.

The Project B rerun is from `outputs/ProjectB_GT_niulaoshi_20261112/ProjectB_GT_niulaoshi`. The local verification receipt (`local_run_verification.json` in this translation) records 18 complete dynamic-model retrains across three pooling methods, three seeds, and development and final stages, alongside 18 verified static controls. Source, data, weights, per-example predictions, plots, and hashes were available. Sampled replay for 36 models had a probability difference of zero. The new dynamic weights and aggregate metrics reproduced. The training occurred on November 12, 2026. The November 17 work verified, presented, and packaged those results.

The original run ZIP was 114,528,697 bytes with SHA-256 `b732275b75092bf4738373965749db6405218fe29b9e25908d861d953d3fd424`.

## Four Presentation Evidence Sections

### 1. Original Task and Supplementary Data Protocol

- The coursework calls for 100 Geolife users in 2010–2012, rush hours 07:00–09:00 and 17:00–19:00, and speed below 40% of the historical mean for the same period for five continuous minutes.
- All 578 windows from the original project are unknown. Training is blocked, and unknown labels are excluded from the negative class.
- The NYC DOT public table `i4gi-tjb9` provided 54 files and 1,019,891 raw rows from 2026-05-11 to 2026-11-11. The latest source timestamp was 2026-11-11T17:12:06, and the final day is incomplete.
- The graph has 121 fixed directed segments, 56 geometric candidate directed edges, and 54 isolated nodes. A local graph has up to five nodes.
- The proxy defines two consecutive 10-minute speed estimates each below 40% of their past baselines. Each estimate needs both 5-minute halves. Baselines use at least three qualifying prior days for the same road, period, and weekday or weekend type and are frozen after training. This proxy does not establish minute-by-minute low speed under the original five-minute definition.
- Final training [05-11, 08-31) has 171,672 known windows and 4,649 positives. Calibration [08-31, 09-28) has 47,780 known and 1,476 positives. Backtesting [09-28, 11-12) has 80,536 known, 2,866 positive, 77,670 negative, and 50,144 unknown windows, with 796 positive events over 45 days. Known coverage is 61.6284% and positive prevalence among known windows is 3.5587%.
- The label-distribution plot can show known positives, known negatives, and unknowns separately. The model is binary.

### 2. Directed Graph Transformer and Dynamic Edge Weights

- A score for target bin t is issued at t−10 minutes. The newest dynamic feature bin is t−2, covering [t−20, t−10). Speeds from label bins t−1 and t are excluded.
- The 52 features cover past speed ratios and trends, coverage, historical support, road attributes, and time. Training-split imputation and standardization precede concatenation with 52 missingness indicators, giving input shape [B, N, 104].
- The target is node 0 of a fixed one-hop graph. Attention sees physical predecessors and self-loops. There are two layers and four heads of width 8, with total width 32. The input projects from 104 to 32 features.
- The edge gate uses eight legal past values and missingness indicators at each endpoint. Source, destination, and absolute difference make 48 inputs to an MLP of widths 48→16→4. Its log prior is log(2) × tanh(...), giving a multiplier in [0.5, 2] initially equal to 1. If either endpoint's latest relative speed is missing, the multiplier is 1. Self-loops also use 1. The gate adds 1,704 parameters and no new road edges.
- Attention scores are QKᵀ / sqrt(8) + log(edge prior) + directed mask. After softmax, the layer aggregates V, then applies residuals, layer normalization, and feed-forward processing.
- Mean, Max, and Attention are separate models. Pooling uses valid nodes in the target subgraph, and a 32-dimensional pooled vector plus the 32-dimensional center vector feeds the 64-dimensional classifier for the segment label.
- Training uses weighted BCE, Adam with learning rate 0.001, batch size 128, 12 epochs, two CPU threads per process, and seeds 11, 22, and 33 on the same support and budget. Mean development AP selected `dynamic_mean`. The dynamic final models saved epoch 11. Thresholds were fitted on calibration and then frozen.

### 3. Static and Dynamic Pooling Results

All results use the same 80,536 known test windows. AP and positive F1 show mean ± sample SD across three seeds. The main threshold optimizes calibration F1 within false-positive constraints.

| Model | AP | Positive F1 | Balanced accuracy | Recall | FP/day | Recovery FP/day |
| --- | --- | --- | --- | --- | --- | --- |
| Static Mean | .65526±.03477 | .64552±.01139 | .78922 | .58700 | 14.756 | 7.578 |
| Dynamic Mean (selected on development) | .66644±.02427 | .65005±.01952 | .79189 | .59223 | 14.600 | 7.504 |
| Static Max | .65914±.02618 | .65514±.01582 | .79548 | .59944 | 14.652 | 7.630 |
| Dynamic Max | .65913±.01439 | .64398±.00531 | .78330 | .57432 | 13.341 | 6.852 |
| Static Attention | .66382±.03318 | .63880±.01963 | .78319 | .57467 | 14.289 | 7.459 |
| Dynamic Attention | .66866±.01345 | .65407±.00789 | .79407 | .59653 | 14.489 | 7.348 |

Dynamic Mean gains 0.01118 AP and 0.00453 F1 over its static counterpart. Dynamic Attention gains 0.01528 F1, and Dynamic Max loses 0.01115 F1. All three dynamic models have mean AP and F1 above 0.60. The main model exceeds the deployment budgets of 12 total and 6 recovery false positives per day. Attention has the higher test results for some metrics, while Mean is the development-selected model.

Dynamic Mean accuracy is .97733 with a 96.44% majority class, so report the class-sensitive metrics. AP is `average_precision_score`.

### 4. False Positives at Matched Recall and Verification

The post hoc ranking diagnostic uses common support and the same true-positive count at recall 0.6001396. Its threshold is for comparison.

| Pooling | Static FP/day | Dynamic FP/day | Static recovery FP/day | Dynamic recovery FP/day |
| --- | --- | --- | --- | --- |
| Mean | 15.3778 | 14.7481 | 7.8148 | 7.5704 |
| Max | 14.4444 | 14.8667 | 7.5407 | 7.3556 |
| Attention | 15.5407 | 14.6815 | 7.8222 | 7.3852 |

Mean reduced both matched-recall false-positive rates in 3/3 seeds. Conditional 95% intervals from 1,000 paired date-block replicates span zero for the F1 change [-.003986, .012004] and the recovery FP60/day change [-.585185, .089074]. That evidence leaves the stability of a recovery benefit unresolved. Attention's recovery FP60/day interval is [-.933333, -.073889], with no multiple-comparison correction. Max's F1 degradation also matters.

Acceptance covered 81 tests, including 75 original engineering and six portability tests. All 36 model replays had zero probability difference. Both prepared arrays were rebuilt from packaged raw CSVs with identical byte hashes. Weights for 18 new dynamic retrains matched their references individually. The run provides plots, commands, exit codes, timings, and models.

The NYC results are a temporal backtest on disclosed historical dates. The original Geolife and Chicago tasks still have unmet verification conditions. Segment speed estimates, geometric-neighbor topology, isolated nodes, missing coverage, and unavailable arrival timestamps add limitations. The road overlap across time splits evaluates temporal generalization on the same roads.

## Result Figures and Sources

The original package root is `outputs/ProjectB_GT_niulaoshi_20261112/ProjectB_GT_niulaoshi/`.

- `reference/figures/paired_PR.png` shows precision–recall curves for three pooling methods across three seeds.
- `reference/figures/paired_metrics.png` shows static and dynamic AP, F1, and recall with error bars.
- `reference/figures/matched_recall_FP.png` compares total and recovery false positives near 60% recall.
- `reference/figures/development_curves.png` shows development AP across training for seed 11.
- The deck's blue and red plots use `reference/results/summary.json`, `paired_summary.json`, `operating_points.csv`, and `matched_recall.csv`. The original plotting values were recorded in `work/final_course_20261117/B_evidence.json`.
- `reference/results/dynamic_selection.json` records the development selection rule, primary model, and epoch.
- `reference/evidence/cohort_final.json` records calendar, sample sizes, unknowns, and class distributions.
- `reference/results/paired_date_blocks.csv` records paired date-block intervals.
- `reference/method_and_data_contract.md` and `src/dynamic_model.py` record methods and implementation.
- `local_run_verification.json`, `local_run_acceptance.md`, and `validation_logs/` record the fresh-directory run.
- `run.py` is the run entry point, with `run_niulaoshi.cmd` for Conda. It defaults to verification. `prepare` rebuilds arrays, `replay` replays, `predict` scores, and `train` fits 18 dynamic models and reuses 18 verified static controls.

## Final Delivery Scope

The main Project B delivery is the extracted runnable NYC package with source, public data, prepared arrays, weights, results, and logs. The original Geolife project is retained only as a course comparison with source, requirements, and a summary of why training was blocked. Keep the old `ProjectB.zip`, `data/*.pkl`, and private matched and unmatched coordinate tables out of the public submission. The old directory contains private trajectory and executable pickle caches, with no formal training result. The new public NYC package contains no private Geolife tracks.

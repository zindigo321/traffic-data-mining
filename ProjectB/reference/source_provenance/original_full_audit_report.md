# Project B Full-Data Experiment Audit Report

This translated provenance record describes the historical full-run archive. Relative references to its evidence, results, and figures use that archive layout.

The latest authorization removed calendar waits, minimum coverage, and minimum event counts as entry conditions. The experiment combined all verified cached records with currently obtainable New York continuation data and fitted 36 neural models and six deterministic baselines. Five full-data recipes use all known samples within their training periods. The sixth is a registered step-matched sampling control. Unknown windows receive scores and count toward coverage.

The source contains 1,019,891 records, including 15,353 added over the last three days. The latest timestamp in the snapshot is 2026-11-11T17:12:06.000. The study runs from 2026-05-11 to 11-11 and covers 121 fixed directed segments and the original two peak periods. The public V5/V6 holdouts and November 9–11 records are part of this retrospective protocol. Results are released as a time-split retrospective evaluation. Independent validation on unseen future dates requires a separately reserved period. Source snapshot (`evidence/source_snapshot.json`) · Raw-file verification (`evidence/sources_cached.json`)

| Partition | Dates, end exclusive | Fixed windows | Known | Positive | Negative | Unknown | Events |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 2026-05-11 to 2026-08-31 | 325,248 | 171,672 | 4,649 | 167,023 | 153,576 | 1,503 |
| calibration | 2026-08-31 to 2026-09-28 | 81,312 | 47,780 | 1,476 | 46,304 | 33,532 | 433 |
| retrospective_test | 2026-09-28 to 2026-11-12 | 130,680 | 80,536 | 2,866 | 77,670 | 50,144 | 796 |

The final source day does not include all evening-peak records. The fixed grid retains the unknown windows and reports observed coverage. Daily false-positive rates use the full registered calendar days as the denominator. Window false positives and event metrics are reported separately. Window keys are disjoint across training, calibration, and testing. The training baseline updates daily from earlier qualifying observations. Calibration and testing use frozen training history. Extending training may therefore change a same-day label relative to an earlier frozen fold while preserving the original two-bin 40% rule. Partitions and unknown reasons (`evidence/cohort_final.json`)

Mean development AP across three seeds selected Graph Transformer Max as the primary model. Final training saves the median of its development checkpoint epochs. All models completed 12 epochs. Thresholds were fitted only on calibration from 2026-08-31 to 09-27. Retrospective scores for 2026-09-28 to 11-11 took no part in model or threshold selection.

| Retrospective test mean over three seeds | AP | BA | F1 | Recall | Total FP/day | Recovery FP/day |
|---|---:|---:|---:|---:|---:|---:|
| Graph Transformer Mean | 0.6553 | 0.7892 | 0.6455 | 0.5870 | 14.756 | 7.578 |
| Graph Transformer Max | 0.6591 | 0.7955 | 0.6551 | 0.5994 | 14.652 | 7.630 |
| Graph Transformer Attention | 0.6638 | 0.7832 | 0.6388 | 0.5747 | 14.289 | 7.459 |
| Full-data center only | 0.6804 | 0.7962 | 0.6548 | 0.6011 | 14.896 | 7.548 |
| Full-data center plus within-group ranking | 0.6685 | 0.7781 | 0.6322 | 0.5641 | 13.748 | 6.511 |
| Sampled center with step matching | 0.6144 | 0.7367 | 0.5669 | 0.4813 | 13.741 | 5.815 |
| Full-data logistic regression | 0.6212 | 0.6927 | 0.5091 | 0.3908 | 9.200 | 4.600 |
| Admissible lag20 rule | 0.6422 | 0.7470 | 0.6015 | 0.5000 | 10.356 | 5.444 |
| Majority class | 0.0356 | 0.5000 | 0.0000 | 0.0000 | 0.000 | 0.000 |

The preselected model's test AP is 0.6591±0.0262 and its F1 is 0.6551±0.0158. Deterministic baselines fit once, so their standard deviations are N/A. The 12/6 budgets are mean daily false-positive limits for choosing a calibration threshold. The table reports the actual test false positives in full. Operating points by model and seed (`results/operating_points.csv`)

| Primary model threshold rule | AP | BA | F1 | Recall | Total FP/day | Recovery FP/day |
|---|---:|---:|---:|---:|---:|---:|
| constrained | 0.6591 | 0.7955 | 0.6551 | 0.5994 | 14.652 | 7.630 |
| max_f1 | 0.6591 | 0.8429 | 0.6768 | 0.6995 | 23.452 | 10.000 |

The maximum-F1 operating point quantifies the trade-off between false positives and recall. Both thresholds are fitted on calibration only. Test values are retained when scores miss their targets or false positives exceed budget.

| Matched recall at 60% on full common support | Total FP/day | Recovery FP/day | Actual recall |
|---|---:|---:|---:|
| Graph Transformer Mean | 15.378 | 7.815 | 0.6001 |
| Graph Transformer Max | 14.444 | 7.541 | 0.6001 |
| Graph Transformer Attention | 15.541 | 7.822 | 0.6001 |
| Full-data center only | 14.881 | 7.370 | 0.6001 |
| Full-data center plus within-group ranking | 15.874 | 7.052 | 0.6001 |
| Sampled center with step matching | 22.089 | 7.874 | 0.6001 |
| Full-data logistic regression | 18.400 | 8.067 | 0.6001 |
| Admissible lag20 rule | 15.800 | 7.622 | 0.6001 |
| Majority class | 1062.356 | 12.600 | 0.6001 |

Matched recall uses the same support and the same number of true positives. Tied scores are ordered by fixed sample key. Operating thresholds and matched-recall cutoffs are recorded separately. AP, recall, and false negatives are retained for the admissible t−2 low-speed, non-low-speed, and missing groups. Calendar-date block resampling uses 1000 draws with shared weights across models and seeds. The intervals are conditional on the fixed models, thresholds, and retrospective sample. Group metrics (`results/groups.csv`) · Matched-recall details (`results/matched_recall.csv`) · Paired intervals (`results/paired_date_blocks.csv`)

Each full-data neural recipe visits every known sample in its training partition once per epoch. Training ID files support independent verification. The sampling control keeps 12000 negatives and all positives and cycles through sampled IDs to match the same total visit count. All six recipes use exactly the same number of steps per epoch. Mean, Max, and Attention share training IDs, random seeds, features, and backbone. Pooling is their only difference. Initial class counts, actual training visits, execution times, peak memory, and parameter counts are in the resource table. Training resources (`results/training_resources.csv`)

The experiment passed 63 engineering tests. Independent verification recomputed 84 operating points and 336 matched-recall points. Replay of 36 neural models returned a maximum score difference of 0.0. Weights load with `weights_only=True` and NPZ files with `allow_pickle=False`. Real-data results and synthetic engineering tests are stored in separate directories. Verification receipt (`evidence/verification_replay.json`) · Engineering tests (`synthetic/status.json`)

The source is public New York DOT segment speed estimates and the fixed road geometry. Speeds are in mph. Source timestamps are timezone-naive local times. There is no separate observation arrival timestamp, so `data_as_of` is the offline availability proxy. Source rows are filtered by the original speed and travel-time quality rules. Missing speeds, insufficient history, and unconfirmed low speed remain unknown. The package contains curves, parameters, per-sample float32 probabilities and logits and float64 scores, sampled and full-data IDs, and the preprocessor. Large raw files and feature arrays remain at external read-only paths. Shard SHA hashes, queries, and reconstruction commands are retained. [Official NYC data](https://data.cityofnewyork.us/d/i4gi-tjb9)

Three-pooling PR plot (`figures/test_PR_three_pooling.png`) · Metrics and false positives (`figures/test_pooling_and_controls.png`) · Development training curves (`figures/development_curves.png`)

Verify in this order.

1. Check the manifest, `SHA256SUMS`, and original V10/V5/V6 signatures. Verify each of the 54 raw shards against `source_snapshot`.
2. Rebuild 121 nodes and 56 candidate directed edges from the fixed geometry. Recompute the full original grid, historical baseline, original labels, and 52 strictly ahead-of-time features. Verify the prepared data and partition key digests.
3. Check that each training ID set matches the full set of known indices, the sampling control matches visit counts, and calibration and test keys are disjoint.
4. Confirm the sequence of development selection, final training, threshold freezing, and test-metric computation. Recompute confusion counts, AP, BA, F1, recall, false positives, and matched recall for each seed, and replay a sample from the saved weights.
5. Extract into a fresh directory, check the manifest, then use the entry points to rebuild features and recompute results. See the extraction verification receipt next to the archive for the steps that were executed. [Single-command and staged entry points](original_full_README.md)

This round opened access to the experiment and completed it on available data. The original Geolife coursework and Chicago continuation retain their own evidence. The full-data New York results are retrospective temporal validation on public supplemental data.

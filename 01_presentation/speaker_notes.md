# Speaker Notes

## Slide 1

This report covers two independent projects. Project A studies spatial density and hotspots in Geolife trajectories. Project B uses a directed Graph Transformer to issue warnings with a strict 10-minute lead. The NYC public-data backtest is a supplemental task. The original Geolife task still lacks sufficient labels.

Evidence

- ProjectA/results/analysis_summary.json
- ProjectB/reference/results/summary.json
- archive/legacy-presentation/traffic_data_mining_course_defense.pptx

## Slide 2

The selected 1,108 files contained 2,029,261 points. Region and quality filters retained 1,691,926, minute thinning retained 101,019, and the configured sample contained 53,786 points across 1,102 trajectories. Cleaning removed invalid points, duplicates, same-time conflicts, and jump endpoints. Distances use EPSG:32650 projected coordinates. The main clustering uses only spatial x/y coordinates. Time is used for cleaning and later time-of-day summaries.

Evidence

- ProjectA/results/cleaning_summary.csv
- ProjectA/results/analysis_summary.json
- ProjectA/config.json

## Slide 3

The DBSCAN grid contains 15 settings. KMeans tests K from 2 through 10. The inertia elbow rule selected K = 5 independently of the silhouette score. The methods favor different cluster shapes and cover different sets of points, so the comparison also uses the same 30,040-point DBSCAN non-noise support.

Evidence

- ProjectA/results/dbscan_experiments.csv
- ProjectA/results/kmeans_experiments.csv
- ProjectA/results/common_sample_comparison.csv

## Slide 4

The main DBSCAN setting produced 174 clusters, with 30,040 points in clusters and 23,746 labeled noise. The 500 m grid aggregates only non-noise points from the main analysis sample. Color represents log10(1 + point count). The figure omits personal identifiers and point-level tracks. Noise denotes low density under the model, with no verified abnormal-driving label.

Evidence

- ProjectA/results/point_labels.csv.gz (restricted original)
- Figures/A_grid500m.csv
- ProjectA/results/dbscan_experiments.csv

## Slide 5

Independent refitting reproduced every label and the ARI calculation. The ARI comes from one fixed 90% subsample. No multi-seed confidence interval was estimated. On shared support, silhouette scores were 0.108585 for DBSCAN and 0.427705 for KMeans. Silhouette measures compactness and does not establish the spatial relevance of density hotspots. The low-density rule flagged 115 trajectories, with no true abnormal-driving labels available.

Evidence

- ProjectA/results/analysis_summary.json
- ProjectA/results/common_sample_comparison.csv
- ProjectA/results/trajectory_candidates.csv (restricted original)
- Evidence/A_evidence.json

## Slide 6

The original course Project B training status remains blocked. All 578 windows have insufficient history. The NYC proxy is a separate task. The 99.165% nearest-road matching rate measures matching coverage, while classification accuracy requires labels. The full snapshot has 537,240 fixed windows, with 299,988 known and 237,252 unknown. The backtest has 80,536 known windows, a positive rate of 3.56%, and coverage of 61.63%. The source supplies official aggregate speed estimates. Two 10-minute low-speed estimates do not establish continuous minute-level low speed.

Evidence

- Evidence/Geolife_original_B/training_status.json
- Evidence/Geolife_original_B/pipeline_summary.json
- ProjectB/reference/evidence/cohort_final.json
- Course PDF, physical pages 11–15

## Slide 7

The [B, N, 104] input projects to 32 features. Two attention blocks use four heads with eight features per head. The score is QKᵀ/sqrt(8) + log(w). The mask sets nonphysical edges to negative infinity and transposes the source–target orientation so a target query sees predecessor keys. The dynamic edge MLP has dimensions 48 → 16 → 4, with 852 parameters per layer and 1,704 total. Its inputs include past speed ratios, trends, support, and missingness flags. Self-loops and missing key speeds use a multiplier of 1. Each pooling variant trains independently. A simulated decision at 07:30 predicts 07:40–07:50 using observations from 07:20–07:30 and earlier.

Evidence

- ProjectB/src/dynamic_model.py
- ProjectB/src/v10/v9/v8/v7/v7_prepare.py
- ProjectB/src/dynamic_train.py
- ProjectB/reference/evidence/graph.json

## Slide 8

Three seeds were used for each dynamic pooling model. Static AP/F1 was 0.65526/0.64552 for Mean, 0.65914/0.65514 for Max, and 0.66382/0.63880 for Attention. Dynamic AP/F1 was 0.66644/0.65005, 0.65913/0.64398, and 0.66866/0.65408, respectively. AP uses average_precision_score. Dynamic Max lost F1. Development mean AP selected Dynamic Mean before the test. The higher test Attention result does not change that selection. All models use the same test support. Unknown labels are excluded from primary evaluation, with all difficult known windows retained.

Evidence

- ProjectB/reference/results/summary.json
- ProjectB/reference/results/development_metrics.csv
- ProjectB/reference/results/dynamic_selection.json
- Figures/B_plot_values.csv

## Slide 9

This ranking comparison uses the same true-positive count at 60% recall on the full known support. Threshold selection uses a separate calibration split. Mean total false positives declined from 15.3778 to 14.7481 per day and recovery false positives declined from 7.8148 to 7.5704 per day. All three seeds moved in the same direction, though the recovery interval from 1,000 date-block bootstrap resamples crosses zero. The frozen operating threshold maximized F1 subject to the calibration constraints. Average backtest false positives still exceed both budgets. NYC test dates were already public, so the backtest lacks a new independent holdout.

Evidence

- ProjectB/reference/results/matched_recall.csv
- ProjectB/reference/results/paired_date_blocks.csv
- ProjectB/reference/results/operating_points.csv

## Slide 10

Project A contributes interpretable spatial analysis of hotspots, noise, and candidates. Supervised classification accuracy requires labels that were unavailable. Project B implements a Transformer constrained by the actual directed graph with dynamic edges. Thresholds were frozen after independent calibration. Results cover three pooling comparisons and false alarm limits. The release excludes plaintext individual trajectories and pickle files. A full Project A rerun requires an authorized original archive. Project B ships the public snapshot, arrays, and weights. The original Geolife requirements for user count, years, and five-minute persistence remain distinct from the supplemental NYC backtest.

Evidence

- Evidence/A_evidence.json
- ProjectA/evidence/
- ProjectB/local_run_verification.json
- README_submission.md

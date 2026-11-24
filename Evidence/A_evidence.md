# Project A Evidence Review

Review date: November 17, 2026

## Findings

The original project was read-only. The review read CSV.GZ files without loading pickles or modifying source, data, or results. Independent full-sample DBSCAN and K-means refits reproduced the labels in `point_labels.csv.gz` value for value, and the report's key numbers matched the CSVs. All nine original engineering tests passed. See `A_evidence.json`, `A_tests.log`, and `A_tests_receipt.json`.

The original `checksums.sha256` listed 56 files. Fifty-three existed, three under `references/` were missing, and 14 current hashes differed. `ProjectA.zip` also lacked `references/`. The public submission uses its own manifest and checksum list, with the previous audit outcome retained as historical evidence.

## Coursework Requirements

Physical pages 9–10 of `G:/course-jiaotongshujuwajue/ppt/Lecture1-logistics-DM2026V3-part1.pdf` cover environment setup, Geolife loading, cleaning and features, DBSCAN sensitivity, a K-means comparison, hotspots across subjects, unusual trajectories, scatterplots and heatmaps, and the report. The optional temporal analysis was completed. Road-network matching and detection of parking or congestion events were outside the implementation.

Page 8 gives 30% for assignments, 60% for the group project (20% each for A, B, and the presentation), and 10% for quizzes and participation. Project A has no Project B requirement for 100 users in 2010–2012. The course slides use “vehicles,” but Geolife contains mixed-mode trajectories. See the English slide extraction in `course_A_requirements_pages9_10.txt`.

## Data and Workflow

- Original ZIP: `G:/course-jiaotongshujuwajue/dataset/Geolife Trajectories 1.3.zip`, 353,488,286 bytes. Its SHA-256, `16c2caed998e80828acd5fe7b15404593d1752f2a1524505633c23fdfac8a066`, matches the recorded metadata.
- The ZIP inventory has 182 users, 18,670 trajectory files, and 24,876,978 points. Sixty-nine users have mode label files. Project A did not use them for supervised classification.
- The Beijing rectangle spans longitude 116.1–116.7° and latitude 39.7–40.2°. The cleaning interval covers 2007-04-01 up to but excluding 2012-09-01. The actual sample covers 2007-09-03 09:52:08 to 2011-12-27 05:18:00 UTC.
- The first 30 users by ID with at least 1,000 original regional points were eligible. For each user, rank trajectory files with points in the region by the hash of the seed and ZIP member name, then retain up to 40. This selected 1,108 files and 2,029,261 original points.
- Per-trajectory cleaning removed invalid coordinates or times, exact duplicates, conflicting timestamps, and both endpoints of jumps with adjacent speed over 100 m/s. Speed edges did not cross rejected points or gaps over 300 seconds. Cleaning left 2,021,748 points and region filtering left 1,691,926.
- Keeping the first point per trajectory-minute left 101,019 representative points. WGS84 coordinates were projected to EPSG:32650 in meters. A fixed seed capped each user at 2,000 points, giving 53,786 points, 30 users, and 1,102 trajectories.
- Clustering uses projected x and y. Time, speed, and elevation support preprocessing, diagnostics, and optional time-period summaries.
- Unsupervised evaluation uses the same fixed sample throughout. There is no supervised split or classification accuracy, AP, or F1 measure.

Original source locations include `src/data.py:67` for cleaning, `:97` for selection and sampling, and `:108` for projection. In `src/analysis.py`, `:33` starts the analysis, `:38` defines x/y, `:48` runs the DBSCAN grid, `:70` covers K-means, `:89` compares shared support, `:97` covers personal hotspots, and `:142` runs the perturbation check. See `run.py:22` for the historical entry point and `src/verify.py:7` for its consistency check.

## Recomputed Results

| Item | Verified result |
| --- | --- |
| Main DBSCAN | eps = 50 m, min_samples = 20, 174 clusters |
| DBSCAN non-noise support | 30,040 points, 55.8509649% coverage |
| DBSCAN noise | 23,746 points, 44.1490351% noise |
| Main DBSCAN silhouette | 0.10858512065142913, on 2,000 sampled non-noise points |
| K-means | K = 5 by the geometric elbow |
| K-means full-support silhouette | 0.3652873096107069, on 2,000 of the 53,786 points |
| K-means common-support silhouette | 0.42770490260565147, on 2,000 of the 30,040 DBSCAN non-noise points |
| 90% perturbation | 48,407 points, ARI = 0.9063785102960764 including noise, 155 clusters, 46.5986% noise |
| Low-density candidates | 115 trajectories with at least 20 analysis points and at least 80% noise. No verified anomalous-driving label. |
| Independent refits | Full-sample DBSCAN and K-means K = 5 labels matched value for value |
| Engineering tests | 9 passed, exit code 0 |

DBSCAN used 15 combinations of eps = [30, 50, 100, 200, 500] m and min_samples = [5, 10, 20]. K-means evaluated K = 2–10. Silhouette scores measure internal cluster structure and reflect different methods and cluster counts. The fair comparison uses 30,040 common-support points with an internal 2,000-point sample. The all-point K-means score of 0.365 and non-noise DBSCAN score of 0.109 use different supports.

At fixed min_samples = 20, increasing eps from 30 to 500 m reduced noise from 60.8002% to 4.51976%. The grid minimum of 1.1806% uses eps = 500 and min_samples = 5. The initial nine configurations were expanded with 30 m and 50 m cases during development. The final choice was made after that exploration.

## Presentation Results

1. The 53,786 fixed analysis points form 174 density clusters, covering 55.85% with 44.15% noise. A coarse-grid heatmap and cluster sizes show the scale.
2. The noise curve at fixed min_samples shows parameter sensitivity. On the common 30,040-point support, silhouettes are 0.109 for DBSCAN and 0.428 for K-means. The methods have different objectives and cluster counts.
3. The 115 low-density candidates and the perturbation ARI of 0.906 are exploratory results. Individual routes and space-time identifiers are unnecessary for showing them.

## Figure and Privacy Review

The original directory `G:/course-jiaotongshujuwajue/ProjectA/results/figures/` contains 11 plots numbered 01–11.

- `02_cleaning.png`, `04_sensitivity.png`, `09_diagnostics.png`, and `11_time.png` show counts, curves, and distributions without individual space-time tracks.
- `05_heatmap.png` aggregates over 500 m cells and has no user or time linkage. The published design uses aggregate source tables.
- Keep `07_users.png` and `08_cases.png` private because they show individual hotspots, trajectory IDs, and paths.
- `01_region.png`, `03_dbscan.png`, `06_comparison.png`, and `10_parameter_examples.png` contain precise point scatterplots. Use the 500 m map or cluster sizes for public presentation.
- The original hotspot CSV has precise centers and bounds, including single-user clusters. Remove `lon`, `lat`, `x_min`, `x_max`, `y_min`, and `y_max` for public tables or use spatial statistics with at least three users. The `users` field is a count of distinct users.
- For `user_comparison.csv`, anonymize user labels and retain counts and proportions. For `trajectory_candidates.csv`, publish the noise distribution, candidate totals, and anonymous counts, excluding trajectory IDs, times, and spatial bounds.
- The original ZIP, point-level CSV.GZ files, `quality_issues.csv.gz`, `raw_to_sample.csv.gz`, labels, and personal time series stay private. The code, configuration, and external read-only data instructions support authorized reproduction.

The historical plots used YaHei and default Latin fonts. The original deck's new design specified SimSun and Times New Roman and regenerated its white, black, blue, and red figures from verified CSVs.

## Entry-Point Scope

The original `python run.py --data <Geolife.zip> --out <new-directory>` rebuilds the pipeline. Its default output was `results/`, so the public wrapper uses a fresh output path.

The original `run.py --stage verify` still performs a stat and hash of the default raw ZIP and `src.verify` requires sensitive point-level files. The public instructor entry point checks aggregate tables and files separately. The original nine engineering tests were also run in `niulaoshi` without the ZIP.

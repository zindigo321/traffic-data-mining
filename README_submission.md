# Final Traffic Data Mining Submission Guide

Submission date: November 17, 2026

The package includes source, configuration, measured results, figures, a combined report, and a 10-slide presentation for Project A and the Project B rerun. The deck uses a white background, black type, and blue and red accents. The PDF is for quick viewing. The speaker notes follow the slide order.

## 1. Package Navigation

```text
FinalCourse_ProjectAB/
  README_submission.md           This guide
  01_presentation/              10-slide PPTX, PDF, and speaker notes
  ProjectA/                      Privacy-safe aggregate submission, code, config, results, and evidence
  ProjectB/                      Runnable package, public data, weights, results, and logs
  Figures/                       PNG and SVG figures and plotting CSVs
  Report/experiment_report.md
  Evidence/                      Verification, coursework requirements, provenance, and acceptance scope
  manifest.csv
  SHA256SUMS
```

Start with the [presentation](01_presentation/) and then read the [combined report](Report/experiment_report.md). The two project directories contain the supporting methods, measurements, and run evidence.

On a machine with Conda configured, run `verify_submission.cmd` to verify all package hashes and run both default project checks using `niulaoshi`. The equivalent command is `python verify_submission.py --run`. Use `python verify_submission.py` for hashes only. Results go into a fresh `runs/` directory.

## 2. Quick Verification

Run these commands from the extracted package root in Anaconda Prompt.

```bat
conda activate niulaoshi
python ProjectA/run.py
python ProjectB/run.py
```

The default entry points verify Project A's aggregate results and Project B's per-example metrics. Output goes into a new directory under each project's `runs/`, with command, duration, and exit-code records. The existing Conda environment can be used directly.

For further checks, run the following.

```bat
python ProjectA/run.py tests
python ProjectB/run.py tests
python ProjectB/run.py replay
python ProjectB/run.py predict
```

Project B includes public data, prepared arrays, and weights. Replay loads the models and replays a sample. Prediction performs a forward pass and exports predictions on the fixed historical test windows.

## 3. Full Reruns from Data

The Project A package excludes the original ZIP and point-level files to protect precise individual trajectories. Obtain Geolife Trajectories 1.3 legitimately and choose a new output directory.

```bat
python ProjectA/run.py full --data "D:\data\Geolife Trajectories 1.3.zip" --out "D:\CourseRuns\A_new"
```

The original ZIP used for byte-for-byte verification has SHA-256 `16c2caed998e80828acd5fe7b15404593d1752f2a1524505633c23fdfac8a066`.

Project B can run using the frozen public snapshot in the package.

```bat
python ProjectB/run.py prepare
python ProjectB/run.py train --output "D:\CourseRuns\B_new"
```

Full Project B training fits 18 dynamic models across three pooling methods, three seeds, and two training stages, development and final. It separately verifies and reuses 18 static controls. The submission packaging preserves the training configuration and evaluation protocol. The commands use frozen dates and the existing dependencies. Full-matrix runtime depends on CPU load. See the project logs for measured resource use.

## 4. Main Results and Interpretation

- Project A used 30 users, 1,102 trajectories, and 53,786 analysis points. The main DBSCAN configuration, 50 m and 20 points, produced 174 clusters, 55.85% coverage, and 44.15% noise. The geometric elbow selected K = 5. Silhouettes were approximately 0.109 and 0.428 on the common support of 30,040 non-noise DBSCAN points. A single 90% perturbation produced ARI of approximately 0.906. These are internal measures. The 115 low-density candidates have no verified anomalous-driving labels.
- The original Project B Geolife pipeline labeled all 578 windows unknown, blocking formal training. The 99.165% figure is a nearest-road match rate. The rerun uses public NYC segment speed estimates, a strict 10-minute lead time, and a directed Graph Transformer with dynamic edge weights. Development-selected Mean achieved AP of 0.6664 ± 0.0243 and positive-class F1 of 0.6501 ± 0.0195 on 80,536 known test windows, reported as the mean ± sample SD across three seeds.
- Total and recovery false positives were 14.600 and 7.504 per day, above the respective budgets of 12 and 6. The disclosed dates constitute a historical backtest. The original 100-user, 2010–2012, Beijing road network, and five-minute continuous-congestion requirements remain separate from the NYC task.

The combined and project reports record data cutoffs, protocol differences, degradations across pooling methods, and statistical limitations.

## 5. Privacy, Integrity, and Acceptance Scope

Project A supplies aggregate statistics and 500 m plots. The public package excludes precise individual points, trajectories, point-level labels, links between times and identities, and pickle caches. A full point-level rerun requires authorized local data. Project B uses official public segment-level speed estimates. The package contains no credentials or Conda environment binaries.

The original Project A manifest had three missing entries and 14 hash mismatches. The audit records these and verifies the current point table by refitting the models. The root manifest and checksum list cover the actual submission files. `SHA256SUMS` excludes itself. ZIP hashes and fresh-extraction acceptance receipts are external to the ZIP or supplied with the delivery notes.

The original nine Project A engineering tests passed, and independent refits reproduced DBSCAN and K-means labels on the current point table. The packaged instructor entry point checks aggregate consistency. Point-level silhouette or ARI recomputation requires the underlying data. Project B's prior full-run acceptance covered 81 tests, zero probability differences in the replay of 36 models, byte-identical arrays rebuilt from raw CSVs, and reproduction of weights and metrics for 18 dynamic retrains. See the actual logs and receipts for each package verification.

## Package Acceptance on November 17, 2026

The candidate ZIP was extracted into a fresh directory and tested in the existing `niulaoshi` environment. The root hash check, both default verification entries, Project A's nine tests, Project B's 81 tests, and replay of 36 models passed. See [fresh-directory acceptance](Evidence/fresh_directory_acceptance/) for the original commands, exit codes, timings, environment, and console output. After those logs were collected, that delivery added evidence and updated instructions while preserving its code, data, models, figures, and deck. The final ZIP had a separate integrity check and a delivery checksum file alongside it.

# Traffic Data Mining Coursework and Projects

This repository contains weekly assignments, Project A, the Project B rerun, and the final presentation. The archive was assembled on November 17, 2026. Data cutoffs and experiment dates are recorded in the supporting evidence.

| Directory | Contents |
| --- | --- |
| [Assignments](assignments/) | Week 1–3 PDFs and editable Week 3 notes |
| [Project A](ProjectA/) | Geolife trajectory cleaning, DBSCAN and K-means hotspot discovery, and low-density candidate analysis |
| [Project B](ProjectB/) | A 10-minute-ahead congestion proxy using a Graph Transformer with dynamic edge weights, three pooling methods, and three seeds |
| [Final presentation](01_presentation/) | A 10-slide deck, PDF preview, and speaker notes |
| [Experiment report](Report/experiment_report.md) | Methods, results, and how they address the course requirements |
| [Figures](Figures/) | Result plots and their source values |
| [Evidence](Evidence/) | Provenance, data contracts, model and metric checks, and execution logs |
| [Archive](archive/legacy-presentation/) | The original defense materials for historical context |

## Main Results

- Project A used 30 users and 53,786 analysis points. DBSCAN at 50 m and 20 points produced 174 clusters, with 55.85% coverage and 44.15% noise. The elbow rule selected K = 5 for K-means. Internal clustering metrics and adjusted Rand index (ARI) measure clustering properties. The 115 low-density candidates lack anomalous-driving ground truth.
- All 578 windows from the original Project B Geolife pipeline were labeled unknown, blocking supervised training. The rerun uses public NYC segment speed estimates for a separate proxy task. The original course requirements for 100 users, 2010–2012 data, and five continuous minutes of congestion remain unmet.
- The dynamic Mean model, selected on development data, achieved AP of 0.6664 ± 0.0243 and positive-class F1 of 0.6501 ± 0.0195 across three seeds on 80,536 known NYC historical test windows. Total and recovery false positives were 14.600 and 7.504 per day, above the respective budgets of 12 and 6. This evaluation is a backtest on previously disclosed dates.

## Run the Checks

From the repository root, use the existing `niulaoshi` environment.

```bat
conda activate niulaoshi
python verify_submission.py --run
python ProjectA/run.py tests
python ProjectB/run.py tests
python ProjectB/run.py replay
```

Runs create fresh output directories under `runs/`. Project A verifies aggregate results by default. A full Project A rerun requires a legitimately obtained copy of Geolife Trajectories 1.3.

```bat
python ProjectA/run.py full --data "D:\data\Geolife Trajectories 1.3.zip" --out "D:\CourseRuns\A_new"
python ProjectB/run.py prepare
python ProjectB/run.py train --output "D:\CourseRuns\B_new"
```

Project B uses the included frozen public NYC snapshot, prepared arrays, and model weights. The training command fits 18 dynamic models and verifies and reuses 18 static controls. See each project’s requirements file for dependencies.

## Public Scope and Integrity

The public copy excludes private Geolife tracks, individual precise-point tables, pickle files, instructor slides, and the original course archives. See the [Project B data sources and usage notes](ProjectB/data_sources_and_usage.md) for the public segment data, snapshot, transformations, and terms. Publication does not transfer rights in source or third-party material. This repository grants no license on behalf of other rights holders. `PrivacyReview.json` records the removal of bylines, student IDs, personal author properties, account names, and local user directory identifiers.

`manifest.csv` and `SHA256SUMS` cover the current anonymous public copy. Project B also has its own integrity manifest. Historical records use anonymized path mappings. Their original-byte hashes document the historical artifacts, while the current manifests identify the files in this revision. `.gitattributes` preserves file bytes by disabling line-ending conversion. Write new runs under `runs/` to preserve the reference results.

The English edition translates public filenames, captions, and report text. The [Project B source identity receipt](ProjectB/reference/evidence/english_source_identity.json) links the English verifier source hashes to the original experiment freeze. The [original archive revision](https://github.com/zindigo321/traffic-data-mining/tree/8bea3c796b3fec43db25a6694dbbf83626f01e12) provides the original spellings and bytes for historically cited artifacts.

See the [submission guide](README_submission.md) for package navigation and verification and the [GitHub publishing notes](docs/GITHUB_UPLOAD.md) for the original publication workflow.

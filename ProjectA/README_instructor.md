# Project A Instructor Submission

Project A discovers travel hotspots in Geolife. The code and clustering results come from the original project. This public copy includes aggregate results and privacy-safe figures. Individual points and precise trajectories are excluded.

Use the existing `niulaoshi` environment. From this directory, run:

```bat
conda activate niulaoshi
python run.py
python run.py tests
```

The default offline check works without the original ZIP. It verifies aggregate totals, coverage and noise, the K-means elbow, common support, density bias, time shares, figure files, and privacy-related columns. A full rerun recomputes the point-level analysis.

To rerun from the raw data, obtain Geolife Trajectories 1.3 legitimately and provide a new output directory.

```bat
python run.py full --data "D:\data\Geolife Trajectories 1.3.zip" --out "D:\ProjectA_runs\new_run"
```

The original ZIP remains read-only. The full rerun produces local point-level verification outputs, which should be treated as private. `run_original.py` retains the historical workflow and its old default paths. Use `run.py full` to specify the actual input explicitly.

Start with [the experiment report](results/experiment_report.md), then consult `results/analysis_summary.json`, `results/dbscan_experiments.csv`, `results/common_sample_comparison.csv`, and `evidence/`. The sample has 53,786 points, 174 DBSCAN clusters, five K-means clusters, and 115 low-density candidates. ARI and silhouette scores measure clustering properties.

`evidence/original_checksums.sha256` records the original checksums. Some original entries are absent or differ from the current files. The root submission manifest verifies the actual public files. `evidence/source_provenance.json` records source hashes and anonymization transformations.

# Project B Graph Transformer Local Runtime Package

This package contains real public data, the complete Python source, preprocessors, model weights, and reference results. The directed road-segment Graph Transformer supports Mean, Max, and Attention pooling with dynamic edge weights. CPU execution is the default. The launcher runs offline in the existing `niulaoshi` Conda environment. Extract the package to any writable path, including paths with spaces.

## Quick Start

In Anaconda Prompt, enter the extracted `ProjectB_GT_niulaoshi` folder.

```bat
conda activate niulaoshi
python run.py check
python run.py
```

`python run.py` independently recomputes the reference results by default and reports `Exit: 0` on success. Each run writes its output to a new `runs/<timestamp>_<command>/` directory containing logs and an exit code. `02_verify.cmd` uses the active environment when available and otherwise invokes `conda run -n niulaoshi`. The interpreter used in local verification was `D:\conda_envs\niulaoshi\python.exe`.

```bat
D:\conda_envs\niulaoshi\python.exe -B -X utf8 run.py replay
```

In PyCharm or VS Code, select the `niulaoshi` interpreter and launch `run.py` at the package root. Leave the argument empty to recompute the metrics or pass `train` to run the full retraining protocol. The launcher resolves input paths relative to its source file, independent of the IDE working directory.

## Commands

| Command | Purpose |
| --- | --- |
| `python run.py check` | Check the environment, package manifest, 54 raw shards, road geometry, and both prepared arrays. |
| `python run.py verify` | Independently recompute AP, F1, confusion counts, subgroup metrics, matched-recall points, and thresholds from per-example scores. |
| `python run.py replay` | Sample and replay all 36 models with the bundled data, preprocessors, and weights loaded with `weights_only=True`. |
| `python run.py tests` | Run 75 model and dynamic-edge engineering tests plus 6 portability, environment, and integrity tests. |
| `python run.py smoke` | Run backward passes, optimization, save, and load for all three pooling methods on 32 real training examples for engineering verification. |
| `python run.py prepare` | Rebuild the graph, labels, and features using the bundled 54 raw CSVs and road geometry, then check the array hashes. |
| `python run.py predict --pool mean --seed 11` | Export frozen-model predictions for all 130,680 historical test windows, including windows with unknown labels. |
| `python run.py train` | Reproduce all 18 dynamic training runs across three pooling methods, three seeds, and the development and final splits. Reuse the 18 verified static controls. |

The full protocol runs for 12 epochs with batch size 128, 2 threads per process, at most 4 concurrent jobs, and seeds 11, 22, and 33. Smoke runs provide engineering evidence only. Runtime depends on CPU performance and system load. See the actual timings in the run logs. Allow at least 8 GB of available memory and 2 GB of additional disk space for new outputs.

Choose a separate output directory.

```bat
python run.py train --output "D:\ProjectB_runs\dynamic_new"
```

For a staged run, use the same output directory for every phase. Run `initialize` once with a new directory.

```bat
python run.py train --phase initialize --output "D:\ProjectB_runs\stage_run"
python run.py train --phase tests --output "D:\ProjectB_runs\stage_run"
python run.py train --phase development --output "D:\ProjectB_runs\stage_run"
python run.py train --phase select --output "D:\ProjectB_runs\stage_run"
python run.py train --phase final --output "D:\ProjectB_runs\stage_run"
python run.py train --phase freeze --output "D:\ProjectB_runs\stage_run"
python run.py train --phase evaluate --output "D:\ProjectB_runs\stage_run"
python run.py train --phase analysis --output "D:\ProjectB_runs\stage_run"
python run.py train --phase verify --output "D:\ProjectB_runs\stage_run"
python run.py train --phase report --output "D:\ProjectB_runs\stage_run"
```

The output directory receives new results. The launcher protects `reference/`, `data/`, and `src/`. If a phase fails, check `logs/` for the full exception and exit code. Partial training results are preserved. Once the model and thresholds are frozen, training cannot continue in that run. The launcher checks the manifest before executing, so changes to source or configuration cause the integrity check to fail. If you change the model, labels, or hyperparameters, use a separate project, register the revised protocol, update the package manifest, and retrain the equivalent static controls. The bundled workflow uses the frozen configuration.

## Files

- `run.py` is the unified entry point. `run_niulaoshi.cmd` launches it through Conda.
- `src/config.json` defines the original data, graph, labels, and training hyperparameters. `src/dynamic_model.py` implements dynamic edge weights. `src/dynamic_train.py` and `src/dynamic_score.py` handle training and scoring. `src/portable_tasks.py` resolves packaged inputs.
- `data/raw/` holds 54 official NYC aggregate speed shards. `data/geometry/roads.csv` contains road geometry. JSON records under `data/` contain the original queries, versions, timestamps, and hashes.
- `data/prepared/development.npz` and `data/prepared/final.npz` contain the full real datasets and load with `allow_pickle=False`.
- `reference/` contains the dynamic edge-weight audit records, including freeze receipts, per-example predictions, 36 dynamic and static models, logs, figures, and reports. Historical absolute paths document provenance. The portable launcher resolves data paths from the package.
- `requirements-lock.txt` and `requirements.txt` record the verified dependencies. The existing `niulaoshi` environment already contains them.
- `local_run_verification.json` records the acceptance scope. `runtime_adaptation_notes.md` describes changes from the earlier package. `manifest.csv` and `SHA256SUMS` record package integrity.

The statements in `reference/` about the absence of a second full retraining describe the earlier audit. The portable package was separately used to perform 18 full training runs in a fresh directory with spaces. See `local_run_verification.json` and `validation_logs/` for the actual execution, hashes, and results. The supplemental `reference/evidence/english_source_identity.json` binds the English source hashes to the original threshold-freeze receipt while preserving the original recorded hashes.

Short numerical log filenames avoid Windows path-length limits. Each `validation_logs/` subdirectory has a `_name_map.json` mapping them to historical filenames. The recorded commands document the original executions.

## Data and Results

The frozen NYC i4gi-tjb9 dataset covers 121 directed road segments in 10-minute bins during 07:00–09:00 and 17:00–19:00 local time. The latest dynamic features use bin t−2. Speeds from t−1 and t are reserved for labels. The past speed baseline, two-bin 40% proxy, known/unknown classes, and temporal splits follow the registered protocol. All known windows are evaluated. Unknown windows receive predictions and are excluded from supervised evaluation. The last source timestamp is 2026-11-11T17:12:06.

The reference test has 80,536 known windows and 2,866 positives. Mean AP/F1 across the three seeds was 0.6664/0.6501 for dynamic Mean, 0.6591/0.6440 for dynamic Max, and 0.6687/0.6541 for dynamic Attention. Mean was preselected on development. The reference run exceeded the false-positive budgets. These results are a historical backtest on already published dates. An independent blind test remains outstanding. The Chicago and Geolife tasks remain blocked.

The package runs in Conda without requiring the original G: project path, old task archives, API keys, or network access. The ZIP contains project files and dependency specifications.

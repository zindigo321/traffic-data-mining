# Runtime Adaptation Notes

The original audit archive is recorded in `reference/`. Its model and threshold freeze receipts identify the historical experiment. The English public copy has updated manifests and checksums for localized documents and source files.

`src/` contains the portable working copy. The recorded computational adaptation is in `full_data.py`. Once `PROJECTB_PREPARED_DIR` is set, the loader uses the two bundled arrays and verifies the same bytes against the hashes in the historical `prepared.json`. It never falls back to a legacy drive path in portable mode. Graphs, labels, features, models, loss, pooling, and thresholds continue to use the registered computation. See `full_data.diff` for the original adaptation.

`run.py`, `portable_tasks.py`, `sitecustomize.py`, and the CMD launchers provide portable execution. Data reconstruction calls the original `v5_graph`, `v5_prepare.grid`, and `full_prepare.construct/stats` routines. It redirects `source_files` to the bundled CSVs with matching hashes. Training initialization copies the hashed static controls from `reference/`, registers the bundled snapshot and relative paths, then runs the existing dynamic training, selection, freeze, and evaluation workflow. The 18 new dynamic training runs and 18 reused static controls are counted separately.

Every command uses the active `niulaoshi` Python interpreter. `sitecustomize` applies through the child process's `PYTHONPATH` and rejects access to legacy task paths. Extraction paths may contain spaces or use a different drive. Arrays load with `allow_pickle=False`, and model weights load with `weights_only=True`.

Metric recomputation, model replay, and retraining are separate commands. See `local_run_verification.json` for the verified scope. A smoke run supplies only engineering evidence.

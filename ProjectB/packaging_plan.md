# Portable Runtime Package Plan

Package the completed dynamic edge-weight experiment as a relocatable Python project ZIP that runs in the existing `niulaoshi` Conda environment. Protect the audit archive and existing runs.

Include the full model, graph, feature, training, and evaluation source, the fixed public NYC snapshot with 54 shards and road geometry, the two verified prepared arrays, model weights, preprocessors, reference results, relative path configuration, Conda entry points, dependencies, and runtime documentation. Keep source and usage terms with the data.

Provide `check`, `verify`, `replay`, `tests`, `smoke`, `prepare`, `predict`, and `train`. The default `verify` independently recomputes established metrics. `replay` loads saved models safely. `smoke` runs the three pooling methods on a real training mini-batch for engineering verification. `prepare` rebuilds the bundled snapshot. `train` runs the registered 18-run dynamic training matrix and reuses the matching 18 verified static models. Write new results under `runs/` or another fresh directory.

Limit the portability adaptation to data lookup, bundled input verification, and startup. Retain the graph construction, labels, features, dynamic edge weights, loss, split, pooling, model parameters, and scoring algorithms from the registered experiment. Track relative path adaptations separately from the historical freeze receipts.

Acceptance covers the actual `niulaoshi` interpreter and dependencies, extraction to a fresh path with spaces, manifest and hash checks, operation independently of the original task paths, 75 engineering tests, metric recomputation, sampled replay of 36 models, raw data reconstruction, real three-pooling mini-batch forward and backward passes and save/load, and full test prediction export. Verify the complete development and final training matrix under the original protocol before claiming full retraining. Record commands, timings, and exit codes. The package may run offline using an existing Conda installation.

Estimated sizes were 76 MB for the original dynamic audit ZIP, 36 MB for prepared arrays, and 55 MB for compressible raw CSVs. The package excludes the original large private trajectory archive. Results remain a historical NYC backtest. A separate blind test and the original Chicago protocol still await their own evidence.

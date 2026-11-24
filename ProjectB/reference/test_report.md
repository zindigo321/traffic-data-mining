# Testing and Review Report

All 75 engineering tests passed. The root suite ran 19 tests, seven original and 12 new dynamic-edge tests, and inherited suites ran 56. See `logs/*engineering_dynamic*.json` and the corresponding logs for actual commands, standard output, exit codes, and timings. The synthetic fixtures provide engineering evidence only. See `synthetic/engineering_status.json`.

New tests cover zero log edge weights equaling the static output across three poolings, directed edge coordinates, self edges, padding, bounds, and missing-endpoint fallback to 1. They also cover output changes from nonunit weights and structural masks, identical isolated-graph output, batch isolation and node permutation and center mapping for all poolings, rejection of empty graphs, invalid centers, and NaN, backward passes and safe weights-only save and load for all poolings, single-batch overfitting, and feature, weight, and score invariance when label bins and future observations are changed, including at neighbors.

Real-data verification recomputed 72 operating points, 288 matched-recall points, and 216 subgroup metrics. Sampled replay passed for 36 models with a maximum probability difference of 0.0. Same-support training IDs matched exactly, and unique known examples and updates per epoch were consistent. Unknown windows were excluded from supervised evaluation. Static summaries matched the original audit. See `evidence/verification_dynamic_replay.json`.

The static review covered security, compatibility, error handling, resource management, change scope, types, and shapes. Dynamic modules were added alongside the source for the baseline. Replay checks hashes and loads weights safely. Float32 logits and probabilities are promoted exactly to float64 scores. See `reports/change_notes.md`. Python assertions were enabled for the tests.

This audit did not perform a second full retraining in its clean extraction directory, a new blind future test, or new Chicago or Geolife training. The historical extraction verification receipt records the scope of its metric recomputation and sampled replay. The historical backtest alone cannot establish stable gains on new dates.

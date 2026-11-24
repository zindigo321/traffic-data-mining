# Local Runtime Acceptance in niulaoshi

The verified workflow used the actual `niulaoshi` interpreter in a fresh directory with spaces. It covered the CMD and Conda launchers, raw input reconstruction, metric recomputation, 36-model replay, real mini-batch training for all three pooling methods, tests, and full retraining. The runs used the existing environment.

Full retraining completed 18 dynamic model fits at 12 epochs each and reused 18 verified static controls. All new weight hashes matched the reference experiment, and all aggregate metrics matched value for value. The logs identify the full training runs separately from the smoke runs.

All 81 tests passed, comprising 75 model engineering tests and 6 portability tests. Both prepared arrays matched their original hashes after reconstruction from the bundled inputs. The maximum probability difference across the 36-model replay was zero. Prediction export included 130,680 test windows, including 50,144 with unknown labels.

See `local_run_verification.json` and `validation_logs/` for commands, interpreters, exit codes, timings, and console logs. The final ZIP extraction receipt belongs outside the ZIP to avoid self-reference.

The portability change preserves the registered model and its historical results. The test dates were already public. Independent evaluation on new dates remains outstanding.

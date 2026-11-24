# Changes from the Original Full Experiment

The original package SHA-256 was `0436558b017bb6c5d52d0ab2103ae52f2315b970e4a9b9cf615c7b4a9f0c1c47`. The original project and audit package were read-only. Implementation used an independent copy of the source.

Added `dynamic_model.py`, `dynamic_check.py`, `dynamic_train.py`, `dynamic_score.py`, `dynamic_analysis.py`, `dynamic_verify.py`, `dynamic_report.py`, `run_dynamic.py`, `finish_dynamic.py`, and `tests/test_dynamic.py`. The configuration registers one dynamic-edge recipe with a three-pooling matrix. The baseline model, features, graph, and labels retained the original implementation. Each layer adds 852 parameters and uses the existing feature cutoffs and roads.

Copied 18 static graph models for both stages from the original package. Their hashes are recorded in `evidence/registration_dynamic.json`, and all were safely replayed on sampled inputs. Existing center-only and other baseline aggregates remain in `evidence/static_reference_summary.json` for historical same-support reference. This run added 18 dynamic fits.

After training began, a one-time NumPy load in `dynamic_train.py` was wrapped in a context manager to release its file handle. Computation, sampling, models, and configuration stayed within the registered protocol. The new `--no-replay` option supports offline metric recomputation from bundled predictions when external arrays are unavailable.

The initial manual launch failed while resolving its relative working directory, before the check or training phases began. Retrying from the absolute project parent succeeded. The recorded commands capture the actual check and subsequent phases.

# Project B Paired Dynamic Edge Weight Experiment Plan

Registered on 2026-11-12 before implementation or execution. The existing full experiment and audit archive were read-only inputs.

## Evidence and Scope

The original evidence is the full Project B Graph Transformer audit archive dated 2026-11-12, with expected SHA-256 `0436558b017bb6c5d52d0ab2103ae52f2315b970e4a9b9cf615c7b4a9f0c1c47`. Before implementation, check the ZIP, manifest, checksum list, both prepared arrays, and static model weights. Use the fixed official NYC i4gi-tjb9 snapshot with 121 directed road segments, the original peak periods, and 10-minute bins. The original proxy label requires two consecutive bins below 40% of their respective historical mean speeds. Supervised learning uses all known windows. Unknown windows remain in coverage records.

This is the authorized full temporal backtest on dates whose existing test scores are already public. The original Geolife and Chicago tasks remain blocked. Use the existing data, labels, and availability cutoffs.

## Fixed Dates and Budget

- Development training is [2026-05-11, 2026-08-03), and development validation is [2026-08-03, 2026-08-31).
- Final training is [2026-05-11, 2026-08-31), independent calibration is [2026-08-31, 2026-09-28), and historical testing is [2026-09-28, 2026-11-12). The right endpoints are excluded.
- Use the verified CPU interpreter `D:/conda_envs/niulaoshi/python.exe` with 2 threads per process and up to 4 training processes.
- Features are 52 legal values plus 52 missingness indicators. The latest dynamic input is bin t−2. Speeds in t−1 and t are label inputs and stay out of the edge weights. Fit preprocessing on training only and freeze historical features for evaluation.
- Run 12 epochs with batch size 128, Adam learning rate 0.001, width 32, 4 heads, and 2 layers. Use the same full training IDs and shuffle order. Seeds are 11, 22, and 33.

## Dynamic Recipe and Paired Matrix

Keep the static adjacency constraints. For a physical edge from source to target, concatenate the endpoints' past relative speed, trend, coverage, and missingness descriptors in order. A small MLP in each layer produces a bounded log edge weight for each attention head, added to the layer's QK attention score. The self-edge weight is 1. If the latest relative speed is missing, the edge weight falls back to 1. The mask excludes nonphysical edges. Weights range from 0.5 to 2 and initialize at 1. The recipe uses fixed hyperparameters and asserts the selected columns against the existing feature dictionary.

Train dynamic Mean, Max, and Attention pooling for three seeds each in the development and final phases, for 18 new fits. The two stages of the three completed static pooling methods supply 18 matched controls on the same support and update budget. Verify hashes and replay those controls. Center-only, logistic regression, lag-rule, and majority-class results remain historical same-support comparisons. Count reused models separately from new fits.

Select each development checkpoint by development AP. For final training, use the median of the three development checkpoints for that pooling method, execute all 12 epochs, and retain weights from the fixed epoch. Select the primary dynamic model by mean development AP across three seeds, then development F1, then the Mean, Max, Attention registration order. The original preselected static control uses Max pooling. Freeze selection, weights, and threshold receipts before computing this backtest. The backtest does not select a second recipe.

## Thresholds and Evaluation

Select the primary threshold on calibration by maximizing positive-class F1 under 12 total and 6 recovery false positives per day. Retain the calibration max-F1 operating point separately. Freeze thresholds before backtest scoring. Report across-seed means and sample standard deviations, average precision (AP), ROC-AUC, balanced accuracy, accuracy, precision, recall, F1, TP/FP/FN/TN, total and recovery FP per day, coverage, and event metrics. Compare false positives at 50%, 60%, 70%, and 80% matched recall as retrospective ranking diagnostics. The primary comparison is dynamic minus static with the same pooling. Use 1,000 paired date-block bootstrap samples and describe the conditional intervals and limits from multiple comparisons.

Report whether the three-seed AP and positive-class F1 means both reach 0.60. This is a performance result. Check backtest false-positive budgets separately. There is no requirement to target 70% accuracy.

## Implementation, Testing, and Delivery

1. Verify inputs and record actual paths, hashes, versions, resources, and commands.
2. Copy source into the working directory and add only the dynamic modules and entry point. Read large inputs from their source locations.
3. Test zero-initialized log edge weights against the static model, orientation, edgeless and isolated graphs, changes to edge weights, bounds, missingness fallback, self edges, padding, batch and center mapping, backward passes, single-batch overfitting, safe save and replay, exclusion of unknown labels, and invariance of current edge weights and scores to changes in t−1, t, or future observations.
4. Complete development training, selection, final training, and independent calibration freeze before a single pass at backtest scoring. Retain per-example float32 logits and probabilities plus exact float64 scores for threshold comparisons.
5. Report connected and isolated road groups and t−2 low-speed, regular, and missing groups. Sample and retain actual edge weights and their distributions. Record resource costs and ground interpretations in evidence.
6. Deliver the standalone entry point, results, reports, and the dynamic edge-weight audit ZIP with its extracted folder. The manifest and checksum list avoid self-reference. Extract to a new directory and verify hashes, independently recompute metrics, replay samples from all static and dynamic models, and run the new tests. Record whether a second full training run took place.

Limits include 54 isolated segments out of 121, 56 candidate edges, and limited geometry coverage. Edge weights cannot create missing network edges. There is no independent arrival timestamp. Positive and missing proportions drift in later splits, and the test dates were already public. Preserve failures and their logs and report their scope.

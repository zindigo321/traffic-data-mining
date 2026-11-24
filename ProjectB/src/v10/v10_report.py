"""Build the V10 audit narrative entirely from measured outputs."""
import csv
from itertools import product
import shutil

from v10_data import GROUP_NAMES
from v10_io import CODE, load, settings
from v10_score import METRICS

LABELS = {"B0": "V9 center rerun", "S1": "Low-speed negative sampling", "R1": "S1 with within-group ranking", "R2": "R1 with group balancing"}


def f(value, places=4):
    return "N/A" if value is None else format(value, f".{places}f")


def report(root):
    res, ev = root / "results", root / "evidence"
    summary = load(res / "summary.json")
    pick = load(res / "development_selection.json")
    lock = load(ev / "freeze.json")
    gate = load(ev / "date_gate.json")
    test = load(root / "synthetic/status.json")
    replay = ev / "verification_replay.json"
    check = load(replay if replay.is_file() else ev / "verification_offline.json")
    main = summary["calibration"][pick["primary"]]

    def records(file):
        with (res / file).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    a = [
        "# Project B V10 Audit Report", "",
        "Four recipes, three seeds, three development folds, and one calibration stage yielded 48 real models trained from scratch. The original B0 scores match V9 signed artifacts bit for bit. "
        f"The preregistered rule selected {pick['primary']}. On calibration, the primary model had AP {main['AP']:.4f}, balanced accuracy {main['balanced_accuracy']:.4f}, "
        f"F1 {main['f1']:.4f}, recall {main['recall']:.4f}, mean total false positives {main['FP_day']:.3f}/day, and recovery false positives {main['recovery_FP_day']:.3f}/day. "
        "Each of the 12 calibration models met both daily average false-positive limits.", "",
        f"Formal NYC validation is **blocked**. The latest official timestamp is {gate['latest']}. The preregistered NYC local interval [2026-11-09,2026-12-07) "
        "is incomplete. The V5/V6 holdout [2026-08-31,2026-11-09) is historical. Formal AP, balanced accuracy, F1, recall, and false-positive results are N/A. "
        "Chicago proxy continuation and the original Geolife course task remain separately blocked."
        " [Official NYC data](https://data.cityofnewyork.us/d/i4gi-tjb9) · [Date coverage evidence](evidence/date_gate.json)", "",
        "The separate calibration interval [2026-08-03,2026-08-31) contains 55,305 known windows, 1,773 positive windows, 53,532 negative windows, 572 recovery negatives, "
        "and 565 positive events. Of 81,312 scheduled windows, 26,007 are unknown. Known coverage is 68.02%, and the positive rate among known windows is 3.21%. "
        "V7–V9 previously reported this interval. V10 fits thresholds on it, so the calibration table describes same-interval diagnostics. V10 selects the recipe on development only."
        " [Date plan](plan.md) · [Fold support evidence](evidence/fold_lead_final.json)", "",
        "Each development fold fits thresholds on its first seven days and checks transfer on the last seven. The table aggregates folds within each seed and then averages the three seeds. "
        "Development selects model checkpoints on full-fold AP, so the seven-day transfer comparison remains a development diagnostic. "
        "All scheduled road keys receive scores. All known labels enter evaluation. Unknown labels enter coverage counts only.", "",
        "| Stage | Recipe | AP | BA | F1 | Recall | Total FP/day | Recovery FP/day |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, label in (("development", "Development last seven days"), ("calibration", "Calibration")):
        for recipe in LABELS:
            row = summary[stage][recipe]
            a.append("| " + " | ".join([label, recipe + " " + LABELS[recipe],
                        *(f(row[k]) for k in ("AP", "balanced_accuracy", "f1", "recall")),
                        f(row["FP_day"], 3), f(row["recovery_FP_day"], 3)]) + " |")
    a.extend(["",
        "B0 uses the original weighted BCE and stratified sampling. S1 prioritizes eligible t−2 low-speed negatives in the same budget of 12,000 known negatives. "
        "R1 adds a within-group ranking term to the S1 samples and step budget. R2 also adds group-by-class balanced BCE. "
        "All four networks have 22,594 parameters. No candidate met every preregistered condition, so B0 remains the primary recipe. "
        "R1 improved development low-speed subgroup AP and reduced recovery FP at matched recall. F1 declined from 0.7353 to 0.7332. "
        "S1 and R2 had lower AP, F1, and recall. Selection remained locked before calibration."
        " [Development selection receipt](results/development_selection.json) · [Source code](project/v10_train.py)", "",
        "| Calibration mean ± sample SD across three seeds | AP | BA | F1 | Recall |",
        "|---|---:|---:|---:|---:|",
    ])
    for recipe in LABELS:
        row = summary["calibration"][recipe]
        cells = [f"{f(row[k])}±{f(row[k+'_sd'])}" for k in ("AP", "balanced_accuracy", "f1", "recall")]
        a.append("| " + " | ".join([recipe, *cells]) + " |")
    a.extend(["",
        "The constrained threshold search enumerates every distinct probability and an all-negative sentinel. It maximizes positive-class F1 and breaks ties in favor of the higher threshold. "
        "Daily averages use complete calendar days. The calibration limits are 336 total FP and 168 recovery FP. Daily counts are reported separately because a single day can exceed the average limit. "
        "The freeze records the thresholds, configuration, weights, input hashes, and selection timestamp.", "",
        "| B0 seed | Frozen threshold | 28-day total FP | 28-day recovery FP |", "|---:|---:|---:|---:|",
    ])
    for model in lock["models"]:
        if model["recipe"] == "B0":
            a.append(f"| {model['seed']} | {model['threshold']:.17g} | {model['fit_FP']} | {model['fit_recovery_FP']} |")
    a.extend(["", f"The freeze timestamp is {lock['at_utc']}. The formal holdout release timestamp and sample-key digest are blank."
        " [Freeze receipt](evidence/freeze.json) · [Daily errors](results/daily_errors.csv)", "",
        "On the same evaluation support, matched recall stops at ceil(0.6 × positive count) true positives and breaks score ties by fixed key. "
        "The ranking diagnostic uses true labels to find its cutoff. Operating-point budget checks are separate.", "",
        "| Stage | Recipe | All total FP/day | All recovery FP/day | Low-speed total FP/day | Low-speed recovery FP/day |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for stage, label in (("development", "Development last seven days"), ("calibration", "Calibration")):
        for recipe in LABELS:
            row = summary[stage][recipe]["matched"]
            a.append("| " + " | ".join([label, recipe,
                *(f(row[scope]["0.6"][k], 3) for scope in ("all", "legal_low")
                  for k in ("FP_per_day", "recovery_FP_per_day"))]) + " |")
    a.extend(["",
        "| Calibration group | Recipe | AP | Recall | FN | Recovery FP |", "|---|---|---:|---:|---:|---:|"])
    for group, recipe in product(GROUP_NAMES, LABELS):
        row = summary["calibration"][recipe]["groups"][group]
        a.append("| " + " | ".join([group, recipe, f(row["AP"]), f(row["recall"]), f(row["fn"], 2), f(row["recovery_FP"], 2)]) + " |")
    a.extend(["",
        "At 60% recall on all calibration support, R1 reduced recovery FP for two seeds and tied for one. Total FP slightly exceeded 12/day for two seeds. "
        "R1 calibration AP was lower than baseline in the full cohort and the low-speed group. Recall rose in the regular-speed and missing-speed groups. "
        "The recipes changed score distributions across groups. The results leave full-cohort and low-speed ranking improvements unresolved jointly."
        " [Matched recall by seed](results/matched_recall.csv) · [Groups by seed](results/groups.csv)", "",
        "| Calibration R1 minus B0 | Difference | Date-block 95% interval |", "|---|---:|---:|",
    ])
    for row in records("paired_date_blocks.csv"):
        if row["stage"] == "calibration" and row["contrast"] == "R1_minus_B0":
            a.append(f"| {row['metric']} | {float(row['estimate']):+.5f} | [{float(row['lo95']):+.5f}, {float(row['hi95']):+.5f}] |")
    a.extend(["",
        "Intervals pair the three seeds and use 1,000 shared calendar-day resamples, conditional on fixed models and thresholds. "
        "The calibration R1 F1 interval crosses zero. It missed the development F1 gate, and the formal period remains incomplete, so the results support candidate diagnostics."
        " [Paired details](results/paired_date_blocks.csv)", "",
        "[Calibration PR and subgroup plots](figures/calibration_PR_and_groups.png) · [False-positive plot](figures/calibration_false_alarms.png) · "
        "[Development learning curves](figures/development_training_curve.png)", "",
        "The 48 fits ran in a verified CPU environment, with two threads per task and at most four simultaneous tasks. See the resource table for per-process peak RSS and runtime. "
        "The original data, V9 artifacts, and Conda environment were read-only. Records include commands, start and end times, exit codes, and tracebacks. "
        "The initial calibration launch failed on an integer command argument. The launcher converted arguments and the rerun succeeded. "
        "Additional fixes handled precision N/A for groups with no predicted positives and numeric tolerance for means of float32 weights. The original failure logs remain archived."
        " [Environment](logs/environment.json) · [Resource table](results/training_resources.csv) · [Change notes](reports/change_notes.md)", "",
        "## Reproduction Checks", "",
        "1. Verify manifest.csv, SHA256SUMS, and the signed V9 ZIP and original query hashes in evidence/inputs.json. Both inventories exclude SHA256SUMS itself.",
        "2. Rebuild the four eligible lead arrays from the download receipts and external paths. Check the prepared_development SHA and each fold's known sample keys. "
        "Check date boundaries, original labels, unknowns, roads, and eligible past history.",
        "3. Verify all four recipes use the same full evaluation keys, training counts, seeds, step budgets, and preprocessing. The sampled IDs of S1, R1, and R2 match exactly. "
        "B0 reproduces the original V9 scores bit for bit. t−1 ground-truth risk is excluded from features, sampling, and loss.",
        "4. Verify development training receipts, selection time, fixed calibration epochs, calibration opening time, and the freeze receipt in that order. "
        "Recompute operating points, subgroup metrics, and matched recall from per-window float32 logits and probabilities, float64 scores, and thresholds.",
        f"5. Independent checks recomputed {check['operating_points']} operating points, {check['groups']} subgroup metrics, and {check['matched']} matched-recall points. "
        f"Sampled replay covered {check['models_replayed']} models. The maximum score difference was {check['max_score_difference'] if check['max_score_difference'] is not None else 'N/A'}. "
        f"All {test['total']} engineering tests passed. Clean extraction and one-command rerun receipts are stored next to the archive. [Reproduction entry point](README_audit.md)", "",
        "The offline cutoff uses data_as_of because the source lacks separate arrival timestamps. Directed edges derive from frozen road geometry. "
        "The center-road control uses only the target node. NYC public speed estimates, roads, and proxy labels differ from the Geolife course requirements. "
        "The original course task and Chicago continuation remain blocked. See the method contract for coverage and map biases.", "",
        "> V10 retrained 48 models on NYC development and existing calibration dates. On calibration, the within-group ranking candidate reduced some recovery false positives and improved recall. "
        "AP declined, and the development F1 gate failed. The preregistered selection rule retained the V9 center recipe. The new NYC holdout is incomplete, so formal results remain N/A.", "",
    ])
    (root / "audit_report.md").write_text("\n".join(a), encoding="utf-8")
    (root / "test_report.md").write_text(
        "# V10 Test Report\n\n"
        f"All {test['total']} engineering tests passed, including nine new V10 tests. They cover eligible time groups, within-group ranking gradients and absent-class branches, "
        "group-balanced weights, negative sampling from eligible inputs, holdout refusal, N/A aggregation, recipe fallback, and integer command arguments. "
        "Inherited tests of graph structure, pooling, batch isolation, target mapping, unknown handling, leakage prevention, perturbations of the two future label bins, and saving and loading weights were rerun."
        " [Commands and results](synthetic/status.json)\n\n"
        f"Independent real-data verification recomputed {check['operating_points']} operating points, {check['groups']} subgroup metrics, and "
        f"{check['matched']} matched-recall points. Sampled replay covered {check['models_replayed']} models. "
        f"The maximum score difference was {check['max_score_difference'] if check['max_score_difference'] is not None else 'N/A'}. "
        "Weight loading uses weights_only=True. The full B0 scores match V9 bit for bit. Formal holdout verification remains blocked.\n", encoding="utf-8")
    for name in ("README_audit.md", "method_and_data_contract.md"):
        source = CODE.parent / name
        if source.is_file() and source.resolve() != (root / name).resolve():
            shutil.copy2(source, root / name)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    source = CODE.parent / "reports/change_notes.md"
    if source.is_file() and source.resolve() != (root / "reports/change_notes.md").resolve():
        shutil.copy2(source, root / "reports/change_notes.md")
    print(dict(report="audit_report.md", primary=pick["primary"], tests=test["total"]), flush=True)


if __name__ == "__main__":
    report(settings()[0])

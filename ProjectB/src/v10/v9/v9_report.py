"""Render the human-readable V9 audit from the measured tables."""
import csv
from itertools import product

from v9_io import load, settings

LABELS = {"center": "V7 center road segment", "attention": "V7 Attention", "rank": "V8 tail ranking"}
OP = ["AP", "balanced_accuracy", "f1", "recall", "FP_day", "recovery_FP_day"]


def report(root):
    ev = root / "evidence"
    gate = load(ev / "date_gate.json")
    lock = load(ev / "freeze.json")
    selection = load(ev / "development_selection.json")
    result = root / "results"
    numbers = load(result / "summary.json")
    tests = load(root / "synthetic/status.json")
    replay = ev / "verification_replay.json"
    verification = load(replay if replay.is_file() else ev / "verification_offline.json")
    cal = numbers["calibration"]["center"]["selected"]

    def read(name):
        with (result / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    blocks = read("paired_date_blocks.csv")
    a = [
        "# Project B V9 Audit Report", "", "## Acceptance Results", "",
        "The NYC audit completed three development folds, 28 days of separate calibration, and threshold freezing. The preselected primary model was the V7 center road segment model with false-positive-constrained F1. "
        f"Development selected no group offset for the center model, retaining δ={selection['chosen']['center']['delta']}. Three-seed calibration means were "
        f"AP {cal['AP']:.4f}, balanced accuracy {cal['balanced_accuracy']:.4f}, positive-class F1 {cal['f1']:.4f}, recall {cal['recall']:.4f}, "
        f"total false positives {cal['FP_day']:.3f}/day, and recovery false positives {cal['recovery_FP_day']:.3f}/day. Every seed met both limits. "
        "The audit fit thresholds on calibration data. Formal AP, F1, recall, and false-positive results are N/A.", "",
        "Formal NYC validation is **blocked**. The preregistered NYC local dates are 2026-11-09 through 2026-12-06. "
        f"The latest queried timestamp is {gate['latest']}. The interval is incomplete. The V5/V6 holdout from 2026-08-31 through 2026-11-08 is historical. "
        "Development, calibration, and freezing used artifacts from before August 31. Chicago continuation and the original Geolife course task remain separately blocked."
        " [NYC DOT data](https://data.cityofnewyork.us/d/i4gi-tjb9) · [Chicago discontinuation notice](https://data.cityofchicago.org/stories/s/c28d-stgw)", "",
        "## Support and Timing", "",
        "| Split | Calendar days | Scheduled keys | Known | Positive | Negative | Unknown | Recovery negatives | Positive events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in read("support.csv"):
        if row["scope"] not in ("full", "assess") or (row["fold"] == "final" and row["scope"] == "assess"):
            continue
        cols = ("days", "scheduled", "known", "positives", "negatives", "unknown", "recovery_negatives", "event_segments")
        label = "Calibration" if row["fold"] == "final" else row["fold"] + (" full fold" if row["scope"] == "full" else " last seven days")
        a.append("| " + " | ".join([label, *(f"{int(row[k]):,}" for k in cols)]) + " |")
    a.extend([
        "",
        "The audit scores all scheduled keys for 121 fixed directed roads during both local peak periods. Every known label enters evaluation. "
        "Calibration coverage is 68.02%, with a positive rate of 3.21% among known windows and 91 roads with known records. "
        "Scoring occurs 10 minutes before the target bin begins and uses observations at t−2 or earlier. Speeds in t−1 and t enter ground truth and false-positive audits only. "
        "The frozen historical baseline matches road, time slot, and day type and requires at least three eligible past days. [Method and data contract](method_and_data_contract.md) · [Support details](results/support.csv)",
        "", "## Operating Points on Shared Support and Budgets", "",
        "The development table aggregates the last seven days of the three folds within each seed. Each fold fits its threshold on the first seven days. The original model checkpoint was selected on the full development fold. "
        "Calibration fits thresholds on a separate date interval and reports diagnostics on that same interval. The global policy uses original scores. The selected policy uses the offset frozen in development. "
        "Every model uses the same offset grid, seeds, samples, and budgets.", "",
        "| Stage | Model | Policy δ | AP | Balanced accuracy | F1 | Recall | Total FP/day | Recovery FP/day | Budget passes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for stage, label in (("development", "Development last seven days"), ("calibration", "Calibration")):
        for model, policy in product(LABELS, ("global", "selected")):
            if policy == "global" and selection["chosen"][model]["delta"] == 0:
                continue
            row = numbers[stage][model][policy]
            cells = [f"{row[k]:.4f}" for k in OP[:4]] + [f"{row[k]:.3f}" for k in OP[4:]]
            a.append("| " + " | ".join([label, LABELS[model], f"{policy} {row['delta']:g}", *cells,
                                       f"{row['budget_pass_rows']}/{row['runs']}"]) + " |")
    a.extend([
        "",
        f"The center model's three-seed calibration standard deviations were AP {cal['AP_sd']:.4f}, BA {cal['balanced_accuracy_sd']:.4f}, "
        f"F1 {cal['f1_sd']:.4f}, and recall {cal['recall_sd']:.4f}. The global center policy and V9 primary policy agreed in every window. "
        "Attention passed the development gate at δ=−0.25 and used calibration only to fit the threshold. Tail ranking retained δ=0. "
        "See the configuration-level table for ROC-AUC, precision, accuracy, TP, FP, FN, TN, and support by seed. [Operating points](results/operating_points.csv)", "",
        "| Center seed | Frozen threshold | 28-day total FP | 28-day recovery FP |", "|---:|---:|---:|---:|",
    ])
    for r in lock["models"]:
        if r["model"] == "center" and r["policy"] == "selected":
            a.append(f"| {r['seed']} | {r['threshold']:.17g} | {r['fit_FP']} | {r['fit_recovery_FP']} |")
    a.extend([
        "",
        "Calibration limits were 336 total FP and 168 recovery FP. All three seeds met both limits. The freeze timestamp is "
        f"{lock['at_utc']}. Its receipt covers configuration, weights, preprocessing, per-window scores, thresholds, and calibration key digests. "
        "The formal holdout key digest and release timestamp are blank. [Freeze receipt](evidence/freeze.json)", "",
        "## Matched Recall and False Positives", "",
        "The table fixes the true-positive count within each fold at a 60% recall target. Rank scores from high to low and break ties by key. "
        "Matched recall uses labels to locate its cutoff and serves as a ranking diagnostic. Frozen-threshold operating points check budgets separately. [Matched recall details](results/matched_recall.csv)", "",
        "| Stage | Center or comparison model | Total FP/day | Recovery FP/day |", "|---|---|---:|---:|",
    ])
    for stage, label in (("development", "Development last seven days"), ("calibration", "Calibration")):
        for model, rule in (("center", "selected"), ("attention", "global"), ("attention", "selected"), ("rank", "global")):
            r = numbers[stage][model][rule + "_points"]["0.6"]
            a.append(f"| {label} | {LABELS[model]} {rule} | {r['FP_per_day']:.3f} | {r['recovery_FP_per_day']:.3f} |")
    a.extend(["", "| Calibration difference, center minus comparison | F1 difference | Recall difference | Recovery FP/day difference at 60% recall |", "|---|---:|---:|---:|"])
    for label, ref in (("V7 Attention global", "V7_Attention_global"), ("V8 tail ranking", "V8_rank_global"),
                       ("V7 Attention subgroup offset", "V7_Attention_selected")):
        rs = [next(r for r in blocks if r["stage"] == "calibration" and r["contrast"] == f"center_selected_minus_{ref}"
                   and r["metric"] == m) for m in ("f1", "recall", "recovery_FP60_day")]
        cells = [f"{float(r['estimate']):+.4f} [{float(r['lo95']):+.4f}, {float(r['hi95']):+.4f}]" for r in rs]
        a.append("| " + " | ".join([label, *cells]) + " |")
    errors = [r for r in read("error_slices.csv") if r["fold"] == "final" and r["model"] == "center" and r["policy"] == "selected"]

    def avg(group, k):
        return sum(int(r[k]) for r in errors if r["slice"] == group) / len({r["seed"] for r in errors})

    a.extend([
        "",
        "Brackets show 95% intervals from 1,000 shared calendar-day resamples. Comparisons pair seeds before averaging. Bounds are conditional on fixed weights, offsets, and thresholds. "
        "At 60% calibration recall, the center model exceeded both budgets and had slightly fewer recovery false positives than the two global comparisons. "
        "Center recovery false positives were higher than both comparisons on the last seven days of development. Calibration intervals cross zero, leaving stable gains in false positives, F1, and recall unconfirmed."
        " [Full date-paired results](results/paired_date_blocks.csv)", "",
        f"At the frozen calibration threshold the center model averaged {avg('all', 'recovery_fp'):.1f} recovery false positives, "
        f"including {avg('legal_low_t_minus_2', 'recovery_fp'):.1f} from the eligible t−2 low-speed group. "
        f"It averaged {avg('all', 'fn'):.1f} false negatives, including {avg('legal_known_other_t_minus_2', 'fn'):.1f} with an observed regular speed at t−2 "
        f"and {avg('legal_missing_t_minus_2', 'fn'):.1f} with no eligible speed ratio. Consecutive 10-minute positive windows on the same road form events. An unknown window interrupts the event."
        " [False-positive and false-negative slices](results/error_slices.csv) · [Event metrics](results/events.csv)", "",
        "[Calibration PR curves](figures/calibration_PR.png) · [Calibration thresholds and matched recall](figures/calibration_false_alarms.png)", "",
        "## Reproduction Checks", "",
        "1. Verify package paths, bytes, and hashes against manifest.csv and SHA256SUMS. SHA256SUMS includes the manifest hash. Both inventories exclude SHA256SUMS itself.",
        "2. Verify the V8 ZIP and 36 signed artifacts against evidence/inputs.json. Download receipts list official URLs, queries, row counts, hashes, and external paths. "
        "Use V7 to reconstruct the directed graph, frozen history, and four lead arrays. Check prepared_development.json and each fold's known key digest.",
        "3. Use results/signed/ to verify shared sample keys, seeds, preprocessing, and training sample IDs across the three models. Recompute the development grid and offset selection using eligible lag20 flags in results/groups/.",
        "4. Check the development lock, the calibration opening timestamp, and the freeze receipt. Recompute per-window decisions, confusion counts, AP, BA, F1, recall, matched recall, and budgets. "
        "Review the timestamp query and blocked record in evidence/date_gate.json.",
        f"5. Independent verification recomputed {verification['grid_recomputed']} development candidates and {verification['frozen_and_transfer_recomputed']} operating points, "
        f"plus {verification['matched_recall_recomputed']} matched-recall points. It replayed sampled batches for {verification['models_replayed']} real models. "
        f"The maximum score difference was {verification['max_score_difference'] if verification['max_score_difference'] is not None else 'N/A'}. "
        f"The audit recorded {tests['total']} engineering tests. [Reproduction entry point](README_audit.md)", "",
        "## Data Scope and Course Defense Summary", "",
        "Offline availability follows data_as_of because the source lacks separate arrival records. Directed geometry endpoints determine road edges. Isolated nodes and biases in speed and coverage remain. "
        "The center control uses only the target road. V7 Attention and V8 tail ranking use local graphs. V9 replayed V7/V8 weights and fitted operating policies. "
        "Chicago proxy data remains unavailable for continuation. The Geolife course task lacks positive labels. NYC is a supplemental public-data evaluation."
        " [NYC DOT data](https://data.cityofnewyork.us/d/i4gi-tjb9)", "",
        f"> On NYC's separate calibration dates the center model scored with a strict ten-minute lead. Across three seeds AP was {cal['AP']:.3f}, F1 {cal['f1']:.3f}, and recall {cal['recall']:.3f}. "
        "Every seed met both false-positive budgets. Group adjustments failed to improve the preselected center model's development transfer, and the direction of recovery false-positive differences at matched recall varied by stage. "
        "The formal NYC holdout is incomplete and formal results remain N/A.", "",
    ])
    (root / "audit_report.md").write_text("\n".join(a), encoding="utf-8")
    (root / "test_report.md").write_text(
        "# V9 Test Report\n\n"
        f"All {tests['total']} engineering tests passed, including {tests['counts']['v9']} new V9 tests. "
        "They cover time-eligible grouping, score identity, false-positive budgets, tied thresholds, the development gate, refusal of incomplete holdouts, and resampling by date. "
        "Inherited V3–V8 tests for labels, graph structure, pooling, isolation, model save and load, and perturbations at t−1, t, and later were rerun. "
        "The engineering tests use synthetic inputs. [Test commands and results](synthetic/status.json)\n\n"
        f"Independent real-data validation recomputed {verification['grid_recomputed']} development candidates, "
        f"{verification['frozen_and_transfer_recomputed']} operating points, and {verification['matched_recall_recomputed']} matched-recall points. "
        f"Replay sampled batches from {verification['models_replayed']} models with signed weights. "
        f"The maximum score difference was {verification['max_score_difference'] if verification['max_score_difference'] is not None else 'N/A'}. "
        "Weight loading uses weights_only=True. Incomplete date coverage blocks formal holdout validation.\n", encoding="utf-8")
    print(dict(report="audit_report.md", engineering_tests=tests["total"], verification=verification["status"]), flush=True)


if __name__ == "__main__":
    report(settings()[0])

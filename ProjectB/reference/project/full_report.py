"""Human-readable audit of the actually executed full-data experiment."""
from pathlib import Path
import shutil

from full_io import CODE,load,settings
from full_score import ALL,MODELS

LABELS={"graph_mean":"Graph Transformer Mean","graph_max":"Graph Transformer Max",
        "graph_attention":"Graph Transformer Attention","center_full":"Full-data center segment",
        "center_rank":"Full-data center with within-group ranking",
        "center_sampled_matched":"Sampled center with matched training steps",
        "logistic":"Full-data logistic regression","lag20_rule":"Admissible lag20 rule","majority":"Majority class"}


def fmt(x,d=4):
    return "N/A" if x is None else format(x,f".{d}f")


def report(root):
    ev=root/"evidence"
    summary=load(root/"results/summary.json")
    matched=load(root/"results/group_matched_summary.json")
    pick=load(root/"results/development_selection.json")
    freeze=load(ev/"model_threshold_freeze.json")
    prepared=load(ev/"prepared.json")
    source=load(ev/"source_snapshot.json")
    tests=load(root/"synthetic/status.json")
    verification=load(ev/"verification_replay.json")
    main=summary["2"]["constrained"][pick["primary"]]
    cohort=load(ev/"cohort_final.json")["parts"]
    a=["# Project B Full-Data Experiment Audit","",
       "The authorized run removed calendar waiting, minimum coverage, and event count restrictions from the experiment entry point. "
       "It combined checked caches and the available NYC extension and fitted 36 neural models and six deterministic baselines. "
       "Five full-data configurations use all known training examples. The sixth is a documented sample-based control with matched steps. "
       "Unknown windows receive scores and count toward coverage.","",
       f"The source contained {prepared['source_funnel']['rows']:,} records, with {source['new_rows']:,} added over the final three days. "
       f"The latest source timestamp is {source['latest_source_time']}. The snapshot covers 2026-05-11 through 2026-11-11, 121 fixed directed segments, and the original two peak periods. "
       "The previously public V5/V6 holdouts and November 9–11 observations are included in this temporal backtest. "
       "Independent forward validation requires unused future dates. "
       "[Source snapshot](evidence/source_snapshot.json) · [Raw file checks](evidence/sources_cached.json)","",
       "| Split | Dates (end exclusive) | Scheduled | Known | Positive | Negative | Unknown | Positive events |",
       "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in cohort:
        a.append("| "+" | ".join([r["part"],r["start"]+" to "+r["stop_exclusive"],
                    *(f"{r[k]:,}" for k in ("scheduled","known","positive","negative","unknown","positive_events"))])+" |")
    a.extend(["",
        "The last source day has incomplete evening-peak observations. Scheduled windows retain unknowns and report observed coverage. "
        "Daily false positives use the full registered calendar as the denominator. Window-level errors and event metrics are reported separately. "
        "Training, calibration, and test window keys are disjoint. In training, past eligible observations update baselines daily. Calibration and test use frozen training history. "
        "Expanding training can therefore change same-day labels relative to earlier frozen folds. The two-bin rule at 40% stays fixed. "
        "[Splits and reasons for unknowns](evidence/cohort_final.json)","",
        f"Mean development AP across three seeds selected {LABELS[pick['primary']]}. The final saved epoch is the median of the corresponding saved development epochs. "
        "All models run for 12 epochs. Thresholds use only the calibration dates from 2026-08-31 through 09-27. "
        "The backtest from 2026-09-28 through 11-11 does not inform model or threshold selection.","",
        "| Backtest mean across three seeds | AP | BA | F1 | Recall | Total FP/day | Recovery FP/day |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for model in ALL:
        r=summary["2"]["constrained"][model]
        a.append("| "+" | ".join([LABELS[model],*(fmt(r[k]) for k in ("AP","balanced_accuracy","f1","recall")),
                                  fmt(r["FP_day"],3),fmt(r["recovery_FP_day"],3)])+" |")
    a.extend(["",
        f"The preselected primary model has test AP of {fmt(main['AP'])}±{fmt(main['AP_sd'])} "
        f"and F1 of {fmt(main['f1'])}±{fmt(main['f1_sd'])}. "
        "Each deterministic baseline fits once, so its SD is N/A. "
        "The calibration threshold budgets are 12 total and six recovery false positives per day. The table reports the actual test results. "
        "[Operating points by model and seed](results/operating_points.csv)","",
        "| Primary model threshold rule | AP | BA | F1 | Recall | Total FP/day | Recovery FP/day |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for policy in ("constrained","max_f1"):
        r=summary["2"][policy][pick["primary"]]
        a.append("| "+" | ".join([policy,*(fmt(r[k]) for k in ("AP","balanced_accuracy","f1","recall")),
                                  fmt(r["FP_day"],3),fmt(r["recovery_FP_day"],3)])+" |")
    a.extend(["",
        "The maximum-F1 operating point illustrates the trade-off between false positives and recall. Both thresholds come from calibration only. "
        "The report retains test scores below their targets and false positives above their budgets.","",
        "| Full-data common support at 60% matched recall | Total FP/day | Recovery FP/day | Achieved recall |",
        "|---|---:|---:|---:|"])
    for model in ALL:
        r=matched["2"][model]["matched"]["all"]["0.6"]
        a.append("| "+" | ".join([LABELS[model],fmt(r["FP_per_day"],3),
                                  fmt(r["recovery_FP_per_day"],3),fmt(r["achieved_recall"])])+" |")
    a.extend(["",
        "Matched recall uses common support and equal true-positive counts. A fixed example key breaks score ties. "
        "Operating thresholds and matched-recall cutoffs have separate records. AP, recall, and false negatives are tracked for admissible t−2 low-speed, non-low-speed, and missing groups. "
        "The 1,000 date-block replicates share weights across models and seeds. Intervals are conditional on the fixed models, thresholds, and this backtest sample. "
        "[Group metrics](results/groups.csv) · [Matched recall details](results/matched_recall.csv) · [Paired intervals](results/paired_date_blocks.csv)","",
        "The full-data neural configurations visit every known example in each training split once per epoch, independently checked against the training IDs. "
        "The sampled control retains 12,000 negatives and all positives and cycles its IDs to the same total visits. "
        "All six configurations share identical step counts per epoch. Mean, Max, and Attention share training IDs, seeds, features, and backbone and differ in pooling. "
        "See the resource table for original class counts, actual training visits, runtime, peak memory, and parameters. "
        "[Training resources](results/training_resources.csv)","",
        f"{tests['total']} engineering tests passed. Independent checks recomputed 84 operating points and 336 matched-recall points. "
        f"Replay covered {verification['neural_replayed']} neural models with maximum score difference {verification['maximum_neural_error']}. "
        "Weights are loaded with weights_only=True and NPZ arrays with allow_pickle=False. "
        "Real-data outputs and synthetic engineering tests are stored separately. "
        "[Verification receipt](evidence/verification_replay.json) · [Engineering tests](synthetic/status.json)","",
        "The data source is public NYC DOT segment speed estimates with the established road geometry. Raw speeds are in mph and timestamps are local without a UTC offset. "
        "Independent observation-arrival times are unavailable, so data_as_of determines offline availability. "
        "The original speed and travel-time quality rules filter the raw records. Missing speeds, insufficient history, and unconfirmed low speed remain unknown. "
        "The package includes learning curves, parameters, per-example float32 probabilities and logits, float64 scores, sampled and full training IDs, and preprocessors. "
        "Large raw and feature arrays are in read-only external paths with saved file hashes, queries, and rebuild commands. "
        "[Official NYC data](https://data.cityofnewyork.us/d/i4gi-tjb9)","",
        "[Precision–recall curves for three pooling methods](figures/test_PR_three_pooling.png) · [Metrics and false positives](figures/test_pooling_and_controls.png) · "
        "[Development learning curves](figures/development_curves.png)","",
        "Verify the experiment in the following order.","",
        "1. Check the manifest, SHA256SUMS, and original V10/V5/V6 signatures. Use source_snapshot to verify all 54 raw files.",
        "2. Rebuild 121 nodes and 56 candidate directed edges from the fixed geometry. Recompute the full grid, baselines, labels, and 52 strict-lead features. Check the prepared arrays and split key summaries.",
        "3. Check training IDs against all known indices, step matching for the sampled control, and disjoint calibration and test keys.",
        "4. Check the chronology of development selection, final training, threshold freezing, and test metric computation. Recompute per-seed confusion counts, AP, BA, F1, recall, false positives, and matched recall. Replay a sample with the saved weights.",
        "5. Extract into a fresh directory, check the manifest, and use the documented entry points to rebuild features and recompute results. The extraction verification receipt accompanying the ZIP records the actual run. "
        "[Quick and staged entry points](README_full_experiment.md)","",
        "This run completed the experiment on available data. The original Geolife coursework and Chicago extension follow their existing evidence records. "
        "The NYC result is a temporal backtest on supplementary public data.",""])
    (root/"audit_report.md").write_text("\n".join(a),encoding="utf-8")
    (root/"test_report.md").write_text(
        f"# Full-Data Experiment Test Report\n\n{tests['total']} engineering tests passed. Seven additions cover all known indices, exclusion of unknown labels, matching optimizer steps, "
        "handling low coverage and single-class metrics, and keeping calibration and test labels out of historical updates.\n\n"
        f"Replay covered {verification['neural_replayed']} neural models with maximum score difference {verification['maximum_neural_error']}. "
        "Six deterministic baselines were also checked. Real metrics use the full temporal split. Synthetic examples exercise engineering logic.\n",encoding="utf-8")
    for filename in ("README_full_experiment.md","method_and_data_contract.md"):
        original=CODE.parent/filename
        if original.exists() and original.resolve()!=(root/filename).resolve():
            shutil.copy2(original,root/filename)
    (root/"reports").mkdir(parents=True,exist_ok=True)
    original=CODE.parent/"reports/change_notes.md"
    if original.exists() and original.resolve()!=(root/"reports/change_notes.md").resolve():
        shutil.copy2(original,root/"reports/change_notes.md")
    print(dict(report="audit_report.md",primary=pick["primary"],test_AP=main["AP"],test_F1=main["f1"]),flush=True)


if __name__=="__main__":
    report(settings()[0])

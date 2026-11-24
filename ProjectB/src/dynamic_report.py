"""Write the handoff from measured files; never synthesize missing performances."""
from pathlib import Path
import difflib
import numpy as np
import pandas as pd

from full_io import CODE,load,save,settings,sha
from dynamic_score import ALL,DYNAMIC,STATIC


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def text(root,name,value):
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value.strip()+'\n',encoding='utf-8')


def report(root,cfg):
    summary=load(root/'results/summary.json')
    dev=load(root/'results/dynamic_selection.json')
    paired=load(root/'results/paired_summary.json')
    freeze=load(root/'evidence/dynamic_threshold_freeze.json')
    verify=load(root/'evidence/verification_dynamic_replay.json')
    tests=load(root/'synthetic/engineering_status.json')
    cohort=load(root/'evidence/cohort_final.json')['parts']
    graph=load(root/'evidence/graph.json')
    primary=dev['primary']
    p=summary['2']['constrained'][primary]
    baseline=summary['2']['constrained']['graph_'+primary.split('_')[-1]]
    precision=lambda v:f'{v:.4f}' if v is not None else 'N/A'
    pm=lambda m,k:f"{precision(m[k])} ± {precision(m[k+'_sd'])}"
    rows=[]
    for model in ALL:
        m=summary['2']['constrained'][model]
        rows.append([model,pm(m,'AP'),pm(m,'f1'),precision(m['balanced_accuracy']),precision(m['recall']),
                     f"{m['FP_day']:.3f}",f"{m['recovery_FP_day']:.3f}"])
    metric_table=table(['Recipe','AP (mean ± sample SD)','Positive F1 (mean ± sample SD)','Balanced accuracy','Recall','Total FP/day','Recovery FP/day'],rows)
    matched_rows=[]
    for model in ALL:
        m=paired['2'][model]['matched60']
        matched_rows.append([model,precision(m['achieved_recall']),f"{m['FP_per_day']:.3f}",f"{m['recovery_FP_per_day']:.3f}"])
    matched_table=table(['Recipe','Achieved recall','Total FP/day','Recovery FP/day'],matched_rows)
    points=pd.read_csv(root/'results/matched_recall.csv',float_precision='round_trip')
    signs=[]
    deltas=[]
    for pool in ('mean','max','attention'):
        a=points[(points.model=='dynamic_'+pool)&(points.phase==2)&(points.scope=='all')&(points.target_recall==.6)].sort_values('seed')
        b=points[(points.model=='graph_'+pool)&(points.phase==2)&(points.scope=='all')&(points.target_recall==.6)].sort_values('seed')
        total=(a.FP_per_day.to_numpy()<b.FP_per_day.to_numpy())
        recovery=(a.recovery_FP_per_day.to_numpy()<b.recovery_FP_per_day.to_numpy())
        signs.append(f'{pool}: total false positives fell for {int(total.sum())}/3 seeds, recovery false positives for {int(recovery.sum())}/3, and both for {int((total&recovery).sum())}/3.')
        ma=summary['2']['constrained']['dynamic_'+pool]
        mb=summary['2']['constrained']['graph_'+pool]
        deltas.append(['dynamic_'+pool+' − graph_'+pool,f"{ma['AP']-mb['AP']:+.4f}",f"{ma['f1']-mb['f1']:+.4f}",
            f"{a.FP_per_day.mean()-b.FP_per_day.mean():+.3f}",f"{a.recovery_FP_per_day.mean()-b.recovery_FP_per_day.mean():+.3f}"])
    delta_table=table(['Same-pooling paired difference','ΔAP','ΔF1','ΔTotal FP/day at matched recall','ΔRecovery FP/day at matched recall'],deltas)
    intervals=pd.read_csv(root/'results/paired_date_blocks.csv',float_precision='round_trip')
    interval_rows=[[r.contrast,r.metric,f'{r.low95:+.4f}',f'{r.high95:+.4f}'] for r in intervals.itertuples()
        if r.phase==2 and r.metric in ('f1','FP60_day','recovery_FP60_day')]
    interval_table=table(['Pair','Metric','Conditional 95% lower bound','Conditional 95% upper bound'],interval_rows)
    environment=load(root/'logs/environment_dynamic.json')
    outcome='met the backtest metric targets' if p['AP']>=.6 and p['f1']>=.6 else 'fell short of the backtest metric targets'
    budgets=table(['Pooling','Backtest AP ≥ 0.60 and F1 ≥ 0.60 (three-seed means)','Total FP ≤ 12/day','Recovery FP ≤ 6/day'],[
        [model,'Yes' if summary['2']['constrained'][model]['AP']>=.6 and summary['2']['constrained'][model]['f1']>=.6 else 'No',
         'Yes' if summary['2']['constrained'][model]['FP_day']<=12 else 'No',
         'Yes' if summary['2']['constrained'][model]['recovery_FP_day']<=6 else 'No'] for model in DYNAMIC])
    resources=pd.read_csv(root/'results/training_resources.csv')
    new=resources[resources.new_training]
    clean_path=root/'evidence/clean_qa_prepackage.json'
    clean=load(clean_path) if clean_path.exists() else None
    commands='''```powershell
$python = 'D:/conda_envs/niulaoshi/python.exe'
# From the extracted audit package root, recompute metrics and thresholds from bundled predictions.
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_verify_new --no-replay
# Supply both externally verified prepared arrays to sample replay all 36 models.
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_replay_new
# Rerun all 18 dynamic fits in a fresh directory and verify and reuse the 18 static models.
& $python -B -X utf8 project/run_dynamic.py --phase all --output G:/gpt-file/Codex/work/ProjectB_dynamic_refit_new
```

Choose an unused output directory. For individual phases, replace `--phase all` with check, tests, development, select, final, freeze, evaluate, analysis, verify, or report. The full entry point copies initialization files. Preserve the existing registration receipt when running check.
'''
    method='''# Dynamic Edge Weight Method and Data Timing Contract

Nodes are the 121 original directed road segments. The original source-to-target directions, 56 candidate edges, 54 isolated nodes, and one-hop local graphs are used. Graphs in a batch are independent. The target maps to local node 0 and attends only to physical predecessors and itself. Pooling covers the target's local subgraph. The input tensor `[B,N,104]` contains 52 train-fitted imputed and standardized values plus their 52 missingness indicators, with `N ≤ 5`. See `evidence/cohort_final.json` for the 52 fields. The experiment uses public segment-level data and copied no additional private trajectories.

The descriptors use `lag20_relative_speed`, `lag30_relative_speed`, `lag40_relative_speed`, `lag20_minus_lag30_ratio`, `legal_usable_count`, `lag20_message_count`, `lag20_history_days`, and `lag20_half_last_ratio`. Each endpoint contributes eight values and eight original missingness indicators. Values are clipped to standardized ±5 for edge descriptors only. Concatenating source, target, and their elementwise absolute difference yields 48 dimensions. The latest dynamic information is bin t−2. The t−1 and t speeds used in the label and future observations are excluded. Here t is the target window start, scoring occurs at t−10 minutes, and t−2 is [t−20,t−10).

For layer l, head h, and physical edge j to i:

`b[l,h,i,j] = log(2) * tanh(MLP_l([u_j,u_i,abs(u_j-u_i)]))[h]`

`w[l,h,i,j] = exp(b[l,h,i,j]) ∈ [0.5,2]`

`attention = softmax_j(Q_i K_j / sqrt(8) + b[l,h,i,j] + directed_mask)`

The MLP has layers 48→16→4 with GELU. It adds 852 parameters per layer and 1,704 across both layers. Zero initialization of the final linear layer gives all edges initial weight 1 and preserves the backbone and classifier initialization. Self-edge weights stay at 1. A missing endpoint lag20 relative speed also yields weight 1. The mask excludes nonphysical edges. The original QK attention is already input-dependent. Dynamic edge weights add an explicit, auditable multiplicative prior based on legal past traffic attributes. Dynamic parameters learn only from training labels.

Each of the two layers has a `[B,4,N,N]` prior tensor. Physical source-to-target edges map to source keys and target queries. Local attention uses dense computation. Per-layer backbone cost is `O(BN²D + BND²)`, with edge MLP cost `O(BN²(48×16+16×4))` and additional space `O(BHN²)`. Edge weights create no additional roads or cross-batch connections. An isolated graph has only the unit self edge, so toggling dynamic weights produces exactly identical outputs for the same frozen backbone.

Mean pooling averages valid node vectors elementwise. Max pooling takes an elementwise maximum over valid nodes. Attention pooling uses `a_i=softmax(vᵀh_i+c)` and `sum(a_i h_i)`. Each costs `O(BND)`. Concatenate the target node and pooled vectors into `[B,64]` and output one binary logit per target road segment. Reject empty graphs and invalid centers. Padding is excluded from pooling and valid edges.

Use New York local time, 07:00–09:00 and 17:00–19:00, and the original two-bin low-speed label, historical support, and unknown definitions. Past observations in training form the historical baseline and are frozen in subsequent periods. Fit standardization on training inputs only. Edge weights do not depend on current ground truth, risk, later labels, evaluation classes, or sampling. Risk is a t−1 low-speed indicator used for independent calibration of false-positive limits and retrospective recovery diagnostics. Training, routing, and edge weight computations exclude it.

The original data sources, download times, queries, and hashes were inherited and verified against `source_snapshot.json`. This run neither fetched new dates nor rechecked the source terms. Usage rights follow the original terms. See `source_provenance/` for the official source and usage records. This historical audit directory refers to the read-only speed shards and prepared arrays in `evidence/path_mapping.json` and `evidence/prepared.json`. The full input-to-metric pipeline is in `project/full_*.py`. Offline `data_as_of` provides the original assumed cutoff, with no separate observation-arrival timestamp.
'''
    text(root,'method_and_data_contract.md',method)
    audit=f'''# Project B Dynamic Edge Weight Audit Report

## Findings and Acceptance

The NYC public-data experiment completed 18 new dynamic fits across three pooling methods, three seeds, and two stages, development and final. The paired comparison reused and verified 18 static models from the earlier audit. Development AP selected **{primary}**. Across three backtest seeds, AP was {pm(p,'AP')}, positive-class F1 was {pm(p,'f1')}, balanced accuracy was {precision(p['balanced_accuracy'])}, and recall was {precision(p['recall'])}. The model **{outcome}**. It recorded {p['FP_day']:.3f} total false positives per day and {p['recovery_FP_day']:.3f} recovery false positives per day. The table reports each budget separately. ROC-AUC was {precision(p['ROC_AUC'])}, precision was {precision(p['precision'])}, and accuracy was {precision(p['accuracy'])}. With roughly 3.56% positives, interpret accuracy alongside positive-class metrics.

The test dates had already been disclosed. This experiment compares variants over those same dates and samples. An independent blind test on new dates remains pending. The original Geolife course experiment and Chicago protocol remain blocked or N/A. NYC serves as a public proxy with different users, years, road geometry, and a different label from the original Geolife requirement for at least five continuous minutes.

{budgets}

## Data, Labels, and Evaluation Support

The fixed snapshot of official dataset i4gi-tjb9 contains 54 shards and 1,019,891 raw rows. The latest data_as_of is 2026-11-11T17:12:06. See evidence/source_snapshot.json and registration_dynamic.json for paths, queries, and hashes. The graph has 121 fixed road segments, 56 candidate edges, and 54 isolated roads. Dates use NYC local time. The last source day was incomplete. All scheduled windows were retained, with unknown labels where the evidence was insufficient.

{table(['Split','Dates (end exclusive)','Days','Known windows','Positive','Negative','Unknown','Positive events'],[[r['part'],r['start']+' → '+r['stop_exclusive'],r['days'],r['known'],r['positive'],r['negative'],r['unknown'],r['positive_events']] for r in cohort])}

Supervised training uses every known window and sees each training ID once per epoch. Unknown labels remain separate from negatives. Within each development and final stage, all three poolings and three seeds use identical training IDs, normalization, evaluation sample keys, and labels. Each stage has a 12-epoch budget. The test set has 80,536 known windows, with 2,866 positives ({2866/80536:.4%}). Coverage is 80,536 of 130,680 windows ({80536/130680:.4%}). Each model also scores all 50,144 unknown test windows. The main metrics exclude those windows.

The original full evaluation protocol supplies the time splits and window boundaries. Roads overlap across splits, so the evaluation measures forward-in-time generalization for known roads. The public table contains aggregate segment speed estimates and no person or trajectory IDs. Generalization across roads or people and separation by person remain outside the evaluation. Observation arrival delays remain an offline assumption.

## Implementation and Paired Results

See method_and_data_contract.md for the dynamic edge formula, tensors, direction, missingness and self-loop behavior, and complexity. The implementation is in project/dynamic_model.py. Existing parameters, optimization, and full-sample training follow the static protocol. The two edge MLP layers add 1,704 parameters. With zero gate initialization, the dynamic and static outputs matched exactly in testing.

These backtest results use **false-positive-constrained F1 thresholds frozen on calibration data**. The three seeds are 11, 22, and 33. SD is sample standard deviation. AP uses sklearn.average_precision_score and the original AUC definition remains separate.

{metric_table}

{delta_table}

The tables include gains and declines in AP and F1. Pooling, seeds, and thresholds were chosen independently of the test outcomes. The selected model's AP difference from the static model with the same pooling is {p['AP']-baseline['AP']:+.4f}. Its F1 difference is {p['f1']-baseline['f1']:+.4f}. The previously selected static Max is reported separately because comparisons across poolings combine multiple effects.

## Recovery False Positives at Matched Recall

The principal diagnostic compares scores at 60% recall over every known test window with the same true-positive count. Ties are resolved by sample_id. This retrospective ranking diagnostic does not provide an operating threshold. See matched_recall.csv for 50%, 70%, and 80% recall and the t−2 low-speed subgroup.

{matched_table}

{' '.join(signs)}

{interval_table}

The date-block bootstrap uses 1,000 resamples with shared date weights. Its intervals are conditional on the fitted models, fixed thresholds, and observed dates. They exclude retraining and new-road population uncertainty and do not adjust for multiple comparisons. An interval crossing zero leaves the direction uncertain. AP is outside these bootstrap intervals. Its three-seed SD appears in the main table.

## False Positives, Graph Structure, and Edge Weights

results/groups.csv separates windows with low, regular, or missing speed at t−2 and reports AP, F1, confusion counts, and recovery false positives. topology_groups.csv separates connected and isolated roads. daily_errors.csv gives counts by date. operating_points.csv retains event-level results using the original grouping of consecutive 10-minute bins on the same road. These counts are separate from window-level metrics.

With seed 1811, the edge-weight audit randomly sampled up to 1,024 connected assessment graphs independently of labels and saved them under results/edge_weights/. edge_multiplier_summary.csv reports medians, quantiles, bounds, and the fallback-to-1 share for physical edges. edge_neutralization_diagnostic.csv turns off only the edge prior on **the same trained backbone** without refitting the threshold. It tests the mechanism independently of paired training and model selection.

The 54 isolated roads and relatively few geometric candidate edges limit the reach of edge weights. Toggling the edge prior does not affect an isolated graph on a fixed backbone. Retraining can change the shared parameters, so isolated-group differences across models do not measure direct edge propagation. Endpoint proximity in the road geometry, conflicting metadata, estimated speeds, and biases in missing observations remain limitations.

## Freezing, Execution, and Test Evidence

Development selection followed {dev['selection_rule']}. Final epochs: {dev['epochs']}. Weights and thresholds were frozen at {freeze['frozen_utc']} UTC. The freeze record SHA-256 is {sha(root/'evidence/dynamic_threshold_freeze.json')}. Threshold fitting used only the separate 28-day calibration split. Development fixed the epoch budget, and final training ran all 12 epochs. Final dynamic training saved calibration predictions first and scored the test split after freezing. Static test predictions came from the previously disclosed read-only controls.

All {tests['total']} engineering tests passed, including 12 new dynamic tests. Synthetic fixtures cover forward and backward passes, overfitting, graph structure, pooling, batch isolation, center mapping, direction, missing-value fallback, weight bounds, safe persistence, and invariance of current features, weights, and scores when modifying t−1, t, or future data. Independent recomputation covered {verify['operating_points']} operating points, {verify['matched_points']} matched-recall points, and {verify['group_metrics']} subgroup metrics. Sampled replay covered all 36 models, 18 reused static and 18 new dynamic, with maximum probability difference {verify['maximum_probability_error']}. Static headline metrics matched the original package exactly.

CPU: {environment['cpu']}. Python: {environment['python'].splitlines()[0]}. Torch: {environment['packages']['torch']}. Each process used {cfg['cpu_threads']} threads, with up to {cfg['workers']} tasks. The sum of fit-and-score durations was {new.fit_and_score_seconds.sum():.1f} seconds across parallel tasks. See training_resources.csv for resources by model and logs/ for actual commands, exit codes, durations, and errors. The experiment used the existing Conda environment and dependencies. Weight loading uses weights_only=True and NPZ loading uses allow_pickle=False.

A second full run of the 18-model training matrix in a clean directory remains outside this audit. Clean extraction verified manifests and hashes, recomputed metrics, replayed weights, and ran tests. See extraction_verification.json at the delivery root. {'Prepackaging checks passed for recomputation and replay in a clean directory. See evidence/clean_qa_prepackage.json.' if clean else 'Refer to extraction_verification.json for the final status because the prepackaging copy check has not been registered.'}

## Verification and Delivery

1. Check the external ZIP SHA-256, manifest.csv, and SHA256SUMS. The checksum file includes the manifest and excludes itself.
2. Use evidence/path_mapping.json to find the read-only original audit package and prepared arrays. Review each shard's query, row count, and SHA-256 against source_snapshot.json. Source terms and collection evidence are in source_provenance/.
3. Verify the plan, configuration, source, prepared arrays, and frozen weight hashes. Check the ordering of selection, final training, threshold freezing, and test scoring. An independent future test remains pending.
4. Recompute thresholds and full metrics from assessment.npz, training_ids.npz, and curve.csv in results/models and from final dynamic predictions in results/evaluation. Verify coverage, unknowns, events, and per-day denominators across all eligible samples, including windows without risk flags or edges.
5. Run project/run_dynamic.py --phase verify in a separate output directory. Then load the hash-verified training preprocessors and weights_only weights for sampled replay.

{commands}

For the course defense, describe this as a measured full historical NYC backtest of explicit dynamic edge priors in a directed Graph Transformer with three pooling methods, three seeds, the original proxy labels, and a strict 10-minute lead. Report gains, declines, and false-positive budgets individually. A new blind test remains pending, and the original Geolife and Chicago experiments remain blocked.
'''
    text(root,'audit_report.md',audit)
    text(root,'README_dynamic_edge_weights.md',f'''# Project B Dynamic Edge Weight Handoff

The full NYC historical backtest completed 18 new dynamic fits and paired them with 18 verified, reused static models. Development selected **{primary}**. AP was {pm(p,'AP')} and positive-class F1 was {pm(p,'f1')}. The model **{outcome}**. It recorded {p['FP_day']:.3f} total false positives per day and {p['recovery_FP_day']:.3f} recovery false positives per day. See the audit report for separate false-positive budget results.

The test dates had already been disclosed. An independent blind test remains pending. The original Geolife and Chicago protocols remain blocked. The experiment uses the existing NYC snapshot with the original labels, full difficult-window coverage, and the original accuracy definition.

{metric_table}

Start with audit_report.md, method_and_data_contract.md, and plan.md. Review the results in results/operating_points.csv, matched_recall.csv, and paired_date_blocks.csv. The freeze receipt is evidence/dynamic_threshold_freeze.json. See logs/ and test_report.md for execution and tests.

The code entry point is project/run_dynamic.py. dynamic_model.py implements bounded edge weights, dynamic_train.py runs full training, dynamic_score.py handles selection, freezing, and backtesting, and dynamic_verify.py independently recomputes and replays results. Configuration is in project/config.json. Original input preprocessing is available in project/full_prepare.py and its inherited modules.

{commands}

Offline recomputation uses bundled predictions and the original Python dependencies. Sampled replay also requires the two read-only arrays listed in evidence/prepared.json. A new training run needs the original full audit package and actual raw snapshot paths. The original large public table, private Geolife trajectories, and old large archives are outside this handoff. Verify hashes before replay. Dependency versions are in logs/environment_dynamic.json and project/requirements_actual.txt.

manifest.csv lists all files except itself and SHA256SUMS. SHA256SUMS includes the manifest and excludes itself. The extraction verification receipt and ZIP hash are stored alongside the archive to avoid self-reference.
''')
    text(root,'test_report.md',f'''# Testing and Review Report

All {tests['total']} engineering tests passed. The root suite ran 19 tests, seven original and 12 new. Inherited suites ran 56. See logs/*engineering_dynamic*.json and the corresponding logs for commands, standard output, exit codes, and timings. Synthetic fixtures provide engineering evidence. See synthetic/engineering_status.json.

New tests cover outputs matching static weights at zero across three poolings, directed edges, self-loops, padding, bounds, missing-endpoint fallback to 1, effects of nonunit weights and structural masks, and identical isolated-graph outputs when toggling weights. They also cover batch isolation and node permutation with center mapping across the three poolings, rejection of empty graphs, invalid centers, and NaN, backward passes and weights_only save/load across all three poolings, single-batch overfitting, and invariance of current features, weights, and scores when changing the label bins or future observations, including at neighboring roads.

Real-data verification passed for {verify['operating_points']} operating points, {verify['matched_points']} matched-recall points, and {verify['group_metrics']} subgroup metrics. Sampled replay passed for 36 models, with a maximum probability difference of {verify['maximum_probability_error']}. Same-support training IDs matched exactly, and the numbers of unique known samples and updates per epoch agreed. Supervised evaluation excludes unknown labels. Static metrics match the original package exactly. See evidence/verification_dynamic_replay.json.

The static review covered data and artifact handling, compatibility, error handling, resource cleanup, change scope, types, and shapes. Dynamic modules were added alongside the original code. Replay verifies file hashes and uses pickle-free inputs. Float32 logits and probabilities are promoted exactly to float64 scores. See reports/change_notes.md for review findings. Python assertions were enabled.

This audit covers clean extraction, recomputation, and replay as recorded in extraction_verification.json. A second clean training matrix, a blind independent test on new dates, and new Chicago or Geolife training remain pending. Generalization to new dates remains unverified.
''')
    mapping=load(root/'evidence/path_mapping.json')
    source=Path(mapping['source_bundle'])
    changes=f'''# Changes from the Original Full Experiment

The original package SHA-256 was `0436558b017bb6c5d52d0ab2103ae52f2315b970e4a9b9cf615c7b4a9f0c1c47`. The original project and audit package were read-only. Implementation used an independent copy of the source.

Added `dynamic_model.py`, `dynamic_check.py`, `dynamic_train.py`, `dynamic_score.py`, `dynamic_analysis.py`, `dynamic_verify.py`, `dynamic_report.py`, `run_dynamic.py`, `finish_dynamic.py`, and `tests/test_dynamic.py`. The configuration registers one dynamic-edge recipe with a three-pooling matrix. The baseline model, features, graph, and labels retained the original implementation. Each layer adds 852 parameters and uses the existing feature cutoffs and roads.

Copied 18 static graph models for both stages from the original package. Their hashes are recorded in `evidence/registration_dynamic.json`, and all were safely replayed on sampled inputs. Existing center-only and other baseline aggregates remain in `evidence/static_reference_summary.json` for historical same-support reference. This run added 18 dynamic fits.

After training began, a one-time NumPy load in `dynamic_train.py` was wrapped in a context manager to release its file handle. Computation, sampling, models, and configuration stayed within the registered protocol. All selection rules followed the registered plan. The new `--no-replay` option supports offline metric recomputation from bundled predictions when external arrays are unavailable.

The initial manual launch failed while resolving its relative working directory, before the check or training phases began. Retrying from the absolute project parent succeeded. The recorded commands capture the actual check and subsequent phases.
'''
    text(root,'reports/change_notes.md',changes)
    if source.exists():
        diff=''.join(difflib.unified_diff((source/'project/config.json').read_text(encoding='utf-8').splitlines(True),
              (CODE/'config.json').read_text(encoding='utf-8').splitlines(True),fromfile='original/config.json',tofile='dynamic/config.json'))
        text(root,'reports/config.diff',diff)
    text(root,'project/requirements_actual.txt','\n'.join(f'{k}=={v}' for k,v in environment['packages'].items()))
    save(root/'results/delivery_summary.json',dict(primary=primary,AP=p['AP'],AP_sd=p['AP_sd'],F1=p['f1'],F1_sd=p['f1_sd'],
        balanced_accuracy=p['balanced_accuracy'],recall=p['recall'],FP_day=p['FP_day'],recovery_FP_day=p['recovery_FP_day'],
        goal_met=p['AP']>=.6 and p['f1']>=.6,total_FP_budget_met=p['FP_day']<=12,recovery_FP_budget_met=p['recovery_FP_day']<=6,
        independent_validation=False,new_dynamic_fits=18,reused_static_fits=18,static_pair_AP_delta=p['AP']-baseline['AP'],
        static_pair_F1_delta=p['f1']-baseline['f1']))
    print(dict(primary=primary,report='audit_report.md',engineering_tests=tests['total'],goal=outcome),flush=True)


if __name__=='__main__':report(*settings())

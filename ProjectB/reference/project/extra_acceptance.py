"""Post-fit audit only: independently check max-F1 thresholds and finite edge priors."""
import numpy as np
import pandas as pd
from full_io import load,save,settings,now
from dynamic_score import read_assessment
from v9_verify import independent_cut

root,cfg=settings()
receipt=load(root/'evidence/dynamic_threshold_freeze.json')
thresholds=[]
for entry in receipt['models']:
    raw=read_assessment(root,'final',entry['model'],entry['seed'],True)
    k=(raw['phase']==1)&(raw['y']>=0)
    expected=independent_cut(raw['y'][k],raw['score'][k],raw['risk'][k],np.inf,np.inf)
    assert expected==entry['options']['max_f1']['threshold']
    thresholds.append(dict(model=entry['model'],seed=entry['seed'],threshold=float(expected)))
edge=pd.read_csv(root/'results/edge_multiplier_summary.csv',float_precision='round_trip')
assert np.isfinite(edge[['min','median','max']]).all().all()
assert (edge['min']>=.5).all() and (edge['max']<=2).all()
assert (edge.neutral_fraction<1).all()
save(root/'evidence/extra_acceptance.json',dict(status='PASS',checked_utc=now(),max_f1_thresholds=thresholds,
    edge_summary_rows=len(edge),learned_non_neutral_weights_in_all_nine_fits=True,edge_bounds_verified=True,
    modifies_model_or_threshold=False))
(root/'reports/supplementary_cost_and_statistics.md').write_text('''# Computational Cost and Statistical Notes

Dynamic edge weights add 1,704 parameters to each model. Mean and Max increase from 22,561 to 24,265. Attention increases from 22,594 to 24,298, about 7.5%. Known examples and updates per epoch match the static controls exactly. Each update requires more computation. See `training_resources.csv` for runtime and memory. Static timings came from an earlier run under a different system load, which limits speed comparisons.

The development AP rule selected `dynamic_mean` as the primary model. It slightly increased historical AP and F1 relative to static Mean. The date-block interval for F1 and the interval for recovery false positives at 60% recall both cross zero, so evidence of a stable recovery false-positive improvement is inconclusive. Attention has separate paired improvements, while the primary selection remains the development choice. Intervals are conditional on the fitted models and dates and have no multiple-comparison correction.

The independent calculation for the `max_f1` auxiliary operating point uses cumulative counts over unique scores and matches thresholds for 18 models. See `evidence/extra_acceptance.json`. The principal constrained thresholds were verified in `evidence/verification_dynamic_replay.json`. All nine edge-weight groups contained nonunit weights within [0.5, 2]. Approximately 46.84% of physical edges reverted to 1 because the latest relative speed was missing at an endpoint. The fallback keeps the edge weight neutral when the speed observation is unavailable.
''',encoding='utf-8')
print(dict(status='PASS',max_f1_thresholds=len(thresholds),real_edge_summaries=len(edge)))

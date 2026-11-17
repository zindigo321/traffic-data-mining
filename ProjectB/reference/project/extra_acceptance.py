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
(root/'reports/成本与统计补充说明.md').write_text('''# 成本与统计说明

动态边权每个模型额外 1704 参数；Mean/Max 从 22561 到 24265，Attention 从 22594 到 24298，约增加 7.5%。每轮已知样本和更新数与静态完全一致，但每次更新的计算量增加，不能把相同步数说成完全相同 FLOPs。实际耗时、内存见 training_resources.csv；静态耗时来自旧运行，当时系统负载不同，不据此宣称动态模型更快。

主模型依登记开发 AP 仍为 dynamic_mean。其回测 AP/F1 对静态 Mean 小幅上升，F1 日期块区间跨零，60% 召回下恢复误报差的区间也跨零，因此尚无稳定恢复误报改善的证据。Attention 的配对结果另有改善，但未据测试改换主模型。全部区间条件于现有模型和日期，未作多重比较校正。

max_f1 辅助工作点另用独立唯一分数组累计方法重算，18 个模型阈值一致，见 evidence/extra_acceptance.json；主 constrained 阈值已在 verification_dynamic_replay.json 复核。真实边权汇总 9 组均有非单位边权，处于 [0.5,2]；物理边约 46.84% 因端点最新相对速度缺失退回 1，边权不会补齐缺失观测。
''',encoding='utf-8')
print(dict(status='PASS',max_f1_thresholds=len(thresholds),real_edge_summaries=len(edge)))

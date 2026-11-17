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
    metric_table=table(['配方','AP（均值±样本SD）','正类 F1（均值±样本SD）','平衡准确率','召回','总FP/日','恢复FP/日'],rows)
    matched_rows=[]
    for model in ALL:
        m=paired['2'][model]['matched60']
        matched_rows.append([model,precision(m['achieved_recall']),f"{m['FP_per_day']:.3f}",f"{m['recovery_FP_per_day']:.3f}"])
    matched_table=table(['配方','实际召回','总 FP/日','恢复 FP/日'],matched_rows)
    points=pd.read_csv(root/'results/matched_recall.csv',float_precision='round_trip')
    signs=[]
    deltas=[]
    for pool in ('mean','max','attention'):
        a=points[(points.model=='dynamic_'+pool)&(points.phase==2)&(points.scope=='all')&(points.target_recall==.6)].sort_values('seed')
        b=points[(points.model=='graph_'+pool)&(points.phase==2)&(points.scope=='all')&(points.target_recall==.6)].sort_values('seed')
        total=(a.FP_per_day.to_numpy()<b.FP_per_day.to_numpy())
        recovery=(a.recovery_FP_per_day.to_numpy()<b.recovery_FP_per_day.to_numpy())
        signs.append(f'{pool}：总误报下降 {int(total.sum())}/3 种子，恢复误报下降 {int(recovery.sum())}/3 种子，同时下降 {int((total&recovery).sum())}/3 种子。')
        ma=summary['2']['constrained']['dynamic_'+pool]
        mb=summary['2']['constrained']['graph_'+pool]
        deltas.append(['dynamic_'+pool+' − graph_'+pool,f"{ma['AP']-mb['AP']:+.4f}",f"{ma['f1']-mb['f1']:+.4f}",
            f"{a.FP_per_day.mean()-b.FP_per_day.mean():+.3f}",f"{a.recovery_FP_per_day.mean()-b.recovery_FP_per_day.mean():+.3f}"])
    delta_table=table(['同池化配对差','ΔAP','ΔF1','Δ等召回总FP/日','Δ等召回恢复FP/日'],deltas)
    intervals=pd.read_csv(root/'results/paired_date_blocks.csv',float_precision='round_trip')
    interval_rows=[[r.contrast,r.metric,f'{r.low95:+.4f}',f'{r.high95:+.4f}'] for r in intervals.itertuples()
        if r.phase==2 and r.metric in ('f1','FP60_day','recovery_FP60_day')]
    interval_table=table(['配对','指标','条件性95%下界','条件性95%上界'],interval_rows)
    environment=load(root/'logs/environment_dynamic.json')
    outcome='达到回测数值目标' if p['AP']>=.6 and p['f1']>=.6 else '未达到回测数值目标'
    budgets=table(['池化','回测 AP≥.60 且 F1≥.60（三种子均值）','总FP≤12/日','恢复FP≤6/日'],[
        [model,'是' if summary['2']['constrained'][model]['AP']>=.6 and summary['2']['constrained'][model]['f1']>=.6 else '否',
         '是' if summary['2']['constrained'][model]['FP_day']<=12 else '否',
         '是' if summary['2']['constrained'][model]['recovery_FP_day']<=6 else '否'] for model in DYNAMIC])
    resources=pd.read_csv(root/'results/training_resources.csv')
    new=resources[resources.new_training]
    clean_path=root/'evidence/clean_qa_prepackage.json'
    clean=load(clean_path) if clean_path.exists() else None
    commands='''```powershell
$python = 'D:/conda_envs/niulaoshi/python.exe'
# 在解压审计包根目录执行：仅依赖包内预测，独立重算指标与阈值
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_verify_new --no-replay
# 加上外部已核验 prepared 数组，抽样重放全部 36 个模型
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_replay_new
# 新目录完整重跑 18 次动态训练；静态 18 个模型仍从原审计包核验后复用
& $python -B -X utf8 project/run_dynamic.py --phase all --output G:/gpt-file/Codex/work/ProjectB_dynamic_refit_new
```
输出目录必须选择尚未使用的名称。分阶段命令将 `--phase all` 替换为 check、tests、development、select、final、freeze、evaluate、analysis、verify、report；初始化资料通过完整入口复制。不能用 check 覆盖既有注册收据。
'''
    method='''# 动态边权方法与数据时点合同

节点是原 121 条有向路段；静态边 source→target 的方向、56 条候选边、54 个孤立节点及一跳局部图不变。每批图独立，目标映射为局部节点 0；只从现有物理前驱和自身取注意力，池化只覆盖目标的局部子图。输入 [B,N,104] 为 52 个训练段填补/标准化值及其 52 个缺失位，N≤5。52 列名称见 evidence/cohort_final.json。没有新复制私人轨迹。

动态描述选取 lag20/30/40_relative_speed、lag20_minus_lag30_ratio、legal_usable_count、lag20_message_count、lag20_history_days、lag20_half_last_ratio。每端使用 8 个值（仅边描述裁剪到标准化 ±5）与 8 个原始缺失位，源/目标/两端绝对差拼成 48 维。最新动态信息来自 t−2 格，标签涉及 t−1、t 两格速度及未来观测完全不进入描述。t 为目标窗口起点，出分时点 t−10；t−2 格为 [t−20,t−10)。

对第 l 层、第 h 头、物理边 j→i：

`b[l,h,i,j] = log(2) * tanh(MLP_l([u_j,u_i,abs(u_j-u_i)]))[h]`；
`w[l,h,i,j] = exp(b[l,h,i,j]) ∈ [0.5,2]`；
`attention = softmax_j(Q_i K_j / sqrt(8) + b[l,h,i,j] + directed_mask)`。

MLP 为 48→16→4、GELU。每层额外 852 参数，两层共 1704；最后线性层零初始化使初始所有边权等于 1，且不扰动原骨干/分类器的随机初始化。self 边权恒为 1；任一端 lag20 相对速度缺失时退回 1；非物理边保持 mask 禁止。原 QK 已有随输入变化的注意力；本轮新增的是显式、可审计、由合法过去交通属性驱动的乘性边先验，不声称基线没有自适应注意力。所有动态参数只由训练标签学习。

动态边先验张量 [B,4,N,N]，层数为 2；物理 source→target 转置成 query target/key source。仍采用 dense 局部注意力，非稀疏算法：骨干每层 O(BN²D + BND²)，边 MLP 每层 O(BN²(48×16+16×4))，额外空间 O(BHN²)。边权不会产生新的路网边，也不连接跨批图。孤立图自环为 1，固定骨干下开关动态边权输出严格相同。

Mean：有效节点向量逐维均值；Max：有效节点逐维最大；Attention：a_i=softmax(vᵀh_i+c)，池化=sum(a_i h_i)。三者池化 O(BND)，目标节点向量与池化向量拼为 [B,64]，输出每个目标路段的二分类 logit。空图和无效中心拒绝；padding 不参加池化或有效边。

纽约本地时间、07–09/17–19、原双格低速/历史支持/unknown 定义全部继承。训练段过去信息形成历史基准并在后段冻结；标准化仅训练输入拟合。边权不使用当前真值、risk、后续标签、评估类别或采样结果。risk 为 t−1 是否低速，只用于独立校准误报约束和事后恢复误报诊断，未用于训练/路由/边权。

原数据来源、下载时间/查询/哈希由 source_snapshot.json 继承并实际重核；本轮没有联网获取续期数据，也没有重新检查许可网页。官方公开表及原采集条款证据见 source_provenance/；未对官方数据添加新的许可授权。只读速度聚合分片和 prepared 数组不入包，通过 evidence/path_mapping.json、prepared.json 映射；完整原始到指标流水线仍保留在 project/full_*.py。离线 data_as_of 没有独立到达时间，以原假设截止，不把这一假设说成实时可用性已经证实。
'''
    text(root,'方法与数据合同.md',method)
    audit=f'''# Project B 动态边权实验审计报告

## 结论和验收

真实 NYC 公开数据实验已完成。新增 18 次动态训练（三种池化 × 三种子 × 开发/最终），复用并核验 18 个静态模型；静态模型不计作本轮新训练。主模型由开发 AP 预选为 **{primary}**，三种子回测 AP={pm(p,'AP')}，正类 F1={pm(p,'f1')}，平衡准确率={precision(p['balanced_accuracy'])}，召回={precision(p['recall'])}，**{outcome}**。总 FP/日={p['FP_day']:.3f}，恢复 FP/日={p['recovery_FP_day']:.3f}；误报预算是否维持分别见下表。另报 ROC-AUC={precision(p['ROC_AUC'])}、Precision={precision(p['precision'])}、Accuracy={precision(p['accuracy'])}；在正例仅约 3.56% 时，Accuracy 不等同于正类识别效果。

原回测期结果早已公开，本轮是同支持时间回测消融；没有全新盲测，也没有通过此次修改补足未来独立留出。Geolife 原课程和 Chicago 原口径仍为 blocked/N/A。NYC 仅为公开代理任务，与原 Geolife 用户数、年份、北京路网和至少五分钟定义不能混称。

{budgets}

## 数据、标签与完整支持

官方数据 i4gi-tjb9，固定快照 54 个分片、1,019,891 原始行，最新 data_as_of=2026-11-11T17:12:06。路径/查询/哈希见 evidence/source_snapshot.json 与 registration_dynamic.json。固定道路数 121、候选边 56、孤立道路 54。日期使用 NYC 本地时间。最终源日不完整仍保留所有固定窗口，并将缺证据真值留为 unknown。

{table(['部分','日期（右端不含）','天','已知窗','正例','负例','unknown','正例事件段'],[[r['part'],r['start']+' → '+r['stop_exclusive'],r['days'],r['known'],r['positive'],r['negative'],r['unknown'],r['positive_events']] for r in cohort])}

所有监督训练使用全量已知窗口，每轮每条训练 ID 一次；unknown 不变成负类。三池化/三种子在开发和最终各自完全相同的训练 IDs、归一化、评估样本键和标签上比较；各阶段 12 轮执行预算保持一致。test known=80,536，正例=2,866，阳性率={2866/80536:.4%}；覆盖=80,536/130,680={80536/130680:.4%}。逐模型预测还包含 50,144 个 test unknown，全部出分，但主指标排除。

时间划分及窗口边界延续原全量协议。道路在各段重叠，任务是已知道路向后时间泛化，不是跨路段或跨人员实验。公开表为聚合路段估速，没有人员/轨迹 ID，不能声称已核验人员隔离。未知到达延迟仍是离线假设。

## 实现和本轮配对结果

动态边权公式、张量、方向、缺失/自环规则和复杂度见 方法与数据合同.md；核心实现 project/dynamic_model.py。所有参数/优化/全量采样与原静态保持一致，新增边 MLP 两层共 1704 参数。Gate 零初始化与静态输出精确一致已实际测试。

以下是**校准段冻结的误报约束 F1 阈值**在回测上的结果，均为 11/22/33 三种子，SD 为样本标准差；AP 是 sklearn.average_precision_score，没有改 AUC 定义。

{metric_table}

{delta_table}

AP 与 F1 具体改善或退化均如表保留；不会按测试挑选池化、种子或阈值。选定模型与同池化静态的 AP 差={p['AP']-baseline['AP']:+.4f}，F1 差={p['f1']-baseline['f1']:+.4f}。原预选静态 Max 仍单列，不能用不同池化差异全部归因于边权。

## 等召回恢复误报和统计局限

主诊断点为全量 test known 支持上的 60% 召回，固定相同 TP 数；分数并列按 sample_id 处理。它是事后排序诊断，不能视作已经可部署的测试阈值。50/70/80% 及 t−2 低速组单列在 matched_recall.csv。

{matched_table}

{' '.join(signs)}

{interval_table}

日期块 bootstrap 使用 1000 次共同日期权重，条件于现有模型、固定阈值和现有日期；不覆盖模型重训或新道路总体的不确定性，未作多重比较校正。区间跨零时不宣称稳定优势。AP 不在这些 bootstrap 区间之内，其三种子 SD 在主表。

## 误报、结构和边权证据

results/groups.csv 分 t−2 已低速、未低速、缺失组，含 AP、F1、混淆及恢复误报；topology_groups.csv 分 connected/isolated；daily_errors.csv 含逐日计数。事件级结果沿原同道路连续十分钟归并，直接在 operating_points.csv 中保留，与窗口级指标分开。

真实边权抽样固定种子 1811，按有连接的 assessment 图随机抽最多 1024 个，不按标签抽样；保存到 results/edge_weights/。edge_multiplier_summary.csv 仅统计物理边，含中位数、分位数、边界和退回 1 的比例。edge_neutralization_diagnostic.csv 在**同一训练后骨干**上仅关闭边先验且不重新拟合阈值，用于检验机制；它不是另一次配对训练，也不用于选型。

54 个孤立路段和几何候选边较少，限制边权作用范围。即使孤立图在同一固定骨干上开关无差异，重新训练共享骨干后全局参数会变化，不能把孤立组跨模型差异解释为直接边传播。原地图几何端点近邻、冲突元数据、估速和缺失分布偏差继续存在。

## 冻结、运行和测试证据

开发配方：{dev['selection_rule']}。最终 epoch：{dev['epochs']}。权重/阈值冻结 UTC={freeze['frozen_utc']}，SHA-256={sha(root/'evidence/dynamic_threshold_freeze.json')}。阈值只拟合 28 天独立校准段，不读当前 test 分数；开发定 epoch 后完整执行 12 轮。动态 final 训练只先保存 calibration 预测，在冻结后才生成 test 分数；静态 test 分数来自此前已公开的只读对照。

{tests['total']} 项工程测试通过，其中新增 12 项动态测试。工程样本仅测试前向、反向、过拟合、图/池化/批次/中心映射、方向、缺失退回、权重范围、安全保存和改写 t−1/t/未来观测不改变当前特征/边权/分数；不将合成分数列为真实成绩。独立重算 {verify['operating_points']} 个工作点、{verify['matched_points']} 个等召回点、{verify['group_metrics']} 个分组指标；抽样重放全部 36 个模型（18 复用静态，18 新动态），最大概率差={verify['maximum_probability_error']}。静态主指标逐值等同旧包。

CPU={environment['cpu']}，Python={environment['python'].splitlines()[0]}，torch={environment['packages']['torch']}，每进程 {cfg['cpu_threads']} 线程、最多 {cfg['workers']} 任务。新训练 fit+score 时长之和={new.fit_and_score_seconds.sum():.1f} 秒（并行之和，非总墙钟），每模型资源详见 training_resources.csv，真实命令、退出码、用时、错误保存在 logs/。没有修改现有 Conda 环境，没有安装依赖。权重仅 weights_only=True，NPZ allow_pickle=False。

本轮没有在干净目录第二次完整训练 18 个模型；干净解压只复核清单/哈希、重新算指标、重放权重和运行测试，范围见交付根目录 解压复核.json。{'封包前干净文件夹重算和重放已通过，见 evidence/clean_qa_prepackage.json。' if clean else '封包前复制复核状态尚未登记，最终以解压复核.json 为准。'}

## 复核路径和交付

1. 核验 ZIP 外部 SHA，再核验 manifest.csv、SHA256SUMS。其自身不循环包含；清单哈希在 SHA256SUMS。
2. 按 evidence/path_mapping.json 定位只读原审计包和 prepared 数组，按 source_snapshot.json 逐分片复核查询、行数、SHA。来源采集与条款证据在 source_provenance/。
3. 核验 plan、config、source、prepared 与冻结权重哈希，检查选择/最终训练/阈值冻结/测试评分的时间顺序。未来独立测试仍未执行。
4. 从 results/models 的 assessment.npz、training_ids.npz、curve.csv 及 results/evaluation 的动态最终预测，复算阈值与全量指标，核对 coverage/unknown/事件和日分母；不得只取高风险或有边的容易样本。
5. 通过 project/run_dynamic.py --phase verify 在独立输出目录复算，再加载已哈希的训练预处理器和 weights_only 权重抽样重放。

{commands}

答辩可用表述：在 NYC 原代理标签和严格提前十分钟口径的全量时间回测中，完成有向 Graph Transformer 三池化、三种子的动态边先验消融；结果为实测，提升/退化及误报预算分别报告。公开回测不能替代未来盲测，原 Geolife/Chicago 阻塞未被本实验解除。
'''
    text(root,'审计报告.md',audit)
    text(root,'README_动态边权入口.md',f'''# Project B 动态边权实验交付

真实 NYC 全量时间回测已完成：18 次新动态训练，18 个已核验静态模型配对复用。主模型 {primary}；AP={pm(p,'AP')}，正类 F1={pm(p,'f1')}，{outcome}。总FP/日={p['FP_day']:.3f}，恢复FP/日={p['recovery_FP_day']:.3f}。误报预算在回测是否成立见审计报告，数值达标不等于预算达标。

原测试已公开，这是回测消融；新独立盲测尚未完成。Geolife 与 Chicago 原口径继续 blocked。本轮沿用已有纽约快照，没有修改标签、剔除困难测试窗口或降低准确率来凑数。

{metric_table}

阅读顺序：审计报告.md → 方法与数据合同.md → plan.md → results/operating_points.csv、matched_recall.csv、paired_date_blocks.csv → evidence/dynamic_threshold_freeze.json → logs/ → 测试报告.md。

代码：project/dynamic_model.py（有界边权），dynamic_train.py（全量训练），dynamic_score.py（开发选择/校准冻结/回测），dynamic_verify.py（独立复算与安全重放），run_dynamic.py（一键入口）。配置 project/config.json；原输入预处理模块保留在 project/full_prepare.py 及继承代码内。

{commands}

离线指标复算只需包内预测及原 Python 依赖；重放还需 evidence/prepared.json 中两份只读数组；完整新训练还需原全量审计包及其实际原始快照路径。原始公开大表、私人 Geolife 轨迹和旧巨大压缩包未复制。先核验文件哈希后才能重放。依赖实际版本见 logs/environment_dynamic.json 和 project/requirements_actual.txt。

清单 manifest.csv 枚举除自身和 SHA256SUMS 外所有文件；SHA256SUMS 包含 manifest，不包含自身。解压复核收据及 ZIP 整体哈希放在交付目录外层，避免自指循环。
''')
    text(root,'测试报告.md',f'''# 测试与审查报告

实际工程测试 {tests['total']} 项通过：根 tests 19 项（原 7 + 新增动态 12），其余继承 56。原命令、stdout、退出码和用时见 logs/*engineering_dynamic*.json/log。合成数据仅工程用途，见 synthetic/engineering_status.json。

新增覆盖：零权重等于静态（三池化）、有向边坐标/自环/padding/范围、缺失端点退回 1、非单位边权确实改变输出、结构 mask 影响输出、孤立图开关一致、batch 隔离（三池化）、节点重排与中心映射（三池化）、空图/无效中心/NaN 拒绝、反向与 weights_only 保存加载（三池化）、单批过拟合、改写标签两格与未来（含邻居）不能改变当前特征/边权/分数。

真实验收：{verify['operating_points']} 工作点、{verify['matched_points']} 等召回点、{verify['group_metrics']} 分组复算通过；模型 36 个抽样重放通过，最大概率差 {verify['maximum_probability_error']}。同支持训练 IDs 完全相同、每轮唯一已知样本数和更新预算一致、unknown 未进入监督评价；静态汇总精确等于旧包。见 evidence/verification_dynamic_replay.json。

静态审查按 code-review 技能的安全、破坏性变更、错误处理、资源释放、规范/变更纪律、类型/形状检查执行。保留原源文件，新增动态模块；没有凭据、不执行 pickle，路径重放须哈希校验；数值精度 float32 logits/probability，float64 score 由 float32 精确提升。规范问题修复记录见 reports/变更说明.md。未使用 Python assert 的优化模式执行。

未执行：第二次完整干净训练矩阵、全新未来独立盲测、Chicago/Geolife 新真实训练。干净解压复算/重放范围另见 解压复核.json。回测结果不足以确认对新日期总体的稳定收益。
''')
    mapping=load(root/'evidence/path_mapping.json')
    source=Path(mapping['source_bundle'])
    changes=f'''# 相对原全量实验的变更

原包 SHA-256：0436558b017bb6c5d52d0ab2103ae52f2315b970e4a9b9cf615c7b4a9f0c1c47。原项目和包只读，代码独立复制。

新增 dynamic_model/check/train/score/analysis/verify/report.py、run_dynamic.py、finish_dynamic.py 和 tests/test_dynamic.py；config.json 增加唯一 dynamic_edge 配方和三池化矩阵。原模型/特征/图/标签实现未改。每层增加 852 参数；不新增特征时点或道路。

静态两阶段 18 个 Graph 模型从原包复制，哈希在 evidence/registration_dynamic.json，均安全抽样重放，不声称本轮重新训练。旧中心路段等基线完整聚合保留在 evidence/static_reference_summary.json，只作历史同支持参考；本轮新增 fit 仅动态 18 个。

训练开启后将 dynamic_train.py 单次 np.load 检查改为 with 上下文释放文件句柄，不改运算、采样、模型或配置。新增入口支持只依赖包内预测的 --no-replay，避免外部大数据不在时阻断离线指标复算。所有选择规则保持登记计划。

初次手工启动用相对工作目录解析失败（尚未运行 check 或训练），随后改用项目父目录绝对路径并成功；实际实验 check 和以后命令均由 record 保存。保留事实记录，不将失败视为成功实验。
'''
    text(root,'reports/变更说明.md',changes)
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
    print(dict(primary=primary,report='审计报告.md',engineering_tests=tests['total'],goal=outcome),flush=True)


if __name__=='__main__':report(*settings())

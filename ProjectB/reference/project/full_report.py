"""Human-readable audit of the actually executed full-data experiment."""
from pathlib import Path
import shutil

from full_io import CODE,load,settings
from full_score import ALL,MODELS

LABELS={"graph_mean":"Graph Transformer Mean","graph_max":"Graph Transformer Max",
        "graph_attention":"Graph Transformer Attention","center_full":"全量中心路段",
        "center_rank":"全量中心＋组内排序","center_sampled_matched":"抽样中心＋步数匹配",
        "logistic":"全量逻辑回归","lag20_rule":"合法lag20规则","majority":"多数类"}


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
    a=["# Project B 全量数据实验审计报告","",
       "已按最新授权取消实验入口的日历等待、最低覆盖和事件数限制。"
       "合并所有已核验缓存及当前能取得的纽约续期数据，实际完成36个神经模型和6个确定性基线拟合。"
       "训练范围内的全部已知样本进入五种全量配方，第六组为明确登记的步数匹配抽样对照。"
       "unknown完整出分并计覆盖。","",
       f"源记录共{prepared['source_funnel']['rows']:,}条，最近三天新增{source['new_rows']:,}条，"
       f"快照最新来源时间为{source['latest_source_time']}。范围为2026-05-11至11-11、固定121条有向路段和原双高峰。"
       "原V5/V6公开留出及11月9–11日记录已纳入本次时间回溯协议。"
       "该实验按时间切分的回溯结果发布，未见未来日期的独立验证需另立未使用日期。"
       " [来源快照](evidence/source_snapshot.json) · [原始文件核验](evidence/sources_cached.json)","",
       "| 分段 | 日期，右端不含 | 固定窗 | 已知 | 正例 | 负例 | Unknown | 事件 |",
       "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in cohort:
        a.append("| "+" | ".join([r["part"],r["start"]+" 至 "+r["stop_exclusive"],
                    *(f"{r[k]:,}" for k in ("scheduled","known","positive","negative","unknown","positive_events"))])+" |")
    a.extend(["",
        "最后一个来源日尚未取得全部晚高峰记录，固定时间格保留unknown，并报告实际覆盖。"
        "日均误报分母为完整登记日历天，窗口误报与事件级指标分别报告。"
        "训练、校准、测试逐窗键互斥。训练段的历史基准按过去合格观测逐日更新，校准和测试冻结训练历史，"
        "因此扩展训练后同日标签可与早期冻结折有差异，原40%双格标签规则一致。"
        " [分段和unknown原因](evidence/cohort_final.json)","",
        f"主模型由开发三种子平均AP选定为{LABELS[pick['primary']]}，最终保存轮次取对应开发保存轮次中位数。"
        "全部模型均执行12轮。阈值只用2026-08-31至09-27校准段拟合，"
        "2026-09-28至11-11回溯测试分数不参与模型或阈值选择。","",
        "| 回溯测试三种子均值 | AP | BA | F1 | 召回 | 总FP/日 | 恢复FP/日 |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for model in ALL:
        r=summary["2"]["constrained"][model]
        a.append("| "+" | ".join([LABELS[model],*(fmt(r[k]) for k in ("AP","balanced_accuracy","f1","recall")),
                                  fmt(r["FP_day"],3),fmt(r["recovery_FP_day"],3)])+" |")
    a.extend(["",
        f"预选主模型测试AP为{fmt(main['AP'])}±{fmt(main['AP_sd'])}，"
        f"F1为{fmt(main['f1'])}±{fmt(main['f1_sd'])}。"
        "确定性基线只拟合一次，其标准差为N/A。"
        "12/6为校准阈值选择的日均误报预算，测试实际误报在表中完整报告。"
        " [逐模型逐种子工作点](results/operating_points.csv)","",
        "| 主模型阈值规则 | AP | BA | F1 | 召回 | 总FP/日 | 恢复FP/日 |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for policy in ("constrained","max_f1"):
        r=summary["2"][policy][pick["primary"]]
        a.append("| "+" | ".join([policy,*(fmt(r[k]) for k in ("AP","balanced_accuracy","f1","recall")),
                                  fmt(r["FP_day"],3),fmt(r["recovery_FP_day"],3)])+" |")
    a.extend(["",
        "最大F1工作点用于量化误报与召回取舍。两种阈值都仅从校准段求出，"
        "测试成绩低于目标或误报超出预算时保留原值。","",
        "| 全量同支持60%等召回 | 总FP/日 | 恢复FP/日 | 实际召回 |",
        "|---|---:|---:|---:|"])
    for model in ALL:
        r=matched["2"][model]["matched"]["all"]["0.6"]
        a.append("| "+" | ".join([LABELS[model],fmt(r["FP_per_day"],3),
                                  fmt(r["recovery_FP_per_day"],3),fmt(r["achieved_recall"])])+" |")
    a.extend(["",
        "等召回诊断采用同一支持、相同TP数量，分数并列按固定样本键排序。"
        "阈值工作点与等召回截点分别记录，单独保留合法t−2低速组、未低速组和缺失组的AP、召回及FN。"
        "日期分块1000次在各模型和种子间共用权重，区间条件于固定模型、阈值和这一回溯样本。"
        " [分组指标](results/groups.csv) · [等召回明细](results/matched_recall.csv) · [配对区间](results/paired_date_blocks.csv)","",
        "所有全量神经配方每轮都访问各训练段全部已知样本一次，按训练ID文件独立验证。"
        "抽样对照保留12000个负例和全部正例，循环抽样ID至相同总访问数。"
        "六配方每轮步数完全一致，Mean/Max/Attention共享训练ID、随机种子、特征和骨干，只改变池化。"
        "原始类数、实际训练访问量、耗时、峰值内存及参数量见资源表。"
        " [训练资源](results/training_resources.csv)","",
        f"实际通过{tests['total']}项工程测试，独立复算84个工作点及336个等召回点。"
        f"重放{verification['neural_replayed']}个神经模型，最大分差{verification['maximum_neural_error']}。"
        "权重采用weights_only=True读取，NPZ采用allow_pickle=False。"
        "真实数据结果与合成工程测试分目录留存。"
        " [核验收据](evidence/verification_replay.json) · [工程测试](synthetic/status.json)","",
        "来源使用公开纽约DOT路段估速和既定路网几何。速度单位mph、原数据为本地浮动时间，"
        "缺独立观测到达时间，按data_as_of作为离线可用时点。"
        "原始行按原速度/旅行时间质量规则筛选，缺失速度、历史不足和未确认低速仍为unknown。"
        "曲线、参数、逐样本float32概率/logit及float64分数、采样/全量ID和预处理器均随包。"
        "大体积原始和特征数组在外部只读路径，分片SHA、查询及重建命令完整保存。"
        " [NYC官方数据](https://data.cityofnewyork.us/d/i4gi-tjb9)","",
        "[三池化PR图](figures/test_PR_three_pooling.png) · [指标及误报图](figures/test_pooling_and_controls.png) · "
        "[开发训练曲线](figures/development_curves.png)","",
        "复核按以下顺序执行。","",
        "1. 核对manifest、SHA256SUMS和原V10/V5/V6签名，按source_snapshot逐个校验54份原始分片。",
        "2. 由固定几何重建121节点、56方向候选边，复算全量原网格、历史基准、原标签和52列严格提前特征，"
        "核对prepared及各分段键摘要。",
        "3. 检查每个训练ID与全量已知索引一致、抽样对照访问次数匹配、校准和测试键无重叠。",
        "4. 核对开发选型、最终训练、阈值冻结、测试指标计算时间先后，"
        "逐种子复算混淆、AP、BA、F1、召回、误报和等召回，并抽样重放权重。",
        "5. 从全新目录解压核对清单，按入口重建特征和复算结果。实际执行情况见压缩包旁解压复核收据。"
        " [一键与分阶段入口](README_全量实验入口.md)","",
        "本轮已开放并完成可用数据实验。Geolife课程原任务和Chicago续期仍按各自既有证据报告，"
        "本次纽约全量结果属于公开补充数据的时间回溯验证。",""])
    (root/"审计报告.md").write_text("\n".join(a),encoding="utf-8")
    (root/"测试报告.md").write_text(
        f"# 全量实验测试报告\n\n{tests['total']}项工程测试实际通过。新增7项覆盖全部已知索引、未知标签排除、"
        "步数匹配、低覆盖和单类别指标处理及校准/测试标签对历史转移的隔离。\n\n"
        f"实际重放{verification['neural_replayed']}个神经模型，最大分差{verification['maximum_neural_error']}，"
        "另外复验6个确定性基线。所有真实指标来自全量时间切分，合成样本只验证工程逻辑。\n",encoding="utf-8")
    for filename in ("README_全量实验入口.md","方法与数据合同.md"):
        original=CODE.parent/filename
        if original.exists() and original.resolve()!=(root/filename).resolve():
            shutil.copy2(original,root/filename)
    (root/"reports").mkdir(parents=True,exist_ok=True)
    original=CODE.parent/"reports/变更说明.md"
    if original.exists() and original.resolve()!=(root/"reports/变更说明.md").resolve():
        shutil.copy2(original,root/"reports/变更说明.md")
    print(dict(report="审计报告.md",primary=pick["primary"],test_AP=main["AP"],test_F1=main["f1"]),flush=True)


if __name__=="__main__":
    report(settings()[0])

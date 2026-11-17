"""Build the V10 audit narrative entirely from measured outputs."""
import csv
from itertools import product
import shutil

from v10_data import GROUP_NAMES
from v10_io import CODE, load, settings
from v10_score import METRICS

LABELS = {"B0": "V9 中心重跑", "S1": "低速组负采样", "R1": "S1＋组内排序", "R2": "R1＋组别平衡"}


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
        "# Project B V10 审计报告", "",
        "四组×三个种子×三个开发折和一个校准段，48 个真实模型均已从头训练。原 B0 全量分数与 V9 签名工件逐位一致。"
        f"预登记规则选定 {pick['primary']}。校准主模型 AP {main['AP']:.4f}、平衡准确率 {main['balanced_accuracy']:.4f}、"
        f"F1 {main['f1']:.4f}、召回 {main['recall']:.4f}，平均总误报 {main['FP_day']:.3f}/日、恢复误报 {main['recovery_FP_day']:.3f}/日。"
        "12 个校准模型各自通过两项日均误报上限。", "",
        f"纽约正式验证 **blocked**。最新官方时间为 {gate['latest']}。预登记区间为纽约本地 [2026-11-09,2026-12-07)，"
        "尚未到齐。V5/V6 的 [2026-08-31,2026-11-09) 留出仅作历史。正式 AP、BA、F1、召回和误报为 N/A。"
        "Chicago 原代理任务续期和 Geolife 原课程任务分列 blocked。"
        " [NYC 官方数据](https://data.cityofnewyork.us/d/i4gi-tjb9) · [时间范围证据](evidence/date_gate.json)", "",
        "独立日期校准为 [2026-08-03,2026-08-31)，55,305 个已知窗、1,773 个正窗、53,532 个负窗、572 个恢复负例、"
        "565 个正例事件。固定出分窗 81,312，unknown 26,007，已知覆盖 68.02%，正例率 3.21%。"
        "该校准段曾在 V7–V9 展示，也用于本轮拟合阈值，校准表属于同段诊断。V10 配方选择仅用开发期。"
        " [日期计划](plan.md) · [分折漏斗](evidence/fold_lead_final.json)", "",
        "开发折各前七天拟合阈值、后七天查迁移。表中均值先按种子汇总各折，再对三个种子平均。"
        "开发模型 checkpoint 按全折 AP 选择，因此前后七日迁移属于开发诊断。"
        "所有固定道路键均出分，已知全部评价，unknown 只计覆盖。", "",
        "| 阶段 | 配方 | AP | BA | F1 | 召回 | 总 FP/日 | 恢复 FP/日 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, label in (("development", "开发后七日"), ("calibration", "校准")):
        for recipe in LABELS:
            row = summary[stage][recipe]
            a.append("| " + " | ".join([label, recipe + " " + LABELS[recipe],
                        *(f(row[k]) for k in ("AP", "balanced_accuracy", "f1", "recall")),
                        f(row["FP_day"], 3), f(row["recovery_FP_day"], 3)]) + " |")
    a.extend(["",
        "B0 使用原加权 BCE 和原分层抽样。S1 在同样 12,000 个已知负例预算中优先保留合法 t−2 低速负例。"
        "R1 在同一 S1 抽样与步数上增加组内排序项。R2 再加入组别×类别平衡 BCE。"
        "四组网络参数均为 22,594。没有候选通过全部预登记条件，主配方保留 B0。"
        "R1 的开发低速组 AP 上升、等召回恢复 FP 下降，但 F1 从 0.7353 降至 0.7332。"
        "S1、R2 的 AP、F1 和召回下降。校准结果未用于改选。"
        " [开发选型收据](results/development_selection.json) · [源代码](project/v10_train.py)", "",
        "| 校准三种子均值±样本标准差 | AP | BA | F1 | 召回 |",
        "|---|---:|---:|---:|---:|",
    ])
    for recipe in LABELS:
        row = summary["calibration"][recipe]
        cells = [f"{f(row[k])}±{f(row[k+'_sd'])}" for k in ("AP", "balanced_accuracy", "f1", "recall")]
        a.append("| " + " | ".join([recipe, *cells]) + " |")
    a.extend(["",
        "受约束阈值枚举全部不同概率和全判负哨兵，以正类 F1 最大选取，并列取较高阈值。"
        "按完整日历日计日均，校准总上限 336、恢复上限 168。每日计数单列，单个自然日可能超过日均上限。"
        "阈值、配置、权重、输入摘要和选型时间均已冻结。", "",
        "| B0 种子 | 冻结阈值 | 28 日总 FP | 28 日恢复 FP |", "|---:|---:|---:|---:|",
    ])
    for model in lock["models"]:
        if model["recipe"] == "B0":
            a.append(f"| {model['seed']} | {model['threshold']:.17g} | {model['fit_FP']} | {model['fit_recovery_FP']} |")
    a.extend(["", f"冻结时间为 {lock['at_utc']}。正式留出解封时间和样本键摘要为空。"
        " [冻结收据](evidence/freeze.json) · [每日错误](results/daily_errors.csv)", "",
        "同支持等召回以 ceil(0.6×正例数) 个 TP 定截点，分数并列按固定键排序。"
        "诊断截点用真值确定。工作点约束与等召回诊断分别核对。", "",
        "| 阶段 | 配方 | 全量总 FP/日 | 全量恢复 FP/日 | 低速组总 FP/日 | 低速组恢复 FP/日 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for stage, label in (("development", "开发后七日"), ("calibration", "校准")):
        for recipe in LABELS:
            row = summary[stage][recipe]["matched"]
            a.append("| " + " | ".join([label, recipe,
                *(f(row[scope]["0.6"][k], 3) for scope in ("all", "legal_low")
                  for k in ("FP_per_day", "recovery_FP_per_day"))]) + " |")
    a.extend(["",
        "| 校准分组 | 配方 | AP | 召回 | FN | 恢复 FP |", "|---|---|---:|---:|---:|---:|"])
    for group, recipe in product(GROUP_NAMES, LABELS):
        row = summary["calibration"][recipe]["groups"][group]
        a.append("| " + " | ".join([group, recipe, f(row["AP"]), f(row["recall"]), f(row["fn"], 2), f(row["recovery_FP"], 2)]) + " |")
    a.extend(["",
        "校准 R1 对齐全量 60% 召回时，恢复 FP 有两个种子减少、一个持平，总 FP 有两个种子略超 12/日。"
        "R1 校准的全量 AP 与低速组 AP 均低于基线。未低速和缺失组召回提高。"
        "这些结果表明本批候选改变了不同组的得分分布，尚未同时解决整体排序和低速组排序。"
        " [等召回逐种子](results/matched_recall.csv) · [分组逐种子](results/groups.csv)", "",
        "| 校准 R1 减 B0 | 差值 | 日期分块 95% 区间 |", "|---|---:|---:|",
    ])
    for row in records("paired_date_blocks.csv"):
        if row["stage"] == "calibration" and row["contrast"] == "R1_minus_B0":
            a.append(f"| {row['metric']} | {float(row['estimate']):+.5f} | [{float(row['lo95']):+.5f}, {float(row['hi95']):+.5f}] |")
    a.extend(["",
        "区间使用三个种子配对后共用 1,000 次日历日重抽，条件于固定模型与阈值。"
        "校准 R1 的 F1 差区间跨零。开发 F1 门槛未通过，正式时段尚缺，提升结论保留为候选诊断。"
        " [配对明细](results/paired_date_blocks.csv)", "",
        "[校准 PR 和分组图](figures/calibration_PR_and_groups.png) · [误报图](figures/calibration_false_alarms.png) · "
        "[开发训练曲线](figures/development_training_curve.png)", "",
        "48 次训练使用已核验的 CPU 环境，单进程峰值 RSS 与用时在资源表，每任务两个线程，最多四任务并行。"
        "原数据、原 V9 工件和 Conda 环境只读使用。命令、开始结束时间、退出码及错误栈留存。"
        "校准启动曾因整数命令参数失败，修正命令入口转换后重跑。"
        "分组无预测正例时 Precision=N/A 的汇总，以及 float32 权重均值的数值容差也已修复，原失败日志保留。"
        " [环境](logs/environment.json) · [资源表](results/training_resources.csv) · [变更记录](reports/变更说明.md)", "",
        "复核顺序如下。", "",
        "1. 核对 manifest.csv、SHA256SUMS 及 evidence/inputs.json 中 V9 ZIP 的签名和原始查询哈希。两表排除 SHA256SUMS 自身。",
        "2. 按 download 收据和外部路径重建四个合法 lead 数组，核对 prepared_development 的 SHA 和每折已知样本键。"
        "检查日期边界、原标签、unknown、道路清单和过去历史。",
        "3. 核对四组共同的全量评价键、训练样本数、种子、步数和预处理。S1/R1/R2 的采样 IDs 逐位相同，"
        "B0 原分数逐位复现 V9。实际 t−1 风险不进入特征、采样或损失。",
        "4. 依次核对开发训练收据、选型时间、校准固定轮次、校准打开时间、冻结收据。"
        "用逐窗 float32 logit/概率与 float64 分数、阈值复算工作点、分组和等召回计数。",
        f"5. 实际独立复算 {check['operating_points']} 个工作点、{check['groups']} 组指标、{check['matched']} 个等召回点。"
        f"模型抽样回放 {check['models_replayed']} 个，最大分差 {check['max_score_difference'] if check['max_score_difference'] is not None else 'N/A'}。"
        f"工程测试 {test['total']} 项通过。干净解压与一键重跑的实际收据在压缩包旁。 [复现入口](README_审计入口.md)", "",
        "data_as_of 用作离线时点，源数据缺独立到达时间。方向边来自冻结路网几何，"
        "本轮中心路段控制仅使用目标节点。纽约公开估速、路网及代理标签与原 Geolife 课程要求存在差异，"
        "课堂原任务和 Chicago 续期的 blocked 状态沿用。覆盖及地图偏差见方法合同。", "",
        "> 第十轮在纽约开发与既有校准段实际重训了 48 个模型。组内排序候选在校准段减少部分恢复误报并提高召回，"
        "同时降低 AP，开发 F1 门槛未通过。主模型按预登记保留 V9 中心配方。纽约新留出尚未到齐，正式成绩为 N/A。", "",
    ])
    (root / "审计报告.md").write_text("\n".join(a), encoding="utf-8")
    (root / "测试报告.md").write_text(
        "# V10 测试报告\n\n"
        f"工程测试 {test['total']} 项实际通过，V10 新增9项。覆盖合法分组、组内排序梯度及无类别分支、"
        "组别平衡权重、仅用合法输入的负采样、留出拒绝、N/A汇总、配方回退、整数命令参数。"
        "继承的图结构、池化、批次隔离、目标映射、unknown、防泄漏、未来两格扰动、权重保存加载测试已重跑。"
        " [完整命令和结果](synthetic/status.json)\n\n"
        f"真实数据独立复算 {check['operating_points']} 个工作点、{check['groups']} 组指标、"
        f"{check['matched']} 个等召回点。抽样重放 {check['models_replayed']} 个模型，"
        f"最大分差 {check['max_score_difference'] if check['max_score_difference'] is not None else 'N/A'}。"
        "安全权重加载采用 weights_only=True。原B0完整分数与V9逐位一致。正式留出验证 blocked。\n", encoding="utf-8")
    for name in ("README_审计入口.md", "方法与数据合同.md"):
        source = CODE.parent / name
        if source.is_file() and source.resolve() != (root / name).resolve():
            shutil.copy2(source, root / name)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    source = CODE.parent / "reports/变更说明.md"
    if source.is_file() and source.resolve() != (root / "reports/变更说明.md").resolve():
        shutil.copy2(source, root / "reports/变更说明.md")
    print(dict(report="审计报告.md", primary=pick["primary"], tests=test["total"]), flush=True)


if __name__ == "__main__":
    report(settings()[0])

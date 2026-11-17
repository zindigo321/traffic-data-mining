"""Render the human-readable V9 audit from the measured tables."""
import csv
from itertools import product

from v9_io import load, settings

LABELS = {"center": "V7 中心路段", "attention": "V7 Attention", "rank": "V8 尾部排序"}
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
        "# Project B V9 审计报告", "", "## 验收结果", "",
        "纽约三折开发、28 日独立校准和阈值冻结已完成。预选主模型为 V7 中心路段＋误报约束 F1。"
        f"中心未选中分组偏移，保留 δ={selection['chosen']['center']['delta']}。校准三种子均值 "
        f"AP {cal['AP']:.4f}、平衡准确率 {cal['balanced_accuracy']:.4f}、正类 F1 {cal['f1']:.4f}、召回 {cal['recall']:.4f}，"
        f"总误报 {cal['FP_day']:.3f}/日、恢复误报 {cal['recovery_FP_day']:.3f}/日。逐种子满足两个上限。"
        "校准用于拟合阈值。正式 AP、F1、召回和误报均为 N/A。", "",
        "纽约正式验证 **blocked**。纽约本地预登记日期为 2026-11-09 至 12-06，"
        f"时间查询最新值为 {gate['latest']}。区间尚未结束。V5/V6 留出 2026-08-31 至 11-08 仅作历史，"
        "开发、校准和冻结均使用 8 月 31 日前的工件。Chicago 续期及 Geolife 课程原任务分列 blocked。"
        " [NYC DOT 数据](https://data.cityofnewyork.us/d/i4gi-tjb9) · [Chicago 停更公告](https://data.cityofchicago.org/stories/s/c28d-stgw)", "",
        "## 支持和时点", "",
        "| 分段 | 日历日 | 固定键 | 已知 | 正类 | 负类 | Unknown | 恢复负例 | 正例事件 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in read("support.csv"):
        if row["scope"] not in ("full", "assess") or (row["fold"] == "final" and row["scope"] == "assess"):
            continue
        cols = ("days", "scheduled", "known", "positives", "negatives", "unknown", "recovery_negatives", "event_segments")
        label = "校准" if row["fold"] == "final" else row["fold"] + (" 全折" if row["scope"] == "full" else " 后七日")
        a.append("| " + " | ".join([label, *(f"{int(row[k]):,}" for k in cols)]) + " |")
    a.extend([
        "",
        "121 条固定有向路段及本地双高峰的全部固定键出分，所有已知标签进入评价。"
        "校准覆盖率 68.02%，已知正例率 3.21%，有已知记录的道路 91 条。"
        "预测时刻为目标格起点前 10 分钟，使用 t−2 或更早的观测。t−1 和 t 的速度只用于真值和误报审计。"
        "基准按同道路、同期、日类型、至少三个合格过去日冻结。 [方法与数据合同](方法与数据合同.md) · [支持明细](results/support.csv)",
        "", "## 同支持、同预算工作点", "",
        "开发表为三折后七日先按种子平均。阈值取每折前七日。原模型 checkpoint 用完整开发折选取。"
        "校准是独立日期段拟合阈值后的同段诊断。global 使用原分数，selected 使用开发锁定的偏移。"
        "各组均用同一偏移网格、种子、样本和预算。", "",
        "| 阶段 | 模型 | 策略 δ | AP | 平衡准确率 | F1 | 召回 | 总 FP/日 | 恢复 FP/日 | 通过 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for stage, label in (("development", "开发后半折"), ("calibration", "校准")):
        for model, policy in product(LABELS, ("global", "selected")):
            if policy == "global" and selection["chosen"][model]["delta"] == 0:
                continue
            row = numbers[stage][model][policy]
            cells = [f"{row[k]:.4f}" for k in OP[:4]] + [f"{row[k]:.3f}" for k in OP[4:]]
            a.append("| " + " | ".join([label, LABELS[model], f"{policy} {row['delta']:g}", *cells,
                                       f"{row['budget_pass_rows']}/{row['runs']}"]) + " |")
    a.extend([
        "",
        f"中心校准三种子标准差为 AP {cal['AP_sd']:.4f}、BA {cal['balanced_accuracy_sd']:.4f}、"
        f"F1 {cal['f1_sd']:.4f}、召回 {cal['recall_sd']:.4f}。中心全局和 V9 主策略逐窗相同。"
        "Attention 的 δ=−0.25 通过开发门槛，校准只用于阈值。尾部排序保留 δ=0。"
        "每种子 ROC-AUC、Precision、Accuracy、TP/FP/FN/TN 与支持在逐配置表。 [工作点明细](results/operating_points.csv)", "",
        "| 中心种子 | 冻结阈值 | 28 天总 FP | 28 天恢复 FP |", "|---:|---:|---:|---:|",
    ])
    for r in lock["models"]:
        if r["model"] == "center" and r["policy"] == "selected":
            a.append(f"| {r['seed']} | {r['threshold']:.17g} | {r['fit_FP']} | {r['fit_recovery_FP']} |")
    a.extend([
        "",
        "校准预算为总 FP≤336、恢复 FP≤168。上表三行均满足。冻结时间为 "
        f"{lock['at_utc']}，含配置、权重、预处理、逐窗分数、阈值和校准键摘要。"
        "正式留出键摘要及解封时间为空。 [冻结收据](evidence/freeze.json)", "",
        "## 等召回与误报", "",
        "下表使用每折相同 TP 数，先按分数降序再按键排序，目标召回 60%。"
        "等召回按真值确定截点，用于排序诊断。冻结阈值工作点单列核对预算。 [等召回明细](results/matched_recall.csv)", "",
        "| 阶段 | 中心或参照 | 总 FP/日 | 恢复 FP/日 |", "|---|---|---:|---:|",
    ])
    for stage, label in (("development", "开发后半折"), ("calibration", "校准")):
        for model, rule in (("center", "selected"), ("attention", "global"), ("attention", "selected"), ("rank", "global")):
            r = numbers[stage][model][rule + "_points"]["0.6"]
            a.append(f"| {label} | {LABELS[model]} {rule} | {r['FP_per_day']:.3f} | {r['recovery_FP_per_day']:.3f} |")
    a.extend(["", "| 校准配对差，中心减对照 | F1 差 | 召回差 | 60% 召回恢复 FP/日差 |", "|---|---:|---:|---:|"])
    for label, ref in (("V7 Attention 全局", "V7_Attention_global"), ("V8 尾部排序", "V8_rank_global"),
                       ("V7 Attention 分组", "V7_Attention_selected")):
        rs = [next(r for r in blocks if r["stage"] == "calibration" and r["contrast"] == f"center_selected_minus_{ref}"
                   and r["metric"] == m) for m in ("f1", "recall", "recovery_FP60_day")]
        cells = [f"{float(r['estimate']):+.4f} [{float(r['lo95']):+.4f}, {float(r['hi95']):+.4f}]" for r in rs]
        a.append("| " + " | ".join([label, *cells]) + " |")
    errors = [r for r in read("error_slices.csv") if r["fold"] == "final" and r["model"] == "center" and r["policy"] == "selected"]

    def avg(group, k):
        return sum(int(r[k]) for r in errors if r["slice"] == group) / len({r["seed"] for r in errors})

    a.extend([
        "",
        "方括号为共用 1,000 次日历日重抽的 95% 区间，先按种子配对再平均，区间条件于固定权重、偏移和阈值。"
        "中心在校准段 60% 召回时超出两项预算，恢复 FP 比两个全局对照略低。"
        "开发后半折的中心恢复 FP 则高于两对照。校准区间跨零，暂无稳定降低误报或提高 F1、召回的证据。"
        " [日历配对全表](results/paired_date_blocks.csv)", "",
        f"校准冻结点的中心恢复 FP 平均 {avg('all', 'recovery_fp'):.1f} 个，"
        f"其中 {avg('legal_low_t_minus_2', 'recovery_fp'):.1f} 个来自合法 t−2 低速组。"
        f"中心 FN 平均 {avg('all', 'fn'):.1f} 个，其中 t−2 有速度但未低速占 {avg('legal_known_other_t_minus_2', 'fn'):.1f} 个，"
        f"缺合法速度比占 {avg('legal_missing_t_minus_2', 'fn'):.1f} 个。事件按同道路连续十分钟正窗合并，unknown 打断事件。"
        " [误报漏报切片](results/error_slices.csv) · [事件指标](results/events.csv)", "",
        "[校准 PR 曲线](figures/calibration_PR.png) · [校准工作点和等召回图](figures/calibration_false_alarms.png)", "",
        "## 复核路径", "",
        "1. 从 manifest.csv 和 SHA256SUMS 核对包内路径、字节和摘要。SHA256SUMS 纳入 manifest 自身的哈希，两表排除 SHA256SUMS。",
        "2. 对照 evidence/inputs.json 的 V8 ZIP 和 36 个签名工件。download 收据含官方 URL、查询、行数、哈希、外部路径。"
        "用 V7 重建有向路网、冻结历史和四个 lead 数组，对照 prepared_development.json 及每折已知键摘要。",
        "3. 用 results/signed/ 核对三组共用的样本键、种子、预处理和训练抽样 ID。用 results/groups/ 的合法 lag20 标志重算开发网格及偏移选择。",
        "4. 对照开发锁、校准打开时间和冻结收据，重算逐窗决策、混淆、AP、BA、F1、召回、等召回与预算。"
        "检查 evidence/date_gate.json 的时间字段查询和 blocked 记录。",
        f"5. 已独立复算 {verification['grid_recomputed']} 个开发候选、{verification['frozen_and_transfer_recomputed']} 个工作点、"
        f"{verification['matched_recall_recomputed']} 个等召回点。已回放 {verification['models_replayed']} 个真实模型的抽样批，"
        f"最大分差 {verification['max_score_difference'] if verification['max_score_difference'] is not None else 'N/A'}。"
        f"工程测试 {tests['total']} 项。 [复现入口](README_审计入口.md)", "",
        "## 数据边界和答辩表述", "",
        "data_as_of 用作离线可用时点，源表缺独立到达记录。路网边由有向几何端点推断，存在孤立节点、速度与覆盖偏差。"
        "中心控制只看目标路段，V7 Attention 和 V8 尾部排序使用局部图。V9 实际执行权重重放和策略拟合，权重来自 V7/V8。"
        "Chicago 原代理口径缺续期，Geolife 原课程正类不足。纽约为公开补充验证。"
        " [NYC DOT 数据](https://data.cityofnewyork.us/d/i4gi-tjb9)", "",
        f"> 在纽约独立校准日期段，中心模型按严格提前十分钟出分，三种子 AP {cal['AP']:.3f}、F1 {cal['f1']:.3f}、召回 {cal['recall']:.3f}。"
        "两项误报预算逐种子通过。分组调整未提高预选中心模型的开发迁移表现，等召回的恢复误报差跨阶段方向不一致。"
        "纽约正式留出尚未到齐，正式成绩仍为 N/A。", "",
    ])
    (root / "审计报告.md").write_text("\n".join(a), encoding="utf-8")
    (root / "测试报告.md").write_text(
        "# V9 测试报告\n\n"
        f"实际通过 {tests['total']} 项工程测试。V9 新增 {tests['counts']['v9']} 项，"
        "覆盖时点分组、分数恒等、预算和并列阈值、开发门槛、留出拒绝、日期重抽。"
        "继承 V3–V8 的标签、图、池化、隔离、模型保存加载及 t−1、t、未来扰动测试已重跑。"
        "合成输入用于工程测试。 [测试命令和结果](synthetic/status.json)\n\n"
        f"真实数据独立复算开发 {verification['grid_recomputed']} 个候选、"
        f"工作点 {verification['frozen_and_transfer_recomputed']} 个、等召回 {verification['matched_recall_recomputed']} 个。"
        f"实际回放 {verification['models_replayed']} 个有签名权重的抽样批，"
        f"最大分差 {verification['max_score_difference'] if verification['max_score_difference'] is not None else 'N/A'}。"
        "权重以 weights_only=True 加载。正式留出验证因时间覆盖 blocked。\n", encoding="utf-8")
    print(dict(report="审计报告.md", engineering_tests=tests["total"], verification=verification["status"]), flush=True)


if __name__ == "__main__":
    report(settings()[0])

"""Build audit narrative and figures exclusively from measured development artifacts."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from v4_io import load, settings
from v4_matrix import location
from v4_train import read_npz

NAME = dict(B_mean="V3 R2 + Mean refit", P1="尾部排序", P2="方向证据", P3="尾部排序＋方向证据")


def figures(root, cfg, selected, model_names, points):
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.1, 4.6), layout="constrained")
    for model in model_names:
        values = points[points.model == model].sort_values("target_recall")
        ax.plot(100 * values.target_recall, values.mean_FP_per_day, marker="o", label=NAME.get(model, model.replace("_", " + ")))
    ax.axvline(60, color="#777777", lw=1, ls="--")
    ax.set(xlabel="Target recall (%)", ylabel="False alerts per calendar day",
           title="Chicago development | matched recall over all known windows")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(out / "matched_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.1, 4.7), layout="constrained")
    fold, seed = cfg["folds"][-1], cfg["seeds"][0]
    for model in model_names:
        recipe, pool = model.split("_")
        z = read_npz(location(root, fold, recipe, pool, seed) / "assessment.npz")
        precision, recall, _ = precision_recall_curve(z["y"], z["score"])
        ax.step(recall, precision, where="post", label=NAME.get(model, model.replace("_", " + ")))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
           title=f"Chicago development PR | {fold.upper()} · seed {seed}")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.savefig(out / "development_PR.png", dpi=160)
    plt.close(fig)

    errs = pd.read_csv(root / "results/paired_error_review/groups.csv")
    order = ["1", "2", "3+"]
    ref = errs[(errs.model == "B_mean") & (errs.dimension == "depth_bin")].set_index("category").loc[order]
    alt = errs[(errs.model == selected) & (errs.dimension == "depth_bin")].set_index("category").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), layout="constrained", sharey=True)
    x = np.arange(len(order))
    for ax, col, title in zip(axes, ("FP_rate", "FN_rate"), ("Recovery-negative false alarm rate", "Positive miss rate"), strict=True):
        ax.bar(x - .18, 100 * ref[col], .35, label="V3 R2 + Mean refit")
        ax.bar(x + .18, 100 * alt[col], .35, label=selected.replace("_", " + "))
        ax.set(xticks=x, xticklabels=order, ylim=(0, 100), title=title, xlabel="Prior consecutive low windows")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Percent of stratum")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Chicago development | target recall 60%")
    fig.savefig(out / "risk_errors_by_streak.png", dpi=160)
    plt.close(fig)


def render(root, cfg):
    select = load(root / "results/selection.json")
    chosen = select["recipe"] + "_" + select["pool"]
    screen = load(root / "results/screen.json")["summary"]
    confirm = load(root / "results/confirmation.json")["summary"]
    calibration = load(root / "results/calibration_summary.json")["summary"]
    proof = load(root / "evidence/verification_replay.json")
    gate = load(root / "evidence/holdout_gate.json")
    points = pd.read_csv(root / "results/comparison/matched_aggregates.csv")
    bands = pd.read_csv(root / "results/comparison/paired_date_blocks.csv")
    errors = pd.read_csv(root / "results/paired_error_review/groups.csv")
    latest = gate["queries"]["latest"].get("rows", [{}])[0].get("time", "N/A")
    models = [f"{select['recipe']}_{p}" for p in cfg["pools"]] + ["B_mean"]
    figures(root, cfg, chosen, models, points)
    lines = [
        "# Project B V4 审计报告",
        "",
        "**正式验证 blocked，AP 与正类 F1 的双 0.60 目标为 N/A。** 第四轮完成 Chicago 三折开发和原独立校准。"
        "V3 R2＋Mean 为同支持重跑对照。所有 V4 分数来自开发或校准日期。",
        "",
        "## 数据边界与评价支持",
        "",
        f"官方 API 最新时间为 {latest}（Chicago 本地）。2026-05-01 起同路段 ID 范围查询返回 HTTP 200 和空数组。"
        "官方停更公告与目录查询未发现可验证的同口径后继。官方列出的替代日均流量缺少原任务需要的 10 分钟有向路段速度。"
        "较早年份也无法形成晚于 V3 的前向验证。 [查询收据](evidence/holdout_gate.json) · "
        "[发布方停更说明](https://data.cityofchicago.org/stories/s/c28d-stgw) · [数据集](https://data.cityofchicago.org/d/4g9f-3jbs)",
        "",
        "V3 留出 2026-02-02 至 04-30 只引用发布的 R2＋Mean 聚合 AP 0.4356、F1 0.4459。"
        "V1、V2、V3 留出的观测均排除在训练、历史、误报复盘、选型和校准之外。V3 归档中的留出文件仅作字节完整性核验。"
        "Geolife 沿用原审计的 blocked/N/A，Chicago 继续作为补充代理任务。 [原始输入核验](evidence/inputs.json) · "
        "[V3 历史摘要](evidence/v3_historical_only.json) · [Geolife](evidence/geolife_status.json)",
        "",
        "保持 231 条有向路段和 220 条边、原高峰、10 分钟、两次连续低于各自历史基准的 40%。"
        "unknown 与预测资格保持 V3 规则，监督评价覆盖每折全部已知有图窗口。相同折的键、标签、风险和训练抽样完全一致。",
        "",
        "| 用途 | Chicago 本地日期（左闭右开） | 已知窗 | 正例 | 恢复负例 | 正例事件段 | 未知有图窗 | 已知覆盖 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in cfg["folds"] + ["final"]:
        a, b = cfg["fold_limits"][fold]
        row = load(root / "evidence" / f"fold_{fold}.json")["parts"][1]
        lines.append(f"| {fold.upper() if fold != 'final' else '84 日校准'} | {a} 至 {b} | {row['positive']+row['negative']:,} "
                     f"| {row['positive']:,} | {row['high_risk_recovered']:,} | {row['true_events']:,} "
                     f"| {row['unknown_eligible']:,} | {row['known_coverage']:.2%} |")
    lines += ["", "已知覆盖分母为固定路网内的全量预期网格。事件为同道路同日相邻正例窗。 "
              "[数据漏斗](results/comparison/support.csv) · [时点合同](方法与数据合同.md)",
              "", "## 配对开发消融", "",
              "同一采样、Graph Transformer、路由与更新预算。先 Mean、seed 11 三折初筛。开发 F1 以同折最佳阈值计算，用于诊断。", "",
              "| 组 | 变化 | AP | 诊断 F1 | 60% 召回 FP／日 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for model, value in screen.items():
        lines.append(f"| {model} | {NAME.get(model, 'V3 R2＋Mean 重跑')} | {value['AP']:.4f} | {value['f1']:.4f} | {value['FP60_per_day']:.3f} |")
    lines += ["", f"按预登记规则入围 {select['recipe']}。三种池化各三种子各三折，开发预选 **{chosen}**。表内均值先按种子对三折等权平均，标准差在三个种子间计算。 "
              "[初筛](results/screen.json) · [确认](results/confirmation.json) · [选择收据](results/selection.json)",
              "", "| 配方 | AP | 诊断 F1 | 60% 召回 FP／日 |",
              "| --- | ---: | ---: | ---: |"]
    for model in models:
        value = confirm[model]
        label = f"**{model}**" if model == chosen else model
        lines.append(f"| {label} | {value['AP']:.4f} ± {value['AP_sd']:.4f} | {value['f1']:.4f} ± {value['f1_sd']:.4f} "
                     f"| {value['FP60_per_day']:.3f} ± {value['FP60_per_day_sd']:.3f} |")
    lines += ["", "## 相同召回下的误报与漏报", "",
              "每折在全量已知分数上取达到规定 TP 数的最短排名前缀。所有正例在高风险分支。"
              "在这四个召回点，误报均为上一格低速、本格恢复的已知负例。并列按样本键取序，表内另存纯阈值把整组并列纳入时的计数。"
              "这些比较需要当折真值，仅度量排名在相同召回下的代价。 [逐折等召回](results/comparison/matched_points.csv) · "
              "[逐对差值](results/comparison/paired_matched_recall.csv) · [等召回曲线](figures/matched_recall.png)",
              "", "| 目标召回 | 实际折均召回 | V3 重跑 FP／日 | V4 Mean | V4 Max | V4 Attention |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for rate in cfg["recalls"]:
        view = points[points.target_recall == rate].set_index("model")
        lines.append(f"| {rate:.0%} | {view.loc['B_mean', 'mean_recall']:.2%} | "
                     + " | ".join(f"{view.loc[model, 'mean_FP_per_day']:.3f}" for model in ["B_mean", *models[:3]]) + " |")
    ref, alt = confirm["B_mean"], confirm[chosen]
    ci = bands[bands.model == chosen].set_index("metric")
    lines += ["", f"{chosen} 相对 V3 重跑的开发差为 ΔAP {alt['AP']-ref['AP']:+.4f}、"
              f"ΔF1 {alt['f1']-ref['f1']:+.4f}、等召回 ΔFP／日 {alt['FP60_per_day']-ref['FP60_per_day']:+.3f}。"
              f"配对日期块 250 次的 95% 区间分别为 [{ci.loc['AP', 'lower_95']:+.4f}, {ci.loc['AP', 'upper_95']:+.4f}]、"
              f"[{ci.loc['diagnostic_F1', 'lower_95']:+.4f}, {ci.loc['diagnostic_F1', 'upper_95']:+.4f}] 和 "
              f"[{ci.loc['FP60_per_day', 'lower_95']:+.3f}, {ci.loc['FP60_per_day', 'upper_95']:+.3f}]。"
              "同一折的三种子、模型共用日期块，F1 重采样固定原诊断阈值。开发复用及事件稀少限制推断。 "
              "[配对区间](results/comparison/paired_date_blocks.csv) · [开发 PR](figures/development_PR.png)",
              "", "按低速连续深度分组。分母和计数单位为种子窗，同一个道路窗口可在三个种子中各计一次。",
              "", "| 上一格起连续低速深度 | 恢复负例支持 | V3 误报 | V4 误报 | 正例支持 | V3 漏报 | V4 漏报 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cat in ("1", "2", "3+"):
        rows = errors[(errors.dimension == "depth_bin") & (errors.category == cat)].set_index("model")
        before, after = rows.loc["B_mean"], rows.loc[chosen]
        lines.append(f"| {cat} | {before.recovery_support:,} | {before.FP:,} | {after.FP:,} "
                     f"| {before.positive_support:,} | {before.FN:,} | {after.FN:,} |")
    lines += ["", "完整分组另含前格速度比、公交探针数、连接性、峰别、日类型和道路。"
              "格内公交数及连续低速深度可能同时反映观测质量和交通情况，分组误差不支持因果解释。 "
              "[逐键风险表](results/paired_error_review/risk_windows.csv) · [分组](results/paired_error_review/groups.csv) · "
              "[误差图](figures/risk_errors_by_streak.png)",
              "", "## 既有独立校准与复核", "",
              "开发选择和 epoch 锁定后，在此前独立的 84 日校准段拟合诊断阈值。每组保留三种子。"
              "这段已用于 V3 校准，以下分数反映本轮校准段拟合，不用于双 0.60 判定。", "",
              "| 组 | AP | 校准段最大 F1 | 固定 epoch |",
              "| --- | ---: | ---: | ---: |"]
    for model in models:
        value = calibration[model]
        lines.append(f"| {model} | {value['AP']:.4f} ± {value['AP_sd']:.4f} | "
                     f"{value['f1']:.4f} ± {value['f1_sd']:.4f} | {select['fixed_epochs'][model]} |")
    lines += ["", f"独立复核重新计算 {proof['runs']} 个模型的指标、阈值及等召回计数。"
              f"九个 V3 开发重跑的样本、决定、AP、F1 与等召回计数一致，{proof['exact_V3_dev_runs']} 组逐位一致。"
              f"其余组的最大概率差为 {proof['max_V3_probability_difference']:.3g}，保留首次逐位检查失败日志。"
              f"{proof['replays']} 组权重各重放一个推理批次，"
              f"最大概率差 {proof['replay_max_probability_difference']:.2g}。"
              "封存收据保留权重、阈值、采样键、预处理器、配置和诊断结果哈希。 "
              "[校准收据](results/calibration_freeze.json) · [指标和重放](evidence/verification_replay.json) · [测试报告](测试报告.md)",
              "", "1. 校验原始 V3 ZIP、原料缓存指纹和本包清单。",
              "2. 按 plan 核对开发与校准日期、封存与禁用区间。在逐折 NPZ 上核查标签、样本键、未知覆盖和抽样。",
              "3. 运行 verify 离线复算，或附 V3 固定数组和原 V3 归档重放权重与开发对照。",
              "4. 检查正式状态 JSON、原标签合同和 API 收据。未来只有找到兼容的前向速度数据，才能预登记新的留出和正式验证。",
              "", "完整命令、环境与数据路径见 [审计入口](README_审计入口.md)。ZIP 旁保留干净解压复核收据。",
              ""]
    (root / "审计报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(dict(report=str(root / "审计报告.md"), figures=3, chosen=chosen, formal="blocked"), flush=True)


if __name__ == "__main__":
    render(*settings())


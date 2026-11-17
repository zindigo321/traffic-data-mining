"""Render the V6 audit, keeping both paired questions visible."""
from datetime import date,timedelta
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from v6_io import V5,load,settings
sys.path.insert(0,str(V5))
from v5_prepare import read_npz

LABELS={"X1_mean":"X1＋Mean（V5 对照）","X1_max":"X1＋Max","X1_attention":"X1＋Attention（预选）",
        "B_mean":"V4 P1＋Mean 重跑"}
FIGLABELS={"X1_mean":"V5 X1 + Mean refit","X1_max":"X1 + Max","X1_attention":"X1 + Attention (primary)",
           "B_mean":"V4 P1 + Mean refit"}
CONTROLS={"V5_X1_Mean":"V5 X1＋Mean","V4_P1_Mean":"V4 P1＋Mean"}
COLORS=["#525c66","#af7128","#196d91","#765193"]


def images(root,counts,aggregates,paired):
    dest=root/"figures"
    dest.mkdir(exist_ok=True)
    order=list(LABELS)
    fig,ax=plt.subplots(figsize=(8.9,4.8),layout="constrained")
    for name,color in zip(order,COLORS,strict=True):
        r,p=name.split("_")
        view=counts[(counts.recipe==r)&(counts.pool==p)].groupby("target_recall").FP_per_day.mean()
        ax.plot(100*view.index,view.values,marker="o",color=color,label=FIGLABELS[name])
    ax.axvline(60,lw=1,ls="--",color="#666666")
    ax.set(xlabel="Target recall (%)",ylabel="Recovery false alerts per calendar day",
           title="NYC V6 independent holdout | matched recall")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.25)
    fig.savefig(dest/"matched_recall.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.9,4.8),layout="constrained")
    for name,color in zip(order,COLORS,strict=True):
        raw=read_npz(root/"results/holdout"/f"{name}_s11"/"all_eligible.npz")
        include=raw["y"]>=0
        precision,recall,_=precision_recall_curve(raw["y"][include],raw["score"][include])
        ax.step(recall,precision,where="post",label=FIGLABELS[name],color=color)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",title="NYC V6 independent holdout | seed 11")
    ax.grid(alpha=.25)
    ax.legend(frameon=False,loc="lower left")
    fig.savefig(dest/"holdout_PR.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout="constrained")
    names=list(paired)
    change=np.array([paired[k]["mean_FP_per_day_delta"] for k in names])
    edges=np.array([paired[k]["FP_per_day_95"] for k in names])
    ax.errorbar(change,np.arange(len(names)),xerr=np.abs(edges.T-change),fmt="o",capsize=5,color="#196d91")
    ax.axvline(0,color="#777777",lw=1,ls="--")
    ax.set(yticks=np.arange(len(names)),yticklabels=["vs V5 X1 + Mean","vs V4 P1 + Mean"],
           xlabel="Attention minus control, recovery false alerts/day",title="NYC V6 | matched 60% recall, paired 95% date intervals")
    ax.grid(axis="x",alpha=.25)
    fig.savefig(dest/"paired_date_intervals.png",dpi=160)
    plt.close(fig)


def render(root,cfg):
    state=load(root/"evidence/status.json")
    proof=load(root/"evidence/verification_replay.json")
    gate=load(root/"evidence/date_gate.json")
    integrity=load(root/"evidence/v5_integrity.json")
    pre=load(root/"evidence/registration.json")
    if state["status"]!="complete":
        (root/"审计报告.md").write_text(
            "# Project B V6 审计报告\n\n**纽约正式验证 blocked／N/A。Chicago 原口径 blocked。**\n\n"
            +state["reason"]+"\n\n12 组重跑及校准、日期证据、测试和日志随包。\n",encoding="utf-8")
        return
    out=root/"results/holdout"
    lock=load(root/"evidence/freeze.json")
    unlock=load(root/"evidence/unlock.json")
    prior=load(root/"evidence/v5_historical_only.json")
    graph=load(root/"evidence/graph.json")
    data=load(root/"evidence/fold_holdout.json")
    support=data["parts"][0]
    receipt=load(root/"evidence/holdout_support.json")
    cal=load(root/"evidence/fold_final.json")["parts"]
    aggregates=load(out/"aggregates.json")
    comparisons=load(out/"comparisons.json")
    points=pd.read_csv(out/"matched_recall.csv",float_precision="round_trip")
    pairs=pd.read_csv(out/"paired.csv",float_precision="round_trip")
    current=pairs[pairs.target_recall==cfg["main_recall"]]
    images(root,points,aggregates,comparisons)
    a,b=map(date.fromisoformat,cfg["holdout"])
    inclusive=b-timedelta(days=1)
    actual=current.iloc[0].achieved_recall
    success="达到" if state["goal_reached"] else "未达到"
    both="达到" if state["stable_vs_both"] else "未达到"
    pin="、".join(f"{k}={v}" for k,v in cfg["epochs"].items())
    lines=[
        "# Project B V6 审计报告",
        "",
        f"**预选 X1＋Attention 在纽约新留出{success}双 0.60，{both}相对两个对照均稳定降低恢复误报的条件。** "
        f"三种子平均 AP {state['AP']:.4f}，正类 F1 {state['F1']:.4f}。Chicago 原口径继续 **blocked／N/A**，Geolife 分列 blocked。",
        "",
        "## 日期、样本与证据",
        "",
        f"V5 ZIP 的 {integrity['manifest_rows']} 条 manifest 和 {integrity['hashes']} 项 SHA 通过。"
        "旧纽约留出 2026-08-31 至 2026-10-11 全段从 V6 训练、统计、校准及评分隔离，只引用发布报告的聚合结果。"
        f"新日期由无速度的官方计数确定，登记于 {pre['at_utc']}。 "
        "[V5 核验](evidence/v5_integrity.json) · [历史参考](evidence/v5_historical_only.json) · "
        "[日期闸门](evidence/date_gate.json) · [预登记](registered_holdout.md)",
        "",
        f"纽约新留出为 **{a} 至 {inclusive}**，共 {gate['calendar_days']} 天，{gate['substantial_days']} 天达到发布门槛。"
        "10 月 12 日无观测，10 月 13 日早高峰缺报。原日历和道路分母保留。"
        f"新分片原始 {data['funnel']['rows']:,} 行，其中原名单内 {data['funnel']['rows']-data['funnel']['outside_pinned_roster']:,} 行，"
        f"对照时间计数 {receipt['raw_roster_count_in_metadata_gate']:,} 行。"
        f"固定 {graph['road_count']} 条有向路段、{graph['edges']} 条接续边，正式已知支持来自 {support['known_roads']} 条。 "
        "[NYC DOT 官方源](https://data.cityofnewyork.us/d/i4gi-tjb9) · [分片和哈希](evidence/download_holdout_sealed.json) · [图](evidence/graph.json)",
        "",
        "| 范围 | 合格窗 | 已知窗 | 正例 | 恢复负例 | 代理事件段 | 已知覆盖 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label,c in (("最终训练",cal[0]),("独立校准",cal[1]),("全新纽约留出",support)):
        lines.append(f"| {label} | {c['eligible']:,} | {c['positive']+c['negative']:,} | {c['positive']:,} | "
                     f"{c['high_risk_recovered']:,} | {c['true_events']:,} | {c['known_coverage']:.2%} |")
    lines+=["",
        f"正式有 {support['expected']:,} 个预期网格，{support['unknown_eligible']:,} 个有图 unknown 出分后排除监督评价。"
        f"正式已知正例率 {support['positive']/state['n']:.2%}。纽约十分钟估速由两个半格组成，"
        "与 Chicago 公交探针不同。几何接续、路段缺报、提供方估速延续及季节限制跨城解释。 "
        "[留出漏斗](evidence/fold_holdout.json) · [原口径状态](evidence/chicago_status.json) · [Chicago 官方说明](https://data.cityofchicago.org/stories/s/c28d-stgw)",
        "",
        "## 训练、校准与一次解封",
        "",
        "重新从头拟合四组三种子。X1＋Mean 兼作池化组和 V5 对照。"
        "三个旧开发折、最终训练、固定拓扑和历史都用 V5 的许可范围重建并校验，旧留出数据未接入。"
        f"检查点来自 V5 旧解封前的开发配置（{pin}），每次跑满 12 轮。"
        f"统一在独立 28 日校准按最大正类 F1 定阈值。{lock['frozen_utc']} 冻结，{unlock['unlocked_utc']} 解封。 "
        "[开发数组](evidence/prepared_development.json) · [校准](results/calibration.csv) · [冻结](evidence/freeze.json) · [解封](evidence/unlock.json)",
        "",
        "首次适配器误把固定检查点表传入总轮次，拟合起步即报错。适配层在解封前修正，"
        "保持登记日期、12 轮预算、固定检查点和全部对照。错误日志及部分文件保留。 "
        "[修复收据](evidence/trainer_adapter_fix.json)",
        "",
        "## 全量新留出实测",
        "",
        "| 配方 | AP 均值 ± SD | 正类 F1 均值 ± SD | ROC-AUC | Precision | Recall | 冻结阈值误报／日 | 等召回恢复误报／日 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key,name in LABELS.items():
        s=aggregates[key]
        lines.append(f"| {name} | {s['AP']:.4f} ± {s['AP_sd']:.4f} | {s['f1']:.4f} ± {s['f1_sd']:.4f} | "
                     f"{s['ROC_AUC']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | "
                     f"{s['fp_per_day']:.3f} | {s['FP60_per_day']:.3f} |")
    lines+=["",
        "AP 用 average_precision_score，F1 用解封前的校准阈值。种子、Accuracy、TP/FP/FN/TN、事件结果及资源见实测表。 "
        "[逐种子指标](results/holdout/metrics.csv) · [PR](figures/holdout_PR.png)",
        "",
        "## 两项配对等召回结论",
        "",
        f"主要召回点实际为 **{actual:.3%}**，每个种子每个模型同样检出 **{int(current.iloc[0].TP)}** 个正例。"
        "此点误报全部为上一格低速、当前恢复的已知负例。日期分块 1,000 次共用日期抽样，95% 差区间按 Attention 减对照。"
        "等召回点用真值确定排序截面，冻结阈值单列。V5 对照固定 X1 特征，主要比较池化；V4 对照比较整体配方。 ",
        "",
        "| 对照 | 对照 FP／日 | Attention FP／日 | 日均差 95% 区间 | 下降种子 | 达到稳定性 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key,s in comparisons.items():
        lo,hi=s["FP_per_day_95"]
        stable="是" if s["stable"] else "否"
        lines.append(f"| {CONTROLS[key]} | {s['control_FP_per_day']:.3f} | {s['primary_FP_per_day']:.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {s['seeds_with_fewer_FP']}/3 | {stable} |")
    lines+=["",
        "固定支持的恢复误报和正例漏报按连续低速深度、峰别、路段和日期分列。日期区间反映本月条件，稀缺月份和道路泛化仍待验证。 "
        "[逐种子配对](results/holdout/paired.csv) · [日期块](results/holdout/date_blocks.csv) · "
        "[等召回曲线](figures/matched_recall.png) · [误差分组](results/holdout/errors_grouped.csv)",
        "",
        "## 复核",
        "",
        f"独立复算 {proof['calibrations']} 组校准和 {proof['heldout_runs']} 组正式结果，"
        f"抽样重放 {proof['calibration_replayed']+proof['heldout_replayed']} 组预测，最大概率差 {proof['max_probability_difference']:.3g}。"
        "28 项工程测试仅验证实现。 "
        "[复算](evidence/verification_replay.json) · [测试报告](测试报告.md)",
        "",
        "1. 验签 V5 原始 ZIP、本包 manifest 和外部原料。检查原 V5 留出未进入重跑。",
        "2. 用旧开发和新封存分片重建图、历史及准备数组。核对登记、日期及未知分母。",
        "3. 核查模型和阈值冻结、唯一解封时间与同支持的逐窗分数。",
        "4. 运行 verify 独立复算阈值、全量指标、等召回配对与区间。使用经过 SHA 校验的数组安全重放。",
        "",
        "[一键复现及完整导航](README_审计入口.md)。干净解压收据与 ZIP 同级。",
        ""]
    (root/"审计报告.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"审计报告.md"),primary=state["primary"],goal=state["goal_reached"],
          stable_vs_both=state["stable_vs_both"]),flush=True)


if __name__=="__main__":
    render(*settings())

"""Generate the audit from real external measurements and frozen receipts."""
from datetime import date,timedelta

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from v5_io import load,settings
from v5_prepare import read_npz


def readable_dates(interval):
    a,b=map(date.fromisoformat,interval)
    return f"{a} 至 {b-timedelta(days=1)}"


def plots(root,metrics,points,choice,groups):
    out=root/"figures"
    out.mkdir(exist_ok=True)
    colors=["#166e89","#9c5c17","#674f9b","#414a54"]
    labels=["V5 Mean","V5 Max","V5 Attention","V4 P1 + Mean refit"]
    fig,ax=plt.subplots(figsize=(8.7,4.7),layout="constrained")
    for name,label,col in zip(groups,labels,colors,strict=True):
        recipe,pool=name.split("_")
        view=points[(points.recipe==recipe)&(points.pool==pool)].groupby("target_recall").FP_per_day.mean()
        ax.plot(100*view.index,view.to_numpy(),label=label,color=col,marker="o")
    ax.axvline(60,color="#777777",lw=1,ls="--")
    ax.set(xlabel="Target recall (%)",ylabel="Recovery false alerts per calendar day",
           title="NYC independent holdout | identical known support")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.25)
    fig.savefig(out/"matched_recall.png",dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.7,4.7),layout="constrained")
    for name,label,col in zip(groups,labels,colors,strict=True):
        recipe,pool=name.split("_")
        row=metrics[(metrics.recipe==recipe)&(metrics.pool==pool)&(metrics.seed==11)].iloc[0]
        z=read_npz(root/"results/holdout"/row.job/"all_eligible.npz")
        known=z["y"]>=0
        precision,recall,_=precision_recall_curve(z["y"][known],z["score"][known])
        ax.step(recall,precision,where="post",color=col,label=f"{label} (AP {row.AP:.3f})")
    ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02),
           title="NYC independent holdout | PR curves, seed 11")
    ax.legend(frameon=False,loc="lower left")
    ax.grid(alpha=.25)
    fig.savefig(out/"holdout_PR.png",dpi=160)
    plt.close(fig)

    data=load(root/"results/holdout/aggregates.json")
    fig,ax=plt.subplots(figsize=(9.4,4.6),layout="constrained")
    x=np.arange(len(groups))
    for metric,delta,color,text in (("AP",-.18,"#166e89","AP"),("f1",.18,"#a5682a","Positive F1")):
        ax.bar(x+delta,[data[g][metric] for g in groups],.34,yerr=[data[g][metric+"_sd"] for g in groups],
               capsize=4,color=color,label=text)
    ax.axhline(.6,lw=1,ls="--",color="#333333",label="Registered 0.60")
    ax.set(ylim=(0,1.05),ylabel="Three-seed mean ± SD",xticks=x,xticklabels=labels,
           title="NYC independent holdout | frozen selection and threshold")
    ax.legend(frameon=False,ncols=3,loc="lower center")
    ax.grid(axis="y",alpha=.22)
    fig.savefig(out/"holdout_comparison.png",dpi=160)
    plt.close(fig)


def render(root,cfg):
    out=root/"results/holdout"
    state=load(root/"evidence/status.json")
    graph=load(root/"evidence/graph.json")
    lock=load(root/"evidence/freeze.json")
    opened=load(root/"evidence/unlock.json")
    dev=load(root/"evidence/prepared_development.json")
    verification=load(root/"evidence/verification_replay.json")
    recipe,pool=lock["choice"]["recipe"],lock["choice"]["pool"]
    selected=f"{recipe}_{pool}"
    groups=[f"{recipe}_{p}" for p in cfg["pools"]]+["B_mean"]
    labels=[f"{recipe}＋Mean",f"{recipe}＋Max",f"{recipe}＋Attention","V4 P1＋Mean 同支持重跑"]
    names=dict(zip(groups,labels,strict=True))
    aggregate=load(out/"aggregates.json")
    screen=load(root/"results/screen.json")["summary"]
    metrics=pd.read_csv(out/"metrics.csv",float_precision="round_trip")
    points=pd.read_csv(out/"matched_recall.csv",float_precision="round_trip")
    pairs=pd.read_csv(out/"paired.csv",float_precision="round_trip")
    bands=pd.read_csv(out/"date_blocks.csv",float_precision="round_trip")
    plots(root,metrics,points,selected,groups)
    ref=aggregate["B_mean"]
    best=aggregate[selected]
    p_main=pairs[(pairs.recipe==recipe)&(pairs.pool==pool)&(pairs.target_recall==cfg["main_recall"])]
    held=load(root/"evidence/fold_holdout.json")["parts"][0]
    day=readable_dates(cfg["holdout"])
    raw_ratio=dev["source_funnel"]["implied_miles_over_polyline_miles_p10_median_p90"]
    stable="达到" if state["stable_recovery_FP_reduction"] else "未达到"
    goal="达到" if state["goal_reached"] else "未达到"
    lines=[
        "# Project B V5 审计报告",
        "",
        f"**纽约外部留出{goal}双 0.60 目标，{stable}稳定降低恢复误报的预登记条件。** "
        f"开发预选 {names[selected]} 的 AP {state['AP']:.4f}、正类 F1 {state['F1']:.4f}。Chicago 原口径正式状态继续 **blocked / N/A**。",
        "",
        "## 1. 数据边界、协议与证据",
        "",
        "Chicago 4g9f-3jbs 的时间查询仍以 2026-04-30 为末日，5 月 1 日起同道路范围返回空。官方停更后的日流量缺少同口径路段估速。"
        "V4 Chicago 开发结果与以往留出仅作历史参考。Geolife 继续分列 blocked。 "
        "[查询收据](evidence/feasibility.json) · [Chicago 状态](evidence/chicago_status.json) · "
        "[官方停更说明](https://data.cityofchicago.org/stories/s/c28d-stgw) · [Geolife](evidence/geolife_prior.json)",
        "",
        f"纽约采用 NYC DOT i4gi-tjb9。按无速度元数据选择日期，注册于 {load(root/'evidence/registration.json')['at_utc']}。"
        "官方数据为主要干道的点到点估速。纽约使用其本地日历、两半格汇总、自己构造的有向图与全新拟合权重。"
        "双城市结果不合并。字段字典将 status 列为遗留字段，已在解析速度前锁定勘误。"
        f"开发快照中，估速与行程时间的隐含英里相对几何长度的中位比为 {raw_ratio[1]:.3f}。 "
        "[预登记](protocol_nyc.md) · [字段勘误](amendment_nyc.md) · "
        "[NYC DOT 官方数据](https://data.cityofnewyork.us/d/i4gi-tjb9) · "
        "[NYC DOT 字段说明](https://www1.nyc.gov/html/dot/downloads/pdf/metadata-trafficspeeds.pdf)",
        "",
        f"锚点 125 个 ID，{graph['road_count']} 个保留并固定，{graph['edges']} 条有向边，{graph['isolated']} 个孤立节点，"
        f"最大实际子图 {graph['max_subgraph_nodes']} 节点。{graph['metadata_link_conflicts']} 个 ID 有多个元数据记录版本，按锚点模式选择。"
        f"正式已知窗实际来自 {held['known_roads']} 条路段，所有固定路段的预期网格仍计入覆盖。"
        "近端连接可能缺少立交和物理转向信息。 "
        "[节点和边](evidence/graph.json) · [时点与标签](方法与数据合同.md)",
        "",
        "| 阶段 | 纽约本地日期 | 全部已知窗 | 正例 | 恢复负例 | 代理事件段 | 已知覆盖 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    periods={**cfg["folds"],"final":cfg["calibration"],"holdout":cfg["holdout"]}
    for key,interval in periods.items():
        cov=load(root/"evidence"/f"fold_{key}.json")["parts"][0 if key=="holdout" else 1]
        label={"final":"校准","holdout":"正式留出"}.get(key,key.upper())
        lines.append(f"| {label} | {readable_dates(interval)} | {cov['positive']+cov['negative']:,} | "
                     f"{cov['positive']:,} | {cov['high_risk_recovered']:,} | {cov['true_events']:,} | {cov['known_coverage']:.2%} |")
    lines += ["",
        f"正式留出 {held['total_days']} 天，固定网格 {held['expected']:,} 个，合格 {held['eligible']:,} 个，"
        f"其中 {held['unknown_eligible']:,} 个 unknown 保留出分并排除监督评价。正例率 {held['positive']/state['n']:.2%}，"
        f"上一格低速的已知风险组正例率 {held['positive']/(held['positive']+held['high_risk_recovered']):.2%}。"
        "风险组基准比例和 Chicago 明显不同。两个半格可能反映同一提供方的估计延续，代理标签不证明真实连续五分钟拥堵。 "
        "[漏斗和标签](evidence/fold_holdout.json) · [完整支持摘要](evidence/holdout_support.json)",
        "",
        "## 2. 开发选择和正式冻结",
        "",
        "| 三折单种子初筛 | AP | 同折诊断 F1 | 60% 召回误报／日 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name,value in screen.items():
        lines.append(f"| {name} | {value['AP']:.4f} | {value['f1']:.4f} | {value['FP60_per_day']:.3f} |")
    lines += ["",
        f"初筛选择 {recipe}，三池化确认后按开发 AP 预选 {names[selected]}。"
        "B 使用 V4 P1 配方，在纽约同图、同样本、同步数和同阈值规则重跑。"
        "选择完成后，12 组模型在同一 28 日独立校准用最大 F1 定阈值。"
        f"冻结 {lock['frozen_utc']}，解封 {opened['unlocked_utc']}。"
        "日历、图、代码、预处理、权重与阈值哈希均在解封前记录。 "
        "[确认](results/confirmation.csv) · [冻结](evidence/freeze.json) · [解封](evidence/unlock.json)",
        "",
        "## 3. 纽约全量留出实测",
        "",
        "| 配方 | AP 均值 ± SD | 正类 F1 均值 ± SD | ROC-AUC | Precision | Recall | FP／日（冻结阈值） | 事件 F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in groups:
        a=aggregate[g]
        name=f"**{names[g]}（预选）**" if g==selected else names[g]
        lines.append(f"| {name} | {a['AP']:.4f} ± {a['AP_sd']:.4f} | {a['f1']:.4f} ± {a['f1_sd']:.4f} | "
                     f"{a['ROC_AUC']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['fp_per_day']:.3f} | {a['event_f1']:.4f} |")
    lines += ["",
        f"预选组对 V4 同支持重跑 ΔAP {best['AP']-ref['AP']:+.4f}、ΔF1 {best['f1']-ref['f1']:+.4f}。"
        "V4 重跑也超过双 0.60。冻结阈值的混淆计数、Accuracy、每种子 TP/FP/FN/TN 与事件统计见指标表。"
        "正式判定以预选池化为准，三池化都保留。 "
        "[逐配置](results/holdout/metrics.csv) · [比较图](figures/holdout_comparison.png) · [PR](figures/holdout_PR.png)",
        "",
        "## 4. 等召回恢复误报",
        "",
        f"主要比较在实际召回 {p_main.iloc[0].actual_recall:.3%}，每模型检出 {int(p_main.iloc[0].TP):,} 个正例。"
        "所有在此点出现的误报均为前一格低速、当前恢复。日期块在所有种子与配对模型间共享。",
        "",
        "| 种子 | V4 恢复 FP | V5 预选恢复 FP | V5−V4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in p_main.itertuples():
        lines.append(f"| {row.seed} | {row.V4_recovery_FP} | {row.V5_recovery_FP} | {row.delta_recovery_FP:+d} |")
    lo,hi=state["delta_FP_95"]
    decrease=(ref["FP60_per_day"]-best["FP60_per_day"])/ref["FP60_per_day"]
    lines += ["",
        f"三种子平均恢复 FP／日由 {ref['FP60_per_day']:.3f} 降至 {best['FP60_per_day']:.3f}（{decrease:.1%}）。"
        f"配对日期块 1000 次的 95% 差区间为 [{lo:+.3f}, {hi:+.3f}] 个／日，"
        f"{state['seeds_with_fewer_FP']} 个种子下降。预登记要求至少两个种子下降且区间上界低于零，因此稳定改善{stable}标准。"
        "召回点使用真值确定排序截面，实际部署阈值仍取独立校准。 "
        "[完整等召回](results/holdout/matched_recall.csv) · [日期块区间](results/holdout/date_blocks.csv) · "
        "[误报漏报分组](results/holdout/errors_grouped.csv) · [误报曲线](figures/matched_recall.png)",
        "",
        "## 5. 复核与局限",
        "",
        f"独立验证复算 {verification['development_and_calibration_runs']} 组开发与校准、{verification['heldout_runs']} 组留出，"
        f"抽样安全重放 {verification['dev_replayed']+verification['heldout_replayed']} 组，"
        f"最大概率差 {verification['max_probability_difference']:.2g}。"
        "工程用例合计 23 项，合成只检验实现。运行命令、退出码、用时、硬件和种子在 logs。"
        "道路和季节固定、路网几何接续、路段缺报、不同探针和标签富集均限制跨城市解释。 "
        "[独立复算](evidence/verification_replay.json) · [测试报告](测试报告.md)",
        "",
        "1. 校验本包 manifest、SHA256SUMS、原 V4 ZIP 与外部纽约分片哈希。",
        "2. 从锚点几何和开发 CSV 复建四个准备数组与冻结历史，核对按日历分段、未知标签、同折支持与抽样。",
        "3. 核对注册、勘误、开发选择、阈值、权重和冻结解封顺序。",
        "4. 从全部合格逐窗分数筛 known 复算 AP、F1、混淆、等召回和日期配对区间，按固定权重抽样重放。",
        "",
        "[运行命令和导航](README_审计入口.md)。干净解压收据与 ZIP 同级。",
        ""]
    (root/"审计报告.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"审计报告.md"),figures=3,selected=selected,external_goal=state["goal_reached"],
               stable_recovery_FP_reduction=state["stable_recovery_FP_reduction"]),flush=True)


if __name__=="__main__":
    render(*settings())

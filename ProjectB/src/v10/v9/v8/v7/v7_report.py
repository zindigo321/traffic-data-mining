"""Readable figures and measured-only V7 audit report."""
from datetime import date,timedelta
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from v7_io import V6,load,settings
from v7_train import folder,read_npz

LABELS={"lead_attention":"提前 Attention（预选）","lead_mean":"提前 Mean","lead_max":"提前 Max",
        "lead_center":"提前仅中心","lead_logit":"提前逻辑回归","lead_lag":"提前滞后规则","lead_majority":"提前全判非拥堵",
        "current_attention":"当前 Attention","current_rule":"当前上一格低速规则","current_lag":"当前相对速度规则",
        "current_majority":"当前全判非拥堵"}
FIG={"lead_attention":"Warning attention","lead_center":"Warning center only","lead_logit":"Logistic regression",
     "lead_lag":"Lag-2 rule","current_attention":"Current-window attention"}


def period(bounds):
    a,b=map(date.fromisoformat,bounds)
    return f"{a} 至 {b-timedelta(days=1)}"


def plots(root,cfg):
    p=root/"figures"
    p.mkdir(exist_ok=True)
    frame=pd.read_csv(root/"results/analysis/matched_recall.csv")
    frame=frame[(frame.stage=="development")&(frame.support=="full")]
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.8),layout="constrained")
    for model in ("lead_attention","lead_center","lead_logit","lead_lag"):
        d=frame[frame.model==model].groupby("target_recall")[["FP_per_day","recovery_FP_per_day"]].mean()
        for ax,metric,title in zip(axes,("FP_per_day","recovery_FP_per_day"),("All false alerts/day","Recovery false alerts/day"),strict=True):
            ax.plot(100*d.index,d[metric],label=FIG[model],marker="o")
            ax.axvline(60,color="#777777",alpha=.25,lw=1,ls="--")
            ax.set(xlabel="Target recall (%)",ylabel=title)
            ax.grid(axis="y",alpha=.22)
    axes[0].legend(frameon=False)
    fig.suptitle("V7 NYC development | matched recall on all known warning windows")
    fig.savefig(p/"matched_recall.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,4.8),layout="constrained")
    for model in ("lead_attention","lead_center","lead_logit","current_attention"):
        seed=None if model=="lead_logit" else cfg["seeds"][0]
        z=read_npz(folder(root,"f3",model,seed)/"assessment.npz")
        known=z["y"]>=0
        if model.startswith("lead_"):
            known &= z["current_eligible"]
        precision,recall,_=precision_recall_curve(z["y"][known],z["score"][known])
        ax.step(recall,precision,where="post",label=FIG[model])
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",title="V7 F3 development | shared known windows")
    ax.legend(frameon=False)
    ax.grid(alpha=.22)
    fig.savefig(p/"common_PR_f3.png",dpi=160)
    plt.close(fig)


def fmt(value,sd=None):
    if value is None:
        return "N/A"
    return f"{value:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")


def metrics_table(metrics,models):
    rows=["| 配方 | 平衡准确率 | AP | 正类 F1 | 60% 召回总 FP／日 | 60% 召回恢复 FP／日 |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        val=metrics[m]
        scores=" | ".join(fmt(val[k],val[k+"_sd"]) for k in ("balanced_accuracy","AP","f1"))
        rows.append(f"| {LABELS[m]} | {scores} | {val['FP60_day']:.3f} | {val['recovery_FP60_day']:.3f} |")
    return rows


def render(root,cfg):
    s=load(root/"evidence/status.json")
    g=load(root/"evidence/date_gate.json")
    proof=load(root/"evidence/verification_replay.json")
    freeze=load(root/"evidence/calibration_freeze.json")
    graph=load(root/"evidence/graph.json")
    pre=load(root/"evidence/prereg.json")
    stats=load(root/"results/analysis/aggregates.json")
    ci=pd.read_csv(root/"results/analysis/date_blocks.csv")
    plots(root,cfg)
    d=stats["development"]["full"]["lead_attention"]
    cal=stats["calibration"]["full"]["lead_attention"]
    a=stats["development"]["common"]["current_attention"]
    b=stats["development"]["common"]["lead_attention"]
    lines=[
        "# Project B V7 审计报告",
        "",
        "**三折开发和独立校准已完成。纽约正式验证 blocked／N/A，Chicago 原口径和 Geolife 分列 blocked。** "
        "提前十分钟预警按 t−2 及更早输入，原当前格判定按 t−1 及更早输入，两个协议共用原标签。",
        "",
        "## 日期、证据和实际支持",
        "",
        f"新留出登记为纽约本地 **{period(cfg['holdout'])}**。官方时间查询最新 {g['latest'] or 'N/A'}，登记区间未结束。"
        "V7 未传输、解析或评估这段速度和标签。旧 V5、V6 留出 2026-08-31 至 11-08 只作历史参考。"
        "源核验只读原 ZIP 指纹、代码和许可的开发原料。 "
        "[日期证据](evidence/date_gate.json) · [计划与禁用区间](plan.md) · [NYC DOT 数据](https://data.cityofnewyork.us/d/i4gi-tjb9) · "
        "[Chicago 停更](https://data.cityofchicago.org/stories/s/c28d-stgw)",
        "",
        f"固定 {graph['road_count']} 条有向路段、{graph['edges']} 条候选边。A 复建数组逐字节匹配 V6，"
        "B 的全部真值与 A 逐格相同。B 按固定日历全量出分，原 A 保留上一格资格。共同支持为 A 的已知键。 "
        "[原料验签](evidence/inputs.json) · [复建与真值对齐](evidence/prepared_development.json) · [时点合同](方法与数据合同.md)",
        "",
        "| 评估段 | 本地日期 | B 固定出分键 | B 全量已知 | B unknown | A／B 共同已知 | 正例 | 恢复负例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold,bounds in {**cfg["folds"],"final":cfg["calibration"]}.items():
        row=load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]
        lines.append(f"| {'独立校准' if fold=='final' else fold.upper()} | {period(bounds)} | {row['expected']:,} | "
                     f"{row['positive']+row['negative']:,} | {row['unknown_eligible']:,} | {row['current_known']:,} | "
                     f"{row['positive']:,} | {row['high_risk_recovered']:,} |")
    lines += ["",
        "unknown 保留在出分文件和数据漏斗，监督评价只用已知。A、B 共同键用 B 在其全量已知样本上定出的原阈值计算。"
        "数据量不足的开峰滞后按缺失位处理。 "
        "[逐折支持示例](evidence/fold_lead_f1.json) · [全量及共同指标](results/analysis/phase_metrics.csv)",
        "",
        "## 开发三折实测",
        "",
        "各神经组按种子 11、22、33 先作三折等权平均，再报告种子均值和标准差。规则和逻辑回归每折一次。"
        "开发使用同折选择 AP checkpoint 并计算诊断阈值。", ""]
    leads=["lead_attention","lead_mean","lead_max","lead_center","lead_logit","lead_lag","lead_majority"]
    lines+=metrics_table(stats["development"]["full"],leads)
    lines+=["",
        f"在共同已知键上，B Attention 的 AP／正类 F1 为 {b['AP']:.4f}／{b['f1']:.4f}，"
        f"A 当前 Attention 为 {a['AP']:.4f}／{a['f1']:.4f}。"
        "提前协议移除了标签中的上一格动态量，并改用全量已知训练和无路由评分，差值反映两个协议的整体难度。"
        f"B 全量 BA 为 {d['balanced_accuracy']:.4f}，风险组两类等权准确率为 {d['risk_balanced_accuracy']:.4f}。 "
        "[共同键 PR](figures/common_PR_f3.png) · [共同样本指标](results/analysis/phase_metrics.csv)",
        "",
        "## 28 日独立校准诊断",
        "",
        f"先用开发折锁定中位 epoch，再拟合最终模型。B 用校准最大 BA，A 用原最大 F1。"
        f"冻结时间 {freeze['frozen_utc']}。表内为该校准段的诊断值。 "
        "[冻结权重和阈值](evidence/calibration_freeze.json) · [检查点和种子](results/development_receipt.json)",
        ""]
    lines+=metrics_table(stats["calibration"]["full"],leads[:-1])
    lines+=["",
        f"校准期 B Attention 的 AP／BA／正类 F1 为 {cal['AP']:.4f}／{cal['balanced_accuracy']:.4f}／{cal['f1']:.4f}。"
        "BA 最大的工作点和以 F1 最大为目标的工作点会有不同取舍。全部混淆、分类阈值、ROC-AUC、覆盖、成本和逐种子结果随包。 "
        "[阈值和混淆](results/analysis/phase_metrics.csv)",
        "",
        "## 等召回误报和配对区间",
        "",
        "从全量已知分数中取达到相同 TP 数的前缀，主点目标 60%。总误报和实际上一格低速后恢复的误报分列。"
        "置信区间以相同日历日期、1,000 次共享日期块重采样，报告 Attention 减基线。表为三折开发诊断。 ",
        "",
        "| 基线 | ΔAP（95% 区间） | Δ总 FP／日（95% 区间） | Δ恢复 FP／日（95% 区间） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("lead_lag","lead_logit","lead_center"):
        pick=ci[(ci.stage=="development")&(ci.baseline==name)].set_index("metric")
        terms=[]
        for k in ("AP","FP60_day","recovery_FP60_day"):
            row=pick.loc[k]
            terms.append(f"{row.difference:+.4f} [{row.low_95:+.4f}, {row.high_95:+.4f}]")
        lines.append(f"| {LABELS[name]} | "+" | ".join(terms)+" |")
    lines+=["",
        "与中心消融相比，恢复 FP 在主点减少，AP 和总 FP 的差值区间跨零。校准段的中心模型 AP 略高。"
        "三折分数支持提前排序改善的开发证据，跨新日期表现待正式留出。 "
        "[全部召回点](results/analysis/matched_recall.csv) · [日期块](results/analysis/date_blocks.csv) · [误报曲线](figures/matched_recall.png)",
        "",
        "## 验收与复核",
        "",
        "改写 t−1 和 t 起的中心、邻居观测后，B 输入和三池化分数一致，真值改变。另检查滞后边界、冻结先验及采样独立于实际风险。"
        f"复算 {proof['assessments']} 组配置、{proof['matched_rows']} 行等召回记录，"
        f"抽样重放 {proof['replayed']} 组，最大分数差 {proof['max_score_difference']:.3g}。"
        "34 项工程测试均只证明工程机制。 "
        "[复算和权重重放](evidence/verification_replay.json) · [测试报告](测试报告.md)",
        "",
        "1. 核验 V6 原 ZIP、V7 清单、开发缓存和固定图的 SHA。",
        "2. 重建两协议的开发数组，检查旧日期、真值、观测时点和全量键。",
        "3. 从逐窗 NPZ 独立复算 AP、BA、F1、混淆、阈值及等召回，再抽样安全重放。",
        "4. 对照正式状态、日期闸门和冻结模型。发布未覆盖全部预登记日期时保持 blocked。",
        "",
        "[一键复现和审计导航](README_审计入口.md)。干净解压收据和 ZIP 同级。",
        ""]
    (root/"审计报告.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"审计报告.md"),NYC_formal=s["NYC_formal"],development_AP=d["AP"],development_F1=d["f1"]),flush=True)


if __name__=="__main__":
    render(*settings())

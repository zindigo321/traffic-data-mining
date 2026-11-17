"""Measured-only V8 audit of global operating points and equal recall."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from v8_io import load,settings

NAMES={"attention":"V7 Attention","center":"V7 仅中心路段","rank":"V8 尾部排序"}
RULES={"max_ba":"最大 BA","max_f1":"最大 F1","constrained":"误报约束 F1"}
ENG={"max_ba":"Max BA","max_f1":"Max F1","constrained":"Constrained F1"}


def plots(root,cfg,policies,model):
    out=root/"figures"
    out.mkdir(exist_ok=True)
    order=("max_ba","max_f1","constrained")
    values=policies["calibration"][model]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout="constrained")
    x=np.arange(3)
    for key,off,title,color in (("balanced_accuracy",-.18,"Balanced accuracy","#36688d"),("f1",.18,"Positive F1","#ce7d3e")):
        axes[0].bar(x+off,[values[k][key] for k in order],.35,label=title,color=color)
    axes[0].set(ylim=(0,1),ylabel="Metric",xticks=x,xticklabels=[ENG[k] for k in order])
    for key,off,title,color in (("fp_per_day",-.18,"Total false alerts","#36688d"),
                                ("recovery_FP_day",.18,"Recovery false alerts","#ce7d3e")):
        axes[1].bar(x+off,[values[k][key] for k in order],.35,label=title,color=color)
    axes[1].axhline(cfg["FP_budget_day"],ls="--",color="#36688d",lw=1,alpha=.65)
    axes[1].axhline(cfg["recovery_budget_day"],ls="--",color="#ce7d3e",lw=1,alpha=.65)
    axes[1].set(ylabel="False alerts per calendar day",xticks=x,xticklabels=[ENG[k] for k in order])
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(axis="y",alpha=.22)
    fig.suptitle("NYC V8 | one scoring model, three thresholds | calibration diagnostics")
    fig.savefig(out/"calibration_thresholds.png",dpi=160)
    plt.close(fig)
    points=pd.read_csv(root/"results/analysis/matched_recall.csv")
    points=points[points.stage=="development"]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout="constrained")
    for name,label in (("attention","V7 attention"),("center","V7 center only"),("rank","V8 tail ranking")):
        rows=points[points.model==name].groupby("target_recall")[["FP_per_day","recovery_FP_per_day"]].mean()
        for ax,key,title in zip(axes,("FP_per_day","recovery_FP_per_day"),("All false alerts/day","Recovery false alerts/day"),strict=True):
            ax.plot(100*rows.index,rows[key],label=label,marker="o")
            ax.set(xlabel="Target recall (%)",ylabel=title)
            ax.grid(axis="y",alpha=.23)
    for ax in axes:
        ax.axvline(60,ls="--",color="#777777",lw=1)
        ax.legend(frameon=False)
    fig.suptitle("NYC V8 | full known support | matched recall on development folds")
    fig.savefig(out/"matched_recall.png",dpi=160)
    plt.close(fig)


def operating(rows,model,choice=None):
    output=[]
    for rule in ("max_ba","max_f1","constrained"):
        a=rows[model][rule]
        label=RULES[rule]+("（冻结）" if rule==choice else "")
        output.append(f"| {label} | {a['AP']:.4f} | {a['balanced_accuracy']:.4f} | {a['f1']:.4f} | "
                      f"{a['recall']:.4f} | {a['fp_per_day']:.3f} | {a['recovery_FP_day']:.3f} |")
    return output


def render(root,cfg):
    base=root/"results/analysis"
    summary=load(base/"threshold_summary.json")
    ranking=load(base/"ranking_summary.json")
    status=load(root/"evidence/status.json")
    gate=load(root/"evidence/date_gate.json")
    lock=load(root/"evidence/freeze.json")
    proof=load(root/"evidence/verification_replay.json")
    bands=pd.read_csv(base/"paired_date_blocks.csv")
    model,rule=status["frozen_model"],status["frozen_rule"]
    plots(root,cfg,summary,model)
    old=summary["calibration"]["attention"]["max_ba"]
    chosen=summary["calibration"][model][rule]
    dev=summary["development_transfer"][model]
    bounds=bands[(bands.stage=="development")&(bands.contrast=="rank_minus_attention")].set_index("metric")
    rank=ranking["development"]
    n=sum(load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]["positive"]+
          load(root/"evidence"/f"fold_lead_{fold}.json")["parts"][1]["negative"] for fold in cfg["folds"])
    support=load(root/"evidence/fold_lead_final.json")["parts"][1]
    head=["| 阈值 | AP | BA | 正类 F1 | Recall | 总 FP／日 | 恢复 FP／日 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines=[
        "# Project B V8 审计报告",
        "",
        f"**已完成阈值迁移诊断和 28 日独立校准，冻结 {NAMES[model]}＋{RULES[rule]}。** "
        "纽约正式验证 blocked／N/A，Chicago 原口径、Geolife 各自 blocked。",
        "",
        "## 时间合同、真值和样本",
        "",
        "固定纽约本地 2026-11-09 至 12-06 新留出，V5/V6 日期 08-31 至 11-08 禁止用于训练、校准和选型。"
        f"本次官方时间查询最新 {gate['latest'] or 'N/A'}，登记日期未结束。无新速度、标签或正式结果。"
        "只用 V7 的经过校验的严格提前输入，t−1、t 与未来速度仍只在真值侧。 "
        "[日期闸门](evidence/date_gate.json) · [预登记计划](plan.md) · [NYC DOT 数据](https://data.cityofnewyork.us/d/i4gi-tjb9) · "
        "[Chicago 停更说明](https://data.cityofchicago.org/stories/s/c28d-stgw)",
        "",
        f"三折全量已知支持合计 {n:,} 窗，28 日校准 {support['positive']+support['negative']:,} 个已知窗、"
        f"{support['positive']:,} 个正例、{support['high_risk_recovered']:,} 个恢复负例，{support['unknown_eligible']:,} 个 unknown。"
        "同一折的 3 个模型、3 个种子共用相同键、标签、风险审计和训练抽样。"
        "复用 V7 Attention 与中心 24 组验签预测和权重；新增 Attention 尾部排序从头训练 12 次。 "
        "[输入证据](evidence/inputs.json) · [校准支持](evidence/fold_lead_final.json) · [方法与时点](方法与数据合同.md)",
        "",
        "## 等召回排序对照",
        "",
        "在全量已知评分上取相同 TP 的前缀。三种阈值只改变工作点；同一模型 AP 和固定召回排序相同。"
        "下表是各阶段的三种子均值，开发先按种子平均三折。",
        "",
        "| 阶段 | 评分模型 | AP | 60% 召回总 FP／日 | 60% 召回恢复 FP／日 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for stage,label in (("development","三折开发"),("calibration","独立校准")):
        for name in ("attention","center","rank"):
            a=ranking[stage][name]
            lines.append(f"| {label} | {NAMES[name]} | {a['AP']:.4f} | {a['FP60_day']:.3f} | {a['recovery_FP60_day']:.3f} |")
    delta=rank["rank"]["recovery_FP60_day"]-rank["attention"]["recovery_FP60_day"]
    inter=bounds.loc["recovery_FP60_day"]
    lines+=["",
        f"排序候选符合预登记的开发准入，主点恢复 FP 日差 {delta:+.3f}、两个种子下降。"
        f"相对 V7 Attention 的 1,000 次日期分块 95% 区间为 [{inter.low_95:+.3f}, {inter.high_95:+.3f}]。"
        "校准段相对两组基线的同召回恢复误报略高。现有证据支持阈值取舍，排序改善仍待独立日期确认。 "
        "[配对区间](results/analysis/paired_date_blocks.csv) · [完整等召回](results/analysis/matched_recall.csv) · [曲线](figures/matched_recall.png)",
        "",
        "## 开发阈值迁移与独立校准",
        "",
        "每个开发折前七日定阈值，后七日评估迁移，模型 checkpoint 沿用整折选出的权重。"
        "28 日独立校准表为同段拟合阈值的诊断，完整阈值和逐种子计数均保留。预先设总 FP／日≤12、恢复 FP／日≤6。",
        "",
        f"### {NAMES[model]} 三折后七日",
        "",
        *head,
        *operating(summary["development_transfer"],model),
        "",
        f"### {NAMES[model]} 28 日独立校准",
        "",
        *head,
        *operating(summary["calibration"],model,rule),
        "",
        f"选择规则比较显示，最大 F1 的开发迁移满足平均预算，独立校准超预算。"
        f"{RULES[rule]} 在两段满足冻结条件，三种子阈值在 {lock['at_utc']} 锁定。"
        f"相对 V7 Attention 原最大 BA 阈值，校准正类 F1 {old['f1']:.4f}→{chosen['f1']:.4f}，"
        f"恢复 FP／日 {old['recovery_FP_day']:.3f}→{chosen['recovery_FP_day']:.3f}，"
        f"召回 {old['recall']:.4f}→{chosen['recall']:.4f}。 "
        "[逐行阈值、误报与混淆](results/analysis/thresholds.csv) · [冻结收据](evidence/freeze.json) · [校准图](figures/calibration_thresholds.png)",
        "",
        "| 同校准误报约束 | 正类 F1 | BA | 恢复 FP／日 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("attention","center","rank"):
        v=summary["calibration"][name]["constrained"]
        lines.append(f"| {NAMES[name]} | {v['f1']:.4f} | {v['balanced_accuracy']:.4f} | {v['recovery_FP_day']:.3f} |")
    lines += ["",
        "三种评分采用同一约束后校准 F1 接近，V7 两组略高。正类 F1 的主要变化来自工作点调整。"
        "预算是在校准及开发迁移中量化，正式日期的误报与召回仍待完整覆盖。 "
        "[三组阈值汇总](results/analysis/threshold_summary.json)",
        "",
        "## 验收",
        "",
        f"独立复算 {proof['score_sets']} 组分数、{proof['threshold_evaluations']} 个阈值诊断和 {proof['matched_points']} 个等召回点，"
        f"抽样重放 {proof['replayed']} 组，最大分数差 {proof['max_score_difference']:.2g}。"
        "40 项工程测试只用于实现核查。冻结哈希、命令、实际退出码、数据映射及干净解压复核随包。 "
        "[复算](evidence/verification_replay.json) · [测试](测试报告.md) · [复现](README_审计入口.md)",
        ""]
    (root/"审计报告.md").write_text("\n".join(lines),encoding="utf-8")
    print(dict(report=str(root/"审计报告.md"),model=model,rule=rule,formal=status["NYC_formal"]),flush=True)


if __name__=="__main__":
    render(*settings())

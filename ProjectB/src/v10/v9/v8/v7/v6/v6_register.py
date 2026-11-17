"""Freeze exact V6 dates, model identities and metrics before transport."""
from datetime import date,timedelta
import os
from pathlib import Path

from v6_check import EPOCHS,OLD
from v6_io import CONFIG,V5,load,now,save,sha


def register(root):
    if CONFIG.exists() or (root/"evidence/registration.json").exists():
        raise FileExistsError("Registration already exists")
    pre=load(root/"evidence/plan_receipt.json")
    gate=load(root/"evidence/date_gate.json")
    prior=load(root/"evidence/v5_integrity.json")
    if sha(root/"plan.md")!=pre["sha256"] or pre["sha256"]!=gate["preregistration_plan_sha256"]:
        raise ValueError("Preregistration plan changed")
    if gate["observation_values_read"] is not False or prior["status"]!="PASS":
        raise ValueError("Evidence gate incomplete")
    new=gate["interval"]
    status="eligible" if gate["status"]=="eligible" else "blocked"
    if status=="eligible":
        if not (new[0]>=OLD[1] and new[1]>new[0] and gate["substantial_days"]>=14):
            raise ValueError("Independent calendar failed")
        wording=f"**{new[0]} 至 {date.fromisoformat(new[1])-timedelta(days=1)}**"
    else:
        wording="**blocked／N/A**"
    cfg=load(V5/"config.json")
    if cfg["holdout"]!=OLD:
        raise ValueError("Prior test boundaries changed")
    cfg.update(holdout=new or [OLD[1],OLD[1]],formal_eligible=status=="eligible",prior_holdout=OLD,
        primary_recipe="X1",primary_pool="attention",epochs=EPOCHS,
        controls=[["X1","mean"],["B","mean"]],minimum_affected_seeds=2,recipes=["X1"])
    save(CONFIG,cfg)
    note=f"""# V6 纽约独立留出登记

纽约本地留出 {wording}。候选起止、发布上限、每个高峰计数和失败日期均见 evidence/date_gate.json。实际登记日历为左闭右开 {new}。区间含 {gate['calendar_days']} 个日历日，其中 {gate['substantial_days']} 个通过双高峰报告覆盖闸门，所有日历日都保留。日期选择只用了官方时间、路段 ID 和计数。

V5 旧留出 2026-08-31 至 2026-10-11 全部禁用；仅查既有聚合结果。Chicago 原口径和 Geolife 保持 blocked。图、标签、特征时点、unknown、训练和全量评价规则沿用 V5。V6 重建并核验既有开发数据，只有新留出速度和标签在本轮首次使用。

正式配方 X1＋Attention，两个配对对照 X1＋Mean 和 V4 P1＋Mean。X1＋Mean 同时作为三池化组之一。种子 11、22、33。配置继承 V5 解封前锁定 epoch，训练后各自使用同一 28 日校准阈值规则。稳定性看 60% 等召回的配对平均误报差、至少两种子下降及共享日期块区间上界低于零，分别判定两对照。正式双 0.60 只看预选 Attention。完整计划另见 plan.md。

新分片传输在本登记之后，标签和速度在模型与阈值冻结后才解析。
"""
    path=root/"registered_holdout.md"
    path.write_text(note,encoding="utf-8")
    entry=dict(at_utc=now(),status=status,interval=new,prior_holdout=OLD,primary=["X1","attention"],controls=cfg["controls"],
        plan_sha256=sha(root/"plan.md"),date_gate_sha256=sha(root/"evidence/date_gate.json"),
        calendar_sha256=sha(path),config_sha256=sha(CONFIG),epochs=EPOCHS,holdout_values_read=False)
    save(root/"evidence/registration.json",entry)
    print(entry,flush=True)


def verify_registration(root,cfg):
    row=load(root/"evidence/registration.json")
    checks={root/"plan.md":row["plan_sha256"],root/"registered_holdout.md":row["calendar_sha256"],
            CONFIG:row["config_sha256"],root/"evidence/date_gate.json":row["date_gate_sha256"]}
    if any(sha(file)!=h for file,h in checks.items()) or row["holdout_values_read"] is not False:
        raise ValueError("Registered calendar or evidence changed")
    if (row["status"]=="eligible")!=cfg["formal_eligible"] or row["primary"]!=[cfg["primary_recipe"],cfg["primary_pool"]]:
        raise ValueError("Fixed primary or eligibility changed")
    if cfg["formal_eligible"] and row["interval"]!=cfg["holdout"]:
        raise ValueError("Holdout interval differs from preregistration")
    return row


if __name__=="__main__":
    register(Path(os.environ["GT6_ROOT"]))

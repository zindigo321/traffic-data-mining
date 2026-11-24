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
        wording=f"**{new[0]} to {date.fromisoformat(new[1])-timedelta(days=1)}**"
    else:
        wording="**blocked／N/A**"
    cfg=load(V5/"config.json")
    if cfg["holdout"]!=OLD:
        raise ValueError("Prior test boundaries changed")
    cfg.update(holdout=new or [OLD[1],OLD[1]],formal_eligible=status=="eligible",prior_holdout=OLD,
        primary_recipe="X1",primary_pool="attention",epochs=EPOCHS,
        controls=[["X1","mean"],["B","mean"]],minimum_affected_seeds=2,recipes=["X1"])
    save(CONFIG,cfg)
    note=f"""# V6 New York Independence holdout Registration

NYC local-time holdout {wording}. See evidence/date_gate.json for candidate endpoints, publication cutoff, counts for each peak period, and failed dates. The registered calendar is left-inclusive and right-exclusive {new}. It contains {gate['calendar_days']} calendar days, of which {gate['substantial_days']} pass the reporting coverage gate for both peaks. All calendar days are retained. Date selection used only official timestamps, road IDs, and counts.

Exclude the entire previous V5 holdout from 2026-08-31 through 2026-10-11. Use only its existing aggregates for context. The original Chicago protocol and Geolife remain blocked. V6 follows V5 for the graph, labels, feature timing, unknown labels, training, and full evaluation. V6 rebuilds and verifies the existing development data. Only the new holdout speeds and labels are used for the first time.

The formal recipe is X1 plus Attention. Paired controls are X1 plus Mean and V4 P1 plus Mean. X1 plus Mean is also one of the three pooling configurations. Seeds are 11, 22, and 33. Use the locked epochs from before V5 unsealing, then apply the same threshold rule on 28 days of calibration. For each control, assess stability at 60% matched recall using the paired average false-positive difference, reductions in at least two seeds, and a shared date-block interval with upper bound below zero. The formal dual 0.60 target applies to the preselected Attention model. See plan.md for the complete plan.

Transfer the new shards after registration. Parse labels and speeds only after freezing the models and thresholds.
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

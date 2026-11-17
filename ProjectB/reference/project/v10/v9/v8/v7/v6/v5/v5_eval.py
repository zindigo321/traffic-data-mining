"""One-shot external test, paired matched recall and shared calendar-block comparisons."""
from datetime import date
from pathlib import Path
import time

import numpy as np
import pandas as pd

from v5_io import load,now,save,settings,sha
from v5_lock import transport,verify_lock
from v5_prepare import fingerprint_keys,holdout,read_npz
from v5_train import model_for,study_for,typed_score
from v4_analysis import bootstrap_scores
from v4_matrix import average
from v4_metrics import matched_points
from evaluation import measure
from experiment import predict
from v1.graph import TrainPreprocessor
from v3_train import risk_metrics


def paired(out,records,scores,cfg):
    lookup={(r["recipe"],r["pool"],r["seed"]):r for r in records}
    baseline={s:lookup["B","mean",s] for s in cfg["seeds"]}
    points={k:load(out/r["job"]/"matched_recall.json") for k,r in lookup.items()}
    diffs=[]
    for (r,p,s),values in points.items():
        if r=="B":
            continue
        for a,b in zip(points["B","mean",s],values,strict=True):
            if any(a[k]!=b[k] for k in ("target_recall","achieved_recall","TP","FN")):
                raise ValueError("Different matched recall support")
            diffs.append(dict(recipe=r,pool=p,seed=s,target_recall=b["target_recall"],actual_recall=b["achieved_recall"],
                              TP=b["TP"],V4_recovery_FP=a["recovery_FP"],V5_recovery_FP=b["recovery_FP"],
                              delta_FP=b["FP"]-a["FP"],delta_recovery_FP=b["recovery_FP"]-a["recovery_FP"],
                              delta_FP_per_day=b["FP_per_day"]-a["FP_per_day"],
                              delta_AP=lookup[r,p,s]["AP"]-baseline[s]["AP"],
                              delta_F1=lookup[r,p,s]["f1"]-baseline[s]["f1"]))
    pd.DataFrame(diffs).to_csv(out/"paired.csv",index=False)
    days=(date.fromisoformat(cfg["holdout"][1])-date.fromisoformat(cfg["holdout"][0])).days
    first=(date.fromisoformat(cfg["holdout"][0])-date(1970,1,1)).days
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    weights=rng.multinomial(days,np.full(days,1/days),size=cfg["bootstrap_reps"])
    daily={key:bootstrap_scores(z,weights,first,cfg["main_recall"]) for key,z in scores.items()}
    base_mean=np.mean([daily["B","mean",s] for s in cfg["seeds"]],axis=0)
    rows=[]
    summary={}
    for recipe,pool in dict.fromkeys((r,p) for r,p,_ in daily):
        means=np.mean([daily[recipe,pool,s] for s in cfg["seeds"]],axis=0)
        for typ,values in (("absolute",means),("vs_B",means-base_mean)):
            for j,k in enumerate(("AP","F1_frozen","FP60_per_day")):
                use=values[:,j]
                use=use[np.isfinite(use)]
                lo,hi=np.percentile(use,[2.5,97.5])
                row=dict(recipe=recipe,pool=pool,kind=typ,metric=k,low_95=lo,high_95=hi,valid_replicates=len(use))
                rows.append(row)
                summary[recipe,pool,typ,k]=row
    pd.DataFrame(rows).to_csv(out/"date_blocks.csv",index=False)
    return diffs,summary


def error_review(out,records,all_scores,data,keys,cfg,cov):
    merged=[]
    s=study_for(data,"B")
    idx=s.val
    day=keys.ravel()[idx]//1000//144
    minute=data["minute"][idx//s.nroad]
    prior_ratio=s.ratio[idx]
    depth=data["x"].reshape(-1,data["x"].shape[-1])[idx,31].astype(int)
    for row in records:
        z=all_scores[row["recipe"],row["pool"],row["seed"]]
        _,positions=matched_points(z["sample_id"],z["y"],z["score"],z["risk"],[cfg["main_recall"]],cov["total_days"])
        decision=np.zeros(len(z["y"]),dtype=bool)
        decision[positions[cfg["main_recall"]]]=True
        use=z["risk"]
        out_label=np.select([(z["y"]==1)&decision,z["y"]==1,decision],["TP","FN","FP"],default="TN")
        merged.append(pd.DataFrame(dict(recipe=row["recipe"],pool=row["pool"],seed=row["seed"],sample_id=z["sample_id"][use],
                  y=z["y"][use],score=z["score"][use],matched_recall_outcome=out_label[use],
                  day=pd.to_datetime(day[use],unit="D").strftime("%Y-%m-%d"),
                  peak=np.where(minute[use]%1440<720,"AM","PM"),
                  prior_ratio=prior_ratio[use],prior_low_depth=depth[use],
                  road=z["sample_id"][use]%1000)))
    frame=pd.concat(merged,ignore_index=True)
    frame.to_csv(out/"risk_windows.csv",index=False,float_format="%.17g")
    results=[]
    for key in ("peak","prior_low_depth","road"):
        groups=frame.groupby(["recipe","pool","seed",key,"matched_recall_outcome"],sort=True).size().reset_index(name="windows")
        groups.insert(3,"dimension",key)
        groups=groups.rename(columns={key:"category"})
        results.append(groups)
    pd.concat(results).to_csv(out/"errors_grouped.csv",index=False)


def evaluate(root,cfg):
    start=time.perf_counter()
    ev=root/"evidence"
    if any((root/name).exists() for name in ("evidence/unlock.json","prepared/holdout.npz","results/holdout")):
        raise PermissionError("This external holdout has already been opened")
    lock=verify_lock(root,cfg)
    receipt=transport(root,cfg)
    save(ev/"unlock.json",dict(unlocked_utc=now(),freeze_sha256=sha(ev/"freeze.json"),
        transport_sha256=sha(ev/"download_holdout_sealed.json"),holdout=cfg["holdout"],
        shards=receipt["complete"],prior_holdout_values_read=False))
    data,keys,cover=holdout(root,cfg,lock)
    eligible=data["eligible"]
    known=eligible&(data["y"]>=0)
    if not ((data["y"][known]==1).any() and (data["y"][known]==0).any()
            and (data["risk"][known]&(data["y"][known]==0)).any()):
        save(ev/"status.json",dict(status="blocked",nyc_formal="blocked",Chicago_formal="blocked",reason="Test lacks required labels or recovery support",AP=None,F1=None))
        raise ValueError("Known positive or recovery support absent")
    out=root/"results/holdout"
    out.mkdir()
    np.savez_compressed(out/"support.npz",sample_id=keys.ravel(),eligible=eligible.ravel(),y=data["y"].ravel(),
        risk=data["risk"].ravel(),known_ids=keys[known],eligible_ids=keys[eligible])
    save(ev/"holdout_support.json",dict(all_sha256=fingerprint_keys(keys),eligible_sha256=fingerprint_keys(keys[eligible]),
        known_sha256=fingerprint_keys(keys[known]),expected=int(keys.size),eligible=int(eligible.sum()),known=int(known.sum()),
        positive=int((known&(data["y"]==1)).sum()),date_interval=cfg["holdout"],
        prepared_sha256=sha(root/"prepared/holdout.npz"),at_utc=now()))
    records,scores=[],{}
    points_csv=[]
    for spec in lock["models"]:
        began=time.perf_counter()
        source=root/"results/calibration"/spec["job"]
        model=out/spec["job"]
        model.mkdir()
        opt=load(source/"options.json")
        study=study_for(data,spec["recipe"])
        study.configure(TrainPreprocessor.load(source/"preprocessor.npz"))
        net=model_for(cfg,study,opt,source/"weight.pt")
        positions=np.flatnonzero(study.eligible)
        raw=typed_score(study,positions,*predict(net,study,positions,cfg))
        np.testing.assert_array_equal(raw["sample_id"],keys[eligible])
        cut=spec["threshold"]
        raw.update(threshold=np.float64(cut),pred=raw["score"]>=cut)
        np.savez_compressed(model/"all_eligible.npz",**raw)
        take=raw["y"]>=0
        z={k:v[take] for k,v in raw.items() if k!="threshold"}
        np.testing.assert_array_equal(z["sample_id"],keys[known])
        stats=measure(z["y"],z["score"],cut,z["sample_id"],cover)
        risk=risk_metrics(z,cut)
        fixed,_=matched_points(z["sample_id"],z["y"],z["score"],z["risk"],cfg["recalls"],cover["total_days"])
        save(model/"matched_recall.json",fixed)
        points_csv.extend({**{k:spec[k] for k in ("recipe","pool","seed","selected")},**row} for row in fixed)
        main=next(x for x in fixed if x["target_recall"]==cfg["main_recall"])
        assert main["other_FP"]==0
        cal=load(source/"status.json")
        row=dict(**{k:spec[k] for k in ("recipe","pool","seed","job","selected")},**stats,**risk,
                 FP60=main["FP"],TP60=main["TP"],FP60_per_day=main["FP_per_day"],eval_seconds=time.perf_counter()-began,
                 fit_seconds=cal["fit_seconds"],peak_RSS_bytes=cal["peak_RSS_bytes"],weight_sha256=spec["hashes"]["weight.pt"],
                 forecast_contract="Uses only information before target 10-minute grid")
        save(model/"status.json",row)
        records.append(row)
        scores[spec["recipe"],spec["pool"],spec["seed"]]=z
        print({k:row[k] for k in ("job","AP","f1","FP60_per_day","fp","tp","fn")},flush=True)
    pd.DataFrame(records).to_csv(out/"metrics.csv",index=False,na_rep="N/A")
    pd.DataFrame(points_csv).to_csv(out/"matched_recall.csv",index=False)
    groups={f"{r}_{p}":average([s for s in records if (s["recipe"],s["pool"])==(r,p)])
            for r,p in dict.fromkeys((x["recipe"],x["pool"]) for x in records)}
    save(out/"aggregates.json",groups)
    diffs,ci=paired(out,records,scores,cfg)
    error_review(out,records,scores,data,keys,cfg,cover)
    recipe,pool=lock["choice"]["recipe"],lock["choice"]["pool"]
    selected=f"{recipe}_{pool}"
    stats=groups[selected]
    primary=[row for row in diffs if (row["recipe"],row["pool"],row["target_recall"])==(recipe,pool,cfg["main_recall"])]
    upper=ci[recipe,pool,"vs_B","FP60_per_day"]["high_95"]
    lower=ci[recipe,pool,"vs_B","FP60_per_day"]["low_95"]
    negative=sum(row["delta_recovery_FP"]<0 for row in primary)
    delta=float(np.mean([row["delta_FP_per_day"] for row in primary]))
    result=dict(status="complete",nyc_formal="complete",Chicago_formal="blocked",Geolife_formal="blocked",
        chosen=selected,AP=stats["AP"],F1=stats["f1"],goal=cfg["goal"],
        goal_reached=stats["AP"]>=cfg["goal"] and stats["f1"]>=cfg["goal"],
        AP_gap=cfg["goal"]-stats["AP"],F1_gap=cfg["goal"]-stats["f1"],
        stable_recovery_FP_reduction=bool(delta<0 and negative>=2 and upper<0),
        matched_recall=cfg["main_recall"],delta_FP_per_day=delta,delta_FP_95=[lower,upper],seeds_with_fewer_FP=negative,
        n=int(known.sum()),positive=cover["positive"],events=cover["true_events"],recovery_negatives=cover["high_risk_recovered"],
        heldout=cfg["holdout"],evaluation_seconds=time.perf_counter()-start,finished_utc=now())
    save(ev/"status.json",result)
    print(result,flush=True)


if __name__=="__main__":
    evaluate(*settings())

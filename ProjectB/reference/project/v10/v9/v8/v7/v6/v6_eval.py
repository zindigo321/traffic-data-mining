"""One independent evaluation of the frozen V6 model family."""
import sys
import time

import numpy as np
import pandas as pd

from v6_io import V5,load,now,save,settings,sha
from v6_lock import check_transport,verify_lock
from v6_register import verify_registration
from v6_stats import compare,publish
sys.path.insert(0,str(V5))
from v5_prepare import fingerprint_keys,holdout
from v5_train import model_for,study_for,typed_score
from v5_eval import error_review
from v4_matrix import average
from v4_metrics import matched_points
from evaluation import measure
from experiment import predict
from v1.graph import TrainPreprocessor
from v3_train import risk_metrics


def evaluate(root,cfg):
    t0=time.perf_counter()
    if not cfg["formal_eligible"]:
        print("NYC formal evaluation blocked by coverage gate",flush=True)
        return
    if any((root/name).exists() for name in ("evidence/unlock.json","prepared/holdout.npz","results/holdout")):
        raise PermissionError("This holdout has already been opened")
    lock=verify_lock(root,cfg)
    receipt=check_transport(root,cfg,verify_registration(root,cfg))
    ev=root/"evidence"
    save(ev/"unlock.json",dict(unlocked_utc=now(),freeze_sha256=sha(ev/"freeze.json"),
        transport_sha256=sha(ev/"download_holdout_sealed.json"),interval=cfg["holdout"],shards=receipt["complete"],
        old_V5_holdout_parsed=False,prior_V6_holdout_parsed=False))
    data,keys,cover=holdout(root,cfg,lock)
    eligible=data["eligible"]
    known=eligible&(data["y"]>=0)
    if not (cover["positive"] and cover["negative"] and cover["high_risk_recovered"]):
        save(ev/"status.json",dict(status="blocked",NYC_formal="blocked",Chicago_formal="blocked",
             AP=None,F1=None,goal_reached=None,reason="New registered support lacks a class or recoveries"))
        return
    out=root/"results/holdout"
    out.mkdir()
    np.savez_compressed(out/"support.npz",sample_id=keys.ravel(),y=data["y"].ravel(),
        eligible=eligible.ravel(),risk=data["risk"].ravel(),eligible_ids=keys[eligible],known_ids=keys[known])
    gate=load(ev/"date_gate.json")
    observed_gate=sum(p["rows"] for day in gate["daily_counts"].values() for p in day.values())
    save(ev/"holdout_support.json",dict(expected=int(keys.size),eligible=int(eligible.sum()),known=int(known.sum()),
        all_sha256=fingerprint_keys(keys),eligible_sha256=fingerprint_keys(keys[eligible]),
        known_sha256=fingerprint_keys(keys[known]),prepared_sha256=sha(root/"prepared/holdout.npz"),
        interval=cfg["holdout"],raw_roster_count_in_metadata_gate=observed_gate,at_utc=now()))
    records,scores,points={}, {}, {}
    for spec in lock["models"]:
        began=time.perf_counter()
        job=spec["job"]
        source=root/"results/calibration"/job
        dest=out/job
        dest.mkdir()
        opt=load(source/"options.json")
        study=study_for(data,spec["recipe"])
        study.configure(TrainPreprocessor.load(source/"preprocessor.npz"))
        net=model_for(cfg,study,opt,source/"weight.pt")
        ids=np.flatnonzero(study.eligible)
        all_scores=typed_score(study,ids,*predict(net,study,ids,cfg))
        cutoff=spec["threshold"]
        all_scores.update(threshold=np.float64(cutoff),pred=all_scores["score"]>=cutoff)
        np.testing.assert_array_equal(all_scores["sample_id"],keys[eligible])
        np.savez_compressed(dest/"all_eligible.npz",**all_scores)
        z={k:v[all_scores["y"]>=0] for k,v in all_scores.items() if k!="threshold"}
        np.testing.assert_array_equal(z["sample_id"],keys[known])
        stats=measure(z["y"],z["score"],cutoff,z["sample_id"],cover)
        risk=risk_metrics(z,cutoff)
        fixed,_=matched_points(z["sample_id"],z["y"],z["score"],z["risk"],cfg["recalls"],cover["total_days"])
        main=next(row for row in fixed if row["target_recall"]==cfg["main_recall"])
        cal=load(source/"status.json")
        keyspec={k:spec[k] for k in ("job","recipe","pool","seed","primary")}
        result=dict(**keyspec,**stats,**risk,FP60=main["FP"],TP60=main["TP"],FP60_per_day=main["FP_per_day"],
            fit_seconds=cal["fit_seconds"],peak_RSS_bytes=cal["peak_RSS_bytes"],eval_seconds=time.perf_counter()-began,
            weight_sha256=spec["hashes"]["weight.pt"])
        save(dest/"status.json",result)
        save(dest/"matched_recall.json",fixed)
        key=(spec["recipe"],spec["pool"],spec["seed"])
        records[key],scores[key],points[key]=result,z,fixed
        print({k:result[k] for k in ("job","AP","f1","FP60_per_day","tp","fp","fn")},flush=True)
    rows=list(records.values())
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False,na_rep="N/A")
    pd.DataFrame([dict(recipe=key[0],pool=key[1],seed=key[2],**row)
                  for key,values in points.items() for row in values]).to_csv(out/"matched_recall.csv",index=False)
    aggregates={f"{r}_{p}":average([row for row in rows if (row["recipe"],row["pool"])==(r,p)])
                for r,p in dict.fromkeys((x["recipe"],x["pool"]) for x in rows)}
    save(out/"aggregates.json",aggregates)
    changes=compare(cfg,records,scores,points)
    publish(out,changes)
    error_review(out,rows,scores,data,keys,cfg,cover)
    primary="_".join(lock["primary"])
    formal=aggregates[primary]
    result=dict(status="complete",NYC_formal="complete",Chicago_formal="blocked",Geolife_formal="blocked",
        primary=primary,AP=formal["AP"],F1=formal["f1"],goal=cfg["goal"],
        goal_reached=formal["AP"]>=cfg["goal"] and formal["f1"]>=cfg["goal"],
        stable_vs_both=all(x["stable"] for x in changes[2].values()),comparisons=changes[2],
        n=int(known.sum()),positive=cover["positive"],recoveries=cover["high_risk_recovered"],events=cover["true_events"],
        holdout=cfg["holdout"],V5_holdout_used=False,evaluation_seconds=time.perf_counter()-t0,finished_utc=now())
    save(ev/"status.json",result)
    print(result,flush=True)


if __name__=="__main__":
    evaluate(*settings())

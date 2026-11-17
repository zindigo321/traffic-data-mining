"""Rebuild current-grain truth and construct independently truncated lead features."""
import sys

import numpy as np
import pandas as pd

from v7_io import V6,load,now,save,settings,sha
from v7_check import registered
sys.path.insert(0,str(V6/"v5"))
from v5_prepare import (build,grid,fold,local_dates,times_for,minute_values,read_npz,aligned_lag,take_lag,
                        causal_history,truth_only,inputs_only,historical_transitions,fingerprint_keys,key_matrix)
from v1.prepare_chicago import build_features
from prepare_chicago import REASONS

FEATURES=[
    "lag20_mph","lag30_mph","lag40_mph","lag20_bus_count","lag20_message_count","legal_speed_std",
    "lag20_minus_lag30_mph","length_miles","direction_east","direction_north","cutoff_sin","cutoff_cos","weekend",
]+[f"lag{lag}_{name}" for lag in (20,30,40) for name in ("relative_speed","deviation","history_days")]+[
    "lag30_bus_count","lag30_message_count","lag40_bus_count","lag40_message_count",
    "lag50_ratio","lag60_ratio","lag70_ratio","legal_usable_count","legal_low_count","legal_low_depth",
    "legal_mean_ratio","legal_ratio_range","lag20_minus_lag30_ratio","lag20_minus_lag40_ratio",
    "past_road_continuation_rate","past_road_daytype_peak_rate","past_road_support","past_road_daytype_peak_support",
]+[f"lag{lag}_{name}" for lag in (20,30) for name in ("half_first_ratio","half_last_ratio","half_change",
    "half_first_log_reports","half_last_log_reports")]+["lag20_minus_lag30_last_half","lag20_last_half_low"]


def ratio(x,history):
    return np.divide(x,history,out=np.full_like(x,np.nan,dtype=float),where=history>0)


def forecast_inputs(raw,history,support,phase,roads,prior,cfg):
    # Each lag addresses a bin ending at or before decision t-10.
    times=raw["times"]
    lag=[aligned_lag(times,phase,10*j) for j in range(2,8)]
    base,_=build_features(raw["speed"],raw["bus"],raw["message"],times,lag[:3],roads,-cfg["lead_minutes"])
    rel=[]
    extended=[]
    for i,ix in enumerate(lag):
        speed,h=take_lag(raw["speed"],ix),take_lag(history,ix)
        value=ratio(speed,h)
        rel.append(value)
        if i<3:
            extended.extend((value,np.where(h>0,speed-cfg["ratio"]*h,np.nan),take_lag(support.astype(float),ix)))
    for ix in lag[1:3]:
        extended.extend((take_lag(raw["bus"],ix),take_lag(raw["message"],ix)))
    values=np.stack(rel,axis=-1)
    valid=np.isfinite(values)
    low=valid&(values<cfg["ratio"])
    n=valid.sum(axis=-1)
    lo=np.where(valid,values,np.inf).min(axis=-1)
    hi=np.where(valid,values,-np.inf).max(axis=-1)
    span=np.subtract(hi,lo,out=np.full_like(lo,np.nan),where=n>1)
    mean=np.divide(np.nansum(values,axis=-1),n,out=np.full_like(lo,np.nan),where=n>0)
    depth=np.cumprod(low,axis=-1).sum(axis=-1)
    tail=[*rel[3:],n,low.sum(axis=-1),depth,mean,span,rel[0]-rel[1],rel[0]-rel[2]]
    half,last=[],[]
    for ix in lag[:2]:
        h=take_lag(history,ix)
        a,b=[ratio(take_lag(raw["half"][:,:,k],ix),h) for k in (0,1)]
        counts=[np.log1p(take_lag(raw["half_count"][:,:,k].astype(float),ix)) for k in (0,1)]
        half.extend([a,b,b-a,*counts])
        last.append(b)
    half.extend([last[0]-last[1],np.where(np.isfinite(last[0]),last[0]<cfg["ratio"],np.nan)])
    x=np.concatenate([base,np.stack(extended+tail,axis=-1),prior,np.stack(half,axis=-1)],axis=-1).astype(np.float32)
    if x.shape[-1]!=len(FEATURES) or len(FEATURES)!=52:
        raise ValueError("Invalid feature layout")
    return x,rel[0]


def cohort(data,reason,calendar,cfg):
    y=data["y"]
    keys=key_matrix(data["minute"],data["road_ids"])
    rows=[]
    for part in (0,1):
        phase=data["split"]==part
        z=y[phase]
        risk=data["risk"][phase]
        available=data["eligible"][phase]
        shared=data["current_eligible"][phase]
        known=available&(z>=0)
        common=shared&(z>=0)
        rows.append(dict(part="train" if part==0 else "assessment",expected=int(z.size),eligible=int(available.sum()),
            positive=int((known&(z==1)).sum()),negative=int((known&(z==0)).sum()),
            unknown_eligible=int((available&(z<0)).sum()),high_risk_recovered=int((known&risk&(z==0)).sum()),
            total_days=int(phase.sum()//24),known_coverage=float(known.mean()),known_roads=int(known.any(axis=0).sum()),
            current_known=int(common.sum()),current_unknown=int((shared&(z<0)).sum()),
            known_keys_sha256=fingerprint_keys(keys[phase][known]),common_keys_sha256=fingerprint_keys(keys[phase][common]),
            reasons={name:int((phase_reason==code).sum()) for phase_reason in [reason[phase]] for code,name in REASONS.items()},
            legal_lag2_missing=int((~np.isfinite(data["ratio_prior"][phase])).sum())))
    return dict(assessment_interval=calendar,parts=rows,feature_names=FEATURES,lead_minutes=cfg["lead_minutes"],
                truth_inherited=True,all_scheduled_keys_scored=True)


def create_lead(root,cfg,name,full,table,graph,period):
    train=local_dates([[cfg["train_start"],period[0]]])
    assessment=local_dates([period])
    if train[-1]>=assessment[0] or assessment[-1]>=np.datetime64(cfg["embargo"][0]):
        raise ValueError("Leaking or nonforward split")
    dates=np.r_[train,assessment]
    times=times_for(dates,cfg)
    i=full["times"].get_indexer(times)
    if (i<0).any():
        raise ValueError("Prepared grid incomplete")
    raw={k:full[k][i] for k in full if k!="times"}
    raw["times"]=times
    day_phase=np.r_[np.zeros(len(train),dtype=np.int8),np.ones(len(assessment),dtype=np.int8)]
    phase=np.repeat(day_phase,24)
    weekends=np.asarray(pd.to_datetime(dates).dayofweek>=5)
    cube=raw["speed"].reshape(len(dates),24,len(table))
    history,support=causal_history(cube,weekends,day_phase,cfg["minimum_history_days"])
    current=inputs_only(raw,history,support,phase,table,cfg)
    y,reason,_=truth_only(raw,history,current["lag"],cfg)
    # This state uses completed earlier training days only and freezes in assessment.
    prior,_=historical_transitions(y,current["risk"],current["eligible"],day_phase,weekends)
    x,rel=forecast_inputs(raw,history,support,phase,table,prior,cfg)
    a=read_npz(root/"prepared"/f"{name}.npz")
    for left,right in ((a["y"],y),(a["eligible"],current["eligible"]),(a["risk"],current["risk"])):
        np.testing.assert_array_equal(left,right)
    data=dict(x=x,y=y,aux=np.full_like(y,-1),ratio_prior=rel,eligible=np.ones_like(y,dtype=bool),
              risk=current["risk"],current_eligible=current["eligible"],split=phase,minute=minute_values(times),**graph)
    out=root/"prepared"/f"lead_{name}.npz"
    np.savez_compressed(out,**data)
    summary=cohort(data,reason,period,cfg)
    save(root/"evidence"/f"fold_lead_{name}.json",summary)
    print(dict(fold=name,assess=summary["parts"][1],sha256=sha(out)),flush=True)
    return sha(out)


def prepare(root,cfg):
    registered(root)
    if (root/"evidence/calibration_freeze.json").exists():
        raise PermissionError("This run is frozen")
    old=load(root/"evidence/prior_prepared.json")
    table,graph=build(root,cfg)
    compare=load(root/"evidence/prior_graph.json")
    actual=load(root/"evidence/graph.json")
    for key in ("adjacency_sha256","roster_sha256","edge_sha256"):
        assert actual[key]==compare[key]
    interval=[cfg["train_start"],cfg["calibration"][1]]
    full,funnel=grid(root,cfg,"development",table,interval)
    phases={**cfg["folds"],"final":cfg["calibration"]}
    cur={name:fold(root,cfg,name,full,table,graph,period,funnel) for name,period in phases.items()}
    assert cur==old["hashes"] and sha(root/"prepared/frozen_history.npz")==old["history_sha256"]
    lead={name:create_lead(root,cfg,name,full,table,graph,period) for name,period in phases.items()}
    receipt=dict(status="PASS",utc=now(),hashes=cur,lead_hashes=lead,history_sha256=old["history_sha256"],
        old_A_byte_identical=True,A_B_truth_identical=True,holdout_parsed=False,embargo=cfg["embargo"],
        source_funnel=funnel,lead_feature_names=FEATURES,latest_dynamic_lag_minutes=20)
    save(root/"evidence/prepared_development.json",receipt)
    print({k:v for k,v in receipt.items() if k not in ("source_funnel","lead_feature_names")},flush=True)


if __name__=="__main__":
    prepare(*settings())


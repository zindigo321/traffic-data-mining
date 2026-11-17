"""Rebuild the unchanged V2 proxy and add lag-only recovery features."""
from pathlib import Path
import sys

from v3_io import CODE, context, load, save, sha, now
sys.path.insert(0,str(CODE/"v2"))
from v1 import runtime

import numpy as np
import pandas as pd
from prepare_chicago import (
    BASE_FEATURES, FEATURE_NAMES, aligned_lag, causal_history, check_days, fingerprint_keys,
    frozen_history, grid_from_raw, inputs_only, key_matrix, local_dates, minute_values, read_extension,
    read_npz, summarize, take_lag, times_for, truth_only,
)

EXTRA = [
    "lag40_ratio", "lag50_ratio", "lag60_ratio", "six_lag_usable_count", "six_lag_low_count",
    "contiguous_prior_low_depth", "six_lag_mean_ratio", "six_lag_range_ratio",
    "lag1_minus_lag2_ratio", "lag1_minus_lag3_ratio",
    "prior_road_continuation_rate", "prior_daytype_peak_road_continuation_rate",
    "log1p_prior_road_risk_count", "log1p_prior_daytype_peak_risk_count",
]


def history_raw(root,cfg):
    ev=load(root/"evidence/prior_inputs.json")
    path=Path(ev["legacy_csv"])
    if sha(path)!=ev["legacy_sha256"]:
        raise ValueError("Legacy CSV changed")
    # Load only the source's pre-V1-test rows into this analysis.
    base=pd.read_csv(path)
    base["time"]=pd.to_datetime(base["time"],format="ISO8601")
    start,end=map(pd.Timestamp,cfg["history_intervals"][0])
    base=base.loc[(base["time"]>=start)&(base["time"]<end)].copy()
    chunks=[base]
    for row in ev["historical_development"]:
        source=Path(row["path"])
        if sha(source)!=row["sha256"]:
            raise ValueError("Historical shard changed")
        frame=pd.read_csv(source)
        if len(frame)!=row["rows"]:
            raise ValueError("Historical row count differs")
        frame["time"]=pd.to_datetime(frame["time"],format="ISO8601")
        if not ((frame["time"]>=pd.Timestamp(row["start"]))&
                (frame["time"]<pd.Timestamp(row["stop"]))).all():
            raise ValueError("Historical source record is out of its interval")
        chunks.append(frame)
    return pd.concat(chunks,ignore_index=True)


def roads(root):
    table=pd.read_csv(root/"evidence/roads.csv",float_precision="round_trip",keep_default_na=False).set_index("segment_id")
    graph=read_npz(root/"prepared/topology.npz")
    np.testing.assert_array_equal(table.index.to_numpy(),graph["road_ids"])
    return table,graph


def temporal_extras(raw,hist,phase,ratio):
    times=raw["times"]
    ratios=[]
    streak=np.zeros(raw["speed"].shape,dtype=np.uint8)
    keep=np.ones_like(streak,dtype=bool)
    count=np.zeros_like(streak)
    low_count=np.zeros_like(streak)
    total=np.zeros_like(raw["speed"],dtype=np.float32)
    lo=np.full_like(total,np.inf)
    hi=np.full_like(total,-np.inf)
    for j in range(1,7):
        ix=aligned_lag(times,phase,10*j)
        speed,base=take_lag(raw["speed"],ix),take_lag(hist,ix)
        value=np.divide(speed,base,out=np.full_like(speed,np.nan),where=base>0)
        valid=np.isfinite(value)
        low=(value<ratio)&valid
        keep &= low
        streak += keep
        count += valid
        low_count += low
        v=value.astype(np.float32)
        ratios.append(v)
        total += np.where(valid,v,0)
        np.minimum(lo,np.where(valid,v,np.inf),out=lo)
        np.maximum(hi,np.where(valid,v,-np.inf),out=hi)
    mean=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
    span=np.full_like(mean,np.nan)
    np.subtract(hi,lo,out=span,where=count>1)
    return [*ratios[3:],count,low_count,streak,mean,span,ratios[0]-ratios[1],ratios[0]-ratios[2]]


def initial_state(nroads):
    return dict(
        road_n=np.zeros(nroads,dtype=np.int64),road_p=np.zeros(nroads,dtype=np.int64),
        group_n=np.zeros((2,2,nroads),dtype=np.int64),group_p=np.zeros((2,2,nroads),dtype=np.int64),
    )


def features_for_prior(state,weekend):
    nr,pr=state["road_n"],state["road_p"]
    ng,pg=state["group_n"][weekend],state["group_p"][weekend]
    global_rate=(pr.sum()+1)/(nr.sum()+5)
    road_rate=(pr+12*global_rate)/(nr+12)
    group_rate=(pg+8*road_rate)/(ng+8)
    return np.stack([np.broadcast_to(road_rate,ng.shape),group_rate,
                     np.broadcast_to(np.log1p(nr),ng.shape),np.log1p(ng)],axis=-1)


def historical_transitions(y,risk,eligible,day_phase,weekend):
    """Output the prior-day rate before updating on the day's qualifying risk examples."""
    d,r=len(day_phase),y.shape[1]
    valid=(risk&eligible&(y>=0)).reshape(d,2,12,r)
    yes=(risk&eligible&(y==1)).reshape(d,2,12,r)
    n,p=valid.sum(axis=2),yes.sum(axis=2)
    state=initial_state(r)
    out=np.empty((d,2,r,4),dtype=np.float32)
    for i,(part,daytype) in enumerate(zip(day_phase,weekend,strict=True)):
        out[i]=features_for_prior(state,int(daytype))
        if part==0:
            state["road_n"]+=n[i].sum(axis=0)
            state["road_p"]+=p[i].sum(axis=0)
            state["group_n"][int(daytype)]+=n[i]
            state["group_p"][int(daytype)]+=p[i]
    return np.repeat(out,12,axis=1).reshape(y.shape[0],r,4),state


def frozen_transitions(state,weekend):
    vals=np.stack([features_for_prior(state,i) for i in (0,1)]).astype(np.float32)
    sub=vals[weekend.astype(np.int8)]
    return np.repeat(sub,12,axis=1).reshape(len(weekend)*24,sub.shape[2],4)


def recovery_target(speed,hist,y):
    relative=np.divide(speed,hist,out=np.full_like(speed,np.nan),where=(hist>0)&(y>=0))
    return np.log1p(relative).astype(np.float32)


def write_summary(root,name,arr,reasons,names,funnel):
    path=root/"evidence"/f"fold_{name}.json"
    summarize(path,arr,reasons,names,funnel)
    stat=load(path)
    for part,row in zip(np.unique(arr["split"]),stat["parts"],strict=True):
        mask=arr["split"]==part
        ext=arr["x"][mask,:,len(FEATURE_NAMES):]
        row["extra_feature_missing"]={key:float((~np.isfinite(ext[...,i])).mean()) for i,key in enumerate(EXTRA)}
        known=arr["eligible"][mask]&(arr["y"][mask]>=0)
        if ((arr["y"][mask]==1)&known&~arr["risk"][mask]).any():
            raise ValueError("A positive violates the two-low route definition")
    save(path,stat)
    return stat


def fold(root,cfg,name,grid,table,graph,train,assess,source_funnel):
    check_days(cfg,train,assess)
    dates=np.concatenate([train,assess])
    ts=times_for(dates,cfg)
    index=grid["times"].get_indexer(ts)
    if (index<0).any():
        raise ValueError("Requested split is outside source grid")
    raw={key:grid[key][index] for key in ("speed","bus","message","observed")}
    raw["times"]=ts
    day_phase=np.r_[np.zeros(len(train),dtype=np.int8),np.ones(len(assess),dtype=np.int8)]
    phase=np.repeat(day_phase,24)
    weekend=np.asarray(pd.to_datetime(dates).dayofweek>=5)
    cube=raw["speed"].reshape(len(dates),24,len(table))
    history,n=causal_history(cube,weekend,day_phase,cfg["minimum_history_days"])
    base=inputs_only(raw,history,n,phase,table,cfg)
    y,reasons,aux=truth_only(raw,history,base.pop("lag"),cfg)
    prior,state=historical_transitions(y,base["risk"],base["eligible"],day_phase,weekend)
    ext=temporal_extras(raw,history,phase,cfg["ratio"])
    x=np.concatenate([base.pop("x"),np.stack(ext,axis=-1).astype(np.float32),prior],axis=-1)
    arr=dict(**base,x=x,y=y,aux=aux,recovery_target=recovery_target(raw["speed"],history,y),
             split=phase,minute=minute_values(ts),**graph)
    p=root/"prepared"/f"{name}.npz"
    np.savez_compressed(p,**arr)
    names={0:"train",1:"calibration" if name=="final" else "development"}
    write_summary(root,name,arr,reasons,names,source_funnel)
    if name=="final":
        np.savez_compressed(root/"prepared/frozen_history.npz",**frozen_history(cube,weekend,day_phase,cfg),**state)
    return sha(p)


def make_holdout(root,cfg,contract):
    if not (root/"evidence/holdout_unlock.json").is_file():
        raise PermissionError("No freeze-verified unlock")
    table,graph=roads(root)
    dates=local_dates([cfg["holdout"]])
    check_days(cfg,np.empty(0,dtype="datetime64[D]"),dates)
    raw,funnel=grid_from_raw(read_extension(root,"holdout_sealed"),dates,table,cfg)
    shape=raw["speed"].shape
    phase=np.ones(shape[0],dtype=np.int8)
    weekend=np.asarray(pd.to_datetime(dates).dayofweek>=5,dtype=np.int8)
    state=read_npz(root/"prepared/frozen_history.npz")
    h=state["reference"][weekend].reshape(shape)
    n=state["days"][weekend].reshape(shape)
    base=inputs_only(raw,h,n,phase,table,cfg)
    ext=temporal_extras(raw,h,phase,cfg["ratio"])
    prior=frozen_transitions(state,weekend)
    x=np.concatenate([base.pop("x"),np.stack(ext,axis=-1).astype(np.float32),prior],axis=-1)
    minutes=minute_values(raw["times"])
    keys=key_matrix(minutes,graph["road_ids"])
    if fingerprint_keys(keys)!=contract["expected_keys_sha256"]:
        raise ValueError("Holdout grid differs from frozen contract")
    save(root/"evidence/holdout_features.json",dict(at_utc=now(),known_truth_used=False,
         eligible_count=int(base["eligible"].sum()),eligible_sha256=fingerprint_keys(keys[base["eligible"]]),
         all_keys_sha256=fingerprint_keys(keys)))
    y,reasons,aux=truth_only(raw,h,base.pop("lag"),cfg)
    arr=dict(**base,x=x,y=y,aux=aux,split=phase,minute=minutes,**graph)
    p=root/"prepared/holdout.npz"
    np.savez_compressed(p,**arr)
    cover=write_summary(root,"holdout",arr,reasons,{1:"holdout"},funnel)["parts"][0]
    return arr,keys,cover,sha(p)


def main():
    root,cfg=context()
    receipt=load(root/"evidence/download_development.json")
    if receipt["status"]!="complete":
        save(root/"evidence/blocked.json",dict(stage="prepare",reason="Fixed new development/calibration download incomplete"))
        return
    table,graph=roads(root)
    sources=pd.concat([history_raw(root,cfg),read_extension(root,"development")],ignore_index=True)
    dates=local_dates(cfg["history_intervals"]+[cfg["development_download"]])
    grid,funnel=grid_from_raw(sources,dates,table,cfg)
    del sources
    result={}
    for name,opt in cfg["folds"].items():
        result[name]=fold(root,cfg,name,grid,table,graph,local_dates(opt["train"]),local_dates([opt["valid"]]),funnel)
    result["final"]=fold(root,cfg,"final",grid,table,graph,local_dates(cfg["final_train"]),local_dates([cfg["calibration"]]),funnel)
    save(root/"evidence/prepared_development.json",dict(hashes=result,features_v2=FEATURE_NAMES,features_added=EXTRA,
         source=funnel,excluded=cfg["exclusions"],v2_holdout_used=False))


if __name__=="__main__":
    main()


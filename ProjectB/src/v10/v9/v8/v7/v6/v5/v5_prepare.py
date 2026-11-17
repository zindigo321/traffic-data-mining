"""Construct disjoint local-time NY data arrays and lag-only half-grid inputs."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from v5_io import V4, load, now, save, settings, sha
from v5_download import source_files, verify_registration
from v5_graph import build

sys.path.insert(0,str(V4))
# v4_train sets the published V3/v2 module paths.
from v4_train import read_npz
from v3_prepare import EXTRA, frozen_transitions, historical_transitions, temporal_extras
from prepare_chicago import (
    FEATURE_NAMES, aligned_lag, causal_history, fingerprint_keys, frozen_history, inputs_only,
    key_matrix, local_dates, minute_values, summarize, take_lag, times_for, truth_only,
)

HALF_NAMES=[
    f"lag{10*j}_{field}" for j in (1,2)
    for field in ("half_first_ratio","half_last_ratio","half_ratio_change","half_first_log_reports","half_last_log_reports")
]+["lag10_last_half_minus_lag20_last_half","lag10_last_half_low"]
BASIC_FEATURES=FEATURE_NAMES+EXTRA


def verify_amendment(root):
    ev=load(root/"evidence/amendment.json")
    if (ev["sha256"]!=sha(root/"amendment_nyc.md") or ev["observation_values_read"] is not False
            or ev["status_filters_speed"] is not False or ev["dictionary_sha256"]!=load(root/"evidence/dictionary_receipt.json")["sha256"]):
        raise ValueError("Status amendment does not match official dictionary")
    return ev


def grid(root,cfg,name,table,interval):
    pieces=[]
    for path,receipt in source_files(root,name):
        frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        if len(frame)!=receipt["rows"]:
            raise ValueError("Source CSV row count differs")
        times=pd.to_datetime(frame["data_as_of"],format="ISO8601")
        if times.dt.tz is not None or times.isna().any():
            raise ValueError("NY source must be a floating local time")
        start,stop,peak=path.stem.split("_")
        if not ((times>=pd.Timestamp(start))&(times<pd.Timestamp(stop))&times.dt.hour.isin(
                (7,8) if peak=="am" else (17,18))).all():
            raise ValueError(f"Observation outside shard calendar: {path}")
        frame["stamp"]=times
        pieces.append(frame)
    frame=pd.concat(pieces,ignore_index=True)
    dates=local_dates([interval])
    times=times_for(dates,cfg)
    index=pd.Index(table["link_id"])
    codes=index.get_indexer(frame["link_id"])
    registered=codes>=0
    frame["road"]=codes
    values=pd.to_numeric(frame["speed"],errors="coerce")
    duration=pd.to_numeric(frame["travel_time"],errors="coerce")
    quality=values.gt(0)&values.le(cfg["source"]["max_speed_mph"])&duration.gt(0)&np.isfinite(duration)
    known=quality&registered
    implied=(values[known]*duration[known]/3600).to_numpy(dtype=float)
    known_length=table["length"].to_numpy()[codes[known]]
    implied_ratio=implied/known_length
    status_counts=frame["status"].value_counts().to_dict()
    good_status=frame.loc[known,"status"].value_counts().to_dict()
    funnel=dict(rows=len(frame),outside_pinned_roster=int((~registered).sum()),invalid_in_roster=int((registered&~quality).sum()),
                valid_rows=int(known.sum()),status_counts=status_counts,valid_rows_by_status=good_status,
                status_filters_speed=False,implied_miles_over_polyline_miles_p10_median_p90=np.quantile(implied_ratio,[.1,.5,.9]).tolist()
                if len(implied_ratio) else None)
    full=frame.loc[registered,["stamp","road"]].copy()
    full["ten"]=full["stamp"].dt.floor("10min")
    ix=times.get_indexer(full["ten"])
    if (ix<0).any():
        raise ValueError("Registered source row outside phase time grid")
    shape=(len(times),len(table))
    observed=np.zeros(shape,dtype=bool)
    observed[ix,full["road"].to_numpy()]=True
    frame=frame.loc[known,["stamp","road"]].assign(speed=values[known]).copy()
    exact=frame.groupby(["stamp","road"],as_index=False)["speed"].median()
    exact["ten"]=exact["stamp"].dt.floor("10min")
    exact["half"]=(exact["stamp"].dt.minute%10>=5).astype(np.int8)
    groups=exact.groupby(["ten","road","half"],as_index=False)["speed"].agg(["median","count"]).reset_index(drop=True)
    shape3=(*shape,2)
    half=np.full(shape3,np.nan,dtype=np.float64)
    count=np.zeros(shape3,dtype=np.int32)
    ti=times.get_indexer(groups["ten"])
    if (ti<0).any():
        raise ValueError("Half bins out of phase")
    rr,hh=groups["road"].to_numpy(),groups["half"].to_numpy()
    half[ti,rr,hh]=groups["median"].to_numpy()
    count[ti,rr,hh]=groups["count"].to_numpy()
    speed=(half[:,:,0]+half[:,:,1])/2
    structural=np.full(shape,np.nan,dtype=np.float64)
    funnel.update(duplicate_valid_timestamps=len(frame)-len(exact),usable=int(np.isfinite(speed).sum()),
                  observed=int(observed.sum()),expected=int(observed.size),
                  first_half_only=int(((count[:,:,0]>0)&(count[:,:,1]==0)).sum()),
                  last_half_only=int(((count[:,:,1]>0)&(count[:,:,0]==0)).sum()),
                  source_days=int(full["stamp"].dt.normalize().nunique()),
                  registered_days=len(dates),zero_observation_days=[str(d) for d in dates
                      if not np.any(observed[np.asarray(times.normalize()==pd.Timestamp(d))])])
    return dict(speed=speed,bus=structural,message=structural.copy(),observed=observed,
                half=half,half_count=count,times=times),funnel


def half_features(raw,history,phase,ratio):
    result=[]
    last=[]
    for j in (1,2):
        lag=aligned_lag(raw["times"],phase,10*j)
        base=take_lag(history,lag)
        a,b=(take_lag(raw["half"][:,:,k],lag) for k in (0,1))
        relative_a=np.divide(a,base,out=np.full_like(a,np.nan),where=base>0)
        relative_b=np.divide(b,base,out=np.full_like(b,np.nan),where=base>0)
        counts=[np.log1p(take_lag(raw["half_count"][:,:,k].astype(float),lag)) for k in (0,1)]
        result.extend([relative_a,relative_b,relative_b-relative_a,*counts])
        last.append(relative_b)
    result.extend([last[0]-last[1],np.where(np.isfinite(last[0]),last[0]<ratio,np.nan)])
    return np.stack(result,axis=-1).astype(np.float32)


def ancillary(raw,history,phase,cfg):
    ext=np.stack(temporal_extras(raw,history,phase,cfg["ratio"]),axis=-1).astype(np.float32)
    return ext,half_features(raw,history,phase,cfg["ratio"])


def target(raw,history,y):
    valid=(raw["speed"]>0)&(history>0)&(y>=0)
    relative=np.divide(raw["speed"],history,out=np.full_like(history,np.nan),where=valid)
    return np.log(relative).astype(np.float32)


def summary(root,name,data,reasons,roles,funnel):
    path=root/"evidence"/f"fold_{name}.json"
    summarize(path,data,reasons,roles,funnel)
    result=load(path)
    for code,row in zip(np.unique(data["split"]),result["parts"],strict=True):
        x=data["x"][data["split"]==code]
        row["extra_missing"]={field:float((~np.isfinite(x[...,k])).mean())
                              for k,field in enumerate(BASIC_FEATURES+HALF_NAMES) if k>=len(FEATURE_NAMES)}
    save(path,result)
    return result


def fold(root,cfg,name,dev,table,graph,assessment,funnel):
    train=local_dates([[cfg["train_start"],assessment[0]]])
    test=local_dates([assessment])
    if len(train)==0 or train[-1]>=test[0] or test[-1]>=np.datetime64(cfg["holdout"][0]):
        raise ValueError("Overlapping or nonforward training/evaluation")
    dates=np.r_[train,test]
    times=times_for(dates,cfg)
    at=dev["times"].get_indexer(times)
    if (at<0).any():
        raise ValueError("Evaluation outside permitted development data")
    raw={k:dev[k][at] for k in dev if k!="times"}
    raw["times"]=times
    dphase=np.r_[np.zeros(len(train),dtype=np.int8),np.ones(len(test),dtype=np.int8)]
    phase=np.repeat(dphase,24)
    weekday=np.asarray(pd.to_datetime(dates).dayofweek>=5)
    cube=raw["speed"].reshape(len(dates),24,len(table))
    hist,n=causal_history(cube,weekday,dphase,cfg["minimum_history_days"])
    base=inputs_only(raw,hist,n,phase,table,cfg)
    y,reasons,aux=truth_only(raw,hist,base.pop("lag"),cfg)
    prior,state=historical_transitions(y,base["risk"],base["eligible"],dphase,weekday)
    ext,new=ancillary(raw,hist,phase,cfg)
    x=np.concatenate([base.pop("x"),ext,prior,new],axis=-1)
    data=dict(**base,x=x,y=y,aux=aux,recovery_target=target(raw,hist,y),split=phase,
              minute=minute_values(times),**graph)
    output=root/"prepared"/f"{name}.npz"
    np.savez_compressed(output,**data)
    stat=summary(root,name,data,reasons,{0:"train",1:"calibration" if name=="final" else "development"},funnel)
    if name=="final":
        np.savez_compressed(root/"prepared/frozen_history.npz",**frozen_history(cube,weekday,dphase,cfg),**state)
    print(dict(fold=name,counts=[{key:s[key] for key in ("part","expected","eligible","positive","negative","true_events","high_risk_recovered")}
                                 for s in stat["parts"]],sha256=sha(output)),flush=True)
    return sha(output)


def prepare(root,cfg):
    registration=verify_registration(root,cfg)
    amendment=verify_amendment(root)
    if (root/"evidence/unlock.json").exists():
        raise PermissionError("Development cannot be rebuilt within the unlocked run")
    table,graph=build(root,cfg)
    interval=[cfg["train_start"],cfg["calibration"][1]]
    data,funnel=grid(root,cfg,"development",table,interval)
    hashes={name:fold(root,cfg,name,data,table,graph,dates,funnel)
            for name,dates in {**cfg["folds"],"final":cfg["calibration"]}.items()}
    receipt=dict(prepared_utc=now(),registered_utc=registration["at_utc"],amendment_utc=amendment["at_utc"],
                 hashes=hashes,history_sha256=sha(root/"prepared/frozen_history.npz"),source_funnel=funnel,
                 features_v4_analog=BASIC_FEATURES,features_added=HALF_NAMES,holdout_parsed=False,
                 time_zone=cfg["zone"],support="complete known eligible, identical per fold")
    save(root/"evidence/prepared_development.json",receipt)


def holdout(root,cfg,lock):
    table=pd.read_csv(root/"evidence/nyc_roads.csv",dtype={"link_id":str},index_col="road_code",float_precision="round_trip",keep_default_na=False)
    graph=read_npz(root/"prepared/topology.npz")
    np.testing.assert_array_equal(table.index.to_numpy(),graph["road_ids"])
    state=read_npz(root/"prepared/frozen_history.npz")
    raw,funnel=grid(root,cfg,"holdout_sealed",table,cfg["holdout"])
    d=local_dates([cfg["holdout"]])
    weekend=np.asarray(pd.to_datetime(d).dayofweek>=5,dtype=np.int8)
    shape=raw["speed"].shape
    h=state["reference"][weekend].reshape(shape)
    n=state["days"][weekend].reshape(shape)
    phase=np.ones(shape[0],dtype=np.int8)
    base=inputs_only(raw,h,n,phase,table,cfg)
    ext,new=ancillary(raw,h,phase,cfg)
    prior=frozen_transitions(state,weekend)
    minutes=minute_values(raw["times"])
    keys=key_matrix(minutes,graph["road_ids"])
    if fingerprint_keys(keys)!=lock["expected_grid_sha256"]:
        raise ValueError("Holdout differs from fixed calendar and network")
    save(root/"evidence/holdout_features.json",dict(at_utc=now(),eligible=int(base["eligible"].sum()),
         eligible_sha256=fingerprint_keys(keys[base["eligible"]]),expected_grid_sha256=fingerprint_keys(keys),
         target_truth_built=False))
    y,reasons,aux=truth_only(raw,h,base.pop("lag"),cfg)
    x=np.concatenate([base.pop("x"),ext,prior,new],axis=-1)
    data=dict(**base,x=x,y=y,aux=aux,split=phase,minute=minutes,**graph)
    np.savez_compressed(root/"prepared/holdout.npz",**data)
    support=summary(root,"holdout",data,reasons,{1:"holdout"},funnel)
    return data,keys,support["parts"][0]


if __name__=="__main__":
    prepare(*settings())

"""Build the complete chronological NYC grid using the original label and lead contracts."""
import sys

import numpy as np
import pandas as pd

from full_check import register
from full_io import V10,load,now,save,settings,sha

sys.path.insert(0,str(V10))
from v10_data import study_for
import v5_prepare
from v7_prepare import (FEATURES,forecast_inputs,causal_history,inputs_only,truth_only,historical_transitions,
                         local_dates,times_for,minute_values,key_matrix,fingerprint_keys)
from v5_graph import build
from prepare_chicago import REASONS
from evaluation import spans


def sources(root,name):
    assert name=="all_available"
    snapshot=load(root/"evidence/source_snapshot.json")
    for row in snapshot["files"]:
        from pathlib import Path
        path=Path(row["path"])
        assert sha(path)==row["sha256"] and path.stat().st_size==row["bytes"]
        yield path,row


def construct(raw,table,graph,cfg,train_end,cal_end,stop):
    times=raw["times"]
    select=times<pd.Timestamp(stop)
    raw={k:v[select] for k,v in raw.items()}
    times=raw["times"]
    dates=local_dates([[cfg["corpus"][0],stop]])
    day_phase=np.where(dates<np.datetime64(train_end),0,
                       np.where(dates<np.datetime64(cal_end),1,2)).astype(np.int8)
    phase=np.repeat(day_phase,24)
    weekend=np.asarray(pd.to_datetime(dates).dayofweek>=5)
    cube=raw["speed"].reshape(len(dates),24,len(table))
    history,support=causal_history(cube,weekend,day_phase,cfg["minimum_history_days"])
    current=inputs_only(raw,history,support,phase,table,cfg)
    y,reasons,_=truth_only(raw,history,current["lag"],cfg)
    prior,state=historical_transitions(y,current["risk"],current["eligible"],day_phase,weekend)
    x,relative=forecast_inputs(raw,history,support,phase,table,prior,cfg)
    data=dict(x=x,y=y,label_reason=reasons,aux=np.full_like(y,-1),ratio_prior=relative,
              eligible=np.ones_like(y,dtype=bool),risk=current["risk"],current_eligible=current["eligible"],
              split=phase,minute=minute_values(times),**graph)
    return data


def stats(data,cfg,name):
    keys=key_matrix(data["minute"],data["road_ids"])
    names={0:"train",1:"development" if name=="development" else "calibration",2:"retrospective_test"}
    rows=[]
    for role in np.unique(data["split"]):
        m=data["split"]==role
        y=data["y"][m]
        keep=y>=0
        known_keys=keys[m][keep]
        pos=y==1
        r=data["risk"][m]
        z=data["label_reason"][m]
        minute=data["minute"][m]
        rows.append(dict(part=names[int(role)],phase=int(role),scheduled=int(y.size),known=int(keep.sum()),
            positive=int(pos.sum()),negative=int((y==0).sum()),unknown=int((y<0).sum()),known_fraction=float(keep.mean()),
            days=len(np.unique(minute//1440)),start=str(np.datetime64(int(minute.min()//1440),"D")),
            stop_exclusive=str(np.datetime64(int(minute.max()//1440+1),"D")),known_roads=int(keep.any(0).sum()),
            positive_events=sum(map(len,spans(keys[m][pos]).values())),recovery_negatives=int((r&(y==0)).sum()),
            legal_lag20_missing=int((~np.isfinite(data["ratio_prior"][m])).sum()),
            unknown_and_label_reasons={label:int((z==code).sum()) for code,label in REASONS.items()},
            known_key_sha256=fingerprint_keys(known_keys),all_key_sha256=fingerprint_keys(keys[m])))
    return dict(stage=name,parts=rows,feature_names=FEATURES,all_windows_scored=True,latest_dynamic_lag_minutes=20,
                no_minimum_coverage_or_days_gate=True,history_and_transition_updates_only_train=True)


def prepare(root,cfg):
    register(root,cfg)
    table,graph=build(root,cfg)
    expected=load(root/"evidence/original_graph.json")
    actual=load(root/"evidence/graph.json")
    for key in ("roster_sha256","adjacency_sha256","edge_sha256"):
        assert actual[key]==expected[key]
    v5_prepare.source_files=sources
    raw,funnel=v5_prepare.grid(root,cfg,"all_available",table,cfg["corpus"])
    np.savez_compressed(root/"prepared/raw_grid.npz",**{k:v for k,v in raw.items() if k!="times"},
                        minute=minute_values(raw["times"]))
    hashes={}
    for name,train_end,cal_end,stop in (
        ("development",cfg["development_train"][1],cfg["development_assess"][1],cfg["development_assess"][1]),
        ("final",cfg["full_train"][1],cfg["full_calibration"][1],cfg["corpus"][1])):
        data=construct(raw,table,graph,cfg,train_end,cal_end,stop)
        path=root/"prepared"/(name+".npz")
        np.savez_compressed(path,**data)
        audit=stats(data,cfg,name)
        save(root/"evidence"/f"cohort_{name}.json",audit)
        hashes[name]=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)
        print(dict(stage=name,parts=[{k:r[k] for k in ("part","scheduled","known","positive","unknown","days")} for r in audit["parts"]]),flush=True)
    save(root/"evidence/prepared.json",dict(status="PASS",at_utc=now(),files=hashes,source_funnel=funnel,
         source_snapshot_sha256=sha(root/"evidence/source_snapshot.json"),raw_grid_sha256=sha(root/"prepared/raw_grid.npz"),
         training_full_known=True,new_legacy_holdouts_included=True,evaluation_protocol="chronological retrospective",
         latest_dynamic_lag_minutes=20,unknown_retained=True))
    print(dict(status="PASS",hashes={k:v["sha256"] for k,v in hashes.items()},source_rows=funnel["rows"]),flush=True)


if __name__=="__main__":
    prepare(*settings())

"""Rebuild V5's allowed development geometry, labels, and frozen history."""
import sys

from v6_io import V5,load,now,save,settings,sha
from v6_register import verify_registration
sys.path.insert(0,str(V5))
from v5_prepare import BASIC_FEATURES,HALF_NAMES,build,fold,grid


def prepare(root,cfg):
    reg=verify_registration(root,cfg)
    if (root/"evidence/unlock.json").exists() or (root/"evidence/freeze.json").exists():
        raise PermissionError("Cannot rebuild a frozen run")
    original=load(root/"evidence/v5_prepared_development.json")
    original_graph=load(root/"evidence/v5_graph.json")
    table,graph=build(root,cfg)
    generated=load(root/"evidence/graph.json")
    for name in ("adjacency_sha256","roster_sha256","edge_sha256","road_count","edges"):
        if original_graph[name]!=generated[name]:
            raise ValueError(f"Graph is not the fixed V5 graph: {name}")
    interval=[cfg["train_start"],cfg["calibration"][1]]
    if interval[1]>cfg["prior_holdout"][0]:
        raise ValueError("Prior holdout is forbidden in training/history")
    data,funnel=grid(root,cfg,"development",table,interval)
    hashes={name:fold(root,cfg,name,data,table,graph,period,funnel)
            for name,period in {**cfg["folds"],"final":cfg["calibration"]}.items()}
    history=sha(root/"prepared/frozen_history.npz")
    if hashes!=original["hashes"] or history!=original["history_sha256"]:
        raise ValueError("V6 rebuilt V5 development arrays are not byte-identical")
    entry=dict(prepared_utc=now(),registered_utc=reg["at_utc"],hashes=hashes,history_sha256=history,
        v5_development_identical=True,source_funnel=funnel,features_v4_analog=BASIC_FEATURES,features_added=HALF_NAMES,
        holdout_parsed=False,excluded_prior_holdout=cfg["prior_holdout"],prior_holdout_used=False,time_zone=cfg["zone"])
    save(root/"evidence/prepared_development.json",entry)
    print({k:v for k,v in entry.items() if k not in ("source_funnel","features_v4_analog","features_added")},flush=True)


if __name__=="__main__":
    prepare(*settings())

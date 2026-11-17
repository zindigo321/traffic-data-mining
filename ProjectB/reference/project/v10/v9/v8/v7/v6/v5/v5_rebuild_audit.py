"""Check that fixed raw snapshots rebuild every prepared array after evaluation."""
import argparse
from pathlib import Path

from v5_io import load,now,save,settings,sha
from v5_lock import verify_lock
from v5_prepare import holdout

def rebuild(artifact,root,cfg):
    lock=verify_lock(artifact,cfg)
    prior=load(artifact/"evidence/unlock.json")
    if prior["freeze_sha256"]!=sha(artifact/"evidence/freeze.json") or prior["unlocked_utc"]<=lock["frozen_utc"]:
        raise ValueError("Source evaluation never opened under the pinned contract")
    if root.resolve()==artifact.resolve() or (root/"evidence/unlock.json").exists():
        raise ValueError("Reconstruction requires independent scratch")
    receipts={}
    for name in ("geometry","development","holdout_sealed"):
        old=load(artifact/"evidence"/f"download_{name}.json")
        new=load(root/"evidence"/f"download_{name}.json")
        assert old["files"]==new["files"]
        receipts[name]=len(new["files"])
    source=load(artifact/"evidence/prepared_development.json")
    actual=load(root/"evidence/prepared_development.json")
    if actual["hashes"]!=source["hashes"] or actual["history_sha256"]!=source["history_sha256"]:
        raise ValueError("Prepared development reconstruction differs")
    assert sha(root/"prepared/topology.npz")==sha(artifact/"prepared/topology.npz")
    assert sha(root/"evidence/nyc_roads.csv")==sha(artifact/"evidence/nyc_roads.csv")
    save(root/"evidence/unlock.json",dict(at_utc=now(),source_unlock_sha256=sha(artifact/"evidence/unlock.json"),
        source_freeze_sha256=prior["freeze_sha256"],audit_reconstruction_only=True))
    holdout(root,cfg,lock)
    expected=load(artifact/"evidence/holdout_support.json")["prepared_sha256"]
    assert sha(root/"prepared/holdout.npz")==expected
    result=dict(status="PASS",files_verified=receipts,development=actual["hashes"],
                frozen_history=actual["history_sha256"],topology=sha(root/"prepared/topology.npz"),holdout=expected,
                rebuilt_utc=now(),audit_only=True)
    save(root/"evidence/reconstructed.json",result)
    print(result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    args=p.parse_args()
    rebuild(args.artifacts,*settings())

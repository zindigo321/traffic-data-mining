"""Rebuild V6 input arrays in independent audit scratch."""
import argparse
from pathlib import Path
import sys

from v6_io import V5,load,now,save,settings,sha
from v6_lock import verify_lock
sys.path.insert(0,str(V5))
from v5_prepare import holdout


def rebuild(artifact,root,cfg):
    if root.resolve()==artifact.resolve() or (root/"evidence/unlock.json").exists():
        raise ValueError("A separate, unopened audit scratch is required")
    original=load(artifact/"evidence/prepared_development.json")
    actual=load(root/"evidence/prepared_development.json")
    assert original["hashes"]==actual["hashes"] and original["history_sha256"]==actual["history_sha256"]
    assert sha(root/"prepared/topology.npz")==sha(artifact/"prepared/topology.npz")
    state=load(artifact/"evidence/status.json")
    files={}
    for role in ("geometry","development",*(["holdout_sealed"] if state["status"]=="complete" else [])):
        source=load(artifact/"evidence"/f"download_{role}.json")
        cached=load(root/"evidence"/f"download_{role}.json")
        assert source["files"]==cached["files"]
        files[role]=len(source["files"])
    new_digest=None
    if state["status"]=="complete":
        lock=verify_lock(artifact,cfg)
        opened=load(artifact/"evidence/unlock.json")
        assert opened["freeze_sha256"]==sha(artifact/"evidence/freeze.json")
        assert lock["frozen_utc"]<opened["unlocked_utc"] and not opened["old_V5_holdout_parsed"]
        save(root/"evidence/unlock.json",dict(at_utc=now(),source_freeze_sha256=opened["freeze_sha256"],
              source_unlock_sha256=sha(artifact/"evidence/unlock.json"),audit_only=True))
        holdout(root,cfg,lock)
        new_digest=sha(root/"prepared/holdout.npz")
        assert new_digest==load(artifact/"evidence/holdout_support.json")["prepared_sha256"]
    result=dict(status="PASS",files=files,development=actual["hashes"],frozen_history=actual["history_sha256"],
         graph=sha(root/"prepared/topology.npz"),new_holdout=new_digest,audit_only=True,rebuilt_utc=now())
    save(root/"evidence/reconstructed.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    args=p.parse_args()
    rebuild(args.artifacts,*settings())

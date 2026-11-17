"""Full-known supervised indices and immutable chronological experiment inputs."""
import sys
import os
from pathlib import Path

import numpy as np

from full_io import V10,load,sha
sys.path.insert(0,str(V10))
from v10_data import groups,study_for,build,predict,set_seed,read_npz,TrainPreprocessor
from v7_train import sample_training

MODELS=("graph_mean","graph_max","graph_attention","center_full","center_rank","center_sampled_matched")


def arrays(root,stage,alternate=None):
    if stage not in ("development","final"):
        raise ValueError("Unknown experiment stage")
    info=load(root/"evidence/prepared.json")["files"][stage]
    portable=os.environ.get("PROJECTB_PREPARED_DIR")
    # Explicit portable binding has priority. Never silently read a historic drive.
    p=(Path(portable)/(stage+".npz") if portable else
       alternate/"prepared"/(stage+".npz") if alternate else Path(info["path"]))
    assert sha(p)==info["sha256"]
    data=read_npz(p)
    assert data["eligible"].all() and data["x"].shape[-1]==52 and np.isin(data["y"],[-1,0,1]).all()
    return data


def training_ids(study,cfg,model):
    full=study.train.copy()
    if model=="center_sampled_matched":
        selected,_=sample_training(study,cfg)
    else:
        selected=full
    assert len(np.unique(selected))==len(selected) and np.isin(selected,full).all()
    return selected,len(full)


def epoch_order(selected,full_count,rng):
    return np.resize(rng.permutation(selected),full_count)


def folder(root,stage,model,seed):
    return root/"results/models"/stage/(model+(f"_s{seed}" if seed is not None else ""))


def central(model):
    return model.startswith("center") or model=="logistic"


def pool(model):
    return model.removeprefix("graph_") if model.startswith("graph_") else "attention"

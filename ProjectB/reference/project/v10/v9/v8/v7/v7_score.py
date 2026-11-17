"""Independent threshold rules and full-support metrics."""
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from v7_io import V6
sys.path.insert(0,str(V6/"v5"))
from v5_train import measure,threshold
from v4_metrics import matched_points


def balanced_cutoff(y,score):
    order=np.argsort(-score,kind="stable")
    p=score[order]
    z=y[order]
    npos,nneg=int(z.sum()),int((z==0).sum())
    if not npos or not nneg:
        raise ValueError("Both classes are required to calibrate")
    end=np.r_[p[1:]!=p[:-1],True]
    tp=(z==1).cumsum()[end]
    fp=(z==0).cumsum()[end]
    ba=np.r_[.5,.5*(tp/npos+1-fp/nneg)]
    candidates=np.r_[np.nextafter(p[0],np.inf),p[end]]
    # Descending thresholds. First maximum (including all-negative) is conservative.
    best=np.flatnonzero(np.isclose(ba,ba.max(),rtol=0,atol=1e-14))[0]
    return float(candidates[best])


def cutoff(model,y,score):
    if model.endswith("_majority") or model=="current_rule":
        return .5
    if model.startswith("lead_"):
        return balanced_cutoff(y,score)
    return threshold(y,score)


def evaluate(z,cut,cfg,cover):
    y,p=z["y"],z["score"]
    stats=measure(y,p,cut,z["sample_id"],cover)
    assert len(y)==cover["positive"]+cover["negative"]
    pred=p>=cut
    risk=z["risk"]
    stats.update(balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                 recovery_FP=int((pred&risk&(y==0)).sum()),
                 recovery_count=int((risk&(y==0)).sum()),
                 risk_accuracy=float(np.mean(pred[risk]==y[risk])),
                 risk_balanced_accuracy=float(balanced_accuracy_score(y[risk],pred[risk])))
    points,_=matched_points(z["sample_id"],y,p,risk,cfg["recalls"],cover["total_days"])
    for row in points:
        row["recovery_FP_per_day"]=row["recovery_FP"]/cover["total_days"]
    main=next(row for row in points if row["target_recall"]==cfg["main_recall"])
    stats.update(FP60_day=main["FP_per_day"],recovery_FP60_day=main["recovery_FP_per_day"])
    return stats,points


def slice_scores(z,mask):
    return {k:v[mask] for k,v in z.items() if isinstance(v,np.ndarray) and v.ndim==1}

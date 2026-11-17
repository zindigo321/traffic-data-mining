"""Three scalar cutoff policies over one unchanged ranking."""
import numpy as np


def budgets(days,cfg):
    if days<=0:
        raise ValueError("Use positive calendar day counts")
    return int(days*cfg["FP_budget_day"]),int(days*cfg["recovery_budget_day"])


def cutoffs(y,score,risk,days,cfg):
    if not np.isin(y,[0,1]).all() or not np.isfinite(score).all():
        raise ValueError("Cutoff fitting uses known scores only")
    order=np.argsort(-score,kind="stable")
    p=score[order]
    truth=y[order]==1
    npos=int(truth.sum())
    nneg=len(y)-npos
    if not npos or not nneg:
        raise ValueError("Both calibration classes are required")
    ends=np.r_[p[:-1]!=p[1:],True]
    tp=np.r_[0,np.cumsum(truth)[ends]]
    fp=np.r_[0,np.cumsum(~truth)[ends]]
    hard=np.r_[0,np.cumsum((~truth)&risk[order])[ends]]
    values=np.r_[np.nextafter(p[0],np.inf),p[ends]]
    ba=.5*(tp/npos+1-fp/nneg)
    f1=2*tp/(npos+tp+fp)
    total,recovery=budgets(days,cfg)
    feasible=(fp<=total)&(hard<=recovery)
    result={}
    for name,metric,allow in (
        ("max_ba",ba,np.ones_like(feasible)),("max_f1",f1,np.ones_like(feasible)),("constrained",f1,feasible)
    ):
        index=np.flatnonzero(allow&np.isclose(metric,metric[allow].max(),rtol=0,atol=1e-14))[0]
        result[name]=dict(threshold=float(values[index]),fit_TP=int(tp[index]),fit_FP=int(fp[index]),
            fit_recovery_FP=int(hard[index]),fit_BA=float(ba[index]),fit_F1=float(f1[index]),fit_days=days,
            fit_FP_cap=total,fit_recovery_cap=recovery,fit_FP_budget_ok=bool(feasible[index]))
    return result


def transfer_masks(keys,interval,early_days):
    from datetime import date
    a,b=map(date.fromisoformat,interval)
    days=(b-a).days
    base=(a-date(1970,1,1)).days
    day=keys//1000//144-base
    if not (0<early_days<days) or not ((day>=0)&(day<days)).all():
        raise ValueError("Invalid threshold transfer calendar")
    return day<early_days,day>=early_days,days

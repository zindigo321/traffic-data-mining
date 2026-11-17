"""Evaluation on the complete known support with fixed score and event conventions."""
from collections import defaultdict

from v1 import runtime

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def sigmoid(x):
    from scipy.special import expit
    return expit(x)


def rank_score(p, risk, route):
    p = p.astype(np.float64)
    if not route:
        return p
    return np.where(risk, .51 + .49*p, .49*p)


def threshold(y, p):
    if not len(y) or not (np.any(y == 1) and np.any(y == 0)):
        return None
    precision, recall, cutoffs = precision_recall_curve(y,p)
    den = precision[:-1] + recall[:-1]
    f1 = np.divide(2*precision[:-1]*recall[:-1],den,out=np.zeros_like(den),where=den>0)
    c = cutoffs[np.isclose(f1,f1.max(),atol=1e-12,rtol=0)]
    return float(c[np.lexsort((c,np.abs(c-.5)))[0]])


def counts(y, pred):
    return dict(tp=int(((y==1)&pred).sum()),fp=int(((y==0)&pred).sum()),
                fn=int(((y==1)&~pred).sum()),tn=int(((y==0)&~pred).sum()))


def spans(keys):
    # key = civil-epoch 10-minute slot * 1000 + road
    times, road = np.divmod(keys, 1000)
    day = times // 144
    idx = np.lexsort((times,road,day))
    t, r, d = times[idx],road[idx],day[idx]
    if not len(t):
        return {}
    breaks = np.r_[True,(np.diff(t)!=1)|(np.diff(r)!=0)|(np.diff(d)!=0)]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:]-1,len(t)-1]
    result = defaultdict(list)
    for lo,hi in zip(starts,ends,strict=True):
        result[int(d[lo]),int(r[lo])].append((int(t[lo]),int(t[hi])))
    return result


def event_metrics(keys, y, pred):
    actual, alerts = spans(keys[y == 1]),spans(keys[pred])
    n_true = sum(map(len,actual.values()))
    n_alert = sum(map(len,alerts.values()))
    matched = 0
    for key,left in actual.items():
        right = alerts.get(key,())
        i = j = 0
        while i < len(left) and j < len(right):
            a,b = left[i],right[j]
            if a[1] < b[0]:
                i += 1
            elif b[1] < a[0]:
                j += 1
            else:
                matched += 1
                i += 1
                j += 1
    return dict(true_events=n_true, alert_events=n_alert, matched_events=matched,
                event_precision=matched/n_alert if n_alert else None,
                event_recall=matched/n_true if n_true else None,
                event_f1=2*matched/(n_alert+n_true) if n_alert+n_true else None)


def measure(y, p, cutoff, keys, coverage):
    if not np.isin(y, [0,1]).all() or not np.isfinite(p).all() or len(np.unique(keys)) != len(y):
        raise ValueError("Invalid known support or scores")
    pos, neg = int(y.sum()), int(len(y)-y.sum())
    measures = dict(n=len(y),positive=pos,negative=neg,positive_rate=pos/len(y) if len(y) else None,
                    AP=float(average_precision_score(y,p)) if pos and neg else None,
                    ROC_AUC=float(roc_auc_score(y,p)) if pos and neg else None,threshold=cutoff,
                    known_coverage=coverage["known_coverage"],eligible=coverage["eligible"],
                    expected=coverage["expected"],calendar_days=coverage["total_days"],
                    unknown_eligible=coverage["unknown_eligible"])
    if cutoff is None:
        measures.update({k:None for k in ("tp","fp","fn","tn","precision","recall","f1","accuracy","fp_per_day",
                                          "alert_events","matched_events","event_precision","event_recall","event_f1")})
        measures["true_events"] = sum(map(len,spans(keys[y==1]).values()))
        return measures
    pred = p >= cutoff
    c = counts(y,pred)
    tp,fp,fn,tn = [c[k] for k in ("tp","fp","fn","tn")]
    return dict(**measures, **c, precision=tp/(tp+fp) if tp+fp else None, recall=tp/pos if pos else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                accuracy=(tp+tn)/len(y) if len(y) else None, fp_per_day=fp/coverage["total_days"],
                **event_metrics(keys,y,pred))



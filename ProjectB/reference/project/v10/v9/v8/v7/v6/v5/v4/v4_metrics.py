"""Matched-recall diagnostics on all known eligible windows."""
import math

import numpy as np


def matched_points(keys, y, score, risk, targets, calendar_days):
    if (not np.isfinite(score).all() or not np.isin(y, [0, 1]).all()
            or len(np.unique(keys)) != len(keys) or not (0 < y.sum() < len(y))
            or calendar_days <= 0):
        raise ValueError("Invalid matched-recall support")
    order = np.lexsort((keys, -score))
    positive_positions = np.flatnonzero(y[order] == 1)
    rows, selections = [], {}
    for rate in targets:
        k = math.ceil(rate * len(positive_positions))
        if not 0 < k <= len(positive_positions):
            raise ValueError("Invalid recall target")
        position = positive_positions[k - 1]
        indices = order[:position + 1]
        n_alert = len(indices)
        fp = n_alert - k
        recoveries = int((risk[indices] & (y[indices] == 0)).sum())
        tied = score >= score[order[position]]
        rows.append(dict(target_recall=rate, achieved_recall=k / len(positive_positions), TP=k, FP=fp,
                         FN=len(positive_positions) - k, TN=len(keys) - len(positive_positions) - fp,
                         recovery_FP=recoveries, other_FP=fp - recoveries, FP_per_day=fp / calendar_days,
                         precision=k / n_alert, F1=2 * k / (n_alert + len(positive_positions)),
                         boundary_score=float(score[order[position]]), boundary_key=int(keys[order[position]]),
                         score_only_TP=int(y[tied].sum()), score_only_FP=int(tied.sum() - y[tied].sum())))
        selections[rate] = indices
    return rows, selections

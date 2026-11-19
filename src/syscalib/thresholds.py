"""Threshold strategies. They see ONLY calibration scores (never evaluation data).

Timeline convention (no look-ahead):
    T[0] = T0 from the initial clean calibration set.
    T[k] = update(T[k-1], calibration batch k)      k = 1..K-1
    evaluation batch j (1-indexed) is scored with T[j-1].
"""
import numpy as np


def q(scores, level):
    return float(np.quantile(scores, level))


def fixed(T0, cal_batches, **_):
    return np.full(len(cal_batches) + 1, T0)


def quantile(T0, cal_batches, alpha, **_):
    T = [T0]
    for S in cal_batches:
        T.append(q(S, alpha))
    return np.array(T)


def bacr_original(T0, cal_batches, alpha, delta, **_):
    """Original BACR, unchanged: screen at the OLD threshold, quantile of the
    accepted scores, clip movement to +/- delta. Candidate <= T_old always."""
    T = [T0]
    for S in cal_batches:
        told = T[-1]
        acc = S[S <= told]
        cand = q(acc, alpha) if acc.size else told
        T.append(float(np.clip(cand, told - delta, told + delta)))
    return np.array(T)


def bacr_budgeted(T0, cal_batches, alpha, delta, eps_budget, **_):
    """Modified BACR ('budgeted quantile, bounded movement'):
        C_k = Q_{alpha (1 - eps_budget)}(S_k);  T_k = clip(C_k, T_{k-1} +/- delta).
    No screening at the old threshold, so it can move up. If at most a fraction
    eps_budget of the batch is contamination sitting above the benign scores, the
    target benign coverage alpha is still reached; if contamination is smaller, the
    threshold is conservatively low (FPR = 1 - alpha (1 - eps_budget) when clean)."""
    T = [T0]
    lvl = alpha * (1.0 - eps_budget)
    for S in cal_batches:
        told = T[-1]
        cand = q(S, lvl)
        T.append(float(np.clip(cand, told - delta, told + delta)))
    return np.array(T)


STRATEGIES = {
    "fixed": fixed,
    "quantile": quantile,
    "bacr_orig": bacr_original,
    "bacr_budget": bacr_budgeted,
}
# 'oracle' is handled in experiment.py (needs a clean per-phase benign reference).

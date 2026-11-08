"""Kaplan-Meier with Greenwood standard errors (weighted version for the causal analysis)."""
from statistics import NormalDist

import numpy as np


def kaplan_meier(time, event, weight=None):
    """Returns event times, S(t) just after each time, and the Greenwood sum (unweighted case)."""
    w = np.ones_like(time, dtype=float) if weight is None else np.asarray(weight, float)
    order = np.argsort(time, kind="stable")
    t, d, w = time[order], event[order].astype(float), w[order]
    uniq, start = np.unique(t, return_index=True)
    deaths = np.add.reduceat(w * d, start)
    at_risk = w.sum() - np.concatenate([[0.0], np.cumsum(np.add.reduceat(w, start))[:-1]])
    S = np.cumprod(1 - deaths / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(np.where(at_risk > deaths, deaths / (at_risk * (at_risk - deaths)), 0.0))
    return uniq, S, gw


def km_at(time, event, t, weight=None, level=0.95):
    """S(t), its Greenwood standard error and a log(-log) confidence interval (default 95 %)."""
    uniq, S, gw = kaplan_meier(np.asarray(time, float), np.asarray(event, int), weight)
    i = np.searchsorted(uniq, t, side="right") - 1
    if i < 0:
        return {"S": 1.0, "se": 0.0, "lo": 1.0, "hi": 1.0, "greenwood": 0.0}
    s, g = S[i], gw[i]
    se = s * np.sqrt(g)
    # log(-log) interval: stays inside [0, 1]
    z = NormalDist().inv_cdf(0.5 + level / 2)
    half = z * np.sqrt(g) / np.log(s) if 0 < s < 1 else 0.0
    lo, hi = s ** np.exp(-half), s ** np.exp(half)
    return {"S": float(s), "se": float(se), "lo": float(min(lo, hi)), "hi": float(max(lo, hi)), "greenwood": float(g)}


def max_followup(time):
    return float(np.max(time))

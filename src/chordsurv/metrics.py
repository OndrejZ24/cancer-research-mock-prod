"""Censoring-aware metrics at a fixed horizon, and the censoring distribution used for IPCW."""
import numpy as np
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from .config import HORIZON_MONTHS


def censoring_survival(time, event, weight=None):
    """Kaplan-Meier estimate of the censoring distribution G(t) = P(C > t).
    Ties: on a day with both, deaths come first, so patients who die that day are not at risk of being censored
    (with this convention the IPCW share of deaths equals the Kaplan-Meier risk exactly).
    Returns G evaluated just before t."""
    w = np.ones_like(time) if weight is None else weight
    order = np.argsort(time, kind="stable")
    t, c, w = time[order], (event[order] == 0), w[order]
    uniq, start = np.unique(t, return_index=True)
    cens = np.add.reduceat(w * c, start)
    deaths = np.add.reduceat(w * ~c, start)
    at_risk = w.sum() - np.concatenate([[0.0], np.cumsum(np.add.reduceat(w, start))[:-1]])
    exposed = at_risk - deaths                                # 0 when everyone left dies that day: factor 1
    G = np.cumprod(1.0 - np.divide(cens, exposed, out=np.zeros_like(cens), where=exposed > 0))

    def G_before(x):
        i = np.searchsorted(uniq, x, side="left") - 1          # last censoring time strictly before x
        return np.where(i >= 0, G[np.clip(i, 0, None)], 1.0)
    return G_before


def ipcw_horizon_weights(time, event, horizon=HORIZON_MONTHS, weight=None):
    """Weights for a binary "died within the horizon" target: 1/G(T-) for deaths before the horizon,
    1/G(horizon) for patients alive past it, 0 for patients censored before it (their status is unknown)."""
    G = censoring_survival(time, event, weight)
    died = (time <= horizon) & (event == 1)
    alive = time > horizon
    w = np.zeros_like(time, dtype=float)
    w[died] = 1.0 / G(time[died])
    w[alive] = 1.0 / G(np.full(alive.sum(), horizon, dtype=float) + 1e-9)
    return w, died.astype(float)


def evaluate(time, event, risk, horizon=HORIZON_MONTHS):
    """Brier score, cumulative/dynamic AUC and Uno's C-index truncated at the horizon.
    The censoring distribution is estimated on the evaluated patients themselves: censoring depends on era
    (2 % before the horizon in the training eras, 36 % in the newest), so a G borrowed from other patients would
    give the wrong weights."""
    y = Surv.from_arrays(event.astype(bool), time)
    risk = np.clip(np.asarray(risk, float), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score(y, y, (1 - risk).reshape(-1, 1), [horizon])[1][0]),
        "auc": float(cumulative_dynamic_auc(y, y, risk, [horizon])[0][0]),
        "cindex": float(concordance_index_ipcw(y, y, risk, tau=horizon)[0]),
        "mean_risk": float(risk.mean()),
    }


def observed_risk(time, event, horizon=HORIZON_MONTHS):
    """Kaplan-Meier risk of death by the horizon (the observed counterpart of a mean predicted risk)."""
    order = np.argsort(time, kind="stable")
    t, d = time[order], event[order]
    uniq, start = np.unique(t, return_index=True)
    deaths = np.add.reduceat(d, start)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(np.diff(np.append(start, len(t))))[:-1]])
    S = np.cumprod(1 - deaths / at_risk)
    i = np.searchsorted(uniq, horizon, side="right") - 1
    return float(1 - S[i]) if i >= 0 else 0.0

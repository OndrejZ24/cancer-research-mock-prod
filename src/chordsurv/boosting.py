"""Gradient boosting (XGBoost) for the risk of death within the horizon.

Never import this module and `nnsurv` (torch) in the same process: on macOS, xgboost links Homebrew's libomp and
torch ships its own copy. Two OpenMP runtimes in one process crash (torch imported first) or deadlock (xgboost
imported first)."""
import numpy as np
import xgboost as xgb

from .data import SEED
from .metrics import ipcw_horizon_weights
from .models import H, es_split, truncate


def xgb_fit(Z, time, event, weight, rows, n_patients, params, objective="ipcw", seed=SEED, max_rounds=5000,
            patience=100, nthread=8):
    """objective="ipcw": binary log loss for death within the horizon, IPCW-weighted (patients censored before
    the horizon get weight 0, the others 1/G). objective="cox": Cox partial likelihood (the alternative criterion),
    turned into a 12-month risk with a Breslow baseline. Trees are added until the loss on an inner 15 % of
    patients has not improved for `patience` rounds."""
    es = es_split(n_patients, rows, seed=seed)
    if objective == "ipcw":
        w_ipcw, label = ipcw_horizon_weights(time, event, weight=weight)
        w = weight * w_ipcw
        keep = w > 0
        obj = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        t, d = truncate(time, event)
        label, w, keep = np.where(d == 1, t, -t), weight, np.ones(len(t), bool)
        obj = {"objective": "survival:cox", "eval_metric": "cox-nloglik"}
    p = {**obj, "tree_method": "hist", "seed": seed, "nthread": nthread, **params}
    fit_rows, es_rows = keep & ~es, keep & es
    dtrain = xgb.DMatrix(Z[fit_rows], label=label[fit_rows], weight=w[fit_rows])
    des = xgb.DMatrix(Z[es_rows], label=label[es_rows], weight=w[es_rows])
    history = {}
    booster = xgb.train(p, dtrain, num_boost_round=max_rounds, evals=[(dtrain, "train"), (des, "es")],
                        early_stopping_rounds=patience, evals_result=history, verbose_eval=False)
    model = {"booster": booster, "objective": objective, "best_iteration": booster.best_iteration,
             "history": {k: v[obj["eval_metric"]] for k, v in history.items()}, "params": p}
    if objective == "cox":
        t, d = truncate(time[fit_rows], event[fit_rows])
        f = booster.predict(dtrain, output_margin=True, iteration_range=(0, booster.best_iteration + 1))
        model["H0"] = breslow_at(t, d, w[fit_rows], f, H)
    return model


def breslow_at(t, d, w, f, horizon):
    """Breslow estimate of the baseline cumulative hazard at the horizon (weighted)."""
    shift = f.max()
    risk = w * np.exp(f - shift)
    order = np.argsort(-t, kind="stable")
    ts, cum = t[order], np.cumsum(risk[order])
    times = np.unique(t[(d == 1) & (t <= horizon)])
    at_risk = cum[len(ts) - 1 - np.searchsorted(ts[::-1], times, side="left")]
    deaths = np.bincount(np.searchsorted(times, t[(d == 1) & (t <= horizon)]),
                         weights=w[(d == 1) & (t <= horizon)], minlength=len(times))
    return float((deaths / at_risk).sum() * np.exp(-shift))


def xgb_risk(model, Z):
    out = model["booster"].predict(xgb.DMatrix(Z), output_margin=True,
                                   iteration_range=(0, model["best_iteration"] + 1))
    if model["objective"] == "ipcw":
        return 1.0 / (1.0 + np.exp(-out))
    return 1.0 - np.exp(-model["H0"] * np.exp(out))


def gain_importance(model, names, top=15):
    score = model["booster"].get_score(importance_type="total_gain")
    total = sum(score.values())
    out = sorted(((names[int(k[1:])], v / total) for k, v in score.items()), key=lambda x: -x[1])
    return out[:top]

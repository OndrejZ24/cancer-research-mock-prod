"""Hyperparameter search by 5-fold cross-validation inside the training eras, the forward-in-time check and
the final fit. Every step that learns from data (preprocessing, interactions, smoking prevalences, IPCW
weights, early stopping) happens inside the training part of each fold, through LandmarkModel.design()."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import SEED, feature_columns
from .metrics import evaluate
from .models import LandmarkModel, coxnet_fit, coxnet_risk, km_by_type
from .preprocess import collapse

METRICS = ["brier", "auc", "cindex", "mean_risk"]
COXNET_L1 = (0.1, 0.5, 1.0)


def fold_designs(df, folds, learner, feature_set, groups):
    out = []
    for tr, va in folds:
        m = LandmarkModel(learner, feature_set)
        m.columns_ = feature_columns(groups, feature_set)
        a, b = df.iloc[tr], df.iloc[va]
        Za, wa, ra = m.design(a, fit=True)
        Zb, wb, rb = m.design(b)
        out.append({"va": va, "Za": Za, "wa": wa, "ra": ra, "Zb": Zb, "wb": wb, "rb": rb,
                    "ta": a.time.to_numpy(float)[ra], "ea": a.event.to_numpy(int)[ra],
                    "n_a": len(a), "n_b": len(b), "n_features": Za.shape[1]})
    return out


def score(df, va, risk):
    return evaluate(df.time.to_numpy(float)[va], df.event.to_numpy(int)[va], risk)


def summarise(rows, keys):
    """Mean and standard error over folds for every configuration."""
    r = pd.DataFrame(rows)
    g = r.groupby(keys, dropna=False)
    out = g[METRICS].mean()
    for m in ("brier", "auc", "cindex"):
        out[f"{m}_se"] = g[m].std() / np.sqrt(g[m].count())
    out["n_folds"] = g["brier"].count()
    return out.reset_index()


# ---------------------------------------------------------------- Kaplan-Meier by cancer type (no tuning)

def cv_km(df, folds):
    rows, oof = [], np.full(len(df), np.nan)
    for k, (tr, va) in enumerate(folds):
        a = df.iloc[tr]
        km = km_by_type(a.cancer_type.to_numpy(), a.time.to_numpy(float), a.event.to_numpy(int))
        risk = df.cancer_type.iloc[va].map(km).to_numpy(float)
        oof[va] = risk
        rows.append({"fold": k, **score(df, va, risk)})
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------- penalised Cox: grid over l1_ratio x alpha path

def _coxnet_fold(d, l1, grid, tol):
    model = coxnet_fit(d["Za"], d["ta"], d["ea"], l1, alphas=grid, tol=tol)
    risks = np.full((d["n_b"], len(grid)), np.nan)
    nonzero = np.full(len(grid), np.nan)
    for i, a in enumerate(model.alphas_):                     # the path can stop early (deviance saturates)
        risks[:, i] = collapse(coxnet_risk(model, d["Zb"], a), d["wb"], d["rb"], d["n_b"])
        nonzero[i] = (model.coef_[:, i] != 0).sum()
    return risks, nonzero, model.warnings_


def alpha_grid(df, groups, feature_set, l1, n_alphas=40, alpha_min_ratio=0.002):
    """Common grid for all folds: from the smallest penalty that sets every coefficient to zero on the full
    training data, down to alpha_min_ratio times that, equally spaced on a log scale."""
    m = LandmarkModel("coxnet", feature_set)
    m.columns_ = feature_columns(groups, feature_set)
    Z, _, _ = m.design(df, fit=True)
    # a 2-point path whose second penalty is 0.99 * alpha_max: only alpha_max is needed, and a path that
    # jumps straight to a tiny penalty would be a slow, nearly unpenalised fit
    a_max = coxnet_fit(Z, df.time.to_numpy(float), df.event.to_numpy(int), l1, n_alphas=2,
                       alpha_min_ratio=0.99, tol=1e-7).alphas_[0]
    return a_max * np.logspace(0, np.log10(alpha_min_ratio), n_alphas)


def cv_coxnet(df, folds, groups, feature_set, l1_grid=(0.1, 0.5, 1.0), tol=1e-9, n_jobs=5):
    designs = fold_designs(df, folds, "coxnet", feature_set, groups)
    rows, oof, grids, warns = [], {}, {}, []
    for l1 in l1_grid:
        grid = alpha_grid(df, groups, feature_set, l1)
        grids[l1] = grid
        res = Parallel(n_jobs=n_jobs)(delayed(_coxnet_fold)(d, l1, grid, tol) for d in designs)
        oof[l1] = np.full((len(df), len(grid)), np.nan)
        for k, (d, (risks, nonzero, w)) in enumerate(zip(designs, res)):
            oof[l1][d["va"]] = risks
            warns += w
            for i, a in enumerate(grid):
                m = score(df, d["va"], risks[:, i]) if not np.isnan(risks[0, i]) else dict.fromkeys(METRICS)
                rows.append({"l1_ratio": l1, "alpha_index": i, "alpha": a, "fold": k, "nonzero": nonzero[i], **m})
    search = summarise(rows, ["l1_ratio", "alpha_index", "alpha"])
    search = search.merge(pd.DataFrame(rows).groupby(["l1_ratio", "alpha_index"]).nonzero.mean().reset_index())
    return search, pd.DataFrame(rows), oof, grids, warns, designs[0]["n_features"]


# ---------------------------------------------------------------- XGBoost: random search

def sample_xgb(n, seed=SEED):
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return [{"learning_rate": lu(0.01, 0.3), "max_depth": int(rng.integers(2, 9)),
             "min_child_weight": lu(1, 100), "subsample": float(rng.uniform(0.5, 1.0)),
             "colsample_bytree": float(rng.uniform(0.2, 1.0)), "reg_lambda": lu(0.1, 100),
             "reg_alpha": lu(1e-3, 10)} for _ in range(n)]


def _xgb_task(d, cfg, objective, nthread):
    from .boosting import xgb_fit, xgb_risk
    m = xgb_fit(d["Za"], d["ta"], d["ea"], d["wa"], d["ra"], d["n_a"], cfg, objective=objective, nthread=nthread)
    risk = collapse(xgb_risk(m, d["Zb"]), d["wb"], d["rb"], d["n_b"])
    return risk, m["best_iteration"] + 1


def cv_xgb(df, folds, groups, feature_set, configs, objective="ipcw", n_jobs=5, nthread=2, designs=None):
    designs = designs or fold_designs(df, folds, "xgb", feature_set, groups)
    tasks = [(c, k) for c in range(len(configs)) for k in range(len(designs))]
    res = Parallel(n_jobs=n_jobs)(delayed(_xgb_task)(designs[k], configs[c], objective, nthread) for c, k in tasks)
    rows, oof = [], np.full((len(df), len(configs)), np.nan)
    for (c, k), (risk, n_trees) in zip(tasks, res):
        oof[designs[k]["va"], c] = risk
        rows.append({"config": c, "fold": k, "n_trees": n_trees, **configs[c], **score(df, designs[k]["va"], risk)})
    rows = pd.DataFrame(rows)
    search = summarise(rows, ["config"]).merge(rows.groupby("config").n_trees.agg(["mean", "max"])
                                                .add_prefix("trees_").reset_index())
    search = search.merge(pd.DataFrame(configs).rename_axis("config").reset_index())
    return search, rows, oof, designs


# ---------------------------------------------------------------- neural network: random search

def sample_nn(n, seed=SEED):
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return [{"width": int(rng.choice([32, 64, 128, 256])), "depth": int(rng.integers(1, 4)),
             "dropout": float(rng.uniform(0.0, 0.5)), "lr": lu(1e-4, 3e-3), "weight_decay": lu(1e-6, 1e-1)}
            for _ in range(n)]


def _nn_task(d, cfg, seed):
    import torch
    torch.set_num_threads(1)
    from .nnsurv import nn_fit, nn_risk
    m = nn_fit(d["Za"], d["ta"], d["ea"], d["wa"], d["ra"], d["n_a"], cfg, seed=seed)
    risk = collapse(nn_risk([m], d["Zb"]), d["wb"], d["rb"], d["n_b"])
    return risk, {"best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"], "es_loss": m["es_loss"],
                  "grad_norm": m["grad_norm"], "n_params": m["n_params"]}


def cv_nn(df, folds, groups, feature_set, configs, seeds=(SEED,), n_jobs=8, designs=None):
    designs = designs or fold_designs(df, folds, "nn", feature_set, groups)
    tasks = [(c, k, s) for c in range(len(configs)) for k in range(len(designs)) for s in seeds]
    res = Parallel(n_jobs=n_jobs)(delayed(_nn_task)(designs[k], configs[c], s) for c, k, s in tasks)
    rows, oof = [], {}
    for (c, k, s), (risk, info) in zip(tasks, res):
        oof.setdefault((c, s), np.full(len(df), np.nan))[designs[k]["va"]] = risk
        rows.append({"config": c, "fold": k, "seed": s, **configs[c], **info, **score(df, designs[k]["va"], risk)})
    rows = pd.DataFrame(rows)
    search = summarise(rows, ["config", "seed"]).merge(
        rows.groupby(["config", "seed"])[["best_epoch", "epochs_run", "grad_norm"]].mean().reset_index())
    search = search.merge(pd.DataFrame(configs).rename_axis("config").reset_index())
    return search, rows, oof, designs


# ---------------------------------------------------------------- forward check and per-type metrics

def forward_check(df, groups, learner, feature_set, params, seeds=(SEED,)):
    """Train on the two oldest eras, validate on IMPACT468: how much do the chosen settings lose on later
    patients? (A check of the tuning, not a test: IMPACT468 is part of the training data.)"""
    old, new = df[df.era.isin(["IMPACT341", "IMPACT410"])], df[df.era.eq("IMPACT468")]
    m = LandmarkModel(learner, feature_set, params, seeds).fit(old, groups)
    return evaluate(new.time.to_numpy(float), new.event.to_numpy(int), m.predict_risk(new))


def by_type(df, risk):
    out = {}
    for k, g in df.assign(risk=risk).groupby("cancer_type"):
        out[k] = {**evaluate(g.time.to_numpy(float), g.event.to_numpy(int), g.risk.to_numpy()), "n": len(g)}
    return pd.DataFrame(out).T

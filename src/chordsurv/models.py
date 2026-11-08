"""Learners for the risk of death within the horizon (12 months after the 6-month landmark).

Every learner works on preprocessed matrices; `LandmarkModel` wraps preprocessing + learner so that raw
landmark-set rows go in and risks come out (used for the final fits, the test evaluation and production).
XGBoost (boosting.py) and torch (nnsurv.py) are imported only when needed: they bring two different OpenMP
runtimes, which crash or deadlock when loaded into the same process on macOS."""
import warnings

import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from .config import HORIZON_MONTHS
from .data import SEED, feature_columns, prepare
from .metrics import observed_risk
from .preprocess import Preprocessor, SmokingExpander, TypeInteractions, collapse

H = float(HORIZON_MONTHS)


def truncate(time, event, horizon=H):
    """Administrative censoring at the horizon: only the first 12 months after the landmark matter for the target,
    and the proportional-hazards violation found in Part I is mostly early vs late follow-up."""
    return np.minimum(time, horizon), ((event == 1) & (time <= horizon)).astype(int)


def es_split(n_patients, rows, frac=0.15, seed=SEED):
    """Inner early-stopping split by patient (both rows of an expanded patient fall on the same side)."""
    rng = np.random.default_rng(seed)
    es_patients = rng.random(n_patients) < frac
    return es_patients[rows]


# ---------------------------------------------------------------- baseline: Kaplan-Meier by cancer type

def km_by_type(cancer_type, time, event, horizon=H):
    return {k: observed_risk(time[cancer_type == k], event[cancer_type == k], horizon)
            for k in np.unique(cancer_type)}


# ---------------------------------------------------------------- penalised Cox (elastic net)

def coxnet_fit(Z, time, event, l1_ratio, alphas=None, n_alphas=40, alpha_min_ratio=0.002, tol=1e-9):
    """Elastic-net Cox partial likelihood (Breslow ties), coordinate descent over a path of penalties.
    The default tolerance (1e-7) stops before the optimum at small penalties (KKT check), hence 1e-9 / 1e-10."""
    t, d = truncate(time, event)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, n_alphas=n_alphas,
                                       alpha_min_ratio=alpha_min_ratio, fit_baseline_model=True,
                                       tol=tol, max_iter=1_000_000).fit(Z.astype(np.float64),
                                                                       Surv.from_arrays(d.astype(bool), t))
    model.warnings_ = [str(w.message) for w in caught]
    return model


def coxnet_risk(model, Z, alpha, horizon=H):
    S = model.predict_survival_function(Z.astype(np.float64), alpha=alpha, return_array=True)
    i = np.searchsorted(model.unique_times_, horizon, side="right") - 1
    return 1.0 - S[:, i]


def cox_kkt(model, Z, time, event, alpha_index):
    """Optimality check of the elastic-net Cox solution (Breslow ties, objective = -loglik/n + penalty).
    Returns the largest violation of the KKT conditions relative to alpha: about 0 = converged to the optimum."""
    t, d = truncate(time, event)
    X = Z.astype(np.float64)
    beta = model.coef_[:, alpha_index]
    alpha, l1 = model.alphas_[alpha_index], model.l1_ratio
    eta = X @ beta
    eta -= eta.max()
    r = np.exp(eta)
    order = np.argsort(-t, kind="stable")                    # descending time: cumulative sums = risk sets
    ts, rs, Xs = t[order], r[order], X[order]
    cum_r, cum_rx = np.cumsum(rs), np.cumsum(rs[:, None] * Xs, axis=0)
    # risk set of an event at time u = everyone with time >= u: last index with ts >= u
    last = len(ts) - 1 - np.searchsorted(ts[::-1], t, side="left")
    ev = d == 1
    grad = -(X[ev].sum(axis=0) - (cum_rx[last[ev]] / cum_r[last[ev], None]).sum(axis=0)) / len(t)
    nz = beta != 0
    viol = np.empty_like(beta)
    viol[nz] = np.abs(grad[nz] + alpha * ((1 - l1) * beta[nz] + l1 * np.sign(beta[nz])))
    viol[~nz] = np.maximum(np.abs(grad[~nz]) - alpha * l1, 0)
    return float(viol.max() / alpha)


# ---------------------------------------------------------------- discrete-time targets (neural network)

N_INTERVALS = 12


def interval_targets(time, event, horizon=H, n=N_INTERVALS):
    """Discrete-time likelihood terms: survived[i, j] = 1 if patient i is known to survive interval j,
    died[i, j] = 1 if i died in interval j. A patient censored in the second half of an interval counts as having
    survived it (Gensheimer & Narasimhan 2019)."""
    width = horizon / n
    k = np.minimum(np.floor(np.minimum(time, horizon) / width).astype(int), n - 1)
    died_h = (event == 1) & (time <= horizon)
    alive = time > horizon
    J = np.arange(n)
    survived = (J[None, :] < k[:, None]).astype(np.float32)
    survived[alive] = 1.0
    half = ~died_h & ~alive & (np.minimum(time, horizon) / width - k >= 0.5)
    survived[half, k[half]] = 1.0
    died = np.zeros_like(survived)
    died[died_h, k[died_h]] = 1.0
    return survived, died



# ---------------------------------------------------------------- end-to-end model

class LandmarkModel:
    """Raw landmark-set rows -> risk of death within 12 months after the landmark.

    learner: "km" | "coxnet" | "xgb" | "xgb_cox" | "nn"; feature_set: see data.feature_columns."""

    def __init__(self, learner, feature_set, params=None, seeds=(SEED,)):
        self.learner, self.feature_set, self.params, self.seeds = learner, feature_set, dict(params or {}), seeds

    @property
    def flexible(self):
        return self.learner in ("xgb", "xgb_cox", "nn")

    def design(self, df, fit=False):
        """Preprocessed matrix, row weights and the patient position of each row."""
        X = prepare(df, self.columns_)
        if fit:
            self.pre_ = Preprocessor(scale=self.learner in ("coxnet", "nn")).fit(X)
            self.expander_ = SmokingExpander().fit(X) if self.flexible else None
        if self.flexible:
            X, w, rows = self.expander_.expand(X)
        else:
            w, rows = np.ones(len(X)), np.arange(len(X))
        Z = self.pre_.transform(X)
        if self.learner == "coxnet":
            if fit:
                self.inter_ = TypeInteractions().fit(Z, X.cancer_type, self.pre_.feature_names_)
            Z = self.inter_.transform(Z, X.cancer_type)
        return Z, w, rows

    @property
    def feature_names_(self):
        return self.inter_.feature_names_ if self.learner == "coxnet" else self.pre_.feature_names_

    def fit(self, df, groups, tol=1e-10):
        time, event = df.time.to_numpy(float), df.event.to_numpy(int)
        if self.learner == "km":                               # uses the cancer type only
            self.columns_ = ["cancer_type"]
            self.km_ = km_by_type(df.cancer_type.to_numpy(), time, event)
            return self
        self.columns_ = feature_columns(groups, self.feature_set)
        Z, w, rows = self.design(df, fit=True)
        t, e = time[rows], event[rows]
        if self.learner == "coxnet":
            alphas = np.asarray(self.params["alphas"])        # the CV grid; the path is fitted down to the chosen alpha
            self.model_ = coxnet_fit(Z, t, e, self.params["l1_ratio"], alphas=alphas[alphas >= self.params["alpha"] * 0.999],
                                     tol=tol)
            self.alpha_index_ = len(self.model_.alphas_) - 1
            self.alpha_ = float(self.model_.alphas_[-1])       # = the chosen alpha unless the path stopped earlier
            if not np.isclose(self.alpha_, self.params["alpha"]):
                warnings.warn(f"Coxnet path stopped at alpha {self.alpha_:.3g} before the chosen {self.params['alpha']:.3g}")
        elif self.learner in ("xgb", "xgb_cox"):
            from .boosting import xgb_fit
            xgb_params = {k: v for k, v in self.params.items() if k != "nthread"}
            self.model_ = xgb_fit(Z, t, e, w, rows, len(df), xgb_params, seed=self.seeds[0],
                                  objective="ipcw" if self.learner == "xgb" else "cox")
        elif self.learner == "nn":
            from .nnsurv import nn_fit
            self.model_ = [nn_fit(Z, t, e, w, rows, len(df), self.params, seed=s) for s in self.seeds]
        return self

    def predict_risk(self, df):
        if self.learner == "km":
            return df.cancer_type.map(self.km_).to_numpy(float)
        Z, w, rows = self.design(df)
        if self.learner == "coxnet":
            r = coxnet_risk(self.model_, Z, self.alpha_)
        elif self.flexible and self.learner != "nn":
            from .boosting import xgb_risk
            r = xgb_risk(self.model_, Z)
        else:
            from .nnsurv import nn_risk
            r = nn_risk(self.model_, Z)
        return collapse(r, w, rows, len(df))

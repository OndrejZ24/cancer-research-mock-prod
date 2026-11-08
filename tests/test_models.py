"""Tests of the Part II training code on synthetic data (no licensed data needed).

xgboost and torch load two different OpenMP runtimes that crash or deadlock in one process (macOS), so every
test that needs one of them runs in its own Python subprocess."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chordsurv.data import prepare                                       # noqa: E402
from chordsurv.metrics import censoring_survival, ipcw_horizon_weights, observed_risk  # noqa: E402
from chordsurv.models import LandmarkModel, coxnet_fit, cox_kkt, interval_targets  # noqa: E402
from chordsurv.preprocess import Preprocessor, SmokingExpander, collapse  # noqa: E402
from chordsurv.synthetic import make_landmark_like                        # noqa: E402


def run_isolated(code):
    """Run `code` in a fresh interpreter; its last stdout line must be JSON."""
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n" + code],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


# ---------------------------------------------------------------- censoring weights

def test_censoring_survival_by_hand():
    time = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 0, 1])
    G = censoring_survival(time, event)
    # day 2: a death and a censoring; the death comes first, so 3 patients are at risk of censoring (G = 2/3);
    # day 3: 1 censoring of 2 at risk (G = 1/3). G is evaluated just before t.
    assert np.allclose(G(np.array([1.0, 2.0, 2.5, 3.0, 3.5])), [1.0, 1.0, 2 / 3, 2 / 3, 1 / 3])


def test_ipcw_mean_equals_kaplan_meier_risk():
    df, _ = make_landmark_like(3000, seed=1)
    t, e = df.time.to_numpy(), df.event.to_numpy()
    w, died = ipcw_horizon_weights(t, e, 12.0)
    assert np.isclose((w * died).sum() / w.sum(), observed_risk(t, e, 12.0), atol=1e-10)
    assert (w[(t <= 12) & (e == 0)] == 0).all()                         # unknown status gets no weight


# ---------------------------------------------------------------- preprocessing

def test_preprocessor_is_fold_safe_and_handles_new_data():
    df, groups = make_landmark_like(2000, seed=2)
    cols = groups["static"] + groups["landmark"] + groups["genomic"]
    train, new = prepare(df.iloc[:1500], cols), prepare(df.iloc[1500:], cols).copy()
    pre = Preprocessor().fit(train)
    names = pre.feature_names_
    assert "mut_RARE" not in names                                       # below 1 % prevalence
    assert "mut_NEWGENE" in names and "mut_NEWGENE_missing" not in names  # untested gene: no indicator
    assert "never_smoker_missing" not in names and "lm_ecog_missing" in names
    new.loc[new.index[:5], "cancer_type_detailed"] = "A type never seen in training"
    new.loc[new.index[:5], "tmb"] = 1e6                                  # far outside the training range
    Z = pre.transform(new)
    assert Z.shape == (500, len(names)) and np.isfinite(Z).all()
    with pytest.raises(ValueError):
        pre.transform(new.drop(columns=["tmb"]))


def test_smoking_expansion_weights_and_collapse():
    df, groups = make_landmark_like(500, seed=3)
    X = prepare(df, groups["static"])
    ex = SmokingExpander().fit(X)
    Xe, w, rows = ex.expand(X)
    assert Xe.never_smoker.notna().all()
    assert np.allclose(np.bincount(rows, weights=w), 1.0)                # each patient's rows sum to weight 1
    risk = np.where(Xe.never_smoker.to_numpy() == 1, 0.1, 0.3)
    out = collapse(risk, w, rows, len(X))
    unknown = X.never_smoker.isna().to_numpy()
    p = X.cancer_type.map(ex.p_).to_numpy()
    assert np.allclose(out[unknown], 0.1 * p[unknown] + 0.3 * (1 - p[unknown]))


# ---------------------------------------------------------------- learners

def test_interval_targets_by_hand():
    time = np.array([2.3, 2.7, 2.2, 15.0, 12.0])
    event = np.array([1, 0, 0, 1, 1])
    s, d = interval_targets(time, event, 12.0, 12)
    assert s[0, :2].all() and not s[0, 2:].any() and d[0, 2] == 1 and d[0].sum() == 1
    assert s[1, :3].all() and not s[1, 3:].any() and d[1].sum() == 0    # censored in the second half of month 3
    assert s[2, :2].all() and not s[2, 2:].any() and d[2].sum() == 0    # censored in the first half
    assert s[3].all() and d[3].sum() == 0                                # alive past the horizon
    assert d[4, 11] == 1 and s[4, :11].all()                            # death exactly at month 12


def test_coxnet_reaches_the_optimum():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((1500, 20))
    T = rng.exponential(np.exp(-(X[:, :4] @ np.array([0.8, -0.5, 0.4, 0.3]))) * 10)
    t, e = np.minimum(T, 15.0), (T <= 15.0).astype(int)
    m = coxnet_fit(X, t, e, l1_ratio=0.5, n_alphas=20, alpha_min_ratio=0.01, tol=1e-10)
    assert cox_kkt(m, X, t, e, len(m.alphas_) - 1) < 1e-2


@pytest.mark.parametrize("learner", ["km", "coxnet"])
def test_landmark_model_in_process(learner):
    df, groups = make_landmark_like(2000, seed=5)
    train, test = df.iloc[:1500], df.iloc[1500:]
    params = {}
    if learner == "coxnet":
        params = {"l1_ratio": 0.5, "alpha": 0.01, "alphas": list(np.logspace(-0.5, -2, 15))}
    risk = LandmarkModel(learner, "landmark", params).fit(train, groups).predict_risk(test)
    again = LandmarkModel(learner, "landmark", params).fit(train, groups).predict_risk(test)
    assert np.all((risk > 0) & (risk < 1)) and np.array_equal(risk, again)
    if learner == "coxnet":
        died = ((test.time <= 12) & (test.event == 1)).to_numpy().astype(int)
        known = ((test.time > 12) | (died == 1)).to_numpy()
        assert auc(died[known], risk[known]) > 0.65


ISOLATED = """
import json, numpy as np
from chordsurv.models import LandmarkModel
from chordsurv.synthetic import make_landmark_like
df, groups = make_landmark_like(2000, seed=5)
train, test = df.iloc[:1500], df.iloc[1500:]
params = {params}
r1 = LandmarkModel("{learner}", "landmark", params, seeds={seeds}).fit(train, groups).predict_risk(test)
r2 = LandmarkModel("{learner}", "landmark", params, seeds={seeds}).fit(train, groups).predict_risk(test)
died = ((test.time <= 12) & (test.event == 1)).to_numpy()
known = ((test.time > 12) | died).to_numpy()
pos, neg = r1[known & died], r1[known & ~died]
print(json.dumps({{"in_range": bool(np.all((r1 > 0) & (r1 < 1))), "same": bool(np.array_equal(r1, r2)),
                  "auc": float((pos[:, None] > neg[None, :]).mean())}}))
"""

XGB = {"learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
       "reg_lambda": 1.0, "reg_alpha": 0.0}
NN = {"width": 32, "depth": 1, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4}


@pytest.mark.parametrize("learner,params,seeds", [("xgb", XGB, (42,)), ("xgb_cox", XGB, (42,)),
                                                  ("nn", NN, (42, 43))])
def test_landmark_model_isolated(learner, params, seeds):
    out = run_isolated(ISOLATED.format(learner=learner, params=params, seeds=seeds))
    assert out["in_range"] and out["same"] and out["auc"] > 0.65


def test_breslow_equals_nelson_aalen_without_covariates():
    out = run_isolated("""
import json, numpy as np
from lifelines import NelsonAalenFitter
from chordsurv.boosting import breslow_at
rng = np.random.default_rng(6)
t = np.round(rng.exponential(10, 800), 1) + 0.1
d = (rng.random(800) < 0.7).astype(int)
H = breslow_at(t, d, np.ones(800), np.zeros(800), 12.0)
na = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(t, d)
print(json.dumps({"breslow": H, "nelson_aalen": float(na.cumulative_hazard_at_times(12.0).iloc[0])}))
""")
    assert np.isclose(out["breslow"], out["nelson_aalen"], rtol=1e-6)


def test_xgboost_and_torch_cannot_share_a_process_is_documented():
    # guard: the package itself must not import xgboost or torch at import time
    out = run_isolated("""
import json, sys
import chordsurv.models, chordsurv.tuning, chordsurv.preprocess, chordsurv.data, chordsurv.metrics
print(json.dumps({"xgboost": "xgboost" in sys.modules, "torch": "torch" in sys.modules}))
""")
    assert not out["xgboost"] and not out["torch"]

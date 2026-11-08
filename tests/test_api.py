"""Production tests of the public interface, on synthetic data (they run in GitHub Actions).

Grouped by the checks a tool must pass: same answers as the model, only information known at the landmark,
sensible behaviour on bad inputs, reproducibility, honest uncertainty. XGBoost/torch run in subprocesses
(two OpenMP runtimes cannot share one process)."""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chordsurv.api import InputError, InputWarning, RiskPredictor               # noqa: E402
from chordsurv.cli import _demo_params                                          # noqa: E402
from chordsurv.models import LandmarkModel                                      # noqa: E402
from chordsurv.survival import km_at                                            # noqa: E402
from chordsurv.synthetic import make_landmark_like                              # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "coxnet_synthetic.csv"


def data():
    df, groups = make_landmark_like(2500, seed=11)
    return df.iloc[:2000], df.iloc[2000:].drop(columns=["time", "event"]), groups


def fitted(learner="coxnet"):
    train, new, groups = data()
    return RiskPredictor(learner, "landmark", params=_demo_params(learner)).fit(train, groups), new, train, groups


def run_isolated(code):
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n" + code],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- 1. the same answers as the model

def test_predictor_returns_the_model_predictions():
    model, new, train, groups = fitted()
    direct = LandmarkModel("coxnet", "landmark", _demo_params("coxnet")).fit(train, groups).predict_risk(new)
    assert np.array_equal(model.predict(new).risk_12m.to_numpy(), direct)


def test_saved_cases_are_reproduced():
    """Golden file: risks of 20 synthetic patients written when the model was validated."""
    model, new, _, _ = fitted()
    expected = pd.read_csv(GOLDEN, index_col=0)
    got = model.predict(new.loc[expected.index]).risk_12m
    assert np.allclose(got.to_numpy(), expected.risk_12m.to_numpy(), atol=1e-6)


@pytest.mark.parametrize("learner", ["xgb", "nn"])
def test_predictor_returns_the_model_predictions_flexible(learner):
    out = run_isolated(f"""
import json, numpy as np
from chordsurv.api import RiskPredictor
from chordsurv.cli import _demo_params
from chordsurv.models import LandmarkModel
from chordsurv.synthetic import make_landmark_like
df, groups = make_landmark_like(2500, seed=11)
train, new = df.iloc[:2000], df.iloc[2000:].drop(columns=["time", "event"])
p = _demo_params("{learner}")
a = RiskPredictor("{learner}", "landmark", params=p, seeds=(42,)).fit(train, groups).predict(new).risk_12m.to_numpy()
b = LandmarkModel("{learner}", "landmark", p, seeds=(42,)).fit(train, groups).predict_risk(new)
c = RiskPredictor("{learner}", "landmark", params=p, seeds=(42,)).fit(train, groups).predict(new).risk_12m.to_numpy()
print(json.dumps({{"same_as_model": bool(np.array_equal(a, b)), "repeatable": bool(np.array_equal(a, c))}}))
""")
    assert out["same_as_model"] and out["repeatable"]


# ---------------------------------------------------------------- 2. only what is known at the landmark

def test_outcome_columns_are_never_inputs():
    model, new, _, _ = fitted()
    assert not {"time", "event", "os_months", "OS_MONTHS", "OS_STATUS", "CURRENT_AGE_DEID"} & set(model.profile_["columns"])
    with_outcome = new.assign(time=50.0, event=1)                     # a leaked outcome must not change the risk
    assert np.array_equal(model.predict(new).risk_12m, model.predict(with_outcome).risk_12m)


def test_patient_not_alive_at_landmark_is_refused():
    model, new, _, _ = fitted()
    with pytest.raises(InputError, match="not alive at the 6-month landmark"):
        model.predict(new.assign(time=np.r_[-1.0, np.full(len(new) - 1, 5.0)]))


# ---------------------------------------------------------------- 3. bad inputs: refuse or warn, never nonsense

@pytest.mark.parametrize("column,value,message", [
    ("lm_ecog", 7, "impossible value"),
    ("win_prog_max", 1.3, "impossible value"),
    ("tmb", -2.0, "impossible value"),
    ("mut_TP53", 2, "must be 0 or 1"),
    ("cancer_type", "Melanoma", "only covers"),
    ("lm_ecog", "two", "not a number"),
])
def test_impossible_values_are_refused(column, value, message):
    model, new, _, _ = fitted()
    bad = new.copy()
    if isinstance(value, str) and column != "cancer_type":
        bad[column] = bad[column].astype(object)
    bad.iloc[3, bad.columns.get_loc(column)] = value
    with pytest.raises(InputError, match=message):
        model.predict(bad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InputWarning)
        out = model.predict(bad, on_error="drop")
        clean = model.predict(new)
    assert np.isnan(out.risk_12m.iloc[3]) and "error" in out.issues.iloc[3]
    keep = np.arange(len(new)) != 3
    assert np.allclose(out.risk_12m.to_numpy()[keep], clean.risk_12m.to_numpy()[keep])


def test_missing_column_is_refused():
    model, new, _, _ = fitted()
    with pytest.raises(InputError, match="missing"):
        model.predict(new.drop(columns=["lm_ecog"]))


def test_out_of_range_and_unseen_categories_warn_but_predict():
    model, new, _, _ = fitted()
    odd = new.copy()
    odd.iloc[0, odd.columns.get_loc("tmb")] = 5000.0
    odd.iloc[1, odd.columns.get_loc("cancer_type_detailed")] = "A subtype never seen"
    with pytest.warns(InputWarning) as rec:
        out = model.predict(odd)
    msgs = " | ".join(str(w.message) for w in rec)
    assert "outside the training range" in msgs and "not seen in training" in msgs
    assert out.risk_12m.notna().all() and (out.risk_12m.between(0, 1)).all()


def test_missing_values_are_handled():
    model, new, _, _ = fitted()
    holey = new.copy()
    holey.loc[holey.index[:10], ["lm_ecog", "win_prog_max", "tmb"]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InputWarning)
        out = model.predict(holey)
    assert out.risk_12m.notna().all()
    assert "months 0-6" in "".join(str(i.message) for i in model.check(holey))


# ---------------------------------------------------------------- 4. reproducible

def test_save_and_load_give_identical_predictions(tmp_path):
    model, new, _, _ = fitted()
    path = model.save(tmp_path / "model.pkl")
    again = RiskPredictor.load(path)
    assert np.array_equal(model.predict(new).risk_12m, again.predict(new).risk_12m)
    assert again.meta_["data_fingerprint"] == model.meta_["data_fingerprint"]
    assert again.meta_["versions"]["scikit-survival"]


def test_cli_demo_runs():
    r = subprocess.run([sys.executable, "-m", "chordsurv.cli", "demo"], capture_output=True, text=True, timeout=600,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "refused" in r.stdout and "risk_12m" in r.stdout


# ---------------------------------------------------------------- 5. honest uncertainty

@pytest.mark.parametrize("level", [0.80, 0.95])
def test_kaplan_meier_intervals_cover_the_truth(level):
    """Simulated cohorts with a known S(24): the log(-log) interval must cover it at about its nominal rate."""
    rng = np.random.default_rng(7)
    rate, truth, hits, reps = 0.03, np.exp(-0.03 * 24), 0, 600
    for _ in range(reps):
        T, C = rng.exponential(1 / rate, 400), rng.uniform(6, 60, 400)
        k = km_at(np.minimum(T, C), (T <= C).astype(int), 24.0, level=level)
        hits += k["lo"] <= truth <= k["hi"]
    se = np.sqrt(level * (1 - level) / reps)
    assert abs(hits / reps - level) < 3 * se + 0.01, hits / reps

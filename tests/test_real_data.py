"""Checks on the licensed MSK-CHORD data and the Part II artifacts. They run only where the data are present
(see data/README.md) and are skipped elsewhere, e.g. in GitHub Actions."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chordsurv.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR   # noqa: E402

HAVE_DATA = (PROCESSED_DIR / "landmark.parquet").exists()
HAVE_ARTIFACTS = (ARTIFACTS_DIR / "part2" / "model_coxnet_landmark.pkl").exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="needs the licensed MSK-CHORD data (see data/README.md)")


@pytest.fixture(scope="module")
def train():
    from chordsurv.data import load_landmark
    lm = load_landmark()
    return lm[~lm.test_era]


@pytest.fixture(scope="module")
def coxnet(train):
    from chordsurv.api import RiskPredictor
    from chordsurv.data import load_groups
    return RiskPredictor("coxnet", "landmark").fit(train, load_groups())


def test_validation_accepts_every_training_patient(coxnet, train):
    assert not [i for i in coxnet.check(train) if i.level == "error"]


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="needs the Part II artifacts")
def test_package_reproduces_the_part2_cox_model(coxnet, train):
    import pickle
    with open(ARTIFACTS_DIR / "part2" / "model_coxnet_landmark.pkl", "rb") as f:
        part2 = pickle.load(f)
    cases = train.sample(300, random_state=1)
    assert np.allclose(coxnet.predict(cases).risk_12m.to_numpy(), part2.predict_risk(cases), atol=1e-9)


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="needs the Part II artifacts")
def test_package_reproduces_the_part2_xgboost_model():
    code = f"""
import sys, json, pickle, numpy as np; sys.path.insert(0, {str(ROOT / 'src')!r})
from chordsurv.api import RiskPredictor
from chordsurv.data import load_landmark, load_groups
lm = load_landmark(); train = lm[~lm.test_era]
m = RiskPredictor("xgb", "landmark").fit(train, load_groups())
part2 = pickle.load(open({str(ARTIFACTS_DIR / 'part2' / 'model_xgb_landmark.pkl')!r}, "rb"))
cases = train.sample(300, random_state=1)
print(json.dumps({{"same": bool(np.array_equal(m.predict(cases).risk_12m.to_numpy(), part2.predict_risk(cases)))}}))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["same"]


@pytest.mark.skipif(not (ARTIFACTS_DIR / "part2" / "oof_xgb_landmark.parquet").exists(), reason="needs Part II OOF")
def test_out_of_fold_risk_is_calibrated_in_the_large(train):
    from chordsurv.metrics import observed_risk
    oof = pd.read_parquet(ARTIFACTS_DIR / "part2" / "oof_xgb_landmark.parquet").risk.reindex(train.index)
    observed = observed_risk(train.time.to_numpy(float), train.event.to_numpy(int))
    assert abs(oof.mean() - observed) < 0.015


def test_describe_reproduces_the_part2_table():
    from chordsurv.estimates import survival_by_type
    table = survival_by_type(pd.read_parquet(PROCESSED_DIR / "patients.parquet"), B=0)[0]
    saved = pd.read_csv(RESULTS_DIR / "part2" / "inference" / "s24_by_type.csv", index_col=0)
    for c in ("S", "se", "lo", "hi", "S_era_standardised"):
        assert np.allclose(table[c], saved.loc[table.index, c], rtol=1e-6)


@pytest.mark.skipif(not Path(RAW_DIR).exists(), reason="needs the raw export")
def test_real_landmark_features_use_no_later_events(train):
    """Cut every timeline at day 182 (a treatment still running stays running) and recompute: nothing changes."""
    from chordsurv import features as F
    from chordsurv.load import load_raw
    raw = load_raw()
    ids = pd.Index(train.sample(1000, random_state=2).index)
    keep = {k: v[v.PATIENT_ID.isin(ids)] for k, v in raw["timelines"].items()}
    cut = {}
    for k, v in keep.items():
        v = v[v.START_DATE <= 182].copy()
        if "STOP_DATE" in v:
            v["STOP_DATE"] = v.STOP_DATE.clip(upper=182)
        cut[k] = v
    full, trunc = {"timelines": keep}, {"timelines": cut}
    for r in (full, trunc):
        d0 = F.clinical_at(r, ids, 0)
        r["out"] = pd.concat([F.clinical_at(r, ids, 182), F.window_features(r, ids, d0), F.intensity_features(r, ids)],
                             axis=1)
    pd.testing.assert_frame_equal(full["out"], trunc["out"])

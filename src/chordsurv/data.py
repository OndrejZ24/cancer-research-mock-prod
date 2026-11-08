"""Processed datasets, feature sets, the prediction target and the cross-validation folds."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import HORIZON_MONTHS, LABS, PROCESSED_DIR

SEED = 42
CATEGORICAL = ["cancer_type", "cancer_type_detailed", "sample_type", "metastatic_site", "msi_type", "stage_dx",
               "sex", "race"]
GENOMIC_PREFIXES = ("mut_", "amp_", "del_", "fus_")
PDL1 = ["d0_pdl1_positive", "lm_pdl1_positive"]
BREAST = "Breast Cancer"


def load_groups():
    return json.loads((PROCESSED_DIR / "feature_groups.json").read_text())


def load_landmark():
    """Landmark set (one row per patient alive at month 6) with the genomic features of the first sample."""
    lm = pd.read_parquet(PROCESSED_DIR / "landmark.parquet")
    return lm.join(pd.read_parquet(PROCESSED_DIR / "genomics.parquet"))


def feature_columns(groups, feature_set):
    """Columns of a feature set.
    day0: known at sequencing | landmark: + months 0-6 | landmark_counts: + MSK report counts (MSK only) |
    landmark_no_pdl1: landmark without PD-L1 (its meaning drifts between eras)."""
    day0 = groups["static"] + groups["day0"] + groups["genomic"]
    landmark = day0 + groups["landmark"] + groups["window"]
    sets = {"day0": day0, "landmark": landmark, "landmark_counts": landmark + groups["intensity"],
            "landmark_no_pdl1": [c for c in landmark if c not in PDL1]}
    cols = sets[feature_set]
    assert not set(cols) & set(groups["never_use"]), "leaky column in feature set"
    return cols


def _log1p_columns(cols):
    labs = [f"{p}_{lab}" for p in ("d0", "lm") for lab in LABS]
    counts = [c for c in cols if c.startswith("cnt_")]
    return [c for c in ["tmb", "msi_score", "dx_to_seq_days", *labs, *counts] if c in cols]


def prepare(df, cols):
    """Deterministic recoding (nothing is learned from data, so it is safe outside the folds)."""
    X = df[cols].copy()
    # receptor status is only defined for breast cancer; the few values elsewhere belong to an earlier breast tumour
    for c in ("hr_positive", "her2_positive"):
        if c in X:
            X[c] = X[c].where(df.cancer_type.eq(BREAST), 0.0)
    if "metastatic_site" in X:
        X["metastatic_site"] = X.metastatic_site.str.strip().str.lower()   # "Lymph Node" vs "Lymph node"
    for c in _log1p_columns(cols):
        X[c] = np.log1p(X[c].clip(lower=0))                                # heavy right tails (TMB, CEA, ...)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    return X


def target(df, horizon=HORIZON_MONTHS):
    """Months from the landmark to death or censoring, and the status at the horizon:
    1 = died within the horizon, 0 = known alive at the horizon, NaN = censored before the horizon."""
    time, event = df.time.to_numpy(float), df.event.to_numpy(int)
    status = np.where((time <= horizon) & (event == 1), 1.0, np.where(time > horizon, 0.0, np.nan))
    return time, event, status


def make_folds(df, n_splits=5, seed=SEED):
    """Patient-level K folds, stratified by cancer type x era x status at the horizon."""
    _, _, status = target(df)
    strata = df.cancer_type + "|" + df.era + "|" + pd.Series(status, index=df.index).fillna(-1).astype(str)
    counts = strata.map(strata.value_counts())
    strata = strata.where(counts >= n_splits, df.cancer_type + "|small")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(df)), strata)]

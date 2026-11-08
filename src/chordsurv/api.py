"""Public interface: fit the 6-month landmark model with the Part II settings and predict the risk of death
within 12 months for new patients, refusing inputs that would give nonsense.

    from chordsurv.api import RiskPredictor
    model = RiskPredictor("xgb", "landmark").fit(train_table, groups)
    out = model.predict(new_table)          # DataFrame: risk_12m + issues per patient
    model.save("model.pkl"); RiskPredictor.load("model.pkl")
"""
import hashlib
import json
import pickle
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from .config import HORIZON_MONTHS, LANDMARK_MONTHS
from .data import CATEGORICAL, GENOMIC_PREFIXES, SEED
from .models import LandmarkModel

LEARNERS = ("km", "coxnet", "xgb", "nn")
FEATURE_SETS = ("day0", "landmark", "landmark_counts", "landmark_no_pdl1")

BINARY = re.compile(r"^pre_tx_|_site_|^win_new_|^win_tx_|on_systemic_tx|^win_radiation$|^win_surgery$|any_radiology"
                    r"|^hr_positive$|^her2_positive$|^never_smoker$|^hispanic$|pdl1_positive|mmr_absent|prior_meds")
PROBABILITY = re.compile(r"prog_|presence_")
MONTHS_0_6 = ("win_prog_max", "win_ecog_last", "lm_ecog", "lm_prog_max_recent", "win_presence_mean")
PACKAGES = ("numpy", "pandas", "scikit-learn", "scikit-survival", "xgboost", "torch")


class InputError(ValueError):
    """The input cannot be used for a prediction."""


class InputWarning(UserWarning):
    """The prediction is possible but less trustworthy (outside the training data)."""


@dataclass
class Issue:
    level: str          # "error" or "warning"
    column: str
    message: str
    rows: list          # positions of the affected patients (empty = the whole table)


def chosen_settings(learner, feature_set):
    """Hyperparameters chosen by cross-validation in Part II (shipped with the package; no data)."""
    if learner == "km":
        return {}
    s = json.loads(resources.files("chordsurv").joinpath("settings.json").read_text())
    return dict(s[learner][feature_set])


def _version(p):
    try:
        return version(p)
    except PackageNotFoundError:
        return None


def _rule(col):
    """Physically possible values of a feature column: ("binary",), ("range", lo, hi) or None (any value)."""
    if col in CATEGORICAL or col.endswith("logratio"):
        return None
    if col.startswith(GENOMIC_PREFIXES) or BINARY.search(col):
        return ("binary",)
    if PROBABILITY.search(col):
        return ("range", 0.0, 1.0)
    if col in ("d0_ecog", "lm_ecog", "win_ecog_last"):
        return ("range", 0.0, 5.0)
    if col == "win_ecog_change":
        return ("range", -5.0, 5.0)
    if col == "tumor_purity":
        return ("range", 0.0, 100.0)
    return ("range", 0.0, np.inf)                       # counts, tumour markers, TMB, MSI, days


class RiskPredictor:
    """A LandmarkModel plus what it needs in production: the training profile used to check new inputs,
    the settings, versions and a fingerprint of the training data."""

    def __init__(self, learner="xgb", feature_set="landmark", params=None, seeds=None):
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}")
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.learner, self.feature_set = learner, feature_set
        self.params = dict(params) if params is not None else chosen_settings(learner, feature_set)
        if seeds is None:
            seeds = tuple(json.loads(resources.files("chordsurv").joinpath("settings.json").read_text())["nn_seeds"]) \
                if learner == "nn" else (SEED,)
        self.seeds = tuple(seeds)

    # ------------------------------------------------------------------ training
    def fit(self, df, groups):
        """df: one row per patient alive at the landmark, with `time` (months from the landmark), `event` and the
        feature columns (see `chordsurv build`). groups: the feature groups (feature_groups.json)."""
        for c in ("time", "event", "cancer_type"):
            if c not in df:
                raise InputError(f"training table needs the column '{c}'")
        if (df.time <= 0).any():
            raise InputError("training patients must be alive after the landmark (time > 0)")
        self.model_ = LandmarkModel(self.learner, self.feature_set, self.params, self.seeds).fit(df, groups)
        cols = [c for c in self.model_.columns_ if c in df]
        num = df[[c for c in cols if c not in CATEGORICAL]].apply(pd.to_numeric, errors="coerce")
        self.profile_ = {
            "columns": list(self.model_.columns_),
            "cancer_types": sorted(df.cancer_type.dropna().unique()),
            "categories": {c: sorted(map(str, df[c].dropna().unique())) for c in cols if c in CATEGORICAL},
            "min": num.min().to_dict(), "max": num.max().to_dict(),
        }
        fingerprint = pd.util.hash_pandas_object(df[["time", "event"]].sort_index(), index=True).values.tobytes()
        died = (df.time <= HORIZON_MONTHS) & (df.event == 1)
        self.meta_ = {
            "learner": self.learner, "feature_set": self.feature_set, "params": self.params, "seeds": self.seeds,
            "landmark_months": LANDMARK_MONTHS, "horizon_months": HORIZON_MONTHS,
            "n_train": len(df), "deaths_within_horizon": int(died.sum()),
            "eras": sorted(df.era.unique()) if "era" in df else None,
            "data_fingerprint": hashlib.sha256(fingerprint).hexdigest()[:16],
            "versions": {p: _version(p) for p in ("chordsurv", *PACKAGES)},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return self

    # ------------------------------------------------------------------ input checks
    def check(self, df):
        """All problems of an input table, without predicting."""
        issues = []
        missing = [c for c in self.profile_["columns"] if c not in df]
        if missing:
            issues.append(Issue("error", ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""),
                                f"{len(missing)} feature columns are missing", []))
            return issues
        pos = lambda mask: list(np.flatnonzero(np.asarray(mask)))

        ct = df.cancer_type
        bad = ~ct.isin(self.profile_["cancer_types"])
        if bad.any():
            issues.append(Issue("error", "cancer_type", "the model only covers " + ", ".join(
                self.profile_["cancer_types"]), pos(bad)))
        # outcome columns are never used as inputs; if present, they can still show a patient is not at the landmark
        if "time" in df:
            dead = pd.to_numeric(df.time, errors="coerce") <= 0
            if dead.any():
                issues.append(Issue("error", "time", "patient not alive at the 6-month landmark", pos(dead)))
        if "os_months" in df:
            early = pd.to_numeric(df.os_months, errors="coerce") < LANDMARK_MONTHS
            if early.any():
                issues.append(Issue("error", "os_months", "follow-up ends before the 6-month landmark", pos(early)))

        for c in self.profile_["columns"]:
            if c in CATEGORICAL:
                if c != "cancer_type":
                    seen = set(self.profile_["categories"].get(c, []))
                    new = df[c].notna() & ~df[c].astype(str).isin(seen)
                    if new.any():
                        issues.append(Issue("warning", c, "category not seen in training (treated as a rare "
                                            "category)", pos(new)))
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            notnum = df[c].notna() & x.isna()
            if notnum.any():
                issues.append(Issue("error", c, "not a number", pos(notnum)))
            rule = _rule(c)
            if rule and rule[0] == "binary":
                wrong = x.notna() & ~x.isin([0, 1])
                if wrong.any():
                    issues.append(Issue("error", c, "must be 0 or 1", pos(wrong)))
                continue                                 # 0/1 columns cannot be "outside the range" in a useful way
            if rule:
                wrong = x.notna() & ((x < rule[1]) | (x > rule[2]))
                if wrong.any():
                    issues.append(Issue("error", c, f"impossible value (allowed {rule[1]:g} to {rule[2]:g})",
                                        pos(wrong)))
                if rule[2] <= 100:                       # bounded scales (probabilities, ECOG, purity)
                    continue
            lo, hi = self.profile_["min"].get(c), self.profile_["max"].get(c)
            if lo is not None and not np.isnan(lo):
                outside = x.notna() & ((x < lo) | (x > hi))
                if outside.any():
                    issues.append(Issue("warning", c, f"outside the training range [{lo:g}, {hi:g}]", pos(outside)))
        if self.feature_set != "day0":
            cols = [c for c in MONTHS_0_6 if c in df]
            if cols:
                blind = df[cols].isna().all(axis=1)
                if blind.any():
                    issues.append(Issue("warning", ", ".join(cols), "no information from months 0-6: the risk "
                                        "rests on day-0 data", pos(blind)))
        return issues

    # ------------------------------------------------------------------ prediction
    def predict(self, df, on_error="raise"):
        """Risk of death within 12 months after the landmark, one row per patient.
        on_error="raise": stop on any row error; "drop": return NaN for those rows and predict the rest."""
        issues = self.check(df)
        table_errors = [i for i in issues if i.level == "error" and not i.rows]
        row_errors = [i for i in issues if i.level == "error" and i.rows]
        if table_errors or (row_errors and on_error == "raise"):
            raise InputError("; ".join(f"{i.column}: {i.message}" + (f" ({len(i.rows)} rows)" if i.rows else "")
                                       for i in table_errors + row_errors))
        bad = sorted({r for i in row_errors for r in i.rows})
        for i in issues:
            if i.level == "warning":
                warnings.warn(f"{i.column}: {i.message} ({len(i.rows)} patients)", InputWarning, stacklevel=2)
        flags = [[] for _ in range(len(df))]
        for i in issues:
            for r in i.rows:
                flags[r].append(f"{i.level}: {i.column}")
        risk = np.full(len(df), np.nan)
        ok = np.setdiff1d(np.arange(len(df)), bad)
        if len(ok):
            risk[ok] = self.model_.predict_risk(df.iloc[ok])
        return pd.DataFrame({"risk_12m": risk, "issues": ["; ".join(f) for f in flags]}, index=df.index)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path):
        """Only load files you created yourself: pickle can run code."""
        with open(path, "rb") as f:
            m = pickle.load(f)
        now = {p: _version(p) for p in PACKAGES}
        changed = {p: (v, now[p]) for p, v in m.meta_["versions"].items() if p in now and v and now[p] and v != now[p]}
        if changed:
            warnings.warn(f"package versions differ from training: {changed}; predictions may differ slightly",
                          InputWarning, stacklevel=2)
        return m

    def card(self):
        """Short description of the fitted model."""
        m = self.meta_
        return (f"{m['learner']} on '{m['feature_set']}' features | trained on {m['n_train']:,} patients "
                f"({m['deaths_within_horizon']:,} deaths within {m['horizon_months']} months), eras {m['eras']} | "
                f"data fingerprint {m['data_fingerprint']} | created {m['created']}")

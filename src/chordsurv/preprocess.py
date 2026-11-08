"""Fold-safe preprocessing: everything learned from data is learned in fit() from the training rows only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, GENOMIC_PREFIXES

UNKNOWN_SMOKING = "never_smoker"


class Preprocessor(BaseEstimator, TransformerMixin):
    """DataFrame (from data.prepare) -> float32 matrix.

    - genomic indicators: kept if observed in >= `min_observed` of training rows and altered in >= `min_prevalence`;
      a gene that was not tested (NaN) is filled with its training prevalence, without an indicator, because
      untested = older panel = era, and era must not become a predictor;
    - other numeric columns: training median + a missing-value indicator ("not measured" is informative);
    - columns in `no_indicator` (smoking): filled with the training prevalence within the cancer type and no
      indicator, because their missingness depends on follow-up after the landmark (REVIEW_LOG 3.1);
    - categorical: one-hot, categories with < `min_category` training rows pooled, unseen categories -> pooled;
    - constant columns dropped; optional standardisation.
    """

    def __init__(self, categorical=tuple(CATEGORICAL), no_indicator=(UNKNOWN_SMOKING,), min_prevalence=0.01,
                 min_observed=0.10, min_category=50, scale=True):
        self.categorical = categorical
        self.no_indicator = no_indicator
        self.min_prevalence = min_prevalence
        self.min_observed = min_observed
        self.min_category = min_category
        self.scale = scale

    def fit(self, X, y=None):
        self.columns_in_ = list(X.columns)
        cat = [c for c in self.categorical if c in X]
        gen = [c for c in X if c.startswith(GENOMIC_PREFIXES)]
        num = [c for c in X if c not in cat and c not in gen]

        observed, prevalence = X[gen].notna().mean(), X[gen].mean()
        keep = (observed >= self.min_observed) & prevalence.between(self.min_prevalence, 1 - self.min_prevalence)
        self.genomic_ = list(prevalence.index[keep])
        self.genomic_fill_ = prevalence[self.genomic_]

        self.numeric_ = [c for c in num if X[c].nunique() > 1]
        self.indicators_ = [c for c in self.numeric_ if X[c].isna().any() and c not in self.no_indicator]
        self.medians_ = X[self.numeric_].median()
        self.group_fill_ = {c: X.groupby("cancer_type")[c].mean() for c in self.no_indicator if c in self.numeric_}

        self.categorical_ = cat
        self.encoder_ = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=self.min_category,
                                      sparse_output=False).fit(self._cat(X))

        Z, names = self._assemble(X)
        sd = Z.std(axis=0)
        self.keep_ = sd > 0
        self.feature_names_ = list(np.array(names)[self.keep_])
        self.mean_ = Z[:, self.keep_].mean(axis=0) if self.scale else 0.0
        self.sd_ = sd[self.keep_] if self.scale else 1.0
        return self

    def _cat(self, X):
        return X[self.categorical_].astype(object).where(X[self.categorical_].notna(), "missing").astype(str)

    def _assemble(self, X):
        num = X[self.numeric_].astype(float)
        ind = num[self.indicators_].isna().astype(float).add_suffix("_missing")
        for c, fill in self.group_fill_.items():
            num[c] = num[c].fillna(X.cancer_type.map(fill)).fillna(fill.mean())
        num = num.fillna(self.medians_)
        gen = X[self.genomic_].astype(float).fillna(self.genomic_fill_)
        cat = self.encoder_.transform(self._cat(X))
        names = (list(num.columns) + list(ind.columns) + self.genomic_
                 + list(self.encoder_.get_feature_names_out()))
        return np.hstack([num.to_numpy(), ind.to_numpy(), gen.to_numpy(), cat]), names

    def transform(self, X):
        missing = [c for c in self.columns_in_ if c not in X]
        if missing:
            raise ValueError(f"missing input columns: {missing[:5]}")
        Z = self._assemble(X)[0][:, self.keep_]
        return ((Z - self.mean_) / self.sd_).astype(np.float32)


class TypeInteractions(BaseEstimator, TransformerMixin):
    """Adds feature x cancer-type columns (for the penalised Cox model): coefficient for type k = shared effect +
    type-specific deviation, and the penalty shrinks the deviations towards zero unless the data support them.
    An interaction is created only if the feature varies within that type (>= `min_count` rows off its mode)."""

    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, Z, cancer_type, names):
        self.types_ = sorted(cancer_type.unique())
        base = [j for j, n in enumerate(names) if not n.startswith("cancer_type")]
        self.pairs_ = []
        for k in self.types_:
            rows = (cancer_type == k).to_numpy()
            for j in base:
                col = Z[rows, j]
                vals, counts = np.unique(col, return_counts=True)
                if rows.sum() - counts.max() >= self.min_count:
                    self.pairs_.append((j, k))
        self.feature_names_ = list(names) + [f"{names[j]}:{k}" for j, k in self.pairs_]
        inter = self._inter(Z, cancer_type)
        self.mean_, self.sd_ = inter.mean(axis=0), inter.std(axis=0)
        return self

    def _inter(self, Z, cancer_type):
        out = np.empty((Z.shape[0], len(self.pairs_)), dtype=np.float32)
        for i, (j, k) in enumerate(self.pairs_):
            out[:, i] = Z[:, j] * (cancer_type == k).to_numpy()
        return out

    def transform(self, Z, cancer_type):
        inter = (self._inter(Z, cancer_type) - self.mean_) / self.sd_
        return np.hstack([Z, inter.astype(np.float32)])


class SmokingExpander:
    """Fractional imputation of unknown smoking for flexible models (trees, neural nets).
    A median or mean fill would be a value that trees can isolate, and they would then learn "unknown",
    which partly encodes survival after the landmark. Instead, a patient with unknown smoking becomes two rows,
    never-smoker with weight p and smoker with weight 1 - p (p = training share of never-smokers in the cancer
    type). Training uses the weights, and predictions are averaged with them."""

    def __init__(self, column=UNKNOWN_SMOKING):
        self.column = column

    def fit(self, X):
        self.p_ = X.groupby("cancer_type")[self.column].mean()
        self.p_all_ = X[self.column].mean()
        return self

    def expand(self, X):
        """Returns the expanded frame, the row weights and the position of each row's patient in X."""
        pos = np.arange(len(X))
        unknown = X[self.column].isna().to_numpy()
        p = X.cancer_type.map(self.p_).fillna(self.p_all_).to_numpy()
        never, smoker = X.iloc[unknown].copy(), X.iloc[unknown].copy()
        never[self.column], smoker[self.column] = 1.0, 0.0
        out = pd.concat([X.iloc[~unknown], never, smoker])
        weight = np.concatenate([np.ones((~unknown).sum()), p[unknown], 1 - p[unknown]])
        rows = np.concatenate([pos[~unknown], pos[unknown], pos[unknown]])
        return out, weight, rows


def collapse(values, weight, rows, n):
    """Weighted average of per-row predictions back to one value per patient."""
    return np.bincount(rows, weights=values * weight, minlength=n) / np.bincount(rows, weights=weight, minlength=n)

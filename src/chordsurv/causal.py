"""Target-trial emulation: immunotherapy-containing first regimen vs chemotherapy alone in stage 4 NSCLC.
Stabilised inverse-probability-of-treatment weights from a logistic propensity model, weighted Kaplan-Meier
survival at 24 months from treatment start, weighted Cox hazard ratio, bootstrap and E-value."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter

from .survival import km_at

HORIZON = 24.0


def calendar_proxy(index):
    """MSK patient IDs (the letter P and 7 digits) are assigned in order of registration, so the number is a rough proxy for
    calendar time, which the de-identified dates do not give. In units of 10,000 registrations."""
    return index.str.extract(r"P-(\d+)", expand=False).astype(float).to_numpy() / 10_000


def confounders(d, pdl1=True, calendar=False):
    """Factors known at treatment start that plausibly drive both the choice and survival (Part I notebook, §3).
    calendar=True adds the patient-ID calendar proxy (linear + quadratic) for a sensitivity analysis."""
    Z = pd.DataFrame(index=d.index)
    Z["era_505"] = d.era.eq("IMPACT505").astype(float)
    Z["squamous"] = d.cancer_type_detailed.eq("Lung Squamous Cell Carcinoma").astype(float)
    Z["adenocarcinoma"] = d.cancer_type_detailed.eq("Lung Adenocarcinoma").astype(float)
    Z["male"] = d.sex.eq("Male").astype(float)
    Z["white"] = d.race.eq("White").astype(float)
    Z["never_smoker"] = d.never_smoker.fillna(d.never_smoker.mean())          # imputed without an indicator
    Z["ecog_0"] = d.tz_ecog.eq(0).astype(float)
    Z["ecog_2plus"] = d.tz_ecog.ge(2).astype(float)
    Z["ecog_missing"] = d.tz_ecog.isna().astype(float)
    if pdl1:
        Z["pdl1_positive"] = d.tz_pdl1_positive.eq(1).astype(float)
        Z["pdl1_not_recorded"] = d.tz_pdl1_positive.isna().astype(float)
    for s in ["cns_brain", "liver", "bone"]:
        Z[f"site_{s}"] = d[f"tz_site_{s}"]
    Z["n_sites"] = d.tz_n_sites
    Z["log_tmb"] = np.log1p(d.tmb)
    for g in ["KRAS", "STK11", "KEAP1", "TP53"]:
        Z[f"mut_{g}"] = d[f"mut_{g}"].fillna(0)
    Z["log_days_dx_to_seq"] = np.log1p(d.dx_to_seq_days.clip(lower=0).fillna(d.dx_to_seq_days.median()))
    if calendar:
        cal = calendar_proxy(d.index)
        Z["calendar"] = cal - cal.mean()
        Z["calendar_sq"] = Z.calendar ** 2
    return Z


def propensity(Z, A):
    """Logistic regression by Newton-Raphson (statsmodels). Returns the scores and convergence diagnostics."""
    X = sm.add_constant(Z, has_constant="add")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.Logit(A, X).fit(method="newton", maxiter=100, tol=1e-10, disp=0)
    score = X.T @ (A - fit.predict(X))                           # gradient of the log-likelihood at the estimate
    info = {"converged": bool(fit.mle_retvals["converged"]), "iterations": int(fit.mle_retvals["iterations"]),
            "max_abs_gradient": float(np.abs(score).max()), "warnings": [str(w.message) for w in caught]}
    return fit.predict(X).to_numpy(), fit, info


def weights(A, ps, kind="ate"):
    """ate: stabilised IPTW, P(A=a)/P(A=a|x); overlap: 1-ps for treated, ps for controls (ATO)."""
    A = np.asarray(A)
    if kind == "ate":
        return np.where(A == 1, A.mean() / ps, (1 - A.mean()) / (1 - ps))
    return np.where(A == 1, 1 - ps, ps)


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def smd(Z, A, w=None):
    A = np.asarray(A)
    w = np.ones(len(A)) if w is None else np.asarray(w)
    out = {}
    for c in Z:
        x = Z[c].to_numpy(float)
        m1, m0 = np.average(x[A == 1], weights=w[A == 1]), np.average(x[A == 0], weights=w[A == 0])
        v1 = np.average((x[A == 1] - m1) ** 2, weights=w[A == 1])
        v0 = np.average((x[A == 0] - m0) ** 2, weights=w[A == 0])
        out[c] = (m1 - m0) / np.sqrt((v1 + v0) / 2) if v1 + v0 > 0 else 0.0
    return pd.Series(out)


def outcome(time, event, A, w, horizon=HORIZON):
    """Weighted S(24) by arm, their difference, and the weighted Cox HR over the first 24 months (robust SE)."""
    time, event, A = np.asarray(time, float), np.asarray(event, int), np.asarray(A)
    s1, s0 = km_at(time[A == 1], event[A == 1], horizon, w[A == 1]), km_at(time[A == 0], event[A == 0], horizon,
                                                                            w[A == 0])
    t, d = np.minimum(time, horizon), ((event == 1) & (time <= horizon)).astype(int)
    df = pd.DataFrame({"t": t, "d": d, "a": A, "w": w})
    cph = CoxPHFitter().fit(df, "t", "d", weights_col="w", robust=True)
    return {"S1": s1["S"], "S0": s0["S"], "diff": s1["S"] - s0["S"], "hr": float(cph.hazard_ratios_["a"]),
            "log_hr": float(cph.params_["a"]), "log_hr_se_robust": float(cph.standard_errors_["a"]),
            "cox_converged": True}


def estimate(E, kind="ate", pdl1=True, trim=None, calendar=False):
    Z, A = confounders(E, pdl1=pdl1, calendar=calendar), E.treated.to_numpy()
    ps, fit, info = propensity(Z, A)
    w = weights(A, ps, kind)
    if trim is not None:                                          # truncate extreme weights at percentiles
        lo, hi = np.percentile(w, [trim, 100 - trim])
        w = np.clip(w, lo, hi)
    res = outcome(E.time, E.event, A, w)
    res.update({"n_treated": int(A.sum()), "n_control": int((1 - A).sum()), "ess_treated": ess(w[A == 1]),
                "ess_control": ess(w[A == 0]), "w_min": float(w.min()), "w_max": float(w.max()),
                "ps_min": float(ps.min()), "ps_max": float(ps.max()),
                "share_ps_outside_05_95": float(((ps < 0.05) | (ps > 0.95)).mean()),
                "max_abs_smd_weighted": float(smd(Z, A, w).abs().max()),
                "max_abs_smd_unweighted": float(smd(Z, A).abs().max()), "propensity": info})
    return res, ps, w, fit


def bootstrap(E, B=1000, seed=42, **kw):
    """Patient-level bootstrap that re-estimates the propensity model in every replicate."""
    rng = np.random.default_rng(seed)
    out, failed = [], 0
    for _ in range(B):
        idx = rng.integers(0, len(E), len(E))
        try:
            r = estimate(E.iloc[idx], **kw)[0]
            out.append({k: r[k] for k in ("S1", "S0", "diff", "hr", "log_hr")})
        except Exception:                                         # e.g. a replicate with perfect separation
            failed += 1
    return pd.DataFrame(out), failed


def e_value(hr, ci_limit=None):
    """E-value for a hazard ratio with a common outcome (VanderWeele & Ding 2017):
    RR ~ (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)), then E = RR + sqrt(RR (RR - 1)) on the side away from 1."""
    def ev(h):
        rr = (1 - 0.5 ** np.sqrt(h)) / (1 - 0.5 ** np.sqrt(1 / h))
        rr = 1 / rr if rr < 1 else rr
        return float(rr + np.sqrt(rr * (rr - 1)))
    point = ev(hr)
    if ci_limit is None:
        return point, None
    crosses = (hr < 1 and ci_limit >= 1) or (hr > 1 and ci_limit <= 1)
    return point, 1.0 if crosses else ev(ci_limit)

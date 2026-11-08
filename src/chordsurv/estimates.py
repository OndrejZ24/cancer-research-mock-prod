"""The inferential (D1) and causal (C1) estimates as functions: re-run them on a new data release."""
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import causal as C
from .config import DAYS_PER_MONTH, PROCESSED_DIR
from .survival import kaplan_meier, km_at

D1_MONTHS = 24.0
CAUSAL_GENES = [f"mut_{g}" for g in ("KRAS", "STK11", "KEAP1", "TP53")]


def median_survival(t, e):
    uniq, S, _ = kaplan_meier(t, e)
    below = np.flatnonzero(S <= 0.5)
    return float(uniq[below[0]]) if len(below) else None


def survival_by_type(patients, months=D1_MONTHS, B=1000, seed=42):
    """S_c(months) per cancer type: Kaplan-Meier, Greenwood SE, log(-log) CI, a patient bootstrap SE and the
    era-standardised estimate. Returns (table, era table, pairwise differences, log-rank result)."""
    from lifelines.statistics import multivariate_logrank_test
    p = patients[patients.in_study]                              # drops patients without a solid-tumour panel
    rng = np.random.default_rng(seed)
    rows, era_rows = [], []
    for ct, g in p.groupby("cancer_type"):
        t, e = g.os_months.to_numpy(float), g.event.to_numpy(int)
        km = km_at(t, e, months)
        boot = [km_at(t[i], e[i], months)["S"] for i in rng.integers(0, len(g), (B, len(g)))] if B else [np.nan]
        std_S, std_var = 0.0, 0.0                                # weights = the type's share of patients by era
        for era, ge in g.groupby("era"):
            te, ee = ge.os_months.to_numpy(float), ge.event.to_numpy(int)
            k = km_at(te, ee, months)
            w = len(ge) / len(g)
            std_S, std_var = std_S + w * k["S"], std_var + w ** 2 * k["se"] ** 2
            era_rows.append({"cancer_type": ct, "era": era, "n": len(ge), "deaths": int(ee.sum()),
                             "at_risk_24": int((te >= months).sum()), "weight": w, **k})
        rows.append({"cancer_type": ct, "n": len(g), "deaths": int(e.sum()), "at_risk_24": int((t >= months).sum()),
                     "censored_before_24": int(((t < months) & (e == 0)).sum()), **km,
                     "se_bootstrap": float(np.std(boot, ddof=1)) if B else np.nan,
                     "S_era_standardised": std_S, "se_era_standardised": float(np.sqrt(std_var)),
                     "median_months": median_survival(t, e)})
    table = pd.DataFrame(rows).set_index("cancer_type")
    diffs = []
    for a, b in combinations(table.index, 2):
        d = table.loc[a, "S"] - table.loc[b, "S"]
        se = np.hypot(table.loc[a, "se"], table.loc[b, "se"])
        diffs.append({"a": a, "b": b, "diff": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se})
    lr = multivariate_logrank_test(p.os_months, p.cancer_type, p.event)
    return table, pd.DataFrame(era_rows), pd.DataFrame(diffs), lr


def causal_cohort_table(processed_dir=PROCESSED_DIR):
    d = Path(processed_dir)
    E = pd.read_parquet(d / "causal.parquet").join(pd.read_parquet(d / "genomics.parquet", columns=CAUSAL_GENES))
    return E[E.eligible].copy()


def grace_period(E, days):
    """Strategy = immunotherapy started within `days` of the first treatment (pembrolizumab is often added at
    cycle 2, once PD-L1 is known). Time zero moves to that day, among patients alive then, so that nobody is
    assigned to an arm by having survived long enough to receive the drug (no immortal time)."""
    G = E[E.time > days / DAYS_PER_MONTH].copy()
    G["treated"] = (G.days_to_immuno <= days).astype(int)
    G["time"] = G.time - days / DAYS_PER_MONTH
    return G


def causal_variants(E):
    """The primary analysis and the sensitivity analyses: name -> (data, estimate() arguments)."""
    return {
        "primary_iptw": (E, dict(kind="ate")),
        "without_pdl1": (E, dict(kind="ate", pdl1=False)),
        "weights_truncated_1_99": (E, dict(kind="ate", trim=1)),
        "overlap_weights": (E, dict(kind="overlap")),
        "immuno_plus_chemo_only": (E[(E.treated == 0) | E.immuno_with_chemo], dict(kind="ate")),
        "grace_period_42d": (grace_period(E, 42), dict(kind="ate")),
        "calendar_time_proxy": (E, dict(kind="ate", calendar=True)),
        "calendar_time_proxy_overlap": (E, dict(kind="overlap", calendar=True)),
    }


def treatment_effect(d, B=1000, seed=42, **kw):
    """Point estimate + bootstrap (propensity re-fitted in each replicate) + E-value for one analysis."""
    t0 = time.time()
    res, ps, w, fit = C.estimate(d, **kw)
    boot, failed = C.bootstrap(d, B=B, seed=seed, **kw)
    q = lambda c: [float(boot[c].quantile(0.025)), float(boot[c].quantile(0.975))]
    hr_ci = [float(np.exp(boot.log_hr.quantile(0.025))), float(np.exp(boot.log_hr.quantile(0.975)))]
    e_point, e_ci = C.e_value(res["hr"], hr_ci[1] if res["hr"] < 1 else hr_ci[0])
    res.update({"bootstrap_reps": len(boot), "bootstrap_failed": failed,
                "diff_se_boot": float(boot["diff"].std()), "diff_ci_boot": q("diff"),
                "S1_ci_boot": q("S1"), "S0_ci_boot": q("S0"),
                "log_hr_se_boot": float(boot.log_hr.std()), "hr_ci_boot": hr_ci,
                "hr_ci_robust": [float(np.exp(res["log_hr"] - 1.96 * res["log_hr_se_robust"])),
                                 float(np.exp(res["log_hr"] + 1.96 * res["log_hr_se_robust"]))],
                "e_value_point": e_point, "e_value_ci": e_ci, "seconds": round(time.time() - t0, 1)})
    return res, w, fit


def km_curves(d, w, max_month=36):
    out = []
    for arm, label in ((1, "immunotherapy"), (0, "chemotherapy")):
        m = d.treated.to_numpy() == arm
        for wt, wname in ((w[m], "weighted"), (np.ones(m.sum()), "crude")):
            uniq, S, _ = kaplan_meier(d.time.to_numpy(float)[m], d.event.to_numpy(int)[m], wt)
            keep = uniq <= max_month
            out += [{"arm": label, "weights": wname, "month": u, "S": s} for u, s in zip(uniq[keep], S[keep])]
    return out

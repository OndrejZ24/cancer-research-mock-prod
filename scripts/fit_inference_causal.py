"""Part II, description/inference (D1) and causal (C1) questions (same as `chordsurv describe` / `effect`).

Usage:  python scripts/fit_inference_causal.py [--boot 1000] [--only inference|causal]
Writes aggregate results to results/part2/inference/ and results/part2/causal/."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chordsurv import causal as C                                          # noqa: E402
from chordsurv import estimates as EST                                     # noqa: E402
from chordsurv.config import PROCESSED_DIR, RESULTS_DIR                    # noqa: E402


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return o


def inference(B, seed=42):
    out = RESULTS_DIR / "part2" / "inference"
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PROCESSED_DIR / "patients.parquet")
    table, eras, diffs, lr = EST.survival_by_type(p, B=B, seed=seed)
    table.to_csv(out / "s24_by_type.csv")
    eras.to_csv(out / "s24_by_type_era.csv", index=False)
    diffs.to_csv(out / "s24_differences.csv", index=False)
    summary = {"estimand": "S_c(24) = P(alive 24 months after first sequencing), MSK-sequenced patients",
               "estimator": "Kaplan-Meier; Greenwood SE; 95 % log(-log) CI; era-standardised KM as alternative",
               "n": int(p.in_study.sum()), "bootstrap_reps": B, "logrank_chi2": float(lr.test_statistic),
               "logrank_df": int(lr.degrees_of_freedom), "logrank_p": float(lr.p_value),
               "max_abs_pooled_minus_standardised_pp": float(100 * (table.S - table.S_era_standardised).abs().max())}
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    print(table[["n", "S", "lo", "hi", "se", "se_bootstrap", "S_era_standardised"]].round(4))
    print(f"log-rank chi2 {lr.test_statistic:.0f}, p = {lr.p_value:.1e}")


def causal(B, seed=42):
    out = RESULTS_DIR / "part2" / "causal"
    out.mkdir(parents=True, exist_ok=True)
    E = EST.causal_cohort_table()
    summary, curves = {}, []
    for name, (d, kw) in EST.causal_variants(E).items():
        res, w, fit = EST.treatment_effect(d, B=B, seed=seed, **kw)
        summary[name] = res
        if name == "primary_iptw":
            Z = C.confounders(d)
            pd.DataFrame({"unweighted": C.smd(Z, d.treated), "weighted": C.smd(Z, d.treated, w)}).to_csv(
                out / "balance_smd.csv")
            pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues}).to_csv(
                out / "propensity_coefficients.csv")
            summary["crude_unweighted"] = C.outcome(d.time, d.event, d.treated.to_numpy(), np.ones(len(d)))
            curves = EST.km_curves(d, w)
        hr_ci = res["hr_ci_boot"]
        print(f"{name:26s} S1 {res['S1']:.3f} S0 {res['S0']:.3f} diff {res['diff']:+.3f} "
              f"[{res['diff_ci_boot'][0]:+.3f}, {res['diff_ci_boot'][1]:+.3f}] HR {res['hr']:.2f} "
              f"[{hr_ci[0]:.2f}, {hr_ci[1]:.2f}] ESS {res['ess_treated']:.0f}/{res['ess_control']:.0f} "
              f"E-value {res['e_value_point']:.2f} ({res['seconds']}s)", flush=True)
    pd.DataFrame(curves).to_csv(out / "weighted_km_curves.csv", index=False)
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--boot", type=int, default=1000)
    a.add_argument("--only", choices=["inference", "causal"])
    args = a.parse_args()
    if args.only != "causal":
        inference(args.boot)
    if args.only != "inference":
        causal(args.boot)


if __name__ == "__main__":
    main()

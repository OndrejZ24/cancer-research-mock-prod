"""Command line: chordsurv {build, train, predict, describe, effect, demo}. Run `chordsurv <command> -h`."""
import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR


def _read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)


def _landmark_table(processed_dir):
    d = Path(processed_dir)
    return pd.read_parquet(d / "landmark.parquet").join(pd.read_parquet(d / "genomics.parquet"))


def cmd_build(a):
    from .pipeline import build
    build(a.raw_dir, a.processed_dir)


def cmd_train(a):
    from .api import RiskPredictor
    df = _landmark_table(a.processed_dir)
    if a.eras == "training":
        df = df[~df.test_era]                  # the newest era stays untouched for the final evaluation
    groups = json.loads((Path(a.processed_dir) / "feature_groups.json").read_text())
    model = RiskPredictor(a.learner, a.features).fit(df, groups)
    model.save(a.out)
    print(model.card())
    print(f"saved to {a.out}")


def cmd_predict(a):
    from .api import RiskPredictor
    model = RiskPredictor.load(a.model)
    df = _read_table(a.input) if a.input else _landmark_table(a.processed_dir)
    if a.patients:
        ids = [x.strip() for x in Path(a.patients).read_text().split() if x.strip()]
        df = df.loc[df.index.intersection(ids)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = model.predict(df, on_error=a.on_error)
    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)
    out.to_csv(a.out)
    print(f"{out.risk_12m.notna().sum():,} risks written to {a.out} "
          f"({out.risk_12m.isna().sum():,} refused, {(out.issues != '').sum():,} with issues)")


def cmd_describe(a):
    from .estimates import survival_by_type
    p = pd.read_parquet(Path(a.processed_dir) / "patients.parquet")
    table, _, _, lr = survival_by_type(p, months=a.months, B=a.boot)
    cols = ["n", "deaths", "S", "lo", "hi", "se", "se_bootstrap", "S_era_standardised"]
    print(table[cols].round(4).to_string())
    print(f"log-rank chi2 = {lr.test_statistic:.1f} on {lr.degrees_of_freedom} df")
    if a.out:
        table.to_csv(a.out)


def cmd_effect(a):
    from .estimates import causal_cohort_table, causal_variants, treatment_effect
    variants = causal_variants(causal_cohort_table(a.processed_dir))
    names = list(variants) if a.analysis == "all" else [a.analysis]
    for name in names:
        d, kw = variants[name]
        r, _, _ = treatment_effect(d, B=a.boot, **kw)
        print(f"{name:28s} S(24) {r['S1']:.1%} vs {r['S0']:.1%}, difference {r['diff']:+.1%} "
              f"[{r['diff_ci_boot'][0]:+.1%}, {r['diff_ci_boot'][1]:+.1%}], HR {r['hr']:.2f} "
              f"[{r['hr_ci_boot'][0]:.2f}, {r['hr_ci_boot'][1]:.2f}], ESS {r['ess_treated']:.0f}/{r['ess_control']:.0f}")


def cmd_demo(a):
    """End to end on synthetic data (no licensed data needed): train, predict, and refuse bad inputs."""
    from .api import InputError, RiskPredictor
    from .synthetic import make_landmark_like
    df, groups = make_landmark_like(3000, seed=1)
    train, new = df.iloc[:2500], df.iloc[2500:2506].drop(columns=["time", "event"]).copy()
    model = RiskPredictor(a.learner, "landmark", params=_demo_params(a.learner)).fit(train, groups)
    print(model.card(), "\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        print(model.predict(new).round(3).to_string())
    for w in caught:
        print(f"warning: {w.message}")
    print()
    new.iloc[0, new.columns.get_loc("lm_ecog")] = 7                          # impossible
    new.iloc[1, new.columns.get_loc("cancer_type")] = "Melanoma"              # not covered
    new.iloc[2, new.columns.get_loc("tmb")] = 900.0                           # far outside the training range
    try:
        model.predict(new)
    except InputError as e:
        print(f"refused: {e}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(model.predict(new, on_error="drop").round(3).to_string())


def _demo_params(learner):
    # small, fast settings for synthetic data (the Part II settings were tuned on the real data)
    return {"km": {}, "coxnet": {"l1_ratio": 0.5, "alpha": 0.01, "alphas": [10 ** (-0.5 - 1.5 * i / 14) for i in range(15)]},
            "xgb": {"learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5, "subsample": 0.8,
                    "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
            "nn": {"width": 32, "depth": 1, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4}}[learner]


def main(argv=None):
    p = argparse.ArgumentParser(prog="chordsurv", description="12-month risk of death at the 6-month landmark "
                                "(MSK-CHORD), survival by cancer type, and immunotherapy vs chemotherapy.")
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="processed tables (default: %(default)s)")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="raw cBioPortal export -> processed tables")
    b.add_argument("--raw-dir", default=str(RAW_DIR), help="unpacked msk_chord_2024 folder (default: %(default)s)")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="fit a model with the settings chosen in Part II")
    t.add_argument("--learner", choices=["km", "coxnet", "xgb", "nn"], default="xgb")
    t.add_argument("--features", choices=["day0", "landmark", "landmark_counts", "landmark_no_pdl1"],
                   default="landmark")
    t.add_argument("--eras", choices=["training", "all"], default="training",
                   help="'training' keeps IMPACT505 out (for evaluation); 'all' after the evaluation")
    t.add_argument("--out", default="model.pkl")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("predict", help="12-month risk for patients at the landmark")
    r.add_argument("--model", default="model.pkl")
    r.add_argument("--input", help="table with the feature columns (.parquet or .csv, index = patient id); "
                   "default: the processed landmark table")
    r.add_argument("--patients", help="text file with patient ids to keep")
    r.add_argument("--on-error", choices=["raise", "drop"], default="raise")
    r.add_argument("--out", default="risks.csv")
    r.set_defaults(func=cmd_predict)

    d = sub.add_parser("describe", help="survival by cancer type (Kaplan-Meier with 95%% CIs)")
    d.add_argument("--months", type=float, default=24.0)
    d.add_argument("--boot", type=int, default=0, help="bootstrap replicates for a second SE (0 = skip)")
    d.add_argument("--out")
    d.set_defaults(func=cmd_describe)

    e = sub.add_parser("effect", help="immunotherapy vs chemotherapy in stage 4 NSCLC (IPTW)")
    e.add_argument("--analysis", default="primary_iptw", help="primary_iptw, without_pdl1, ..., or all")
    e.add_argument("--boot", type=int, default=1000)
    e.set_defaults(func=cmd_effect)

    m = sub.add_parser("demo", help="run everything on synthetic data (no licensed data needed)")
    m.add_argument("--learner", choices=["km", "coxnet", "xgb", "nn"], default="coxnet")
    m.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

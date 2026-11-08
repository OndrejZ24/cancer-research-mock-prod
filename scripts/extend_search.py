"""Part II: extend the search where the chosen value lies at the edge of its range (the assignment's rule).

Usage:  python scripts/extend_search.py {xgb,nn}      (separate processes: xgboost and torch cannot share one)
- XGBoost: the chosen learning rate on the landmark features (0.0115) is close to the lower bound (0.01); the chosen
  settings are refitted in the same 5 folds with smaller learning rates (more trees, higher round limit).
- Neural network: the chosen width (256) is the largest offered; the chosen settings are refitted with width 512.
Only if the CV Brier improves by more than one standard error would the chosen value change."""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chordsurv import tuning as T                                          # noqa: E402
from chordsurv.config import RESULTS_DIR                                    # noqa: E402
from chordsurv.data import load_groups, load_landmark, make_folds           # noqa: E402
from chordsurv.preprocess import collapse                                   # noqa: E402

warnings.filterwarnings("ignore", message="The least populated class")


def _task(d, cfg):
    from chordsurv.boosting import xgb_fit, xgb_risk
    m = xgb_fit(d["Za"], d["ta"], d["ea"], d["wa"], d["ra"], d["n_a"], cfg, max_rounds=20000, nthread=2)
    return collapse(xgb_risk(m, d["Zb"]), d["wb"], d["rb"], d["n_b"]), m["best_iteration"] + 1


def xgb_edge(fs="landmark", lower=(0.006, 0.003)):
    out = RESULTS_DIR / "part2" / "xgb"
    chosen = json.loads((out / "summary.json").read_text())["feature_sets"][fs]["params"]
    rates = (chosen["learning_rate"], *lower)              # the exact chosen value first: reproduces the search
    lm = load_landmark()
    df = lm[~lm.test_era].copy()
    folds, groups = make_folds(df), load_groups()
    designs = T.fold_designs(df, folds, "xgb", fs, groups)
    rows = []
    for lr in rates:
        cfg = {**chosen, "learning_rate": lr}
        res = Parallel(n_jobs=5)(delayed(_task)(d, cfg) for d in designs)
        for k, (d, (risk, trees)) in enumerate(zip(designs, res)):
            rows.append({"learning_rate": lr, "fold": k, "trees": trees, **T.score(df, d["va"], risk)})
        r = pd.DataFrame(rows)
        r = r[r.learning_rate == lr]
        print(f"learning rate {lr}: CV brier {r.brier.mean():.5f} ± {r.brier.std() / np.sqrt(5):.5f}, "
              f"auc {r.auc.mean():.4f}, trees {r.trees.mean():.0f} (max {r.trees.max()})", flush=True)
    rows = pd.DataFrame(rows)
    rows.to_csv(out / f"edge_extension_{fs}.csv", index=False)
    s = rows.groupby("learning_rate").agg(brier=("brier", "mean"), se=("brier", lambda x: x.std() / np.sqrt(len(x))),
                                          auc=("auc", "mean"), trees=("trees", "mean"))
    print(s)


def nn_edge(fs="landmark", widths=(256, 512)):
    out = RESULTS_DIR / "part2" / "nn"
    chosen = json.loads((out / "summary.json").read_text())["feature_sets"][fs]["params"]
    lm = load_landmark()
    df = lm[~lm.test_era].copy()
    folds, groups = make_folds(df), load_groups()
    search, rows, _, _ = T.cv_nn(df, folds, groups, fs, [{**chosen, "width": w} for w in widths], seeds=(42,))
    rows["width"] = rows.config.map(dict(enumerate(widths)))
    rows.to_csv(out / f"edge_extension_{fs}.csv", index=False)
    print(search.assign(width=search.config.map(dict(enumerate(widths))))[["width", "brier", "brier_se", "auc", "best_epoch"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("learner", choices=["xgb", "nn"])
    p.add_argument("--feature-set", default="landmark")
    args = p.parse_args()
    {"xgb": xgb_edge, "nn": nn_edge}[args.learner](fs=args.feature_set)


if __name__ == "__main__":
    main()

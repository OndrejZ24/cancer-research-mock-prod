"""Part II, prediction: tune, check and fit one learner on the training eras (IMPACT341/410/468) only.

Usage:  python scripts/train_prediction.py {km,coxnet,xgb,nn}

Each learner runs in its own process: xgboost and torch bring two OpenMP runtimes that cannot share one
(see src/chordsurv/boosting.py). The test era (IMPACT505) is dropped before anything else happens.
Writes aggregate results to results/part2/<learner>/ and models + out-of-fold risks to artifacts/part2/."""
import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chordsurv import tuning as T                                             # noqa: E402
from chordsurv.config import ARTIFACTS_DIR, RESULTS_DIR                         # noqa: E402
from chordsurv.data import SEED, load_groups, load_landmark, make_folds         # noqa: E402
from chordsurv.metrics import evaluate                                          # noqa: E402
from chordsurv.models import LandmarkModel, coxnet_fit, cox_kkt                 # noqa: E402

FEATURE_SETS = ["day0", "landmark", "landmark_counts"]   # tuned separately
SENSITIVITY = "landmark_no_pdl1"                         # landmark settings reused, PD-L1 removed
NN_SEEDS = (42, 43, 44, 45, 46)
N_XGB, N_NN = 40, 30

warnings.filterwarnings("ignore", message="The least populated class")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def cv_summary(rows):
    r = pd.DataFrame(rows)
    return {m: {"mean": float(r[m].mean()), "se": float(r[m].std() / np.sqrt(len(r)))}
            for m in ("brier", "auc", "cindex", "mean_risk")}


class Run:
    def __init__(self, learner, quick=False):
        self.learner = learner
        self.out = RESULTS_DIR / ("part2_quick" if quick else "part2") / learner
        self.art = ARTIFACTS_DIR / ("part2_quick" if quick else "part2")
        self.out.mkdir(parents=True, exist_ok=True)
        self.art.mkdir(parents=True, exist_ok=True)
        lm = load_landmark()
        self.df = lm[~lm.test_era].copy()                    # the test era never enters this script
        assert self.df.era.isin(["IMPACT341", "IMPACT410", "IMPACT468"]).all()
        self.groups = load_groups()
        self.folds = make_folds(self.df)
        self.summary = {"learner": learner, "n_train": len(self.df), "n_deaths_12m":
                        int(((self.df.time <= 12) & (self.df.event == 1)).sum()), "feature_sets": {}}
        self.t0 = time.time()

    def log(self, msg):
        print(f"[{time.time() - self.t0:7.1f}s] {self.learner}: {msg}", flush=True)

    def save_oof(self, fs, risk):
        pd.DataFrame({"risk": risk}, index=self.df.index).to_parquet(self.art / f"oof_{self.learner}_{fs}.parquet")

    def final_fit(self, fs, params, seeds=(SEED,), learner=None):
        learner = learner or self.learner
        m = LandmarkModel(learner, fs, params, seeds).fit(self.df, self.groups)
        with open(self.art / f"model_{learner}_{fs}.pkl", "wb") as f:
            pickle.dump(m, f)
        return m

    def finish_fs(self, fs, params, fold_rows, oof, extra=None, seeds=(SEED,)):
        T.by_type(self.df, oof).to_csv(self.out / f"by_type_{fs}.csv")
        pd.DataFrame(fold_rows).to_csv(self.out / f"folds_{fs}.csv", index=False)
        self.save_oof(fs, oof)
        fwd = T.forward_check(self.df, self.groups, self.learner, fs, params, seeds)
        self.summary["feature_sets"][fs] = {"params": params, "cv": cv_summary(fold_rows),
                                            "cv_pooled_oof": evaluate(self.df.time.to_numpy(float),
                                                                      self.df.event.to_numpy(int), oof),
                                            "forward_468": fwd, **(extra or {})}
        self.log(f"{fs}: CV brier {np.mean([r['brier'] for r in fold_rows]):.4f} | forward IMPACT468 "
                 f"brier {fwd['brier']:.4f} auc {fwd['auc']:.3f}")

    def write(self):
        self.summary["seconds"] = round(time.time() - self.t0, 1)
        (self.out / "summary.json").write_text(json.dumps(jsonable(self.summary), indent=1))
        self.log("summary saved")


# ---------------------------------------------------------------- learners

def run_km(run):
    rows, oof = T.cv_km(run.df, run.folds)
    run.finish_fs("km", {}, rows.to_dict("records"), oof)
    m = run.final_fit("km", {})
    run.summary["feature_sets"]["km"]["risk_12m_by_type"] = m.km_


def run_coxnet(run):
    for fs in FEATURE_SETS + [SENSITIVITY]:
        tuned = fs != SENSITIVITY
        if tuned:
            search, rows, oof, grids, warns, n_feat = T.cv_coxnet(run.df, run.folds, run.groups, fs,
                                                                  l1_grid=T.COXNET_L1)
            search.to_csv(run.out / f"search_{fs}.csv", index=False)
            ok = search[search.n_folds == len(run.folds)]
            best = ok.loc[ok.brier.idxmin()]
            # one-standard-error rule, reported for comparison: the largest penalty within 1 SE of the best
            within = ok[(ok.l1_ratio == best.l1_ratio) & (ok.brier <= best.brier + best.brier_se)]
            one_se = within.loc[within.alpha.idxmax()]
            l1, idx = float(best.l1_ratio), int(best.alpha_index)
            grid = grids[l1]
            params = {"l1_ratio": l1, "alpha": float(grid[idx]), "alphas": grid.tolist()}
            fold_rows = rows[(rows.l1_ratio == l1) & (rows.alpha_index == idx)].to_dict("records")
            oof_best = oof[l1][:, idx]
            extra = {"search_space": {"l1_ratio": list(grids), "alpha": "40 values, log scale, from alpha_max "
                                      "(all coefficients zero) to 0.002 alpha_max"},
                     "chosen_alpha_index": idx, "edge_of_grid": idx in (0, len(grid) - 1),
                     "nonzero_cv_mean": float(best.nonzero), "n_features_with_interactions": n_feat,
                     "one_se_rule": {"alpha": float(one_se.alpha), "nonzero": float(one_se.nonzero),
                                     "brier": float(one_se.brier)},
                     "cv_warnings": sorted(set(warns))}
            if fs == "landmark":
                run.landmark_params = params
        else:
            params = run.landmark_params
            oof_best, fold_rows = np.full(len(run.df), np.nan), []
            designs = T.fold_designs(run.df, run.folds, "coxnet", fs, run.groups)
            grid = np.asarray(params["alphas"])
            grid = grid[grid >= params["alpha"] * 0.999]
            for k, d in enumerate(designs):
                risks, _, _ = T._coxnet_fold(d, params["l1_ratio"], grid, 1e-9)
                r = risks[:, -1]
                oof_best[d["va"]] = r
                fold_rows.append({"fold": k, **T.score(run.df, d["va"], r)})
            extra = {"note": "landmark settings reused, PD-L1 columns removed"}

        m = run.final_fit(fs, params)
        Z, _, _ = m.design(run.df)
        t, e = run.df.time.to_numpy(float), run.df.event.to_numpy(int)
        beta = m.model_.coef_[:, m.alpha_index_]
        names = np.array(m.feature_names_)
        is_inter = np.array([":" in n for n in names])
        top = np.argsort(-np.abs(beta))[:20]
        extra.update({"final": {
            "kkt_violation_rel_tol1e-10": cox_kkt(m.model_, Z, t, e, m.alpha_index_),
            "nonzero": int((beta != 0).sum()),
            "nonzero_main": int(((beta != 0) & ~is_inter).sum()),
            "nonzero_interactions": int(((beta != 0) & is_inter).sum()),
            "path_length": len(m.model_.alphas_), "warnings": m.model_.warnings_,
            "top_coefficients": [(names[j], float(beta[j]), float(np.exp(beta[j]))) for j in top if beta[j] != 0]}})
        if fs == "landmark":                                   # what the default tolerance would have returned
            loose = coxnet_fit(Z, t, e, params["l1_ratio"], alphas=m.model_.alphas_, tol=1e-7)
            from chordsurv.models import coxnet_risk
            extra["final"]["default_tol_1e-7"] = {
                "kkt_violation_rel": cox_kkt(loose, Z, t, e, len(loose.alphas_) - 1),
                "max_abs_risk_difference": float(np.abs(coxnet_risk(loose, Z, params["alpha"])
                                                        - coxnet_risk(m.model_, Z, params["alpha"])).max()),
                "max_abs_coef_difference": float(np.abs(loose.coef_[:, -1] - beta).max())}
        run.finish_fs(fs, params, fold_rows, oof_best, extra)
        run.write()


def run_xgb(run):
    from chordsurv.boosting import gain_importance
    configs = T.sample_xgb(N_XGB)
    for fs in FEATURE_SETS + [SENSITIVITY]:
        if fs != SENSITIVITY:
            search, rows, oof, designs = T.cv_xgb(run.df, run.folds, run.groups, fs, configs)
            search.to_csv(run.out / f"search_{fs}.csv", index=False)
            best = search.loc[search.brier.idxmin()]
            c = int(best.config)
            params, fold_rows, oof_best = configs[c], rows[rows.config == c].to_dict("records"), oof[:, c]
            extra = {"search_space": {"learning_rate": "log-uniform [0.01, 0.3]", "max_depth": "2-8",
                                      "min_child_weight": "log-uniform [1, 100]", "subsample": "[0.5, 1]",
                                      "colsample_bytree": "[0.2, 1]", "reg_lambda": "log-uniform [0.1, 100]",
                                      "reg_alpha": "log-uniform [0.001, 10]",
                                      "n_trees": "early stopping, 100 rounds patience, max 5000"},
                     "n_configs": len(configs), "chosen_config": c,
                     "cv_trees_mean": float(best.trees_mean), "cv_trees_max": float(best.trees_max)}
            if fs == "landmark":
                run.landmark_params = params
                # alternative criterion: same trees settings, Cox partial likelihood instead of IPCW log loss
                s2, r2, o2, _ = T.cv_xgb(run.df, run.folds, run.groups, fs, [params], objective="cox",
                                         designs=designs)
                r2.to_csv(run.out / "folds_landmark_coxloss.csv", index=False)
                run.save_oof("landmark_coxloss", o2[:, 0])
                T.by_type(run.df, o2[:, 0]).to_csv(run.out / "by_type_landmark_coxloss.csv")
                mc = run.final_fit(fs, params, learner="xgb_cox")
                extra["alternative_criterion_cox"] = {
                    "cv": cv_summary(r2.to_dict("records")),
                    "cv_trees_mean": float(r2.n_trees.mean()),
                    "spearman_with_ipcw_oof": float(pd.Series(o2[:, 0]).corr(pd.Series(oof_best), method="spearman")),
                    "mean_abs_risk_difference": float(np.abs(o2[:, 0] - oof_best).mean()),
                    "final_trees": int(mc.model_["best_iteration"] + 1)}
        else:
            params = run.landmark_params
            _, rows, oof, _ = T.cv_xgb(run.df, run.folds, run.groups, fs, [params])
            fold_rows, oof_best = rows.to_dict("records"), oof[:, 0]
            extra = {"note": "landmark settings reused, PD-L1 columns removed"}
        m = run.final_fit(fs, params)
        hist = m.model_["history"]
        extra["final"] = {"trees": int(m.model_["best_iteration"] + 1), "max_rounds": 5000, "patience": 100,
                          "stopped_by_early_stopping": len(hist["es"]) < 5000,
                          "es_logloss_best": float(min(hist["es"])),
                          "importance_total_gain": gain_importance(m.model_, m.feature_names_)}
        pd.DataFrame(hist).to_csv(run.out / f"final_curve_{fs}.csv", index_label="round")
        run.finish_fs(fs, params, fold_rows, oof_best, extra)
        run.write()


def run_nn(run):
    configs = T.sample_nn(N_NN)
    for fs in FEATURE_SETS + [SENSITIVITY]:
        designs = None
        if fs != SENSITIVITY:
            search, rows, oof, designs = T.cv_nn(run.df, run.folds, run.groups, fs, configs)
            search.to_csv(run.out / f"search_{fs}.csv", index=False)
            c = int(search.loc[search.brier.idxmin()].config)
            params = configs[c]
            extra = {"search_space": {"width": [32, 64, 128, 256], "depth": "1-3", "dropout": "[0, 0.5]",
                                      "lr": "log-uniform [1e-4, 3e-3]", "weight_decay": "log-uniform [1e-6, 0.1]",
                                      "epochs": "early stopping, patience 20, max 300", "batch": 512,
                                      "optimizer": "AdamW"},
                     "n_configs": len(configs), "chosen_config": c}
            if fs == "landmark":
                run.landmark_params = params
        else:
            params, extra = run.landmark_params, {"note": "landmark settings reused, PD-L1 columns removed"}
        # the chosen settings with five seeds: spread across starts, and the ensemble that is deployed
        s5, r5, o5, _ = T.cv_nn(run.df, run.folds, run.groups, fs, [params], seeds=NN_SEEDS, designs=designs)
        r5.to_csv(run.out / f"seeds_{fs}.csv", index=False)
        ens = np.mean([o5[(0, s)] for s in NN_SEEDS], axis=0)
        fold_rows = [{"fold": k, **T.score(run.df, va, ens[va])} for k, (_, va) in enumerate(run.folds)]
        per_seed = s5.set_index("seed").brier
        extra["seeds"] = {"cv_brier_by_seed": per_seed.to_dict(), "spread": float(per_seed.max() - per_seed.min())}
        m = run.final_fit(fs, params, seeds=NN_SEEDS)
        extra["final"] = [{k: v for k, v in r.items() if k not in ("net", "curve")} for r in m.model_]
        pd.DataFrame([(s["seed"], e, tr, va) for s in m.model_ for e, (tr, va) in enumerate(s["curve"], 1)],
                     columns=["seed", "epoch", "train_loss", "es_loss"]).to_csv(run.out / f"final_curve_{fs}.csv",
                                                                                index=False)
        run.finish_fs(fs, params, fold_rows, ens, extra, seeds=NN_SEEDS)
        run.write()


def main():
    global FEATURE_SETS, NN_SEEDS, N_XGB, N_NN
    p = argparse.ArgumentParser()
    p.add_argument("learner", choices=["km", "coxnet", "xgb", "nn"])
    p.add_argument("--quick", action="store_true", help="smoke test: tiny search, written to *_quick folders")
    args = p.parse_args()
    learner = args.learner
    if args.quick:
        FEATURE_SETS, NN_SEEDS, N_XGB, N_NN = ["landmark"], (42, 43), 2, 2
        T.COXNET_L1 = (0.5,)
    run = Run(learner, quick=args.quick)
    run.log(f"{len(run.df)} training patients, {len(run.folds)} folds")
    {"km": run_km, "coxnet": run_coxnet, "xgb": run_xgb, "nn": run_nn}[learner](run)
    run.write()


if __name__ == "__main__":
    main()

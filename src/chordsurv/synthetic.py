"""Synthetic data shaped like the landmark set (same column names and types, invented values).
Used by the tests and the usage example, so that neither needs the licensed MSK-CHORD data."""
import numpy as np
import pandas as pd

TYPES = ["Breast Cancer", "Colorectal Cancer", "Non-Small Cell Lung Cancer", "Pancreatic Cancer", "Prostate Cancer"]


def make_landmark_like(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ct = rng.choice(TYPES, n, p=[0.22, 0.23, 0.3, 0.11, 0.14])
    df = pd.DataFrame(index=pd.Index([f"SYN-{i:05d}" for i in range(n)], name="PATIENT_ID"))
    df["cancer_type"] = ct
    df["cancer_type_detailed"] = np.where(rng.random(n) < 0.8, ct, "Other " + ct)
    df["sample_type"] = rng.choice(["Primary", "Metastasis"], n, p=[0.65, 0.35])
    df["metastatic_site"] = np.where(df.sample_type.eq("Metastasis"), rng.choice(["Liver", "Lymph Node", "Bone"], n),
                                     "Not Applicable")
    df["msi_type"] = rng.choice(["Stable", "Instable", None], n, p=[0.9, 0.04, 0.06])
    df["stage_dx"] = rng.choice(["Stage 1-3", "Stage 4", None], n, p=[0.6, 0.38, 0.02])
    df["sex"] = np.where(ct == "Prostate Cancer", "Male", np.where(ct == "Breast Cancer", "Female",
                                                                  rng.choice(["Male", "Female"], n)))
    df["race"] = rng.choice(["White", "Black", "Asian", None], n, p=[0.75, 0.08, 0.1, 0.07])
    df["tmb"] = rng.lognormal(1.2, 0.8, n)
    df["never_smoker"] = np.where(rng.random(n) < 0.05, np.nan, (rng.random(n) < 0.5).astype(float))
    breast = ct == "Breast Cancer"
    df["hr_positive"] = np.where(breast, (rng.random(n) < 0.8).astype(float), np.nan)
    df["her2_positive"] = np.where(breast, (rng.random(n) < 0.17).astype(float), np.nan)
    df["d0_ecog"] = np.where(rng.random(n) < 0.45, np.nan, rng.choice([0, 1, 2, 3], n, p=[0.5, 0.35, 0.12, 0.03]))
    df["lm_ecog"] = np.where(rng.random(n) < 0.35, np.nan, rng.choice([0, 1, 2, 3], n, p=[0.45, 0.35, 0.15, 0.05]))
    df["win_prog_max"] = np.where(rng.random(n) < 0.2, np.nan, rng.beta(0.6, 1.5, n))
    df["win_on_systemic_tx_at_landmark"] = (rng.random(n) < 0.6).astype(float)
    df["cnt_n_radiology_reports"] = rng.poisson(4, n).astype(float)
    df["mut_TP53"] = (rng.random(n) < np.where(ct == "Colorectal Cancer", 0.7, 0.4)).astype(float)
    df["mut_KRAS"] = (rng.random(n) < np.where(ct == "Pancreatic Cancer", 0.9, 0.2)).astype(float)
    df["mut_RARE"] = (rng.random(n) < 0.002).astype(float)                       # below 1 %: filtered out
    df["mut_NEWGENE"] = np.where(rng.random(n) < 0.3, np.nan, (rng.random(n) < 0.05).astype(float))  # untested
    df["era"] = rng.choice(["IMPACT341", "IMPACT410", "IMPACT468"], n, p=[0.1, 0.25, 0.65])
    df["test_era"] = False

    type_effect = pd.Series([0.0, 0.2, 0.6, 1.2, -0.3], index=TYPES)
    lp = (type_effect[ct].to_numpy() + 0.7 * df.stage_dx.eq("Stage 4") + 0.5 * df.lm_ecog.fillna(1)
          + 1.8 * df.win_prog_max.fillna(0.3) + 0.4 * df.mut_TP53 - 0.3 * df.win_on_systemic_tx_at_landmark)
    T = rng.exponential(1 / (0.012 * np.exp(lp.to_numpy())))
    C = rng.uniform(3, 60, n)
    df["time"] = np.round(np.minimum(T, C), 2) + 0.01
    df["event"] = (T <= C).astype(int)
    groups = {"static": ["cancer_type", "cancer_type_detailed", "sample_type", "metastatic_site", "msi_type",
                         "stage_dx", "sex", "race", "tmb", "never_smoker", "hr_positive", "her2_positive"],
              "day0": ["d0_ecog"], "landmark": ["lm_ecog"],
              "window": ["win_prog_max", "win_on_systemic_tx_at_landmark"],
              "intensity": ["cnt_n_radiology_reports"],
              "genomic": ["mut_TP53", "mut_KRAS", "mut_RARE", "mut_NEWGENE"],
              "never_use": ["CURRENT_AGE_DEID"]}
    return df, groups

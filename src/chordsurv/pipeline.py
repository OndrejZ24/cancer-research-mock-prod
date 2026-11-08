"""Raw cBioPortal export -> the analysis tables (patients, landmark set, genomics, causal cohort)."""
import json
from pathlib import Path

from . import cohorts, features as F
from .config import LANDMARK_DAY, LEAKY_COLUMNS, PROCESSED_DIR, RAW_DIR
from .load import load_raw

STATIC = ["cancer_type", "cancer_type_detailed", "sample_type", "metastatic_site", "tmb", "msi_score", "msi_type",
          "tumor_purity", "stage_dx", "dx_to_seq_days", "n_primaries_by_day0", "sex", "race", "hispanic",
          "never_smoker", "hr_positive", "her2_positive"]
# missingness of these carries outcome information: impute them without a missing-value indicator
IMPUTE_WITHOUT_INDICATOR = ["never_smoker"]


def build(raw_dir=RAW_DIR, out_dir=PROCESSED_DIR, verbose=True):
    """Build every table from the raw export. Features only use events dated on or before their cutoff:
    day 0 (sequencing), day 182 (the 6-month landmark) or the first treatment start (causal cohort)."""
    say = print if verbose else (lambda *a, **k: None)
    say(f"reading {raw_dir}")
    raw = load_raw(raw_dir)
    base = cohorts.patient_base(raw)
    pre_tx = [c for c in base.columns if c.startswith("pre_tx_")]

    mut = F.harmonise_mutations(raw["mutations"])
    coverage = F.panel_coverage(mut, raw["panels"])
    genomics = F.genomic_features(base.sample_id, base.era, mut, raw["cna"], raw["sv"], coverage)

    d0 = F.clinical_at(raw, base.index, 0).add_prefix("d0_")
    patients = base.join(d0)

    lm = cohorts.landmark_cohort(base)
    lm_clin = F.clinical_at(raw, lm.index, LANDMARK_DAY).add_prefix("lm_")
    win = F.window_features(raw, lm.index, d0.loc[lm.index].rename(columns=lambda c: c[3:])).add_prefix("win_")
    cnt = F.intensity_features(raw, lm.index).add_prefix("cnt_")
    landmark = lm.join(d0).join(lm_clin).join(win).join(cnt)

    causal = cohorts.causal_cohort(raw, base, mut)
    tz = F.clinical_at(raw, causal.index, causal.first_start.fillna(0)).add_prefix("tz_")
    causal = causal.join(tz)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patients.to_parquet(out_dir / "patients.parquet")
    landmark.to_parquet(out_dir / "landmark.parquet")
    genomics.to_parquet(out_dir / "genomics.parquet")
    causal.to_parquet(out_dir / "causal.parquet")
    groups = {
        "static": STATIC + pre_tx,
        "day0": list(d0.columns),
        "landmark": list(lm_clin.columns),
        "window": list(win.columns),
        "intensity": list(cnt.columns),
        "genomic": list(genomics.columns),
        "time_zero": list(tz.columns),
        "impute_without_indicator": IMPUTE_WITHOUT_INDICATOR,
        "never_use": list(LEAKY_COLUMNS) + ["smoking"],
    }
    (out_dir / "feature_groups.json").write_text(json.dumps(groups, indent=1))
    say(f"patients {patients.shape} | landmark {landmark.shape} | genomics {genomics.shape} | "
        f"causal {causal.shape} (eligible {int(causal.eligible.sum())})")
    return {"patients": patients, "landmark": landmark, "genomics": genomics, "causal": causal, "groups": groups}

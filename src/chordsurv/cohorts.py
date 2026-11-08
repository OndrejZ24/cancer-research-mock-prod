"""Analysis populations: all patients (description), the 6-month landmark set (prediction), and the
stage 4 NSCLC first-treatment cohort (causal question)."""
import numpy as np
import pandas as pd

from .config import DAYS_PER_MONTH, LANDMARK_MONTHS, SOLID_PANELS, TEST_ERA, TREATMENT_CLASSES, UNKNOWN_CODES
from .features import slug

RACE = {"White": "White", "Black Or African American": "Black", "Asian-Far East/Indian Subcont": "Asian",
        "Native American-Am Ind/Alaska": "Other", "Native Hawaiian Or Pacific Isl": "Other", "Other": "Other"}
# OncoTree files these under NSCLC, but they are neuroendocrine or salivary-type tumours with different biology
# and standard treatment (and were excluded from KEYNOTE-189): not part of the causal comparison.
NON_CONVENTIONAL_NSCLC = {"Large Cell Neuroendocrine Carcinoma", "Lung Carcinoid", "Atypical Lung Carcinoid",
                          "Lung Neuroendocrine Tumor", "Adenoid Cystic Carcinoma of the Lung",
                          "Mucoepidermoid Carcinoma of the Lung", "NUT Carcinoma of the Lung",
                          "Pleuropulmonary Blastoma", "Lung Adenocarcinoma In Situ"}
# ICD-O topography codes of each sequenced cancer type (C21 = rectum/anal overlap, recorded for some rectal cancers)
TOPOGRAPHY = {"Non-Small Cell Lung Cancer": ("C34",), "Colorectal Cancer": ("C18", "C19", "C20", "C21"),
              "Breast Cancer": ("C50",), "Prostate Cancer": ("C61",), "Pancreatic Cancer": ("C25",)}


def _clean(s):
    return s.where(~s.isin(UNKNOWN_CODES))


def patient_base(raw):
    """One row per patient: the first sequenced sample defines day 0, cancer type and era."""
    pt = raw["patient"].set_index("PATIENT_ID")
    seq_day = raw["timelines"]["specimen"].set_index("SAMPLE_ID").START_DATE
    s = raw["sample"].assign(seq_day=lambda d: d.SAMPLE_ID.map(seq_day))
    first = s.sort_values(["PATIENT_ID", "seq_day", "SAMPLE_ID"]).drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    # Patients with several primaries have several registry rows: use the diagnosis of the sequenced cancer
    # (matched by ICD-O topography), falling back to the earliest diagnosis when none matches.
    dx = raw["timelines"]["diagnosis"].copy()
    topo = dx.DX_DESCRIPTION.str.extract(r"\|\s*(C\d{2})")[0]
    dx["matches"] = [t in TOPOGRAPHY.get(c, ()) for t, c in zip(topo, dx.PATIENT_ID.map(first.CANCER_TYPE))]
    dx = dx.sort_values(["PATIENT_ID", "matches", "START_DATE"], ascending=[True, False, True])
    first_dx = dx.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")

    b = pd.DataFrame(index=pt.index)
    b["sample_id"] = first.SAMPLE_ID
    b["first_seq_day"] = first.seq_day
    b["era"] = first.SAMPLE_ID.map(raw["panels"].mutations)
    b["in_study"] = b.era.isin(SOLID_PANELS)
    b["test_era"] = b.era.eq(TEST_ERA)
    b["os_months"] = pt.OS_MONTHS
    b["event"] = pt.OS_STATUS.str.startswith("1").astype(int)

    b["cancer_type"] = first.CANCER_TYPE
    b["cancer_type_detailed"] = first.CANCER_TYPE_DETAILED
    b["sample_type"] = first.SAMPLE_TYPE
    b["metastatic_site"] = first.METASTATIC_SITE
    b["tmb"] = first.TMB_NONSYNONYMOUS
    b["msi_score"] = first.MSI_SCORE.where(first.MSI_SCORE >= 0)        # -1 = "could not be computed"
    b["msi_type"] = first.MSI_TYPE.replace({"Do Not Report": "Do not report"})
    b["tumor_purity"] = first.TUMOR_PURITY

    b["dx_matches_sequenced_cancer"] = first_dx.matches.reindex(b.index, fill_value=False)
    # stage is only known at day 0 if the diagnosis was recorded by then (50 diagnoses are dated after day 0)
    known = first_dx.START_DATE <= 0
    b["stage_dx"] = first_dx.STAGE_CDM_DERIVED.where(known)
    b["dx_to_seq_days"] = (-first_dx.START_DATE).where(known)
    b["n_primaries_by_day0"] = dx[dx.START_DATE <= 0].groupby("PATIENT_ID").size().reindex(b.index, fill_value=0)

    b["sex"] = _clean(pt.GENDER)
    b["race"] = pt.RACE.map(RACE)                     # refused / unknown / no value -> NaN
    eth = _clean(pt.ETHNICITY)
    b["hispanic"] = np.where(eth.isna(), np.nan, np.where(eth.eq("Non-Spanish; Non-Hispanic"), 0.0, 1.0))
    b["smoking"] = pt.SMOKING_PREDICTIONS_3_CLASSES   # raw, for description only
    # 'Unknown' means the NLP found no smoking section; it is more common in patients who die sooner (fewer
    # notes), so it carries follow-up information: model it as missing and impute WITHOUT a missing indicator.
    b["never_smoker"] = pt.SMOKING_PREDICTIONS_3_CLASSES.map({"Never": 1.0, "Former/Current Smoker": 0.0})
    b["hr_positive"] = pt.HR.map({"Yes": 1.0, "No": 0.0})
    b["her2_positive"] = pt.HER2.map({"Yes": 1.0, "No": 0.0})

    tx = raw["timelines"]["treatment"]
    before = tx[tx.START_DATE < 0]
    for c in TREATMENT_CLASSES:
        b[f"pre_tx_{slug(c)}"] = b.index.isin(before.loc[before.SUBTYPE.eq(c), "PATIENT_ID"]).astype(float)
    b["pre_tx_any"] = b.index.isin(before.PATIENT_ID).astype(float)
    return b


def landmark_cohort(base, landmark_months=LANDMARK_MONTHS):
    """Patients alive and in follow-up at the landmark; time is measured from the landmark."""
    lm = base[base.in_study & (base.os_months > landmark_months)].copy()
    lm["time"] = lm.os_months - landmark_months
    return lm


def causal_cohort(raw, base, mut, window_days=60, regimen_days=14,
                  later_eras=("IMPACT468", "IMPACT505")):
    """Stage 4 NSCLC of conventional histology, no EGFR-sensitising mutation or ALK fusion, no systemic treatment
    before sequencing, first systemic regimen started within `window_days` after sequencing.
    Time zero = treatment start."""
    tx = raw["timelines"]["treatment"]
    systemic = tx[tx.SUBTYPE.isin(["Chemo", "Immuno", "Targeted", "Biologic", "Investigational"])]
    c = base[base.cancer_type.eq("Non-Small Cell Lung Cancer") & base.stage_dx.eq("Stage 4")].copy()

    egfr = mut[mut.gene.eq("EGFR") & (mut.HGVSp_Short.eq("p.L858R") | (
        mut.Variant_Classification.eq("In_Frame_Del")
        & mut.HGVSp_Short.str.contains(r"p\.[EL]7(?:4[5-9]|5[0-3])", na=False)))]
    sv = raw["sv"]
    alk = sv.loc[sv.Site1_Hugo_Symbol.eq("ALK") | sv.Site2_Hugo_Symbol.eq("ALK"), "Sample_Id"]
    c["egfr_sensitising"] = c.sample_id.isin(egfr.Tumor_Sample_Barcode)
    c["alk_fusion"] = c.sample_id.isin(alk)
    c["treated_before_seq"] = c.index.isin(systemic.loc[systemic.START_DATE < 0, "PATIENT_ID"])

    after = systemic[systemic.START_DATE >= 0]
    c["first_start"] = after.groupby("PATIENT_ID").START_DATE.min().reindex(c.index)
    w = after.merge(c[["first_start"]], left_on="PATIENT_ID", right_index=True)
    w = w[w.START_DATE.between(w.first_start, w.first_start + regimen_days)]
    classes = w.groupby("PATIENT_ID").SUBTYPE.agg(set)
    c["regimen"] = classes.map(
        lambda s: "immunotherapy" if "Immuno" in s else
        "targeted_or_investigational" if s & {"Targeted", "Investigational"} else
        "chemotherapy" if "Chemo" in s else "other").reindex(c.index)
    c.loc[~c.first_start.between(0, window_days), "regimen"] = "no_start_in_window"
    c["immuno_with_chemo"] = c.index.isin(w.loc[w.SUBTYPE.eq("Chemo"), "PATIENT_ID"]) & c.regimen.eq("immunotherapy")
    # first immunotherapy relative to the first treatment start: post-baseline information, used only by the
    # grace-period sensitivity analysis (immunotherapy added at cycle 2 still belongs to the first-line strategy)
    io = after[after.SUBTYPE.eq("Immuno")].groupby("PATIENT_ID").START_DATE.min().reindex(c.index)
    c["days_to_immuno"] = io - c.first_start

    c["conventional_histology"] = ~c.cancer_type_detailed.isin(NON_CONVENTIONAL_NSCLC)
    c["eligible"] = (c.conventional_histology & ~c.egfr_sensitising & ~c.alk_fusion & ~c.treated_before_seq
                     & c.regimen.isin(["immunotherapy", "chemotherapy"]) & c.era.isin(later_eras))
    c["treated"] = c.regimen.eq("immunotherapy").astype(int)
    # outcome from time zero (treatment start); everyone eligible is alive at treatment start by construction
    c["time"] = c.os_months - c.first_start / DAYS_PER_MONTH
    return c

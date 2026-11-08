"""Paths and study-design constants shared by every part of the project."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("CHORD_DATA_DIR", PROJECT_DIR.parent / "data" / "msk_chord_2024"))
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"        # aggregate numbers only (shareable)
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"    # fitted models and patient-level predictions (never shared)

# Time: every timeline date is in days relative to the patient's first sequencing (day 0).
DAYS_PER_MONTH = 30.4375
LANDMARK_MONTHS = 6
LANDMARK_DAY = 182          # events dated <= day 182 are known at the 6-month landmark (182.6 days)
HORIZON_MONTHS = 12         # predict death within 12 months after the landmark
RECENT_DAYS = 90            # "latest value" window for ECOG, labs and radiology before a cutoff

# Eras = sequencing panel of the first sample. The newest era is the untouched temporal test set.
TRAIN_ERAS = ("IMPACT341", "IMPACT410", "IMPACT468")
TEST_ERA = "IMPACT505"
SOLID_PANELS = TRAIN_ERAS + (TEST_ERA,)

# Gene symbols renamed between panel versions (older name -> current HGNC name).
GENE_ALIASES = {"MLL": "KMT2A", "MLL2": "KMT2D", "MLL3": "KMT2C", "MLL4": "KMT2B", "MYCL1": "MYCL"}
# Protein-altering variant classes; flanks, introns, UTRs and silent variants are dropped.
CODING_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del",
    "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
}
FUSION_GENES = ("ALK", "RET", "ROS1", "NTRK1", "NTRK2", "NTRK3", "ERG", "FGFR2", "FGFR3", "BRAF", "MET", "NRG1")

UNKNOWN_CODES = {"Unknown", "Pt Refused To Answer", "No Value Entered", "Unknown whether Spanish or not"}
SITES = ("Adrenal Glands", "Bone", "CNS/Brain", "Intra-Abdominal", "Liver", "Lung", "Lymph Nodes", "Other",
         "Pleura", "Reproductive Organs")
LABS = {"cea": "cea_labs", "psa": "psa_labs", "ca15_3": "ca_15-3_labs", "ca19_9": "ca_19-9_labs"}
TREATMENT_CLASSES = ("Chemo", "Immuno", "Targeted", "Hormone", "Biologic", "Investigational", "Bone Treatment")

# Patient-level columns that summarise the whole follow-up or encode the outcome: never used as features.
LEAKY_COLUMNS = (
    "CURRENT_AGE_DEID",          # age at death for deceased patients (see REVIEW_LOG pass 3)
    "STAGE_HIGHEST_RECORDED", "NUM_ICDO_DX", "HISTORY_OF_PDL1", "PRIOR_MED_TO_MSK",
    "GLEASON_FIRST_REPORTED", "GLEASON_HIGHEST_REPORTED", "OS_MONTHS", "OS_STATUS",
    "ADRENAL_GLANDS", "BONE", "CNS_BRAIN", "INTRA_ABDOMINAL", "LIVER", "LUNG", "LYMPH_NODES", "OTHER",
    "PLEURA", "REPRODUCTIVE_ORGANS",
)

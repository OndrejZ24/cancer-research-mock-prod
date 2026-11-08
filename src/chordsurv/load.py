"""Read the raw cBioPortal export of MSK-CHORD."""
from pathlib import Path

import pandas as pd

from .config import RAW_DIR


def read_clinical(name, raw_dir=RAW_DIR):
    # cBioPortal clinical files start with four '#' metadata lines (label, description, type, priority)
    return pd.read_csv(Path(raw_dir) / name, sep="\t", skiprows=4)


def read_timeline(name, raw_dir=RAW_DIR):
    # Some timelines repeat identical rows (e.g. 3,378 in radiation); duplicates would inflate counts.
    t = pd.read_csv(Path(raw_dir) / f"data_timeline_{name}.txt", sep="\t", low_memory=False)
    return t.drop_duplicates().reset_index(drop=True)


def load_raw(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    timelines = {p.stem.removeprefix("data_timeline_"): read_timeline(p.stem.removeprefix("data_timeline_"), raw_dir)
                 for p in sorted(raw_dir.glob("data_timeline_*.txt"))}
    return {
        "patient": read_clinical("data_clinical_patient.txt", raw_dir),
        "sample": read_clinical("data_clinical_sample.txt", raw_dir),
        "timelines": timelines,
        "mutations": pd.read_csv(raw_dir / "data_mutations.txt", sep="\t", low_memory=False,
                                 usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification",
                                          "HGVSp_Short", "Mutation_Status"]),
        "cna": pd.read_csv(raw_dir / "data_cna.txt", sep="\t", index_col=0),
        "sv": pd.read_csv(raw_dir / "data_sv.txt", sep="\t", low_memory=False,
                          usecols=["Sample_Id", "Site1_Hugo_Symbol", "Site2_Hugo_Symbol"]),
        "panels": pd.read_csv(raw_dir / "data_gene_panel_matrix.txt", sep="\t").set_index("SAMPLE_ID"),
    }

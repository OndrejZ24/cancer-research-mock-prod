"""Unit tests on small synthetic data (no MSK-CHORD data needed). Run: pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chordsurv import cohorts, features as F  # noqa: E402

P = pd.Index(["A", "B"], name="PATIENT_ID")


def timeline(rows, cols):
    return pd.DataFrame(rows, columns=["PATIENT_ID", "START_DATE", *cols])


def synthetic_raw():
    empty_lab = timeline([], ["RESULT"])
    return {"timelines": {
        "performance_status": timeline([("A", -10, 1), ("A", 0, 2), ("A", 5, 4), ("B", -200, 0)], ["ECOG"]),
        "cea_labs": timeline([("A", -1, "3.0"), ("A", 50, "90.0")], ["RESULT"]),
        "psa_labs": empty_lab, "ca_15-3_labs": empty_lab, "ca_19-9_labs": empty_lab,
        "tumor_sites": timeline([("A", -30, "Bone"), ("A", 100, "Liver")], ["TUMOR_SITE"]),
        "progression": timeline([("A", -5, 0.1), ("A", 60, 0.9)], ["NLP_PROGRESSION_PROBABILITY"]),
        "cancer_presence": timeline([("A", -5, 0.7)], ["NLP_HAS_CANCER_PROBABILITY"]),
        "pdl1": timeline([("A", 10, "Yes")], ["PDL1_POSITIVE"]),
        "mmr": timeline([], ["MMR_ABSENT"]), "gleason": timeline([], ["GLEASON_SCORE"]),
        "prior_meds": timeline([("B", 20, "Prior medications to MSK")], ["PRIOR_MED_TO_MSK"]),
        "treatment": pd.DataFrame([("A", 30, 400, "Immuno", "PEMBROLIZUMAB"), ("B", 10, 20, "Chemo", "CISPLATIN")],
                                  columns=["PATIENT_ID", "START_DATE", "STOP_DATE", "SUBTYPE", "AGENT"]),
        "radiation": timeline([], []), "surgery": timeline([], []),
    }}


def test_day0_ignores_future_events():
    d0 = F.clinical_at(synthetic_raw(), P, 0)
    assert d0.at["A", "ecog"] == 2                     # day 5 value (4) is in the future
    assert np.isnan(d0.at["B", "ecog"])                # day -200 is outside the 90-day window
    assert d0.at["A", "cea"] == 3.0
    assert d0.at["A", "site_bone"] == 1 and d0.at["A", "site_liver"] == 0
    assert np.isnan(d0.at["A", "pdl1_positive"])       # PD-L1 recorded on day 10
    assert np.isnan(d0.at["B", "prior_meds"])          # assessed on day 20


def test_per_patient_cutoff():
    cut = pd.Series({"A": 0, "B": 30})
    out = F.clinical_at(synthetic_raw(), P, cut)
    assert out.at["B", "prior_meds"] == 1.0 and np.isnan(out.at["A", "pdl1_positive"])


def test_window_features():
    raw = synthetic_raw()
    w = F.window_features(raw, P, F.clinical_at(raw, P, 0), start=1, end=182)
    assert w.at["A", "new_liver"] == 1 and w.at["A", "new_bone"] == 0      # bone was already known at day 0
    assert w.at["A", "prog_max"] == 0.9 and np.isnan(w.at["B", "prog_max"])
    assert w.at["A", "on_systemic_tx_at_landmark"] == 1                    # started day 30, stops day 400
    assert w.at["B", "on_systemic_tx_at_landmark"] == 0                    # stopped day 20
    assert np.isclose(w.at["A", "cea_logratio"], np.log1p(90) - np.log1p(3))


def test_landmark_features_ignore_events_after_day_182():
    """Everything known at the 6-month landmark is unchanged when later events are added."""
    raw, later = synthetic_raw(), synthetic_raw()
    tl = later["timelines"]
    add = lambda name, rows: pd.concat([tl[name], timeline(rows, list(tl[name].columns[2:]))], ignore_index=True)
    tl["progression"] = add("progression", [("A", 200, 0.99), ("B", 300, 0.95)])
    tl["performance_status"] = add("performance_status", [("A", 190, 4), ("B", 250, 3)])
    tl["tumor_sites"] = add("tumor_sites", [("A", 183, "CNS/Brain"), ("B", 400, "Liver")])
    tl["cea_labs"] = add("cea_labs", [("A", 260, "500.0")])
    tl["treatment"] = pd.concat([tl["treatment"], pd.DataFrame([("B", 200, 300, "Immuno", "NIVOLUMAB")],
                                columns=tl["treatment"].columns)], ignore_index=True)
    for r in (raw, later):
        r["lm"] = F.clinical_at(r, P, 182)
        r["win"] = F.window_features(r, P, F.clinical_at(r, P, 0))
        r["cnt"] = F.intensity_features(r, P)
    for k in ("lm", "win", "cnt"):
        pd.testing.assert_frame_equal(raw[k], later[k])


def test_uncovered_gene_is_missing_not_zero():
    mut = F.harmonise_mutations(pd.DataFrame({
        "Hugo_Symbol": ["MLL2", "TP53", "TP53"], "Tumor_Sample_Barcode": ["s1", "s2", "s3"],
        "Variant_Classification": ["Missense_Mutation", "Missense_Mutation", "Intron"],
        "HGVSp_Short": ["", "", ""], "Mutation_Status": ["SOMATIC"] * 3}))
    assert set(mut.gene) == {"KMT2D", "TP53"} and len(mut) == 2              # alias renamed, intron dropped
    coverage = {"IMPACT341": {"TP53"}, "IMPACT410": {"TP53"}, "IMPACT468": {"TP53", "KMT2D"}, "IMPACT505": {"TP53", "KMT2D"}}
    first = pd.Series({"A": "s1", "B": "s2"})
    panel = pd.Series({"A": "IMPACT468", "B": "IMPACT341"})
    cna = pd.DataFrame(0, index=["TP53", "KMT2D"], columns=["s1", "s2"])
    sv = pd.DataFrame(columns=["Sample_Id", "Site1_Hugo_Symbol", "Site2_Hugo_Symbol"])
    g = F.genomic_features(first, panel, mut, cna, sv, coverage)
    assert g.at["A", "mut_KMT2D"] == 1 and g.at["A", "mut_TP53"] == 0
    assert np.isnan(g.at["B", "mut_KMT2D"]) and np.isnan(g.at["B", "amp_KMT2D"])   # not on IMPACT341


def test_landmark_cohort():
    base = pd.DataFrame({"in_study": [True, True, False], "os_months": [6.0, 6.5, 30.0]}, index=["A", "B", "C"])
    lm = cohorts.landmark_cohort(base)
    assert list(lm.index) == ["B"] and np.isclose(lm.time.iloc[0], 0.5)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")

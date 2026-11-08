"""Features. Every function only uses events dated on or before the cutoff it is given."""
import re

import numpy as np
import pandas as pd

from .config import (CODING_CLASSES, FUSION_GENES, GENE_ALIASES, LABS, LANDMARK_DAY, RECENT_DAYS, SITES,
                     SOLID_PANELS, TREATMENT_CLASSES)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _cutoff(cutoff, index):
    # a scalar day or a per-patient Series of days -> per-patient Series named "cut"
    if np.isscalar(cutoff):
        return pd.Series(float(cutoff), index=index, name="cut")
    return cutoff.reindex(index).astype(float).rename("cut")


def _window(t, cut, lo=-np.inf, hi=0.0):
    # rows with cut + lo <= START_DATE <= cut + hi, for the patients in `cut`
    # (unnamed index + reset: an empty `t` would otherwise come back indexed by PATIENT_ID and clash with the column)
    m = t.merge(cut.rename_axis(None), left_on="PATIENT_ID", right_index=True).reset_index(drop=True)
    return m[(m.START_DATE >= m.cut + lo) & (m.START_DATE <= m.cut + hi)]


def _last(w, col):
    # value of the latest event per patient; ties on the same day resolved by the larger value (deterministic)
    return w.sort_values(["PATIENT_ID", "START_DATE", col]).groupby("PATIENT_ID")[col].last()


def _labs(raw, name):
    t = raw["timelines"][LABS[name]]
    return t.assign(RESULT=pd.to_numeric(t.RESULT, errors="coerce")).dropna(subset=["RESULT"])


def clinical_at(raw, index, cutoff):
    """Clinical state known at `cutoff` (scalar or per-patient days): latest values and history so far."""
    tl, cut, out = raw["timelines"], _cutoff(cutoff, index), pd.DataFrame(index=index)
    recent = lambda t: _window(t, cut, lo=-RECENT_DAYS)
    ever = lambda t: _window(t, cut)

    out["ecog"] = _last(recent(tl["performance_status"]), "ECOG")
    for name in LABS:
        out[name] = _last(recent(_labs(raw, name)), "RESULT")

    sites = ever(tl["tumor_sites"])
    for s in SITES:
        out[f"site_{slug(s)}"] = index.isin(sites.loc[sites.TUMOR_SITE.eq(s), "PATIENT_ID"]).astype(float)
    out["n_sites"] = out[[f"site_{slug(s)}" for s in SITES]].sum(axis=1)
    # no radiology report at all means "not imaged", not "no metastases"
    imaged = pd.concat([ever(tl[n]).PATIENT_ID for n in ("progression", "tumor_sites", "cancer_presence")])
    out["any_radiology"] = index.isin(imaged).astype(float)

    out["pdl1_positive"] = _last(ever(tl["pdl1"]), "PDL1_POSITIVE").map({"Yes": 1.0, "No": 0.0})
    out["mmr_absent"] = _last(ever(tl["mmr"]), "MMR_ABSENT").astype(float)
    out["gleason_max"] = ever(tl["gleason"]).groupby("PATIENT_ID").GLEASON_SCORE.max()
    out["prior_meds"] = _last(ever(tl["prior_meds"]), "PRIOR_MED_TO_MSK").map(
        {"Prior medications to MSK": 1.0, "No prior medications": 0.0})

    prog = recent(tl["progression"])
    out["prog_max_recent"] = prog.groupby("PATIENT_ID").NLP_PROGRESSION_PROBABILITY.max()
    out["presence_last_recent"] = _last(recent(tl["cancer_presence"]), "NLP_HAS_CANCER_PROBABILITY")
    return out


def window_features(raw, index, baseline, start=1, end=LANDMARK_DAY):
    """What happened between day `start` and day `end` (the landmark), relative to the day-0 state."""
    tl, cut, out = raw["timelines"], _cutoff(end, index), pd.DataFrame(index=index)
    win = lambda t: _window(t, cut, lo=start - end)

    prog = win(tl["progression"])
    g = prog.groupby("PATIENT_ID").NLP_PROGRESSION_PROBABILITY
    out["prog_max"], out["prog_mean"] = g.max(), g.mean()
    out["prog_last"] = _last(prog, "NLP_PROGRESSION_PROBABILITY")
    out["presence_mean"] = win(tl["cancer_presence"]).groupby("PATIENT_ID").NLP_HAS_CANCER_PROBABILITY.mean()

    first_report = (tl["tumor_sites"].groupby(["PATIENT_ID", "TUMOR_SITE"]).START_DATE.min().unstack()
                    .reindex(columns=list(SITES)))
    for s in SITES:
        out[f"new_{slug(s)}"] = first_report[s].between(start, end).reindex(index, fill_value=False).astype(float)
    out["n_new_sites"] = out[[f"new_{slug(s)}" for s in SITES]].sum(axis=1)

    out["ecog_last"] = _last(win(tl["performance_status"]), "ECOG")
    out["ecog_change"] = out.ecog_last - baseline["ecog"]
    for name in LABS:
        last = _last(win(_labs(raw, name)), "RESULT")
        out[f"{name}_logratio"] = np.log1p(last) - np.log1p(baseline[name])

    tx = win(tl["treatment"])
    for c in TREATMENT_CLASSES:
        out[f"tx_{slug(c)}"] = index.isin(tx.loc[tx.SUBTYPE.eq(c), "PATIENT_ID"]).astype(float)
    out["n_new_agents"] = tx.groupby("PATIENT_ID").AGENT.nunique().reindex(index, fill_value=0)
    # a treatment started by the landmark and not stopped before it is known to be ongoing at the landmark
    t = tl["treatment"]
    ongoing = t[t.SUBTYPE.isin(["Chemo", "Immuno", "Targeted", "Hormone", "Biologic", "Investigational"])
                & (t.START_DATE <= end) & (t.STOP_DATE >= end)]
    out["on_systemic_tx_at_landmark"] = index.isin(ongoing.PATIENT_ID).astype(float)
    out["radiation"] = index.isin(win(tl["radiation"]).PATIENT_ID).astype(float)
    out["surgery"] = index.isin(win(tl["surgery"]).PATIENT_ID).astype(float)
    return out


def intensity_features(raw, index, start=1, end=LANDMARK_DAY):
    """How intensively MSK observed the patient in the window: specific to MSK's care process."""
    tl, cut, out = raw["timelines"], _cutoff(end, index), pd.DataFrame(index=index)
    win = lambda t: _window(t, cut, lo=start - end)
    out["n_radiology_reports"] = win(tl["progression"]).groupby("PATIENT_ID").size()
    out["n_lab_tests"] = pd.concat([win(_labs(raw, n)) for n in LABS]).groupby("PATIENT_ID").size()
    out["n_ecog_assessments"] = win(tl["performance_status"]).groupby("PATIENT_ID").size()
    return out.reindex(index).fillna(0.0)


# ---------------------------------------------------------------- genomics

def harmonise_mutations(mut):
    m = mut[mut.Variant_Classification.isin(CODING_CLASSES)].copy()
    m["gene"] = m.Hugo_Symbol.replace(GENE_ALIASES)
    return m


def panel_coverage(mut, panels):
    # Gene lists are not part of the download: a gene counts as covered by a panel if any sample
    # sequenced on that panel has a call in it.
    return mut.groupby(mut.Tumor_Sample_Barcode.map(panels.mutations)).gene.apply(set).to_dict()


def _mask_uncovered(m, sample_panel, coverage):
    # a gene that is not on the patient's panel was never tested: NaN, not 0
    for panel in sample_panel.dropna().unique():
        rows = sample_panel.index[sample_panel.eq(panel)]
        missing = [g for g in m.columns if g not in coverage.get(panel, set())]
        m.loc[rows, missing] = np.nan
    m.loc[sample_panel.index[~sample_panel.isin(SOLID_PANELS)]] = np.nan
    return m


def genomic_features(first_sample, sample_panel, mut, cna, sv, coverage):
    """Mutation, copy-number and fusion indicators of the first sequenced sample (one row per patient)."""
    genes = sorted(set().union(*(coverage[p] for p in SOLID_PANELS)))
    hit = mut[mut.Tumor_Sample_Barcode.isin(first_sample)]
    muts = pd.crosstab(hit.Tumor_Sample_Barcode, hit.gene).gt(0).astype(float)
    muts = muts.reindex(index=first_sample.values, columns=genes, fill_value=0.0).set_axis(first_sample.index)
    muts = _mask_uncovered(muts, sample_panel, coverage).add_prefix("mut_")

    # copy number: genes outside a sample's panel are stored as 0 in the raw matrix, so they are masked too
    cna_genes = [g for g in genes if g in cna.index]
    values = cna.loc[cna_genes].T.reindex(first_sample.values).set_axis(first_sample.index)
    amp = (values >= 2).astype(float).where(values.notna())
    dele = (values <= -1.5).astype(float).where(values.notna())
    amp = _mask_uncovered(amp, sample_panel, coverage).add_prefix("amp_")
    dele = _mask_uncovered(dele, sample_panel, coverage).add_prefix("del_")

    fus = pd.DataFrame(index=first_sample.index)
    for g in FUSION_GENES:
        ids = sv.loc[sv.Site1_Hugo_Symbol.eq(g) | sv.Site2_Hugo_Symbol.eq(g), "Sample_Id"]
        fus[f"fus_{g}"] = first_sample.isin(ids).astype(float)
    return pd.concat([muts, amp, dele, fus], axis=1).astype("float32")

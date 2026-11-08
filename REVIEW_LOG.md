# Review log: data and code checks before the first training

Each issue lists the root cause, not just the symptom, and the resolution: **fixed** (code changed) or
**limitation** (fixing it would be wrong or impossible; stated in the reports).

All checks were rerun on the final build: `patients` 24,950 · `landmark` 21,598 · `genomics` 24,950 × 1,529 ·
`causal` 3,292 stage 4 NSCLC (656 eligible).

## Found while writing the pipeline

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| E1 | `CURRENT_AGE_DEID` grows with survival time among patients who died: 0.82 years per year of survival, adjusted for era and cancer type (Part I notebook, section 4) | The field is age at the patient's last known date. For deceased patients that date is death, so age = age at sequencing + survival. Age at sequencing can't be recovered without calendar dates | **Fixed:** excluded from all features. **Limitation:** age, a known prognostic factor and a plausible confounder, is unavailable |
| E2 | The same gene appears under two names, by era (`MLL2`/`KMT2D`, `MLL3`/`KMT2C`, `MLL`/`KMT2A`, `MLL4`/`KMT2B`, `MYCL1`/`MYCL`) | Older panels were annotated with gene symbols that HGNC later renamed. The split follows the era, so it would silently break a temporal test | **Fixed:** harmonised to current symbols (KMT2D prevalence now 5.6–8.0% in every era) |
| E3 | In the copy-number matrix, genes absent from a sample's panel are stored as 0 ("no change"). Genes added in later panels have exactly 0% alterations on older panels vs 0.4–0.5% on the newest | The matrix is the union of all panels, and cells for untested genes were filled with 0 instead of left empty | **Fixed:** masked to NaN using panel coverage |
| E4 | Exact duplicate timeline rows (radiation 3,378, tumour sites 1,997, cancer presence 1,227, treatment 871) | The export emits one row per source record; identical rows would inflate the count features | **Fixed:** exact duplicates dropped on load |
| E5 | Panel gene lists are not in the download | cBioPortal points to an external reference | **Limitation:** coverage is inferred (a gene counts as covered by a panel if any sample on it has a call). A covered gene that was never mutated on a panel becomes NaN instead of 0, which is harmless because its prevalence is about 0 |

## Pass 1: raw data integrity and time origin
Checked:
- first sequencing is day 0 for every patient;
- no event is dated after the end of survival;
- survival time is always > 0;
- era counts;
- value ranges.

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 1.1 | MSI score of −1 in 91 samples | It's a "could not be computed" sentinel: every one of them has MSI type "Do not report" | **Fixed:** set to NaN |
| 1.2 | For 54 patients, the earliest registry diagnosis is a different cancer from the one sequenced (e.g. a lung tumour sequenced, while the earliest diagnosis was a breast cancer decades earlier) | Patients with several primary cancers have several registry rows, and the earliest was taken regardless of site. Some rectal cancers also carry topography code C21 | **Fixed:** use the diagnosis whose ICD-O topography matches the sequenced cancer (99.96% match; the other 10 patients have no diagnosis row at all) |
| 1.3 | 50 diagnoses are dated after sequencing | Registry entry for a diagnosis made after the first sequencing | **Fixed:** stage and time since diagnosis count only if the diagnosis was made by day 0 (61 patients have no stage) |

## Pass 2: cohorts and target definitions
Checked:
- **Landmark set:** 21,598 patients, all with time > 0. Horizon status: 3,884 died within 12 months, 15,219 known alive, 2,495 censored before month 18.
- **Treatment arms:** classification hand-checked on random patients.

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 2.1 | A "chemotherapy" patient received cisplatin + etoposide, a neuroendocrine regimen | OncoTree files large-cell neuroendocrine carcinoma, carcinoids and salivary-type tumours under NSCLC. They differ in biology and standard treatment, and were excluded from KEYNOTE-189 | **Fixed (causal cohort only):** restricted to conventional NSCLC histologies, giving 656 eligible (470 immunotherapy vs 186 chemotherapy; was 479 vs 201). The prediction model keeps all patients and uses the detailed type as a feature |
| 2.2 | 102 deaths within 7 days after the landmark | Genuine: patients close to death at month 6 | No action |

## Pass 3: leakage, i.e. is every input known at prediction time?
- **Truncation test:** every timeline was cut off at the prediction day (day 0, or day 182; a running treatment's stop date censored at the cutoff) and all features recomputed. **Zero differences** for day-0, landmark, window and count features. The same test at each patient's treatment start gave zero differences for the causal features (pass 6).

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 3.1 | Smoking "Unknown" is associated with shorter survival (logit coefficient on log survival −0.33, p ≈ 10⁻²⁶) | The NLP needs a "Smoking Status" section in a clinical note. Patients who die sooner have fewer notes, so "Unknown" partly encodes follow-up length | **Fixed:** `never_smoker` (1/0) with Unknown as missing, listed in `impute_without_indicator`. Training must impute it **without** a missing-value indicator |
| 3.2 | `HR` and `HER2` have no description and no date | Patient-level fields from pathology; receptor status can be re-tested on later biopsies | **Limitation:** kept (fundamental in breast cancer), with a possible small look-ahead |
| 3.3 | Smoking itself has no date | Extracted from notes, mostly 2015 onwards | **Limitation:** smoking history rarely changes; low risk |

## Pass 4: feature correctness
Checked:
- **Hand-checks:** three random patients recomputed through an independent code path all match.
- **Genomics:** aliases merged; untested genes masked (TP53/KRAS missing only for the 2 haematology-panel patients).
- **Copy-number sanity:** ERBB2 amplification in 12.6% of breast cancers, CDKN2A deletion in 14.6% of pancreatic and 10.6% of NSCLC cancers, AR amplification in 8.3% of prostate cancers.

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 4.1 | Odd gene IDs (`CTD-2561B21.3`, `AC004906.3`, `RP1-85F18.6`) | Variant annotation picked an overlapping non-coding or clone transcript | **Handled:** all are rare, so the ≥ 1% training-prevalence filter removes them |
| 4.2 | 102 genomic columns are observed only on the newest panel | Genes added to IMPACT505 | **Limitation:** dropped by the prevalence filter (never seen in training) |

## Pass 5: train/test split and drift
- **Split:** train (IMPACT341/410/468) 15,595 patients, 7,531 deaths; test (IMPACT505) 6,003 patients, 1,188 deaths; no overlap.
- **Test censoring:** in the test set 36% are censored before the 12-month horizon, so test metrics must use censoring weights (IPCW).
- **Drift:** most features move little (standardised difference < 0.25).

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 5.1 | PD-L1 drifts strongly (standardised difference −0.67). Older eras: recorded for 11%, 78% positive. Newest: recorded for 28%, 47% positive | Testing practice changed: selective testing early on, routine testing once immunotherapy depended on it. Both the value and its missingness change meaning | **Limitation:** Part II compares models with and without PD-L1 |
| 5.2 | ECOG recording rose (day-0 missing 52% → 40%) | Changes in EHR documentation | **Limitation:** expect some drift in "ECOG missing" |
| 5.3 | Cancer mix shifts (breast 25% → 15% of the landmark set) | Case-mix drift across eras | Metrics are reported within each cancer type |
| 5.4 | Only era-level time order is available, so there's no rolling validation inside the training eras | Dates are de-identified | Tune with K-fold CV inside the training eras, plus a forward check (train IMPACT341+410, validate IMPACT468) |

## Pass 6: causal cohort
- **Time-zero truncation test:** zero differences. Outcome time from treatment start is always > 0.
- **ECOG at treatment start:** recorded for 86–88%. **PD-L1 at treatment start:** recorded for 52% of the chemotherapy arm vs 76% of the immunotherapy arm (testing is part of the decision).

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 6.1 | 53% of the chemotherapy arm later receive immunotherapy (median 135 days after start) | Second-line immunotherapy after chemotherapy | **Estimand:** start with immunotherapy vs start with chemotherapy alone, crossover allowed, like intention-to-treat (KEYNOTE-189 also allowed crossover) |
| 6.2 | The newest era has only 36 chemotherapy patients | Immunotherapy became standard | **Limitation:** wide intervals; era is used as a confounder |

## Pass 7: code robustness (unit tests on synthetic data, `tests/test_features.py`)
| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 7.1 | Crash when a timeline is empty for the patients given | pandas `merge(left_on=…, right_index=True)` on an empty frame returns the right index, named like the `PATIENT_ID` column | **Fixed** (outputs on the full data unchanged) |
| 7.2 | Crash when a metastatic site never occurs in the input | Columns were taken from the data, not from the fixed site list | **Fixed** (outputs unchanged) |

## Pass 8: training code (Part II), checked before the full run
Checked against independent implementations:
- Kaplan–Meier, Greenwood and the log(−log) CI against lifelines (identical);
- the censoring distribution against lifelines (identical);
- the KKT gradient on synthetic data, with and without ties.

New tests in `tests/test_models.py` (synthetic data only, 18 tests in total).

| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 8.1 | XGBoost crashes (segfault) when torch was imported first; torch training hangs when XGBoost was imported first | Two OpenMP runtimes in one process: xgboost links Homebrew's `libomp`, torch ships its own copy | **Fixed:** xgboost and torch are imported only inside the functions that need them; each learner trains in its own process; tests that need them run in subprocesses; a test guards that the package imports neither |
| 8.2 | Elastic-net Cox with the default tolerance (1e-7): the KKT optimality conditions are violated by up to 18% of α at small penalties | The stopping rule watches the size of coefficient updates, not optimality. Slow progress near the optimum looks like convergence | **Fixed:** tolerance 1e-9 in CV, 1e-10 for the final fit; the KKT violation is reported. Predictions change only in the 5th decimal |
| 8.3 | Unknown smoking imputed with a single value: trees and neural nets can isolate that value and relearn "unknown" (36% vs 18% 12-month deaths) | Any distinct fill value is a split point for flexible models | **Fixed:** fractional imputation for XGBoost and the network (one row as never-smoker with weight p, one as smoker with weight 1 − p; predictions averaged). The linear Cox model uses the within-type prevalence, which it cannot isolate |
| 8.4 | Metrics with censoring weights borrowed from other patients would mis-weight the test era | Censoring depends on era (2% vs 36% censored before the horizon) | **Fixed:** the censoring distribution is estimated on the evaluated patients themselves |
| 8.5 | The IPCW share of deaths differed from the Kaplan–Meier risk by 2·10⁻⁵ | Tie convention: patients dying on a day were also counted as at risk of censoring that day | **Fixed:** deaths come before censorings; the identity is now exact and the weights sum to n |
| 8.6 | `NaN` in the censoring curve if everyone still at risk dies on the last day | 0/0 | **Fixed** (factor 1) |
| 8.7 | HR/HER2 values in 0.1% of non-breast patients | They belong to an earlier breast cancer | **Fixed:** set to 0 outside breast cancer |
| 8.8 | `metastatic_site` has duplicate categories ("Lymph Node" vs "Lymph node") | Free-text entry | **Fixed:** lower-cased |
| 8.9 | Finding the largest useful penalty took minutes | A 2-point path jumped straight to a tiny penalty (a nearly unpenalised fit with 2,125 columns) | **Fixed:** 2-point path with the second point at 0.99·α_max (2 s) |
| 8.10 | Causal: 14 of 186 chemotherapy-arm patients (8%) added immunotherapy at cycle 2; 19 (10%) within 42 days | The 14-day regimen window is shorter than one 21-day chemotherapy cycle. Pembrolizumab is often added once PD-L1 is known | **Sensitivity analysis:** strategy defined by day 42, time zero moved to day 42 (no immortal time). Primary definition kept as pre-specified |
| 8.11 | Causal: the IPTW estimate is close to null (HR ≈ 1.0) vs KEYNOTE-189 (HR 0.49) | Not a coding error: arms, agents, time zero and events were hand-checked. Calendar time within an era is a confounder that the era flag misses. MSK patient IDs follow registration order. Within IMPACT468, chemotherapy-alone patients have a median ID of 23,110 vs 39,215 for immunotherapy, and the immunotherapy share rises from 22% to 91% across ID quintiles. Further causes: unrecorded reasons for chemotherapy alone in the immunotherapy era; 53% second-line immunotherapy in the chemotherapy arm | **Sensitivity analysis:** adjust for the patient-ID calendar proxy. **Limitation (result):** the remaining gap to KEYNOTE-189 is reported as a failed benchmark check, consistent with the Part II verdict "not identified" |
| 8.12 | A rerun of the chosen XGBoost settings gave different tree counts (fold 0: 1,070 vs 1,488) | I typed the learning rate rounded (0.0115 instead of 0.011520). XGBoost itself is deterministic: the same trees for 1, 2 or 8 threads and any round limit | **Fixed:** the exact value is read from the saved results. Side finding: early stopping's loss is flat near its minimum, so the number of trees is poorly determined while the CV error is not (Brier within 0.0002) |

## Changes vs the submitted Problem Definition Reports
- **Part I, §4:** "age … taken at data release" is wrong. Age is at death for deceased patients and is excluded (E1).
- **Part II, causal:** stage 4 NSCLC 3,301 → 3,292 (diagnosis matching, 1.2). Comparison 479 vs 201 → **470 vs 186** (histology restriction, 2.1).
- **Part II, prediction:**
  - Landmark set 21,599 → 21,598 (one haematology-panel patient excluded).
  - Smoking is now `never_smoker`, with Unknown imputed without an indicator (3.1).
  - PD-L1 is flagged for drift (5.1).
- **Part II, prediction, validation:**
  - **Tuning:** the report said "tune on held-out IMPACT468 patients". Hyperparameters are now chosen by 5-fold CV inside the training eras, stratified by cancer type × era × 12-month status.
  - **Why:** more data per decision and lower variance, and genes added in IMPACT468 can be learned.
  - **Forward check:** train IMPACT341+410, validate IMPACT468, reported for the chosen settings.
- **Part II, prediction, models:** the report compared a Cox model with a random survival forest. Following Model Development Part I, the candidates are:
  - elastic-net Cox with cancer-type interactions;
  - XGBoost on the IPCW 12-month log loss;
  - a neural discrete-time model.

## Pass 9: production (Part III), before the release
| # | Issue | Root cause | Resolution |
|---|---|---|---|
| 9.1 | A negative value (e.g. TMB −2) would be silently clipped to 0 before the model | `prepare()` clips before `log1p`, which is right for valid data but hides impossible inputs | **Fixed:** `RiskPredictor.check` refuses impossible values before any transformation |
| 9.2 | The synthetic golden file would be missing in GitHub Actions | `*.csv` in `.gitignore` (meant for data) also matched `tests/golden/*.csv` | **Fixed:** explicit exception. A clean copy of the tracked files passes 38 tests (6 skipped: licensed data) |
| 9.3 | A patient ID with clinical details in this log (finding 1.2) | Copied from a data check | **Fixed:** anonymised before the repository was created. No patient IDs in tracked files (scan) |
| 9.4 | Range warnings fired for 0/1 gene columns and bounded probabilities, which is noise | The training min/max rule was applied to every column | **Fixed:** range warnings only for unbounded columns (counts, markers, TMB, ratios) |

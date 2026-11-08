# chordsurv: 12-month risk of death at the 6-month follow-up (MSK-CHORD)

For a patient alive 6 months after their first tumour sequencing, `chordsurv` predicts the risk of dying within the next 12 months. It covers five cancers: non-small cell lung, breast, colorectal, prostate and pancreatic. It uses what is known at sequencing and what happened in months 0–6: NLP progression scores from radiology reports, new metastatic sites, ECOG, tumour markers and treatments. It also re-estimates the two other results of the project:
- 24-month survival by cancer type, with confidence intervals;
- the comparison of immunotherapy vs chemotherapy as first treatment in stage 4 lung cancer.

**For whom:** researchers with access to MSK-CHORD (or a later release in the same cBioPortal format) who want to:
- refit the model on new data;
- score patients at their 6-month follow-up;
- re-run the estimates.

The models and settings come from the Model Development Report (Parts I and II, semester project in Modelling in Data Science, VŠE). The risk is a research output, not a clinical tool.

## Install
```bash
pip install "git+https://github.com/OndrejZ24/cancer-research-mock-prod@v0.1.0"
```
Python 3.12 or 3.13. On macOS, XGBoost also needs `brew install libomp`. For development: `uv sync` (versions pinned in `uv.lock`, CPU-only torch on Linux).

## Use
Without any data, `chordsurv demo` shows the whole flow on synthetic patients (train, predict, refuse bad inputs). `notebooks/usage_example.ipynb` does the same step by step.

With the real data (download described in [`data/README.md`](data/README.md)):
```bash
chordsurv build                                     # raw export -> processed tables (about 10 s)
chordsurv train --learner xgb --features landmark   # settings chosen in Part II; training eras only
chordsurv predict --model model.pkl --patients ids.txt --out risks.csv
chordsurv describe --boot 1000                      # 24-month survival by cancer type, 95 % CIs
chordsurv effect --analysis all                     # immunotherapy vs chemotherapy, bootstrap CIs
```
In Python:
```python
from chordsurv.api import RiskPredictor
model = RiskPredictor("xgb", "landmark").fit(train_table, feature_groups)
model.predict(new_patients)        # risk_12m + issues; bad rows are refused (on_error="drop" -> NaN)
```

**Options:**
- **Learners:** `km` (baseline), `coxnet` (elastic-net Cox), `xgb` (XGBoost), `nn` (neural discrete-time).
- **Feature sets:**
  - `day0`: information at sequencing only;
  - `landmark`: the default;
  - `landmark_counts`: adds MSK's report counts, so valid at MSK only;
  - `landmark_no_pdl1`: without PD-L1.

**Inputs:** one row per patient (index = patient id) with the feature columns written by `chordsurv build`.
- **Refused:**
  - missing columns;
  - cancer types other than the five;
  - impossible values (ECOG outside 0–5, probabilities outside [0, 1], negative counts, flags other than 0/1, text in numeric columns);
  - patients not alive at the landmark.
- **Warned:**
  - values outside the training range;
  - categories never seen in training;
  - patients with no information from months 0–6.
- **Missing values:** handled as in training.

## How it is tested
`uv run pytest` runs 44 tests. GitHub Actions runs them on every push, on synthetic data. The real-data tests in `tests/test_real_data.py` run only where the licensed data are present.

| Requirement | Tests |
|---|---|
| Same answers as the Part II model | The predictor returns exactly the `LandmarkModel` predictions (Cox, XGBoost and the network). Saved risks of 20 synthetic patients are reproduced. On the real data, the package reproduces the Part II Cox model (to 1e-9) and XGBoost model (exactly) for 300 patients. |
| Only information known at the landmark | Adding events after day 182 leaves every landmark feature unchanged, on synthetic data and on 1,000 real patients with timelines cut at day 182. Outcome columns are never inputs, and a leaked outcome does not change the risk. |
| Bad inputs | Each impossible value is refused (or its row dropped with NaN while the other rows keep their risks). Out-of-range values and unseen categories give warnings. Missing values are handled. |
| Reproducible | Fixed seeds; identical predictions after save/load and across runs (also checked once: XGBoost gives identical trees with 1, 2 or 8 threads). The model stores the package versions and a fingerprint of its training data, and loading warns when versions differ. |
| Honest uncertainty | Simulated cohorts with known survival: the 80 % and 95 % Kaplan–Meier intervals cover the truth at their nominal rate. On the real data, the out-of-fold mean risk is within 1.5 points of the observed 12-month mortality. |

## What to monitor in use
- **Input drift against the training profile:**
  - share of patients with PD-L1 recorded (11 % → 28 % between eras) and positive;
  - ECOG missingness (52 % → 40 %);
  - cancer-type mix;
  - share of patients with no radiology report in months 0–6.
- **Error over time:** once 12-month outcomes mature, compute the Brier score and calibration by cancer type for each new release.
- **When to retrain:** when the Brier score is clearly worse than in validation, or calibration in the high-risk group drifts. In any case retrain with every new data release, after the final test on the newest era (`chordsurv train --eras all`).

## Layout
- **`src/chordsurv/`:** the package:
  - `pipeline.py` (build), `features.py`, `cohorts.py`, `data.py` (feature sets), `preprocess.py`;
  - `models.py` (`LandmarkModel`), `boosting.py`, `nnsurv.py`;
  - `api.py` (`RiskPredictor`, input checks), `cli.py`;
  - `estimates.py`, `causal.py`, `survival.py`, `metrics.py`;
  - `settings.json` (hyperparameters chosen in Part II), `synthetic.py`.
- **`scripts/`:** reproduce the reports (`train_prediction.py`, `fit_inference_causal.py`, `extend_search.py`).
- **`notebooks/`:** Part I checks, Part II tables and figures, usage example.
- **`results/`:** aggregate results only.
- **`REVIEW_LOG.md`:** every data and code issue found, with its root cause.

XGBoost and torch load two different OpenMP runtimes, which crash or freeze in one process. The package therefore imports each only when needed, and training runs one learner per process.

## Licence and data
The code is for teaching and research. MSK-CHORD is CC BY-NC-ND 4.0, so no data, processed tables, patient-level predictions or fitted models are part of this repository. Cite Jee et al. (2024), *Nature* 636, 728–736, when using the data.

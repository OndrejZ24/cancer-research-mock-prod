# Data

The code needs MSK-CHORD (Memorial Sloan Kettering Cancer Center; Jee et al., *Nature* 2024). The licence is CC BY-NC-ND 4.0, so the data are **not** in this repository and must never be committed.

1. Open https://www.cbioportal.org/study/summary?id=msk_chord_2024 and download the study (the download icon next to the study name gives `msk_chord_2024.tar.gz`).
2. Unpack it so that the text files sit in `../data/msk_chord_2024/` relative to this repository (or anywhere else, and set `CHORD_DATA_DIR`). The files used are:
   - `data_clinical_patient.txt`, `data_clinical_sample.txt`;
   - all `data_timeline_*.txt`;
   - `data_mutations.txt`, `data_cna.txt`, `data_sv.txt`;
   - `data_gene_panel_matrix.txt`.
3. Run `chordsurv build`. It writes the processed tables to `data/processed/` (also git-ignored).

Everything derived from the data at patient level stays local, and `.gitignore` keeps it out of git:
- `data/processed/`;
- fitted models and out-of-fold predictions in `artifacts/`.

The tests in `tests/test_real_data.py` use these files and are skipped when they are missing. All other tests run on synthetic data (`chordsurv.synthetic`).

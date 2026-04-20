# Pipelines

This directory is documentation-only. It does not contain an additional
pipeline implementation layer.

The actual code lives in `surveys/`, `lib/`, `models/`, and `catalog/`.

## 1. Training Workflows

### 1.1 Photo-z Model Training

1. Data download
   - LSDR10: `surveys/LSDR10/download.py`
   - PS1DR2: `surveys/PS1DR2/download.py`
   - unWISE: `surveys/unWISE/download.py`
   - SDSS DR19: this repository currently keeps the cleaning step only in
     `surveys/SDSSDR19/data_clean.py`; raw download follows the external
     workflow described in `legacy/list.md`
   - DESI DR1: this repository currently keeps the cleaning step only in
     `surveys/DESIDR1/data_clean.py`; raw download follows the official
     DESI data release workflow

2. Spectroscopic catalog cleaning
   - SDSS DR19:
     `surveys.SDSSDR19.data_clean.clean_sdss_dr19()`
   - DESI DR1:
     - `surveys.DESIDR1.data_clean.clean_bgs()`
     - `surveys.DESIDR1.data_clean.clean_elg()`
     - `surveys.DESIDR1.data_clean.clean_lrg()`
     - `surveys.DESIDR1.data_clean.merge_clean_catalogs()`

3. Re-bucketing and cross-match
   - RA-sliced FITS regrouping and optional UID assignment:
     `lib.recollect.recollect_fits_by_ra()`
   - Sky cross-match through STILTS:
     `lib.crossmatch.stilts_crossmatch()`

4. Photometric processing and cleaning
   - PS1 extinction correction:
     - `surveys.PS1DR2.dered.query_ebv()`
     - `surveys.PS1DR2.dered.apply_extinction()`
   - LSDR10 training cleaning:
     `surveys.LSDR10.data_clean.clean_for_training()`
   - PS1DR2 training cleaning:
     `surveys.PS1DR2.data_clean.clean_for_training()`
   - unWISE cleaning:
     `surveys.unWISE.data_clean.clean()`

5. Train / validation / test split
   - There is currently no standalone shared split script in this repository.
   - NNC supports two split modes:
     - explicit `train` / `val` / `test` files via
       `models/photoz/NNC/config.yaml`
     - single-file ratio split through `dataset_mode.type = "ratio"` in
       `models/photoz/NNC/core.py`
   - The galaxy classifier ANN follows the same pattern through
     `models/galaxy_clf/ANN/galaxyClf_ann_config.yaml` and
     `models/galaxy_clf/ANN/galaxyClf_ann.py`

6. Model training
   - NNC:
     `python -m models.photoz.NNC.train --config <config.yaml>`
   - ANN:
     `python -m models.photoz.ANN.train --config <config.yaml>`
   - Other experimental model families are under `models/photoz/`

7. Prediction / evaluation
   - NNC:
     `python -m models.photoz.NNC.predict --input ... --model-dir ... --output ...`
   - ANN:
     `python -m models.photoz.ANN.predict --input ... --model-dir ... --output ...`

### 1.2 Galaxy / Non-Galaxy Classifier Training

1. Data source and sample construction
   - LSDR10:
     see `surveys/LSDR10/download_pos_neg_sample.py`
   - PS1DR2:
     non-galaxy samples are downloaded from CasJobs using the
     `iPSFMag - iKronMag` relation

2. Data cleaning and labeling notes
   - LSDR10:
     see `surveys/LSDR10/download_pos_neg_sample.py` and
     `surveys/LSDR10/galaxyclf.ipynb`
   - PS1DR2:
     see `surveys/PS1DR2/download.py`
   - The classifier training code expects the final input table to already
     contain a binary `label` column.

3. Classifier training
   - ANN:
     `python models/galaxy_clf/ANN/galaxyClf_ann.py --config <config.yaml>`
   - XGBoost:
     implementation lives in `models/galaxy_clf/XGBoost/galaxyClf_xgb.py`

4. Train / validation / test split
   - ANN supports both `files` and `ratio` modes through
     `models/galaxy_clf/ANN/galaxyClf_ann_config.yaml`
   - The split logic is implemented inside
     `models/galaxy_clf/ANN/galaxyClf_ann.py`

## 2. Release Data Workflow

1. Raw data source
   - Reuse the survey raw data from the training workflow.
   - PS1DR2 dereddening is handled by `surveys/PS1DR2/dered.py`.

2. Release-data cleaning
   - LSDR10:
     `surveys.LSDR10.data_clean.clean_for_release()`
   - PS1DR2:
     `surveys.PS1DR2.data_clean.clean_for_release()`
   - unWISE:
     `surveys.unWISE.data_clean.clean()`

3. Photo-z prediction and galaxy classification
   - LSDR10:
     `python -m catalog --stage lsdr10`
   - PS1DR2:
     `python -m catalog --stage ps1dr2`
   - The implementation is in `catalog/predict.py`.
   - If `catalog/config.yaml` provides `galaxy_clf_ls` or `galaxy_clf_ps`,
     the catalog prediction stage also computes `galaxy_prob`.

4. Cross-survey merge
   - `python -m catalog --stage merge`
   - Implemented in `catalog/merge.py`

5. Publish and validation
   - `python -m catalog --stage publish`
   - `python -m catalog --stage check`
   - Implemented in `catalog/publish.py`

## 3. Notes

- This directory records workflows only. It does not duplicate logic that
  already exists elsewhere in the repository.
- If a step does not yet have a dedicated script in the current repository,
  that is stated explicitly here instead of introducing a placeholder wrapper.
- Historical notebooks and legacy process notes are kept in `legacy/list.md`.

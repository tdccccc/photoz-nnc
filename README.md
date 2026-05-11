# photoz-nnc

Photometric redshift PDFs via neural network classification for DESI Legacy Imaging Surveys and Pan-STARRS.

This repository accompanies the manuscript:

**Photometric Redshift PDFs via Neural Network Classification for DESI Legacy Imaging Surveys and Pan-STARRS** — [arXiv:2602.01548](https://arxiv.org/abs/2602.01548)

## Status

This repository hosts the data preparation, model training, and catalog
production code used in the paper. Additional examples and user-facing
documentation are still being added.

See [`pipelines/README.md`](pipelines/README.md) for end-to-end workflows.

## Project Overview

We develop a neural network classification (NNC) framework for photometric redshift estimation that:

- predicts full redshift probability density functions (PDFs) rather than only point estimates,
- optimizes the Continuous Ranked Probability Score (CRPS),
- produces well-calibrated photo-z PDFs,
- is applied to both LSDR10 and PS1DR2-based samples.

The associated paper also presents a public photometric redshift catalog and PDF products.

## Data Release

The catalog associated with this work is available at Zenodo:

- https://doi.org/10.5281/zenodo.18410731

For practical use and download efficiency, the public data release provides **40-bin redshift PDFs** obtained by rebinning the native **400-bin model output**.

## Installation

```bash
git clone https://github.com/tdccccc/photoz-nnc.git
cd photoz-nnc
pip install -e .
```

`pip install -e .` builds the C++ extension under `lib/cpp_src` via
pybind11 and exposes `lib`, `surveys`, and `models` as importable
packages.

## Usage

End-to-end training and release workflows are documented in
[`pipelines/README.md`](pipelines/README.md). Common entry points:

- Train NNC: `python -m models.photoz.NNC.train --config <config.yaml>`
- Predict with NNC: `python -m models.photoz.NNC.predict --input ... --model-dir ... --output ...`
- Build the released catalog: `python -m catalog --stage <lsdr10|ps1dr2|merge|publish|check>`

## Repository Layout

- `lib/` — shared utilities (I/O, cross-match, metrics, PIT, recollect, ...)
- `surveys/` — per-survey download and cleaning helpers (LSDR10, PS1DR2, unWISE, SDSSDR19, DESIDR1)
- `models/` — per-algorithm training and prediction code (NNC, ANN, XGBoost, RandomForest, KNN, LSTM, Ensemble; plus galaxy classifier)
- `catalog/` — released-catalog prediction, merge, and publish pipeline
- `pipelines/` — workflow documentation
- `figures/` — figure-generation notebooks
- `legacy/` — original source files and development notes from earlier stages of the project; kept for reference only (paths inside are author-local and not reproducible as-is)

## Citation

If you use the catalog or code associated with this project, please cite the corresponding paper and the Zenodo release.

## Contact

For questions regarding the current repository status or early access inquiries, please contact:

- Jun-Qing Xia — xiajq@bnu.edu.cn
- Da-Chuan Tian — tiandc@mail.bnu.edu.cn

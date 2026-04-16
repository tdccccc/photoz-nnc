# photoz-nnc

Photometric redshift PDFs via neural network classification for DESI Legacy Imaging Surveys and Pan-STARRS.

This repository accompanies the manuscript:

**Photometric Redshift PDFs via Neural Network Classification for DESI Legacy Imaging Surveys and Pan-STARRS**

## Status

This repository has been created in advance of the full public code release.

The codebase is currently being cleaned up, reorganized, and documented.  
A complete and user-friendly release will be uploaded after the final preparation is finished.

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

## Planned Repository Contents

The full release is expected to include:

- model training and inference scripts,
- data preprocessing utilities,
- calibration and evaluation scripts,
- figure-generation notebooks/scripts,
- example workflows for reproducing the main results in the paper.

## Citation

If you use the catalog or code associated with this project, please cite the corresponding paper and the Zenodo release.

## Contact

For questions regarding the current repository status or early access inquiries, please contact:

- Jun-Qing Xia — xiajq@bnu.edu.cn
- Da-Chuan Tian — tiandc@mail.bnu.edu.cn

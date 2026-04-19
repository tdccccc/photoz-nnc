"""ANN photo-z package — regression-based photo-z with an MLP.

Treats photometric redshift estimation as a direct scalar regression:
the model outputs a single value, and training minimises MSE, MAE,
Huber, or Smooth-L1 loss.

Layout
------
* :mod:`model` — :class:`~.model.PhotozRegressor`.
* :mod:`dataset` — :class:`~.dataset.DatasetPhotozRegression`.
* :mod:`core` — training infra (:class:`~.core.Config`,
  :class:`~.core.Trainer`, :class:`~.core.EarlyStopping`, …).
* :mod:`inference` — :class:`~.inference.PhotozRegressionInference`.
* :mod:`train` / :mod:`predict` — CLI entry points.

Registries
----------
:data:`MODELS` and :data:`DATASETS` map the config-yaml string
names to the actual classes. Add new variants here and reference
them by name in ``config.yaml``.
"""

from .dataset import DatasetPhotozRegression
from .model import PhotozRegressor

__all__ = ["MODELS", "DATASETS"]

MODELS = {
    "PhotozRegressor": PhotozRegressor,
}

DATASETS = {
    "DatasetPhotozRegression": DatasetPhotozRegression,
}

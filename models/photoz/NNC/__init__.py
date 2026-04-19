"""NNC photo-z package — neural-network classifier for photo-z.

Treats photometric redshift estimation as an N-bin classification
problem: the model emits one logit per redshift bin; the softmax
over logits is the predicted PDF.

Layout
------
* :mod:`model` — :class:`~.model.PhotozBinningClassifier` (user edits
  to change the architecture).
* :mod:`dataset` — :class:`~.dataset.DatasetPhotozBinned` (user edits
  to change features / label handling).
* :mod:`losses` — the seven training losses and bin/sample weight
  helpers (user picks via ``config.yaml``).
* :mod:`core` — training-time infrastructure (:class:`~.core.Config`,
  :class:`~.core.Trainer`, early stopping, samplers, etc.).
* :mod:`inference` — :class:`~.inference.PhotozBinInference` and
  :class:`~.inference.TemperatureCalibrator`.
* :mod:`train` / :mod:`predict` — CLI entry points.

Registries
----------
:data:`MODELS` and :data:`DATASETS` map the string class names used
in ``config.yaml`` to the actual Python classes. When adding a new
model or dataset variant:

1. Define the class in :mod:`model` / :mod:`dataset`.
2. Add one line to the registry here.
3. Reference it by name in ``config.yaml``.
"""

from .dataset import DatasetPhotozBinned
from .model import PhotozBinningClassifier

__all__ = ["MODELS", "DATASETS"]


#: ``Config.model_class_name`` → model class.
MODELS = {
    "PhotozBinningClassifier": PhotozBinningClassifier,
}

#: ``Config.dataset_class_name`` → dataset class.
DATASETS = {
    "DatasetPhotozBinned": DatasetPhotozBinned,
}

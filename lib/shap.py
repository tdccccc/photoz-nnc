"""SHAP-based feature importance for classification-style photo-z models.

Wraps :class:`shap.KernelExplainer` so we can get per-feature attributions
for NNC-style models that output a probability distribution over
redshift bins. The attributions target the **expected redshift**
``E[z] = sum_i p_i * bin_center_i``, not the raw logits or any
individual bin probability — this keeps the feature importances
interpretable as "how does each input feature push the predicted
redshift up or down".

Dependencies
------------
Install via pip::

    pip install shap torch h5py

The :func:`compute_shap_importance` helper expects an inference-side
object that exposes:

* ``model`` — a callable (typically a :class:`torch.nn.Module`)
  mapping a feature matrix to per-bin logits.
* ``device`` — a :class:`torch.device` compatible value.
* ``_create_dataset(path, mode)`` — factory returning an object
  with ``.features`` (``np.ndarray``), ``.feature_cols``
  (list of names), and ``.bin_centers`` (``np.ndarray`` of
  length ``n_bins``).

This matches the shape of the project's
:class:`PhotozBinInference`, but the helper is agnostic to the
exact class name as long as those attributes are present.

Example:
    >>> from lib.shap import compute_shap_importance
    >>> result = compute_shap_importance(inference, "validation.fits")
    >>> result["importance_df"].head(10)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

__all__ = ["compute_shap_importance"]


def compute_shap_importance(
    inference: Any,
    data_path: str,
    *,
    n_background: int = 200,
    n_samples: int = 2000,
    random_state: int = 42,
    save_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Compute SHAP feature importances for an NNC-style photo-z model.

    Uses :class:`shap.KernelExplainer` to attribute the predicted
    expected redshift to each input feature, averaged over
    ``n_samples`` test rows. Runtime scales roughly as
    ``O(n_samples * n_background * n_features)`` — the default
    ``n_background=200, n_samples=2000`` typically runs in minutes
    on a GPU-backed inference object.

    Args:
        inference: Inference-side object exposing ``model``,
            ``device``, and ``_create_dataset(path, mode)``. See the
            module docstring for the attributes that the returned
            dataset object must provide.
        data_path: Path to the FITS / HDF5 / ... file passed to
            ``inference._create_dataset``. A single ``"validation"``
            split is created internally.
        n_background: Number of rows sampled for the
            KernelExplainer background distribution. Typical range
            50-500; higher is more accurate but slower.
        n_samples: Number of rows whose SHAP values are computed.
            Runtime grows linearly with this.
        random_state: Seed for :func:`numpy.random.seed`, so the
            background / test sub-samples are reproducible across
            runs.
        save_dir: If given, write the SHAP arrays to
            ``{save_dir}/shap_feature_importance.h5`` (gzip
            compressed). The directory is created if missing.

    Returns:
        Dict with keys:

        * ``"shap_values"`` — ``np.ndarray`` of shape
          ``(n_samples, n_features)``.
        * ``"X_test"`` — ``np.ndarray`` of shape
          ``(n_samples, n_features)``, the feature matrix SHAP was
          computed on.
        * ``"feature_names"`` — list of column names.
        * ``"mean_abs_shap"`` — ``np.ndarray[n_features]``, the
          per-feature mean of ``|shap_value|`` across test rows.
        * ``"importance_df"`` — :class:`pandas.DataFrame` with
          columns ``feature``, ``importance``,
          ``importance_normalized``, ``rank``, sorted descending by
          importance.

    Raises:
        AttributeError: ``inference`` is missing one of the
            attributes listed in the module docstring.
        ImportError: ``shap`` or ``torch`` is not installed.
    """
    import shap
    import torch

    np.random.seed(random_state)

    # ---- Load dataset via the inference object's own factory ------------
    ds = inference._create_dataset(data_path, mode="validation")
    X = ds.features
    feature_names = ds.feature_cols
    bin_centers = ds.bin_centers

    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Background samples: {n_background}, test samples: {n_samples}")

    bg_idx = np.random.choice(len(X), min(n_background, len(X)), replace=False)
    test_idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_background = X[bg_idx]
    X_test = X[test_idx]

    # ---- Prediction function: logits -> softmax -> expected redshift ----
    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.FloatTensor(x).to(inference.device)
        inference.model.eval()
        with torch.no_grad():
            logits = inference.model(x_tensor).cpu().numpy()
        # Numerically stable softmax, then weighted sum over bin centers.
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        return (probs * bin_centers).sum(axis=1)

    # ---- SHAP ----------------------------------------------------------
    print("\nBuilding SHAP KernelExplainer ...")
    explainer = shap.KernelExplainer(predict_fn, X_background)
    print("Computing SHAP values (this can take several minutes) ...")
    shap_values = explainer.shap_values(X_test, silent=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # ---- Report --------------------------------------------------------
    print("\nFeature importance (mean |SHAP|):")
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for rank, idx in enumerate(sorted_idx):
        print(
            f"  [{rank + 1:2d}] {feature_names[idx]:20s}: "
            f"{mean_abs_shap[idx]:.6f}"
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": mean_abs_shap,
            "importance_normalized": mean_abs_shap / mean_abs_shap.max(),
        }
    )
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    # ---- Optional persistence ------------------------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "shap_feature_importance.h5")
        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "shap_values", data=shap_values, compression="gzip"
            )
            f.create_dataset("X_test", data=X_test, compression="gzip")
            f.create_dataset(
                "feature_names", data=np.array(feature_names, dtype="S")
            )
            f.create_dataset("mean_abs_shap", data=mean_abs_shap)
            f.create_dataset(
                "mean_abs_shap_normalized",
                data=mean_abs_shap / mean_abs_shap.max(),
            )
        print(f"\nSaved SHAP results to: {file_path}")

    return {
        "shap_values": shap_values,
        "X_test": X_test,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "importance_df": importance_df,
    }

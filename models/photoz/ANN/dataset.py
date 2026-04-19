"""Dataset class for regression-based photo-z training.

:class:`DatasetPhotozRegression` reads a catalog, picks feature
columns, and exposes the scalar redshift label as a ``(1,)`` target
tensor. Optionally adds label noise during training (Gaussian,
uniform, or their redshift-relative variants).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from lib.io import readfile

__all__ = ["DatasetPhotozRegression"]


class DatasetPhotozRegression(Dataset):
    """Photo-z catalog → scalar-label PyTorch dataset.

    Exactly one of ``dataset`` and ``file_path`` must be given.

    Args:
        dataset: In-memory DataFrame. Mutually exclusive with
            ``file_path``.
        file_path: Path to a catalog file (any extension supported
            by :func:`lib.io.readfile`).
        feature_generation: Optional spec for
            :meth:`_choose_features`.
        feature_columns: Explicit feature column names (takes
            priority over ``feature_generation``).
        label_noise: Optional noise config applied only when
            ``mode='train'``. Dict with ``type`` (``'gaussian'``,
            ``'uniform'``, ``'relative_gaussian'``,
            ``'relative_uniform'``) and ``level`` (float, 0 =
            disabled).
        scaler_X: Optional pre-fitted :class:`StandardScaler`.
        mode: ``'train'`` / ``'validation'`` / ``'test'`` /
            ``'inference'``. The first three expose labels.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
        feature_generation: Optional[dict[str, Any]] = None,
        feature_columns: Optional[list[str]] = None,
        label_noise: Optional[dict[str, Any]] = None,
        scaler_X: Optional[StandardScaler] = None,
        mode: str = "train",
    ):
        # --- Source table ---------------------------------------------------
        if dataset is not None and file_path is not None:
            raise ValueError("Cannot specify both 'dataset' and 'file_path'.")
        if dataset is None and file_path is None:
            raise ValueError("Must provide 'dataset' or 'file_path'.")

        if file_path is not None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            self.dataset = readfile(file_path)
        else:
            self.dataset = dataset

        # --- Feature columns ------------------------------------------------
        if feature_columns is not None:
            self.feature_cols = feature_columns
        elif feature_generation is not None:
            self.mag_types = feature_generation.get("mag_types")
            self.color_types = feature_generation.get("color_types")
            self.use_mag = feature_generation.get("use_mag")
            self.use_magerr = feature_generation.get("use_magerr")
            self.use_colorerr = feature_generation.get("use_colorerr")
            self.feature_cols = self._choose_features(
                self.mag_types,
                self.color_types,
                use_mag=self.use_mag,
                use_magerr=self.use_magerr,
                use_colorerr=self.use_colorerr,
            )
        else:
            raise ValueError(
                "Either 'feature_columns' or 'feature_generation' "
                "must be provided."
            )
        self.features = self.dataset[self.feature_cols].values.astype(np.float32)

        # --- Labels ---------------------------------------------------------
        self.mode = mode
        if self.mode in ("train", "validation", "test"):
            self.labels = self.dataset["z"].values.astype(np.float32)

            if label_noise is not None and mode == "train":
                self._apply_label_noise(label_noise)

        # --- Standardisation ------------------------------------------------
        if scaler_X is not None:
            self.scaler_X = scaler_X
            self.features = self.scaler_X.transform(self.features)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _choose_features(
        self,
        mag_types: Optional[list[str]],
        color_types: Optional[list[str]],
        use_mag: bool = True,
        use_magerr: bool = True,
        use_colorerr: bool = True,
    ) -> list[str]:
        """Build the feature column list for the grizy band layout."""
        bands = ["g", "r", "i", "z", "y"]
        colours = ["gr", "ri", "iz", "zy"]
        mag_cols: list[str] = []
        err_cols: list[str] = []
        if color_types:
            for color_type in color_types:
                for c in colours:
                    mag_cols.append(f"{color_type}_{c}")
                    if use_colorerr:
                        err_cols.append(f"{color_type}_{c}_Err")
        if use_mag and mag_types:
            for b in bands:
                for m in mag_types:
                    mag_cols.append(f"{b}{m}Mag_dered")
                    if use_magerr:
                        err_cols.append(f"{b}{m}MagErr")
        return mag_cols + err_cols

    # ------------------------------------------------------------------
    # Label noise
    # ------------------------------------------------------------------

    def _apply_label_noise(self, noise_cfg: dict[str, Any]) -> None:
        """Add noise to training labels for regularisation."""
        noise_type = noise_cfg.get("type", "gaussian")
        level = noise_cfg.get("level", 0.0)
        if level <= 0:
            return

        n = self.labels.shape
        if noise_type == "gaussian":
            noise = np.random.normal(0, level, size=n)
        elif noise_type == "uniform":
            noise = np.random.uniform(-level, level, size=n)
        elif noise_type == "relative_gaussian":
            noise = np.random.normal(0, 1, size=n) * level * (1 + self.labels)
        elif noise_type == "relative_uniform":
            noise = np.random.uniform(-1, 1, size=n) * level * (1 + self.labels)
        else:
            raise ValueError(f"Unknown noise type: {noise_type!r}")

        self.labels = np.maximum(self.labels + noise.astype(np.float32), 0.0)
        logging.info(
            f"Added {noise_type} noise (level={level}) to training labels."
        )

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.FloatTensor(self.features[idx])
        if self.mode in ("train", "validation", "test"):
            return {"input": x, "target": torch.FloatTensor([self.labels[idx]])}
        return {"input": x}

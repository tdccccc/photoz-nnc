"""Dataset class for bin-classifier photo-z training.

:class:`DatasetPhotozBinned` does three jobs that users typically
tweak when re-training on a new survey or feature set:

1. Reads a catalog from disk (or takes a :class:`pandas.DataFrame`
   in-memory) via :func:`lib.io.readfile`.
2. Picks feature columns either from an explicit list or from a
   ``feature_generation`` spec (which is what the project-specific
   grizy helper :meth:`_choose_features` implements).
3. Converts the scalar redshift label into a soft probability vector
   over the bin grid (linear interpolation between neighbouring
   bins, with optional Gaussian broadening).

Downstream training / inference code in :mod:`core` only ever pokes
at :attr:`DatasetPhotozBinned.features`, :attr:`labels_probs`,
:attr:`labels_cont`, :attr:`bin_edges`, :attr:`bin_centers`, and
:attr:`feature_cols`, so you can subclass or replace this module to
support a different feature layout without touching the trainer.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from lib.io import readfile

__all__ = ["DatasetPhotozBinned"]


class DatasetPhotozBinned(Dataset):
    """Photo-z catalog → soft-binned PyTorch dataset.

    Exactly one of ``dataset`` and ``file_path`` must be given. The
    loaded frame must contain every column referenced by
    ``feature_columns`` / ``feature_generation`` and, when
    ``mode != 'predict'``, the ``label_column`` too.

    Args:
        dataset: In-memory DataFrame. Mutually exclusive with
            ``file_path``.
        file_path: Path to a catalog file (any extension supported
            by :func:`lib.io.readfile`). Mutually exclusive with
            ``dataset``.
        feature_generation: Optional spec passed to
            :meth:`_choose_features`; must contain ``mag_types``,
            ``color_types``, ``use_mag``, ``use_magerr``,
            ``use_colorerr``.
        feature_columns: Explicit feature column names. Takes
            priority over ``feature_generation`` when both are given.
        label_column: Name of the scalar redshift column. Defaults
            to ``"z"``.
        binning_config: Required dict with ``z_min``, ``z_max``,
            ``num_bins``, and optional ``soft_label_width`` (in bin
            units; ``None`` disables Gaussian broadening, values >1
            enable it).
        scaler_X: Optional pre-fitted :class:`StandardScaler`.
            When given, ``features`` are transformed with it at
            construction time.
        mode: One of ``"train"``, ``"validation"``, ``"test"``, or
            ``"predict"``. The first three expose the label arrays;
            ``"predict"`` skips label processing.

    Raises:
        ValueError: Both or neither of ``dataset`` / ``file_path``,
            neither ``feature_columns`` nor ``feature_generation``,
            or missing ``binning_config``.
        FileNotFoundError: ``file_path`` does not exist.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
        feature_generation: Optional[dict[str, Any]] = None,
        feature_columns: Optional[list[str]] = None,
        label_column: str = "z",
        binning_config: Optional[dict[str, Any]] = None,
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
                "Either 'feature_columns' or 'feature_generation' must be provided."
            )
        self.features = self.dataset[self.feature_cols].values.astype(np.float32)

        self.label_column = label_column

        # --- Binning grid (must be set before label processing) -------------
        if binning_config is None:
            raise ValueError("binning_config must be provided.")
        self.z_min = binning_config.get("z_min", 0.0)
        self.z_max = binning_config.get("z_max", 1.2)
        self.num_bins = binning_config.get("num_bins", 120)
        # bin units; None disables Gaussian smoothing, >1 enables it.
        self.soft_label_width = binning_config.get("soft_label_width", None)
        self.bin_edges = np.linspace(self.z_min, self.z_max, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # --- Labels ---------------------------------------------------------
        self.mode = mode
        if self.mode in ("train", "validation", "test"):
            self.labels_cont = self.dataset[self.label_column].values.astype(np.float32)
            self.labels_probs = self._z_to_soft_bins(self.labels_cont)

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
        """Build the feature column list for the grizy PS1 / LS layout.

        This helper is survey-specific: it knows about the ``grizy``
        photometric bands and the ``gr / ri / iz / zy`` adjacent-band
        colours. Re-implement or override when adapting to a survey
        with a different band set.
        """
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
    # Soft binning
    # ------------------------------------------------------------------

    def _z_to_soft_bins(self, z_values: np.ndarray) -> np.ndarray:
        """Convert scalar redshifts into per-bin probability vectors.

        Two rows of processing:

        1. Linear split between the two nearest bins: a source at the
           left edge of bin ``k`` gets all its mass in bin ``k``; at
           the right edge it gets it in bin ``k+1``; in between the
           weights interpolate linearly.
        2. (Optional) Gaussian broadening controlled by
           ``soft_label_width`` (in bin units), applied after the
           linear split.

        The output is row-normalised and dithered by a tiny epsilon
        to avoid exact zeros downstream (KL divergence is unstable
        around 0·log 0).
        """
        z = np.clip(z_values, self.z_min, self.z_max - 1e-8)
        bin_indices = np.floor(
            (z - self.z_min) / (self.z_max - self.z_min) * self.num_bins
        ).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        bin_left_edges = self.bin_edges[bin_indices]
        bin_right_edges = self.bin_edges[bin_indices + 1]
        t = (z - bin_left_edges) / (bin_right_edges - bin_left_edges)

        probs = np.zeros((len(z), self.num_bins), dtype=np.float32)
        left_weight = 1 - t
        right_weight = t

        rows = np.arange(len(z))
        probs[rows, bin_indices] = left_weight

        right_indices = bin_indices + 1
        valid = right_indices < self.num_bins
        probs[rows[valid], right_indices[valid]] = right_weight[valid]
        # Rows that already sit in the last bin: fold right_weight back in.
        invalid = ~valid
        probs[rows[invalid], bin_indices[invalid]] += right_weight[invalid]

        if self.soft_label_width is not None and self.soft_label_width > 1:
            width_bins = int(self.soft_label_width)
            sigma_bins = width_bins / 3  # ±3σ covers width_bins
            kernel = np.exp(
                -0.5 * (np.arange(-width_bins, width_bins + 1) / sigma_bins) ** 2
            )
            kernel = kernel / kernel.sum()
            padded = np.pad(
                probs, ((0, 0), (width_bins, width_bins)), mode="edge"
            )
            smoothed = np.array([
                np.convolve(padded[i], kernel, mode="same")[
                    width_bins:-width_bins
                ]
                for i in range(padded.shape[0])
            ])
            probs = smoothed.astype(np.float32)

        # Row-normalise, then add epsilon and renormalise to avoid
        # exact zeros causing numerical issues in downstream loss.
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(probs_sum, 1e-8, None)
        eps = 1e-8
        probs = probs + eps
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.FloatTensor(self.features[idx])
        if self.mode in ("train", "validation", "test"):
            return {
                "input": x,
                "target_prob": torch.FloatTensor(self.labels_probs[idx]),
                "label_z": torch.FloatTensor([self.labels_cont[idx]]),
            }
        return {"input": x}

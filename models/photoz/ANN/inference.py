"""Inference helper for the ANN photo-z regressor.

Loads a saved model directory (``config.yaml`` + ``scaler.pkl`` +
``best_model.pkl``) and exposes :meth:`predict` for point estimates
and optional MC-dropout uncertainty.

Example:
    >>> from models.photoz.ANN.inference import PhotozRegressionInference
    >>> infer = PhotozRegressionInference("runs/experiment_01/")
    >>> infer.predict("catalog.fits")["predictions"]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import Config

__all__ = ["PhotozRegressionInference"]


class PhotozRegressionInference:
    """Load a saved ANN model directory and run inference.

    Args:
        model_dir: Path to the experiment directory containing
            ``config.yaml``, ``scaler.pkl``, and either
            ``best_model.pkl`` or ``checkpoint.pkl``.
        device: Torch device string. Falls back to CPU with a
            warning when CUDA is requested but unavailable.
        batch_size: Default inference batch size.
        num_workers: ``DataLoader`` num_workers.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        batch_size: int = 4096,
        num_workers: int = 4,
    ):
        self.model_dir = Path(model_dir)
        self.device = self._setup_device(device)

        cfg_path = self.model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        self.config = Config()
        self.config.update_from_yaml(str(cfg_path))

        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        self._load_model()
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def _setup_device(device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU.")
                return torch.device("cpu")
            return torch.device(device if device != "cuda" else "cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        from . import MODELS

        model_path = self.model_dir / "best_model.pkl"
        if not model_path.exists():
            model_path = self.model_dir / "checkpoint.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No model file found in {self.model_dir}"
                )
        if hasattr(self.scaler, "n_features_in_"):
            input_dim = self.scaler.n_features_in_
        else:
            raise ValueError("Cannot determine input dimension from scaler.")

        model_params = dict(self.config.model_params)
        model_params["input_dim"] = input_dim
        model_cls = MODELS.get(self.config.model_class_name)
        if model_cls is None:
            raise ValueError(
                f"Model class {self.config.model_class_name!r} is not "
                f"registered; known: {sorted(MODELS)}."
            )
        self.model = model_cls(**model_params)
        self.model.load_state_dict(
            torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        )
        self.model.to(self.device)
        self.model.eval()

    def _create_dataset(self, data_source):
        from . import DATASETS

        dataset_cls = DATASETS[self.config.dataset_class_name]
        kwargs = dict(
            feature_generation=self.config.dataset_params.get("feature_generation"),
            feature_columns=self.config.dataset_params.get("feature_columns"),
            label_noise=None,
            scaler_X=self.scaler,
            mode="inference",
        )
        if isinstance(data_source, str):
            return dataset_cls(file_path=data_source, **kwargs)
        return dataset_cls(dataset=data_source, **kwargs)

    @torch.no_grad()
    def predict(
        self,
        data_source,
        batch_size: Optional[int] = None,
        return_uncertainty: bool = False,
        n_mc_forward: int = 100,
    ) -> dict[str, np.ndarray]:
        """Predict redshifts for the given data.

        Args:
            data_source: File path or DataFrame.
            batch_size: Override the default batch size.
            return_uncertainty: If ``True``, enable dropout at
                inference time and run ``n_mc_forward`` stochastic
                forward passes (MC-dropout) to estimate uncertainty.
            n_mc_forward: Number of forward passes for MC-dropout.

        Returns:
            Dict with ``"predictions"`` (always present) and
            ``"uncertainties"`` (std across MC-dropout passes, only
            present when ``return_uncertainty=True``).
        """
        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

        if return_uncertainty:
            self.model.train()  # enable dropout
            all_preds: list[np.ndarray] = []
            all_uncert: list[np.ndarray] = []

            for batch in dl:
                inputs = batch["input"].to(self.device)
                batch_preds = np.array([
                    self.model(inputs).cpu().numpy().flatten()
                    for _ in range(n_mc_forward)
                ])  # (n_mc_forward, batch_size)
                all_preds.append(batch_preds.mean(axis=0))
                all_uncert.append(batch_preds.std(axis=0))

            self.model.eval()
            return {
                "predictions": np.concatenate(all_preds),
                "uncertainties": np.concatenate(all_uncert),
            }

        all_preds_det: list[np.ndarray] = []
        for batch in dl:
            inputs = batch["input"].to(self.device)
            outputs = self.model(inputs).cpu().numpy().flatten()
            all_preds_det.append(outputs)
        return {"predictions": np.concatenate(all_preds_det)}

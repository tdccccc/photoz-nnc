"""Inference-time helpers for the NNC photo-z model.

Two classes live here:

* :class:`TemperatureCalibrator` — post-hoc temperature scaling.
  Fits a scalar ``T`` against PIT uniformity (KS statistic) or CRPS
  on a labelled validation set; ``T > 1`` flattens overconfident
  PDFs, ``T < 1`` sharpens underconfident ones.
* :class:`PhotozBinInference` — loads a saved model directory
  (``config.yaml`` + ``scaler.pkl`` + ``best_model.pkl``) and
  exposes ``predict``, ``predict_probabilities``,
  ``predict_probabilities_chunked``, and ``calibrate`` methods.

Training-time pieces (``Config``, ``Trainer``, ``EarlyStopping``,
``BalancedBatchSampler``, ...) live in :mod:`core`. This split keeps
the inference path importable without dragging in training-only
dependencies at construction time.

Example:
    >>> from models.photoz.NNC.inference import PhotozBinInference
    >>> infer = PhotozBinInference("runs/my_experiment/")
    >>> out = infer.predict("catalog.fits")
    >>> out["expectation"]
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lib.pit import compute_pit

from .core import Config, expected_redshift_from_probs

__all__ = ["TemperatureCalibrator", "PhotozBinInference"]


# ===========================================================================
# Temperature-scaling calibrator
# ===========================================================================


class TemperatureCalibrator:
    """Post-hoc temperature scaling for classification-style PDFs.

    * ``T > 1`` flattens the distribution (fixes overconfidence).
    * ``T < 1`` sharpens the distribution (fixes underconfidence).
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.fit_history: list[dict[str, Any]] = []

    def fit(
        self,
        logits: np.ndarray,
        true_z: np.ndarray,
        bin_edges: np.ndarray,
        method: str = "minimize_ks",
        verbose: bool = True,
    ) -> "TemperatureCalibrator":
        """Find an optimal temperature on a validation set.

        Args:
            logits: ``(n, n_bins)`` raw model outputs.
            true_z: ``(n,)`` spectroscopic redshifts.
            bin_edges: ``(n_bins + 1,)`` bin edges.
            method: ``"minimize_ks"`` or ``"minimize_crps"``.
            verbose: If ``True``, log the chosen temperature.

        Returns:
            ``self`` (for chaining).
        """
        from scipy.optimize import minimize_scalar

        def _softmax(x: np.ndarray) -> np.ndarray:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

        def objective(T: float) -> float:
            if T <= 0:
                return 1e10
            probs = _softmax(logits / T)
            if method == "minimize_ks":
                # compute_pit already runs the KS test internally.
                return compute_pit(probs, true_z, bin_edges)["ks_stat"]
            if method == "minimize_crps":
                cumsum = np.cumsum(probs, axis=1)
                bin_rights = bin_edges[1:]
                true_cdf = (bin_rights[None, :] <= true_z[:, None]).astype(float)
                bin_widths = bin_edges[1:] - bin_edges[:-1]
                return float(
                    np.mean(
                        np.sum(
                            (cumsum - true_cdf) ** 2 * bin_widths[None, :],
                            axis=1,
                        )
                    )
                )
            raise ValueError(f"Unknown method: {method!r}")

        result = minimize_scalar(
            objective, bounds=(0.1, 10.0), method="bounded"
        )
        self.temperature = float(result.x)
        self.fit_history.append({
            "temperature": self.temperature,
            "objective": float(result.fun),
            "method": method,
        })
        if verbose:
            logging.info(
                f"Temperature scaling: T = {self.temperature:.4f} "
                f"(objective = {result.fun:.6f})"
            )
        return self

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply ``logits / T``."""
        return logits / self.temperature

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Return softmax probabilities after temperature scaling."""
        scaled = self.calibrate_logits(logits)
        e_x = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        """Save ``temperature`` and ``fit_history`` to a joblib pickle."""
        joblib.dump(
            {"temperature": self.temperature, "fit_history": self.fit_history},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TemperatureCalibrator":
        """Inverse of :meth:`save`."""
        data = joblib.load(path)
        calibrator = cls()
        calibrator.temperature = data["temperature"]
        calibrator.fit_history = data.get("fit_history", [])
        return calibrator


# ===========================================================================
# Inference
# ===========================================================================


class PhotozBinInference:
    """Load a saved model directory and run inference on new data."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        batch_size: int = 4096,
        num_workers: int = 4,
    ):
        """Initialise from a ``model_dir`` produced by the trainer.

        Args:
            model_dir: Path to the experiment directory containing
                ``config.yaml``, ``scaler.pkl``, and either
                ``best_model.pkl`` or ``checkpoint.pkl``.
            device: Torch device string (``"cuda"``, ``"cuda:0"``,
                ``"cpu"``). Falls back to CPU with a warning when
                CUDA is requested but unavailable.
            batch_size: Default inference batch size.
            num_workers: ``DataLoader`` num_workers.
        """
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
        self.calibrator: Optional[TemperatureCalibrator] = None

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
        model_params["num_bins"] = self.config.dataset_params[
            "binning_config"
        ]["num_bins"]
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

    def _create_dataset(self, data_source, mode: str = "inference"):
        from . import DATASETS

        dataset_cls = DATASETS[self.config.dataset_class_name]
        kwargs = dict(
            feature_generation=self.config.dataset_params.get("feature_generation"),
            feature_columns=self.config.dataset_params.get("feature_columns"),
            binning_config=self.config.dataset_params.get("binning_config"),
            scaler_X=self.scaler,
            mode=mode,
        )
        if isinstance(data_source, str):
            return dataset_cls(file_path=data_source, **kwargs)
        return dataset_cls(dataset=data_source, **kwargs)

    # ------------------------------------------------------------------
    # Prediction APIs
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        data_source,
        mode: str = "expectation",
        num_samples: int = 1,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        apply_calibration: Optional[bool] = None,
    ) -> dict[str, np.ndarray]:
        """Point-estimate (and optional samples) per source.

        Args:
            data_source: File path or DataFrame.
            mode: ``"expectation"`` returns only the E[z] column.
                ``"sample"`` also draws ``num_samples`` Monte-Carlo
                samples per source (multinomial over bins, uniform
                within the chosen bin).
            num_samples: Number of samples when ``mode='sample'``.
            batch_size: Override the default batch size.
            seed: ``numpy.random`` seed for the sampling mode.
            apply_calibration: ``None`` auto-enables when a
                calibrator is loaded; ``True`` forces; ``False``
                skips.

        Returns:
            Dict with ``"expectation"`` always present, and
            ``"samples"`` (shape ``(n, num_samples)``) when sampling.
        """
        if seed is not None:
            np.random.seed(seed)
        if apply_calibration is None:
            apply_calibration = self.calibrator is not None

        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        all_expect: list[np.ndarray] = []
        all_sampled: list[np.ndarray] = []
        centers = ds.bin_centers

        for batch in dl:
            inputs = batch["input"].to(self.device)
            logits = self.model(inputs).cpu().numpy()
            probs = self._softmax_maybe_calibrated(logits, apply_calibration)

            all_expect.append(expected_redshift_from_probs(probs, centers))

            if mode == "sample":
                batch_samples = []
                for _ in range(num_samples):
                    sampled = []
                    for p in probs:
                        idx = np.random.choice(len(centers), p=p)
                        left, right = ds.bin_edges[idx], ds.bin_edges[idx + 1]
                        sampled.append(np.random.uniform(left, right))
                    batch_samples.append(np.array(sampled))
                all_sampled.append(np.stack(batch_samples, axis=1))

        expect_arr = np.concatenate(all_expect, axis=0)
        if mode == "sample":
            return {
                "expectation": expect_arr,
                "samples": np.concatenate(all_sampled, axis=0),
            }
        return {"expectation": expect_arr}

    @torch.no_grad()
    def predict_probabilities(
        self,
        data_source,
        batch_size: Optional[int] = None,
        apply_calibration: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Return the full per-bin PDF for every source.

        Args:
            data_source: File path or DataFrame.
            batch_size: Override the default batch size.
            apply_calibration: See :meth:`predict`.

        Returns:
            Dict with ``probabilities`` ``(n, n_bins)``, ``bin_centers``,
            ``bin_edges``, ``expectation``, ``calibrated`` (bool),
            and ``temperature`` (1.0 when not calibrated).
        """
        if apply_calibration is None:
            apply_calibration = self.calibrator is not None

        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        centers = ds.bin_centers

        all_probs: list[np.ndarray] = []
        for batch in dl:
            inputs = batch["input"].to(self.device)
            logits = self.model(inputs).cpu().numpy()
            probs = self._softmax_maybe_calibrated(logits, apply_calibration)
            all_probs.append(probs)

        prob_arr = np.concatenate(all_probs, axis=0)
        return {
            "probabilities": prob_arr,
            "bin_centers": centers,
            "bin_edges": ds.bin_edges,
            "expectation": expected_redshift_from_probs(prob_arr, centers),
            "calibrated": apply_calibration and self.calibrator is not None,
            "temperature": (
                self.calibrator.temperature
                if (apply_calibration and self.calibrator) else 1.0
            ),
        }

    @torch.no_grad()
    def predict_probabilities_chunked(
        self,
        data_source,
        save_path: str,
        chunk_size: int = 100_000,
        batch_size: Optional[int] = None,
        compression: str = "gzip",
        apply_calibration: Optional[bool] = None,
    ) -> None:
        """Memory-bounded PDF inference streamed straight to HDF5.

        For very large catalogs (e.g. 7M x 240 bins), materialising
        all probabilities in memory at once is impractical; this
        method streams predictions to an HDF5 file one chunk at a
        time.

        Args:
            data_source: File path or DataFrame.
            save_path: Destination ``.h5`` file.
            chunk_size: Number of rows processed per spill.
            batch_size: Override the default batch size.
            compression: HDF5 compression (``'gzip'`` / ``'lzf'`` /
                ``None``).
            apply_calibration: See :meth:`predict`.
        """
        import h5py

        if not save_path.endswith(".h5"):
            raise ValueError(
                f"save_path must end with '.h5', got: {save_path!r}"
            )
        if apply_calibration is None:
            apply_calibration = self.calibrator is not None

        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size

        total = len(ds)
        num_bins = self.config.dataset_params["binning_config"]["num_bins"]
        centers = ds.bin_centers
        temperature = (
            self.calibrator.temperature
            if (apply_calibration and self.calibrator) else 1.0
        )

        print(f"Processing {total:,} samples with {num_bins} bins...")
        print(
            f"Temperature scaling: "
            f"{'T=' + str(temperature) if apply_calibration else 'disabled'}"
        )
        print(
            f"Estimated uncompressed size: "
            f"{total * num_bins * 4 / 1e9:.2f} GB"
        )

        with h5py.File(save_path, "w") as f:
            prob_dataset = f.create_dataset(
                "probabilities",
                shape=(total, num_bins),
                dtype=np.float32,
                compression=compression,
            )
            expect_dataset = f.create_dataset(
                "expectation",
                shape=(total,),
                dtype=np.float32,
                compression=compression,
            )
            f.create_dataset("bin_centers", data=centers, compression=compression)
            f.create_dataset("bin_edges", data=ds.bin_edges, compression=compression)

            f.attrs["num_samples"] = total
            f.attrs["num_bins"] = num_bins
            f.attrs["z_min"] = self.config.dataset_params["binning_config"]["z_min"]
            f.attrs["z_max"] = self.config.dataset_params["binning_config"]["z_max"]
            f.attrs["chunk_size"] = chunk_size
            f.attrs["compression"] = compression or ""
            f.attrs["calibrated"] = apply_calibration and self.calibrator is not None
            f.attrs["temperature"] = temperature

            processed = 0
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk_dl = DataLoader(
                    Subset(ds, range(start, end)),
                    batch_size=batch_size, shuffle=False,
                    num_workers=self.num_workers,
                )

                chunk_probs: list[np.ndarray] = []
                for batch in chunk_dl:
                    inputs = batch["input"].to(self.device)
                    logits = self.model(inputs).cpu().numpy()
                    probs = self._softmax_maybe_calibrated(
                        logits, apply_calibration
                    )
                    chunk_probs.append(probs)

                chunk_prob_arr = np.concatenate(chunk_probs, axis=0)
                prob_dataset[start:end] = chunk_prob_arr
                expect_dataset[start:end] = expected_redshift_from_probs(
                    chunk_prob_arr, centers
                )

                processed += end - start
                print(f"Processed {processed:,}/{total:,} samples")
                del chunk_probs, chunk_prob_arr
                torch.cuda.empty_cache()

        print(f"Results saved to {save_path}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(
        self,
        data_source,
        method: str = "minimize_ks",
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
        save_path: Optional[str] = None,
        plot: bool = True,
    ) -> dict[str, Any]:
        """Fit or apply temperature scaling on a labelled validation set.

        Three usage modes:

        1. ``temperature=None`` — auto-fit (``minimize_ks`` or
           ``minimize_crps``).
        2. ``temperature=<float>`` — skip optimisation and evaluate
           that specific temperature.
        3. Re-calibrate an existing model on new data.

        Args:
            data_source: File path or DataFrame with true redshifts.
            method: Optimisation target (``"minimize_ks"`` or
                ``"minimize_crps"``). Ignored when ``temperature`` is
                given.
            temperature: Override the auto-fit.
            batch_size: Override the default batch size.
            save_path: If given, save the calibrator to
                ``<save_path>.pkl`` and the comparison plot to
                ``<save_path>.png``.
            plot: Whether to draw the before/after PIT comparison.

        Returns:
            Dict with the chosen ``temperature``, the KS statistics
            before and after, and the full PIT diagnosis dicts.
        """
        ds = self._create_dataset(data_source, mode="validation")
        if batch_size is None:
            batch_size = self.batch_size
        if not hasattr(ds, "labels_cont") or ds.labels_cont is None:
            raise ValueError(
                "Dataset must have true redshifts for calibration."
            )

        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        all_logits: list[np.ndarray] = []
        all_true_z: list[np.ndarray] = []
        for batch in dl:
            inputs = batch["input"].to(self.device)
            all_logits.append(self.model(inputs).cpu().numpy())
            all_true_z.append(batch["label_z"].numpy().flatten())
        logits_arr = np.concatenate(all_logits, axis=0)
        true_z_arr = np.concatenate(all_true_z, axis=0)
        bin_edges = ds.bin_edges

        if temperature is not None:
            self.calibrator = TemperatureCalibrator()
            self.calibrator.temperature = float(temperature)
            print(f"Using fixed temperature: T={temperature:.4f}")
        else:
            self.calibrator = TemperatureCalibrator()
            self.calibrator.fit(
                logits_arr, true_z_arr, bin_edges, method=method
            )

        # Before / after comparison.
        e_x = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
        probs_before = e_x / e_x.sum(axis=1, keepdims=True)
        diag_before = compute_pit(probs_before, true_z_arr, bin_edges)

        probs_after = self.calibrator.calibrate_probs(logits_arr)
        diag_after = compute_pit(probs_after, true_z_arr, bin_edges)

        print(
            f"Before calibration: KS={diag_before['ks_stat']:.4f} | "
            f"{diag_before['suggestion']}"
        )
        print(
            f"After calibration (T={self.calibrator.temperature:.4f}): "
            f"KS={diag_after['ks_stat']:.4f} | {diag_after['suggestion']}"
        )

        if plot:
            _, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, (pit, diag, label) in zip(
                axes,
                [
                    (diag_before["pit_values"], diag_before, "Before (T=1.0)"),
                    (
                        diag_after["pit_values"],
                        diag_after,
                        f"After (T={self.calibrator.temperature:.4f})",
                    ),
                ],
            ):
                ax.hist(
                    pit, bins=20, range=(0, 1), density=True,
                    alpha=0.7, edgecolor="black",
                )
                ax.axhline(y=1, color="r", linestyle="--", linewidth=2)
                ax.set_title(f"{label}, KS={diag['ks_stat']:.4f}")
                ax.set_xlabel("PIT value")
                ax.set_ylabel("Density")
            plt.suptitle("Temperature Scaling Calibration")
            plt.tight_layout()

            if save_path:
                plot_path = str(Path(save_path).with_suffix(".png"))
                plt.savefig(plot_path, dpi=150)
                print(f"Plot saved to {plot_path}")
            plt.show()

        if save_path:
            calibrator_path = str(Path(save_path).with_suffix(".pkl"))
            self.calibrator.save(calibrator_path)
            print(f"Calibrator saved to {calibrator_path}")

        return {
            "temperature": self.calibrator.temperature,
            "ks_before": diag_before["ks_stat"],
            "ks_after": diag_after["ks_stat"],
            "diagnosis_before": diag_before,
            "diagnosis_after": diag_after,
        }

    def load_calibrator(self, path: str) -> "PhotozBinInference":
        """Load a previously-saved :class:`TemperatureCalibrator`."""
        self.calibrator = TemperatureCalibrator.load(path)
        print(f"Calibrator loaded: T={self.calibrator.temperature:.4f}")
        return self

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _softmax_maybe_calibrated(
        self, logits: np.ndarray, apply_calibration: bool,
    ) -> np.ndarray:
        """Softmax, going through the calibrator when available."""
        if apply_calibration and self.calibrator is not None:
            return self.calibrator.calibrate_probs(logits)
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

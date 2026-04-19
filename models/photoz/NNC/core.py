"""Training-time infrastructure for the NNC photo-z model.

This module holds the pieces that a typical user does **not** edit
when changing the model, features, or losses:

* :class:`Config` — dataclass mirroring ``config.yaml``.
* :class:`EarlyStopping` — standard best-checkpoint + patience early
  stopping.
* :func:`get_optimizer`, :func:`get_scheduler` — thin factories
  mapping config names to :mod:`torch.optim` classes.
* :class:`BalancedBatchSampler` — round-robin sampler balancing the
  per-batch distribution across redshift bins.
* :func:`expected_redshift_from_probs`,
  :func:`calculate_regression_metrics` — tiny metric helpers.
* :class:`Trainer` — glues dataset / model / loss / optimizer / PIT
  monitoring / early stopping into a single ``run()`` loop.

Inference-time helpers (``PhotozBinInference``,
``TemperatureCalibrator``) live in :mod:`inference`.

The model / dataset / loss selection happens through
``Config.model_class_name`` / ``Config.dataset_class_name``, resolved
via the :data:`MODELS` / :data:`DATASETS` registries in
``NNC/__init__.py``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset

from lib.io import readfile
from lib.pit import compute_pit, plot_pit

from .losses import (
    anchor_loss,
    combined_loss,
    compute_bin_weights,
    crps_loss,
    emd_loss,
    focal_loss,
    kl_divergence_with_softmax_logits,
    weighted_kl_divergence_loss,
)

__all__ = [
    "Config",
    "EarlyStopping",
    "get_optimizer",
    "get_scheduler",
    "BalancedBatchSampler",
    "expected_redshift_from_probs",
    "calculate_regression_metrics",
    "Trainer",
]


# ===========================================================================
# Config
# ===========================================================================


@dataclass
class Config:
    """Training / inference config, kept in sync with ``config.yaml``.

    Instantiate with defaults, then call :meth:`update_from_yaml` to
    overlay a YAML file, or :meth:`update_from_dict` to overlay a
    plain dict. Unknown keys are silently ignored.
    """

    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "nnc"
    random_seed: int = 2025
    gpu_id: str = "0"

    # --- Model ---
    model_params: dict[str, Any] = field(default_factory=dict)
    model_class_name: str = "PhotozBinningClassifier"

    # --- Data ---
    dataset_params: dict[str, Any] = field(default_factory=dict)
    dataset_class_name: str = "DatasetPhotozBinned"
    dataset_mode: dict[str, Any] = field(default_factory=lambda: {
        "type": "files",
        "paths": {"train": None, "val": None, "test": None},
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    })

    # --- Train ---
    optimizer_config: dict[str, Any] = field(default_factory=lambda: {
        "name": "AdamW",
        "params": {"lr": 5e-4, "weight_decay": 5e-3},
    })
    scheduler_config: dict[str, Any] = field(default_factory=lambda: {
        "name": "ReduceLROnPlateau",
        "params": {"mode": "min", "factor": 0.5, "patience": 5},
    })
    batch_size: int = 4096
    epochs: int = 200
    early_stopping: int = 20

    gradient_clipping: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "method": "norm",
        "max_norm": 1.0,
        "clip_value": 1.0,
    })

    loss_config: dict[str, Any] = field(default_factory=lambda: {
        # 'kl_divergence' | 'weighted_kl' | 'focal' | 'anchor' | 'crps' |
        # 'emd' | 'combined'
        "type": "kl_divergence",
        "bin_weights": {
            "enabled": False,
            # 'inverse_frequency' | 'balanced' | 'sqrt_inverse'
            "method": "inverse_frequency",
        },
        "sample_weights": {"enabled": False},
        "focal": {"alpha": 1.0, "gamma": 2.0},
    })

    sampling_config: dict[str, Any] = field(default_factory=lambda: {
        "balanced_sampling": {
            "enabled": False,
            "bins": 10,
            "z_min": 0.0,
            "z_max": 1.2,
        },
    })

    pit_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "monitor_interval": 1,
        "n_bins": 20,
        "plot": True,
        "plot_interval": 10,
    })

    def update_from_yaml(self, path: str) -> None:
        """Overlay values from a YAML file onto this config."""
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.update_from_dict(cfg)

    def update_from_dict(self, d: dict[str, Any]) -> None:
        """Overlay values from a dict. Unknown keys are ignored."""
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ===========================================================================
# Optimiser / scheduler factories
# ===========================================================================


def get_optimizer(
    parameters: Iterable[nn.Parameter],
    config: dict[str, Any],
) -> optim.Optimizer:
    """Build an optimizer by name from ``torch.optim``."""
    name = config.get("name", "AdamW")
    params = config.get("params", {})
    opt_cls = getattr(optim, name, None)
    if opt_cls is None:
        raise ValueError(f"Unsupported optimizer: {name!r}")
    return opt_cls(parameters, **params)


def get_scheduler(
    optimizer: optim.Optimizer,
    config: Optional[dict[str, Any]],
    training_context: Optional[dict[str, Any]] = None,
) -> Optional[_LRScheduler]:
    """Build an LR scheduler by name; return ``None`` when disabled.

    ``training_context`` may carry ``epochs`` so ``CosineAnnealingLR``
    can default ``T_max`` when the user did not set it.
    """
    if not config or not config.get("name"):
        return None
    name = config["name"]
    params = dict(config.get("params", {}))
    if (
        training_context
        and name == "CosineAnnealingLR"
        and "T_max" not in params
    ):
        params["T_max"] = training_context.get("epochs", 100)
    cls_ = getattr(optim.lr_scheduler, name, None)
    if cls_ is None:
        raise ValueError(f"Unsupported scheduler: {name!r}")
    return cls_(optimizer, **params)


# ===========================================================================
# Early stopping
# ===========================================================================


class EarlyStopping:
    """Early stopping with best-checkpoint bookkeeping.

    Tracks validation loss and stops training when it fails to
    improve for ``patience`` consecutive epochs. The best model is
    saved to both ``checkpoint_path`` (latest best) and
    ``best_model_path`` (kept for later inference).
    """

    def __init__(
        self,
        patience: int,
        checkpoint_path: Union[str, Path],
        best_model_path: Union[str, Path],
        verbose: bool = True,
        delta: float = 0,
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = Path(checkpoint_path)
        self.best_model_path = Path(best_model_path)
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            logging.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.checkpoint_path)
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss


# ===========================================================================
# Balanced batch sampler
# ===========================================================================


class BalancedBatchSampler:
    """Batch sampler that round-robins across redshift bins.

    Helpful when the training distribution is strongly peaked: each
    batch gets roughly equal representation across ``num_bins``
    redshift buckets, regardless of the underlying frequency.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_bins: int = 10,
        z_min: float = 0.0,
        z_max: float = 1.2,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        all_labels = np.array(
            [dataset[i]["label_z"].item() for i in range(len(dataset))]
        )
        bin_edges = np.linspace(z_min, z_max, num_bins + 1)
        bin_indices = np.clip(
            np.digitize(all_labels, bin_edges) - 1, 0, num_bins - 1
        )

        self.bin_to_indices: dict[int, list[int]] = {}
        for i, bin_idx in enumerate(bin_indices):
            self.bin_to_indices.setdefault(int(bin_idx), []).append(i)
        # Drop empty bins (digitize can leave some bins without any row).
        self.bin_to_indices = {
            k: v for k, v in self.bin_to_indices.items() if v
        }
        self.num_bins = len(self.bin_to_indices)

        logging.info(
            f"BalancedBatchSampler: using {self.num_bins} non-empty bins"
        )
        for bin_idx, indices in self.bin_to_indices.items():
            logging.info(f"  Bin {bin_idx}: {len(indices)} samples")

    def __iter__(self):
        if self.shuffle:
            for indices in self.bin_to_indices.values():
                np.random.shuffle(indices)

        bin_iterators = {
            bin_idx: iter(indices)
            for bin_idx, indices in self.bin_to_indices.items()
        }

        samples_per_bin = max(1, self.batch_size // self.num_bins)
        remainder = self.batch_size % self.num_bins

        total_samples = sum(len(v) for v in self.bin_to_indices.values())
        num_batches = total_samples // self.batch_size

        for _ in range(num_batches):
            batch: list[int] = []
            bins = list(self.bin_to_indices.keys())
            if self.shuffle:
                np.random.shuffle(bins)

            for i, bin_idx in enumerate(bins):
                take = samples_per_bin + (1 if i < remainder else 0)
                for _ in range(take):
                    try:
                        batch.append(next(bin_iterators[bin_idx]))
                    except StopIteration:
                        # Restart this bin's iterator when exhausted,
                        # so every batch gets the target counts even
                        # if some bins are smaller than others.
                        indices = self.bin_to_indices[bin_idx]
                        if self.shuffle:
                            np.random.shuffle(indices)
                        bin_iterators[bin_idx] = iter(indices)
                        batch.append(next(bin_iterators[bin_idx]))

            if self.shuffle:
                np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        total_samples = sum(len(v) for v in self.bin_to_indices.values())
        return total_samples // self.batch_size


# ===========================================================================
# Tiny metric helpers
# ===========================================================================


def expected_redshift_from_probs(
    probs: np.ndarray, bin_centers: np.ndarray
) -> np.ndarray:
    """Expected-value point estimate: ``sum_i p_i * bin_center_i``."""
    return (probs * bin_centers[None, :]).sum(axis=1)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    running_loss: float,
    num_batches: int,
) -> dict[str, float]:
    """Return loss/MSE/MAE/R² over a validation or test loader."""
    return {
        "loss": running_loss / max(1, num_batches),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ===========================================================================
# Trainer
# ===========================================================================


class Trainer:
    """Top-level training loop for the NNC photo-z model."""

    def __init__(self, config: Config):
        self.config = config
        self.device: Optional[torch.device] = None
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.logdir: Optional[Path] = None
        self.scaler_X: Optional[StandardScaler] = None

        self.train_metrics = {"loss": [], "mse": [], "mae": [], "r2": []}
        self.val_metrics = {"loss": [], "mse": [], "mae": [], "r2": []}

        self.bin_weights: Optional[Tensor] = None
        self.loss_function = None
        self.bin_edges_tensor: Optional[Tensor] = None
        self._use_crps = False

        self.pit_metrics: dict[str, list[Any]] = {
            "ks_stat": [], "ks_pval": [], "problem": [],
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self._setup_seed()
        self._setup_logging()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss_function()
        self._setup_early_stopping()

    def _setup_seed(self) -> None:
        seed = self.config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_logging(self) -> None:
        self.logdir = Path(self.config.save_dir) / self.config.experiment_name
        self.logdir.mkdir(parents=True, exist_ok=True)
        log_path = self.logdir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )
        logging.info("--- Configuration ---")
        for k, v in asdict(self.config).items():
            logging.info(f"{k}: {v}")
        logging.info("---------------------\n")
        with open(self.logdir / "config.yaml", "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False,
                      sort_keys=False)

    def _setup_device(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Using device: {self.device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _make_dataset(self, source, scaler_X, mode: str):
        """Shared constructor for all 'files' / 'ratio' flows."""
        from . import DATASETS

        dataset_cls = DATASETS[self.config.dataset_class_name]
        kwargs = dict(
            feature_generation=self.config.dataset_params.get("feature_generation"),
            feature_columns=self.config.dataset_params.get("feature_columns"),
            binning_config=self.config.dataset_params.get("binning_config"),
            scaler_X=scaler_X,
            mode=mode,
        )
        if isinstance(source, str):
            return dataset_cls(file_path=source, **kwargs)
        return dataset_cls(dataset=source, **kwargs)

    def _build_trainloader(self, dataset, indices=None):
        """Wrap a dataset in a DataLoader, optionally with a subset index
        list and optionally with :class:`BalancedBatchSampler`."""
        sampling = self.config.sampling_config.get(
            "balanced_sampling",
            {"enabled": False, "bins": 10, "z_min": 0.0, "z_max": 1.2},
        )
        target = Subset(dataset, indices) if indices is not None else dataset

        if sampling.get("enabled", False):
            sampler = BalancedBatchSampler(
                target,
                batch_size=self.config.batch_size,
                num_bins=sampling.get("bins", 10),
                z_min=sampling.get("z_min", 0.0),
                z_max=sampling.get("z_max", 1.2),
                shuffle=True,
            )
            return DataLoader(
                target, batch_sampler=sampler,
                num_workers=4, pin_memory=True,
            )
        return DataLoader(
            target, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )

    def _create_datasets_from_paths(self, paths: dict[str, str]) -> None:
        # Fit scaler on train first.
        tmp_train = self._make_dataset(paths["train"], scaler_X=None, mode="train")
        scaler_X = StandardScaler()
        scaler_X.fit(tmp_train.features)

        train_ds = self._make_dataset(paths["train"], scaler_X, "train")
        val_ds = self._make_dataset(paths["val"], scaler_X, "validation")
        test_ds = self._make_dataset(paths["test"], scaler_X, "test")

        self.scaler_X = scaler_X
        self.trainloader = self._build_trainloader(train_ds)
        self.valloader = DataLoader(
            val_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.testloader = DataLoader(
            test_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        logging.info(
            f"Data loaded from paths: {len(train_ds)} train, "
            f"{len(val_ds)} val, {len(test_ds)} test samples."
        )

    def _create_datasets_from_ratio(self) -> None:
        dataset = readfile(self.config.dataset_params.get("file_path"))
        indices = list(range(len(dataset)))
        ratios = self.config.dataset_mode.get(
            "ratios", {"train": 0.8, "val": 0.1, "test": 0.1}
        )
        val_test_ratio = ratios["val"] + ratios["test"]
        train_idx, temp_idx = train_test_split(
            indices, test_size=val_test_ratio,
            random_state=self.config.random_seed,
        )
        val_size_corr = ratios["val"] / val_test_ratio
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_size_corr),
            random_state=self.config.random_seed,
        )

        temp_ds = self._make_dataset(dataset, scaler_X=None, mode="inference")
        scaler_X = StandardScaler()
        scaler_X.fit(temp_ds.features[np.array(train_idx)])

        train_ds = self._make_dataset(dataset, scaler_X, "train")
        val_ds = self._make_dataset(dataset, scaler_X, "validation")
        test_ds = self._make_dataset(dataset, scaler_X, "test")

        self.scaler_X = scaler_X
        self.trainloader = self._build_trainloader(train_ds, indices=train_idx)
        self.valloader = DataLoader(
            Subset(val_ds, val_idx), batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.testloader = DataLoader(
            Subset(test_ds, test_idx), batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        logging.info(
            f"Data loaded: {len(train_idx)} train, {len(val_idx)} val, "
            f"{len(test_idx)} test samples."
        )

    def _setup_data(self) -> None:
        mode_type = self.config.dataset_mode.get("type", "files")
        if mode_type == "files":
            paths = self.config.dataset_mode.get("paths", {})
            if not all(paths.get(s) for s in ("train", "val", "test")):
                raise ValueError(
                    "When using 'files' mode, all paths "
                    "(train, val, test) must be specified."
                )
            self._create_datasets_from_paths(paths)
        elif mode_type == "ratio":
            self._create_datasets_from_ratio()
        else:
            raise ValueError(f"Unsupported dataset_mode type: {mode_type!r}")
        self._save_scaler()

    def _setup_model(self) -> None:
        from . import MODELS

        sample_batch = next(iter(self.trainloader))
        input_dim = sample_batch["input"].shape[1]
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
        self.model = model_cls(**model_params).to(self.device)
        logging.info(
            f"Model '{self.config.model_class_name}' created with "
            f"input_dim={input_dim}, num_bins={model_params['num_bins']}."
        )

    def _setup_optimizer(self) -> None:
        self.optimizer = get_optimizer(
            self.model.parameters(), self.config.optimizer_config
        )
        self.scheduler = get_scheduler(
            self.optimizer, self.config.scheduler_config,
            {"epochs": self.config.epochs},
        )

    def _setup_loss_function(self) -> None:
        loss_type = self.config.loss_config.get("type", "kl_divergence")

        # Shared bin_edges tensor for CDF-style losses.
        binning = self.config.dataset_params.get("binning_config", {})
        z_min = binning.get("z_min", 0.0)
        z_max = binning.get("z_max", 1.2)
        num_bins = binning.get("num_bins", 120)
        self.bin_edges_tensor = torch.FloatTensor(
            np.linspace(z_min, z_max, num_bins + 1)
        ).to(self.device)

        # Optional bin weights (for weighted KL / some combined losses).
        bw_cfg = self.config.loss_config.get(
            "bin_weights",
            {"enabled": False, "method": "inverse_frequency"},
        )
        if bw_cfg.get("enabled", False):
            bin_centers = self.trainloader.dataset.bin_centers
            all_labels = np.array([
                v for batch in self.trainloader
                for v in batch["label_z"].numpy().flatten()
            ])
            method = bw_cfg.get("method", "inverse_frequency")
            self.bin_weights = torch.FloatTensor(
                compute_bin_weights(bin_centers, all_labels, method)
            ).to(self.device)
            logging.info(
                f"Computed bin weights using method '{method}' "
                f"(range {self.bin_weights.min().item():.3f} - "
                f"{self.bin_weights.max().item():.3f})"
            )

        if loss_type == "kl_divergence":
            self.loss_function = kl_divergence_with_softmax_logits
            self._use_crps = False
        elif loss_type == "weighted_kl":
            self.loss_function = lambda logits, target, true_z=None: (
                weighted_kl_divergence_loss(
                    logits, target, bin_weights=self.bin_weights
                )
            )
            self._use_crps = False
        elif loss_type == "focal":
            focal_cfg = self.config.loss_config.get(
                "focal", {"alpha": 1.0, "gamma": 2.0}
            )
            self.loss_function = lambda logits, target, true_z=None: focal_loss(
                logits, target,
                alpha=focal_cfg.get("alpha", 1.0),
                gamma=focal_cfg.get("gamma", 2.0),
            )
            self._use_crps = False
        elif loss_type == "anchor":
            anchor_cfg = self.config.loss_config.get(
                "anchor",
                {"gamma": 0.5, "slack": 0.05, "anchor": "neg"},
            )
            self.loss_function = lambda logits, target, true_z=None: anchor_loss(
                logits, target,
                gamma=anchor_cfg.get("gamma", 0.5),
                slack=anchor_cfg.get("slack", 0.05),
                anchor=anchor_cfg.get("anchor", "neg"),
            )
            self._use_crps = False
        elif loss_type == "crps":
            self.loss_function = lambda logits, target, true_z: crps_loss(
                logits, true_z, self.bin_edges_tensor
            )
            self._use_crps = True
        elif loss_type == "emd":
            self.loss_function = lambda logits, target, true_z: emd_loss(
                logits, true_z, self.bin_edges_tensor
            )
            self._use_crps = True
        elif loss_type == "combined":
            combined_cfg = self.config.loss_config.get(
                "combined",
                {
                    "components": [{"type": "kl", "weight": 1.0}],
                    "entropy_reg": 0.0,
                    "z_weights": {"enabled": False},
                },
            )
            self.loss_function = lambda logits, target, true_z: combined_loss(
                logits, target, true_z, self.bin_edges_tensor, combined_cfg
            )
            self._use_crps = True
        else:
            raise ValueError(f"Unknown loss type: {loss_type!r}")

        logging.info(f"Using loss function: {loss_type}")

    def _setup_early_stopping(self) -> None:
        if self.config.early_stopping > 0:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping,
                checkpoint_path=self.logdir / "checkpoint.pkl",
                best_model_path=self.logdir / "best_model.pkl",
                verbose=True,
            )

    def _save_scaler(self) -> None:
        scaler_path = self.logdir / "scaler.pkl"
        joblib.dump(self.scaler_X, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")

    # ------------------------------------------------------------------
    # Train / eval steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        all_labels: list[float] = []
        all_preds: list[float] = []
        centers = self.trainloader.dataset.bin_centers

        for batch in self.trainloader:
            inputs = batch["input"].to(self.device)
            target_prob = batch["target_prob"].to(self.device)
            label_z = batch["label_z"]
            label_z_np = label_z.cpu().numpy().flatten()

            self.optimizer.zero_grad()
            logits = self.model(inputs)

            if self._use_crps:
                loss = self.loss_function(
                    logits, target_prob, label_z.to(self.device)
                )
            else:
                loss = self.loss_function(logits, target_prob)
            loss.backward()

            clip_cfg = self.config.gradient_clipping
            if clip_cfg["enabled"]:
                if clip_cfg["method"] == "norm":
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=clip_cfg["max_norm"],
                    )
                elif clip_cfg["method"] == "value":
                    nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        clip_value=clip_cfg["clip_value"],
                    )

            self.optimizer.step()
            running_loss += loss.item()

            probs = F.softmax(logits.detach(), dim=1).cpu().numpy()
            pred_z = expected_redshift_from_probs(probs, centers)
            all_labels.extend(label_z_np)
            all_preds.extend(pred_z)

        return calculate_regression_metrics(
            np.array(all_labels), np.array(all_preds),
            running_loss, len(self.trainloader),
        )

    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader, return_logits: bool = False,
    ):
        """Evaluate on ``loader``; optionally return raw logits for PIT."""
        self.model.eval()
        running_loss = 0.0
        all_labels: list[float] = []
        all_preds: list[float] = []
        all_logits: list[np.ndarray] = []
        centers = loader.dataset.bin_centers
        bin_edges = loader.dataset.bin_edges

        for batch in loader:
            inputs = batch["input"].to(self.device)
            target_prob = batch["target_prob"].to(self.device)
            label_z = batch["label_z"]
            label_z_np = label_z.cpu().numpy().flatten()

            logits = self.model(inputs)
            if self._use_crps:
                loss = self.loss_function(
                    logits, target_prob, label_z.to(self.device)
                )
            else:
                loss = self.loss_function(logits, target_prob)
            running_loss += loss.item()

            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_z = expected_redshift_from_probs(probs, centers)
            all_labels.extend(label_z_np)
            all_preds.extend(pred_z)

            if return_logits:
                all_logits.append(logits.cpu().numpy())

        metrics = calculate_regression_metrics(
            np.array(all_labels), np.array(all_preds),
            running_loss, len(loader),
        )
        if return_logits:
            return (
                metrics,
                np.concatenate(all_logits, axis=0),
                np.array(all_labels),
                bin_edges,
            )
        return metrics

    def _compute_pit_metrics(
        self,
        logits: np.ndarray,
        true_z: np.ndarray,
        bin_edges: np.ndarray,
        epoch: int,
    ) -> dict[str, Any]:
        """Compute PIT dict and optionally save a PIT plot for this epoch."""
        # Numerically stable softmax for inference-time probabilities.
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)

        n_bins = self.config.pit_config.get("n_bins", 20)
        result = compute_pit(
            probs, true_z, bin_edges, n_hist_bins=n_bins
        )

        pit_cfg = self.config.pit_config
        if pit_cfg.get("plot", True):
            plot_interval = pit_cfg.get("plot_interval", 10)
            if (epoch + 1) % plot_interval == 0 or epoch == 0:
                plot_pit(
                    result["pit_values"],
                    save_path=str(self.logdir / f"pit_epoch_{epoch+1:03d}.png"),
                    title=f"PIT Diagnosis - Epoch {epoch+1}",
                    n_hist_bins=n_bins,
                )

        return result

    def _update_and_log(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        duration: float,
        pit_diagnosis: Optional[dict[str, Any]] = None,
    ) -> None:
        for metric in self.train_metrics:
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])

        logging.info(
            f"Epoch {epoch+1}/{self.config.epochs} | "
            f"Duration: {duration:.2f}m"
        )
        logging.info(
            f"\tTrain Loss: {train_metrics['loss']:.4f}, "
            f"MSE: {train_metrics['mse']:.4f}, "
            f"MAE: {train_metrics['mae']:.4f}, "
            f"R2: {train_metrics['r2']:.4f}"
        )
        logging.info(
            f"\tVal   Loss: {val_metrics['loss']:.4f}, "
            f"MSE: {val_metrics['mse']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"R2: {val_metrics['r2']:.4f}"
        )

        if pit_diagnosis is not None:
            self.pit_metrics["ks_stat"].append(pit_diagnosis["ks_stat"])
            self.pit_metrics["ks_pval"].append(pit_diagnosis["ks_pval"])
            self.pit_metrics["problem"].append(pit_diagnosis["problem"])
            mark = "OK" if pit_diagnosis["is_calibrated"] else "--"
            logging.info(
                f"\tPIT: KS={pit_diagnosis['ks_stat']:.4f} [{mark}] | "
                f"{pit_diagnosis['suggestion']}"
            )

        if self.scheduler:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            else:
                self.scheduler.step()
        logging.info(
            f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )

    def _plot_metrics(self) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_map = {"loss": "Loss", "mse": "MSE", "mae": "MAE", "r2": "R2"}
        for ax, (metric, title) in zip(axes.flatten(), metrics_map.items()):
            ax.plot(self.train_metrics[metric], label=f"Train {title}")
            ax.plot(self.val_metrics[metric], label=f"Validation {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.logdir / "metrics_curve.jpg", dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Full training + final test-set evaluation."""
        self._setup()
        logging.info("--- Starting Training ---")

        pit_cfg = self.config.pit_config
        pit_enabled = pit_cfg.get("enabled", True)
        pit_interval = pit_cfg.get("monitor_interval", 1)

        for epoch in range(self.config.epochs):
            start_time = time.time()
            train_metrics = self._train_epoch()

            pit_diagnosis: Optional[dict[str, Any]] = None
            if pit_enabled and (epoch + 1) % pit_interval == 0:
                val_metrics, val_logits, val_true_z, bin_edges = self._evaluate(
                    self.valloader, return_logits=True
                )
                pit_diagnosis = self._compute_pit_metrics(
                    val_logits, val_true_z, bin_edges, epoch
                )
            else:
                val_metrics = self._evaluate(self.valloader)

            duration = (time.time() - start_time) / 60
            self._update_and_log(
                epoch, train_metrics, val_metrics, duration, pit_diagnosis
            )
            self._plot_metrics()

            if self.early_stopping:
                self.early_stopping(val_metrics["loss"], self.model)
                if self.early_stopping.early_stop:
                    logging.info("Early stopping triggered.")
                    break

        logging.info("--- Training Finished ---")
        self._test()
        torch.cuda.empty_cache()

    def _test(self) -> None:
        logging.info("--- Starting Testing ---")
        best_model_path = self.logdir / "best_model.pkl"
        if not best_model_path.exists():
            logging.warning(
                "No best model found; testing with the last checkpoint."
            )
            best_model_path = self.logdir / "checkpoint.pkl"
        self.model.load_state_dict(torch.load(best_model_path))

        metrics, test_logits, test_true_z, bin_edges = self._evaluate(
            self.testloader, return_logits=True
        )
        logging.info("Final Test Set Results:")
        logging.info(
            f"\tTest Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}"
        )

        e_x = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        pit_result = compute_pit(probs, test_true_z, bin_edges)
        mark = "OK" if pit_result["is_calibrated"] else "--"
        logging.info(
            f"\tTest PIT: KS={pit_result['ks_stat']:.4f} [{mark}] | "
            f"{pit_result['suggestion']}"
        )

        plot_pit(
            pit_result["pit_values"],
            save_path=str(self.logdir / "pit_test.png"),
            title="Test Set PIT Diagnosis",
        )

        with open(self.logdir / "test_results.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"Test {k}: {v:.6f}\n")
            f.write("\nPIT Diagnostics:\n")
            f.write(f"KS statistic: {pit_result['ks_stat']:.6f}\n")
            f.write(f"Problem: {pit_result['problem']}\n")
            f.write(f"Suggestion: {pit_result['suggestion']}\n")

"""Training-time infrastructure for the ANN photo-z regressor.

Mirrors :mod:`models.photoz.NNC.core` but for scalar regression
(single-output, MSE/MAE/Huber loss, no binning / PIT / calibration).

* :class:`Config` — dataclass mirroring ``config.yaml``.
* :class:`EarlyStopping` — best-checkpoint + patience.
* :func:`get_optimizer`, :func:`get_scheduler` — thin factories.
* :func:`get_loss_function` — maps config to ``nn.MSELoss`` etc.
* :func:`calculate_regression_metrics` — loss / MSE / MAE / R2 /
  NMAD / bias / outlier fraction.
* :class:`Trainer` — train loop with logging and metric curves.
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
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset

from lib.io import readfile

__all__ = [
    "Config",
    "EarlyStopping",
    "get_optimizer",
    "get_scheduler",
    "get_loss_function",
    "calculate_regression_metrics",
    "Trainer",
]


# ===========================================================================
# Config
# ===========================================================================


@dataclass
class Config:
    """Training / inference config, kept in sync with ``config.yaml``."""

    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "ann"
    random_seed: int = 2025
    gpu_id: str = "0"

    model_params: dict[str, Any] = field(default_factory=dict)
    model_class_name: str = "PhotozRegressor"

    dataset_params: dict[str, Any] = field(default_factory=dict)
    dataset_class_name: str = "DatasetPhotozRegression"
    dataset_mode: dict[str, Any] = field(default_factory=lambda: {
        "type": "files",
        "paths": {"train": None, "val": None, "test": None},
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    })

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
        "type": "mse",
        "huber_delta": 1.0,
        "smooth_l1_beta": 1.0,
    })

    def update_from_yaml(self, path: str) -> None:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.update_from_dict(cfg)

    def update_from_dict(self, d: dict[str, Any]) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ===========================================================================
# Optimiser / scheduler / loss factories
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
    """Build an LR scheduler by name; return ``None`` when disabled."""
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


def get_loss_function(config: dict[str, Any]) -> nn.Module:
    """Map a config dict to a regression loss module.

    Supported ``type`` values: ``"mse"``, ``"mae"``, ``"huber"``,
    ``"smooth_l1"``.
    """
    loss_type = config.get("type", "mse")
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "huber":
        return nn.HuberLoss(delta=config.get("huber_delta", 1.0))
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss(beta=config.get("smooth_l1_beta", 1.0))
    raise ValueError(f"Unknown loss type: {loss_type!r}")


# ===========================================================================
# Early stopping
# ===========================================================================


class EarlyStopping:
    """Early stopping with best-checkpoint bookkeeping."""

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
# Metric helpers
# ===========================================================================


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    running_loss: float,
    num_batches: int,
) -> dict[str, float]:
    """Return loss / MSE / MAE / R2 / NMAD / bias / outlier fraction."""
    residuals = y_pred - y_true
    return {
        "loss": running_loss / max(1, num_batches),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nmad": 1.48 * float(np.median(np.abs(residuals))),
        "bias": float(np.mean(residuals)),
        "outlier_frac": float(np.mean(np.abs(residuals) > 0.15)),
    }


# ===========================================================================
# Trainer
# ===========================================================================


class Trainer:
    """Training loop for the ANN photo-z regressor."""

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
        self.loss_function = None

        self.train_metrics: dict[str, list] = {
            "loss": [], "mse": [], "mae": [], "r2": [],
            "nmad": [], "bias": [], "outlier_frac": [],
        }
        self.val_metrics: dict[str, list] = {
            "loss": [], "mse": [], "mae": [], "r2": [],
            "nmad": [], "bias": [], "outlier_frac": [],
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

    def _make_dataset(self, source, scaler_X, mode, noise_cfg=None):
        from . import DATASETS

        dataset_cls = DATASETS[self.config.dataset_class_name]
        kwargs = dict(
            feature_generation=self.config.dataset_params.get("feature_generation"),
            feature_columns=self.config.dataset_params.get("feature_columns"),
            label_noise=noise_cfg,
            scaler_X=scaler_X,
            mode=mode,
        )
        if isinstance(source, str):
            return dataset_cls(file_path=source, **kwargs)
        return dataset_cls(dataset=source, **kwargs)

    def _create_datasets_from_paths(self, paths: dict[str, str]) -> None:
        noise_cfg = self.config.dataset_params.get("label_noise")
        tmp = self._make_dataset(paths["train"], None, "train", noise_cfg)
        scaler_X = StandardScaler()
        scaler_X.fit(tmp.features)

        train_ds = self._make_dataset(paths["train"], scaler_X, "train", noise_cfg)
        val_ds = self._make_dataset(paths["val"], scaler_X, "validation")
        test_ds = self._make_dataset(paths["test"], scaler_X, "test")

        self.scaler_X = scaler_X
        self.trainloader = DataLoader(
            train_ds, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
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
        noise_cfg = self.config.dataset_params.get("label_noise")
        dataset = readfile(self.config.dataset_params.get("file_path"))
        indices = list(range(len(dataset)))
        ratios = self.config.dataset_mode.get(
            "ratios", {"train": 0.8, "val": 0.1, "test": 0.1},
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

        tmp = self._make_dataset(dataset, None, "inference")
        scaler_X = StandardScaler()
        scaler_X.fit(tmp.features[np.array(train_idx)])

        train_ds = self._make_dataset(dataset, scaler_X, "train", noise_cfg)
        val_ds = self._make_dataset(dataset, scaler_X, "validation")
        test_ds = self._make_dataset(dataset, scaler_X, "test")

        self.scaler_X = scaler_X
        self.trainloader = DataLoader(
            Subset(train_ds, train_idx), batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        self.valloader = DataLoader(
            Subset(val_ds, val_idx), batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.testloader = DataLoader(
            Subset(test_ds, test_idx), batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        logging.info(
            f"Data loaded: {len(train_idx)} train, "
            f"{len(val_idx)} val, {len(test_idx)} test samples."
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
        model_cls = MODELS.get(self.config.model_class_name)
        if model_cls is None:
            raise ValueError(
                f"Model class {self.config.model_class_name!r} is not "
                f"registered; known: {sorted(MODELS)}."
            )
        self.model = model_cls(**model_params).to(self.device)
        logging.info(
            f"Model '{self.config.model_class_name}' created with "
            f"input_dim={input_dim}."
        )

    def _setup_optimizer(self) -> None:
        self.optimizer = get_optimizer(
            self.model.parameters(), self.config.optimizer_config,
        )
        self.scheduler = get_scheduler(
            self.optimizer, self.config.scheduler_config,
            {"epochs": self.config.epochs},
        )

    def _setup_loss_function(self) -> None:
        self.loss_function = get_loss_function(self.config.loss_config)
        logging.info(
            f"Using loss function: {self.config.loss_config.get('type', 'mse')}"
        )

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

        for batch in self.trainloader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
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
            all_labels.extend(targets.cpu().numpy().flatten())
            all_preds.extend(outputs.detach().cpu().numpy().flatten())

        return calculate_regression_metrics(
            np.array(all_labels), np.array(all_preds),
            running_loss, len(self.trainloader),
        )

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_labels: list[float] = []
        all_preds: list[float] = []

        for batch in loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            running_loss += loss.item()
            all_labels.extend(targets.cpu().numpy().flatten())
            all_preds.extend(outputs.cpu().numpy().flatten())

        return calculate_regression_metrics(
            np.array(all_labels), np.array(all_preds),
            running_loss, len(loader),
        )

    def _update_and_log(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        duration: float,
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
            f"R2: {train_metrics['r2']:.4f}, "
            f"NMAD: {train_metrics['nmad']:.4f}"
        )
        logging.info(
            f"\tVal   Loss: {val_metrics['loss']:.4f}, "
            f"MSE: {val_metrics['mse']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"R2: {val_metrics['r2']:.4f}, "
            f"NMAD: {val_metrics['nmad']:.4f}"
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics_map = {
            "loss": "Loss", "mse": "MSE", "mae": "MAE",
            "r2": "R2", "nmad": "NMAD", "outlier_frac": "Outlier Fraction",
        }
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

        for epoch in range(self.config.epochs):
            start_time = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._evaluate(self.valloader)
            duration = (time.time() - start_time) / 60
            self._update_and_log(epoch, train_metrics, val_metrics, duration)
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
        metrics = self._evaluate(self.testloader)
        logging.info("Final Test Set Results:")
        logging.info(
            f"\tTest Loss: {metrics['loss']:.4f}, "
            f"MSE: {metrics['mse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, "
            f"R2: {metrics['r2']:.4f}"
        )
        logging.info(
            f"\tNMAD: {metrics['nmad']:.4f}, "
            f"Bias: {metrics['bias']:.4f}, "
            f"Outlier Frac: {metrics['outlier_frac']:.4f}"
        )
        with open(self.logdir / "test_results.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"Test {k}: {v:.6f}\n")

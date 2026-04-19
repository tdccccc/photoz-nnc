"""Binary galaxy / non-galaxy classifier built on a feed-forward MLP.

Pipeline at a glance
--------------------
1. :class:`DatasetGalaxyClassification` loads a catalog (FITS / HDF5
   / CSV via :func:`lib.io.readfile`), picks feature columns, and
   exposes a scalar ``{0, 1}`` label column.
2. :class:`GalaxyClassifier` is a stack of ``Linear → Norm →
   Activation → Dropout`` blocks (optionally residual) ending in a
   single logit. Sigmoid is applied outside the model, either inside
   :class:`FocalLoss` / :class:`~torch.nn.BCEWithLogitsLoss` during
   training, or explicitly during inference.
3. :class:`Trainer` glues dataset / model / loss / optimizer /
   early-stopping together; :meth:`Trainer.run` executes the full
   training + final test-set evaluation.
4. :class:`GalaxyClassificationInference` loads a finished model
   directory (``config.yaml`` + ``scaler.pkl`` + ``best_model.pkl``)
   and exposes :meth:`predict` for deterministic inference or
   MC-dropout uncertainty.

This module is intentionally kept as a single file (vs. the
package-style split used for the photo-z NNC / ANN models) because
its scope is narrower and it is unlikely to grow further.

Usage:
    python galaxyClf_ann.py --config path/to/config.yaml
"""

import os
import time
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Iterable, Union, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import joblib

# Project-local I/O helpers (unified FITS / HDF5 / CSV reader).
from lib.io import readfile


# ============================
# Config
# ============================
@dataclass
class Config:
    """Training / inference config, kept in sync with ``config.yaml``.

    Instantiate with defaults, then overlay YAML via
    :meth:`update_from_yaml` or a plain dict via
    :meth:`update_from_dict`. Unknown keys in the source are ignored
    silently.

    Top-level sections:

    * **identity** — ``config_path``, ``save_dir``, ``experiment_name``,
      ``random_seed``, ``gpu_id``.
    * **model** — ``model_class_name`` (+ ``model_params`` dict).
    * **data** — ``dataset_class_name`` (+ ``dataset_params`` dict),
      ``dataset_mode`` (either explicit ``files`` paths or ``ratio``
      split).
    * **train** — optimizer / scheduler / batch / epoch / early
      stopping / gradient clipping / loss.
    """

    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "galaxy_classification"
    random_seed: int = 2025
    gpu_id: str = '0'

    # Model
    model_params: Dict[str, Any] = field(default_factory=dict)
    model_class_name: str = 'GalaxyClassifier'

    # Data
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    dataset_class_name: str = 'DatasetGalaxyClassification'
    dataset_mode: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'files',
        'paths': {'train': None, 'val': None, 'test': None},
        'ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
    })

    # Train
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        'name': 'AdamW',
        'params': {'lr': 5e-4, 'weight_decay': 5e-3}
    })
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        'name': 'ReduceLROnPlateau',
        'params': {'mode': 'min', 'factor': 0.5, 'patience': 5}
    })
    batch_size: int = 4096
    epochs: int = 200
    early_stopping: int = 20

    # Gradient clipping
    gradient_clipping: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'norm',
        'max_norm': 1.0,
        'clip_value': 1.0
    })

    # Loss function configuration
    #   type: 'bce' | 'bce_logits' | 'focal'
    loss_config: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'bce_logits',
        'focal_alpha': 1.0,  # For Focal loss (class weighting)
        'focal_gamma': 2.0,  # For Focal loss (focus on hard examples)
    })

    def update_from_yaml(self, path: str):
        """Overlay values from a YAML file onto this config."""
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.update_from_dict(cfg)

    def update_from_dict(self, d: Dict[str, Any]):
        """Overlay values from a dict. Unknown keys are ignored."""
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ============================
# Dataset for classification
# ============================
class DatasetGalaxyClassification(Dataset):
    """Galaxy / non-galaxy catalog → PyTorch dataset.

    Exactly one of ``dataset`` and ``file_path`` must be given. The
    loaded frame must contain every feature column referenced by
    ``feature_columns`` / ``feature_generation`` and, when
    ``mode != 'inference'``, the ``label_column`` too.

    Args:
        dataset: In-memory DataFrame. Mutually exclusive with
            ``file_path``.
        file_path: Path to a catalog (any extension supported by
            :func:`lib.io.readfile`). Mutually exclusive with
            ``dataset``.
        feature_generation: Optional spec passed to
            :meth:`_choose_features`; must contain ``mag_types``,
            ``color_types``, ``use_mag``, ``use_magerr``,
            ``use_colorerr``.
        feature_columns: Explicit list of column names. Takes
            priority over ``feature_generation``.
        label_column: Name of the binary label column (galaxy=1,
            others=0). Defaults to ``"label"``.
        scaler_X: Optional pre-fitted :class:`StandardScaler`;
            applied to ``features`` at construction time.
        mode: ``"train"`` / ``"validation"`` / ``"test"`` /
            ``"inference"``. The first three expose ``labels``;
            ``"inference"`` skips label processing.

    Raises:
        ValueError: Both or neither of ``dataset`` / ``file_path``,
            or neither ``feature_columns`` nor ``feature_generation``.
        FileNotFoundError: ``file_path`` does not exist.
    """

    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_path: str = None,
                 feature_generation: dict = None,
                 feature_columns: list = None,
                 label_column: str = 'label',
                 scaler_X: StandardScaler = None,
                 mode: str = 'train'):

        # --- Source table ---
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

        # --- Feature columns ---
        if feature_columns is not None:
            self.feature_cols = feature_columns
        elif feature_generation is not None:
            self.mag_types = feature_generation.get('mag_types')
            self.color_types = feature_generation.get('color_types')
            self.use_mag = feature_generation.get('use_mag')
            self.use_magerr = feature_generation.get('use_magerr')
            self.use_colorerr = feature_generation.get('use_colorerr')
            self.feature_cols = self._choose_features(self.mag_types,
                                                     self.color_types,
                                                     use_mag=self.use_mag,
                                                     use_magerr=self.use_magerr,
                                                     use_colorerr=self.use_colorerr)
        else:
            raise ValueError("Either 'feature_columns' or 'feature_generation' must be provided")
        self.features = self.dataset[self.feature_cols].values.astype(np.float32)

        # --- Labels ---
        self.mode = mode
        self.label_column = label_column
        if self.mode in ['train', 'validation', 'test']:
            # Binary classification labels: galaxy=1, others=0
            self.labels = self.dataset[label_column].values.astype(np.float32)

        # --- Standardisation ---
        if scaler_X is not None:
            self.scaler_X = scaler_X
            self.features = self.scaler_X.transform(self.features)

    def _choose_features(self, mag_types, color_types, use_mag=True, use_magerr=True, use_colorerr=True):
        """Build the feature column list for the grizy PS1 / LS layout.

        Survey-specific: assumes ``grizy`` bands and the adjacent-band
        colours ``gr / ri / iz / zy``. Override / replace if adapting
        to a different band set.
        """
        bands = ['g', 'r', 'i', 'z', 'y']
        colours = ['gr', 'ri', 'iz', 'zy']
        mag_cols, err_cols = [], []
        if color_types:
            for color_type in color_types:
                for c in colours:
                    mag_cols.append(f'{color_type}_{c}')
                    if use_colorerr:
                        err_cols.append(f'{color_type}_{c}_Err')
        if use_mag and mag_types:
            for b in bands:
                for m in mag_types:
                    mag_cols.append(f'{b}{m}Mag_dered')
                    if use_magerr:
                        err_cols.append(f'{b}{m}MagErr')
        return mag_cols + err_cols

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        if self.mode in ['train', 'validation', 'test']:
            y = torch.FloatTensor([self.labels[idx]])
            return {'input': x, 'target': y}
        return {'input': x}


# ============================
# Model
# ============================
#: String → ``nn.Module`` factory for supported activation functions.
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
    'elu': nn.ELU,
    'silu': nn.SiLU,
}


class GalaxyClassifier(nn.Module):
    """MLP that outputs a single logit for binary classification.

    Architecture: a stack of ``Linear → Norm → Activation → Dropout``
    blocks, optionally with residual skips (a learned projection when
    input/output widths differ, Identity when they match), ending in
    a single linear head. Sigmoid is *not* applied inside ``forward``
    so that the output can feed straight into
    :class:`~torch.nn.BCEWithLogitsLoss` (numerically stable); apply
    ``torch.sigmoid`` explicitly at inference time to get
    probabilities.

    Args:
        input_dim: Number of input features.
        hidden_dims: Widths of the hidden layers. ``None`` falls
            back to ``[512, 256, 128, 64]``.
        dropout_rate: Dropout probability after the activation.
            Pass ``0`` to disable.
        activation_function: Key into :data:`ACTIVATION_FUNCTIONS`.
        use_batch_norm: Insert :class:`~torch.nn.BatchNorm1d` after
            each hidden linear.
        use_layer_norm: Insert :class:`~torch.nn.LayerNorm` instead.
            Ignored (with a warning) when ``use_batch_norm`` is also
            ``True``.
        use_residual: Add a residual skip around every hidden block.

    Raises:
        ValueError: ``activation_function`` is not in
            :data:`ACTIVATION_FUNCTIONS`.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = None,
                 dropout_rate: float = 0.1,
                 activation_function: str = 'relu',
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_residual: bool = True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        if use_batch_norm and use_layer_norm:
            logging.warning("Both BatchNorm and LayerNorm are enabled. Using BatchNorm only.")
            use_layer_norm = False

        if activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        activation_fn = ACTIVATION_FUNCTIONS[activation_function]

        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        in_dim = input_dim
        for h in hidden_dims:
            blocks = [nn.Linear(in_dim, h)]
            if use_batch_norm:
                blocks.append(nn.BatchNorm1d(h))
            elif use_layer_norm:
                blocks.append(nn.LayerNorm(h))
            blocks.append(activation_fn())
            if dropout_rate > 0:
                blocks.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Sequential(*blocks))
            if use_residual:
                # Identity skip when widths match, learned projection
                # otherwise so the residual add is well-defined.
                self.residual_layers.append(nn.Linear(in_dim, h) if in_dim != h else nn.Identity())
            else:
                self.residual_layers.append(None)
            in_dim = h

        # Output layer for binary classification (logits)
        self.output = nn.Linear(in_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming-normal init for linear layers; 1/0 for norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Return a raw logit of shape ``(batch, 1)`` (no sigmoid)."""
        for layer, res in zip(self.layers, self.residual_layers):
            identity = x
            out = layer(x)
            if self.use_residual and res is not None:
                out = out + (identity if isinstance(res, nn.Identity) else res(identity))
            x = out
        # Return logits (no sigmoid here, will be applied in loss or inference)
        return self.output(x)


# ============================
# Optimizer & Scheduler
# ============================
def get_optimizer(parameters: Iterable[torch.nn.Parameter], config: Dict[str, Any]) -> optim.Optimizer:
    """Build an optimizer by name from :mod:`torch.optim`."""
    name = config.get('name', 'AdamW')
    params = config.get('params', {})
    opt_cls = getattr(optim, name, None)
    if opt_cls is None:
        raise ValueError(f"Unsupported optimizer: {name}")
    return opt_cls(parameters, **params)


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], training_context: Dict[str, Any] = None) -> _LRScheduler:
    """Build an LR scheduler by name; return ``None`` when disabled.

    ``training_context`` may carry ``epochs`` so that
    :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` can default
    ``T_max`` when the user did not set it.
    """
    if not config or not config.get('name'):
        return None
    name = config['name']
    params = config.get('params', {})
    if training_context and name == 'CosineAnnealingLR' and 'T_max' not in params:
        params['T_max'] = training_context.get('epochs', 100)
    cls_ = getattr(optim.lr_scheduler, name, None)
    if cls_ is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    return cls_(optimizer, **params)


# ============================
# Utils
# ============================
class EarlyStopping:
    """Early stopping with best-checkpoint bookkeeping.

    Tracks validation loss and stops training when it fails to
    improve (by more than ``delta``) for ``patience`` consecutive
    epochs. The best model is saved to both ``checkpoint_path``
    (latest best) and ``best_model_path`` (kept for later inference).
    """

    def __init__(self, patience: int, checkpoint_path: Union[str, Path], best_model_path: Union[str, Path], verbose: bool = True, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = Path(checkpoint_path)
        self.best_model_path = Path(best_model_path)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.checkpoint_path)
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss


def calculate_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, running_loss: float, num_batches: int) -> Dict[str, float]:
    """Compute standard binary-classification metrics + confusion counts.

    Uses a fixed 0.5 threshold to turn probabilities into hard
    predictions. AUC-ROC is computed from the raw probabilities and
    falls back to 0 when only one class is present in ``y_true``
    (:func:`sklearn.metrics.roc_auc_score` raises in that case).

    Args:
        y_true: ``(n,)`` ground-truth labels (0 / 1).
        y_pred_proba: ``(n,)`` predicted probability of class 1.
        running_loss: Sum of per-batch losses.
        num_batches: Number of batches contributing to
            ``running_loss``.

    Returns:
        Dict with ``loss``, ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``auc``, and confusion-matrix counts ``tp`` / ``fp``
        / ``tn`` / ``fn``.
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_true_int = y_true.astype(int)

    accuracy = accuracy_score(y_true_int, y_pred)
    precision = precision_score(y_true_int, y_pred, zero_division=0)
    recall = recall_score(y_true_int, y_pred, zero_division=0)
    f1 = f1_score(y_true_int, y_pred, zero_division=0)

    # AUC-ROC (requires probability scores)
    try:
        auc = roc_auc_score(y_true_int, y_pred_proba)
    except ValueError:
        auc = 0.0  # If only one class present

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred, labels=[0, 1]).ravel()

    return {
        'loss': running_loss / max(1, num_batches),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


# ============================
# Loss Functions
# ============================
class FocalLoss(nn.Module):
    """Focal loss for binary classification.

    Down-weights the contribution of easy examples via
    ``(1 - p_t) ** gamma``; useful when the positive class is rare.
    ``gamma = 0`` recovers plain BCE (weighted by ``alpha``).

    Args:
        alpha: Global scaling / positive-class weighting. Defaults
            to 1.0.
        gamma: Focusing exponent. Higher = more weight on hard
            examples. Defaults to 2.0.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_loss_function(config: Dict[str, Any]) -> callable:
    """Map a ``loss_config`` dict to a concrete loss module.

    Supported ``type`` values:

    * ``"bce"`` — :class:`~torch.nn.BCELoss` (expects probabilities,
      i.e. a separate sigmoid on the model output).
    * ``"bce_logits"`` — :class:`~torch.nn.BCEWithLogitsLoss`
      (expects logits; numerically more stable; **recommended**).
    * ``"focal"`` — :class:`FocalLoss` (expects logits). Reads
      ``focal_alpha`` and ``focal_gamma`` from ``config``.
    """
    loss_type = config.get('type', 'bce_logits')

    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'bce_logits':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        alpha = config.get('focal_alpha', 1.0)
        gamma = config.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================
# Trainer
# ============================
class Trainer:
    """Top-level training loop for the galaxy classifier.

    One-shot usage: build a :class:`Config`, instantiate ``Trainer``,
    call :meth:`run`. The trainer takes care of seeding, data loader
    construction, scaler fitting + saving, model instantiation,
    per-epoch logging, early stopping, metric curve plotting, and a
    final evaluation on the test set.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.early_stopping = None
        self.logdir = None
        self.scaler_X = None

        self.train_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        self.val_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

        self.loss_function = None

    def _setup(self):
        """Run every setup step in the correct order."""
        self._setup_seed()
        self._setup_logging()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss_function()
        self._setup_early_stopping()

    def _setup_seed(self):
        """Seed torch / numpy + enable deterministic cuDNN."""
        seed = self.config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_logging(self):
        """Create ``{save_dir}/{experiment_name}/``, set up logging,
        and dump a frozen copy of the config."""
        self.logdir = Path(f"{self.config.save_dir}/{self.config.experiment_name}")
        self.logdir.mkdir(parents=True, exist_ok=True)
        log_path = self.logdir / 'training.log'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
        logging.info("--- Configuration ---")
        for k, v in asdict(self.config).items():
            logging.info(f"{k}: {v}")
        logging.info("---------------------\n")
        with open(self.logdir / 'config.yaml', 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False, sort_keys=False)

    def _setup_device(self):
        """Pin CUDA visibility to ``gpu_id`` and pick cuda/cpu."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def _create_datasets_from_paths(self, paths: Dict[str, str]):
        """Build train / val / test dataloaders from explicit paths.

        The scaler is **fit on the training set only** before any
        split is wrapped for iteration.
        """
        # Fit scaler on train
        tmp_train = DatasetGalaxyClassification(file_path=paths['train'],
                                           feature_generation=self.config.dataset_params.get('feature_generation'),
                                           feature_columns=self.config.dataset_params.get('feature_columns'),
                                           label_column=self.config.dataset_params.get('label_column', 'label'),
                                           scaler_X=None,
                                           mode='train')
        scaler_X = StandardScaler()
        scaler_X.fit(tmp_train.features)

        train_ds = DatasetGalaxyClassification(file_path=paths['train'],
                                          feature_generation=self.config.dataset_params.get('feature_generation'),
                                          feature_columns=self.config.dataset_params.get('feature_columns'),
                                          label_column=self.config.dataset_params.get('label_column', 'label'),
                                          scaler_X=scaler_X,
                                          mode='train')
        val_ds = DatasetGalaxyClassification(file_path=paths['val'],
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        label_column=self.config.dataset_params.get('label_column', 'label'),
                                        scaler_X=scaler_X,
                                        mode='validation')
        test_ds = DatasetGalaxyClassification(file_path=paths['test'],
                                         feature_generation=self.config.dataset_params.get('feature_generation'),
                                         feature_columns=self.config.dataset_params.get('feature_columns'),
                                         label_column=self.config.dataset_params.get('label_column', 'label'),
                                         scaler_X=scaler_X,
                                         mode='test')

        self.scaler_X = scaler_X
        self.trainloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.valloader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.testloader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logging.info(f"Data loaded from paths: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test samples.")

    def _setup_data(self):
        """Dispatch on ``dataset_mode.type`` (``files`` or ``ratio``)."""
        mode_type = self.config.dataset_mode.get('type', 'files')
        if mode_type == 'files':
            paths = self.config.dataset_mode.get('paths', {})
            if not all(paths.get(s) for s in ['train', 'val', 'test']):
                raise ValueError("When using 'files' mode, all paths (train, val, test) must be specified")
            self._create_datasets_from_paths(paths)
        elif mode_type == 'ratio':
            # Single-file mode: random-split once, then wrap each split
            # in a Subset that shares the underlying Dataset + scaler.
            dataset = readfile(self.config.dataset_params.get('file_path'))
            total = len(dataset)
            indices = list(range(total))
            ratios = self.config.dataset_mode.get('ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
            val_test_ratio = ratios['val'] + ratios['test']
            train_idx, temp_idx = train_test_split(indices, test_size=val_test_ratio, random_state=self.config.random_seed)
            val_size_corr = ratios['val'] / val_test_ratio
            val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size_corr), random_state=self.config.random_seed)

            temp_ds = DatasetGalaxyClassification(dataset=dataset,
                                             feature_generation=self.config.dataset_params.get('feature_generation'),
                                             feature_columns=self.config.dataset_params.get('feature_columns'),
                                             label_column=self.config.dataset_params.get('label_column', 'label'),
                                             scaler_X=None, mode='inference')
            scaler_X = StandardScaler()
            scaler_X.fit(temp_ds.features[np.array(train_idx)])

            train_ds = DatasetGalaxyClassification(dataset=dataset,
                                              feature_generation=self.config.dataset_params.get('feature_generation'),
                                              feature_columns=self.config.dataset_params.get('feature_columns'),
                                              label_column=self.config.dataset_params.get('label_column', 'label'),
                                              scaler_X=scaler_X, mode='train')
            val_ds = DatasetGalaxyClassification(dataset=dataset,
                                            feature_generation=self.config.dataset_params.get('feature_generation'),
                                            feature_columns=self.config.dataset_params.get('feature_columns'),
                                            label_column=self.config.dataset_params.get('label_column', 'label'),
                                            scaler_X=scaler_X,
                                            mode='validation')
            test_ds = DatasetGalaxyClassification(dataset=dataset,
                                             feature_generation=self.config.dataset_params.get('feature_generation'),
                                             feature_columns=self.config.dataset_params.get('feature_columns'),
                                             label_column=self.config.dataset_params.get('label_column', 'label'),
                                             scaler_X=scaler_X, mode='test')

            self.scaler_X = scaler_X
            self.trainloader = DataLoader(Subset(train_ds, train_idx), batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            self.valloader = DataLoader(Subset(val_ds, val_idx), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            self.testloader = DataLoader(Subset(test_ds, test_idx), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            logging.info(f"Data loaded: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples.")
        else:
            raise ValueError(f"Unsupported dataset_mode type: {mode_type}")

        self._save_scaler()

    def _setup_model(self):
        """Instantiate the model class named in ``config.model_class_name``.

        Input dim is inferred from the first training batch; the
        class name is resolved from module globals, which means any
        custom classifier defined in this same file is picked up
        automatically.
        """
        sample_batch = next(iter(self.trainloader))
        input_dim = sample_batch['input'].shape[1]
        model_params = self.config.model_params.copy()
        model_params['input_dim'] = input_dim
        model_class = globals().get(self.config.model_class_name)
        if model_class is None:
            raise ValueError(f"Model class '{self.config.model_class_name}' not found.")
        self.model = model_class(**model_params).to(self.device)
        logging.info(f"Model '{self.config.model_class_name}' created with input_dim={input_dim}.")

    def _setup_optimizer(self):
        self.optimizer = get_optimizer(self.model.parameters(), self.config.optimizer_config)
        self.scheduler = get_scheduler(self.optimizer, self.config.scheduler_config, {'epochs': self.config.epochs})

    def _setup_loss_function(self):
        """Build the loss module from ``config.loss_config``."""
        self.loss_function = get_loss_function(self.config.loss_config)
        logging.info(f"Using loss function: {self.config.loss_config.get('type', 'bce_logits')}")

    def _setup_early_stopping(self):
        if self.config.early_stopping > 0:
            self.early_stopping = EarlyStopping(patience=self.config.early_stopping,
                                               checkpoint_path=self.logdir / 'checkpoint.pkl',
                                               best_model_path=self.logdir / 'best_model.pkl',
                                               verbose=True)

    def _save_scaler(self):
        scaler_path = self.logdir / 'scaler.pkl'
        joblib.dump(self.scaler_X, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")

    def _train_epoch(self):
        """Run a single training epoch and return per-epoch metrics."""
        self.model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for batch in self.trainloader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()

            if self.config.gradient_clipping['enabled']:
                if self.config.gradient_clipping['method'] == 'norm':
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clipping['max_norm'])
                elif self.config.gradient_clipping['method'] == 'value':
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.config.gradient_clipping['clip_value'])

            self.optimizer.step()
            running_loss += loss.item()

            # Apply sigmoid to get probabilities for metrics calculation
            pred_proba = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
            true_labels = targets.cpu().numpy().flatten()
            all_labels.extend(true_labels)
            all_preds.extend(pred_proba)

        return calculate_classification_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(self.trainloader))

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on ``loader`` and return aggregate metrics."""
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for batch in loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            pred_proba = torch.sigmoid(outputs).cpu().numpy().flatten()
            true_labels = targets.cpu().numpy().flatten()
            all_labels.extend(true_labels)
            all_preds.extend(pred_proba)

        return calculate_classification_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(loader))

    def _update_and_log(self, epoch, train_metrics, val_metrics, duration):
        """Log one epoch's numbers + step the LR scheduler."""
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])

        logging.info(f"Epoch {epoch+1}/{self.config.epochs} | Duration: {duration:.2f}m")
        logging.info(f"\tTrain Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        logging.info(f"\tVal   Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
        logging.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _plot_metrics(self):
        """Render the 6-panel train / val metric curve and save to disk."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics_map = {'loss': 'Loss', 'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1 Score', 'auc': 'AUC-ROC'}
        for ax, (metric, title) in zip(axes.flatten(), metrics_map.items()):
            ax.plot(self.train_metrics[metric], label=f'Train {title}')
            ax.plot(self.val_metrics[metric], label=f'Validation {title}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.logdir / 'metrics_curve.jpg', dpi=300)
        plt.close()

    def run(self):
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
                self.early_stopping(val_metrics['loss'], self.model)
                if self.early_stopping.early_stop:
                    logging.info("Early stopping triggered.")
                    break
        logging.info("--- Training Finished ---")
        self._test()
        torch.cuda.empty_cache()

    def _test(self):
        """Restore the best checkpoint and evaluate on the test loader."""
        logging.info("--- Starting Testing ---")
        best_model_path = self.logdir / 'best_model.pkl'
        if not best_model_path.exists():
            logging.warning("No best model found. Testing with the last model.")
            best_model_path = self.logdir / 'checkpoint.pkl'
        self.model.load_state_dict(torch.load(best_model_path))
        metrics = self._evaluate(self.testloader)
        logging.info("Final Test Set Results:")
        logging.info(f"\tTest Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        logging.info(f"\tF1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        logging.info(f"\tConfusion Matrix - TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
        with open(self.logdir / 'test_results.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"Test {k}: {v:.6f}\n")


# ============================
# Inference helper
# ============================
class GalaxyClassificationInference:
    """Load a saved model directory and run inference on new data.

    Expects ``model_dir`` to contain the files produced by a
    successful :meth:`Trainer.run` — ``config.yaml``, ``scaler.pkl``,
    and at least one of ``best_model.pkl`` / ``checkpoint.pkl``.

    Args:
        model_dir: Path to the experiment directory.
        device: Torch device string (``"cuda"``, ``"cuda:0"``,
            ``"cpu"``). Falls back to CPU with a warning when CUDA
            is requested but unavailable.
        batch_size: Default inference batch size.
        num_workers: ``DataLoader`` ``num_workers``.
    """

    def __init__(self, model_dir: str, device: str = 'cuda:0', batch_size: int = 4096, num_workers: int = 4):
        self.model_dir = Path(model_dir)
        self.device = self._setup_device(device)
        cfg_path = self.model_dir / 'config.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        self.config = Config()
        self.config.update_from_yaml(str(cfg_path))
        scaler_path = self.model_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        self._load_model()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _setup_device(self, device: str) -> torch.device:
        if device == 'cpu':
            return torch.device('cpu')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device(device if device != 'cuda' else 'cuda')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_model(self):
        """Locate the checkpoint, reconstruct the model, load weights."""
        model_path = self.model_dir / 'best_model.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'checkpoint.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"No model file found in {self.model_dir}")
        if hasattr(self.scaler, 'n_features_in_'):
            input_dim = self.scaler.n_features_in_
        else:
            raise ValueError("Cannot determine input dimension from scaler")
        model_params = self.config.model_params.copy()
        model_params['input_dim'] = input_dim
        model_class = globals().get(self.config.model_class_name)
        self.model = model_class(**model_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def _create_dataset(self, data_source):
        """Wrap ``data_source`` (path or DataFrame) as an inference dataset."""
        if isinstance(data_source, str):
            ds = DatasetGalaxyClassification(file_path=data_source,
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        label_column=self.config.dataset_params.get('label_column', 'label'),
                                        scaler_X=self.scaler, mode='inference')
        else:
            ds = DatasetGalaxyClassification(dataset=data_source,
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        label_column=self.config.dataset_params.get('label_column', 'label'),
                                        scaler_X=self.scaler, mode='inference')
        return ds

    @torch.no_grad()
    def predict(self, data_source, batch_size: int = None, return_uncertainty: bool = False, threshold: float = 0.5):
        """Predict galaxy-class probabilities for the given data.

        Args:
            data_source: Input data (file path or DataFrame).
            batch_size: Override the default batch size.
            return_uncertainty: If ``True``, keep dropout active at
                inference time and run 100 stochastic forward passes
                per batch (MC-dropout), returning the mean as the
                prediction and the standard deviation as the
                uncertainty.
            threshold: Classification threshold applied to the mean
                probability to produce hard ``predictions``.

        Returns:
            Dict with ``probabilities`` ``(n,)`` and ``predictions``
            ``(n,)``; plus ``uncertainties`` ``(n,)`` when
            ``return_uncertainty=True``.
        """
        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        all_probs = []
        all_uncertainties = []

        if return_uncertainty:
            # Monte Carlo dropout for uncertainty estimation
            n_forward = 100
            self.model.train()  # Enable dropout

            for batch in dl:
                inputs = batch['input'].to(self.device)
                batch_preds = []

                for _ in range(n_forward):
                    outputs = torch.sigmoid(self.model(inputs)).cpu().numpy().flatten()
                    batch_preds.append(outputs)

                batch_preds = np.array(batch_preds)  # [n_forward, batch_size]
                mean_pred = batch_preds.mean(axis=0)
                std_pred = batch_preds.std(axis=0)

                all_probs.append(mean_pred)
                all_uncertainties.append(std_pred)

            self.model.eval()  # Disable dropout
            prob_arr = np.concatenate(all_probs, axis=0)
            uncertainty_arr = np.concatenate(all_uncertainties, axis=0)
            class_arr = (prob_arr >= threshold).astype(int)
            return {'probabilities': prob_arr, 'predictions': class_arr, 'uncertainties': uncertainty_arr}
        else:
            for batch in dl:
                inputs = batch['input'].to(self.device)
                outputs = torch.sigmoid(self.model(inputs)).cpu().numpy().flatten()
                all_probs.append(outputs)

            prob_arr = np.concatenate(all_probs, axis=0)
            class_arr = (prob_arr >= threshold).astype(int)
            return {'probabilities': prob_arr, 'predictions': class_arr}


# ============================
# Main
# ============================
def main():
    """CLI entry point: load a YAML config and run training."""
    parser = argparse.ArgumentParser(description="Train a classification model for galaxy classification.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config.yaml file.")
    args = parser.parse_args()

    config = Config()
    if Path(args.config).exists():
        config.update_from_yaml(args.config)
    else:
        logging.warning(f"Config file not found at {args.config}. Using default values.")
    config.config_path = args.config

    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()

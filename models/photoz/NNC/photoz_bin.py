import os
import time
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Iterable, Callable, Union, Tuple

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss

import joblib

# External utils (same as photoz_ann.py)
import cosmic.utils as cu


# ============================
# Config
# ============================
@dataclass
class Config:
    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "photoz_bin"
    random_seed: int = 2025
    gpu_id: str = '0'

    # Model
    model_params: Dict[str, Any] = field(default_factory=dict)
    model_class_name: str = 'PhotozBinningClassifier'

    # Data
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    dataset_class_name: str = 'DatasetPhotozBinned'
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
        'params': {'mode': 'min', 'factor': 0.5, 'patience': 5, 'verbose': True}
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
    loss_config: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'kl_divergence',  # 'kl_divergence', 'weighted_kl', 'focal', 'crps'
        'bin_weights': {
            'enabled': False,
            'method': 'inverse_frequency'  # 'inverse_frequency', 'balanced', 'sqrt_inverse'
        },
        'sample_weights': {
            'enabled': False
        },
        'focal': {
            'alpha': 1.0,
            'gamma': 2.0
        }
    })

    # Balanced sampling configuration
    sampling_config: Dict[str, Any] = field(default_factory=lambda: {
        'balanced_sampling': {
            'enabled': False,
            'bins': 10,  # Number of bins for balanced sampling
            'z_min': 0.0,
            'z_max': 1.2
        }
    })

    # PIT monitoring configuration
    pit_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,              # Whether to compute PIT during training
        'monitor_interval': 1,        # Compute PIT every N epochs
        'n_bins': 20,                 # Number of bins for PIT histogram
        'plot': True,                 # Whether to save PIT plots
        'plot_interval': 10           # Save PIT plot every N epochs
    })

    def update_from_yaml(self, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.update_from_dict(cfg)

    def update_from_dict(self, d: Dict[str, Any]):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ============================
# Dataset with binning
# ============================
class DatasetPhotozBinned(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_path: str = None,
                 feature_generation: dict = None,
                 feature_columns: list = None,
                 label_column: str = 'z',
                 binning_config: dict = None,
                 scaler_X: StandardScaler = None,
                 mode: str = 'train'):

        # data
        if dataset is not None and file_path is not None:
            raise ValueError("Cannot specify both 'dataset' and 'file_path'.")
        if dataset is None and file_path is None:
            raise ValueError("Must provide 'dataset' or 'file_path'.")

        if file_path is not None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            self.dataset = cu.readfile(file_path)
        else:
            self.dataset = dataset

        # feature
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

        # label
        self.label_column = label_column
        
        # binning (must be set before processing labels)
        if binning_config is not None:
            self.z_min = binning_config.get('z_min', 0.0)
            self.z_max = binning_config.get('z_max', 1.2)
            self.num_bins = binning_config.get('num_bins', 120)
            self.soft_label_width = binning_config.get('soft_label_width', None)  # in bin units, None to disable, >1 to enable
            self.bin_edges = np.linspace(self.z_min, self.z_max, self.num_bins + 1)
            self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        else:
            raise ValueError("binning_config must be provided.")

        # mode
        self.mode = mode
        if self.mode in ['train', 'validation', 'test']:
            self.labels_cont = self.dataset[self.label_column].values.astype(np.float32)
            self.labels_probs = self._z_to_soft_bins(self.labels_cont)

        # scaler
        if scaler_X is not None:
            self.scaler_X = scaler_X
            self.features = self.scaler_X.transform(self.features)

    def _choose_features(self, mag_types, color_types, use_mag=True, use_magerr=True, use_colorerr=True):
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

    def _z_to_soft_bins(self, z_values: np.ndarray) -> np.ndarray:
        # Clip to range
        z = np.clip(z_values, self.z_min, self.z_max - 1e-8)
        # Find left bin index
        bin_indices = np.floor((z - self.z_min) / (self.z_max - self.z_min) * self.num_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Linear soft assignment to two nearest bins
        bin_left_edges = self.bin_edges[bin_indices]
        bin_right_edges = self.bin_edges[bin_indices + 1]
        t = (z - bin_left_edges) / (bin_right_edges - bin_left_edges)

        probs = np.zeros((len(z), self.num_bins), dtype=np.float32)
        left_weight = (1 - t)
        right_weight = t

        # Create row indices for advanced indexing
        rows = np.arange(len(z))
        
        # Assign left weights to current bins
        probs[rows, bin_indices] = left_weight
        
        # Handle right weights with boundary check
        right_indices = bin_indices + 1
        valid_mask = right_indices < self.num_bins
        
        # Assign right weights to next bins (where valid)
        probs[rows[valid_mask], right_indices[valid_mask]] = right_weight[valid_mask]
        
        # For samples at the last bin, add right weight to the same bin
        invalid_mask = ~valid_mask
        probs[rows[invalid_mask], bin_indices[invalid_mask]] += right_weight[invalid_mask]

        # Optional Gaussian smoothing
        if self.soft_label_width is not None and self.soft_label_width > 1:
            width_bins = int(self.soft_label_width)
            sigma_bins = width_bins / 3  # so that ±3σ covers width_bins
            kernel = np.exp(-0.5 * (np.arange(-width_bins, width_bins + 1) / sigma_bins) ** 2)
            kernel = kernel / kernel.sum()
            padded = np.pad(probs, ((0, 0), (width_bins, width_bins)), mode='edge')
            smoothed = np.array([
                np.convolve(padded[i], kernel, mode='same')[width_bins:-width_bins] for i in range(padded.shape[0])
            ])
            probs = smoothed.astype(np.float32)

        # Normalize row-wise
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(probs_sum, 1e-8, None)
        # Add tiny epsilon and renormalize to avoid exact zeros causing numerical issues
        eps = 1e-8
        probs = probs + eps
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        if self.mode in ['train', 'validation', 'test']:
            y_prob = torch.FloatTensor(self.labels_probs[idx])
            y_z = torch.FloatTensor([self.labels_cont[idx]])
            return {'input': x, 'target_prob': y_prob, 'label_z': y_z}
        return {'input': x}


# ============================
# Model
# ============================
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
    'elu': nn.ELU,
    'silu': nn.SiLU,
}


class PhotozBinningClassifier(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_bins: int,
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
                self.residual_layers.append(nn.Linear(in_dim, h) if in_dim != h else nn.Identity())
            else:
                self.residual_layers.append(None)
            in_dim = h

        self.classifier = nn.Linear(in_dim, num_bins)
        self._initialize_weights()

    def _initialize_weights(self):
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
        for layer, res in zip(self.layers, self.residual_layers):
            identity = x
            out = layer(x)
            if self.use_residual and res is not None:
                out = out + (identity if isinstance(res, nn.Identity) else res(identity))
            x = out
        logits = self.classifier(x)
        return logits


# ============================
# Optimizer & Scheduler
# ============================
def get_optimizer(parameters: Iterable[torch.nn.Parameter], config: Dict[str, Any]) -> optim.Optimizer:
    name = config.get('name', 'AdamW')
    params = config.get('params', {})
    opt_cls = getattr(optim, name, None)
    if opt_cls is None:
        raise ValueError(f"Unsupported optimizer: {name}")
    return opt_cls(parameters, **params)


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], training_context: Dict[str, Any] = None) -> _LRScheduler:
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


def kl_divergence_with_softmax_logits(logits: Tensor, target_probs: Tensor, reduction='mean') -> Tensor:
    """
    Calculates the cross-entropy loss between model logits and target probabilities.

    This is equivalent to KL divergence for optimization purposes when the target
    distribution is fixed, as the entropy of the target is constant.
    This implementation is numerically stable.

    Args:
        logits (Tensor): Raw, unnormalized output from the model.
        target_probs (Tensor): Target probabilities (soft labels).
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        Tensor: The calculated loss.
    """
    # Stable soft cross-entropy with soft targets
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        return loss.sum()


def weighted_kl_divergence_loss(logits: Tensor, target_probs: Tensor, sample_weights: Tensor = None, 
                               bin_weights: Tensor = None, reduction='mean') -> Tensor:
    """
    Weighted KL divergence loss to handle sample imbalance.
    
    Args:
        logits: Model predictions [batch_size, num_bins]
        target_probs: Target probability distributions [batch_size, num_bins]
        sample_weights: Per-sample weights [batch_size]
        bin_weights: Per-bin weights [num_bins]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Weighted loss
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)  # [batch_size]
    
    # Apply bin-wise weights
    if bin_weights is not None:
        bin_weighted_loss = -(target_probs * log_probs * bin_weights.unsqueeze(0)).sum(dim=1)
        loss = bin_weighted_loss
    
    # Apply sample weights
    if sample_weights is not None:
        loss = loss * sample_weights
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def focal_loss(logits: Tensor, target_probs: Tensor, alpha: float = 1.0, gamma: float = 2.0,
               reduction='mean') -> Tensor:
    """
    Focal loss for addressing class imbalance in soft label setting.

    Args:
        logits: Model predictions [batch_size, num_bins]
        target_probs: Target probability distributions [batch_size, num_bins]
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Focal loss
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    # Calculate focal weight: (1 - p_t)^gamma
    # For soft labels, use expected probability
    pt = (target_probs * probs).sum(dim=1)  # Expected probability
    focal_weight = (1 - pt) ** gamma

    # Standard cross-entropy with soft labels
    ce_loss = -(target_probs * log_probs).sum(dim=1)

    # Apply focal weighting
    loss = alpha * focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def anchor_loss(logits: Tensor, target_probs: Tensor, gamma: float = 0.5,
                slack: float = 0.05, anchor: str = 'neg', reduction='mean') -> Tensor:
    """
    Anchor Loss for modulating loss scale based on prediction difficulty.

    Reference: Ryou et al. "Anchor Loss: Modulating Loss Scale based on
    Prediction Difficulty" (ICCV 2019)

    Used in photometric redshift estimation:
    Lee & Shin "Estimation of Photometric Redshifts. I. Machine-learning
    Inference for Pan-STARRS1 Galaxies Using Neural Networks" (AJ 2021)

    Args:
        logits: Model predictions [batch_size, num_bins]
        target_probs: Target probability distributions [batch_size, num_bins]
        gamma: Modulation strength. γ=0 reduces to standard cross-entropy.
               Typical values: 0.2, 0.5, 1.0
        slack: Small margin to prevent gradient vanishing (default: 0.05)
        anchor: Anchor type - 'neg' (max non-target prob) or 'pos' (target prob)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Anchor loss value
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    # For soft labels, find the peak bin as "target"
    target_bins = target_probs.argmax(dim=1)  # [batch_size]

    # Get target prediction probability
    batch_size = logits.size(0)
    p_target = probs[torch.arange(batch_size, device=logits.device), target_bins]  # [batch_size]

    if anchor == 'neg':
        # Anchor = max probability among non-target bins
        # Create mask for non-target bins
        mask = torch.ones_like(probs)
        mask[torch.arange(batch_size, device=logits.device), target_bins] = 0
        p_neg = (probs * mask).max(dim=1)[0]  # [batch_size]
        q = p_neg
    else:  # anchor == 'pos'
        q = p_target

    # Compute anchor weight: (1 + γ * (q - p_k + slack))^+
    # For soft labels, compute weighted loss across all bins
    # Weight increases when prediction is difficult (q > p_target)
    anchor_weight = torch.clamp(1 + gamma * (q.unsqueeze(1) - probs + slack), min=0)

    # Weighted cross-entropy with soft labels
    loss_per_bin = -target_probs * anchor_weight * log_probs
    loss = loss_per_bin.sum(dim=1)  # [batch_size]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def crps_loss(logits: Tensor, true_z: Tensor, bin_edges: Tensor, reduction='mean') -> Tensor:
    """
    Continuous Ranked Probability Score (CRPS) loss.

    CRPS measures the integrated squared difference between the predicted CDF
    and the empirical CDF (step function at true value). It naturally encourages
    both accuracy and calibration.

    Args:
        logits: Model predictions [batch_size, num_bins]
        true_z: True redshift values [batch_size] or [batch_size, 1]
        bin_edges: Bin edge values [num_bins + 1]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        CRPS loss
    """
    if true_z.dim() == 2:
        true_z = true_z.squeeze(1)

    probs = F.softmax(logits, dim=1)

    # Predicted CDF: cumulative sum of probabilities
    pred_cdf = torch.cumsum(probs, dim=1)  # [batch_size, num_bins]

    # True CDF: step function at true_z
    # CDF(z) = 1 if z >= true_z, else 0
    # For each bin's right edge: true_cdf = 1 if bin_right_edge >= true_z
    bin_rights = bin_edges[1:]  # [num_bins]
    true_cdf = (bin_rights.unsqueeze(0) >= true_z.unsqueeze(1)).float()  # [batch_size, num_bins]

    # CRPS = integral of (F_pred(z) - F_true(z))^2 dz
    bin_widths = bin_edges[1:] - bin_edges[:-1]  # [num_bins]
    crps = torch.sum((pred_cdf - true_cdf)**2 * bin_widths.unsqueeze(0), dim=1)

    if reduction == 'mean':
        return crps.mean()
    elif reduction == 'sum':
        return crps.sum()
    else:
        return crps


def emd_loss(logits: Tensor, true_z: Tensor, bin_edges: Tensor, reduction='mean') -> Tensor:
    """
    Earth Mover's Distance (EMD) / Wasserstein Distance loss.

    Similar to CRPS but uses L1 instead of L2, making it more robust to outliers
    and producing less sharp distributions.

    EMD = ∫|F_pred(z) - F_true(z)| dz

    Args:
        logits: Model predictions [batch_size, num_bins]
        true_z: True redshift values [batch_size] or [batch_size, 1]
        bin_edges: Bin edge values [num_bins + 1]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        EMD loss
    """
    if true_z.dim() == 2:
        true_z = true_z.squeeze(1)

    probs = F.softmax(logits, dim=1)

    # Predicted CDF
    pred_cdf = torch.cumsum(probs, dim=1)  # [batch_size, num_bins]

    # True CDF: step function at true_z
    bin_rights = bin_edges[1:]  # [num_bins]
    true_cdf = (bin_rights.unsqueeze(0) >= true_z.unsqueeze(1)).float()  # [batch_size, num_bins]

    # EMD = integral of |F_pred(z) - F_true(z)| dz
    bin_widths = bin_edges[1:] - bin_edges[:-1]  # [num_bins]
    emd = torch.sum(torch.abs(pred_cdf - true_cdf) * bin_widths.unsqueeze(0), dim=1)

    if reduction == 'mean':
        return emd.mean()
    elif reduction == 'sum':
        return emd.sum()
    else:
        return emd


def combined_loss(logits: Tensor, target_probs: Tensor, true_z: Tensor,
                  bin_edges: Tensor, config: dict, reduction='mean') -> Tensor:
    """
    Combined loss function that can mix multiple loss components.

    Args:
        logits: Model predictions [batch_size, num_bins]
        target_probs: Target probability distributions [batch_size, num_bins]
        true_z: True redshift values [batch_size]
        bin_edges: Bin edge values [num_bins + 1]
        config: Configuration dict with keys:
            - components: list of dicts with 'type' and 'weight'
            - entropy_reg: entropy regularization weight (optional)
            - z_weights: redshift-dependent weighting config (optional)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Combined loss value
    """
    if true_z.dim() == 2:
        true_z = true_z.squeeze(1)

    batch_size = logits.size(0)
    device = logits.device

    # Compute sample weights based on redshift ranges
    sample_weights = torch.ones(batch_size, device=device)
    z_weights_config = config.get('z_weights', None)
    if z_weights_config is not None and z_weights_config.get('enabled', False):
        ranges = z_weights_config.get('ranges', [])
        for range_cfg in ranges:
            z_min, z_max = range_cfg['z_min'], range_cfg['z_max']
            weight = range_cfg['weight']
            mask = (true_z >= z_min) & (true_z < z_max)
            sample_weights[mask] = weight

    # Compute each loss component
    total_loss = torch.zeros(batch_size, device=device)
    components = config.get('components', [{'type': 'kl', 'weight': 1.0}])

    for comp in components:
        loss_type = comp['type']
        weight = comp.get('weight', 1.0)

        if weight == 0:
            continue

        if loss_type == 'kl':
            log_probs = F.log_softmax(logits, dim=1)
            comp_loss = -(target_probs * log_probs).sum(dim=1)
        elif loss_type == 'crps':
            comp_loss = crps_loss(logits, true_z, bin_edges, reduction='none')
        elif loss_type == 'emd':
            comp_loss = emd_loss(logits, true_z, bin_edges, reduction='none')
        elif loss_type == 'focal':
            focal_cfg = comp.get('params', {'alpha': 1.0, 'gamma': 2.0})
            comp_loss = focal_loss(logits, target_probs,
                                   alpha=focal_cfg.get('alpha', 1.0),
                                   gamma=focal_cfg.get('gamma', 2.0),
                                   reduction='none')
        elif loss_type == 'anchor':
            anchor_cfg = comp.get('params', {'gamma': 0.5, 'slack': 0.05, 'anchor': 'neg'})
            comp_loss = anchor_loss(logits, target_probs,
                                    gamma=anchor_cfg.get('gamma', 0.5),
                                    slack=anchor_cfg.get('slack', 0.05),
                                    anchor=anchor_cfg.get('anchor', 'neg'),
                                    reduction='none')
        elif loss_type == 'brier':
            probs = F.softmax(logits, dim=1)
            comp_loss = ((probs - target_probs) ** 2).sum(dim=1)
        else:
            raise ValueError(f"Unknown loss component type: {loss_type}")

        total_loss = total_loss + weight * comp_loss

    # Entropy regularization (encourages smoother distributions)
    entropy_reg = config.get('entropy_reg', 0.0)
    if entropy_reg > 0:
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1)
        total_loss = total_loss - entropy_reg * entropy  # Subtract to encourage higher entropy

    # Apply sample weights
    total_loss = total_loss * sample_weights

    if reduction == 'mean':
        return total_loss.mean()
    elif reduction == 'sum':
        return total_loss.sum()
    else:
        return total_loss


def compute_bin_weights(bin_centers: np.ndarray, labels: np.ndarray, 
                       method: str = 'inverse_frequency') -> np.ndarray:
    """
    Compute weights for each bin based on sample distribution.
    
    Args:
        bin_centers: Center values of each bin
        labels: True redshift values
        method: 'inverse_frequency', 'balanced', or 'sqrt_inverse'
    
    Returns:
        Bin weights array
    """
    # Count samples in each bin
    bin_edges = np.concatenate([
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[:-1] + bin_centers[1:]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]
    ])
    
    counts, _ = np.histogram(labels, bins=bin_edges)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    
    if method == 'inverse_frequency':
        weights = 1.0 / counts
    elif method == 'balanced':
        weights = len(labels) / (len(bin_centers) * counts)
    elif method == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_sample_weights(labels: np.ndarray, bin_centers: np.ndarray, 
                          bin_weights: np.ndarray) -> np.ndarray:
    """
    Assign weights to individual samples based on their redshift bin.
    
    Args:
        labels: True redshift values for each sample
        bin_centers: Center values of each bin
        bin_weights: Precomputed weights for each bin
    
    Returns:
        Sample weights array
    """
    # Find closest bin for each sample
    bin_indices = np.searchsorted(bin_centers, labels, side='left')
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    
    # Assign bin weights to samples
    sample_weights = bin_weights[bin_indices]
    return sample_weights.astype(np.float32)


# ============================
# PIT Computation and Diagnosis
# ============================
def compute_pit_values(probs: np.ndarray, true_z: np.ndarray,
                       bin_edges: np.ndarray) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values.

    If the model is well-calibrated, PIT values should follow a uniform distribution U(0,1).

    Args:
        probs: Predicted probability distributions [n_samples, n_bins]
        true_z: True redshift values [n_samples]
        bin_edges: Bin edge values [n_bins + 1]

    Returns:
        PIT values [n_samples]
    """
    n_samples = len(true_z)
    num_bins = probs.shape[1]

    # Compute CDF
    cumsum = np.cumsum(probs, axis=1)

    # Find bin index for each true_z
    bin_idx = np.digitize(true_z, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    pit_values = np.zeros(n_samples)
    for i in range(n_samples):
        idx = bin_idx[i]
        # CDF at left edge of the bin
        cdf_left = cumsum[i, idx - 1] if idx > 0 else 0.0
        # Fraction within the bin
        bin_left = bin_edges[idx]
        bin_right = bin_edges[idx + 1]
        frac = (true_z[i] - bin_left) / (bin_right - bin_left)
        frac = np.clip(frac, 0, 1)
        # PIT = CDF at true_z position
        pit_values[i] = cdf_left + probs[i, idx] * frac

    return pit_values


def diagnose_pit(pit_values: np.ndarray, n_pit_bins: int = 20) -> Dict[str, Any]:
    """
    Diagnose calibration issues from PIT distribution.

    Args:
        pit_values: PIT values [n_samples]
        n_pit_bins: Number of bins for PIT histogram

    Returns:
        Dictionary with diagnosis results
    """
    from scipy import stats

    # KS test against uniform distribution
    ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')

    # Compute PIT histogram
    hist, edges = np.histogram(pit_values, bins=n_pit_bins, range=(0, 1))
    expected = len(pit_values) / n_pit_bins

    # Analyze distribution shape
    left_quarter = hist[:n_pit_bins // 4].sum()
    right_quarter = hist[-n_pit_bins // 4:].sum()
    center_half = hist[n_pit_bins // 4:-n_pit_bins // 4].sum()

    left_ratio = left_quarter / (expected * (n_pit_bins // 4))
    right_ratio = right_quarter / (expected * (n_pit_bins // 4))
    center_ratio = center_half / (expected * (n_pit_bins // 2))

    # Use KS statistic for calibration assessment (p-value unreliable for large N)
    # KS thresholds: <0.01 excellent, <0.02 good, <0.05 acceptable
    is_calibrated = ks_stat < 0.02

    diagnosis = {
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'is_calibrated': is_calibrated,
        'left_ratio': left_ratio,
        'right_ratio': right_ratio,
        'center_ratio': center_ratio,
        'pit_mean': np.mean(pit_values),
        'pit_std': np.std(pit_values),
    }

    # Determine problem type based on shape analysis
    if left_ratio > 1.3 and right_ratio > 1.3:
        diagnosis['problem'] = 'overconfident'
        diagnosis['suggestion'] = 'PDFs are too narrow. Use temperature > 1 to soften.'
    elif center_ratio > 1.3:
        diagnosis['problem'] = 'underconfident'
        diagnosis['suggestion'] = 'PDFs are too wide. Use temperature < 1 to sharpen.'
    elif left_ratio > 1.5:
        diagnosis['problem'] = 'high_bias'
        diagnosis['suggestion'] = 'Model systematically overestimates redshift.'
    elif right_ratio > 1.5:
        diagnosis['problem'] = 'low_bias'
        diagnosis['suggestion'] = 'Model systematically underestimates redshift.'
    elif ks_stat < 0.01:
        diagnosis['problem'] = 'none'
        diagnosis['suggestion'] = 'Excellent calibration.'
    elif ks_stat < 0.02:
        diagnosis['problem'] = 'none'
        diagnosis['suggestion'] = 'Good calibration.'
    elif ks_stat < 0.05:
        diagnosis['problem'] = 'minor'
        diagnosis['suggestion'] = 'Acceptable. Temperature scaling may help slightly.'
    else:
        diagnosis['problem'] = 'needs_calibration'
        diagnosis['suggestion'] = 'Consider temperature scaling.'

    return diagnosis


def plot_pit_diagnosis(pit_values: np.ndarray, save_path: str = None,
                       title: str = 'PIT Diagnosis'):
    """
    Visualize PIT distribution for calibration diagnosis.

    Args:
        pit_values: PIT values [n_samples]
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. PIT histogram
    axes[0].hist(pit_values, bins=20, range=(0, 1), density=True,
                 alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Ideal (uniform)')
    axes[0].set_xlabel('PIT value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('PIT Histogram')
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # 2. Q-Q plot
    sorted_pit = np.sort(pit_values)
    expected = np.linspace(0, 1, len(pit_values))
    axes[1].plot(expected, sorted_pit, 'b.', alpha=0.1, markersize=1)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal')
    axes[1].set_xlabel('Expected quantile')
    axes[1].set_ylabel('Observed PIT')
    axes[1].set_title('PIT Q-Q Plot')
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    # 3. CDF comparison
    axes[2].plot(sorted_pit, expected, 'b-', linewidth=2, label='Observed CDF')
    axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal (uniform)')
    axes[2].fill_between(sorted_pit, expected, np.linspace(0, 1, len(pit_values)),
                         alpha=0.3, color='blue')
    axes[2].set_xlabel('PIT value')
    axes[2].set_ylabel('CDF')
    axes[2].set_title('PIT CDF Comparison')
    axes[2].legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================
# Temperature Scaling Calibration
# ============================
class TemperatureCalibrator:
    """
    Temperature scaling calibrator for probability distribution calibration.

    - T > 1: Makes distribution flatter (fixes overconfidence)
    - T < 1: Makes distribution sharper (fixes underconfidence)
    """

    def __init__(self):
        self.temperature = 1.0
        self.fit_history = []

    def fit(self, logits: np.ndarray, true_z: np.ndarray, bin_edges: np.ndarray,
            method: str = 'minimize_ks', verbose: bool = True) -> 'TemperatureCalibrator':
        """
        Find optimal temperature on validation set.

        Args:
            logits: Raw model outputs [n_samples, n_bins]
            true_z: True redshift values [n_samples]
            bin_edges: Bin edge values [n_bins + 1]
            method: Optimization target ('minimize_ks', 'minimize_crps')
            verbose: Whether to print progress

        Returns:
            self
        """
        from scipy.optimize import minimize_scalar

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

        def objective(T):
            if T <= 0:
                return 1e10
            scaled_logits = logits / T
            probs = softmax(scaled_logits)
            pit = compute_pit_values(probs, true_z, bin_edges)

            if method == 'minimize_ks':
                from scipy import stats
                ks_stat, _ = stats.kstest(pit, 'uniform')
                return ks_stat
            elif method == 'minimize_crps':
                # Compute CRPS
                cumsum = np.cumsum(probs, axis=1)
                bin_rights = bin_edges[1:]
                true_cdf = (bin_rights[None, :] <= true_z[:, None]).astype(float)
                bin_widths = bin_edges[1:] - bin_edges[:-1]
                crps = np.mean(np.sum((cumsum - true_cdf) ** 2 * bin_widths[None, :], axis=1))
                return crps
            else:
                raise ValueError(f"Unknown method: {method}")

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.fit_history.append({
            'temperature': self.temperature,
            'objective': result.fun,
            'method': method
        })

        if verbose:
            logging.info(f"Temperature scaling: T = {self.temperature:.4f} (objective = {result.fun:.6f})")

        return self

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        scaled = self.calibrate_logits(logits)
        e_x = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def save(self, path: str):
        """Save calibrator to file."""
        joblib.dump({
            'temperature': self.temperature,
            'fit_history': self.fit_history
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TemperatureCalibrator':
        """Load calibrator from file."""
        data = joblib.load(path)
        calibrator = cls()
        calibrator.temperature = data['temperature']
        calibrator.fit_history = data.get('fit_history', [])
        return calibrator


class BalancedBatchSampler:
    """
    Custom batch sampler for balanced sampling across redshift bins.
    """
    def __init__(self, dataset, batch_size: int, num_bins: int = 10, 
                 z_min: float = 0.0, z_max: float = 1.2, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get all labels
        all_labels = []
        for i in range(len(dataset)):
            all_labels.append(dataset[i]['label_z'].item())
        all_labels = np.array(all_labels)
        
        # Create bins and assign samples to bins
        bin_edges = np.linspace(z_min, z_max, num_bins + 1)
        bin_indices = np.digitize(all_labels, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Group indices by bin
        self.bin_to_indices = {}
        for i, bin_idx in enumerate(bin_indices):
            if bin_idx not in self.bin_to_indices:
                self.bin_to_indices[bin_idx] = []
            self.bin_to_indices[bin_idx].append(i)
        
        # Remove empty bins
        self.bin_to_indices = {k: v for k, v in self.bin_to_indices.items() if len(v) > 0}
        self.num_bins = len(self.bin_to_indices)
        
        logging.info(f"BalancedBatchSampler: Using {self.num_bins} non-empty bins")
        for bin_idx, indices in self.bin_to_indices.items():
            logging.info(f"  Bin {bin_idx}: {len(indices)} samples")
    
    def __iter__(self):
        if self.shuffle:
            for indices in self.bin_to_indices.values():
                np.random.shuffle(indices)
        
        # Create iterators for each bin
        bin_iterators = {}
        for bin_idx, indices in self.bin_to_indices.items():
            bin_iterators[bin_idx] = iter(indices * (len(indices) // len(indices) + 1))
        
        samples_per_bin = max(1, self.batch_size // self.num_bins)
        remainder = self.batch_size % self.num_bins
        
        total_samples = sum(len(indices) for indices in self.bin_to_indices.values())
        num_batches = total_samples // self.batch_size
        
        for _ in range(num_batches):
            batch = []
            bins = list(self.bin_to_indices.keys())
            if self.shuffle:
                np.random.shuffle(bins)
            
            for i, bin_idx in enumerate(bins):
                current_samples_per_bin = samples_per_bin + (1 if i < remainder else 0)
                
                for _ in range(current_samples_per_bin):
                    try:
                        batch.append(next(bin_iterators[bin_idx]))
                    except StopIteration:
                        # Restart iterator for this bin
                        indices = self.bin_to_indices[bin_idx]
                        if self.shuffle:
                            np.random.shuffle(indices)
                        bin_iterators[bin_idx] = iter(indices)
                        batch.append(next(bin_iterators[bin_idx]))
            
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch
    
    def __len__(self):
        total_samples = sum(len(indices) for indices in self.bin_to_indices.values())
        return total_samples // self.batch_size


def expected_redshift_from_probs(probs: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    return (probs * bin_centers[None, :]).sum(axis=1)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, running_loss: float, num_batches: int) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'loss': running_loss / max(1, num_batches), 'mse': mse, 'mae': mae, 'r2': r2}


# ============================
# Trainer
# ============================
class Trainer:
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

        self.train_metrics = {'loss': [], 'mse': [], 'mae': [], 'r2': []}
        self.val_metrics = {'loss': [], 'mse': [], 'mae': [], 'r2': []}

        # For handling sample imbalance
        self.bin_weights = None
        self.loss_function = None

        # For CRPS loss
        self.bin_edges_tensor = None

        # For PIT monitoring
        self.pit_metrics = {'ks_stat': [], 'ks_pval': [], 'problem': []}

    def _setup(self):
        self._setup_seed()
        self._setup_logging()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss_function()
        self._setup_early_stopping()

    def _setup_seed(self):
        seed = self.config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_logging(self):
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
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def _create_datasets_from_paths(self, paths: Dict[str, str]):
        # Fit scaler on train
        tmp_train = DatasetPhotozBinned(file_path=paths['train'],
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        binning_config=self.config.dataset_params.get('binning_config'),
                                        scaler_X=None,
                                        mode='train')
        scaler_X = StandardScaler()
        scaler_X.fit(tmp_train.features)

        train_ds = DatasetPhotozBinned(file_path=paths['train'],
                                       feature_generation=self.config.dataset_params.get('feature_generation'),
                                       feature_columns=self.config.dataset_params.get('feature_columns'),
                                       binning_config=self.config.dataset_params.get('binning_config'),
                                       scaler_X=scaler_X,
                                       mode='train')
        val_ds = DatasetPhotozBinned(file_path=paths['val'],
                                     feature_generation=self.config.dataset_params.get('feature_generation'),
                                     feature_columns=self.config.dataset_params.get('feature_columns'),
                                     binning_config=self.config.dataset_params.get('binning_config'),
                                     scaler_X=scaler_X,
                                     mode='validation')
        test_ds = DatasetPhotozBinned(file_path=paths['test'],
                                      feature_generation=self.config.dataset_params.get('feature_generation'),
                                      feature_columns=self.config.dataset_params.get('feature_columns'),
                                      binning_config=self.config.dataset_params.get('binning_config'),
                                      scaler_X=scaler_X,
                                      mode='test')

        self.scaler_X = scaler_X
        
        # Setup training dataloader with optional balanced sampling
        balanced_sampling_config = self.config.sampling_config.get('balanced_sampling', {'enabled': False, 'bins': 10, 'z_min': 0.0, 'z_max': 1.2})
        if balanced_sampling_config.get('enabled', False):
            sampler = BalancedBatchSampler(
                train_ds, 
                batch_size=self.config.batch_size,
                num_bins=balanced_sampling_config.get('bins', 10),
                z_min=balanced_sampling_config.get('z_min', 0.0),
                z_max=balanced_sampling_config.get('z_max', 1.2),
                shuffle=True
            )
            self.trainloader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
        else:
            self.trainloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        self.valloader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.testloader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logging.info(f"Data loaded from paths: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test samples.")

    def _setup_data(self):
        mode_type = self.config.dataset_mode.get('type', 'files')
        if mode_type == 'files':
            paths = self.config.dataset_mode.get('paths', {})
            if not all(paths.get(s) for s in ['train', 'val', 'test']):
                raise ValueError("When using 'files' mode, all paths (train, val, test) must be specified")
            self._create_datasets_from_paths(paths)
        elif mode_type == 'ratio':
            dataset = cu.readfile(self.config.dataset_params.get('file_path'))
            total = len(dataset)
            indices = list(range(total))
            ratios = self.config.dataset_mode.get('ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
            val_test_ratio = ratios['val'] + ratios['test']
            train_idx, temp_idx = train_test_split(indices, test_size=val_test_ratio, random_state=self.config.random_seed)
            val_size_corr = ratios['val'] / val_test_ratio
            val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size_corr), random_state=self.config.random_seed)

            temp_ds = DatasetPhotozBinned(dataset=dataset,
                                          feature_generation=self.config.dataset_params.get('feature_generation'),
                                          feature_columns=self.config.dataset_params.get('feature_columns'),
                                          binning_config=self.config.dataset_params.get('binning_config'),
                                          scaler_X=None, mode='inference')
            scaler_X = StandardScaler(); scaler_X.fit(temp_ds.features[np.array(train_idx)])

            train_ds = DatasetPhotozBinned(dataset=dataset,
                                           feature_generation=self.config.dataset_params.get('feature_generation'),
                                           feature_columns=self.config.dataset_params.get('feature_columns'),
                                           binning_config=self.config.dataset_params.get('binning_config'),
                                           scaler_X=scaler_X, mode='train')
            val_ds = DatasetPhotozBinned(dataset=dataset,
                                         feature_generation=self.config.dataset_params.get('feature_generation'),
                                         feature_columns=self.config.dataset_params.get('feature_columns'),
                                         binning_config=self.config.dataset_params.get('binning_config'),
                                         scaler_X=scaler_X,
                                         mode='validation')
            test_ds = DatasetPhotozBinned(dataset=dataset,
                                          feature_generation=self.config.dataset_params.get('feature_generation'),
                                          feature_columns=self.config.dataset_params.get('feature_columns'),
                                          binning_config=self.config.dataset_params.get('binning_config'),
                                          scaler_X=scaler_X, mode='test')

            self.scaler_X = scaler_X
            
            # Setup training dataloader with optional balanced sampling
            train_subset = Subset(train_ds, train_idx)
            balanced_sampling_config = self.config.sampling_config.get('balanced_sampling', {'enabled': False, 'bins': 10, 'z_min': 0.0, 'z_max': 1.2})
            if balanced_sampling_config.get('enabled', False):
                sampler = BalancedBatchSampler(
                    train_subset, 
                    batch_size=self.config.batch_size,
                    num_bins=balanced_sampling_config.get('bins', 10),
                    z_min=balanced_sampling_config.get('z_min', 0.0),
                    z_max=balanced_sampling_config.get('z_max', 1.2),
                    shuffle=True
                )
                self.trainloader = DataLoader(train_subset, batch_sampler=sampler, num_workers=4, pin_memory=True)
            else:
                self.trainloader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            
            self.valloader = DataLoader(Subset(val_ds, val_idx), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            self.testloader = DataLoader(Subset(test_ds, test_idx), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            logging.info(f"Data loaded: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples.")
        else:
            raise ValueError(f"Unsupported dataset_mode type: {mode_type}")

        self._save_scaler()

    def _setup_model(self):
        sample_batch = next(iter(self.trainloader))
        input_dim = sample_batch['input'].shape[1]
        model_params = self.config.model_params.copy()
        model_params['input_dim'] = input_dim
        model_params['num_bins'] = self.config.dataset_params['binning_config']['num_bins']
        model_class = globals().get(self.config.model_class_name)
        if model_class is None:
            raise ValueError(f"Model class '{self.config.model_class_name}' not found.")
        self.model = model_class(**model_params).to(self.device)
        logging.info(f"Model '{self.config.model_class_name}' created with input_dim={input_dim}, num_bins={self.config.dataset_params['binning_config']['num_bins']}.")

    def _setup_optimizer(self):
        self.optimizer = get_optimizer(self.model.parameters(), self.config.optimizer_config)
        self.scheduler = get_scheduler(self.optimizer, self.config.scheduler_config, {'epochs': self.config.epochs})
    
    def _setup_loss_function(self):
        """Setup loss function and weights for handling sample imbalance."""
        loss_type = self.config.loss_config.get('type', 'kl_divergence')

        # Setup bin_edges tensor for CRPS loss
        binning_config = self.config.dataset_params.get('binning_config', {})
        z_min = binning_config.get('z_min', 0.0)
        z_max = binning_config.get('z_max', 1.2)
        num_bins = binning_config.get('num_bins', 120)
        self.bin_edges_tensor = torch.FloatTensor(
            np.linspace(z_min, z_max, num_bins + 1)
        ).to(self.device)

        bin_weights_config = self.config.loss_config.get('bin_weights', {'enabled': False, 'method': 'inverse_frequency'})
        if bin_weights_config.get('enabled', False):
            # Compute bin weights based on training data distribution
            bin_centers = self.trainloader.dataset.bin_centers

            # Collect all training labels
            all_labels = []
            for batch in self.trainloader:
                all_labels.extend(batch['label_z'].numpy().flatten())
            all_labels = np.array(all_labels)

            # Compute bin weights
            method = bin_weights_config.get('method', 'inverse_frequency')
            self.bin_weights = compute_bin_weights(bin_centers, all_labels, method)
            self.bin_weights = torch.FloatTensor(self.bin_weights).to(self.device)

            logging.info(f"Computed bin weights using method '{method}'")
            logging.info(f"Bin weight range: {self.bin_weights.min().item():.3f} - {self.bin_weights.max().item():.3f}")

        # Setup loss function
        if loss_type == 'kl_divergence':
            self.loss_function = kl_divergence_with_softmax_logits
            self._use_crps = False
        elif loss_type == 'weighted_kl':
            self.loss_function = lambda logits, target_probs, true_z=None: weighted_kl_divergence_loss(
                logits, target_probs, bin_weights=self.bin_weights)
            self._use_crps = False
        elif loss_type == 'focal':
            focal_config = self.config.loss_config.get('focal', {'alpha': 1.0, 'gamma': 2.0})
            alpha = focal_config.get('alpha', 1.0)
            gamma = focal_config.get('gamma', 2.0)
            self.loss_function = lambda logits, target_probs, true_z=None: focal_loss(
                logits, target_probs, alpha=alpha, gamma=gamma)
            self._use_crps = False
        elif loss_type == 'crps':
            # CRPS loss requires true_z values
            self.loss_function = lambda logits, target_probs, true_z: crps_loss(
                logits, true_z, self.bin_edges_tensor)
            self._use_crps = True
        elif loss_type == 'anchor':
            # Anchor loss (Ryou et al. 2019, used in Lee & Shin 2021 for photo-z)
            anchor_config = self.config.loss_config.get('anchor', {'gamma': 0.5, 'slack': 0.05, 'anchor': 'neg'})
            gamma = anchor_config.get('gamma', 0.5)
            slack = anchor_config.get('slack', 0.05)
            anchor_type = anchor_config.get('anchor', 'neg')
            self.loss_function = lambda logits, target_probs, true_z=None: anchor_loss(
                logits, target_probs, gamma=gamma, slack=slack, anchor=anchor_type)
            self._use_crps = False
        elif loss_type == 'emd':
            # Earth Mover's Distance / Wasserstein loss
            self.loss_function = lambda logits, target_probs, true_z: emd_loss(
                logits, true_z, self.bin_edges_tensor)
            self._use_crps = True  # EMD also requires true_z like CRPS
        elif loss_type == 'combined':
            # Combined loss with multiple components and z-dependent weights
            combined_config = self.config.loss_config.get('combined', {
                'components': [{'type': 'kl', 'weight': 1.0}],
                'entropy_reg': 0.0,
                'z_weights': {'enabled': False}
            })
            self.loss_function = lambda logits, target_probs, true_z: combined_loss(
                logits, target_probs, true_z, self.bin_edges_tensor, combined_config)
            self._use_crps = True  # Combined loss requires true_z for CRPS/EMD components
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        logging.info(f"Using loss function: {loss_type}")

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
        self.model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for batch in self.trainloader:
            inputs = batch['input'].to(self.device)
            target_prob = batch['target_prob'].to(self.device)
            label_z = batch['label_z']
            label_z_np = label_z.cpu().numpy().flatten()

            self.optimizer.zero_grad()
            logits = self.model(inputs)

            # Compute loss (CRPS needs true_z, others don't)
            if self._use_crps:
                loss = self.loss_function(logits, target_prob, label_z.to(self.device))
            else:
                loss = self.loss_function(logits, target_prob)
            loss.backward()

            if self.config.gradient_clipping['enabled']:
                if self.config.gradient_clipping['method'] == 'norm':
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clipping['max_norm'])
                elif self.config.gradient_clipping['method'] == 'value':
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.config.gradient_clipping['clip_value'])

            self.optimizer.step()
            running_loss += loss.item()

            probs = F.softmax(logits.detach(), dim=1).cpu().numpy()
            # Expected redshift
            centers = self.trainloader.dataset.bin_centers
            pred_z = expected_redshift_from_probs(probs, centers)
            all_labels.extend(label_z_np)
            all_preds.extend(pred_z)

        return calculate_regression_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(self.trainloader))

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, return_logits: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """
        Evaluate model on a data loader.

        Args:
            loader: DataLoader to evaluate on
            return_logits: If True, also return logits and true_z for PIT computation

        Returns:
            If return_logits=False: metrics dict
            If return_logits=True: (metrics dict, logits array, true_z array)
        """
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds, all_logits = [], [], []
        centers = loader.dataset.bin_centers
        bin_edges = loader.dataset.bin_edges

        for batch in loader:
            inputs = batch['input'].to(self.device)
            target_prob = batch['target_prob'].to(self.device)
            label_z = batch['label_z']
            label_z_np = label_z.cpu().numpy().flatten()

            logits = self.model(inputs)

            # Compute loss
            if self._use_crps:
                loss = self.loss_function(logits, target_prob, label_z.to(self.device))
            else:
                loss = self.loss_function(logits, target_prob)
            running_loss += loss.item()

            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_z = expected_redshift_from_probs(probs, centers)
            all_labels.extend(label_z_np)
            all_preds.extend(pred_z)

            if return_logits:
                all_logits.append(logits.cpu().numpy())

        metrics = calculate_regression_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(loader))

        if return_logits:
            logits_arr = np.concatenate(all_logits, axis=0)
            labels_arr = np.array(all_labels)
            return metrics, logits_arr, labels_arr, bin_edges
        return metrics

    def _compute_pit_metrics(self, logits: np.ndarray, true_z: np.ndarray,
                             bin_edges: np.ndarray, epoch: int) -> Dict[str, Any]:
        """
        Compute PIT metrics and optionally save plots.

        Args:
            logits: Model logits [n_samples, n_bins]
            true_z: True redshift values [n_samples]
            bin_edges: Bin edge values [n_bins + 1]
            epoch: Current epoch number

        Returns:
            PIT diagnosis dictionary
        """
        # Compute probabilities
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)

        # Compute PIT values
        pit_values = compute_pit_values(probs, true_z, bin_edges)

        # Diagnose
        n_bins = self.config.pit_config.get('n_bins', 20)
        diagnosis = diagnose_pit(pit_values, n_pit_bins=n_bins)

        # Save plot if configured
        pit_config = self.config.pit_config
        if pit_config.get('plot', True):
            plot_interval = pit_config.get('plot_interval', 10)
            if (epoch + 1) % plot_interval == 0 or epoch == 0:
                plot_path = self.logdir / f'pit_epoch_{epoch+1:03d}.png'
                plot_pit_diagnosis(pit_values, save_path=str(plot_path),
                                  title=f'PIT Diagnosis - Epoch {epoch+1}')

        return diagnosis

    def _update_and_log(self, epoch, train_metrics, val_metrics, duration, pit_diagnosis=None):
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])

        logging.info(f"Epoch {epoch+1}/{self.config.epochs} | Duration: {duration:.2f}m")
        logging.info(f"\tTrain Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}")
        logging.info(f"\tVal   Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f}")

        # Log PIT metrics if available
        if pit_diagnosis is not None:
            self.pit_metrics['ks_stat'].append(pit_diagnosis['ks_stat'])
            self.pit_metrics['ks_pval'].append(pit_diagnosis['ks_pval'])
            self.pit_metrics['problem'].append(pit_diagnosis['problem'])

            calibrated_str = "✓" if pit_diagnosis['is_calibrated'] else "✗"
            logging.info(f"\tPIT: KS={pit_diagnosis['ks_stat']:.4f} [{calibrated_str}] | {pit_diagnosis['suggestion']}")

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
        logging.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _plot_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_map = {'loss': 'Loss', 'mse': 'MSE', 'mae': 'MAE', 'r2': 'R2'}
        for ax, (metric, title) in zip(axes.flatten(), metrics_map.items()):
            ax.plot(self.train_metrics[metric], label=f'Train {title}')
            ax.plot(self.val_metrics[metric], label=f'Validation {title}')
            ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.savefig(self.logdir / 'metrics_curve.jpg', dpi=300); plt.close()

    def run(self):
        self._setup()
        logging.info("--- Starting Training ---")

        pit_config = self.config.pit_config
        pit_enabled = pit_config.get('enabled', True)
        pit_interval = pit_config.get('monitor_interval', 1)

        for epoch in range(self.config.epochs):
            start_time = time.time()
            train_metrics = self._train_epoch()

            # Evaluate with or without PIT
            pit_diagnosis = None
            if pit_enabled and (epoch + 1) % pit_interval == 0:
                val_metrics, val_logits, val_true_z, bin_edges = self._evaluate(
                    self.valloader, return_logits=True)
                pit_diagnosis = self._compute_pit_metrics(val_logits, val_true_z, bin_edges, epoch)
            else:
                val_metrics = self._evaluate(self.valloader)

            duration = (time.time() - start_time) / 60
            self._update_and_log(epoch, train_metrics, val_metrics, duration, pit_diagnosis)
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
        logging.info("--- Starting Testing ---")
        best_model_path = self.logdir / 'best_model.pkl'
        if not best_model_path.exists():
            logging.warning("No best model found. Testing with the last model.")
            best_model_path = self.logdir / 'checkpoint.pkl'
        self.model.load_state_dict(torch.load(best_model_path))

        # Evaluate with PIT
        metrics, test_logits, test_true_z, bin_edges = self._evaluate(self.testloader, return_logits=True)

        logging.info("Final Test Set Results:")
        logging.info(f"\tTest Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

        # Compute PIT on test set
        e_x = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        pit_values = compute_pit_values(probs, test_true_z, bin_edges)
        pit_diagnosis = diagnose_pit(pit_values)

        calibrated_str = "✓" if pit_diagnosis['is_calibrated'] else "✗"
        logging.info(f"\tTest PIT: KS={pit_diagnosis['ks_stat']:.4f} [{calibrated_str}] | {pit_diagnosis['suggestion']}")

        # Save PIT plot for test set
        plot_pit_diagnosis(pit_values, save_path=str(self.logdir / 'pit_test.png'),
                          title='Test Set PIT Diagnosis')

        # Save results
        with open(self.logdir / 'test_results.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"Test {k}: {v:.6f}\n")
            f.write(f"\nPIT Diagnostics:\n")
            f.write(f"KS statistic: {pit_diagnosis['ks_stat']:.6f}\n")
            f.write(f"Problem: {pit_diagnosis['problem']}\n")
            f.write(f"Suggestion: {pit_diagnosis['suggestion']}\n")


# ============================
# Inference helper
# ============================
class PhotozBinInference:
    def __init__(self, model_dir: str, device: str = 'cuda:0', batch_size: int = 4096,
                 num_workers: int = 4):
        """
        Initialize inference helper.

        Args:
            model_dir: Path to the model directory
            device: Device to use for inference
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
        """
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

        # Calibrator (set via calibrate() method)
        self.calibrator = None

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
        model_params['num_bins'] = self.config.dataset_params['binning_config']['num_bins']
        model_class = globals().get(self.config.model_class_name)
        self.model = model_class(**model_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device); self.model.eval()

    def _create_dataset(self, data_source, mode: str = 'inference'):
        if isinstance(data_source, str):
            ds = DatasetPhotozBinned(file_path=data_source,
                                     feature_generation=self.config.dataset_params.get('feature_generation'),
                                     feature_columns=self.config.dataset_params.get('feature_columns'),
                                     binning_config=self.config.dataset_params.get('binning_config'),
                                     scaler_X=self.scaler, mode=mode)
        else:
            ds = DatasetPhotozBinned(dataset=data_source,
                                     feature_generation=self.config.dataset_params.get('feature_generation'),
                                     feature_columns=self.config.dataset_params.get('feature_columns'),
                                     binning_config=self.config.dataset_params.get('binning_config'),
                                     scaler_X=self.scaler, mode=mode)
        return ds

    @torch.no_grad()
    def predict(self, data_source, mode: str = 'expectation', num_samples: int = 1,
                batch_size: int = None, seed: int = None, apply_calibration: bool = None):
        """
        Predict redshift for given data.

        Args:
            data_source: Input data (file path or DataFrame)
            mode: 'expectation' or 'sample'
            num_samples: Number of samples to generate when mode='sample'
            batch_size: Batch size for inference
            seed: Random seed for sampling
            apply_calibration: Whether to apply temperature scaling. If None, uses self.use_calibration

        Returns:
            Dictionary with predictions
        """
        if seed is not None:
            np.random.seed(seed)

        # apply_calibration: None=auto (use if calibrator exists), True=force, False=skip
        if apply_calibration is None:
            apply_calibration = self.calibrator is not None

        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        all_expect, all_sampled = [], []
        centers = ds.bin_centers

        for batch in dl:
            inputs = batch['input'].to(self.device)
            logits = self.model(inputs).cpu().numpy()

            # Apply temperature scaling if calibrator is available
            if apply_calibration and self.calibrator is not None:
                probs = self.calibrator.calibrate_probs(logits)
            else:
                e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = e_x / e_x.sum(axis=1, keepdims=True)

            expect = expected_redshift_from_probs(probs, centers)
            all_expect.append(expect)

            if mode == 'sample':
                # Multinomial sampling within bin (uniform inside bin)
                batch_samples = []
                for _ in range(num_samples):
                    sampled_list = []
                    for p in probs:
                        idx = np.random.choice(len(centers), p=p)
                        left, right = ds.bin_edges[idx], ds.bin_edges[idx + 1]
                        sampled_list.append(np.random.uniform(left, right))
                    batch_samples.append(np.array(sampled_list))
                all_sampled.append(np.stack(batch_samples, axis=1))

        expect_arr = np.concatenate(all_expect, axis=0)
        if mode == 'sample':
            sample_arr = np.concatenate(all_sampled, axis=0)
            return {'expectation': expect_arr, 'samples': sample_arr}
        return {'expectation': expect_arr}

    @torch.no_grad()
    def predict_probabilities(self, data_source, batch_size: int = None, apply_calibration: bool = None):
        """
        Predict probability distribution over all bins for each sample.

        Args:
            data_source: Input data (file path or DataFrame)
            batch_size: Batch size for inference
            apply_calibration: Whether to apply temperature scaling. If None, uses self.use_calibration

        Returns:
            Dictionary containing:
                - 'probabilities': Array of shape (n_samples, n_bins) with probability for each bin
                - 'bin_centers': Array of bin centers
                - 'bin_edges': Array of bin edges
                - 'expectation': Expected redshift for each sample
                - 'calibrated': Whether temperature scaling was applied
                - 'temperature': Temperature value used (1.0 if not calibrated)
        """
        # apply_calibration: None=auto (use if calibrator exists), True=force, False=skip
        if apply_calibration is None:
            apply_calibration = self.calibrator is not None

        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        all_probs = []
        centers = ds.bin_centers

        for batch in dl:
            inputs = batch['input'].to(self.device)
            logits = self.model(inputs).cpu().numpy()

            # Apply temperature scaling if calibrator is available
            if apply_calibration and self.calibrator is not None:
                probs = self.calibrator.calibrate_probs(logits)
            else:
                e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = e_x / e_x.sum(axis=1, keepdims=True)

            all_probs.append(probs)

        prob_arr = np.concatenate(all_probs, axis=0)
        expect_arr = expected_redshift_from_probs(prob_arr, centers)

        temperature = self.calibrator.temperature if (apply_calibration and self.calibrator) else 1.0

        return {
            'probabilities': prob_arr,
            'bin_centers': centers,
            'bin_edges': ds.bin_edges,
            'expectation': expect_arr,
            'calibrated': apply_calibration and self.calibrator is not None,
            'temperature': temperature
        }

    @torch.no_grad()
    def predict_probabilities_chunked(self, data_source, save_path: str,
                                      chunk_size: int = 100000, batch_size: int = None,
                                      compression: str = 'gzip', apply_calibration: bool = None):
        """
        Predict probability distribution for large datasets using chunked processing.
        Saves results to HDF5 format for memory efficiency.

        This method is memory-efficient for large datasets (e.g., 7M samples × 240 bins).

        Args:
            data_source: Input data (file path or DataFrame)
            save_path: Path to save results (must end with .h5)
            chunk_size: Number of samples to process at once
            batch_size: Batch size for inference
            compression: Compression method ('gzip', 'lzf', None)
            apply_calibration: Whether to apply temperature scaling. If None, uses self.use_calibration

        Returns:
            Dictionary with save metadata
        """
        import h5py

        # apply_calibration: None=auto (use if calibrator exists), True=force, False=skip
        if apply_calibration is None:
            apply_calibration = self.calibrator is not None

        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size

        total_samples = len(ds)
        num_bins = self.config.dataset_params['binning_config']['num_bins']
        centers = ds.bin_centers

        temperature = self.calibrator.temperature if (apply_calibration and self.calibrator) else 1.0

        print(f"Processing {total_samples:,} samples with {num_bins} bins...")
        print(f"Temperature scaling: {'T=' + str(temperature) if apply_calibration else 'disabled'}")
        print(f"Estimated uncompressed size: {total_samples * num_bins * 4 / 1e9:.2f} GB")

        # Check if save_path ends with .h5
        if not save_path.endswith('.h5'):
            raise ValueError(f"save_path must end with '.h5', got: {save_path}")

        save_file = f"{save_path}"
        with h5py.File(save_file, 'w') as f:
            # Create datasets with compression
            prob_dataset = f.create_dataset('probabilities',
                                            shape=(total_samples, num_bins),
                                            dtype=np.float32,
                                            compression=compression)
            expect_dataset = f.create_dataset('expectation',
                                              shape=(total_samples,),
                                              dtype=np.float32,
                                              compression=compression)

            # Save metadata
            f.create_dataset('bin_centers', data=centers, compression=compression)
            f.create_dataset('bin_edges', data=ds.bin_edges, compression=compression)
            f.attrs['num_samples'] = total_samples
            f.attrs['num_bins'] = num_bins
            f.attrs['z_min'] = self.config.dataset_params['binning_config']['z_min']
            f.attrs['z_max'] = self.config.dataset_params['binning_config']['z_max']
            f.attrs['chunk_size'] = chunk_size
            f.attrs['compression'] = compression
            f.attrs['calibrated'] = apply_calibration and self.calibrator is not None
            f.attrs['temperature'] = temperature

            # Process in chunks
            processed = 0
            for chunk_start in range(0, total_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_samples)
                chunk_indices = range(chunk_start, chunk_end)

                # Create subset dataset and dataloader
                chunk_ds = Subset(ds, chunk_indices)
                chunk_dl = DataLoader(chunk_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=self.num_workers)

                # Process chunk
                chunk_probs = []
                for batch in chunk_dl:
                    inputs = batch['input'].to(self.device)
                    logits = self.model(inputs).cpu().numpy()

                    # Apply temperature scaling if calibrator is available
                    if apply_calibration and self.calibrator is not None:
                        probs = self.calibrator.calibrate_probs(logits)
                    else:
                        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        probs = e_x / e_x.sum(axis=1, keepdims=True)

                    chunk_probs.append(probs)
                
                chunk_prob_arr = np.concatenate(chunk_probs, axis=0)
                chunk_expect_arr = expected_redshift_from_probs(chunk_prob_arr, centers)
                
                # Save chunk to file
                prob_dataset[chunk_start:chunk_end] = chunk_prob_arr
                expect_dataset[chunk_start:chunk_end] = chunk_expect_arr
                
                processed += len(chunk_indices)
                print(f"Processed {processed:,}/{total_samples:,} samples")
                
                # Clear memory
                del chunk_probs, chunk_prob_arr, chunk_expect_arr
                torch.cuda.empty_cache()

        print(f"Results saved to {save_file}")

    @torch.no_grad()
    def calibrate(self, data_source, method: str = 'minimize_ks',
                  temperature: float = None, batch_size: int = None,
                  save_path: str = None, plot: bool = True) -> Dict[str, Any]:
        """
        Perform temperature scaling calibration on a dataset with known redshifts.

        This method can be used to:
        1. Auto-fit optimal temperature on a validation/calibration set
        2. Apply a fixed temperature and evaluate its effect
        3. Re-calibrate an existing model on new data

        Args:
            data_source: Input data with true redshifts (file path or DataFrame)
            method: Optimization method ('minimize_ks' or 'minimize_crps')
            temperature: Fixed temperature (None = auto-fit)
            batch_size: Batch size for inference
            save_path: Path to save calibrator (None = don't save)
            plot: Whether to display/save PIT comparison plot

        Returns:
            Dictionary with calibration results
        """
        # Create dataset (needs true redshifts)
        ds = self._create_dataset(data_source, mode='validation')
        if batch_size is None:
            batch_size = self.batch_size

        # Check if dataset has true redshifts
        if not hasattr(ds, 'labels_cont') or ds.labels_cont is None:
            raise ValueError("Dataset must have true redshifts for calibration. "
                           "Make sure the data contains a 'Z' or redshift column.")

        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True)

        # Collect logits and true redshifts
        all_logits, all_true_z = [], []
        for batch in dl:
            inputs = batch['input'].to(self.device)
            logits = self.model(inputs).cpu().numpy()
            true_z = batch['label_z'].numpy().flatten()
            all_logits.append(logits)
            all_true_z.append(true_z)

        logits_arr = np.concatenate(all_logits, axis=0)
        true_z_arr = np.concatenate(all_true_z, axis=0)
        bin_edges = ds.bin_edges

        # Create or update calibrator
        if temperature is not None:
            # Use fixed temperature
            self.calibrator = TemperatureCalibrator()
            self.calibrator.temperature = float(temperature)
            print(f"Using fixed temperature: T={temperature:.4f}")
        else:
            # Auto-fit temperature
            self.calibrator = TemperatureCalibrator()
            self.calibrator.fit(logits_arr, true_z_arr, bin_edges, method=method)

        # Compute PIT before and after calibration
        e_x = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
        probs_before = e_x / e_x.sum(axis=1, keepdims=True)
        pit_before = compute_pit_values(probs_before, true_z_arr, bin_edges)
        diagnosis_before = diagnose_pit(pit_before)

        probs_after = self.calibrator.calibrate_probs(logits_arr)
        pit_after = compute_pit_values(probs_after, true_z_arr, bin_edges)
        diagnosis_after = diagnose_pit(pit_after)

        print(f"Before calibration: KS={diagnosis_before['ks_stat']:.4f} | {diagnosis_before['suggestion']}")
        print(f"After calibration (T={self.calibrator.temperature:.4f}): KS={diagnosis_after['ks_stat']:.4f} | {diagnosis_after['suggestion']}")

        # Plot comparison
        if plot:
            _, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(pit_before, bins=20, range=(0, 1), density=True, alpha=0.7, edgecolor='black')
            axes[0].axhline(y=1, color='r', linestyle='--', linewidth=2)
            axes[0].set_title(f'Before (T=1.0, KS={diagnosis_before["ks_stat"]:.4f})')
            axes[0].set_xlabel('PIT value')
            axes[0].set_ylabel('Density')

            axes[1].hist(pit_after, bins=20, range=(0, 1), density=True, alpha=0.7, edgecolor='black')
            axes[1].axhline(y=1, color='r', linestyle='--', linewidth=2)
            axes[1].set_title(f'After (T={self.calibrator.temperature:.4f}, KS={diagnosis_after["ks_stat"]:.4f})')
            axes[1].set_xlabel('PIT value')
            axes[1].set_ylabel('Density')

            plt.suptitle('Temperature Scaling Calibration')
            plt.tight_layout()

            if save_path:
                plot_path = str(Path(save_path).with_suffix('.png'))
                plt.savefig(plot_path, dpi=150)
                print(f"Plot saved to {plot_path}")
            plt.show()

        # Save calibrator
        if save_path:
            calibrator_path = str(Path(save_path).with_suffix('.pkl'))
            self.calibrator.save(calibrator_path)
            print(f"Calibrator saved to {calibrator_path}")

        return {
            'temperature': self.calibrator.temperature,
            'ks_before': diagnosis_before['ks_stat'],
            'ks_after': diagnosis_after['ks_stat'],
            'diagnosis_before': diagnosis_before,
            'diagnosis_after': diagnosis_after
        }

    def load_calibrator(self, path: str) -> 'PhotozBinInference':
        """
        Load a previously saved temperature calibrator.

        Args:
            path: Path to the saved calibrator (.pkl file)

        Returns:
            self (for method chaining)
        """
        self.calibrator = TemperatureCalibrator.load(path)
        print(f"Calibrator loaded: T={self.calibrator.temperature:.4f}")
        return self


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="Train a binned classification model for photo-z.")
    parser.add_argument('--config', type=str, default='photoz_bin/config.yaml', help="Path to the config.yaml file.")
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



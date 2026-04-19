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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

# External utils (same as original)
import cosmic.utils as cu


# ============================
# Config
# ============================
@dataclass
class Config:
    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "photoz_regression"
    random_seed: int = 2025
    gpu_id: str = '0'

    # Model
    model_params: Dict[str, Any] = field(default_factory=dict)
    model_class_name: str = 'PhotozRegressor'

    # Data
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    dataset_class_name: str = 'DatasetPhotozRegression'
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
        'type': 'mse',  # 'mse', 'mae', 'huber', 'smooth_l1'
        'huber_delta': 1.0,  # For Huber loss
        'smooth_l1_beta': 1.0  # For Smooth L1 loss
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
# Dataset for regression
# ============================
class DatasetPhotozRegression(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_path: str = None,
                 feature_generation: dict = None,
                 feature_columns: list = None,
                 label_noise: dict = None,
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

        # mode
        self.mode = mode
        if self.mode in ['train', 'validation', 'test']:
            self.labels = self.dataset['z'].values.astype(np.float32)

            # Add noise to labels if configured
            self.label_noise = label_noise
            if label_noise is not None and mode == 'train':  # Only add noise during training
                noise_type = label_noise.get('type', 'gaussian')
                noise_level = label_noise.get('level', 0.0)

                if noise_level > 0:
                    if noise_type == 'gaussian':
                        # Gaussian noise with std = noise_level
                        noise = np.random.normal(0, noise_level, size=self.labels.shape)
                        self.labels = self.labels + noise.astype(np.float32)
                    elif noise_type == 'uniform':
                        # Uniform noise in [-noise_level, noise_level]
                        noise = np.random.uniform(-noise_level, noise_level, size=self.labels.shape)
                        self.labels = self.labels + noise.astype(np.float32)
                    elif noise_type == 'relative_gaussian':
                        # Gaussian noise proportional to redshift value
                        # noise_std = noise_level * (1 + z)
                        noise_std = noise_level * (1 + self.labels)
                        noise = np.random.normal(0, 1, size=self.labels.shape) * noise_std
                        self.labels = self.labels + noise.astype(np.float32)
                    elif noise_type == 'relative_uniform':
                        # Uniform noise proportional to redshift value
                        noise_range = noise_level * (1 + self.labels)
                        noise = np.random.uniform(-1, 1, size=self.labels.shape) * noise_range
                        self.labels = self.labels + noise.astype(np.float32)
                    else:
                        raise ValueError(f"Unknown noise type: {noise_type}")

                    # Ensure labels remain positive
                    self.labels = np.maximum(self.labels, 0.0)

                    logging.info(f"Added {noise_type} noise with level {noise_level} to training labels")

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
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
    'elu': nn.ELU,
    'silu': nn.SiLU,
}


class PhotozRegressor(nn.Module):
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
                self.residual_layers.append(nn.Linear(in_dim, h) if in_dim != h else nn.Identity())
            else:
                self.residual_layers.append(None)
            in_dim = h

        # Output layer for single redshift value
        self.output = nn.Linear(in_dim, 1)
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
        return self.output(x)


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


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, running_loss: float, num_batches: int) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate additional photoz-specific metrics
    residuals = y_pred - y_true
    nmad = 1.48 * np.median(np.abs(residuals))  # Normalized MAD
    bias = np.mean(residuals)

    # Outlier fraction (|Δz| > 0.15)
    outlier_threshold = 0.15
    outlier_frac = np.mean(np.abs(residuals) > outlier_threshold)

    return {
        'loss': running_loss / max(1, num_batches),
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'nmad': nmad,
        'bias': bias,
        'outlier_frac': outlier_frac
    }


# ============================
# Loss Functions
# ============================
def get_loss_function(config: Dict[str, Any]) -> callable:
    """Get the loss function based on configuration."""
    loss_type = config.get('type', 'mse')

    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        delta = config.get('huber_delta', 1.0)
        return nn.HuberLoss(delta=delta)
    elif loss_type == 'smooth_l1':
        beta = config.get('smooth_l1_beta', 1.0)
        return nn.SmoothL1Loss(beta=beta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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

        self.train_metrics = {'loss': [], 'mse': [], 'mae': [], 'r2': [], 'nmad': [], 'bias': [], 'outlier_frac': []}
        self.val_metrics = {'loss': [], 'mse': [], 'mae': [], 'r2': [], 'nmad': [], 'bias': [], 'outlier_frac': []}

        self.loss_function = None

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
        tmp_train = DatasetPhotozRegression(file_path=paths['train'],
                                           feature_generation=self.config.dataset_params.get('feature_generation'),
                                           feature_columns=self.config.dataset_params.get('feature_columns'),
                                           label_noise=self.config.dataset_params.get('label_noise'),
                                           scaler_X=None,
                                           mode='train')
        scaler_X = StandardScaler()
        scaler_X.fit(tmp_train.features)

        train_ds = DatasetPhotozRegression(file_path=paths['train'],
                                          feature_generation=self.config.dataset_params.get('feature_generation'),
                                          feature_columns=self.config.dataset_params.get('feature_columns'),
                                          label_noise=self.config.dataset_params.get('label_noise'),
                                          scaler_X=scaler_X,
                                          mode='train')
        val_ds = DatasetPhotozRegression(file_path=paths['val'],
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        label_noise=None,  # No noise for validation
                                        scaler_X=scaler_X,
                                        mode='validation')
        test_ds = DatasetPhotozRegression(file_path=paths['test'],
                                         feature_generation=self.config.dataset_params.get('feature_generation'),
                                         feature_columns=self.config.dataset_params.get('feature_columns'),
                                         label_noise=None,  # No noise for test
                                         scaler_X=scaler_X,
                                         mode='test')

        self.scaler_X = scaler_X
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

            temp_ds = DatasetPhotozRegression(dataset=dataset,
                                             feature_generation=self.config.dataset_params.get('feature_generation'),
                                             feature_columns=self.config.dataset_params.get('feature_columns'),
                                             label_noise=None,  # No noise for scaler fitting
                                             scaler_X=None, mode='inference')
            scaler_X = StandardScaler()
            scaler_X.fit(temp_ds.features[np.array(train_idx)])

            train_ds = DatasetPhotozRegression(dataset=dataset,
                                              feature_generation=self.config.dataset_params.get('feature_generation'),
                                              feature_columns=self.config.dataset_params.get('feature_columns'),
                                              label_noise=self.config.dataset_params.get('label_noise'),
                                              scaler_X=scaler_X, mode='train')
            val_ds = DatasetPhotozRegression(dataset=dataset,
                                            feature_generation=self.config.dataset_params.get('feature_generation'),
                                            feature_columns=self.config.dataset_params.get('feature_columns'),
                                            label_noise=None,  # No noise for validation
                                            scaler_X=scaler_X,
                                            mode='validation')
            test_ds = DatasetPhotozRegression(dataset=dataset,
                                             feature_generation=self.config.dataset_params.get('feature_generation'),
                                             feature_columns=self.config.dataset_params.get('feature_columns'),
                                             label_noise=None,  # No noise for test
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
        """Setup loss function."""
        self.loss_function = get_loss_function(self.config.loss_config)
        logging.info(f"Using loss function: {self.config.loss_config.get('type', 'mse')}")

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

            pred_z = outputs.detach().cpu().numpy().flatten()
            true_z = targets.cpu().numpy().flatten()
            all_labels.extend(true_z)
            all_preds.extend(pred_z)

        return calculate_regression_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(self.trainloader))

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for batch in loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            running_loss += loss.item()

            pred_z = outputs.cpu().numpy().flatten()
            true_z = targets.cpu().numpy().flatten()
            all_labels.extend(true_z)
            all_preds.extend(pred_z)

        return calculate_regression_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(loader))

    def _update_and_log(self, epoch, train_metrics, val_metrics, duration):
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])

        logging.info(f"Epoch {epoch+1}/{self.config.epochs} | Duration: {duration:.2f}m")
        logging.info(f"\tTrain Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}, NMAD: {train_metrics['nmad']:.4f}")
        logging.info(f"\tVal   Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f}, NMAD: {val_metrics['nmad']:.4f}")

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
        logging.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _plot_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics_map = {'loss': 'Loss', 'mse': 'MSE', 'mae': 'MAE', 'r2': 'R2', 'nmad': 'NMAD', 'outlier_frac': 'Outlier Fraction'}
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
        logging.info("--- Starting Testing ---")
        best_model_path = self.logdir / 'best_model.pkl'
        if not best_model_path.exists():
            logging.warning("No best model found. Testing with the last model.")
            best_model_path = self.logdir / 'checkpoint.pkl'
        self.model.load_state_dict(torch.load(best_model_path))
        metrics = self._evaluate(self.testloader)
        logging.info("Final Test Set Results:")
        logging.info(f"\tTest Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        logging.info(f"\tNMAD: {metrics['nmad']:.4f}, Bias: {metrics['bias']:.4f}, Outlier Frac: {metrics['outlier_frac']:.4f}")
        with open(self.logdir / 'test_results.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"Test {k}: {v:.6f}\n")


# ============================
# Inference helper
# ============================
class PhotozRegressionInference:
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
        if isinstance(data_source, str):
            ds = DatasetPhotozRegression(file_path=data_source,
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        label_noise=None,  # No noise during inference
                                        scaler_X=self.scaler, mode='inference')
        else:
            ds = DatasetPhotozRegression(dataset=data_source,
                                        feature_generation=self.config.dataset_params.get('feature_generation'),
                                        feature_columns=self.config.dataset_params.get('feature_columns'),
                                        label_noise=None,  # No noise during inference
                                        scaler_X=self.scaler, mode='inference')
        return ds

    @torch.no_grad()
    def predict(self, data_source, batch_size: int = None, return_uncertainty: bool = False):
        """
        Predict redshift for given data.

        Args:
            data_source: Input data (file path or DataFrame)
            batch_size: Batch size for inference
            return_uncertainty: If True, use dropout for uncertainty estimation

        Returns:
            Dictionary with predictions
        """
        ds = self._create_dataset(data_source)
        if batch_size is None:
            batch_size = self.batch_size
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        all_preds = []
        all_uncertainties = []

        if return_uncertainty:
            # Monte Carlo dropout for uncertainty estimation
            n_forward = 100
            self.model.train()  # Enable dropout

            for batch in dl:
                inputs = batch['input'].to(self.device)
                batch_preds = []

                for _ in range(n_forward):
                    outputs = self.model(inputs).cpu().numpy().flatten()
                    batch_preds.append(outputs)

                batch_preds = np.array(batch_preds)  # [n_forward, batch_size]
                mean_pred = batch_preds.mean(axis=0)
                std_pred = batch_preds.std(axis=0)

                all_preds.append(mean_pred)
                all_uncertainties.append(std_pred)

            self.model.eval()  # Disable dropout
            pred_arr = np.concatenate(all_preds, axis=0)
            uncertainty_arr = np.concatenate(all_uncertainties, axis=0)
            return {'predictions': pred_arr, 'uncertainties': uncertainty_arr}
        else:
            for batch in dl:
                inputs = batch['input'].to(self.device)
                outputs = self.model(inputs).cpu().numpy().flatten()
                all_preds.append(outputs)

            pred_arr = np.concatenate(all_preds, axis=0)
            return {'predictions': pred_arr}


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="Train a regression model for photo-z.")
    parser.add_argument('--config', type=str, default='photoz_ann/config_regression.yaml', help="Path to the config.yaml file.")
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
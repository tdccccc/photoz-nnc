#============================================================
# Import
#============================================================
import os
import time
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Iterable, Callable, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import cosmic.utils as cu
import joblib

#============================================================
# Config
#============================================================
@dataclass
class Config:
    # Experiment setup
    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "base_experiment"
    random_seed: int = 2025
    gpu_id: str = '0'
    
    # Model configuration
    model_class_name: str = 'DeepPhotozNet'
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    dataset_class_name: str = 'Dataset_Photz'
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    dataset_mode: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'ratio',
        'ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'paths': {'train': None, 'val': None, 'test': None}
    })
    
    # Label transformation configuration
    label_transform: str = 'none'  # 'none', 'log1p', 'log', 'sqrt', 'power', 'asinh'
    label_transform_params: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 256
    epochs: int = 100
    early_stopping: int = 30
    loss_type: str = "huber"
    train_augmentation: bool = True
    
    # Gradient clipping configuration
    gradient_clipping: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'norm',  # 'norm' or 'value'
        'max_norm': 1.0,   # for norm clipping
        'clip_value': 1.0  # for value clipping
    })

    def update_from_yaml(self, path: str):
        """Loads and updates configuration from a YAML file."""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        self.update_from_dict(yaml_config)

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Updates configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)



#============================================================
# Dataset
#============================================================
class Dataset_Photz(Dataset):
    
    def __init__(self, 
                 dataset: pd.DataFrame = None,
                 file_path: str = None, 
                 mag_types: list = None,
                 color_types: list = None,
                 use_mag: bool = True,
                 use_magerr: bool = True,
                 use_colorerr: bool = True,
                 scaler_X = None, 
                 mode: str = 'train',
                 label_noise: float = 0.0,
                 label_transform: str = 'none',
                 label_transform_params: dict = None):

        self.mode = mode
        self.file_path = file_path
        self.mag_types = mag_types
        self.color_types = color_types
        self.scaler_X = scaler_X
        self.use_mag = use_mag
        self.use_magerr = use_magerr
        self.use_colorerr = use_colorerr
        self.label_noise = label_noise
        self.label_transform = label_transform
        self.label_transform_params = label_transform_params or {}
        
        # Check that exactly one of dataset or file_path is provided
        if dataset is not None and file_path is not None:
            raise ValueError("Cannot specify both 'dataset' and 'file_path'. Please provide only one.")
        if dataset is None and file_path is None:
            raise ValueError("Must specify either 'dataset' or 'file_path'.")
            
        # Load data from file or use provided dataset
        if file_path is not None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            self.dataset = cu.readfile(file_path)
        else:
            self.dataset = dataset
    
        self.feature_cols = self._choose_features(self.mag_types, self.color_types, use_mag=self.use_mag, 
                                                  use_magerr=self.use_magerr, use_colorerr=self.use_colorerr)
        
        # Extract features from dataset
        self.features = self.dataset[self.feature_cols].values.astype(np.float32)

        # Extract labels if needed
        if self.mode in ['train', 'validation', 'test']:
            self.label = self.dataset['z'].values.astype(np.float32)
            if self.label_noise > 0:
                self.label = self.label + np.random.normal(0, self.label_noise, len(self.label))
            # Apply label transformation
            self.label = self._transform_labels(self.label)
        
        # Apply scaling if scaler is provided
        if scaler_X is not None:
            self.features = self.scaler_X.transform(self.features)
    
    def _choose_features(self, 
                         mag_types: list,
                         color_types: list,
                         use_mag: bool = True,
                         use_magerr: bool = True, 
                         use_colorerr: bool = True):
        bands = ['g', 'r', 'i', 'z', 'y']
        colours = ['gr', 'ri', 'iz', 'zy']
        mag_cols = []
        err_cols = []
        
        # Add color features using color_types
        if color_types:
            for color_type in color_types:
                for colour in colours:
                    mag_cols.extend([f'{color_type}_{colour}'])
                    if use_colorerr:
                        err_cols.extend([f'{color_type}_{colour}_Err'])
        
        # Add magnitude features using mag_types
        if use_mag and mag_types:
            for band in bands:
                for mag_type in mag_types:
                    mag_cols.extend([f'{band}{mag_type}Mag_dered'])
                    if use_magerr:
                        err_cols.extend([f'{band}{mag_type}MagErr'])
        
        combined_cols = mag_cols + err_cols
        return combined_cols
    
    def _transform_labels(self, labels: np.ndarray) -> np.ndarray:
        """Apply transformation to labels to ensure positive outputs."""
        if self.label_transform == 'none':
            return labels
        elif self.label_transform == 'log1p':
            # log(1 + x) transformation - good for values starting from 0
            return np.log1p(labels)
        elif self.label_transform == 'log':
            # log(x + offset) transformation
            offset = self.label_transform_params.get('offset', 1e-8)
            return np.log(labels + offset)
        elif self.label_transform == 'sqrt':
            # sqrt(x) transformation - preserves order, compresses large values
            return np.sqrt(np.maximum(labels, 0))  # Ensure non-negative input
        elif self.label_transform == 'power':
            # x^power transformation
            power = self.label_transform_params.get('power', 0.5)
            return np.power(np.maximum(labels, 0), power)
        elif self.label_transform == 'asinh':
            # inverse hyperbolic sine - handles negative values well
            scale = self.label_transform_params.get('scale', 1.0)
            return np.arcsinh(labels / scale)
        else:
            raise ValueError(f"Unsupported label transform: {self.label_transform}")
    
    @staticmethod
    def inverse_transform_labels(transformed_labels: np.ndarray, 
                                transform_type: str, 
                                transform_params: dict = None) -> np.ndarray:
        """Inverse transformation to get original label scale."""
        if transform_params is None:
            transform_params = {}
            
        if transform_type == 'none':
            return transformed_labels
        elif transform_type == 'log1p':
            return np.expm1(transformed_labels)  # exp(x) - 1
        elif transform_type == 'log':
            offset = transform_params.get('offset', 1e-8)
            return np.exp(transformed_labels) - offset
        elif transform_type == 'sqrt':
            return np.square(transformed_labels)
        elif transform_type == 'power':
            power = transform_params.get('power', 0.5)
            return np.power(transformed_labels, 1.0 / power)
        elif transform_type == 'asinh':
            scale = transform_params.get('scale', 1.0)
            return np.sinh(transformed_labels) * scale
        else:
            raise ValueError(f"Unsupported label transform: {transform_type}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        if self.mode in ['train', 'validation', 'test']:
            label = torch.FloatTensor([self.label[idx]])
            return {'input': features, 'label': label}
        elif self.mode == 'inference':
            return {'input': features}


#============================================================
# Model
#============================================================
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
    'elu': nn.ELU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'softplus': nn.Softplus,
}

class DeepPhotozNet(nn.Module):
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list = None,
                 dropout_rate: float = 0.0,
                 activation_function: str = 'relu',
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_residual: bool = True,
                 output_activation: str = 'relu'):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate (0.0 to disable)
            activation_function: Activation function name
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            output_activation: Output layer activation function ('relu', 'softplus', 'none')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        
        # 获取激活函数
        if activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation function: {activation_function}. "
                           f"Available: {list(ACTIVATION_FUNCTIONS.keys())}")
        
        activation_fn = ACTIVATION_FUNCTIONS[activation_function]
        
        # 检查归一化方法冲突
        if use_batch_norm and use_layer_norm:
            logging.warning("Both BatchNorm and LayerNorm are enabled. Using BatchNorm only.")
            use_layer_norm = False
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 创建主要的特征层
            layer_components = []
            
            # 线性层
            layer_components.append(nn.Linear(in_dim, hidden_dim))
            
            # 归一化层
            if use_batch_norm:
                layer_components.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layer_components.append(nn.LayerNorm(hidden_dim))
            
            # 激活函数
            layer_components.append(activation_fn())
            
            # Dropout
            if dropout_rate > 0:
                layer_components.append(nn.Dropout(dropout_rate))
            
            self.layers.append(nn.Sequential(*layer_components))
            
            # 残差连接层
            if use_residual:
                if in_dim == hidden_dim:
                    self.residual_layers.append(nn.Identity())
                else:
                    self.residual_layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.residual_layers.append(None)
            
            in_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(in_dim, 1)
        
        # 输出激活函数
        if output_activation == 'none':
            self.output_act = nn.Identity()
        elif output_activation in ACTIVATION_FUNCTIONS:
            self.output_act = ACTIVATION_FUNCTIONS[output_activation]()
        else:
            available_functions = list(ACTIVATION_FUNCTIONS.keys()) + ['none']
            raise ValueError(f"Unsupported output activation function: {output_activation}. "
                           f"Available: {available_functions}")
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for layer, residual_layer in zip(self.layers, self.residual_layers):
            identity = x
            
            # 主路径
            out = layer(x)
            
            # 残差连接
            if self.use_residual and residual_layer is not None:
                if isinstance(residual_layer, nn.Identity):
                    out = out + identity
                else:
                    out = out + residual_layer(identity)
            
            x = out
        
        # 输出层
        return self.output_act(self.output_layer(x))
    
    
class DeepPhotozLSTM(nn.Module):
    
    def __init__(self, 
                 input_dim: int, 
                 lstm_hidden_size: int = 32,
                 num_lstm_layers: int = 2,
                 dropout_rate: float = 0.2,
                 fc_hidden_dims: list = None,
                 use_batch_norm: bool = True,
                 output_activation: str = 'relu'):
        """
        LSTM-based photometric redshift model following the architecture in the figure.
        
        Args:
            input_dim: Input feature dimension
            lstm_hidden_size: Hidden size for LSTM layers (default: 32)
            num_lstm_layers: Number of LSTM layers (default: 2)
            dropout_rate: Dropout rate between layers
            fc_hidden_dims: List of fully connected layer dimensions (if None, uses [64])
            use_batch_norm: Whether to use batch normalization
            output_activation: Output layer activation function ('relu', 'softplus', 'none')
        """
        super().__init__()
        
        if fc_hidden_dims is None:
            fc_hidden_dims = [64]
        
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.output_activation = output_activation
        
        # LSTM layers (bidirectional)
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, 
                   batch_first=True, bidirectional=True)
        )
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        if use_batch_norm:
            self.batch_norm_layers.append(nn.BatchNorm1d(lstm_hidden_size * 2))
        
        # Additional LSTM layers
        for i in range(1, num_lstm_layers):
            self.lstm_layers.append(
                nn.LSTM(input_size=lstm_hidden_size * 2, hidden_size=lstm_hidden_size,
                       batch_first=True, bidirectional=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(lstm_hidden_size * 2))
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = lstm_hidden_size * 2  # bidirectional output
        
        for hidden_dim in fc_hidden_dims:
            self.fc_layers.append(nn.Linear(in_features, hidden_dim))
            if use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
            self.fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(in_features, 1)
        
        # Output activation function
        if output_activation == 'none':
            self.output_act = nn.Identity()
        elif output_activation in ACTIVATION_FUNCTIONS:
            self.output_act = ACTIVATION_FUNCTIONS[output_activation]()
        else:
            available_functions = list(ACTIVATION_FUNCTIONS.keys()) + ['none']
            raise ValueError(f"Unsupported output activation function: {output_activation}. "
                           f"Available: {available_functions}")
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        
        # Reshape input for LSTM: (batch, seq_len, input_size)
        # Each feature is treated as a time step
        x = x.unsqueeze(-1)  # (batch, features, 1)
        
        # Process through LSTM layers
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm_layer(x)  # (batch, seq_len, hidden_size * 2)
            
            # Apply dropout
            lstm_out = dropout_layer(lstm_out)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and i < len(self.batch_norm_layers):
                # Reshape for batch norm: (batch * seq_len, features)
                seq_len = lstm_out.size(1)
                lstm_out = lstm_out.contiguous().view(-1, lstm_out.size(-1))
                lstm_out = self.batch_norm_layers[i](lstm_out)
                lstm_out = lstm_out.view(batch_size, seq_len, -1)
            
            x = lstm_out
        
        # Use the last time step output (or mean pooling)
        # Here we use mean pooling across the sequence dimension
        x = torch.mean(x, dim=1)  # (batch, hidden_size * 2)
        
        # Process through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return self.output_act(x)


class DeepPhotozLSTMWithUncertainty(nn.Module):
    
    def __init__(self, 
                 input_dim: int, 
                 lstm_hidden_size: int = 32,
                 num_lstm_layers: int = 2,
                 dropout_rate: float = 0.2,
                 fc_hidden_dims: list = None,
                 use_batch_norm: bool = True,
                 output_activation: str = 'relu'):
        """
        LSTM-based photometric redshift model with uncertainty estimation.
        
        Args:
            input_dim: Input feature dimension
            lstm_hidden_size: Hidden size for LSTM layers (default: 32)
            num_lstm_layers: Number of LSTM layers (default: 2)
            dropout_rate: Dropout rate between layers
            fc_hidden_dims: List of fully connected layer dimensions (if None, uses [64])
            use_batch_norm: Whether to use batch normalization
            output_activation: Output layer activation function for mu ('relu', 'softplus', 'none')
        """
        super().__init__()
        
        if fc_hidden_dims is None:
            fc_hidden_dims = [64]
        
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # LSTM layers (bidirectional)
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, 
                   batch_first=True, bidirectional=True)
        )
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        if use_batch_norm:
            self.batch_norm_layers.append(nn.BatchNorm1d(lstm_hidden_size * 2))
        
        # Additional LSTM layers
        for i in range(1, num_lstm_layers):
            self.lstm_layers.append(
                nn.LSTM(input_size=lstm_hidden_size * 2, hidden_size=lstm_hidden_size,
                       batch_first=True, bidirectional=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(lstm_hidden_size * 2))
        
        # Shared backbone fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = lstm_hidden_size * 2  # bidirectional output
        
        for hidden_dim in fc_hidden_dims:
            self.fc_layers.append(nn.Linear(in_features, hidden_dim))
            if use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
            self.fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        
        # Output heads
        self.mu_head = nn.Linear(in_features, 1)
        self.sigma_head = nn.Linear(in_features, 1)
        
        # Activation for sigma (ensure positive values)
        self.softplus = nn.Softplus()
        
        # Activation for mu
        if output_activation == 'none':
            self.mu_act = nn.Identity()
        elif output_activation in ACTIVATION_FUNCTIONS:
            self.mu_act = ACTIVATION_FUNCTIONS[output_activation]()
        else:
            available_functions = list(ACTIVATION_FUNCTIONS.keys()) + ['none']
            raise ValueError(f"Unsupported output activation function for mu: {output_activation}. "
                           f"Available: {available_functions}")
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """Forward pass."""
        batch_size = x.size(0)
        
        # Reshape input for LSTM: (batch, seq_len, input_size)
        # Each feature is treated as a time step
        x = x.unsqueeze(-1)  # (batch, features, 1)
        
        # Process through LSTM layers
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm_layer(x)  # (batch, seq_len, hidden_size * 2)
            
            # Apply dropout
            lstm_out = dropout_layer(lstm_out)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and i < len(self.batch_norm_layers):
                # Reshape for batch norm: (batch * seq_len, features)
                seq_len = lstm_out.size(1)
                lstm_out = lstm_out.contiguous().view(-1, lstm_out.size(-1))
                lstm_out = self.batch_norm_layers[i](lstm_out)
                lstm_out = lstm_out.view(batch_size, seq_len, -1)
            
            x = lstm_out
        
        # Use mean pooling across the sequence dimension
        x = torch.mean(x, dim=1)  # (batch, hidden_size * 2)
        
        # Process through shared fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Calculate mu and sigma
        mu = self.mu_act(self.mu_head(x))
        sigma = self.softplus(self.sigma_head(x)) + 1e-6
        
        return mu, sigma


class DeepPhotozNetWithUncertainty(nn.Module):
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list = None,
                 dropout_rate: float = 0.0,
                 activation_function: str = 'relu',
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_residual: bool = True,
                 output_activation: str = 'relu'):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate (0.0 to disable)
            activation_function: Activation function name
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            output_activation: Output layer activation function for mu ('relu', 'softplus', 'none')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
        # Get activation function
        if activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation function: {activation_function}. "
                           f"Available: {list(ACTIVATION_FUNCTIONS.keys())}")
        
        activation_fn = ACTIVATION_FUNCTIONS[activation_function]
        
        # Check for normalization conflict
        if use_batch_norm and use_layer_norm:
            logging.warning("Both BatchNorm and LayerNorm are enabled. Using BatchNorm only.")
            use_layer_norm = False
        
        # Backbone layers
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer_components = []
            layer_components.append(nn.Linear(in_dim, hidden_dim))
            
            if use_batch_norm:
                layer_components.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layer_components.append(nn.LayerNorm(hidden_dim))
            
            layer_components.append(activation_fn())
            
            if dropout_rate > 0:
                layer_components.append(nn.Dropout(dropout_rate))
            
            self.layers.append(nn.Sequential(*layer_components))
            
            if use_residual:
                if in_dim == hidden_dim:
                    self.residual_layers.append(nn.Identity())
                else:
                    self.residual_layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.residual_layers.append(None)
            
            in_dim = hidden_dim
        
        # Output heads
        self.mu_head = nn.Linear(in_dim, 1)
        self.sigma_head = nn.Linear(in_dim, 1)
        
        # Activation for sigma
        self.softplus = nn.Softplus()

        # Activation for mu
        if output_activation == 'none':
            self.mu_act = nn.Identity()
        elif output_activation in ACTIVATION_FUNCTIONS:
            self.mu_act = ACTIVATION_FUNCTIONS[output_activation]()
        else:
            available_functions = list(ACTIVATION_FUNCTIONS.keys()) + ['none']
            raise ValueError(f"Unsupported output activation function for mu: {output_activation}. "
                           f"Available: {available_functions}")
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """Forward pass."""
        for layer, residual_layer in zip(self.layers, self.residual_layers):
            identity = x
            
            out = layer(x)
            
            if self.use_residual and residual_layer is not None:
                if isinstance(residual_layer, nn.Identity):
                    out = out + identity
                else:
                    out = out + residual_layer(identity)
            
            x = out
        
        # Calculate mu and sigma
        mu = self.mu_act(self.mu_head(x))
        sigma = self.softplus(self.sigma_head(x)) + 1e-6
        
        return mu, sigma


#============================================================
# Optimizer and Scheduler
#============================================================
def get_optimizer(parameters: Iterable[torch.nn.Parameter], 
                  config: Dict[str, Any]
    ) -> optim.Optimizer:
    
    optimizer_name = config.get("name", "AdamW")
    params = config.get("params", {})
    
    optimizer_cls = getattr(optim, optimizer_name, None)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Please use a valid optimizer from `torch.optim`.")
    
    return optimizer_cls(parameters, **params)

def get_scheduler(optimizer: optim.Optimizer, 
                  config: Dict[str, Any], 
                  training_context: Dict[str, Any] = None
    ) -> _LRScheduler:
    
    if not config or not config.get("name"):
        logging.info("No scheduler configured.")
        return None
    
    scheduler_name = config["name"]
    params = config.get("params", {})
    
    # Special handling for schedulers that need context from the trainer
    if training_context:
        if scheduler_name == 'CosineAnnealingLR' and 'T_max' not in params:
            params['T_max'] = training_context.get('epochs')
            logging.info(f"Inferred 'T_max'={params['T_max']} for CosineAnnealingLR from training context.")
    
    scheduler_cls = getattr(optim.lr_scheduler, scheduler_name, None)
    if scheduler_cls is None:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. Please use a valid scheduler from `torch.optim.lr_scheduler`.")

    return scheduler_cls(optimizer, **params) 


#============================================================
# Utils
#============================================================
class EarlyStopping:
    
    def __init__(self, 
                 patience: int,
                 checkpoint_path: Union[str, Path],
                 best_model_path: Union[str, Path],
                 verbose: bool = True,
                 delta: float = 0,
                 trace_func: Callable = logging.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            checkpoint_path (str or Path): Path to save the latest model checkpoint.
            best_model_path (str or Path): Path to save the best model.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            trace_func (callable): Function to use for logging messages.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.checkpoint_path = Path(checkpoint_path)
        self.best_model_path = Path(best_model_path)
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Checks if training should be stopped."""
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        torch.save(model.state_dict(), self.checkpoint_path)
        # When validation loss is lower, save it as the best model as well
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss


class RegressionLoss:
    
    @staticmethod
    def get_loss_fn(loss_type: str) -> Callable:
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "mae":
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        elif loss_type == 'log_mse':
            # Log-transformed MSE for positive values
            return RegressionLoss.LogMSELoss()
        elif loss_type == 'log_mae':
            # Log-transformed MAE for positive values  
            return RegressionLoss.LogMAELoss()
        elif loss_type == 'relative_mse':
            # Relative MSE: (pred - true)^2 / true^2
            return RegressionLoss.RelativeMSELoss()
        elif loss_type == 'relative_mae':
            # Relative MAE: |pred - true| / true
            return RegressionLoss.RelativeMAELoss()
        elif loss_type == 'log_cosh':
            # Log-cosh loss for positive values
            return RegressionLoss.LogCoshLoss()
        elif loss_type == 'gaussian_nll':
            return nn.GaussianNLLLoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_type}")
    
    class LogMSELoss(nn.Module):
        """Log-transformed MSE loss for positive values: MSE(log(pred+1), log(true+1))"""
        def __init__(self, epsilon: float = 1e-8):
            super().__init__()
            self.epsilon = epsilon
            
        def forward(self, pred: Tensor, true: Tensor) -> Tensor:
            # Add small epsilon to avoid log(0)
            pred_log = torch.log(torch.clamp(pred, min=self.epsilon) + 1)
            true_log = torch.log(torch.clamp(true, min=self.epsilon) + 1)
            return nn.functional.mse_loss(pred_log, true_log)
    
    class LogMAELoss(nn.Module):
        """Log-transformed MAE loss for positive values: MAE(log(pred+1), log(true+1))"""
        def __init__(self, epsilon: float = 1e-8):
            super().__init__()
            self.epsilon = epsilon
            
        def forward(self, pred: Tensor, true: Tensor) -> Tensor:
            pred_log = torch.log(torch.clamp(pred, min=self.epsilon) + 1)
            true_log = torch.log(torch.clamp(true, min=self.epsilon) + 1)
            return nn.functional.l1_loss(pred_log, true_log)
    
    class RelativeMSELoss(nn.Module):
        """Relative MSE loss: mean((pred - true)^2 / true^2)"""
        def __init__(self, epsilon: float = 1e-8):
            super().__init__()
            self.epsilon = epsilon
            
        def forward(self, pred: Tensor, true: Tensor) -> Tensor:
            true_clamped = torch.clamp(torch.abs(true), min=self.epsilon)
            relative_error = (pred - true) / true_clamped
            return torch.mean(relative_error ** 2)
    
    class RelativeMAELoss(nn.Module):
        """Relative MAE loss: mean(|pred - true| / true)"""
        def __init__(self, epsilon: float = 1e-8):
            super().__init__()
            self.epsilon = epsilon
            
        def forward(self, pred: Tensor, true: Tensor) -> Tensor:
            true_clamped = torch.clamp(torch.abs(true), min=self.epsilon)
            relative_error = torch.abs(pred - true) / true_clamped
            return torch.mean(relative_error)
    
    class LogCoshLoss(nn.Module):
        """Log-cosh loss: log(cosh(pred - true)) - robust to outliers"""
        def forward(self, pred: Tensor, true: Tensor) -> Tensor:
            diff = pred - true
            return torch.mean(torch.log(torch.cosh(diff)))

def calculate_metrics(labels, predictions, running_loss, num_batches):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return {
        'loss': running_loss / num_batches,
        'mse': mse,
        'mae': mae,
        'r2': r2
    } 



#============================================================
# Trainer
#============================================================
class Trainer:
    
    def __init__(self, config: Config):
        self.config = config
        self.device = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.early_stopping = None
        self.logdir = None
        
        self.train_metrics = {'loss': [], 'mse': [], 'mae': [], 'r2': []}
        self.val_metrics = {'loss': [], 'mse': [], 'mae': [], 'r2': []}

    def _setup(self):
        """Initializes all components for training."""
        self._setup_seed()
        self._setup_logging()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_early_stopping()

    def _setup_seed(self):
        seed = self.config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_logging(self):
        self.logdir = Path(f'{self.config.save_dir}/{self.config.experiment_name}')
        self.logdir.mkdir(parents=True, exist_ok=True)
        
        log_path = self.logdir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )
        logging.info("--- Configuration ---")
        for key, value in asdict(self.config).items():
            logging.info(f"{key}: {value}")
        logging.info("---------------------\n")
        
        # Save the config file for reproducibility
        with open(self.logdir / 'config.yaml', 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False, sort_keys=False)

    def _setup_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def _setup_data(self):
        mode_type = self.config.dataset_mode.get('type', 'ratio')
        
        if mode_type == 'files':
            # Load datasets directly from file paths
            paths = self.config.dataset_mode.get('paths', {})
            if not all(paths.get(split) for split in ['train', 'val', 'test']):
                raise ValueError("When using 'files' mode, all paths (train, val, test) must be specified")
                
            logging.info("Loading datasets from specified file paths...")
            
            # Create training dataset without scaling to fit scaler
            temp_train_dataset = Dataset_Photz(
                file_path=paths['train'],
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=None,
                mode='train',
                label_noise=self.config.dataset_params.get('label_noise'),
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            
            # Fit scaler on training data
            scaler_X = StandardScaler()
            scaler_X.fit(temp_train_dataset.features)
            
            # Create all datasets with the fitted scaler
            train_dataset = Dataset_Photz(
                file_path=paths['train'],
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=scaler_X,
                mode='train',
                label_noise=self.config.dataset_params.get('label_noise'),
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            
            val_dataset = Dataset_Photz(
                file_path=paths['val'],
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=scaler_X,
                mode='validation',
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            
            test_dataset = Dataset_Photz(
                file_path=paths['test'],
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=scaler_X,
                mode='test',
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            
            self.trainloader = DataLoader(train_dataset, 
                                          batch_size=self.config.batch_size, 
                                          shuffle=True, num_workers=4, pin_memory=True)
            self.valloader = DataLoader(val_dataset, 
                                        batch_size=self.config.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=True)
            self.testloader = DataLoader(test_dataset, 
                                         batch_size=self.config.batch_size, 
                                         shuffle=False, num_workers=4, pin_memory=True)
            
            # Store scaler for inference use
            self.scaler_X = scaler_X
            
            logging.info(f"Data loaded from paths: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
            
        elif mode_type == 'ratio':
            # Use ratio-based splitting
            ratios = self.config.dataset_mode.get('ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
            logging.info(f"Using ratio-based dataset splitting: {ratios}")
            
            # Read data file to get total number of samples for index splitting
            dataset = cu.readfile(self.config.dataset_params.get('file_path'))
            total_samples = len(dataset)
            
            # Split indices first
            indices = list(range(total_samples))
            val_test_ratio = ratios['val'] + ratios['test']
            train_indices, temp_indices = train_test_split(
                indices, test_size=val_test_ratio, 
                random_state=self.config.random_seed
            )
            val_size_corrected = ratios['val'] / val_test_ratio
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=(1 - val_size_corrected), 
                random_state=self.config.random_seed
            )
            
            # Create dataset without scaling to fit scaler on training data only
            temp_dataset = Dataset_Photz(
                file_path=self.config.dataset_params.get('file_path'),
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=None,  # No scaling applied
                mode='inference',
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            
            # Extract training features and fit scaler
            train_features = temp_dataset.features[train_indices]
            scaler_X = StandardScaler()
            scaler_X.fit(train_features)
            
            # Create all datasets with the fitted scaler
            train_dataset = Dataset_Photz(
                file_path=self.config.dataset_params.get('file_path'),
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=scaler_X,
                mode='train',
                label_noise=self.config.dataset_params.get('label_noise'),
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            train_subset = Subset(train_dataset, train_indices)
            
            val_dataset = Dataset_Photz(
                file_path=self.config.dataset_params.get('file_path'),
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=scaler_X,
                mode='validation',
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            val_subset = Subset(val_dataset, val_indices)
            
            test_dataset = Dataset_Photz(
                file_path=self.config.dataset_params.get('file_path'),
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=scaler_X,
                mode='test',
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
            test_subset = Subset(test_dataset, test_indices)

            self.trainloader = DataLoader(train_subset, 
                                          batch_size=self.config.batch_size, 
                                          shuffle=True, 
                                          num_workers=4, 
                                          pin_memory=True)
            self.valloader = DataLoader(val_subset, 
                                        batch_size=self.config.batch_size, 
                                        shuffle=False, 
                                        num_workers=4, 
                                        pin_memory=True)
            self.testloader = DataLoader(test_subset, 
                                         batch_size=self.config.batch_size, 
                                         shuffle=False, 
                                         num_workers=4, 
                                         pin_memory=True)
            
            # Store scaler for inference use
            self.scaler_X = scaler_X
            
            logging.info(f"Data loaded: {len(train_subset)} train, {len(val_subset)} val, {len(test_subset)} test samples.")
        else:
            raise ValueError(f"Unsupported dataset_mode type: {mode_type}. Use 'ratio' or 'files'.")
        
        # Save scaler immediately after data setup
        self._save_scaler()

    def _setup_model(self):
        # Get input dimension from the first training sample
        sample_batch = next(iter(self.trainloader))
        input_dim = sample_batch['input'].shape[1]
        
        # Create model with correct input dimension
        model_params = self.config.model_params.copy()
        model_params['input_dim'] = input_dim
        
        model_class_name = self.config.model_class_name
        model_class = globals().get(model_class_name)
        if model_class is None:
            raise ValueError(f"Model class '{model_class_name}' not found in the current scope.")
            
        self.model = model_class(**model_params).to(self.device)
        logging.info(f"Model '{self.config.model_class_name}' created with input_dim={input_dim}.")

    def _setup_loss(self):
        self.loss_fn = RegressionLoss.get_loss_fn(self.config.loss_type)
        logging.info(f"Using loss function: {self.config.loss_type}")

    def _setup_optimizer(self):
        """Initializes the optimizer and scheduler using factory functions."""
        self.optimizer = get_optimizer(self.model.parameters(), self.config.optimizer_config)
        
        training_context = {'epochs': self.config.epochs}
        self.scheduler = get_scheduler(self.optimizer, self.config.scheduler_config, training_context)
        
        logging.info("Optimizer and scheduler created.")

    def _setup_early_stopping(self):
        if self.config.early_stopping > 0:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping,
                checkpoint_path=self.logdir / 'checkpoint.pkl',
                best_model_path=self.logdir / 'best_model.pkl',
                verbose=True
            )

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        for batch in self.trainloader:
            inputs, label = batch['input'].to(self.device), batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(inputs)
            
            if self.config.model_class_name in ['DeepPhotozNetWithUncertainty', 'DeepPhotozLSTMWithUncertainty']:
                mu, sigma = output
                var = sigma.pow(2)
                loss = self.loss_fn(mu, label, var)
                preds_for_metrics = mu.cpu().detach().numpy()
            else:
                loss = self.loss_fn(output, label)
                preds_for_metrics = output.cpu().detach().numpy()

            loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clipping['enabled']:
                if self.config.gradient_clipping['method'] == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.gradient_clipping['max_norm']
                    )
                elif self.config.gradient_clipping['method'] == 'value':
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), 
                        clip_value=self.config.gradient_clipping['clip_value']
                    )
            
            self.optimizer.step()
            
            running_loss += loss.item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds_for_metrics)
            
        return calculate_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(self.trainloader))

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        for batch in loader:
            inputs, label = batch['input'].to(self.device), batch['label'].to(self.device)
            output = self.model(inputs)

            if self.config.model_class_name in ['DeepPhotozNetWithUncertainty', 'DeepPhotozLSTMWithUncertainty']:
                mu, sigma = output
                var = sigma.pow(2)
                loss = self.loss_fn(mu, label, var)
                preds_for_metrics = mu.cpu().numpy()
            else:
                loss = self.loss_fn(output, label)
                preds_for_metrics = output.cpu().numpy()

            running_loss += loss.item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds_for_metrics)

        return calculate_metrics(np.array(all_labels), np.array(all_preds), running_loss, len(loader))

    def _update_and_log(self, epoch, train_metrics, val_metrics, duration):
        # Update metrics history
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])

        # Log to console and file
        logging.info(f"Epoch {epoch+1}/{self.config.epochs} | Duration: {duration:.2f}m")
        logging.info(f"\tTrain Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}")
        logging.info(f"\tVal   Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f}")

        # Update learning rate scheduler
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
        logging.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")


    def _plot_metrics(self, epoch):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_map = {
            'loss': 'Loss', 'mse': 'Mean Squared Error',
            'mae': 'Mean Absolute Error', 'r2': 'R2 Score'
        }
        
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
        """Main training loop."""
        self._setup()
        logging.info("--- Starting Training ---")
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            train_metrics = self._train_epoch()
            val_metrics = self._evaluate(self.valloader)
            
            duration = (time.time() - start_time) / 60
            self._update_and_log(epoch, train_metrics, val_metrics, duration)
            self._plot_metrics(epoch)
            
            if self.early_stopping:
                self.early_stopping(val_metrics['loss'], self.model)
                if self.early_stopping.early_stop:
                    logging.info("Early stopping triggered.")
                    break
        
        logging.info("--- Training Finished ---")
        self._test()
        torch.cuda.empty_cache()

    def _test(self):
        """Runs evaluation on the test set with the best model."""
        logging.info("--- Starting Testing ---")
        best_model_path = self.logdir / 'best_model.pkl'
        if not best_model_path.exists():
            logging.warning("No best model found. Testing with the last model.")
            best_model_path = self.logdir / 'checkpoint.pkl'
        
        self.model.load_state_dict(torch.load(best_model_path))
        test_metrics = self._evaluate(self.testloader)
        
        logging.info("Final Test Set Results:")
        logging.info(f"\tTest Loss: {test_metrics['loss']:.4f}, MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")
        
        with open(self.logdir / 'test_results.txt', 'w') as f:
            for key, value in test_metrics.items():
                f.write(f"Test {key}: {value:.6f}\n")
    
    def _save_scaler(self):
        """Save the scaler for inference use."""
        import joblib
        scaler_path = self.logdir / 'scaler.pkl'
        joblib.dump(self.scaler_X, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
    
    def load_scaler(self, scaler_path: str):
        """Load a saved scaler for inference."""
        import joblib
        return joblib.load(scaler_path)
                
                
                
#============================================================
# Inference
#============================================================
class PhotozInference:
    
    def __init__(self, 
                 model_dir: str, 
                 device: str = 'cuda:0', 
                 batch_size: int = 4096, 
                 num_workers: int = 4):
        """
        Args:
            model_dir: Directory containing the trained model, scaler, and config
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
        """
        self.model_dir = Path(model_dir)
        self.device = self._setup_device(device)
        
        # Load configuration
        config_path = self.model_dir / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config = Config()
        self.config.update_from_yaml(str(config_path))
        
        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        
        self._load_model()
        
        logging.info(f"PhotozInference initialized with model from {model_dir}")
        logging.info(f"Using device: {self.device}")
        
    def _setup_device(self, device: str) -> torch.device:
        """设置推理设备"""
        if device == 'cpu':
            return torch.device('cpu')
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        
        # Check if a specific GPU is specified (e.g., cuda:0, cuda:1, etc.)
        if device.startswith('cuda:'):
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            
            try:
                gpu_id = int(device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    warnings.warn(f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found), falling back to cuda:0")
                    return torch.device('cuda:0')
                return torch.device(device)
            except (ValueError, IndexError):
                warnings.warn(f"Invalid device format: {device}, falling back to auto mode")
                return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # auto模式
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_model(self):
        """Load the trained model."""
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
        
        model_class_name = self.config.model_class_name
        model_class = globals().get(model_class_name)
        if model_class is None:
            raise ValueError(f"Model class '{model_class_name}' not found in the current scope.")

        self.model = model_class(**model_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Model loaded from {model_path}")
        
    def _set_dropout_mode(self, training: bool) -> None:
        """设置dropout层的训练/评估模式"""
        def set_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train(training)
        
        self.model.apply(set_dropout)
    
    def _create_dataset(self, data_source):
        mode = 'inference'
        
        if isinstance(data_source, str):
            # File path
            dataset = Dataset_Photz(
                file_path=data_source,
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=self.scaler,
                mode=mode,
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
        else:
            # DataFrame
            dataset = Dataset_Photz(
                dataset=data_source,
                mag_types=self.config.dataset_params.get('mag_types'),
                color_types=self.config.dataset_params.get('color_types'),
                use_mag=self.config.dataset_params.get('use_mag'),
                use_magerr=self.config.dataset_params.get('use_magerr'),
                use_colorerr=self.config.dataset_params.get('use_colorerr'),
                scaler_X=self.scaler,
                mode=mode,
                label_noise=0.0,
                label_transform=self.config.label_transform,
                label_transform_params=self.config.label_transform_params
            )
        
        return dataset
    
    def predict(self, data_source, mc_runs: int = 1, batch_size: int = None):
        """
        Prediction with optional uncertainty quantification using Monte Carlo Dropout.
        
        Args:
            data_source: Either file path (str) or pandas DataFrame
            mc_runs: Number of Monte Carlo runs
                - mc_runs=1: Standard inference mode (no uncertainty), returns predictions only
                - mc_runs>1: Monte Carlo mode, returns statistics dictionary with uncertainty
            batch_size: Batch size for inference (if None, uses config batch_size)
        
        Returns:
            When mc_runs=1: numpy array of predictions
            When mc_runs>1: dict containing 'mean', 'std', and 'all_samples'
        """
        dataset = self._create_dataset(data_source)
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        if mc_runs == 1:
            self.model.eval()  # dropout disabled
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch['input'].to(self.device)
                    outputs = self.model(inputs)
                    
                    if self.config.model_class_name in ['DeepPhotozNetWithUncertainty', 'DeepPhotozLSTMWithUncertainty']:
                        # If the model returns mu and sigma, we only use mu for standard prediction
                        mu, _ = outputs
                        predictions.extend(mu.cpu().numpy())
                    else:
                        predictions.extend(outputs.cpu().numpy())
            
            predictions = np.array(predictions).flatten()
            
            if self.config.label_transform != 'none':
                predictions = Dataset_Photz.inverse_transform_labels(
                    predictions, 
                    self.config.label_transform, 
                    self.config.label_transform_params
                )
                
            return predictions
        
        else:
            # Monte Carlo Dropout mode for uncertainty estimation
            all_predictions = []
            
            for sample_idx in range(mc_runs):
                self.model.eval()
                self._set_dropout_mode(training=True) # dropout enabled
                
                sample_predictions = []
                with torch.no_grad():
                    for batch in dataloader:
                        inputs = batch['input'].to(self.device)
                        outputs = self.model(inputs)
                        
                        if self.config.model_class_name in ['DeepPhotozNetWithUncertainty', 'DeepPhotozLSTMWithUncertainty']:
                            # In MC dropout with uncertainty model, we're interested in mu's stability
                            mu, _ = outputs
                            sample_predictions.extend(mu.cpu().numpy())
                        else:
                            sample_predictions.extend(outputs.cpu().numpy())
                
                all_predictions.append(np.array(sample_predictions).flatten())
            
            all_predictions = np.array(all_predictions)
            
            predictions_mean = np.mean(all_predictions, axis=0)
            predictions_std = np.std(all_predictions, axis=0)
            
            if self.config.label_transform != 'none':
                # Transform all samples first, then calculate statistics
                transformed_samples = []
                for sample in all_predictions:
                    transformed_sample = Dataset_Photz.inverse_transform_labels(
                        sample, 
                        self.config.label_transform, 
                        self.config.label_transform_params
                    )
                    transformed_samples.append(transformed_sample)
                
                transformed_samples = np.array(transformed_samples)
                predictions_mean = np.mean(transformed_samples, axis=0)
                predictions_std = np.std(transformed_samples, axis=0)
                
                return {
                    'mean': predictions_mean,
                    'std': predictions_std,
                    # 'all_samples': transformed_samples
                }
            else:
                return {
                    'mean': predictions_mean,
                    'std': predictions_std,
                    # 'all_samples': all_predictions
                }


#============================================================
# Main
#============================================================
def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train a regression model based on a YAML config file.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help="Path to the config.yaml file.")
    
    args = parser.parse_args()
    
    # Create a default config object
    config = Config()
    
    # Update from YAML file, which is the primary source of configuration
    if Path(args.config).exists():
        config.update_from_yaml(args.config)
    else:
        logging.warning(f"Config file not found at {args.config}. Using default values.")
    config.config_path = args.config

    trainer = Trainer(config)
    trainer.run()

if __name__ == '__main__':
    main() 
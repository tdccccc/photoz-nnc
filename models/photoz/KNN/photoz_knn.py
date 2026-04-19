"""k-Nearest-Neighbor photometric redshift estimator.

Implements a KNN regressor that infers a redshift by averaging the known
redshifts of the k nearest training samples in color space. Three distance
metrics are supported:

* ``euclidean`` -- plain L2 distance between feature vectors.
* ``weighted``  -- each color is scaled by ``1/error`` before L2, so noisier
  colors contribute less.
* ``chi_square`` -- ``sum_k[(c1_k - c2_k)^2 / (err1_k^2 + err2_k^2)]``, which
  accounts for the error of *both* sides. Most accurate when per-sample
  errors are informative; requires the custom batched routine implemented
  here (no sklearn/FAISS backend).

Optional backends:

* FAISS (``use_faiss=True``) replaces sklearn's index for euclidean/weighted
  distance, which is materially faster on large training sets.
* A C++ extension (``cosmic.cpp_chi_square_distance``) replaces the numpy
  broadcast loop for the chi-square path when available.

Typical workflow:

1. Build a :class:`PhotozKNNConfig` (defaults from YAML via
   :meth:`~PhotozKNNConfig.update_from_yaml`).
2. Run :class:`KNNTrainer`, which loads data, fits a scaler (optional),
   fits the KNN index, evaluates on validation/test, and persists the
   model, scaler, config, metrics, and a diagnostic plot.
3. For later predictions on new files, use :class:`PhotozKNNInference`.
"""

import os
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Union, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors

import joblib

# External utils (same as photoz_ann.py)
import cosmic.utils as cu

# Try to import C++ chi-square distance function
try:
    from cosmic import cpp_chi_square_distance
    HAS_CPP_CHI_SQUARE = True
except ImportError:
    HAS_CPP_CHI_SQUARE = False
    logging.warning("C++ chi-square distance not available, using numpy implementation")

# Try to import faiss for faster nearest neighbor search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("FAISS not available, using sklearn implementation. Install with: pip install faiss-cpu (or faiss-gpu)")


# ============================
# Config
# ============================
@dataclass
class PhotozKNNConfig:
    """Configuration for :class:`KNNTrainer`.

    Values can be set at construction, loaded from YAML via
    :meth:`update_from_yaml`, or overridden with a dict via
    :meth:`update_from_dict`. The attribute layout mirrors the RF/XGB
    configs so cross-model experiments stay directly comparable.
    """

    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "photoz_knn"
    random_seed: int = 42

    # KNN Model parameters
    knn_params: Dict[str, Any] = field(default_factory=lambda: {
        'k_neighbors': 50,
        'metric': 'chi_square',  # 'euclidean', 'weighted', 'chi_square'
        'algorithm': 'auto',  # 'auto', 'ball_tree', 'kd_tree', 'brute'
        'leaf_size': 30,
        'n_jobs': -1
    })

    # Data
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    dataset_mode: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'files',
        'paths': {'train': None, 'val': None, 'test': None},
        'ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
    })

    # Evaluation optimization parameters
    train_subsample: int = None  # Subsample training set (e.g., 1000000), None = no subsampling
    validation_subsample: int = None  # Subsample validation set (e.g., 50000), None = no subsampling
    test_batch_size: int = 100000  # Batch size for test set evaluation (to reduce memory usage)

    def update_from_yaml(self, path: str):
        """Load a YAML config file and overwrite matching fields."""
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.update_from_dict(cfg)

    def update_from_dict(self, d: Dict[str, Any]):
        """Apply a dict of overrides. Keys not on the dataclass are ignored."""
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ============================
# Dataset for KNN
# ============================
class DatasetPhotozKNN:
    """Feature container for the KNN regressor.

    Loads a catalog (either from ``dataset`` directly or from ``file_path``),
    slices out the requested color / error columns, optionally applies a
    pre-fit :class:`~sklearn.preprocessing.StandardScaler`, and exposes the
    features in three flavours used by the downstream distance metrics:

    * ``features`` -- concatenated ``[colors, errors]``
    * ``color_features`` -- colors only (used by ``chi_square``)
    * ``weighted_features`` -- colors pre-weighted by ``1/error`` (used by
      the ``weighted`` metric)

    Labels are loaded from the ``z`` column when ``mode`` is
    ``'train' | 'validation' | 'test'``.
    """

    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_path: str = None,
                 color_feature_columns: list = None,
                 error_feature_columns: list = None,
                 weight_by_errors: bool = True,
                 error_floor: float = 0.01,
                 normalize_weights: bool = True,
                 scaler_X: StandardScaler = None,
                 mode: str = 'train'):
        """Initialize the dataset.

        Args:
            dataset: Pre-loaded DataFrame. Mutually exclusive with
                ``file_path``.
            file_path: Path to a file readable by :func:`cosmic.utils.readfile`.
                Mutually exclusive with ``dataset``.
            color_feature_columns: Required list of color column names, e.g.
                ``['Kron_gr', 'Kron_ri', ...]``.
            error_feature_columns: Optional list of error column names (one
                per color). Required for the ``weighted`` and ``chi_square``
                metrics.
            weight_by_errors: If ``True``, :meth:`get_weighted_features`
                divides colors by their errors before returning them.
            error_floor: Lower bound applied to errors to avoid division by
                zero. The effective floor is ``min(data_min_error, error_floor)``.
            normalize_weights: Renormalise each sample's weights so they have
                mean 1 -- keeps the overall scale comparable between samples.
            scaler_X: Optional pre-fit scaler applied to ``features``.
            mode: ``'train' | 'validation' | 'test' | 'inference'``. The
                first three also load labels from the ``z`` column.

        Raises:
            ValueError: If both or neither of ``dataset`` / ``file_path``
                are given, or if ``color_feature_columns`` is missing.
            FileNotFoundError: If ``file_path`` does not exist.
        """
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

        # feature selection and weighting parameters
        self.weight_by_errors = weight_by_errors
        self.error_floor = error_floor
        self.normalize_weights = normalize_weights

        if color_feature_columns is None:
            raise ValueError("'color_feature_columns' must be provided")

        self.color_feature_cols = color_feature_columns
        self.error_feature_cols = error_feature_columns if error_feature_columns is not None else []

        # Combine color and error features for full feature set
        self.feature_cols = self.color_feature_cols + self.error_feature_cols

        # Extract features
        self.features = self.dataset[self.feature_cols].values.astype(np.float32)

        # Store indices for color and error features
        self.color_indices = np.arange(len(self.color_feature_cols))
        if len(self.error_feature_cols) > 0:
            self.error_indices = np.arange(len(self.color_feature_cols), len(self.feature_cols))
            self.error_features = self.features[:, self.error_indices]
        else:
            self.error_indices = np.array([])
            self.error_features = None

        # Always store color features separately for potential weighted distance calculation
        self.color_features = self.features[:, self.color_indices]

        # mode
        self.mode = mode
        if self.mode in ['train', 'validation', 'test']:
            self.labels = self.dataset['z'].values.astype(np.float32)

        # scaler
        if scaler_X is not None:
            self.scaler_X = scaler_X
            self.features = self.scaler_X.transform(self.features)
            # Also scale color and error features separately
            self.color_features = self.features[:, self.color_indices]
            if self.error_features is not None:
                self.error_features = self.features[:, self.error_indices]

    def get_weighted_features(self):
        """Return colors pre-weighted by ``1/error`` for the ``weighted`` metric.

        When ``weight_by_errors`` is ``False`` or no error columns were
        provided, the unweighted colors are returned unchanged.
        """
        if self.error_features is None or not self.weight_by_errors:
            return self.color_features

        # Use the minimum positive error value as the floor, or fall back to configured/default value
        positive_errors = self.error_features[self.error_features > 0]
        if len(positive_errors) > 0:
            error_floor = np.minimum(np.min(positive_errors), self.error_floor)
        else:
            error_floor = self.error_floor

        # Ensure errors have a minimum value to avoid division by zero
        safe_errors = np.maximum(self.error_features, error_floor)

        # Weight colors by inverse of their errors
        weights = 1.0 / safe_errors

        if self.normalize_weights:
            # Normalize weights to have mean 1
            weights = weights / np.mean(weights, axis=1, keepdims=True)

        # Apply weights to color features
        weighted_colors = self.color_features * weights

        return weighted_colors

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Index/slice into all feature flavours at once.

        Returns a dict with keys ``features``, ``weighted_features``,
        ``color_features``, ``error_features`` (may be ``None``) and
        ``labels`` (``None`` for inference mode). The downstream trainer
        and inference classes rely on this bundle so they can feed whichever
        flavour the active metric needs.
        """
        result = {
            'features': self.features[idx],
            'weighted_features': self.get_weighted_features()[idx],
            'color_features': self.color_features[idx],
            'labels': self.labels[idx] if hasattr(self, 'labels') else None
        }
        if self.error_features is not None:
            result['error_features'] = self.error_features[idx]
        else:
            result['error_features'] = None
        return result


# ============================
# KNN Model
# ============================
class PhotozKNNRegressor:
    """KNN regressor backing :class:`KNNTrainer`.

    Wraps three distance-metric backends behind a single :meth:`fit` /
    :meth:`predict` API:

    * ``chi_square`` -- custom double-batched distance (query batch x train
      batch) with an optional C++ speedup. Does not use sklearn/FAISS.
    * ``euclidean`` / ``weighted`` with ``use_faiss=False`` -- sklearn's
      :class:`~sklearn.neighbors.NearestNeighbors`.
    * ``euclidean`` / ``weighted`` with ``use_faiss=True`` -- FAISS
      :class:`faiss.IndexFlatL2`. Falls back to sklearn if FAISS is missing.

    Only the training redshifts are stored for later aggregation; no per-tree
    ensembling.
    """

    def __init__(self,
                 k_neighbors: int = 50,
                 metric: str = 'weighted',  # 'euclidean', 'weighted', 'chi_square'
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 n_jobs: int = -1,
                 use_faiss: bool = False):
        """Initialize the regressor.

        Args:
            k_neighbors: Number of neighbors to retrieve per query.
            metric: ``'euclidean' | 'weighted' | 'chi_square'``. See class
                docstring.
            algorithm: Forwarded to sklearn when using the non-FAISS,
                non-chi-square path.
            leaf_size: Forwarded to sklearn (affects ball_tree/kd_tree).
            n_jobs: Forwarded to sklearn.
            use_faiss: Request the FAISS backend for euclidean/weighted.
                Silently falls back to sklearn if FAISS is not installed.
        """

        self.k_neighbors = k_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.use_faiss = use_faiss and HAS_FAISS  # Only use faiss if available

        if self.use_faiss and not HAS_FAISS:
            logging.warning("FAISS requested but not available. Falling back to sklearn.")
            self.use_faiss = False

        # Initialize the sklearn KNN model (not used for chi_square metric or when using faiss)
        if metric in ['weighted', 'chi_square']:
            # For weighted or chi_square distance, we'll use euclidean base
            self.knn_model = NearestNeighbors(
                n_neighbors=k_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric='euclidean',
                n_jobs=n_jobs
            )
        else:
            self.knn_model = NearestNeighbors(
                n_neighbors=k_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                n_jobs=n_jobs
            )

        self.is_fitted = False
        self.train_labels = None
        self.train_features = None
        self.train_color_features = None
        self.train_error_features = None
        self.use_weighted_features = False
        self.faiss_index = None  # FAISS index for fast search

    def fit(self, features: np.ndarray, labels: np.ndarray, weighted_features: np.ndarray = None,
            color_features: np.ndarray = None, error_features: np.ndarray = None):
        """Fit the KNN model.

        Args:
            features: Training features
            labels: Training spectroscopic redshifts
            weighted_features: Features weighted by errors (for weighted distance)
            color_features: Color features only (for chi_square distance)
            error_features: Error features (for chi_square distance)
        """
        self.train_labels = labels.copy()
        self.train_features = features.copy()

        # Store color and error features for chi_square metric
        if color_features is not None:
            self.train_color_features = color_features.copy()
        if error_features is not None:
            self.train_error_features = error_features.copy()

        if self.metric == 'chi_square':
            # For chi_square, we don't use sklearn's NearestNeighbors or FAISS
            # Distance will be computed manually in predict
            if self.train_color_features is None or self.train_error_features is None:
                raise ValueError("chi_square metric requires both color_features and error_features")
            self.use_weighted_features = False
        elif self.use_faiss and self.metric != 'chi_square':
            # Use FAISS for euclidean or weighted distance
            if weighted_features is not None and self.metric == 'weighted':
                self.use_weighted_features = True
                index_features = weighted_features.astype(np.float32)
            else:
                self.use_weighted_features = False
                index_features = features.astype(np.float32)

            # Build FAISS index
            dimension = index_features.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
            self.faiss_index.add(index_features)
            logging.info(f"FAISS index built with {len(index_features)} samples")
        elif weighted_features is not None and self.metric == 'weighted':
            self.use_weighted_features = True
            self.knn_model.fit(weighted_features)
        else:
            self.use_weighted_features = False
            self.knn_model.fit(features)

        self.is_fitted = True
        backend = "FAISS" if self.use_faiss and self.metric != 'chi_square' else "sklearn"
        logging.info(f"KNN model fitted with {len(features)} training samples, k={self.k_neighbors}, metric={self.metric}, backend={backend}")

    def _compute_chi_square_distance(self, query_colors: np.ndarray, query_errors: np.ndarray,
                                     train_batch_size: int = 100000) -> np.ndarray:
        """Compute chi-square distance between query samples and training samples.

        Uses train set batching to reduce memory usage.
        Prefers C++ implementation if available for speed.

        Distance formula: ``d_ij = sum_k[(color1_ik - color2_jk)^2 / (err1_ik^2 + err2_jk^2)]``

        Args:
            query_colors: Query color features, shape ``(n_queries, n_colors)``.
            query_errors: Query error features, shape ``(n_queries, n_colors)``.
            train_batch_size: Batch size for training set (to save memory).

        Returns:
            Distance matrix of shape ``(n_queries, n_train)``.
        """
        n_queries = query_colors.shape[0]
        n_train = self.train_color_features.shape[0]

        # Initialize output
        all_distances = np.zeros((n_queries, n_train), dtype=np.float32)

        # Batch over training samples to reduce memory
        n_train_batches = (n_train + train_batch_size - 1) // train_batch_size

        for train_batch_idx in range(n_train_batches):
            train_start = train_batch_idx * train_batch_size
            train_end = min((train_batch_idx + 1) * train_batch_size, n_train)

            # Get train batch
            train_colors_batch = self.train_color_features[train_start:train_end]
            train_errors_batch = self.train_error_features[train_start:train_end]

            # Use C++ implementation if available, otherwise use numpy
            if HAS_CPP_CHI_SQUARE:
                # C++ version is much faster
                distances_batch = cpp_chi_square_distance(
                    query_colors.astype(np.float32),
                    query_errors.astype(np.float32),
                    train_colors_batch.astype(np.float32),
                    train_errors_batch.astype(np.float32),
                    error_floor=1e-10
                )
            else:
                # Numpy fallback version
                # Expand dimensions for broadcasting
                query_colors_exp = query_colors[:, np.newaxis, :]  # (n_queries, 1, n_colors)
                train_colors_exp = train_colors_batch[np.newaxis, :, :]  # (1, batch_train, n_colors)

                query_errors_exp = query_errors[:, np.newaxis, :]  # (n_queries, 1, n_colors)
                train_errors_exp = train_errors_batch[np.newaxis, :, :]  # (1, batch_train, n_colors)

                # Compute color differences
                color_diff = query_colors_exp - train_colors_exp  # (n_queries, batch_train, n_colors)

                # Compute error sum (with small floor to avoid division by zero)
                error_floor = 1e-10
                error_sum = query_errors_exp**2 + train_errors_exp**2 + error_floor

                # Compute chi-square distance for this train batch
                chi_square_terms = color_diff**2 / error_sum
                distances_batch = np.sum(chi_square_terms, axis=2)  # (n_queries, batch_train)

            # Store in output
            all_distances[:, train_start:train_end] = distances_batch

        return all_distances

    def _find_k_nearest_chi_square(self, query_colors: np.ndarray, query_errors: np.ndarray,
                                   query_batch_size: int = 5000, train_batch_size: int = 100000):
        """Find k nearest neighbors using chi-square distance.

        Uses double batching (query + train) to avoid memory issues with large datasets.

        Args:
            query_colors: Query color features.
            query_errors: Query error features.
            query_batch_size: Number of query samples to process at once.
            train_batch_size: Number of train samples to process at once
                (passed to :meth:`_compute_chi_square_distance`).

        Returns:
            Tuple ``(distances, indices)`` of shape ``(n_queries, k)`` each,
            sorted ascending by distance.
        """
        n_queries = query_colors.shape[0]
        k = min(self.k_neighbors, self.train_color_features.shape[0])

        # Initialize output arrays
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        all_indices = np.zeros((n_queries, k), dtype=np.int32)

        # Process queries in batches to avoid memory overflow
        n_query_batches = (n_queries + query_batch_size - 1) // query_batch_size

        logging.info(f"  Computing chi-square distances: {n_queries} queries in {n_query_batches} batches")

        for query_batch_idx in range(n_query_batches):
            start_idx = query_batch_idx * query_batch_size
            end_idx = min((query_batch_idx + 1) * query_batch_size, n_queries)

            # Compute distances for this query batch (train set is batched inside)
            batch_colors = query_colors[start_idx:end_idx]
            batch_errors = query_errors[start_idx:end_idx]

            batch_distances = self._compute_chi_square_distance(batch_colors, batch_errors,
                                                                train_batch_size=train_batch_size)  # (batch_size, n_train)

            # Find k smallest distances for this batch
            partition_indices = np.argpartition(batch_distances, k-1, axis=1)[:, :k]
            batch_k_distances = np.take_along_axis(batch_distances, partition_indices, axis=1)

            # Sort within the k nearest neighbors
            sort_indices = np.argsort(batch_k_distances, axis=1)
            batch_k_indices = np.take_along_axis(partition_indices, sort_indices, axis=1)
            batch_k_distances = np.take_along_axis(batch_k_distances, sort_indices, axis=1)

            # Store results
            all_distances[start_idx:end_idx] = batch_k_distances
            all_indices[start_idx:end_idx] = batch_k_indices

            # Print progress
            if n_query_batches > 1 and (query_batch_idx + 1) % max(1, n_query_batches // 10) == 0:
                logging.info(f"    Progress: {query_batch_idx + 1}/{n_query_batches} query batches completed")

        return all_distances, all_indices

    def predict(self, features: np.ndarray, weighted_features: np.ndarray = None,
                color_features: np.ndarray = None, error_features: np.ndarray = None,
                aggregation: str = 'median') -> np.ndarray:
        """Predict redshifts using KNN.

        Args:
            features: Query features.
            weighted_features: Query features weighted by errors.
            color_features: Query color features (for ``chi_square`` distance).
            error_features: Query error features (for ``chi_square`` distance).
            aggregation: How to aggregate neighbor redshifts
                (``'median' | 'mean' | 'weighted_mean'``).

        Returns:
            Predicted redshifts.

        Raises:
            ValueError: If the model has not been fitted, required feature
                inputs for the active metric are missing, or the aggregation
                name is unknown.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Choose distance calculation method based on metric and backend
        if self.metric == 'chi_square':
            if color_features is None or error_features is None:
                raise ValueError("chi_square metric requires both color_features and error_features")
            distances, indices = self._find_k_nearest_chi_square(color_features, error_features)
        elif self.use_faiss:
            # Use FAISS for fast search
            if self.use_weighted_features and weighted_features is not None:
                query_features = weighted_features.astype(np.float32)
            else:
                query_features = features.astype(np.float32)
            distances, indices = self.faiss_index.search(query_features, self.k_neighbors)
            indices = indices.astype(np.int64)  # Convert to int64 for indexing
        elif self.use_weighted_features and weighted_features is not None:
            query_features = weighted_features
            distances, indices = self.knn_model.kneighbors(query_features)
        else:
            query_features = features
            distances, indices = self.knn_model.kneighbors(query_features)

        # Get neighbor redshifts
        neighbor_redshifts = self.train_labels[indices]  # Shape: (n_queries, k)

        # Aggregate neighbor redshifts
        if aggregation == 'median':
            predictions = np.median(neighbor_redshifts, axis=1)
        elif aggregation == 'mean':
            predictions = np.mean(neighbor_redshifts, axis=1)
        elif aggregation == 'weighted_mean':
            # Weight by inverse distance (avoid division by zero)
            weights = 1.0 / (distances + 1e-8)
            weights_norm = weights / np.sum(weights, axis=1, keepdims=True)
            predictions = np.sum(neighbor_redshifts * weights_norm, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return predictions

    def predict_with_uncertainty(self, features: np.ndarray, weighted_features: np.ndarray = None,
                                 color_features: np.ndarray = None, error_features: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict redshift with uncertainty estimates.

        Returns:
            Tuple ``(predictions, uncertainties)`` where ``predictions`` is
            the median neighbor redshift and ``uncertainties`` is the
            standard deviation across the k neighbors.

        Raises:
            ValueError: If the model has not been fitted, or required
                feature inputs for the active metric are missing.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Choose distance calculation method based on metric and backend
        if self.metric == 'chi_square':
            if color_features is None or error_features is None:
                raise ValueError("chi_square metric requires both color_features and error_features")
            distances, indices = self._find_k_nearest_chi_square(color_features, error_features)
        elif self.use_faiss:
            # Use FAISS for fast search
            if self.use_weighted_features and weighted_features is not None:
                query_features = weighted_features.astype(np.float32)
            else:
                query_features = features.astype(np.float32)
            distances, indices = self.faiss_index.search(query_features, self.k_neighbors)
            indices = indices.astype(np.int64)  # Convert to int64 for indexing
        elif self.use_weighted_features and weighted_features is not None:
            query_features = weighted_features
            distances, indices = self.knn_model.kneighbors(query_features)
        else:
            query_features = features
            distances, indices = self.knn_model.kneighbors(query_features)

        # Get neighbor redshifts
        neighbor_redshifts = self.train_labels[indices]  # Shape: (n_queries, k)

        # Calculate predictions and uncertainties
        predictions = np.median(neighbor_redshifts, axis=1)
        uncertainties = np.std(neighbor_redshifts, axis=1)

        return predictions, uncertainties

    def get_neighbors(self, features: np.ndarray, weighted_features: np.ndarray = None,
                      color_features: np.ndarray = None, error_features: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get the k nearest neighbors for each query.

        Returns:
            Tuple ``(distances, neighbor_redshifts)`` where each has shape
            ``(n_queries, k)``.

        Raises:
            ValueError: If the model has not been fitted, or required
                feature inputs for the active metric are missing.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before querying neighbors")

        # Choose distance calculation method based on metric and backend
        if self.metric == 'chi_square':
            if color_features is None or error_features is None:
                raise ValueError("chi_square metric requires both color_features and error_features")
            distances, indices = self._find_k_nearest_chi_square(color_features, error_features)
        elif self.use_faiss:
            # Use FAISS for fast search
            if self.use_weighted_features and weighted_features is not None:
                query_features = weighted_features.astype(np.float32)
            else:
                query_features = features.astype(np.float32)
            distances, indices = self.faiss_index.search(query_features, self.k_neighbors)
            indices = indices.astype(np.int64)  # Convert to int64 for indexing
        elif self.use_weighted_features and weighted_features is not None:
            query_features = weighted_features
            distances, indices = self.knn_model.kneighbors(query_features)
        else:
            query_features = features
            distances, indices = self.knn_model.kneighbors(query_features)

        neighbor_redshifts = self.train_labels[indices]

        return distances, neighbor_redshifts


# ============================
# Utils
# ============================
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics for photometric redshift predictions.

    Args:
        y_true: Spectroscopic redshifts.
        y_pred: Predicted redshifts.

    Returns:
        Dict with ``mse``, ``mae``, ``r2``, ``bias``, ``std``, ``nmad`` and
        ``outlier_fraction`` (fraction of ``|residual| > 0.15``).
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Additional photo-z specific metrics
    residuals = y_pred - y_true
    bias = np.mean(residuals)
    std = np.std(residuals)

    # Normalized median absolute deviation (NMAD)
    nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

    # Outlier fraction (|residual| > 0.15)
    outlier_fraction = np.mean(np.abs(residuals) > 0.15)

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'std': std,
        'nmad': nmad,
        'outlier_fraction': outlier_fraction
    }


# ============================
# KNN Trainer
# ============================
class KNNTrainer:
    """End-to-end KNN photo-z trainer.

    Drives the full workflow: seeds the RNG, sets up logging and the save
    directory, builds the train/val/test datasets (with an optional scaler),
    fits the :class:`PhotozKNNRegressor`, evaluates on the validation split,
    then on the test split in batches, and writes the model, scaler, config,
    metrics, and a diagnostic plot under ``save_dir/experiment_name``.

    KNN has no training epochs, so there is no training-set evaluation loop
    (a training sample is trivially its own nearest neighbor).
    """

    def __init__(self, config: PhotozKNNConfig):
        self.config = config
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.logdir = None
        self.scaler_X = None

        # Only track validation metrics (no training set evaluation for KNN)
        self.val_metrics = {'mse': [], 'mae': [], 'r2': [], 'bias': [], 'std': [], 'nmad': [], 'outlier_fraction': []}

    def _setup(self):
        """Run all setup steps: seed, logging, data, model."""
        self._setup_seed()
        self._setup_logging()
        self._setup_data()
        self._setup_model()

    def _setup_seed(self):
        """Seed numpy's RNG from the config."""
        seed = self.config.random_seed
        np.random.seed(seed)

    def _setup_logging(self):
        """Create the save directory, configure logging, and persist the config."""
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

    def _create_datasets_from_paths(self, paths: Dict[str, str]):
        """Build train/val/test :class:`DatasetPhotozKNN` from three file paths.

        Fits a :class:`StandardScaler` on the training set when
        ``dataset_params.use_scaler`` is truthy (default ``True``) and
        shares it across splits.
        """
        # Check if scaler should be used
        use_scaler = self.config.dataset_params.get('use_scaler', True)

        if use_scaler:
            # Fit scaler on train
            tmp_train = DatasetPhotozKNN(file_path=paths['train'],
                                         color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                         error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                         weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                         error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                         normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                         scaler_X=None,
                                         mode='train')
            scaler_X = StandardScaler()
            scaler_X.fit(tmp_train.features)
            logging.info("Feature scaler fitted on training data")
        else:
            scaler_X = None
            logging.info("Feature scaling disabled (use_scaler=False)")

        self.train_dataset = DatasetPhotozKNN(file_path=paths['train'],
                                              color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                              error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                              weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                              error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                              normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                              scaler_X=scaler_X,
                                              mode='train')
        self.val_dataset = DatasetPhotozKNN(file_path=paths['val'],
                                            color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                            error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                            weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                            error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                            normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                            scaler_X=scaler_X,
                                            mode='validation')
        self.test_dataset = DatasetPhotozKNN(file_path=paths['test'],
                                             color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                             error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                             weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                             error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                             normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                             scaler_X=scaler_X,
                                             mode='test')

        self.scaler_X = scaler_X
        logging.info(f"Data loaded from paths: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test samples.")

    def _setup_data(self):
        """Dispatch to ``files`` or ``ratio`` loader based on ``dataset_mode``."""
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

            # Check if scaler should be used
            use_scaler = self.config.dataset_params.get('use_scaler', True)

            if use_scaler:
                temp_ds = DatasetPhotozKNN(dataset=dataset,
                                           color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                           error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                           weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                           error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                           normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                           scaler_X=None, mode='inference')
                scaler_X = StandardScaler()
                scaler_X.fit(temp_ds.features[np.array(train_idx)])
                logging.info("Feature scaler fitted on training data")
            else:
                scaler_X = None
                logging.info("Feature scaling disabled (use_scaler=False)")

            train_ds = DatasetPhotozKNN(dataset=dataset,
                                        color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                        error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                        weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                        error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                        normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                        scaler_X=scaler_X, mode='train')
            val_ds = DatasetPhotozKNN(dataset=dataset,
                                      color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                      error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                      weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                      error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                      normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                      scaler_X=scaler_X,
                                      mode='validation')
            test_ds = DatasetPhotozKNN(dataset=dataset,
                                       color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                       error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                       weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                       error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                       normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                       scaler_X=scaler_X, mode='test')

            self.scaler_X = scaler_X
            # For slicing support
            self.train_dataset = train_ds
            self.val_dataset = val_ds
            self.test_dataset = test_ds
            self.train_indices = train_idx
            self.val_indices = val_idx
            self.test_indices = test_idx
            logging.info(f"Data loaded: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples.")
        else:
            raise ValueError(f"Unsupported dataset_mode type: {mode_type}")

        self._save_scaler()

    def _setup_model(self):
        """Instantiate the :class:`PhotozKNNRegressor` from ``config.knn_params``."""
        knn_params = self.config.knn_params.copy()
        self.model = PhotozKNNRegressor(**knn_params)
        logging.info(f"KNN model created with parameters: {knn_params}")

    def _save_scaler(self):
        """Persist the fitted scaler (when present) to ``logdir/scaler.pkl``."""
        if self.scaler_X is not None:
            scaler_path = self.logdir / 'scaler.pkl'
            joblib.dump(self.scaler_X, scaler_path)
            logging.info(f"Scaler saved to {scaler_path}")
        else:
            logging.info("No scaler to save (use_scaler=False)")

    def _fit_and_evaluate(self):
        """Fit the KNN model and evaluate on validation set only.

        Applies ``train_subsample`` / ``validation_subsample`` when set so
        that very large catalogs still fit in memory.
        """
        # Prepare training data
        if hasattr(self, 'train_indices'):
            # Using ratio mode
            train_indices = self.train_indices
        else:
            # Using files mode
            train_indices = np.arange(len(self.train_dataset))

        # Subsample training set if configured
        if self.config.train_subsample and len(train_indices) > self.config.train_subsample:
            logging.info(f"Subsampling training set: {len(train_indices)} -> {self.config.train_subsample}")
            train_indices = np.random.choice(train_indices, self.config.train_subsample, replace=False)

        train_data = self.train_dataset[train_indices]

        train_features = train_data['features']
        train_weighted_features = train_data['weighted_features']
        train_color_features = train_data['color_features']
        train_error_features = train_data['error_features']
        train_labels = train_data['labels']

        # Fit the model
        logging.info("Fitting KNN model...")
        self.model.fit(train_features, train_labels,
                      weighted_features=train_weighted_features,
                      color_features=train_color_features,
                      error_features=train_error_features)

        # KNN does not need training set evaluation (nearest neighbor is itself)
        # Only evaluate on validation set
        logging.info("Evaluating on validation set...")
        if hasattr(self, 'val_indices'):
            val_indices = self.val_indices
        else:
            val_indices = np.arange(len(self.val_dataset))

        # Subsample validation set if configured
        if self.config.validation_subsample and len(val_indices) > self.config.validation_subsample:
            logging.info(f"Subsampling validation set: {len(val_indices)} -> {self.config.validation_subsample}")
            val_indices = np.random.choice(val_indices, self.config.validation_subsample, replace=False)

        val_data = self.val_dataset[val_indices]

        val_features = val_data['features']
        val_weighted_features = val_data['weighted_features']
        val_color_features = val_data['color_features']
        val_error_features = val_data['error_features']
        val_labels = val_data['labels']
        val_predictions = self.model.predict(val_features,
                                            weighted_features=val_weighted_features,
                                            color_features=val_color_features,
                                            error_features=val_error_features)
        val_metrics = calculate_regression_metrics(val_labels, val_predictions)

        # Store metrics
        for metric in self.val_metrics.keys():
            if metric in val_metrics:
                self.val_metrics[metric].append(val_metrics[metric])

        # Log results
        logging.info("Validation Set Results:")
        for metric, value in val_metrics.items():
            logging.info(f"  Val {metric}: {value:.6f}")

        return val_metrics

    def _test(self):
        """Evaluate on the test set in batches of ``test_batch_size``.

        Writes ``test_results.txt`` (metrics) and ``test_predictions.npz``
        (true/predicted redshifts) to ``logdir``.
        """
        logging.info("--- Starting Testing ---")
        if hasattr(self, 'test_indices'):
            test_indices = self.test_indices
        else:
            test_indices = np.arange(len(self.test_dataset))

        n_test = len(test_indices)
        batch_size = self.config.test_batch_size

        logging.info(f"Evaluating test set in batches: {n_test} samples, batch_size={batch_size}")

        # Batch evaluation to reduce memory usage
        all_predictions = []
        all_labels = []

        n_batches = (n_test + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_test)
            batch_indices = test_indices[start_idx:end_idx]

            logging.info(f"  Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx})...")

            test_data = self.test_dataset[batch_indices]

            test_features = test_data['features']
            test_weighted_features = test_data['weighted_features']
            test_color_features = test_data['color_features']
            test_error_features = test_data['error_features']
            test_labels = test_data['labels']

            test_predictions = self.model.predict(test_features,
                                                 weighted_features=test_weighted_features,
                                                 color_features=test_color_features,
                                                 error_features=test_error_features)

            all_predictions.append(test_predictions)
            all_labels.append(test_labels)

        # Concatenate all batches
        test_predictions = np.concatenate(all_predictions)
        test_labels = np.concatenate(all_labels)

        test_metrics = calculate_regression_metrics(test_labels, test_predictions)

        logging.info("Final Test Set Results:")
        for metric, value in test_metrics.items():
            logging.info(f"  Test {metric}: {value:.6f}")

        # Save detailed results
        test_results = {
            'true_redshifts': test_labels,
            'predicted_redshifts': test_predictions,
            'metrics': test_metrics
        }

        # Save test results
        with open(self.logdir / 'test_results.txt', 'w') as f:
            for k, v in test_metrics.items():
                f.write(f"Test {k}: {v:.6f}\n")

        # Save predictions
        np.savez(self.logdir / 'test_predictions.npz',
                 true_redshifts=test_labels,
                 predicted_redshifts=test_predictions)

        return test_results

    def _save_model(self):
        """Save the fitted KNN model to ``logdir/knn_model.pkl``."""
        model_path = self.logdir / 'knn_model.pkl'
        joblib.dump(self.model, model_path)
        logging.info(f"KNN model saved to {model_path}")

    def _plot_results(self, test_results):
        """Plot predicted-vs-true and residual scatter into ``results_plot.jpg``."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot: predicted vs true
        true_z = test_results['true_redshifts']
        pred_z = test_results['predicted_redshifts']

        axes[0].scatter(true_z, pred_z, alpha=0.5, s=1)
        axes[0].plot([true_z.min(), true_z.max()], [true_z.min(), true_z.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Redshift')
        axes[0].set_ylabel('Predicted Redshift')
        axes[0].set_title('Predicted vs True Redshift')
        axes[0].grid(True)

        # Residual plot
        residuals = pred_z - true_z
        axes[1].scatter(true_z, residuals, alpha=0.5, s=1)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('True Redshift')
        axes[1].set_ylabel('Residual (Pred - True)')
        axes[1].set_title('Residuals')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.logdir / 'results_plot.jpg', dpi=300)
        plt.close()

    def run(self):
        """Run the full setup -> fit -> validate -> test -> save -> plot pipeline."""
        self._setup()
        logging.info("--- Starting KNN Training and Evaluation ---")
        val_metrics = self._fit_and_evaluate()
        test_results = self._test()
        self._save_model()
        self._plot_results(test_results)
        logging.info("--- KNN Training and Evaluation Finished ---")


# ============================
# Inference helper
# ============================
class PhotozKNNInference:
    """Inference helper for trained KNN models.

    Loads a previously trained experiment (config, optional scaler, model)
    from a directory written by :meth:`KNNTrainer.run` and exposes a
    standalone :meth:`predict` / :meth:`get_neighbors` without rebuilding a
    :class:`KNNTrainer` (no data load, no scaler fit).
    """

    def __init__(self, model_dir: str, batch_size: int = 4096):
        """Initialize the inference helper.

        Args:
            model_dir: Path to the experiment directory containing
                ``config.yaml``, ``knn_model.pkl``, and optionally
                ``scaler.pkl``.
            batch_size: Currently recorded on the helper; the underlying
                :class:`PhotozKNNRegressor` handles batching internally for
                chi-square and uses a single vectorised call otherwise.

        Raises:
            FileNotFoundError: If ``config.yaml`` or ``knn_model.pkl`` is
                missing.
        """
        self.model_dir = Path(model_dir)
        cfg_path = self.model_dir / 'config.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        self.config = PhotozKNNConfig(); self.config.update_from_yaml(str(cfg_path))

        # Load scaler if it exists (optional, depends on use_scaler config)
        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
            logging.info("No scaler found, proceeding without feature scaling")

        self._load_model()
        self.batch_size = batch_size

    def _load_model(self):
        """Load the pickled :class:`PhotozKNNRegressor` from ``knn_model.pkl``."""
        model_path = self.model_dir / 'knn_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"No model file found in {self.model_dir}")
        self.model = joblib.load(model_path)

    def _create_dataset(self, data_source):
        """Wrap ``data_source`` (path or DataFrame) in a :class:`DatasetPhotozKNN`."""
        if isinstance(data_source, str):
            ds = DatasetPhotozKNN(file_path=data_source,
                                  color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                  error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                  weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                  error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                  normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                  scaler_X=self.scaler, mode='inference')
        else:
            ds = DatasetPhotozKNN(dataset=data_source,
                                  color_feature_columns=self.config.dataset_params.get('color_feature_columns'),
                                  error_feature_columns=self.config.dataset_params.get('error_feature_columns'),
                                  weight_by_errors=self.config.dataset_params.get('weight_by_errors', True),
                                  error_floor=self.config.dataset_params.get('error_floor', 0.01),
                                  normalize_weights=self.config.dataset_params.get('normalize_weights', True),
                                  scaler_X=self.scaler, mode='inference')
        return ds

    def predict(self, data_source, aggregation: str = 'median', with_uncertainty: bool = False):
        """Predict redshift for given data.

        Args:
            data_source: Input data (file path or DataFrame).
            aggregation: How to aggregate neighbor redshifts
                (``'median' | 'mean' | 'weighted_mean'``).
            with_uncertainty: Whether to return uncertainty estimates.

        Returns:
            Dict with ``'predictions'`` (and ``'uncertainties'`` when
            ``with_uncertainty=True``).
        """
        ds = self._create_dataset(data_source)
        data = ds[:]
        features = data['features']
        weighted_features = data['weighted_features']
        color_features = data['color_features']
        error_features = data['error_features']

        if with_uncertainty:
            predictions, uncertainties = self.model.predict_with_uncertainty(
                features,
                weighted_features=weighted_features,
                color_features=color_features,
                error_features=error_features
            )
            return {'predictions': predictions, 'uncertainties': uncertainties}
        else:
            predictions = self.model.predict(
                features,
                weighted_features=weighted_features,
                color_features=color_features,
                error_features=error_features,
                aggregation=aggregation
            )
            return {'predictions': predictions}

    def get_neighbors(self, data_source):
        """Get the k nearest neighbors for each query sample.

        Args:
            data_source: Input data (file path or DataFrame).

        Returns:
            Dict with ``'distances'`` and ``'neighbor_redshifts'``, each of
            shape ``(n_queries, k)``.
        """
        ds = self._create_dataset(data_source)
        data = ds[:]
        features = data['features']
        weighted_features = data['weighted_features']
        color_features = data['color_features']
        error_features = data['error_features']

        distances, neighbor_redshifts = self.model.get_neighbors(
            features,
            weighted_features=weighted_features,
            color_features=color_features,
            error_features=error_features
        )

        return {
            'distances': distances,
            'neighbor_redshifts': neighbor_redshifts
        }


# ============================
# Main
# ============================
def main():
    """CLI entry point: load a YAML config and run :class:`KNNTrainer`."""
    parser = argparse.ArgumentParser(description="Train a KNN model for photo-z.")
    parser.add_argument('--config', type=str, default='photoz_knn/config.yaml',
                        help="Path to the config.yaml file.")
    args = parser.parse_args()

    config = PhotozKNNConfig()
    if Path(args.config).exists():
        config.update_from_yaml(args.config)
    else:
        logging.warning(f"Config file not found at {args.config}. Using default values.")
    config.config_path = args.config

    trainer = KNNTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()

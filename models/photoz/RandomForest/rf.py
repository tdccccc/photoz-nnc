"""Random Forest photometric redshift estimator.

Thin wrapper around :class:`sklearn.ensemble.RandomForestRegressor` that
shares the configuration and directory conventions used by the NNC
(``photoz_bin``) trainer, so results across model families stay directly
comparable.

Typical workflow:

1. Build a :class:`PhotozRFConfig` (or load it from YAML).
2. Instantiate :class:`PhotozRandomForest` -- this loads the train/val/test
   data and fits a :class:`~sklearn.preprocessing.StandardScaler` on the
   training features.
3. Call :meth:`PhotozRandomForest.fit` and
   :meth:`PhotozRandomForest.evaluate` to train, score, plot, and persist
   results.
4. For later predictions on new files, load the saved experiment with
   :class:`PhotozRFInference`.

Prediction uncertainty is reported as the standard deviation of the
individual tree predictions across ``self.rf.estimators_``.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import cosmic.utils as cu
from cosmic.panstarrs_dr2 import ColorCalculator
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import yaml
from pathlib import Path
import joblib

#============================================================
# Config
#============================================================
@dataclass
class PhotozRFConfig:
    """Configuration for :class:`PhotozRandomForest`.

    Values can be set at construction, loaded from YAML via
    :meth:`update_from_yaml`, or round-tripped with :meth:`save_to_yaml`. The
    attribute layout mirrors the NNC config so cross-model experiments stay
    directly comparable without extra bookkeeping.
    """

    # Experiment setup
    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "base_rf_experiment"
    random_seed: int = 42

    # Dataset configuration (similar to photoz_bin)
    dataset_params: Dict[str, Any] = field(default_factory=lambda: {
        'feature_generation': {
            'mag_types': ['Kron', 'PSF', 'Ap'],
            'color_types': ['Kron', 'Ap'],
            'use_mag': True,
            'use_magerr': True,
            'use_colorerr': False
        },
        'feature_columns': None  # Alternative: specify columns directly
    })

    dataset_mode: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'files',  # 'files' or 'ratio'
        'paths': {
            'train': None,
            'val': None,
            'test': None
        },
        'ratios': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        },
        'data_path': None  # Used when type='ratio'
    })

    # Random Forest parameters
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    })

    # Feature scaling
    use_scaler: bool = True

    # Output settings
    save_model: bool = True
    save_results: bool = True
    plot_results: bool = True

    def update_from_yaml(self, path: str):
        """Load a YAML config file and overwrite matching fields."""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        self.update_from_dict(yaml_config)

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Apply a dict of overrides. Keys not on the dataclass are ignored."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_yaml(self, path: str = None):
        """Save the current config to YAML.

        When ``path`` is ``None`` the file is written to
        ``save_dir/experiment_name/config.yaml``, creating parent directories
        as needed.
        """
        if path is None:
            save_dir = Path(self.save_dir) / self.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / "config.yaml"

        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to: {path}")

    def get_save_dir(self) -> Path:
        """Return (and create) the ``save_dir/experiment_name`` directory."""
        save_dir = Path(self.save_dir) / self.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir


def load_datasets(path):
    """Load a catalog file via :func:`cosmic.utils.readfile`.

    Args:
        path: Path to a FITS/CSV/Parquet file readable by ``cosmic.utils``.

    Returns:
        The loaded :class:`pandas.DataFrame`. Row count is printed as a
        sanity check.
    """
    df = cu.readfile(path)
    print(f"total: {df.shape[0]}")
    return df


def choose_features(feature_generation: dict = None,
                    feature_columns: list = None):
    """Generate feature column names for PS1-style catalogs.

    Args:
        feature_generation: Dict with ``mag_types``, ``color_types``,
            ``use_mag``, ``use_magerr``, ``use_colorerr``. Expanded into PS1
            column names such as ``Kron_gr`` or ``gKronMag_dered``.
        feature_columns: Direct list of column names. Takes precedence over
            ``feature_generation`` when provided -- use this for non-PS1
            surveys whose catalogs use different conventions.

    Returns:
        List of feature column names, with magnitudes/colors first and error
        columns appended after.

    Raises:
        ValueError: If neither ``feature_columns`` nor ``feature_generation``
            is provided.
    """
    if feature_columns is not None:
        return feature_columns

    if feature_generation is None:
        raise ValueError("Either 'feature_columns' or 'feature_generation' must be provided")

    mag_types = feature_generation.get('mag_types', [])
    color_types = feature_generation.get('color_types', [])
    use_mag = feature_generation.get('use_mag', True)
    use_magerr = feature_generation.get('use_magerr', True)
    use_colorerr = feature_generation.get('use_colorerr', False)

    bands = ['g', 'r', 'i', 'z', 'y']
    colours = ['gr', 'ri', 'iz', 'zy']
    mag_cols = []
    err_cols = []

    # Colors first so the ordering matches the NNC feature vector.
    if color_types:
        for color_type in color_types:
            for colour in colours:
                mag_cols.append(f'{color_type}_{colour}')
                if use_colorerr:
                    err_cols.append(f'{color_type}_{colour}_Err')

    if use_mag and mag_types:
        for band in bands:
            for mag_type in mag_types:
                mag_cols.append(f'{band}{mag_type}Mag_dered')
                if use_magerr:
                    err_cols.append(f'{band}{mag_type}MagErr')

    combined_cols = mag_cols + err_cols
    return combined_cols


class PhotozRandomForest:
    """End-to-end Random Forest photo-z trainer.

    On construction the class loads the train/val/test data (either from
    three files or by splitting a single file), fits a
    :class:`~sklearn.preprocessing.StandardScaler` on the training features,
    and prepares an unfitted
    :class:`~sklearn.ensemble.RandomForestRegressor`. Call :meth:`fit` and
    then :meth:`evaluate` to complete the workflow.

    Uncertainty from :meth:`predict` is estimated as the standard deviation
    of the per-tree predictions.
    """

    def __init__(self, config: PhotozRFConfig):
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaler_X = None

        # Generate features based on config
        dataset_params = config.dataset_params
        self.features_names = choose_features(
            feature_generation=dataset_params.get('feature_generation'),
            feature_columns=dataset_params.get('feature_columns')
        )
        self.n_features = len(self.features_names)

        # Initialize Random Forest with config parameters
        model_params = config.model_params
        self.rf = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            max_features=model_params.get('max_features', 'sqrt'),
            random_state=config.random_seed,
            n_jobs=-1
        )
        self.is_trained = False

        # Load data based on config
        self._setup_data()
        self._print_params()

    def _print_params(self):
        """Pretty-print the effective config and feature list."""
        print("=" * 60)
        print("Random Forest Model Parameters")
        print("=" * 60)
        print(f"{'experiment_name':18s}: {self.config.experiment_name}")
        print(f"{'save_dir':18s}: {self.config.save_dir}")

        model_params = self.config.model_params
        print(f"{'n_estimators':18s}: {model_params.get('n_estimators', 100)}")
        print(f"{'max_depth':18s}: {model_params.get('max_depth', None)}")
        print(f"{'min_samples_split':18s}: {model_params.get('min_samples_split', 2)}")
        print(f"{'min_samples_leaf':18s}: {model_params.get('min_samples_leaf', 1)}")
        print(f"{'max_features':18s}: {model_params.get('max_features', 'sqrt')}")
        print(f"{'random_seed':18s}: {self.config.random_seed}")
        print(f"{'use_scaler':18s}: {self.config.use_scaler}")
        print(f"{'n_features':18s}: {self.n_features}")

        features_str = str(self.features_names)
        max_line_length = 80
        print(f"{'features':18s}: ", end="")
        while len(features_str) > max_line_length:
            print(features_str[:max_line_length])
            features_str = features_str[max_line_length:]
            print(" " * 20, end="")
        print(features_str)
        print("=" * 60)

    def _setup_data(self):
        """Load and setup data based on dataset_mode configuration."""
        mode_type = self.config.dataset_mode.get('type', 'files')

        if mode_type == 'files':
            self._load_data_from_files()
        elif mode_type == 'ratio':
            self._load_data_from_ratio()
        else:
            raise ValueError(f"Unsupported dataset_mode type: {mode_type}")

        # Fit scaler on training data if enabled
        if self.config.use_scaler:
            self._fit_scaler()
            self._save_scaler()

    def _load_data_from_files(self):
        """Load data from separate train/val/test files."""
        paths = self.config.dataset_mode.get('paths', {})

        if paths.get('train'):
            print(f"Loading training data from: {paths['train']}")
            self.train_df = load_datasets(paths['train'])
        else:
            raise ValueError("Training data path must be specified in 'files' mode")

        if paths.get('val'):
            print(f"Loading validation data from: {paths['val']}")
            self.val_df = load_datasets(paths['val'])

        if paths.get('test'):
            print(f"Loading test data from: {paths['test']}")
            self.test_df = load_datasets(paths['test'])
        else:
            raise ValueError("Test data path must be specified in 'files' mode")

        print(f"\nData loaded from files:")
        print(f"  trainset: {self.train_df.shape[0]}")
        if self.val_df is not None:
            print(f"  validation set: {self.val_df.shape[0]}")
        print(f"  testset: {self.test_df.shape[0]}")

    def _load_data_from_ratio(self):
        """Load data from a single file and split by ratios."""
        data_path = self.config.dataset_mode.get('data_path')
        if not data_path:
            raise ValueError("'data_path' must be specified when using 'ratio' mode")

        print(f"Loading data from: {data_path}")
        df = load_datasets(data_path)

        ratios = self.config.dataset_mode.get('ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
        train_ratio = ratios.get('train', 0.8)
        val_ratio = ratios.get('val', 0.1)
        test_ratio = ratios.get('test', 0.1)

        # Normalize so the three ratios sum to 1 regardless of user input.
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        # First split: train vs (val + test).
        val_test_ratio = val_ratio + test_ratio
        self.train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=self.config.random_seed
        )

        # Second split: val vs test.
        if val_ratio > 0:
            val_size_in_temp = val_ratio / val_test_ratio
            self.val_df, self.test_df = train_test_split(
                temp_df,
                train_size=val_size_in_temp,
                random_state=self.config.random_seed
            )
        else:
            self.test_df = temp_df
            self.val_df = None

        print(f"\nData split completed:")
        print(f"  trainset: {self.train_df.shape[0]}")
        if self.val_df is not None:
            print(f"  validation set: {self.val_df.shape[0]}")
        print(f"  testset: {self.test_df.shape[0]}")

    def _fit_scaler(self):
        """Fit StandardScaler on training data."""
        self.scaler_X = StandardScaler()
        train_features = self.train_df[self.features_names].values
        self.scaler_X.fit(train_features)
        print("StandardScaler fitted on training data")

    def _save_scaler(self):
        """Save the scaler to disk."""
        if self.scaler_X is not None:
            save_dir = self.config.get_save_dir()
            scaler_path = save_dir / 'scaler.pkl'
            joblib.dump(self.scaler_X, scaler_path)
            print(f"Scaler saved to: {scaler_path}")

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Get features from dataframe, applying scaler if enabled."""
        features = df[self.features_names].values
        if self.config.use_scaler and self.scaler_X is not None:
            features = self.scaler_X.transform(features)
        return features

    def _evaluate(self, y_true, y_pred):
        """Delegate to :func:`cosmic.utils.evaluate_redshift_quality`."""
        metrics = cu.evaluate_redshift_quality(y_pred, y_true)
        return metrics

    def _print_metrics_comparison(self, train_metrics, test_metrics):
        """Print a side-by-side table of train vs. test metrics."""
        print("\n" + "="*60)
        print(f"{'Metric':<20} {'Trainset':<12} {'Testset':<12}")
        print("="*60)
        print(f"{'<Δz_norm>':<20} {train_metrics['mean_bias']:<12.6f} {test_metrics['mean_bias']:<12.6f}")
        print(f"{'std':<20} {train_metrics['std_dev']:<12.6f} {test_metrics['std_dev']:<12.6f}")
        print(f"{'MAD':<20} {train_metrics['mad']:<12.6f} {test_metrics['mad']:<12.6f}")
        print(f"{'outlier':<20} {train_metrics['outlier_fraction']:<12.6f} {test_metrics['outlier_fraction']:<12.6f}")
        print("="*60)

    def _print_metrics_comparison_with_val(self, train_metrics, val_metrics, test_metrics):
        """Print metrics comparison including validation set."""
        print("\n" + "="*80)
        print(f"{'Metric':<20} {'Trainset':<12} {'Validation':<12} {'Testset':<12}")
        print("="*80)
        print(f"{'<bias>':<20} {train_metrics['mean_bias']:<12.6f} {val_metrics['mean_bias']:<12.6f} {test_metrics['mean_bias']:<12.6f}")
        print(f"{'std':<20} {train_metrics['std_dev']:<12.6f} {val_metrics['std_dev']:<12.6f} {test_metrics['std_dev']:<12.6f}")
        print(f"{'MAD':<20} {train_metrics['mad']:<12.6f} {val_metrics['mad']:<12.6f} {test_metrics['mad']:<12.6f}")
        print(f"{'outlier':<20} {train_metrics['outlier_fraction']:<12.6f} {val_metrics['outlier_fraction']:<12.6f} {test_metrics['outlier_fraction']:<12.6f}")
        print("="*80)

    def fit(self):
        """Fit the Random Forest model on training data.

        Also persists the trained model and config to
        ``save_dir/experiment_name`` when ``config.save_model`` is ``True``.
        """
        X_train = self._get_features(self.train_df)
        y_train = self.train_df['z'].values

        print("fitting...")
        self.rf.fit(X_train, y_train)
        self.is_trained = True
        print("done!")

        if self.config.save_model:
            self.save_model()
            self.config.save_to_yaml()

    def evaluate(self):
        """Predict on every loaded split and report/plot/save metrics."""
        # Evaluate on training set
        X_train = self._get_features(self.train_df)
        self.train_pred = self.rf.predict(X_train)
        self.train_df['z_pred'] = self.train_pred
        self.train_metrics = self._evaluate(self.train_df['z'].values, self.train_pred)

        # Evaluate on validation set if available
        if self.val_df is not None:
            X_val = self._get_features(self.val_df)
            self.val_pred = self.rf.predict(X_val)
            self.val_df['z_pred'] = self.val_pred
            self.val_metrics = self._evaluate(self.val_df['z'].values, self.val_pred)

        # Evaluate on test set
        X_test = self._get_features(self.test_df)
        self.test_pred = self.rf.predict(X_test)
        self.test_df['z_pred'] = self.test_pred
        self.test_metrics = self._evaluate(self.test_df['z'].values, self.test_pred)

        if self.val_df is not None:
            self._print_metrics_comparison_with_val(self.train_metrics, self.val_metrics, self.test_metrics)
        else:
            self._print_metrics_comparison(self.train_metrics, self.test_metrics)

        if self.config.plot_results:
            self._plot_results()

        if self.config.save_results:
            self.save_results()

    def _plot_results(self):
        """Save a two-panel hexbin of true vs. predicted redshift."""
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot([0, 1.], [0, 1.], color='gray', linestyle='--')
        plt.hexbin(self.train_df['z'].values, self.train_pred,
                   gridsize=100, bins='log', cmap='Blues', mincnt=1)
        plt.xlabel('True Redshift')
        plt.ylabel('Estimated Redshift')
        plt.xlim(0, 1.)
        plt.ylim(0, 1.)
        plt.title('Training set')
        text = f'<Δz_norm> = {self.train_metrics["mean_bias"]:.5f}\nstd = {self.train_metrics["std_dev"]:.5f}\nMAD = {self.train_metrics["mad"]:.5f}\noutliers = {self.train_metrics["outlier_fraction"]:.5f}'
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=12)

        plt.subplot(122)
        plt.plot([0, 1.], [0, 1.], color='gray', linestyle='--')
        plt.hexbin(self.test_df['z'].values, self.test_pred,
                   gridsize=100, bins='log', cmap='Blues', mincnt=1)
        plt.xlabel('True Redshift')
        plt.ylabel('Estimated Redshift')
        plt.xlim(0, 1.)
        plt.ylim(0, 1.)
        plt.title('Test set')
        text = f'<Δz_norm> = {self.test_metrics["mean_bias"]:.5f}\nstd = {self.test_metrics["std_dev"]:.5f}\nMAD = {self.test_metrics["mad"]:.5f}\noutliers = {self.test_metrics["outlier_fraction"]:.5f}'
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=12)
        plt.tight_layout()

        save_dir = self.config.get_save_dir()
        plot_path = save_dir / f'results.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Results plot saved to: {plot_path}")

    def save_model(self):
        """Save the trained model to ``save_dir/rf_model.joblib``."""
        if not self.is_trained:
            print("Model not trained yet. Cannot save.")
            return

        save_dir = self.config.get_save_dir()
        model_path = save_dir / "rf_model.joblib"
        joblib.dump(self.rf, model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path: str = None):
        """Load a trained model (and its scaler if available) from disk.

        Args:
            model_path: Path to the ``.joblib`` file. Defaults to
                ``save_dir/rf_model.joblib``.
        """
        if model_path is None:
            save_dir = self.config.get_save_dir()
            model_path = save_dir / "rf_model.joblib"

        if os.path.exists(model_path):
            self.rf = joblib.load(model_path)
            self.is_trained = True
            print(f"Model loaded from: {model_path}")

            # Also load scaler if it was saved alongside the model.
            scaler_path = Path(model_path).parent / "scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler_X = joblib.load(scaler_path)
                print(f"Scaler loaded from: {scaler_path}")
        else:
            print(f"Model file not found: {model_path}")

    def save_results(self):
        """Write full result tables and a metrics summary to the save dir.

        Produces ``train_results.fits``, ``val_results.fits`` (when the split
        exists), ``test_results.fits``, per-split prediction files via
        :meth:`_save_predictions`, and a human-readable ``metrics.txt``.
        """
        save_dir = self.config.get_save_dir()

        train_results_path = save_dir / "train_results.fits"
        cu.savefile(self.train_df, train_results_path)

        if hasattr(self, 'val_df') and self.val_df is not None:
            val_results_path = save_dir / "val_results.fits"
            cu.savefile(self.val_df, val_results_path)

        test_results_path = save_dir / "test_results.fits"
        cu.savefile(self.test_df, test_results_path)

        # Companion files with only (z, z_pred, z_err) for quick reloads.
        self._save_predictions()

        metrics_path = save_dir / "metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Train Metrics:\n")
            for k, v in self.train_metrics.items():
                f.write(f"  {k}: {v:.6f}\n")

            if hasattr(self, 'val_metrics'):
                f.write("\nValidation Metrics:\n")
                for k, v in self.val_metrics.items():
                    f.write(f"  {k}: {v:.6f}\n")

            f.write("\nTest Metrics:\n")
            for k, v in self.test_metrics.items():
                f.write(f"  {k}: {v:.6f}\n")
        print(f"Results saved to: {save_dir}")

    def _save_predictions(self):
        """Save per-split prediction files with ``(z, z_pred, z_err)``.

        ``z_err`` is the standard deviation of the per-tree predictions, a
        coarse but cheap uncertainty estimate.
        """
        save_dir = self.config.get_save_dir()

        X_train = self._get_features(self.train_df)
        train_predictions = np.array([tree.predict(X_train) for tree in self.rf.estimators_])
        train_z_err = np.std(train_predictions, axis=0)

        X_test = self._get_features(self.test_df)
        test_predictions = np.array([tree.predict(X_test) for tree in self.rf.estimators_])
        test_z_err = np.std(test_predictions, axis=0)

        train_pred_df = pd.DataFrame({
            'z': self.train_df['z'].values,
            'z_pred': self.train_pred,
            'z_err': train_z_err
        })
        cu.savefile(train_pred_df, save_dir / "predictions_train.fits")

        if self.val_df is not None:
            X_val = self._get_features(self.val_df)
            val_predictions = np.array([tree.predict(X_val) for tree in self.rf.estimators_])
            val_z_err = np.std(val_predictions, axis=0)

            val_pred_df = pd.DataFrame({
                'z': self.val_df['z'].values,
                'z_pred': self.val_pred,
                'z_err': val_z_err
            })
            cu.savefile(val_pred_df, save_dir / "predictions_val.fits")

        test_pred_df = pd.DataFrame({
            'z': self.test_df['z'].values,
            'z_pred': self.test_pred,
            'z_err': test_z_err
        })
        cu.savefile(test_pred_df, save_dir / "predictions_test.fits")

        print(f"Predictions saved to: {save_dir}/predictions_*.fits")

    def predict(self, data_source, return_uncertainty=True):
        """Predict photometric redshifts for new data.

        Args:
            data_source: ``DataFrame``, file path, or numpy array of features.
                A numpy array is assumed to already match ``features_names``
                in order; it is passed through the scaler if one is active.
            return_uncertainty: Whether to compute uncertainty from the
                variance of the per-tree predictions.

        Returns:
            Tuple ``(phot_z, phot_z_err)``. ``phot_z_err`` is ``None`` when
            ``return_uncertainty`` is ``False``.

        Raises:
            ValueError: If :meth:`fit` has not been called or
                ``data_source`` has an unsupported type.
        """
        if not self.is_trained:
            raise ValueError("Model not trained, please call fit() method")

        if isinstance(data_source, pd.DataFrame):
            X_test = self._get_features(data_source)
        elif isinstance(data_source, str):
            df = load_datasets(data_source)
            X_test = self._get_features(df)
        elif isinstance(data_source, np.ndarray):
            X_test = data_source
            if self.config.use_scaler and self.scaler_X is not None:
                X_test = self.scaler_X.transform(X_test)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        phot_z = np.full(len(X_test), np.nan)
        phot_z_err = np.full(len(X_test), np.nan) if return_uncertainty else None

        if return_uncertainty:
            # Averaging across trees gives a cheaper mean than rf.predict while
            # also exposing the per-tree spread for an uncertainty estimate.
            predictions = np.array([tree.predict(X_test) for tree in self.rf.estimators_])
            phot_z = np.mean(predictions, axis=0)
            phot_z_err = np.std(predictions, axis=0)
        else:
            phot_z = self.rf.predict(X_test)

        return phot_z, phot_z_err

    def get_feature_importance(self):
        """Return the Random Forest feature importances.

        Returns:
            Tuple ``(importances, feature_names)``. Both are in the same
            order used during training.

        Raises:
            ValueError: If :meth:`fit` has not been called yet.
        """
        if not self.is_trained:
            raise ValueError("Model not trained, please call fit() method")

        importances = self.rf.feature_importances_

        print("\nFeature importances:")
        for i, (name, importance) in enumerate(zip(self.features_names, importances)):
            print(f"{i+1:2d}. {name:15s}: {importance:.4f}")

        return importances, self.features_names

    def plot_feature_importance(self):
        """Plot feature importances as a sorted bar chart.

        Saves the figure to ``save_dir/feature_importance.png`` when
        ``config.save_results`` is enabled.
        """
        importances, feature_names = self.get_feature_importance()

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Random Forest Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Importance')
        plt.tight_layout()

        if self.config.save_results:
            save_dir = self.config.get_save_dir()
            feature_plot_path = save_dir / "feature_importance.png"
            plt.savefig(feature_plot_path, dpi=300)
            print(f"Feature importance plot saved to: {feature_plot_path}")

        plt.show()

        return importances, feature_names


# ============================================================
# Inference helper (similar to photoz_bin)
# ============================================================
class PhotozRFInference:
    """Inference helper for trained Random Forest models.

    Loads a previously trained experiment (config, scaler, model) from a
    directory written by :meth:`PhotozRandomForest.fit` and exposes a
    standalone :meth:`predict` without rebuilding a
    :class:`PhotozRandomForest` (no data load, no scaler fit).
    """

    def __init__(self, model_dir: str):
        """Initialize the inference helper.

        Args:
            model_dir: Path to the experiment directory containing
                ``config.yaml``, ``rf_model.joblib``, and optionally
                ``scaler.pkl``.

        Raises:
            FileNotFoundError: If ``config.yaml`` or ``rf_model.joblib`` is
                missing.
        """
        self.model_dir = Path(model_dir)

        cfg_path = self.model_dir / 'config.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        self.config = PhotozRFConfig()
        self.config.update_from_yaml(str(cfg_path))

        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler_X = joblib.load(scaler_path)
        else:
            self.scaler_X = None

        model_path = self.model_dir / 'rf_model.joblib'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.rf = joblib.load(model_path)

        self.features_names = choose_features(
            feature_generation=self.config.dataset_params.get('feature_generation'),
            feature_columns=self.config.dataset_params.get('feature_columns')
        )

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Get features from dataframe, applying scaler if available."""
        features = df[self.features_names].values
        if self.scaler_X is not None:
            features = self.scaler_X.transform(features)
        return features

    def predict(self, data_source, return_uncertainty=True):
        """Predict photometric redshifts for new data.

        Args:
            data_source: :class:`pandas.DataFrame` already containing the
                catalog columns, or a path to a file readable by
                :func:`load_datasets`.
            return_uncertainty: Whether to compute uncertainty from the
                per-tree prediction variance.

        Returns:
            Tuple ``(phot_z, phot_z_err)``. ``phot_z_err`` is ``None`` when
            ``return_uncertainty`` is ``False``.
        """
        if isinstance(data_source, str):
            df = load_datasets(data_source)
        else:
            df = data_source

        X_test = self._get_features(df)

        if return_uncertainty:
            predictions = np.array([tree.predict(X_test) for tree in self.rf.estimators_])
            phot_z = np.mean(predictions, axis=0)
            phot_z_err = np.std(predictions, axis=0)
        else:
            phot_z = self.rf.predict(X_test)
            phot_z_err = None

        return phot_z, phot_z_err


# ============================================================
# Example Usage
# ============================================================
if __name__ == '__main__':
    # Minimal end-to-end example using three pre-split FITS files.
    # Replace the placeholder paths below with your own splits.
    config = PhotozRFConfig(
        experiment_name="example_rf",
        save_dir="./experiments",
        random_seed=42,
        use_scaler=True,
        dataset_mode={
            'type': 'files',
            'paths': {
                'train': "/path/to/train.fits",
                'val': "/path/to/val.fits",
                'test': "/path/to/test.fits",
            }
        },
        # dataset_params defaults to PS1-style features (Kron/PSF/Ap mags +
        # Kron/Ap colors). Supply ``feature_columns`` instead for surveys
        # whose catalogs use different column names, e.g.::
        #     dataset_params={'feature_columns': ['dered_mag_g', ...]}
        model_params={
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
        },
    )

    rf = PhotozRandomForest(config=config)
    rf.fit()
    rf.evaluate()
    rf.plot_feature_importance()

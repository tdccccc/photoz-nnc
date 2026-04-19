import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

import cosmic.utils as cu
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import yaml
from pathlib import Path
import joblib

#============================================================
# Config
#============================================================
@dataclass
class GalaxyXGBConfig:
    # Experiment setup
    config_path: str = "config.yaml"
    save_dir: str = "."
    experiment_name: str = "galaxy_classification_xgb"
    random_seed: int = 2025

    # Dataset configuration
    dataset_params: Dict[str, Any] = field(default_factory=lambda: {
        'feature_generation': {
            'mag_types': ['Kron', 'PSF', 'Ap'],
            'color_types': ['Kron', 'Ap'],
            'use_mag': True,
            'use_magerr': True,
            'use_colorerr': False
        },
        'feature_columns': None,  # Alternative: specify columns directly
        'label_column': 'label'   # Column name for classification labels (galaxy=1, others=0)
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

    # XGBoost parameters for classification
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_child_weight': 1,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'scale_pos_weight': 1.0  # For imbalanced datasets
    })

    # Feature scaling
    use_scaler: bool = True

    # Output settings
    save_model: bool = True
    save_results: bool = True
    plot_results: bool = True

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

    def save_to_yaml(self, path: str = None):
        """Saves configuration to a YAML file."""
        if path is None:
            save_dir = Path(self.save_dir) / self.experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / "config.yaml"

        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to: {path}")

    def get_save_dir(self) -> Path:
        """Returns the experiment save directory."""
        save_dir = Path(self.save_dir) / self.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir


def load_datasets(path):
    df = cu.readfile(path)
    print(f"total: {df.shape[0]}")
    return df


def choose_features(feature_generation: dict = None,
                    feature_columns: list = None):
    """
    Generate feature column names based on configuration.

    Args:
        feature_generation: Dict with mag_types, color_types, use_mag, use_magerr, use_colorerr
        feature_columns: Direct list of column names (takes precedence if provided)

    Returns:
        List of feature column names
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

    # Add color features using color_types
    if color_types:
        for color_type in color_types:
            for colour in colours:
                mag_cols.append(f'{color_type}_{colour}')
                if use_colorerr:
                    err_cols.append(f'{color_type}_{colour}_Err')

    # Add magnitude features using mag_types
    if use_mag and mag_types:
        for band in bands:
            for mag_type in mag_types:
                mag_cols.append(f'{band}{mag_type}Mag_dered')
                if use_magerr:
                    err_cols.append(f'{band}{mag_type}MagErr')

    combined_cols = mag_cols + err_cols
    return combined_cols


class GalaxyXGBClassifier:
    def __init__(self, config: GalaxyXGBConfig):
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaler_X = None

        # Get label column name
        self.label_column = config.dataset_params.get('label_column', 'label')

        # Generate features based on config
        dataset_params = config.dataset_params
        self.features_names = choose_features(
            feature_generation=dataset_params.get('feature_generation'),
            feature_columns=dataset_params.get('feature_columns')
        )
        self.n_features = len(self.features_names)

        # Initialize XGBoost Classifier with config parameters
        model_params = config.model_params
        self.early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
        self.xgb_params = {
            'n_estimators': model_params.get('n_estimators', 100),
            'max_depth': model_params.get('max_depth', 6),
            'learning_rate': model_params.get('learning_rate', 0.1),
            'subsample': model_params.get('subsample', 1.0),
            'colsample_bytree': model_params.get('colsample_bytree', 1.0),
            'min_child_weight': model_params.get('min_child_weight', 1),
            'gamma': model_params.get('gamma', 0.0),
            'reg_alpha': model_params.get('reg_alpha', 0.0),
            'reg_lambda': model_params.get('reg_lambda', 1.0),
            'objective': model_params.get('objective', 'binary:logistic'),
            'eval_metric': model_params.get('eval_metric', 'logloss'),
            'scale_pos_weight': model_params.get('scale_pos_weight', 1.0),
            'random_state': config.random_seed,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': self.early_stopping_rounds,
        }

        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.is_trained = False

        # Load data based on config
        self._setup_data()
        self._print_params()

    def _print_params(self):
        print("=" * 60)
        print("XGBoost Classification Model Parameters")
        print("=" * 60)
        print(f"{'experiment_name':18s}: {self.config.experiment_name}")
        print(f"{'save_dir':18s}: {self.config.save_dir}")
        print(f"{'label_column':18s}: {self.label_column}")

        model_params = self.config.model_params
        print(f"{'n_estimators':18s}: {model_params.get('n_estimators', 100)}")
        print(f"{'max_depth':18s}: {model_params.get('max_depth', 6)}")
        print(f"{'learning_rate':18s}: {model_params.get('learning_rate', 0.1)}")
        print(f"{'subsample':18s}: {model_params.get('subsample', 1.0)}")
        print(f"{'colsample_bytree':18s}: {model_params.get('colsample_bytree', 1.0)}")
        print(f"{'min_child_weight':18s}: {model_params.get('min_child_weight', 1)}")
        print(f"{'gamma':18s}: {model_params.get('gamma', 0.0)}")
        print(f"{'reg_alpha':18s}: {model_params.get('reg_alpha', 0.0)}")
        print(f"{'reg_lambda':18s}: {model_params.get('reg_lambda', 1.0)}")
        print(f"{'scale_pos_weight':18s}: {model_params.get('scale_pos_weight', 1.0)}")
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
        """Load data from single file and split by ratios."""
        data_path = self.config.dataset_mode.get('data_path')
        if not data_path:
            raise ValueError("'data_path' must be specified when using 'ratio' mode")

        print(f"Loading data from: {data_path}")
        df = load_datasets(data_path)

        ratios = self.config.dataset_mode.get('ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
        train_ratio = ratios.get('train', 0.8)
        val_ratio = ratios.get('val', 0.1)
        test_ratio = ratios.get('test', 0.1)

        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        # First split: train vs (val + test)
        val_test_ratio = val_ratio + test_ratio
        self.train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=self.config.random_seed
        )

        # Second split: val vs test
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

    def _evaluate(self, y_true, y_pred_proba, threshold=0.5):
        """Evaluate classification performance."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_true_int = y_true.astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true_int, y_pred),
            'precision': precision_score(y_true_int, y_pred, zero_division=0),
            'recall': recall_score(y_true_int, y_pred, zero_division=0),
            'f1': f1_score(y_true_int, y_pred, zero_division=0),
        }

        # AUC-ROC
        try:
            metrics['auc'] = roc_auc_score(y_true_int, y_pred_proba)
        except ValueError:
            metrics['auc'] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred, labels=[0, 1]).ravel()
        metrics['tp'] = tp
        metrics['fp'] = fp
        metrics['tn'] = tn
        metrics['fn'] = fn

        return metrics

    def _print_metrics_comparison(self, train_metrics, test_metrics):
        print("\n" + "="*70)
        print(f"{'Metric':<15} {'Trainset':<12} {'Testset':<12}")
        print("="*70)
        print(f"{'Accuracy':<15} {train_metrics['accuracy']:<12.6f} {test_metrics['accuracy']:<12.6f}")
        print(f"{'Precision':<15} {train_metrics['precision']:<12.6f} {test_metrics['precision']:<12.6f}")
        print(f"{'Recall':<15} {train_metrics['recall']:<12.6f} {test_metrics['recall']:<12.6f}")
        print(f"{'F1 Score':<15} {train_metrics['f1']:<12.6f} {test_metrics['f1']:<12.6f}")
        print(f"{'AUC-ROC':<15} {train_metrics['auc']:<12.6f} {test_metrics['auc']:<12.6f}")
        print("-"*70)
        print(f"{'TP':<15} {train_metrics['tp']:<12} {test_metrics['tp']:<12}")
        print(f"{'FP':<15} {train_metrics['fp']:<12} {test_metrics['fp']:<12}")
        print(f"{'TN':<15} {train_metrics['tn']:<12} {test_metrics['tn']:<12}")
        print(f"{'FN':<15} {train_metrics['fn']:<12} {test_metrics['fn']:<12}")
        print("="*70)

    def _print_metrics_comparison_with_val(self, train_metrics, val_metrics, test_metrics):
        """Print metrics comparison including validation set."""
        print("\n" + "="*90)
        print(f"{'Metric':<15} {'Trainset':<15} {'Validation':<15} {'Testset':<15}")
        print("="*90)
        print(f"{'Accuracy':<15} {train_metrics['accuracy']:<15.6f} {val_metrics['accuracy']:<15.6f} {test_metrics['accuracy']:<15.6f}")
        print(f"{'Precision':<15} {train_metrics['precision']:<15.6f} {val_metrics['precision']:<15.6f} {test_metrics['precision']:<15.6f}")
        print(f"{'Recall':<15} {train_metrics['recall']:<15.6f} {val_metrics['recall']:<15.6f} {test_metrics['recall']:<15.6f}")
        print(f"{'F1 Score':<15} {train_metrics['f1']:<15.6f} {val_metrics['f1']:<15.6f} {test_metrics['f1']:<15.6f}")
        print(f"{'AUC-ROC':<15} {train_metrics['auc']:<15.6f} {val_metrics['auc']:<15.6f} {test_metrics['auc']:<15.6f}")
        print("-"*90)
        print(f"{'TP':<15} {train_metrics['tp']:<15} {val_metrics['tp']:<15} {test_metrics['tp']:<15}")
        print(f"{'FP':<15} {train_metrics['fp']:<15} {val_metrics['fp']:<15} {test_metrics['fp']:<15}")
        print(f"{'TN':<15} {train_metrics['tn']:<15} {val_metrics['tn']:<15} {test_metrics['tn']:<15}")
        print(f"{'FN':<15} {train_metrics['fn']:<15} {val_metrics['fn']:<15} {test_metrics['fn']:<15}")
        print("="*90)

    def fit(self):
        """Fit the XGBoost classifier on training data."""
        X_train = self._get_features(self.train_df)
        y_train = self.train_df[self.label_column].values

        # Use validation set if pre-loaded, otherwise create one from training data
        if self.val_df is not None:
            X_val = self._get_features(self.val_df)
            y_val = self.val_df[self.label_column].values
            print("Using pre-loaded validation set for early stopping...")
        else:
            # Create validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.config.random_seed
            )
            print("Creating validation set from training data for early stopping...")

        print("fitting...")
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        self.is_trained = True
        print("done!")

        # Save model and config if configured
        if self.config.save_model:
            self.save_model()
            self.config.save_to_yaml()

    def evaluate(self):
        """Evaluate the classifier on all datasets."""
        # Evaluate on training set
        X_train = self._get_features(self.train_df)
        self.train_pred_proba = self.xgb_model.predict_proba(X_train)[:, 1]
        self.train_pred = (self.train_pred_proba >= 0.5).astype(int)
        self.train_df['pred_proba'] = self.train_pred_proba
        self.train_df['pred'] = self.train_pred
        self.train_metrics = self._evaluate(self.train_df[self.label_column].values, self.train_pred_proba)

        # Evaluate on validation set if available
        if self.val_df is not None:
            X_val = self._get_features(self.val_df)
            self.val_pred_proba = self.xgb_model.predict_proba(X_val)[:, 1]
            self.val_pred = (self.val_pred_proba >= 0.5).astype(int)
            self.val_df['pred_proba'] = self.val_pred_proba
            self.val_df['pred'] = self.val_pred
            self.val_metrics = self._evaluate(self.val_df[self.label_column].values, self.val_pred_proba)

        # Evaluate on test set
        X_test = self._get_features(self.test_df)
        self.test_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        self.test_pred = (self.test_pred_proba >= 0.5).astype(int)
        self.test_df['pred_proba'] = self.test_pred_proba
        self.test_df['pred'] = self.test_pred
        self.test_metrics = self._evaluate(self.test_df[self.label_column].values, self.test_pred_proba)

        # Print metrics comparison
        if self.val_df is not None:
            self._print_metrics_comparison_with_val(self.train_metrics, self.val_metrics, self.test_metrics)
        else:
            self._print_metrics_comparison(self.train_metrics, self.test_metrics)

        if self.config.plot_results:
            self._plot_results()

        if self.config.save_results:
            self.save_results()

    def _plot_results(self):
        """Plot classification results: confusion matrix and ROC curve."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Training set confusion matrix
        ax = axes[0, 0]
        cm_train = confusion_matrix(self.train_df[self.label_column].values.astype(int),
                                    self.train_pred, labels=[0, 1])
        im = ax.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Training Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Other (0)', 'Galaxy (1)'])
        ax.set_yticklabels(['Other (0)', 'Galaxy (1)'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm_train[i, j], 'd'),
                       ha="center", va="center", color="white" if cm_train[i, j] > cm_train.max()/2 else "black")

        # Test set confusion matrix
        ax = axes[0, 1]
        cm_test = confusion_matrix(self.test_df[self.label_column].values.astype(int),
                                   self.test_pred, labels=[0, 1])
        im = ax.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Test Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Other (0)', 'Galaxy (1)'])
        ax.set_yticklabels(['Other (0)', 'Galaxy (1)'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm_test[i, j], 'd'),
                       ha="center", va="center", color="white" if cm_test[i, j] > cm_test.max()/2 else "black")

        # ROC curves
        ax = axes[1, 0]
        # Training ROC
        fpr_train, tpr_train, _ = roc_curve(self.train_df[self.label_column].values.astype(int), self.train_pred_proba)
        ax.plot(fpr_train, tpr_train, 'b-', label=f'Train (AUC = {self.train_metrics["auc"]:.4f})')
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(self.test_df[self.label_column].values.astype(int), self.test_pred_proba)
        ax.plot(fpr_test, tpr_test, 'r-', label=f'Test (AUC = {self.test_metrics["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True)

        # Probability distribution
        ax = axes[1, 1]
        # Training set probability distribution
        ax.hist(self.train_pred_proba[self.train_df[self.label_column].values == 0], bins=50, alpha=0.5, label='Train Other (0)', color='blue')
        ax.hist(self.train_pred_proba[self.train_df[self.label_column].values == 1], bins=50, alpha=0.5, label='Train Galaxy (1)', color='red')
        ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title('Probability Distribution (Training Set)')
        ax.legend()

        plt.tight_layout()

        save_dir = self.config.get_save_dir()
        plot_path = save_dir / f'results.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Results plot saved to: {plot_path}")
        plt.close()

    def save_model(self):
        """Save the trained model."""
        if not self.is_trained:
            print("Model not trained yet. Cannot save.")
            return

        save_dir = self.config.get_save_dir()
        model_path = save_dir / "xgb_model.json"
        self.xgb_model.save_model(model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path: str = None):
        """Load a trained model and scaler."""
        if model_path is None:
            save_dir = self.config.get_save_dir()
            model_path = save_dir / "xgb_model.json"

        if os.path.exists(model_path):
            self.xgb_model.load_model(model_path)
            self.is_trained = True
            print(f"Model loaded from: {model_path}")

            # Also load scaler if exists
            scaler_path = Path(model_path).parent / "scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler_X = joblib.load(scaler_path)
                print(f"Scaler loaded from: {scaler_path}")
        else:
            print(f"Model file not found: {model_path}")

    def save_results(self):
        """Save training and test results."""
        save_dir = self.config.get_save_dir()

        # Save full results with all columns
        train_results_path = save_dir / "train_results.fits"
        cu.savefile(self.train_df, train_results_path)

        # Save validation results if available
        if hasattr(self, 'val_df') and self.val_df is not None:
            val_results_path = save_dir / "val_results.fits"
            cu.savefile(self.val_df, val_results_path)

        test_results_path = save_dir / "test_results.fits"
        cu.savefile(self.test_df, test_results_path)

        # Save prediction-only files
        self._save_predictions()

        metrics_path = save_dir / "metrics.txt"
        # Write metrics in a more readable format
        with open(metrics_path, 'w') as f:
            f.write("Train Metrics:\n")
            for k, v in self.train_metrics.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.6f}\n")
                else:
                    f.write(f"  {k}: {v}\n")

            # Write validation metrics if available
            if hasattr(self, 'val_metrics'):
                f.write("\nValidation Metrics:\n")
                for k, v in self.val_metrics.items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {v:.6f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")

            f.write("\nTest Metrics:\n")
            for k, v in self.test_metrics.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.6f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
        print(f"Results saved to: {save_dir}")

    def _save_predictions(self):
        """Save prediction results (label, pred, pred_proba) to separate files."""
        save_dir = self.config.get_save_dir()

        # Create prediction DataFrames
        train_pred_df = pd.DataFrame({
            self.label_column: self.train_df[self.label_column].values,
            'pred': self.train_pred,
            'pred_proba': self.train_pred_proba
        })
        cu.savefile(train_pred_df, save_dir / "predictions_train.fits")

        if self.val_df is not None:
            val_pred_df = pd.DataFrame({
                self.label_column: self.val_df[self.label_column].values,
                'pred': self.val_pred,
                'pred_proba': self.val_pred_proba
            })
            cu.savefile(val_pred_df, save_dir / "predictions_val.fits")

        test_pred_df = pd.DataFrame({
            self.label_column: self.test_df[self.label_column].values,
            'pred': self.test_pred,
            'pred_proba': self.test_pred_proba
        })
        cu.savefile(test_pred_df, save_dir / "predictions_test.fits")

        print(f"Predictions saved to: {save_dir}/predictions_*.fits")

    def predict(self, data_source, threshold=0.5):
        """
        Predict galaxy classification for new data.

        Args:
            data_source: DataFrame, file path, or numpy array of features
            threshold: Classification threshold (default 0.5)

        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probability of class 1 (galaxy)
        """
        if not self.is_trained:
            raise ValueError("Model not trained, please call fit() method")

        # Get features from different data sources
        if isinstance(data_source, pd.DataFrame):
            X_test = self._get_features(data_source)
        elif isinstance(data_source, str):
            # File path
            df = load_datasets(data_source)
            X_test = self._get_features(df)
        elif isinstance(data_source, np.ndarray):
            X_test = data_source
            if self.config.use_scaler and self.scaler_X is not None:
                X_test = self.scaler_X.transform(X_test)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        probabilities = self.xgb_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        return predictions, probabilities

    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model not trained, please call fit() method")

        importances = self.xgb_model.feature_importances_

        # Print feature importance
        print("\nFeature Importance:")
        for i, (name, importance) in enumerate(zip(self.features_names, importances)):
            print(f"{i+1:2d}. {name:15s}: {importance:.4f}")

        return importances, self.features_names

    def plot_feature_importance(self):
        importances, feature_names = self.get_feature_importance()

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("XGBoost Feature Importance")
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

    def plot_training_history(self):
        """Plot training history if available."""
        if not self.is_trained:
            print("Model not trained yet.")
            return

        # Get evaluation results from XGBoost
        evals_result = self.xgb_model.evals_result()
        if evals_result:
            plt.figure(figsize=(10, 6))

            # Plot validation loss
            validation_0 = evals_result['validation_0']
            if 'logloss' in validation_0:
                plt.plot(validation_0['logloss'], label='Validation Log Loss')
                plt.xlabel('Boosting Round')
                plt.ylabel('Log Loss')
                plt.title('XGBoost Training History')
                plt.legend()
                plt.grid(True)

                if self.config.save_results:
                    save_dir = self.config.get_save_dir()
                    history_plot_path = save_dir / "training_history.png"
                    plt.savefig(history_plot_path, dpi=300)
                    print(f"Training history plot saved to: {history_plot_path}")

                plt.show()
        else:
            print("No training history available.")


# ============================================================
# Inference helper
# ============================================================
class GalaxyXGBInference:
    """Inference helper for trained XGBoost classification models."""

    def __init__(self, model_dir: str):
        """
        Initialize inference helper.

        Args:
            model_dir: Path to the model directory containing config.yaml and xgb_model.json
        """
        self.model_dir = Path(model_dir)

        # Load config
        cfg_path = self.model_dir / 'config.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        self.config = GalaxyXGBConfig()
        self.config.update_from_yaml(str(cfg_path))

        # Get label column name
        self.label_column = self.config.dataset_params.get('label_column', 'label')

        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler_X = joblib.load(scaler_path)
        else:
            self.scaler_X = None

        # Load model
        model_path = self.model_dir / 'xgb_model.json'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(model_path)

        # Get feature names
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

    def predict(self, data_source, threshold=0.5):
        """
        Predict galaxy classification for new data.

        Args:
            data_source: DataFrame or file path
            threshold: Classification threshold (default 0.5)

        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probability of class 1 (galaxy)
        """
        if isinstance(data_source, str):
            df = load_datasets(data_source)
        else:
            df = data_source

        X_test = self._get_features(df)
        probabilities = self.xgb_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        return predictions, probabilities


# ============================================================
# Example Usage
# ============================================================
if __name__ == '__main__':
    # Example 1: Using 'files' mode with separate train/val/test files
    config = GalaxyXGBConfig(
        experiment_name="2NoSpec_xSpec_NoErr",
        save_dir="./LSDR10/grzW1W2",
        random_seed=2025,
        use_scaler=False,
        dataset_mode={
            'type': 'files',
            'paths': {
                # PS1DR2
                # 'train': "/home/tiandc/photoz/galaxyClf/data/PS1DR2_r_trainset_train.fits",
                # 'val': "/home/tiandc/photoz/galaxyClf/data/PS1DR2_r_trainset_val.fits",
                # 'test': "/home/tiandc/photoz/galaxyClf/data/PS1DR2_r_trainset_test.fits",
                # LSDR10
                # 'train': "/home/tiandc/photoz/galaxyClf/data/LSDR10/NoSpec/LSDR10_NoSpec_train.fits",
                # 'val': "/home/tiandc/photoz/galaxyClf/data/LSDR10/NoSpec/LSDR10_NoSpec_val.fits",
                # 'test': "/home/tiandc/photoz/galaxyClf/data/LSDR10/NoSpec/LSDR10_NoSpec_test.fits",
                # LSDR10 NoSpec_xSpec
                'train': "/home/tiandc/photoz/galaxyClf/data/LSDR10/NoSpec_xSpec/LSDR10_NoSpec_xSpec_train.fits",
                'val': "/home/tiandc/photoz/galaxyClf/data/LSDR10/NoSpec_xSpec/LSDR10_NoSpec_xSpec_val.fits",
                'test': "/home/tiandc/photoz/galaxyClf/data/LSDR10/NoSpec_xSpec/LSDR10_NoSpec_xSpec_test.fits",
            }
        },
        dataset_params={
            'label_column': 'label',  # Column containing labels (galaxy=1, others=0)
            # PS1DR2
            # 'feature_columns': [
            #     # Magnitude features
            #     'gKronMag_dered', 'rKronMag_dered', 'iKronMag_dered', 'zKronMag_dered', 'yKronMag_dered',
            #     'gPSFMag_dered', 'rPSFMag_dered', 'iPSFMag_dered', 'zPSFMag_dered', 'yPSFMag_dered',
            #     'gApMag_dered', 'rApMag_dered', 'iApMag_dered', 'zApMag_dered', 'yApMag_dered',
            #     # Magnitude errors
            #     'gKronMagErr', 'rKronMagErr', 'iKronMagErr', 'zKronMagErr', 'yKronMagErr',
            #     'gPSFMagErr', 'rPSFMagErr', 'iPSFMagErr', 'zPSFMagErr', 'yPSFMagErr',
            #     'gApMagErr', 'rApMagErr', 'iApMagErr', 'zApMagErr', 'yApMagErr',
            # ]
            # LSDR10
            'feature_columns': [
                'dered_mag_g', 'dered_mag_r',
                'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2',
                # 'mag_g_Err', 'mag_r_Err',
                # 'mag_z_Err', 'mag_w1_Err', 'mag_w2_Err',
            ]
        },
        model_params={
            'n_estimators': 500,
            'max_depth': 12,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'early_stopping_rounds': 50,
            'scale_pos_weight': 1.0  # Adjust for imbalanced data
        }
    )

    xgb_model = GalaxyXGBClassifier(config=config)
    xgb_model.fit()
    xgb_model.evaluate()
    xgb_model.plot_feature_importance()
    xgb_model.plot_training_history()


# # Example 2: Using 'ratio' mode to split a single file
# config = GalaxyXGBConfig(
#     experiment_name="galaxy_clf_ratio_split",
#     save_dir=".",
#     random_seed=2025,
#     use_scaler=True,
#     dataset_mode={
#         'type': 'ratio',
#         'data_path': "../data/galaxy_data.fits",
#         'ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
#     },
#     dataset_params={
#         'label_column': 'label',
#         'feature_generation': {
#             'mag_types': ['Kron', 'PSF', 'Ap'],
#             'color_types': ['Kron', 'Ap'],
#             'use_mag': True,
#             'use_magerr': True,
#             'use_colorerr': False
#         }
#     },
#     model_params={
#         'n_estimators': 500,
#         'max_depth': 8,
#         'learning_rate': 0.1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss'
#     }
# )
#
# xgb_model = GalaxyXGBClassifier(config=config)
# xgb_model.fit()
# xgb_model.evaluate()
# xgb_model.plot_feature_importance()
# xgb_model.plot_training_history()


# # Example 3: Inference with a trained model
# inference = GalaxyXGBInference(model_dir="./galaxy_classification/galaxy_classification_xgb")
# predictions, probabilities = inference.predict("path/to/new_data.fits", threshold=0.5)

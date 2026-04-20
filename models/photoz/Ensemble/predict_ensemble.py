"""Ensemble photo-z inference for ANN, Random Forest, and XGBoost models.

This script loads one or more trained model directories, runs prediction on
the requested train / validation / test catalogs, collects the individual
model outputs, computes a mean ensemble prediction for each split, and
summarizes the redshift-quality metrics of both the individual models and
the ensemble result.

The workflow includes:

* loading saved model files, scalers, and YAML configs;
* building the dataset used for inference from catalog files;
* running split-by-split prediction for each enabled model;
* averaging the model predictions to form the ensemble output;
* computing summary metrics for each split;
* writing one output table per split to the experiment directory.
"""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
import yaml
from torch.utils.data import DataLoader

from lib.io import readfile, savefile
from lib.metrics import redshift_quality_metrics
from models.photoz.ANN.core import Config
from models.photoz.ANN.dataset import DatasetPhotozRegression
from models.photoz.ANN.model import PhotozRegressor

warnings.filterwarnings("ignore")


def inverse_transform_labels(
    transformed_labels: np.ndarray,
    transform_type: str,
    transform_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert transformed predictions back to the original redshift scale."""
    if transform_params is None:
        transform_params = {}

    values = np.asarray(transformed_labels)
    if transform_type == "none":
        return values
    if transform_type == "log1p":
        return np.expm1(values)
    if transform_type == "log":
        offset = transform_params.get("offset", 1e-8)
        return np.exp(values) - offset
    if transform_type == "sqrt":
        return np.square(values)
    if transform_type == "power":
        power = transform_params.get("power", 0.5)
        return np.power(values, 1.0 / power)
    if transform_type == "asinh":
        scale = transform_params.get("scale", 1.0)
        return np.sinh(values) * scale
    raise ValueError(f"Unsupported label transform: {transform_type!r}")


def evaluate_redshift_quality(
    z_phot: np.ndarray,
    z_spec: np.ndarray,
) -> dict[str, float]:
    """Return summary metrics for one set of photo-z predictions."""
    metrics = redshift_quality_metrics(z_phot, z_spec)
    return {
        "mean_bias": metrics["bias"],
        "std_dev": metrics["scatter"],
        "mad": metrics["nmad"],
        "outlier_fraction": metrics["outlier_fraction"],
    }


def _apply_output_activation(
    outputs: torch.Tensor,
    activation_name: str,
) -> torch.Tensor:
    """Apply the configured output activation to ANN predictions."""
    if activation_name == "none":
        return outputs
    if activation_name == "relu":
        return F.relu(outputs)
    if activation_name == "softplus":
        return F.softplus(outputs)
    if activation_name == "tanh":
        return torch.tanh(outputs)
    if activation_name == "sigmoid":
        return torch.sigmoid(outputs)
    if activation_name == "elu":
        return F.elu(outputs)
    if activation_name == "gelu":
        return F.gelu(outputs)
    if activation_name == "silu":
        return F.silu(outputs)
    if activation_name == "leaky_relu":
        return F.leaky_relu(outputs, negative_slope=0.01)
    raise ValueError(f"Unsupported output activation: {activation_name!r}")


def _filter_ann_model_params(model_params: dict[str, Any]) -> dict[str, Any]:
    """Keep only the ANN model parameters accepted by ``PhotozRegressor``."""
    accepted = set(inspect.signature(PhotozRegressor).parameters)
    filtered = {k: v for k, v in model_params.items() if k in accepted}
    ignored = sorted(set(model_params) - accepted - {"output_activation"})
    if ignored:
        warnings.warn(
            "Ignoring unsupported ANN model parameters from config: "
            + ", ".join(ignored)
        )
    return filtered


def _remap_ann_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename state-dict keys to match the ANN model definition used here."""
    remapped = dict(state_dict)
    if "output_layer.weight" in remapped:
        remapped["output.weight"] = remapped.pop("output_layer.weight")
    if "output_layer.bias" in remapped:
        remapped["output.bias"] = remapped.pop("output_layer.bias")
    return remapped


def _extract_batch_labels(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Read labels from the batch returned by the dataset."""
    labels = batch.get("target")
    if labels is None:
        labels = batch.get("label")
    if labels is None:
        raise KeyError("Batch does not contain 'target' or 'label'.")
    return labels


def _resolve_dataset_params(config: Any) -> dict[str, Any]:
    """Extract feature settings from the model config."""
    if hasattr(config, "dataset_params"):
        dataset_params = getattr(config, "dataset_params")
    elif isinstance(config, dict):
        dataset_params = config.get("dataset_params", config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    feature_generation = dataset_params.get("feature_generation")
    if feature_generation is None and any(
        key in dataset_params
        for key in ("mag_types", "color_types", "use_mag", "use_magerr", "use_colorerr")
    ):
        feature_generation = {
            "mag_types": dataset_params.get("mag_types"),
            "color_types": dataset_params.get("color_types"),
            "use_mag": dataset_params.get("use_mag", True),
            "use_magerr": dataset_params.get("use_magerr", True),
            "use_colorerr": dataset_params.get("use_colorerr", False),
        }

    return {
        "feature_generation": feature_generation,
        "feature_columns": dataset_params.get("feature_columns"),
    }


def create_common_dataset(
    file_path: str,
    mode: str,
    config: Any,
    scaler=None,
) -> DatasetPhotozRegression:
    """Build the dataset object used for ensemble inference."""
    # Read feature settings from the saved config.
    dataset_cfg = _resolve_dataset_params(config)
    dataset = readfile(file_path)
    return DatasetPhotozRegression(
        dataset=dataset,
        feature_generation=dataset_cfg["feature_generation"],
        feature_columns=dataset_cfg["feature_columns"],
        label_noise=None,
        scaler_X=scaler,
        mode=mode,
    )


class BasePhotozModel(ABC):
    """Abstract interface for one ensemble member."""

    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.config = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the trained model and any sidecar artifacts from disk."""

    @abstractmethod
    def predict_on_files(
        self,
        file_paths: dict[str, str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Predict on each requested data split."""

    def get_model_info(self) -> dict[str, Any]:
        """Return a minimal status summary for the model."""
        return {
            "name": self.model_name,
            "directory": str(self.model_dir),
            "loaded": self.model is not None,
        }


class ANNPhotozModel(BasePhotozModel):
    """ANN-based ensemble member backed by ``models.photoz.ANN``."""

    def __init__(self, model_name: str, model_dir: str):
        super().__init__(model_name, model_dir)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.output_activation = "none"
        self.label_transform = "none"
        self.label_transform_params: dict[str, Any] = {}
        self.raw_config: dict[str, Any] = {}

    def load_model(self) -> None:
        """Load ANN config, scaler, checkpoint, and transform settings."""
        config_path = self.model_dir / "config.yaml"
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"

        # Read model settings from the saved YAML file.
        with open(config_path, "r") as f:
            self.raw_config = yaml.safe_load(f) or {}

        self.config = Config()
        self.config.update_from_dict(self.raw_config)

        self.label_transform = self.raw_config.get("label_transform", "none")
        self.label_transform_params = self.raw_config.get(
            "label_transform_params", {}
        )

        self.scaler = joblib.load(str(scaler_path))
        if not hasattr(self.scaler, "n_features_in_"):
            raise ValueError("Cannot determine input dimension from scaler.")
        input_dim = self.scaler.n_features_in_

        model_params = _filter_ann_model_params(dict(self.config.model_params))
        self.output_activation = self.config.model_params.get(
            "output_activation",
            self.raw_config.get("output_activation", "none"),
        )
        model_params["input_dim"] = input_dim

        self.model = PhotozRegressor(**model_params)
        # Load the trained network weights from disk.
        state_dict = torch.load(
            str(model_path),
            map_location=self.device,
            weights_only=True,
        )
        state_dict = _remap_ann_state_dict(state_dict)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict,
            strict=False,
        )
        if missing_keys:
            warnings.warn(
                "ANN checkpoint is missing parameters after state-dict "
                f"mapping: {missing_keys}"
            )
        if unexpected_keys:
            warnings.warn(
                "ANN checkpoint contains unexpected parameters after "
                f"state-dict mapping: {unexpected_keys}"
            )

        self.model.to(self.device)
        self.model.eval()
        print(f"ANN model '{self.model_name}' loaded from {self.model_dir}")

    def predict_on_files(
        self,
        file_paths: dict[str, str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run ANN inference on each split and return predictions + labels."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\n=== {self.model_name} Model Prediction ===")
        batch_size = getattr(self.config, "batch_size", 4096)
        results: dict[str, dict[str, np.ndarray]] = {}

        for mode, file_path in file_paths.items():
            print(f"  Processing {mode} set: {file_path}")
            # Load one split at a time and collect predictions.
            dataset = create_common_dataset(file_path, mode, self.config, self.scaler)
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            predictions: list[float] = []
            labels: list[float] = []

            with torch.no_grad():
                for batch in data_loader:
                    inputs = batch["input"].to(self.device)
                    batch_labels = _extract_batch_labels(batch).to(self.device)
                    outputs = self.model(inputs)
                    outputs = _apply_output_activation(
                        outputs,
                        self.output_activation,
                    )
                    predictions.extend(outputs.cpu().numpy().flatten())
                    labels.extend(batch_labels.cpu().numpy().flatten())

            predictions_arr = np.asarray(predictions)
            labels_arr = np.asarray(labels)
            if self.label_transform != "none":
                predictions_arr = inverse_transform_labels(
                    predictions_arr,
                    self.label_transform,
                    self.label_transform_params,
                )

            results[mode] = {
                "predictions": predictions_arr,
                "labels": labels_arr,
            }
            print(f"    Completed: {len(predictions_arr)} samples")

        return results


class RFPhotozModel(BasePhotozModel):
    """Random-Forest ensemble member."""

    def load_model(self) -> None:
        """Load the serialized Random Forest and its YAML config."""
        config_path = self.model_dir / "config.yaml"
        model_path = self.model_dir / "rf_model.joblib"

        # Read the saved model settings and estimator.
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model = joblib.load(str(model_path))
        print(f"Random Forest model '{self.model_name}' loaded from {self.model_dir}")

    def predict_on_files(
        self,
        file_paths: dict[str, str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run Random-Forest inference on each split."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\n=== {self.model_name} Model Prediction ===")
        batch_size = self.config.get("batch_size", 4096)
        results: dict[str, dict[str, np.ndarray]] = {}

        for mode, file_path in file_paths.items():
            print(f"  Processing {mode} set: {file_path}")
            # Load one split at a time and collect predictions.
            dataset = create_common_dataset(file_path, mode, self.config, scaler=None)
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            features_list: list[np.ndarray] = []
            labels_list: list[np.ndarray] = []

            for batch in data_loader:
                features_list.append(batch["input"].numpy())
                labels_list.append(_extract_batch_labels(batch).numpy())

            features = np.vstack(features_list)
            labels = np.concatenate(labels_list)
            predictions = self.model.predict(features)

            results[mode] = {
                "predictions": predictions,
                "labels": labels,
            }
            print(f"    Completed: {len(predictions)} samples")

        return results


class XGBPhotozModel(BasePhotozModel):
    """XGBoost ensemble member."""

    def load_model(self) -> None:
        """Load the serialized XGBoost booster and its YAML config."""
        config_path = self.model_dir / "config.yaml"
        model_path = self.model_dir / "xgb_model.json"

        # Read the saved model settings and booster.
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        print(f"XGBoost model '{self.model_name}' loaded from {self.model_dir}")

    def predict_on_files(
        self,
        file_paths: dict[str, str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run XGBoost inference on each split."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"\n=== {self.model_name} Model Prediction ===")
        batch_size = self.config.get("batch_size", 4096)
        results: dict[str, dict[str, np.ndarray]] = {}

        for mode, file_path in file_paths.items():
            print(f"  Processing {mode} set: {file_path}")
            # Load one split at a time and collect predictions.
            dataset = create_common_dataset(file_path, mode, self.config, scaler=None)
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            features_list: list[np.ndarray] = []
            labels_list: list[np.ndarray] = []

            for batch in data_loader:
                features_list.append(batch["input"].numpy())
                labels_list.append(_extract_batch_labels(batch).numpy())

            features = np.vstack(features_list)
            labels = np.concatenate(labels_list)
            dtest = xgb.DMatrix(features)
            predictions = self.model.predict(dtest)

            results[mode] = {
                "predictions": predictions,
                "labels": labels,
            }
            print(f"    Completed: {len(predictions)} samples")

        return results


class EnsemblePhotozPredictor:
    """Manage model loading, per-model inference, and ensemble averaging."""

    def __init__(self, experiment_dir: str = "./ensemble_results"):
        self.models: list[BasePhotozModel] = []
        self.experiment_dir = experiment_dir
        Path(self.experiment_dir).mkdir(exist_ok=True)

    def add_model(self, model: BasePhotozModel) -> None:
        """Register one model in the ensemble."""
        self.models.append(model)
        print(f"Added model: {model.model_name}")

    def load_all_models(self) -> None:
        """Load every registered model and continue past individual failures."""
        print("Loading all models...")
        for model in self.models:
            try:
                model.load_model()
            except Exception as exc:
                print(f"Error loading model {model.model_name}: {exc}")

    def predict_ensemble(
        self,
        file_paths: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        """Run all models and compute the mean ensemble prediction per split."""
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")

        print("=" * 60)
        print("ENSEMBLE PHOTOMETRIC REDSHIFT PREDICTION")
        print("=" * 60)

        all_model_results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for model in self.models:
            print(f"\n--- Running {model.model_name} Model ---")
            try:
                model_results = model.predict_on_files(file_paths)
                all_model_results[model.model_name] = model_results
            except Exception as exc:
                print(f"Error with {model.model_name}: {exc}")
                continue

        ensemble_results: dict[str, dict[str, Any]] = {}
        for split_name in file_paths:
            print(f"\n--- Processing {split_name} ensemble results ---")

            split_results: dict[str, Any] = {
                "labels": None,
                "individual_predictions": {},
                "ensemble_prediction": None,
                "individual_metrics": {},
                "ensemble_metrics": {},
            }
            valid_predictions: list[np.ndarray] = []

            for model_name, model_results in all_model_results.items():
                if split_name not in model_results:
                    continue

                predictions = np.asarray(
                    model_results[split_name]["predictions"]
                ).flatten()
                labels = np.asarray(model_results[split_name]["labels"]).flatten()

                split_results["individual_predictions"][model_name] = predictions
                split_results["individual_metrics"][model_name] = (
                    evaluate_redshift_quality(predictions, labels)
                )
                valid_predictions.append(predictions)

                if split_results["labels"] is None:
                    split_results["labels"] = labels

                print(f"  {model_name}: {len(predictions)} predictions")

            if valid_predictions:
                # Average model predictions to form the ensemble output.
                predictions_array = np.asarray(valid_predictions)
                ensemble_pred = np.mean(predictions_array, axis=0).flatten()
                ensemble_labels = np.asarray(split_results["labels"]).flatten()

                split_results["ensemble_prediction"] = ensemble_pred
                split_results["labels"] = ensemble_labels
                split_results["ensemble_metrics"] = evaluate_redshift_quality(
                    ensemble_pred,
                    ensemble_labels,
                )

                print(
                    "  Ensemble: "
                    f"{len(ensemble_pred)} predictions "
                    f"(mean of {len(valid_predictions)} models)"
                )

            ensemble_results[split_name] = split_results

        return ensemble_results

    def save_results(self, results: dict[str, dict[str, Any]]) -> None:
        """Write one prediction table per split to the experiment directory."""
        print(f"\nSaving results to {self.experiment_dir}...")

        for split_name, split_results in results.items():
            if split_results["labels"] is None:
                continue

            # Flatten arrays before computing metrics and writing tables.
            data_dict = {
                "label": np.asarray(split_results["labels"]).flatten(),
                "ensemble_pred": np.asarray(
                    split_results["ensemble_prediction"]
                ).flatten(),
            }
            for model_name, predictions in split_results[
                "individual_predictions"
            ].items():
                data_dict[f"{model_name}_pred"] = np.asarray(predictions).flatten()

            df = pd.DataFrame(data_dict)
            save_path = Path(self.experiment_dir) / f"predictions_{split_name}.fits"
            savefile(df, str(save_path))

        print("Results saved successfully!")

    def print_metrics_summary(self, results: dict[str, dict[str, Any]]) -> None:
        """Print per-split metrics for each model and for the ensemble."""
        print("\n" + "=" * 80)
        print("ENSEMBLE PREDICTION RESULTS")
        print("=" * 80)

        for split_name, split_results in results.items():
            if split_results["labels"] is None:
                continue

            print(f"\n{split_name.upper()} SET RESULTS:")
            print("-" * 60)

            for model_name, metrics in split_results["individual_metrics"].items():
                print(
                    f"{model_name:<20} | "
                    f"bias: {metrics['mean_bias']:<8.5f} | "
                    f"std: {metrics['std_dev']:<8.5f} | "
                    f"mad: {metrics['mad']:<8.5f} | "
                    f"outliers: {metrics['outlier_fraction']:<8.5f}"
                )

            if split_results["ensemble_metrics"]:
                metrics = split_results["ensemble_metrics"]
                print(
                    f"{'ENSEMBLE':<20} | "
                    f"bias: {metrics['mean_bias']:<8.5f} | "
                    f"std: {metrics['std_dev']:<8.5f} | "
                    f"mad: {metrics['mad']:<8.5f} | "
                    f"outliers: {metrics['outlier_fraction']:<8.5f}"
                )

        print("=" * 80)


def get_dataset_paths() -> dict[str, str]:
    """Return the train / validation / test catalog paths."""
    return {
        "train": "../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_train.fits",
        "validation": "../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_val.fits",
        "test": "../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_test.fits",
    }


def get_model_paths() -> dict[str, str]:
    """Return the trained model directories used by the ensemble script."""
    return {
        "ann": "/home/tiandc/panstarrs/panstarrs_desi/photoz_ann/1SameValTest_chi2>25",
        "rf": "/home/tiandc/panstarrs/panstarrs_desi/photoz_rf/SameValTest_chi2>25",
        "xgb": "/home/tiandc/panstarrs/panstarrs_desi/photoz_xgb/SameValTest_chi2>25",
    }


def get_active_models() -> list[str]:
    """Return the list of ensemble members enabled by default."""
    return ["rf", "xgb"]
    # return ["ann", "rf"]
    # return ["ann", "xgb"]
    # return ["ann", "rf", "xgb"]


def main() -> dict[str, dict[str, Any]]:
    """Run ensemble inference, persist outputs, and print the summary."""
    experiment_dir = "./rf_xgb"
    ensemble = EnsemblePhotozPredictor(experiment_dir)

    model_paths = get_model_paths()
    file_paths = get_dataset_paths()
    active_models = get_active_models()

    model_classes = {
        # "ann": ANNPhotozModel,
        "rf": RFPhotozModel,
        "xgb": XGBPhotozModel,
    }
    model_names = {
        # "ann": "ANN",
        "rf": "RandomForest",
        "xgb": "XGBoost",
    }

    print(f"Active models: {active_models}")
    for model_key in active_models:
        if model_key in model_classes:
            model_class = model_classes[model_key]
            model_name = model_names[model_key]
            model_path = model_paths[model_key]
            ensemble.add_model(model_class(model_name, model_path))
        else:
            print(f"Warning: Unknown model key '{model_key}' ignored")

    ensemble.load_all_models()

    print("Checking dataset file paths:")
    for split_name, file_path in file_paths.items():
        if not Path(file_path).exists():
            print(f"Warning: {split_name} file not found at {file_path}")
        else:
            print(f"  [ok] {split_name}: {file_path}")

    results = ensemble.predict_ensemble(file_paths)
    ensemble.save_results(results)
    ensemble.print_metrics_summary(results)
    return results


if __name__ == "__main__":
    results = main()

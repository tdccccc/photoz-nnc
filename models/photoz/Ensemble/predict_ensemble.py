import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from pathlib import Path
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import cosmic.utils as cu
from cosmic.panstarrs_dr2.photoz_ann import Dataset_Photz, DeepPhotozNet, Config

#============================================================
# Base Model Interface
#============================================================
class BasePhotozModel(ABC):
    """Base class for all photometric redshift models"""
    
    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.config = None
        
    @abstractmethod
    def load_model(self):
        """Load the trained model from disk"""
        pass
    
    @abstractmethod
    def predict_on_files(self, file_paths: Dict[str, str]) -> Dict[str, Dict]:
        """Load data and make predictions for all splits
        
        Args:
            file_paths: Dict with keys 'train', 'val', 'test' and file paths as values
            
        Returns:
            Dict with structure:
            {
                'train': {'predictions': np.ndarray, 'labels': np.ndarray},
                'val': {'predictions': np.ndarray, 'labels': np.ndarray}, 
                'test': {'predictions': np.ndarray, 'labels': np.ndarray}
            }
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'directory': str(self.model_dir),
            'loaded': self.model is not None
        }

#============================================================
# ANN Model Implementation
#============================================================
class ANNPhotozModel(BasePhotozModel):
    """Neural Network model for photometric redshift prediction"""
    
    def __init__(self, model_name: str, model_dir: str):
        super().__init__(model_name, model_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load ANN model, scaler, and config"""
        config_path = self.model_dir / "config.yaml"
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        # Load config
        self.config = Config()
        self.config.update_from_yaml(str(config_path))
        
        # Load scaler
        self.scaler = joblib.load(str(scaler_path))
        
        # Determine input dimension
        if hasattr(self.scaler, 'n_features_in_'):
            input_dim = self.scaler.n_features_in_
        else:
            raise ValueError("Cannot determine input dimension from scaler")
        
        # Load model
        model_params = self.config.model_params.copy()
        model_params['input_dim'] = input_dim
        
        self.model = DeepPhotozNet(**model_params)
        self.model.load_state_dict(torch.load(str(model_path), 
                                             map_location=self.device, 
                                             weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"ANN model '{self.model_name}' loaded from {self.model_dir}")
    
    def predict_on_files(self, file_paths: Dict[str, str]) -> Dict[str, Dict]:
        """ANN模型的完整预测流程"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\n=== {self.model_name} Model Prediction ===")
        
        # 从配置中获取数据参数
        config_dict = {
            'mag_types': self.config.dataset_params.get('mag_types'),
            'color_types': self.config.dataset_params.get('color_types'),
            'use_mag': self.config.dataset_params.get('use_mag'),
            'use_magerr': self.config.dataset_params.get('use_magerr'),
            'use_colorerr': self.config.dataset_params.get('use_colorerr')
        }
        batch_size = getattr(self.config, 'batch_size', 4096)
        
        results = {}
        
        for mode, file_path in file_paths.items():
            print(f"  Processing {mode} set: {file_path}")
            
            # 使用通用数据集创建函数（ANN使用scaler）
            dataset = create_common_dataset(file_path, mode, config_dict, self.scaler)
            
            # 创建数据加载器
            data_loader = DataLoader(dataset, 
                                   batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
                                   pin_memory=True)
            
            # 进行预测
            predictions = []
            labels = []
            
            with torch.no_grad():
                for batch in data_loader:
                    inputs = batch['input'].to(self.device)
                    batch_labels = batch['label'].to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    predictions.extend(outputs.cpu().numpy().flatten())
                    labels.extend(batch_labels.cpu().numpy().flatten())
            
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            # 应用逆变换
            if self.config.label_transform != 'none':
                predictions = Dataset_Photz.inverse_transform_labels(
                    predictions, 
                    self.config.label_transform, 
                    self.config.label_transform_params
                )
            
            results[mode] = {
                'predictions': predictions,
                'labels': labels
            }
            
            print(f"    Completed: {len(predictions)} samples")
        
        return results

#============================================================
# Random Forest Model Implementation
#============================================================
class RFPhotozModel(BasePhotozModel):
    """Random Forest model for photometric redshift prediction"""
    
    def load_model(self):
        """Load RF model and config"""
        config_path = self.model_dir / "config.yaml"
        model_path = self.model_dir / "rf_model.joblib"
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load model
        self.model = joblib.load(str(model_path))
        
        print(f"Random Forest model '{self.model_name}' loaded from {self.model_dir}")
    
    def predict_on_files(self, file_paths: Dict[str, str]) -> Dict[str, Dict]:
        """随机森林模型的完整预测流程"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\n=== {self.model_name} Model Prediction ===")
        
        results = {}
        
        for mode, file_path in file_paths.items():
            print(f"  Processing {mode} set: {file_path}")
            
            # 使用通用数据集创建函数（RF不使用scaler）
            dataset = create_common_dataset(file_path, mode, self.config, scaler=None)
            
            # 创建数据加载器
            data_loader = DataLoader(dataset, 
                                   batch_size=4096, 
        shuffle=False, 
        num_workers=4, 
                                   pin_memory=True)
            
            # 提取特征和标签
            features_list = []
            labels_list = []
            
            for batch in data_loader:
                features_list.append(batch['input'].numpy())
                labels_list.append(batch['label'].numpy())
                
            features = np.vstack(features_list)
            labels = np.concatenate(labels_list)
            
            # 进行预测
            predictions = self.model.predict(features)
            
            results[mode] = {
                'predictions': predictions,
                'labels': labels
            }
            
            print(f"    Completed: {len(predictions)} samples")
        
        return results

#============================================================
# XGBoost Model Implementation
#============================================================
class XGBPhotozModel(BasePhotozModel):
    """XGBoost model for photometric redshift prediction"""
    
    def load_model(self):
        """Load XGB model and config"""
        config_path = self.model_dir / "config.yaml"
        model_path = self.model_dir / "xgb_model.json"
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        print(f"XGBoost model '{self.model_name}' loaded from {self.model_dir}")
    
    def predict_on_files(self, file_paths: Dict[str, str]) -> Dict[str, Dict]:
        """XGBoost模型的完整预测流程"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\n=== {self.model_name} Model Prediction ===")
        
        results = {}
        
        for mode, file_path in file_paths.items():
            print(f"  Processing {mode} set: {file_path}")
            
            # 使用通用数据集创建函数（XGB不使用scaler）
            dataset = create_common_dataset(file_path, mode, self.config, scaler=None)
            
            # 创建数据加载器
            data_loader = DataLoader(dataset, 
                                   batch_size=4096, 
                                   shuffle=False, 
                                   num_workers=4, 
                                   pin_memory=True)
            
            # 提取特征和标签
            features_list = []
            labels_list = []
            
            for batch in data_loader:
                features_list.append(batch['input'].numpy())
                labels_list.append(batch['label'].numpy())
                
            features = np.vstack(features_list)
            labels = np.concatenate(labels_list)
            
            # 创建DMatrix并进行预测
            dtest = xgb.DMatrix(features)
            predictions = self.model.predict(dtest)
            
            results[mode] = {
                'predictions': predictions,
                'labels': labels
            }
            
            print(f"    Completed: {len(predictions)} samples")
        
        return results

#============================================================
# Ensemble Model Manager
#============================================================
class EnsemblePhotozPredictor:
    """Manager for ensemble photometric redshift prediction"""
    
    def __init__(self, experiment_dir: str = "./ensemble_results"):
        self.models: List[BasePhotozModel] = []
        self.experiment_dir = experiment_dir
        Path(self.experiment_dir).mkdir(exist_ok=True)
        
    def add_model(self, model: BasePhotozModel):
        """Add a model to the ensemble"""
        self.models.append(model)
        print(f"Added model: {model.model_name}")
        
    def load_all_models(self):
        """Load all models in the ensemble"""
        print("Loading all models...")
        for model in self.models:
            try:
                model.load_model()
            except Exception as e:
                print(f"Error loading model {model.model_name}: {e}")
                
    def predict_ensemble(self, file_paths: Dict[str, str]) -> Dict[str, Dict]:
        """使用所有模型进行集成预测
        
        Args:
            file_paths: Dict with keys 'train', 'val', 'test' and corresponding file paths
            
        Returns:
            Dict containing all predictions and ensemble results
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")
        
        print("="*60)
        print("ENSEMBLE PHOTOMETRIC REDSHIFT PREDICTION")
        print("="*60)
        
        # 收集所有模型的预测结果
        all_model_results = {}
        
        # 每个模型独立进行预测
        for model in self.models:
            print(f"\n--- Running {model.model_name} Model ---")
            try:
                model_results = model.predict_on_files(file_paths)
                all_model_results[model.model_name] = model_results
            except Exception as e:
                print(f"Error with {model.model_name}: {e}")
                continue
        
        # 整合结果并计算集成预测
        ensemble_results = {}
        
        # 获取数据集分割名称
        split_names = list(file_paths.keys())
        
        for split_name in split_names:
            print(f"\n--- Processing {split_name} ensemble results ---")
            
            split_results = {
                'labels': None,
                'individual_predictions': {},
                'ensemble_prediction': None,
                'individual_metrics': {},
                'ensemble_metrics': {}
            }
            
            # 收集各模型在当前分割上的预测结果
            valid_predictions = []
            
            for model_name, model_results in all_model_results.items():
                if split_name in model_results:
                    predictions = model_results[split_name]['predictions']
                    labels = model_results[split_name]['labels']
                    
                    split_results['individual_predictions'][model_name] = predictions
                    valid_predictions.append(predictions)
                    
                    if split_results['labels'] is None:
                        split_results['labels'] = labels
                    
                    # 确保数组是一维的
                    predictions = np.asarray(predictions).flatten()
                    labels = np.asarray(labels).flatten()

                    # 计算个别模型指标
                    metrics = cu.evaluate_redshift_quality(predictions, labels)  # z_phot, z_spec
                    split_results['individual_metrics'][model_name] = metrics
                    
                    print(f"  {model_name}: {len(predictions)} predictions")
            
            # 计算集成预测（均值）
            if valid_predictions:
                
                predictions_array = np.array(valid_predictions)
                
                split_results['ensemble_prediction'] = np.mean(predictions_array, axis=0)
                
                # 确保集成预测也是一维的并存储回结果
                ensemble_pred = np.asarray(split_results['ensemble_prediction']).flatten()
                ensemble_labels = np.asarray(split_results['labels']).flatten()
                
                # 更新结果为一维数组
                split_results['ensemble_prediction'] = ensemble_pred
                split_results['labels'] = ensemble_labels
                
                # 计算集成指标
                ensemble_metrics = cu.evaluate_redshift_quality(
                    ensemble_pred,    # z_phot
                    ensemble_labels   # z_spec
                )
                split_results['ensemble_metrics'] = ensemble_metrics
                
                print(f"  Ensemble: {len(split_results['ensemble_prediction'])} predictions (mean of {len(valid_predictions)} models)")
            
            ensemble_results[split_name] = split_results
        
        return ensemble_results
    
    def save_results(self, results: Dict[str, Dict]):
        """Save prediction results to files"""
        print(f"\nSaving results to {self.experiment_dir}...")
        
        for split_name, split_results in results.items():
            if split_results['labels'] is None:
                continue
                
            # Prepare data for saving - 确保所有数组都是一维的
            data_dict = {
                'label': np.asarray(split_results['labels']).flatten(),
                'ensemble_pred': np.asarray(split_results['ensemble_prediction']).flatten()
            }
            
            # Add individual model predictions
            for model_name, predictions in split_results['individual_predictions'].items():
                data_dict[f'{model_name}_pred'] = np.asarray(predictions).flatten()
            
            # Create DataFrame and save
            df = pd.DataFrame(data_dict)
            save_path = Path(self.experiment_dir) / f"predictions_{split_name}.fits"
            cu.savefile(df, str(save_path))
            
        print("Results saved successfully!")
    
    def print_metrics_summary(self, results: Dict[str, Dict]):
        """Print a summary of all model metrics"""
        print("\n" + "="*80)
        print("ENSEMBLE PREDICTION RESULTS")
        print("="*80)
        
        for split_name, split_results in results.items():
            if split_results['labels'] is None:
                continue
                
            print(f"\n{split_name.upper()} SET RESULTS:")
            print("-" * 60)
            
            # Print individual model metrics
            for model_name, metrics in split_results['individual_metrics'].items():
                print(f"{model_name:<20} | "
                      f"bias: {metrics['mean_bias']:<8.5f} | "
                      f"std: {metrics['std_dev']:<8.5f} | "
                      f"mad: {metrics['mad']:<8.5f} | "
                      f"outliers: {metrics['outlier_fraction']:<8.5f}")
            
            # Print ensemble metrics
            if split_results['ensemble_metrics']:
                metrics = split_results['ensemble_metrics']
                print(f"{'ENSEMBLE':<20} | "
                      f"bias: {metrics['mean_bias']:<8.5f} | "
                      f"std: {metrics['std_dev']:<8.5f} | "
                      f"mad: {metrics['mad']:<8.5f} | "
                      f"outliers: {metrics['outlier_fraction']:<8.5f}")
        
        print("="*80)

#============================================================
# Configuration
#============================================================
def get_dataset_paths():
    """配置数据集文件路径
    
    Returns:
        Dict: 包含训练、验证、测试集文件路径的字典
    """
    return {
        'train': '../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_train.fits',
        'validation': '../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_val.fits', 
        'test': '../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_test.fits'
    }

def get_model_paths():
    """配置模型路径
    
    Returns:
        Dict: 包含各个模型路径的字典
    """
    return {
        'ann': "/home/tiandc/panstarrs/panstarrs_desi/photoz_ann/1SameValTest_chi2>25",
        'rf': "/home/tiandc/panstarrs/panstarrs_desi/photoz_rf/SameValTest_chi2>25",
        'xgb': "/home/tiandc/panstarrs/panstarrs_desi/photoz_xgb/SameValTest_chi2>25"
    }

def get_active_models():
    """配置要使用的模型
    
    Returns:
        List: 要激活的模型名称列表
    """
    # 可以根据需要修改这里
    return ['rf', 'xgb']  # 只使用RF和XGB
    # return ['ann', 'rf']  # 只使用ANN和RF
    # return ['ann', 'xgb'] # 只使用ANN和XGB
    # return ['ann', 'rf', 'xgb']  # 使用所有三个模型

def create_common_dataset(file_path: str, mode: str, config: Dict, scaler=None):
    """创建通用数据集（所有模型共用）
    
    Args:
        file_path: 数据文件路径
        mode: 数据集模式 ('train', 'validation', 'test')
        config: 配置参数字典
        scaler: 标准化器（ANN使用，RF/XGB为None）
    
    Returns:
        Dataset_Photz: 数据集实例
    """
    return Dataset_Photz(
        file_path=file_path,
        mag_types=config.get('mag_types', ['Kron', 'PSF', 'Ap']),
        use_mag=config.get('use_mag', True),
        scaler_X=scaler,
        mode=mode,
        use_magerr=config.get('use_magerr', True),
        use_colorerr=config.get('use_colorerr', False),
        color_types=config.get('color_types', ['Kron', 'Ap'])
    )

#============================================================
# Main Execution
#============================================================
def main():
    """Main function to run ensemble prediction"""
    
    # Initialize ensemble predictor
    experiment_dir = "./rf_xgb"
    ensemble = EnsemblePhotozPredictor(experiment_dir)
    
    # Get model and dataset paths from configuration
    model_paths = get_model_paths()
    file_paths = get_dataset_paths()
    active_models = get_active_models()
    
    # Add models to ensemble based on configuration
    model_classes = {
        # 'ann': ANNPhotozModel,
        'rf': RFPhotozModel, 
        'xgb': XGBPhotozModel
    }
    
    model_names = {
        # 'ann': "ANN",
        'rf': "RandomForest",
        'xgb': "XGBoost"
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
    
    # Load all models
    ensemble.load_all_models()
    
    # 检查文件是否存在
    print(f"Checking dataset file paths:")
    for split_name, file_path in file_paths.items():
        if not Path(file_path).exists():
            print(f"Warning: {split_name} file not found at {file_path}")
        else:
            print(f"  ✓ {split_name}: {file_path}")
    
    # Run ensemble prediction
    results = ensemble.predict_ensemble(file_paths)
    
    # Save results
    ensemble.save_results(results)
    
    # Print metrics summary
    ensemble.print_metrics_summary(results)
    
    return results

if __name__ == "__main__":
    results = main()

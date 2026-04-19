from __future__ import annotations

# Standard library
import concurrent.futures
import functools
import multiprocessing
import os
import glob
import shutil
import threading
import time
import h5py
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Third-party scientific
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyWarning
from dustmaps.sfd import SFDQuery
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm

# Suppress warnings globally
warnings.filterwarnings('ignore')

# 类型别名
NumericArray = Union[float, np.ndarray, pd.Series]
DataType = Union[pd.DataFrame, np.ndarray, Table]

# 常量
@dataclass(frozen=True)
class Constants:
    PI: float = 3.141592653589793
    DEG2RAD: float = PI / 180.0
    RAD2DEG: float = 180.0 / PI
    

# 配置类
@dataclass(frozen=True)
class CosmologyParams:
    """宇宙学参数配置"""
    OMEGA_LAMBDA: float = 0.7
    OMEGA_M: float = 0.3
    OMEGA_K: float = 0.0
    H_0: float = 70.0  # km/s/Mpc
    Q_EVOLUTION: float = 1.16
    M_SUN_R: float = 4.68
    L_STAR_0: float = 1.4e10  # in L_sun
    

@dataclass(frozen=True)
class FileFormats:
    """支持的文件格式"""
    H5 = '.h5'
    CSV = '.csv'
    DAT = '.dat'
    FITS = '.fits'
    FIT = '.fit'
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.H5, cls.CSV, cls.DAT, cls.FITS, cls.FIT]


# 常量实例
COSMO = CosmologyParams()


# 装饰器：处理数值错误
def handle_numeric_errors(func):
    """处理数值计算中的错误和警告"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                return func(*args, **kwargs)
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                raise ValueError(f"数值计算错误 in {func.__name__}: {str(e)}")
    return wrapper


####################################################
# IO优化版本
####################################################

class FileHandler:
    """文件处理类，统一管理文件读写操作"""
    
    @staticmethod
    def read_file(
        file_path: Union[str, Path], 
        data_format: Literal['dataframe', 'table'] = 'dataframe',
        delimiter: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Table]:
        """
        统一的文件读取接口
        
        Args:
            file_path: 文件路径
            data_format: 返回数据格式 ('dataframe' 或 'table')
            delimiter: 分隔符（用于文本文件）
            **kwargs: 传递给底层读取函数的额外参数
            
        Returns:
            读取的数据
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        # 根据文件类型选择读取方法
        readers = {
            FileFormats.H5: lambda p: pd.read_hdf(p, key='data', mode='r'),
            FileFormats.CSV: lambda p: pd.read_csv(p, **kwargs),
            FileFormats.DAT: lambda p: pd.DataFrame(
                np.loadtxt(p, delimiter=delimiter or ' ', **kwargs)
            ),
            FileFormats.FITS: lambda p: Table.read(p, format='fits', **kwargs),
            FileFormats.FIT: lambda p: Table.read(p, format='fits', **kwargs)
        }
        
        if suffix not in readers:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        # 读取数据
        data = readers[suffix](str(file_path))
        
        # 转换格式
        if data_format.lower() == 'dataframe':
            if isinstance(data, Table):
                data = data.to_pandas()
        elif data_format.lower() == 'table':
            if isinstance(data, pd.DataFrame):
                data = Table.from_pandas(data)
            elif suffix not in [FileFormats.FITS, FileFormats.FIT]:
                raise ValueError(f"文件类型 {suffix} 不支持转换为 Table 格式")
        
        return data
    
    @staticmethod
    def save_file(
        data: DataType,
        file_path: Union[str, Path],
        auto_convert_objects: bool = True,
        delimiter: Optional[str] = None,
        compression: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        统一的文件保存接口
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
            auto_convert_objects: 是否自动转换object类型列
            delimiter: 分隔符（用于文本文件）
            compression: 压缩方式（用于支持压缩的格式）
            **kwargs: 传递给底层保存函数的额外参数
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 预处理数据
        if isinstance(data, pd.DataFrame) and auto_convert_objects:
            data = FileHandler._convert_object_columns(data)
        
        suffix = file_path.suffix.lower()
        
        # 根据文件类型选择保存方法
        if suffix == FileFormats.H5:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            df.to_hdf(
                str(file_path), 
                key='data', 
                mode='w', 
                format='table',
                complib=compression,
                **kwargs
            )
        elif suffix == FileFormats.CSV:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            df.to_csv(str(file_path), index=False, **kwargs)
        elif suffix in [FileFormats.FITS, FileFormats.FIT]:
            if isinstance(data, pd.DataFrame):
                table = Table.from_pandas(data)
            elif isinstance(data, Table):
                table = data
            else:
                table = Table(data)
            table.write(str(file_path), format='fits', overwrite=True, **kwargs)
        elif suffix == FileFormats.DAT:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                array = data.values if isinstance(data, pd.DataFrame) else data.to_numpy()
            else:
                array = np.asarray(data)
            np.savetxt(str(file_path), array, delimiter=delimiter or '\t', **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    @staticmethod
    def _convert_object_columns(df: pd.DataFrame) -> pd.DataFrame:
        """转换DataFrame中的object类型列为数值类型"""
        df = df.copy()
        object_cols = df.select_dtypes(include=['object']).columns
        
        if len(object_cols) > 0:
            for col in object_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass  # 保持原样
            
            # 报告转换结果
            converted_cols = []
            for col in object_cols:
                if df[col].dtype != 'object':
                    converted_cols.append(col)
            
            if converted_cols:
                print(f"已转换 {len(converted_cols)} 个列为数值类型: {', '.join(converted_cols)}")
        
        return df


def fits_to_dataframe(
    fits_path: Union[str, Path],
    hdu_index: int = 1,
    verbose: bool = True,
    return_metadata: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    优化的FITS文件读取函数
    
    Args:
        fits_path: FITS文件路径
        hdu_index: HDU索引
        verbose: 是否打印详细信息
        return_metadata: 是否返回元数据
        
    Returns:
        DataFrame 或 (DataFrame, metadata) 元组
    """
    fits_path = Path(fits_path)
    
    try:
        with fits.open(str(fits_path)) as hdul:
            if verbose:
                print(f"FITS文件信息 ({fits_path.name}):")
                print(hdul.info())
            
            # 验证HDU
            if hdu_index >= len(hdul):
                raise IndexError(f"HDU索引 {hdu_index} 超出范围（共 {len(hdul)} 个HDU）")
            
            if not isinstance(hdul[hdu_index], (fits.BinTableHDU, fits.TableHDU)):
                raise TypeError(f"HDU {hdu_index} 不是表格数据")
            
            # 读取数据
            table = Table(hdul[hdu_index].data)
            df = table.to_pandas()
            
            if verbose:
                print(f"\n数据形状: {df.shape} (行×列)")
                print(f"列名: {', '.join(df.columns[:10])}" + 
                      ("..." if len(df.columns) > 10 else ""))
            
            if return_metadata:
                metadata = dict(hdul[0].header)
                return df, metadata
            return df
            
    except Exception as e:
        raise IOError(f"读取FITS文件失败: {str(e)}")


####################################################
# 物理函数优化版本
####################################################

class Cosmology:
    """宇宙学计算类"""
    
    def __init__(self, params: Optional[CosmologyParams] = None):
        self.params = params or COSMO
    
    @handle_numeric_errors
    def E(self, z: NumericArray) -> NumericArray:
        """计算E(z) = H(z)/H0"""
        z = np.asarray(z, dtype=np.float64)
        result = (
            self.params.OMEGA_M * (1 + z)**3 + 
            self.params.OMEGA_K * (1 + z)**2 + 
            self.params.OMEGA_LAMBDA
        )
        return np.sqrt(result)
    
    @handle_numeric_errors
    def H(self, z: NumericArray) -> NumericArray:
        """计算哈勃参数 H(z) [km/s/Mpc]"""
        return self.params.H_0 * self.E(z)
    
    @handle_numeric_errors
    def angular_diameter_distance(
        self, 
        z: NumericArray, 
        n_steps: int = 1000
    ) -> NumericArray:
        """
        计算角直径距离 [Mpc]
        使用改进的数值积分方法
        """
        z = np.asarray(z, dtype=np.float64)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)
        
        # 使用向量化的梯形积分
        results = np.zeros_like(z, dtype=float)
        
        for i, z_val in enumerate(z):
            if z_val <= 0:
                results[i] = 0
                continue
                
            # 创建积分网格
            z_grid = np.linspace(0, z_val, n_steps)
            integrand = 1.0 / self.E(z_grid)
            
            # 梯形积分
            integral = np.trapezoid(integrand, z_grid)
            
            # 角直径距离
            c_over_H0 = 299792.458 / self.params.H_0  # c/H0 in Mpc
            results[i] = c_over_H0 * integral / (1 + z_val)
        
        return results[0] if scalar_input else results
    
    def angular_scale(self, z: NumericArray) -> NumericArray:
        """计算角尺度 [rad/Mpc]"""
        d_A = self.angular_diameter_distance(z)
        return 1.0 / d_A
    


####################################################
# 统计函数优化版本
####################################################

class Statistics:
    """统计计算类"""
    
    @staticmethod
    def calculate_metrics(
        label: NumericArray,
        prediction: NumericArray,
        mode: Literal['log', 'normal'] = 'log',
        remove_outliers: bool = True,
        outlier_sigma: float = 3.0
    ) -> Dict[str, float]:
        """
        计算预测指标（bias和scatter）
        
        Args:
            label: 真实值
            prediction: 预测值
            mode: 计算模式 ('log' 或 'normal')
            remove_outliers: 是否移除离群值
            outlier_sigma: 离群值阈值（标准差倍数）
            
        Returns:
            包含各种指标的字典
        """
        # 转换为数组并过滤无效值
        label = np.asarray(label)
        prediction = np.asarray(prediction)
        
        if mode == 'log':
            mask = (label > 0) & (prediction > 0)
        else:
            mask = np.ones_like(label, dtype=bool)
        
        mask &= np.isfinite(label) & np.isfinite(prediction)
        
        if not mask.any():
            return {
                'bias_mean': np.nan,
                'bias_median': np.nan,
                'bias_mad': np.nan,
                'scatter_std': np.nan,
                'scatter_mad': np.nan,
                'n_samples': 0
            }
        
        label = label[mask]
        prediction = prediction[mask]
        
        # 计算偏差
        if mode == 'log':
            deviation = np.log10(prediction) - np.log10(label)
        else:
            deviation = prediction - label
        
        # 移除离群值
        if remove_outliers and len(deviation) > 10:
            z_scores = np.abs((deviation - np.median(deviation)) / 
                            (1.4826 * np.median(np.abs(deviation - np.median(deviation)))))
            outlier_mask = z_scores < outlier_sigma
            deviation = deviation[outlier_mask]
            n_outliers = len(z_scores) - np.sum(outlier_mask)
        else:
            n_outliers = 0
        
        # 计算指标
        metrics = {
            'bias_mean': np.mean(deviation),
            'bias_median': 1.4826 * np.median(deviation),
            'scatter_std': np.std(deviation),
            'scatter_mad': 1.4826 * np.median(np.abs(deviation - np.median(deviation))),
            'n_samples': len(deviation),
            'n_outliers': n_outliers
        }
        
        # 添加百分比指标（仅对正值）
        if mode == 'log' and len(label) > 0:
            percent_dev = np.abs(prediction - label) / label * 100
            metrics['percent_mad'] = 1.4826 * np.median(
                np.abs(percent_dev - np.median(percent_dev))
            )
        
        return metrics


####################################################
# 优化的调查统计函数
####################################################

def survey_source_count(
    dir_path: Union[str, Path],
    file_pattern: str = '*.fits',
    max_workers: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    并行统计调查中的源数量
    
    Args:
        dir_path: 目录路径
        file_pattern: 文件匹配模式
        max_workers: 最大工作线程数
        show_progress: 是否显示进度
        
    Returns:
        包含统计信息的字典
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    
    file_list = list(dir_path.glob(file_pattern))
    if not file_list:
        return {
            'total_sources': 0,
            'n_files': 0,
            'files_processed': []
        }
    
    # 线程安全的计数器
    total_sources = 0
    processed_files = []
    lock = threading.Lock()
    
    def process_file(file_path: Path) -> int:
        """处理单个文件"""
        try:
            handler = FileHandler()
            df = handler.read_file(file_path)
            n_sources = len(df)
            
            with lock:
                nonlocal total_sources
                total_sources += n_sources
                processed_files.append(str(file_path))
            
            return n_sources
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return 0
    
    # 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in file_list}
        
        if show_progress:
            from tqdm import tqdm
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Processing files"):
                future.result()
        else:
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    return {
        'total_sources': total_sources,
        'n_files': len(processed_files),
        'files_processed': sorted(processed_files),
        'average_sources_per_file': total_sources / len(processed_files) if processed_files else 0
    }


# 保留原始函数接口以保持向后兼容
readfile = FileHandler.read_file
savefile = FileHandler.save_file

default_cosmo = Cosmology()

@handle_numeric_errors
def m500_to_r500(m500: NumericArray, z: NumericArray) -> NumericArray:
    """
    将M500质量转换为R500半径 [Mpc]
    
    Args:
        m500: M500质量 [10^14 M_sun]
        z: 红移
        
    Returns:
        R500半径 [Mpc]
    """
    m500 = np.asarray(m500, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    G_si = const.G.value  # m³/(kg·s²)
    M_sun_kg = const.M_sun.value  # kg
    Mpc_to_m = 1e6 * const.pc.value  # m
    
    # G in Mpc³/(M_sun·s²)
    G = G_si * (Mpc_to_m**3) / M_sun_kg
    
    H_z = default_cosmo.H(z)  # km/s/Mpc
    
    H_z_si = H_z * 1000 / Mpc_to_m  # 1/s
    
    # 计算临界密度 [M_sun/Mpc³]
    rho_crit_si = 3 * H_z_si**2 / (8 * np.pi * G_si)  # kg/m³
    rho_crit = rho_crit_si * (Mpc_to_m**3) / M_sun_kg  # M_sun/Mpc³
    
    # 计算R500 [Mpc]
    r500_cubed = 3 * m500 * 1e14 / (4 * np.pi * 500 * rho_crit)
    
    return np.cbrt(r500_cubed)

@handle_numeric_errors
def r500_to_m500(r500: NumericArray, z: NumericArray) -> NumericArray:
    """
    将R500半径转换为M500质量 [10^14 M_sun]

    Args:
        r500: R500半径 [Mpc]
        z: 红移

    Returns:
        M500质量 [10^14 M_sun]
    """
    r500 = np.asarray(r500, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    G_si = const.G.value  # m³/(kg·s²)
    M_sun_kg = const.M_sun.value  # kg
    Mpc_to_m = 1e6 * const.pc.value  # m

    H_z = default_cosmo.H(z)  # km/s/Mpc

    H_z_si = H_z * 1000 / Mpc_to_m  # 1/s

    # 计算临界密度 [M_sun/Mpc³]
    rho_crit_si = 3 * H_z_si**2 / (8 * np.pi * G_si)  # kg/m³
    rho_crit = rho_crit_si * (Mpc_to_m**3) / M_sun_kg  # M_sun/Mpc³

    # 从R500计算M500
    # r500_cubed = 3 * m500 * 1e14 / (4 * π * 500 * rho_crit)
    # => m500 * 1e14 = r500_cubed * (4 * π * 500 * rho_crit) / 3
    r500_cubed = r500**3
    m500 = (4 * np.pi * 500 * rho_crit * r500_cubed) / 3 / 1e14

    return m500

def Eofz(z: NumericArray) -> NumericArray:
    return default_cosmo.E(z)

def Hofz(z: NumericArray) -> NumericArray:
    return default_cosmo.H(z)

def disofz(z: NumericArray, rad2ang: bool = True) -> NumericArray:
    d_A = default_cosmo.angular_diameter_distance(z)
    return d_A / Constants.RAD2DEG if rad2ang else d_A


def bias(
    label: NumericArray,
    prediction: NumericArray,
    mode: str = 'log',
    operation: str = 'average'
) -> float:
    """向后兼容的bias函数"""
    stats = Statistics()
    metrics = stats.calculate_metrics(label, prediction, mode)
    
    print(f'bias (average): {metrics["bias_mean"]:.3f}')
    print(f'bias (median): {metrics["bias_median"]:.3f}')
    if 'percent_mad' in metrics:
        print(f'bias (MAD, %): {metrics["percent_mad"]:.2f}%')
    
    if operation == 'average':
        return metrics['bias_mean']
    elif operation == 'median':
        return metrics['bias_median']
    else:
        raise ValueError(f"Unknown operation: {operation}")

def scatter(
    label: NumericArray,
    prediction: NumericArray,
    mode: str = 'log',
    operation: str = 'average'
) -> float:
    """向后兼容的scatter函数"""
    stats = Statistics()
    metrics = stats.calculate_metrics(label, prediction, mode)
    
    print(f'scatter (average): {metrics["scatter_std"]:.3f}')
    print(f'scatter (MAD): {metrics["scatter_mad"]:.3f}')
    
    if operation == 'average':
        return metrics['scatter_std']
    elif operation == 'MAD':
        return metrics['scatter_mad']
    else:
        raise ValueError(f"Unknown operation: {operation}")


# 保留其他原始函数
def specz_slice(sigma_v: float, z: NumericArray) -> NumericArray:
    """光谱红移切片"""
    z = np.asarray(z)
    temp1 = sigma_v * (1 + z) * 1e-5
    temp2 = 25/3 * 1e-3 * (1 + z)
    return np.maximum(temp1, temp2)

def photz_slice(z: NumericArray) -> NumericArray:
    """测光红移切片"""
    z = np.asarray(z)
    return np.where(z <= 0.45, 0.04 * (1 + z), 0.248 * z - 0.0536)

def galaxy_absolute_magnitude(
        mag: NumericArray, 
        redshift: NumericArray
    ) -> NumericArray:
    """计算星系绝对星等"""
    mag = np.asarray(mag)
    redshift = np.asarray(redshift)
    
    # 距离（pc）
    dist = disofz(redshift, rad2ang=False) * 1e6
    
    # 视星等 -> 绝对星等
    absmag = mag - 5 * np.log10(dist) + 5
    
    # 演化修正
    absmag += COSMO.Q_EVOLUTION * redshift
    
    return absmag

def galaxy_luminosity(mag: NumericArray, redshift: NumericArray, 
                      normalize_to_lstar: bool = True) -> NumericArray:
    """
    计算星系光度
    
    参数:
    -----------
    mag : NumericArray
        观测星等（r波段）
    redshift : NumericArray
        红移值
    normalize_to_lstar : bool, optional
        是否归一化到特征光度 L*
        - True: 返回 L/L* (无量纲)
        - False: 返回绝对光度 (以太阳光度为单位)
        默认为 True
    
    返回:
    -----------
    luminosity : NumericArray
        如果 normalize_to_lstar=True: L/L* (无量纲)
        如果 normalize_to_lstar=False: L (以 L_sun 为单位)
    """
    mag = np.asarray(mag)
    redshift = np.asarray(redshift)
    
    # 光度距离（pc）
    dist = disofz(redshift, rad2ang=False) * 1e6
    
    # 视星等 -> 绝对星等
    absmag = mag - 5 * np.log10(dist) + 5
    
    # 演化修正
    absmag += COSMO.Q_EVOLUTION * redshift
    
    # 绝对星等 -> 光度（以太阳光度为单位）
    delta_M = absmag - COSMO.M_SUN_R
    lum_in_sun = 10**(-0.4 * delta_M)
    
    # 是否归一化
    if normalize_to_lstar:
        # 演化的特征光度
        L_star = COSMO.L_STAR_0 * 10**(0.4 * COSMO.Q_EVOLUTION * redshift)
        return lum_in_sun / L_star
    else:
        return lum_in_sun

def add_dis_range(df: pd.DataFrame) -> pd.DataFrame:
    """添加距离相关列"""
    df = df.copy()
    df['dist_ang'] = disofz(df['z'], rad2ang=True)
    df['ang_r500'] = df['r500'] / df['dist_ang']
    df['ang_1mpc'] = 1.0 / df['dist_ang']
    return df


# 保留原始的红移度量函数
def redshift_delta_z_norm(z_phot: NumericArray, z_spec: NumericArray) -> NumericArray:
    """计算归一化红移残差"""
    return (z_phot - z_spec) / (1 + z_spec)

def redshift_outlier_fraction(delta_z_norm: NumericArray, threshold: float = 0.15) -> float:
    """计算异常值比例"""
    return np.mean(np.abs(delta_z_norm) > threshold)

def redshift_mean_bias(delta_z_norm: NumericArray, threshold: float = 0.15) -> float:
    """计算非异常值样本的平均偏差"""
    non_outliers = np.abs(delta_z_norm) <= threshold
    return np.mean(delta_z_norm[non_outliers])

def redshift_std_dev(delta_z_norm: NumericArray, threshold: float = 0.15) -> float:
    """计算非异常值样本的标准差"""
    non_outliers = np.abs(delta_z_norm) <= threshold
    return np.std(delta_z_norm[non_outliers])

def redshift_mad(delta_z_norm: NumericArray) -> float:
    """计算中位绝对偏差"""
    median_dz = np.median(delta_z_norm)
    return 1.4826 * np.median(np.abs(delta_z_norm - median_dz))

def evaluate_redshift_quality(
    z_phot: NumericArray, 
    z_spec: NumericArray, 
    threshold: float = 0.15
    ) -> Dict[str, float]:
    """评估红移质量"""
    delta_z_norm = redshift_delta_z_norm(z_phot, z_spec)
    
    return {
        'mean_bias': redshift_mean_bias(delta_z_norm, threshold),
        'std_dev': redshift_std_dev(delta_z_norm, threshold),
        'mad': redshift_mad(delta_z_norm),
        'outlier_fraction': redshift_outlier_fraction(delta_z_norm, threshold),
    }


def plot_prediction_scatter(
    x, y,
    title='Prediction Results',
    xlabel='True Value',
    ylabel='Predicted Value',
    xlim=None,
    ylim=None,
    figsize=(6, 6),
    log_scale=True,
    density_plot=True,
    cmap='plasma',
    alpha=0.5,
    point_size=1,
    diagonal_line=True,
    diagonal_color='gray',
    diagonal_style='--',
    diagonal_width=2,
    fontsize_title=15,
    fontsize_label=15,
    fontsize_stats=20,
    show_stats=True,
    stats_position=(0.15, 15),
    return_fig=False
):
    """
    绘制预测结果的散点图，支持多种自定义选项
    
    Args:
        x: 真实值数组
        y: 预测值数组
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        xlim: x轴范围，格式为(min, max)
        ylim: y轴范围，格式为(min, max)
        figsize: 图表大小，格式为(width, height)
        log_scale: 是否使用对数坐标轴
        density_plot: 是否使用密度图
        cmap: 颜色映射
        alpha: 点的透明度
        point_size: 点的大小
        diagonal_line: 是否显示对角线
        diagonal_color: 对角线颜色
        diagonal_style: 对角线样式
        diagonal_width: 对角线宽度
        fontsize_title: 标题字体大小
        fontsize_label: 轴标签字体大小
        fontsize_stats: 统计值字体大小
        show_stats: 是否显示统计值
        stats_position: 统计值位置，格式为(x, y)
        return_fig: 是否返回figure对象而不是显示图表
    
    Returns:
        如果return_fig为True，返回matplotlib.figure.Figure对象
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 计算密度（如果需要）
    if density_plot and len(x) > 10:
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, c=z, s=point_size, cmap=cmap, 
                       norm=plt.Normalize(), alpha=alpha)
        except Exception:
            # 如果密度计算失败，使用普通散点图
            plt.scatter(x, y, s=point_size, alpha=alpha)
    else:
        plt.scatter(x, y, s=point_size, alpha=alpha)
    
    # 设置坐标轴范围
    if xlim is None:
        xlim = (np.min(x), np.max(x))
    if ylim is None:
        ylim = (np.min(y), np.max(y))
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # 设置对数坐标轴
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    
    # 添加对角线
    if diagonal_line:
        diag_range = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
        plt.plot(diag_range, diag_range, 
                diagonal_style, lw=diagonal_width, color=diagonal_color)
    
    # 设置标签和标题
    plt.xlabel(xlabel, fontsize=fontsize_label)
    plt.ylabel(ylabel, fontsize=fontsize_label)
    plt.title(title, fontsize=fontsize_title)
    
    # 添加统计信息
    if show_stats:
        bias_value = bias(x, y, mode='log', operation='median')
        scatter_value = scatter(x, y, mode='log', operation='MAD')
        text = f'bias: {bias_value:.3f}\nscatter: {scatter_value:.3f}'
        plt.text(stats_position[0], stats_position[1], text, 
                fontsize=fontsize_stats, ha='left', va='top',
                transform=plt.gca().transAxes if stats_position[0] <= 1 else None)
    
    # 调整布局
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()



#======================================================================
# Recollection data
#======================================================================
def _normalize_table_dtypes(tables):
    """
    标准化表格列的数据类型，使其可以被 vstack 合并。
    对于类型不一致的列，bytes/string 转换为 float32，数值类型统一为 float32。
    """
    if not tables:
        return tables

    # 收集所有列名和其在各表中的类型
    col_types = {}
    for t in tables:
        for col in t.colnames:
            dtype_str = str(t[col].dtype)
            if col not in col_types:
                col_types[col] = set()
            col_types[col].add(dtype_str)

    # 找出类型不一致的列
    problematic_cols = {col: types for col, types in col_types.items() if len(types) > 1}

    if not problematic_cols:
        return tables

    # 对每个表进行类型修正
    normalized_tables = []
    for t in tables:
        t_copy = t.copy()
        for col in problematic_cols:
            if col in t_copy.colnames:
                dtype_str = str(t_copy[col].dtype)
                # 如果是bytes/string类型，尝试转换为float
                if dtype_str.startswith(('|S', 'S', 'bytes', 'U', '<U')):
                    try:
                        # 尝试将字节/字符串转换为float，无法转换的设为nan
                        new_data = []
                        for val in t_copy[col]:
                            try:
                                if isinstance(val, bytes):
                                    val = val.decode('utf-8').strip()
                                else:
                                    val = str(val).strip()
                                if val == '' or val.lower() in ('nan', 'null', 'none', '--'):
                                    new_data.append(np.nan)
                                else:
                                    new_data.append(float(val))
                            except:
                                new_data.append(np.nan)
                        t_copy[col] = np.array(new_data, dtype=np.float32)
                    except Exception:
                        # 转换失败则保持原样
                        pass
                else:
                    # 对于数值类型，统一为float32
                    try:
                        t_copy[col] = t_copy[col].astype(np.float32)
                    except:
                        pass
        normalized_tables.append(t_copy)

    return normalized_tables


def _process_chunk(args):
    """
    由单个进程执行的工作函数。
    处理一部分文件，并将结果分发到临时的分块文件中。
    """
    # 在子进程中也过滤警告
    import warnings
    from astropy.utils.exceptions import AstropyWarning
    warnings.filterwarnings('ignore', category=AstropyWarning)

    file_chunk, worker_id, config = args

    # 解包配置
    temp_dir = config['temp_dir']
    ra_column = config['ra_column']
    data_hdu_index = config['data_hdu_index']
    buffer_flush_size = config['buffer_flush_size']
    bin_size = config['bin_size']
    num_bins = config['num_bins']
    
    # 为每个桶创建缓冲区列表
    buffers = [[] for _ in range(num_bins)]
    # 为每个桶维护一个计数器，避免重复的glob操作
    part_counters = [0 for _ in range(num_bins)]

    for filepath in file_chunk:
        try:
            # 读取整个FITS表
            table = Table.read(filepath, hdu=data_hdu_index)
            
            if ra_column not in table.colnames:
                continue
            
            # 向量化操作：一次性计算所有行的桶索引
            ra_values = table[ra_column] % 360.0
            bin_indices = (ra_values // bin_size).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            
            # 按桶分组，使用mask筛选，只处理有数据的桶
            unique_bins = np.unique(bin_indices)
            for bin_idx in unique_bins:
                mask = (bin_indices == bin_idx)
                sub_table = table[mask]
                buffers[bin_idx].append(sub_table)
                
                # 检查缓冲区大小，超过阈值则写入临时文件
                total_rows = sum(len(t) for t in buffers[bin_idx])
                if total_rows >= buffer_flush_size:
                    # 合并缓冲区中的所有表（先标准化类型）
                    normalized_buffers = _normalize_table_dtypes(buffers[bin_idx])
                    merged_table = vstack(normalized_buffers, metadata_conflicts='silent')
                    
                    # 生成临时文件名，使用计数器而不是glob查找
                    temp_filename = os.path.join(temp_dir, 
                                                f"bin_{bin_idx:02d}_worker_{worker_id:02d}_part_{part_counters[bin_idx]:04d}.fits")
                    
                    # 写入临时文件
                    merged_table.write(temp_filename, format='fits', overwrite=True)
                    
                    # 更新计数器
                    part_counters[bin_idx] += 1
                    
                    # 清空该桶的缓冲区
                    buffers[bin_idx] = []
                    del merged_table
            
            # 释放内存
            del table, ra_values, bin_indices
            
        except Exception as e:
            # 在并行环境中，打印警告而不是中断整个流程
            print(f"[Worker {worker_id}] Warning: Failed to process {os.path.basename(filepath)}. Error: {e}")

    # 处理完所有文件后，清空所有剩余的缓冲区
    for bin_idx in range(num_bins):
        if buffers[bin_idx]:
            normalized_buffers = _normalize_table_dtypes(buffers[bin_idx])
            merged_table = vstack(normalized_buffers, metadata_conflicts='silent')
            
            temp_filename = os.path.join(temp_dir, 
                                        f"bin_{bin_idx:02d}_worker_{worker_id:02d}_part_{part_counters[bin_idx]:04d}.fits")
            
            merged_table.write(temp_filename, format='fits', overwrite=True)
            del merged_table

    return True

def merge_fits_by_ra(
    input_dir,
    output_dir,
    output_filename="jplusdr4_{i:02d}.fits",
    temp_dir="./temp",
    num_processes=None,
    ra_column='alpha_j2000',
    fits_extension="*.fits",
    data_hdu_index=1,
    buffer_flush_size=1000000,
    bin_size=5.0,
    add_uid=True
):
    """
    将多个FITS文件按赤经(RA)分桶合并。

    Parameters
    ----------
    input_dir : str
        输入目录，包含FITS文件的文件夹路径
    output_dir : str
        最终输出目录，存放合并后文件的文件夹路径
    output_filename : str, optional
        输出文件名模板，需要包含 {i:02d} 格式化占位符用于桶索引
        默认为 "jplusdr4_{i:02d}.fits"
    temp_dir : str, optional
        临时工作目录，用于存放中间文件，处理完后会自动删除
        默认为 "./temp"
    num_processes : int, optional
        并行进程数。默认为 CPU 核心数 - 2
    ra_column : str, optional
        FITS文件中的RA列名，默认为 'alpha_j2000'
    fits_extension : str, optional
        FITS文件匹配模式，默认为 "*.fits"
    data_hdu_index : int, optional
        数据所在的HDU索引，默认为 1
    buffer_flush_size : int, optional
        内存缓冲区大小（行数），默认为 1000000
    bin_size : float, optional
        RA分桶大小（度），默认为 5.0 度
    add_uid : bool, optional
        是否为每行添加全局唯一标识符 'uid' 列，默认为 True
        uid 从0开始，按文件顺序（桶索引）连续编号

    Returns
    -------
    dict
        包含处理结果的字典：
        - total_rows: 总行数
        - duration_minutes: 处理时间（分钟）
        - output_files: 输出文件列表
    """
    # 过滤 astropy 警告
    warnings.filterwarnings('ignore', category=AstropyWarning)
    
    start_time = time.time()
    
    # 设置默认进程数
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 2)
    
    # 计算桶数量
    num_bins = int(360 / bin_size)
    
    # 构建配置字典（用于传递给子进程）
    config = {
        'temp_dir': temp_dir,
        'ra_column': ra_column,
        'data_hdu_index': data_hdu_index,
        'buffer_flush_size': buffer_flush_size,
        'bin_size': bin_size,
        'num_bins': num_bins
    }

    # --- 1. 初始化环境 ---
    print("Step 1: Setting up directories...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # --- 2. 准备文件列表并切分 ---
    print(f"\nStep 2: Preparing file list for {num_processes} parallel processes...")
    source_files = sorted(glob.glob(os.path.join(input_dir, fits_extension)))
    if not source_files:
        print(f"Error: No files found matching '{fits_extension}' in '{input_dir}'")
        return {'total_rows': 0, 'duration_minutes': 0, 'output_files': []}
    print(f"Found {len(source_files)} source files.")
    
    # 估算总数据大小
    total_size_gb = sum(os.path.getsize(f) for f in source_files) / (1024**3)
    print(f"Total data size: {total_size_gb:.2f} GB")
    
    # 将文件列表切分成块，分发给每个进程
    chunks = np.array_split(source_files, num_processes)
    tasks = [(list(chunk), i, config) for i, chunk in enumerate(chunks)]

    # --- 3. 并行处理 ---
    print("\nStep 3: Starting parallel processing...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(_process_chunk, tasks)
    
    parallel_duration = time.time() - start_time
    print(f"Parallel processing finished in {parallel_duration / 60:.2f} minutes.")

    # --- 4. 合并每个桶的临时文件 ---
    print("\nStep 4: Consolidating temporary files into final output files...")
    total_rows = 0
    output_files = []
    uid_counter = 0  # 全局uid计数器

    for i in tqdm(range(num_bins), desc="Consolidating bins"):
        temp_files_for_bin = sorted(glob.glob(os.path.join(temp_dir, f"bin_{i:02d}_worker_*.fits")))

        if not temp_files_for_bin:
            continue

        try:
            # vstack可以高效地一次性读取和合并多个文件
            tables = [Table.read(f, hdu=1) for f in temp_files_for_bin]
            tables = _normalize_table_dtypes(tables)
            final_table = vstack(tables, metadata_conflicts='silent')

            # 添加全局唯一标识符
            if add_uid:
                n_rows = len(final_table)
                final_table['uid'] = np.arange(uid_counter, uid_counter + n_rows, dtype=np.int64)
                uid_counter += n_rows

            # 使用模板生成输出文件名
            out_filename = os.path.join(output_dir, output_filename.format(i=i))
            final_table.write(out_filename, format='fits', overwrite=True)

            total_rows += len(final_table)
            output_files.append(out_filename)
            del tables, final_table

        except Exception as e:
            print(f"\nError consolidating bin {i}: {e}")
            print(f"Problematic files might be: {temp_files_for_bin}")

    # --- 5. 清理 ---
    print("\nStep 5: Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    
    total_duration = time.time() - start_time
    print("\n--- All Done! ---")
    print(f"Processing complete in {total_duration / 60:.2f} minutes.")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Final files are located in: {output_dir}")
    
    return {
        'total_rows': total_rows,
        'duration_minutes': total_duration / 60,
        'output_files': output_files
    }



#======================================================================
# Red Sequence Selection
#======================================================================
# 适用于单个星系团的红序筛选
from sklearn.linear_model import RANSACRegressor, LinearRegression
def find_red_sequence_ransac_asym(
    backgal_mag, backgal_color,
    centergal_mag=None, centergal_color=None,
    dcolor_blue=0.2,
    dcolor_red=1.,
    residual_threshold=0.2,
    centergal_residual_threshold=0.2,
    random_state=0
):
    X = np.asarray(backgal_mag).reshape(-1, 1)
    y = np.asarray(backgal_color)

    # RANSAC拟合初步红序
    ransac = RANSACRegressor(
        LinearRegression(),
        residual_threshold=residual_threshold,
        random_state=random_state
    )
    ransac.fit(X, y)

    a = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_

    # BCG一致性检测
    centergal_is_abnormal = False
    if centergal_mag is not None and centergal_color is not None:
        centergal_predicted_color = a * centergal_mag + b
        centergal_residual = centergal_color - centergal_predicted_color

        if np.abs(centergal_residual) <= centergal_residual_threshold:
            # BCG正常：平移直线通过BCG点
            b = centergal_color - a * centergal_mag
        else:
            # BCG异常：维持原拟合
            centergal_is_abnormal = True

    # 筛选红序星系
    residual = y - (a * X[:, 0] + b)
    mask_asym = (residual > -dcolor_blue) & (residual < dcolor_red)

    # 即使BCG异常，也强制包含在结果中
    if centergal_is_abnormal and centergal_mag is not None and centergal_color is not None:
        centergal_mask = (X[:, 0] == centergal_mag) & (y == centergal_color)
        if centergal_mask.any():
            mask_asym[centergal_mask] = True

    return mask_asym

def select_red_sequence_galaxies(
    backgal,
    centergal,
    mag_name='mag_r',
    color_names=['g_r', 'r_i', 'i_z'],
    dcolor_blue=0.2,
    dcolor_red=1.,
    residual_threshold=0.2,
    centergal_residual_threshold=0.2,
    use_bcg_anchor=True,
    random_state=42,
    verbose=True
):
    # 初始化mask
    inlier_mask = np.ones(len(backgal), dtype=bool)

    if len(centergal) != 1:
        raise ValueError(f"centergal应该只有1行（BCG），但有{len(centergal)}行")

    centergal_mag = centergal[mag_name].values[0]

    # 多颜色筛选（取交集）
    for color_name in color_names:
        if color_name not in backgal.columns:
            raise ValueError(f"颜色列 '{color_name}' 不存在于backgal中")
        if color_name not in centergal.columns:
            raise ValueError(f"颜色列 '{color_name}' 不存在于centergal中")

        backgal_mag = backgal[mag_name].values
        backgal_color = backgal[color_name].values
        centergal_color = centergal[color_name].values[0]

        # RANSAC拟合红序
        # 根据use_bcg_anchor参数决定是否使用BCG作为锚点
        if use_bcg_anchor:
            mask = find_red_sequence_ransac_asym(
                backgal_mag, backgal_color,
                centergal_mag=centergal_mag,
                centergal_color=centergal_color,
                dcolor_blue=dcolor_blue,
                dcolor_red=dcolor_red,
                residual_threshold=residual_threshold,
                centergal_residual_threshold=centergal_residual_threshold,
                random_state=random_state
            )
        else:
            mask = find_red_sequence_ransac_asym(
                backgal_mag, backgal_color,
                centergal_mag=None,
                centergal_color=None,
                dcolor_blue=dcolor_blue,
                dcolor_red=dcolor_red,
                residual_threshold=residual_threshold,
                centergal_residual_threshold=centergal_residual_threshold,
                random_state=random_state
            )

        # 取交集
        inlier_mask &= mask

    red_seq_galaxies = backgal[inlier_mask]

    if verbose:
        print(f"Total nearby galaxies: {len(backgal)}")
        print(f"Red sequence galaxies: {len(red_seq_galaxies)}")
        print(f"Red sequence ratio: {len(red_seq_galaxies)/len(backgal)*100:.1f}%")

    return red_seq_galaxies


# 多线程并行优化版本
def _ransac_fit(X, y, residual_threshold=0.2, max_trials=50, random_state=2025):
    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True, copy_X=False),
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=random_state,  # 恢复随机种子
        stop_probability=0.99  # 提前停止
    )
    ransac.fit(X, y)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

def _process_single_cluster_batch(cluster_batch, params):
    dcolor_blue = params['dcolor_blue']
    dcolor_red = params['dcolor_red']
    residual_threshold = params['residual_threshold']
    centergal_residual_threshold = params['centergal_residual_threshold']
    random_state = params['random_state']
    use_bcg_anchor = params['use_bcg_anchor']

    results = []

    for cid, data in cluster_batch:
        n_gal = len(data['gal_mag'])
        if n_gal == 0:
            continue

        masks = np.ones(n_gal, dtype=bool)
        centergal_is_abnormal_any = False

        # 对每个颜色做筛选
        for color_name, gal_color in data['gal_colors'].items():
            # RANSAC拟合
            a, b = _ransac_fit(
                data['gal_mag'].reshape(-1, 1),
                gal_color,
                residual_threshold=residual_threshold,
                max_trials=50,
                random_state=random_state
            )

            # 根据use_bcg_anchor参数决定是否使用BCG作为锚点
            if use_bcg_anchor:
                # BCG一致性检测
                centergal_mag = data['bcg_mag']
                centergal_color = data['bcg_colors'][color_name]
                centergal_predicted_color = a * centergal_mag + b
                centergal_residual = centergal_color - centergal_predicted_color

                if np.abs(centergal_residual) <= centergal_residual_threshold:
                    # BCG正常：平移通过BCG点
                    b = centergal_color - a * centergal_mag
                else:
                    # BCG异常：维持原拟合
                    centergal_is_abnormal_any = True

            # 筛选红序
            residuals = gal_color - (a * data['gal_mag'] + b)
            masks &= (residuals > -dcolor_blue) & (residuals < dcolor_red)

        # 如果使用BCG锚点且BCG异常，强制包含在结果中
        if use_bcg_anchor and centergal_is_abnormal_any:
            centergal_mask = (data['gal_mag'] == data['bcg_mag'])
            for color_name in data['gal_colors'].keys():
                centergal_mask &= (data['gal_colors'][color_name] == data['bcg_colors'][color_name])
            if np.any(centergal_mask):
                masks[centergal_mask] = True

        if np.any(masks):
            selected_indices = data['gal_indices'][masks]
            results.append((cid, selected_indices))

    return results

def _preprocess_data(total_backgal, total_centergal, mag_name, color_names):
    galaxy_groups = total_backgal.groupby('cid', sort=False)
    bcg_indexed = total_centergal.set_index('cid')

    cluster_list = []

    for cid in galaxy_groups.groups.keys():
        if cid not in bcg_indexed.index:
            continue

        gal_group = galaxy_groups.get_group(cid)

        # 跳过星系数少于3个的cluster
        if len(gal_group) < 3:
            continue

        bcg_row = bcg_indexed.loc[cid]

        # 准备cluster数据
        cluster_data = {
            'gal_indices': gal_group.index.values,
            'gal_mag': gal_group[mag_name].values.astype(np.float32),
            'gal_colors': {color: gal_group[color].values.astype(np.float32)
                          for color in color_names},
            'bcg_mag': float(bcg_row[mag_name]),
            'bcg_colors': {color: float(bcg_row[color]) for color in color_names}
        }

        cluster_list.append((cid, cluster_data))

    return cluster_list

def batch_select_red_sequence_galaxies(
    total_backgal,
    total_centergal,
    mag_name='mag_r',
    color_names=['g_r', 'r_i', 'i_z'],
    dcolor_blue=0.2,
    dcolor_red=1.0,
    residual_threshold=0.2,
    centergal_residual_threshold=0.2,
    use_bcg_anchor=True,
    random_state=2025,
    n_jobs=-1,
    batch_size=100,
    show_progress=True
):
    from joblib import cpu_count

    if n_jobs == -1:
        n_jobs = cpu_count()

    print(f"Preprocessing data...")
    print(f"Total original galaxies: {len(total_backgal)}")
    print(f"Total original BCGs / clusters: {len(total_centergal)}")

    # 预处理：按cluster分组
    cluster_list = _preprocess_data(
        total_backgal, total_centergal, mag_name, color_names
    )

    total_clusters = len(cluster_list)
    print(f"Total clusters (with > 3 nearby galaxies): {total_clusters}")

    if total_clusters == 0:
        return pd.DataFrame()

    # 分批处理
    batches = []
    for i in range(0, total_clusters, batch_size):
        batches.append(cluster_list[i:i+batch_size])

    print(f"Split into {len(batches)} batches (batch_size={batch_size})")
    print(f"Processing with {n_jobs} workers...")
    print(f"Random state: {random_state}")

    params = {
        'dcolor_blue': dcolor_blue,
        'dcolor_red': dcolor_red,
        'residual_threshold': residual_threshold,
        'centergal_residual_threshold': centergal_residual_threshold,
        'use_bcg_anchor': use_bcg_anchor,
        'random_state': random_state
    }

    # 并行处理
    if n_jobs == 1:
        all_results = []
        for batch in tqdm(batches, disable=not show_progress):
            batch_results = _process_single_cluster_batch(batch, params)
            all_results.extend(batch_results)
    else:
        batch_results_list = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=5 if show_progress else 0)(
            delayed(_process_single_cluster_batch)(batch, params)
            for batch in batches
        )

        all_results = []
        for batch_results in batch_results_list:
            all_results.extend(batch_results)

    # 构建结果DataFrame
    if all_results:
        all_cids = []
        all_indices = []
        for cid, indices in all_results:
            all_cids.extend([cid] * len(indices))
            all_indices.extend(indices)

        red_seq_all = pd.DataFrame({
            'galaxy_index': all_indices,
            'cid': all_cids
        })
    else:
        red_seq_all = pd.DataFrame()

    print(f"\nResults:")
    print(f"  Clusters with red sequence: {len(all_results)}")
    print(f"  Total red sequence galaxies: {len(red_seq_all)}")
    print(f"  Total input galaxies: {len(total_backgal)}")
    if len(total_backgal) > 0:
        print(f"  Red sequence ratio: {len(red_seq_all)/len(total_backgal)*100:.1f}%")

    return red_seq_all


#======================================================================
# Spatial clustering for member selection
#======================================================================
def hdbscan_cluster_selection(
    red_seq_galaxies,
    bcg_row,
    min_cluster_size=5,
    min_samples=None,
    metric='euclidean',
    cluster_selection_epsilon=0.0,
    select_bcg_cluster=True,
    return_probabilities=True,
    use_physical_coords=True,
    verbose=True
):
    """
    使用HDBSCAN对红序星系进行空间聚类，排除背景污染

    Parameters
    ----------
    red_seq_galaxies : pd.DataFrame
        红序候选星系，需包含'ra', 'dec'列
    bcg_row : pd.DataFrame or pd.Series
        BCG数据（单行），需包含'ra', 'dec'列
    min_cluster_size : int
        最小簇大小。成员数少的团建议5-10，成员多的团建议10-20
    min_samples : int, optional
        核心点的最小邻居数。None则等于min_cluster_size
    metric : str
        距离度量方法
        - 'euclidean': 欧氏距离（适合小范围）
        - 'haversine': 球面距离（适合大范围天球坐标，需弧度制）
    cluster_selection_epsilon : float
        距离阈值。0表示自动选择，可设定物理距离（如0.1度）
    select_bcg_cluster : bool
        是否只选择包含BCG的那个簇
    return_probabilities : bool
        是否返回成员概率
    use_physical_coords : bool
        是否将天球坐标转为投影平面坐标（推荐，避免赤纬扭曲）
    verbose : bool
        是否打印详细信息

    Returns
    -------
    pd.DataFrame
        筛选后的成员星系，包含以下新增列：
        - cluster_label: 聚类标签（-1表示噪声）
        - membership_probability: 成员概率（如果return_probabilities=True）
        - outlier_score: 离群值分数（越高越可能是背景）

    Examples
    --------
    >>> # 单个星系团的红序筛选+空间聚类
    >>> red_seq = select_red_sequence_galaxies(backgal, centergal, ...)
    >>> members = hdbscan_cluster_selection(
    ...     red_seq, centergal,
    ...     min_cluster_size=10,
    ...     verbose=True
    ... )
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("Please install hdbscan: pip install hdbscan")

    if len(red_seq_galaxies) < min_cluster_size:
        if verbose:
            print(f"Warning: Number of red sequence galaxies ({len(red_seq_galaxies)}) < min_cluster_size ({min_cluster_size}), clustering skipped")
        return pd.DataFrame()

    # 提取BCG坐标
    if isinstance(bcg_row, pd.DataFrame):
        if len(bcg_row) != 1:
            raise ValueError(f"bcg_row should be a single row, but has {len(bcg_row)} rows")
        bcg_ra = bcg_row['ra'].values[0]
        bcg_dec = bcg_row['dec'].values[0]
    else:
        bcg_ra = bcg_row['ra']
        bcg_dec = bcg_row['dec']

    # 准备坐标数据
    if use_physical_coords and metric == 'euclidean':
        # 投影到切平面坐标系（以BCG为中心）
        ANG = np.pi / 180.0
        cos_dec = np.cos(ANG * bcg_dec)

        delta_ra = (red_seq_galaxies['ra'].values - bcg_ra) * cos_dec
        delta_dec = red_seq_galaxies['dec'].values - bcg_dec
        coords = np.column_stack([delta_ra, delta_dec])

        if verbose:
            print(f"Using projected coordinates (BCG-centered)")
    elif metric == 'haversine':
        # 天球距离需要弧度制坐标
        coords = np.radians(red_seq_galaxies[['dec', 'ra']].values)  # 注意：haversine需要[lat, lon]
        if verbose:
            print(f"Using haversine metric (spherical distance)")
    else:
        # 直接使用RA/Dec（不推荐，会有赤纬扭曲）
        coords = red_seq_galaxies[['ra', 'dec']].values
        if verbose:
            print(f"Warning: Using raw RA/Dec coordinates may cause declination distortion")

    # HDBSCAN聚类
    if verbose:
        print(f"\nPerforming HDBSCAN clustering...")
        print(f"  Candidate galaxies: {len(red_seq_galaxies)}")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"  min_samples: {min_samples if min_samples else min_cluster_size}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method='eom',  # Excess of Mass
        allow_single_cluster=True  # 允许识别单个簇
    )

    cluster_labels = clusterer.fit_predict(coords)

    # 统计聚类结果
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(cluster_labels == -1)

    if verbose:
        print(f"\nClustering results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        for label in unique_labels:
            if label == -1:
                continue
            count = np.sum(cluster_labels == label)
            print(f"  Cluster {label}: {count} galaxies")

    # 找到包含BCG的簇
    result = red_seq_galaxies.copy()
    result['cluster_label'] = cluster_labels
    result['outlier_score'] = clusterer.outlier_scores_

    if return_probabilities:
        result['membership_probability'] = clusterer.probabilities_

    if select_bcg_cluster:
        # 找到距离BCG最近的星系所在的簇
        if use_physical_coords and metric == 'euclidean':
            dist_to_bcg = np.sqrt(delta_ra**2 + delta_dec**2)
        else:
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            bcg_coord = SkyCoord(ra=bcg_ra*u.deg, dec=bcg_dec*u.deg)
            gal_coords = SkyCoord(
                ra=red_seq_galaxies['ra'].values*u.deg,
                dec=red_seq_galaxies['dec'].values*u.deg
            )
            dist_to_bcg = bcg_coord.separation(gal_coords).deg

        closest_idx = np.argmin(dist_to_bcg)
        bcg_cluster_label = cluster_labels[closest_idx]

        if bcg_cluster_label == -1:
            if verbose:
                print(f"\nWarning: BCG marked as noise! Parameters may be inappropriate or BCG identification incorrect")
                print(f"  BCG position: RA={bcg_ra:.4f}, Dec={bcg_dec:.4f}")
                print(f"  Distance to nearest galaxy: {dist_to_bcg[closest_idx]:.4f} deg")
            # 仍然返回所有非噪声点
            result = result[result['cluster_label'] != -1]
        else:
            if verbose:
                print(f"\nSelected cluster {bcg_cluster_label} (containing BCG)")
            result = result[result['cluster_label'] == bcg_cluster_label]

    if verbose:
        print(f"\nFinal selection: {len(result)} member galaxies")
        if return_probabilities and len(result) > 0:
            print(f"  Membership probability: min={result['membership_probability'].min():.3f}, "
                  f"mean={result['membership_probability'].mean():.3f}, "
                  f"max={result['membership_probability'].max():.3f}")

    return result


def batch_hdbscan_cluster_selection(
    red_seq_all,
    centergal_df,
    min_cluster_size=5,
    min_samples=None,
    n_jobs=-1,
    verbose=True,
    **kwargs
):
    """
    批量对多个星系团的红序星系进行HDBSCAN空间聚类

    Parameters
    ----------
    red_seq_all : pd.DataFrame
        所有星系团的红序星系，需包含'cid'列（星系团ID）
    centergal_df : pd.DataFrame
        所有BCG数据，需包含'cid', 'ra', 'dec'列
    min_cluster_size : int
        最小簇大小，固定值
    min_samples : int, optional
        核心点的最小邻居数
    n_jobs : int
        并行处理的进程数，-1使用所有核心
    verbose : bool
        是否打印进度
    **kwargs : dict
        传递给hdbscan_cluster_selection的其他参数

    Returns
    -------
    pd.DataFrame
        所有星系团的最终成员星系

    Notes
    -----
    如果某个星系团的红序星系数少于min_cluster_size，该星系团会被跳过
    """
    from joblib import Parallel, delayed

    # 按cid分组
    red_seq_groups = red_seq_all.groupby('cid', sort=False)
    bcg_indexed = centergal_df.set_index('cid')

    def process_cluster(cid, red_seq_group):
        if cid not in bcg_indexed.index:
            return None

        # 红序星系数不足，直接跳过
        if len(red_seq_group) < min_cluster_size:
            return None

        bcg = bcg_indexed.loc[[cid]]  # 保持DataFrame格式

        try:
            members = hdbscan_cluster_selection(
                red_seq_group,
                bcg,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                verbose=False,  # 批量处理时关闭详细输出
                **kwargs
            )
            return members
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing cid={cid}: {e}")
            return None

    if verbose:
        print(f"Starting batch HDBSCAN clustering...")
        print(f"  Total clusters: {len(red_seq_groups)}")
        print(f"  Total red sequence galaxies: {len(red_seq_all)}")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"  Parallel workers: {n_jobs if n_jobs > 0 else 'all'}")

    # 并行处理
    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(process_cluster)(cid, group)
        for cid, group in red_seq_groups
    )

    # 合并结果
    valid_results = [r for r in results if r is not None and len(r) > 0]
    n_skipped = len(results) - len(valid_results)

    if len(valid_results) == 0:
        if verbose:
            print("\nWarning: No member galaxies found")
            print(f"  Skipped clusters (red seq < {min_cluster_size}): {n_skipped}")
        return pd.DataFrame()

    final_members = pd.concat(valid_results, ignore_index=True)

    if verbose:
        print(f"\nBatch processing completed:")
        print(f"  Total clusters: {len(red_seq_groups)}")
        print(f"  Successfully processed: {len(valid_results)}")
        print(f"  Skipped clusters (red seq < {min_cluster_size}): {n_skipped}")
        print(f"  Total red sequence galaxies: {len(red_seq_all)}")
        print(f"  Final member galaxies: {len(final_members)}")
        print(f"  Selection rate: {len(final_members)/len(red_seq_all)*100:.1f}%")
        print(f"  Average members per cluster: {len(final_members)/len(valid_results):.1f}")
        if 'membership_probability' in final_members.columns:
            print(f"  Average membership probability: {final_members['membership_probability'].mean():.3f}")

    return final_members


#======================================================================
# Find BCG functions:
# an template usage
#======================================================================
def process_backgal(df):
    df['g_r'] = df['mag_g'] - df['mag_r']
    df['r_i'] = df['mag_r'] - df['mag_i']
    df['i_z'] = df['mag_i'] - df['mag_z']
    df['z_w1'] = df['mag_z'] - df['mag_w1']
    df['w1_w2'] = df['mag_w1'] - df['mag_w2']
    df['g_r_Err'] = np.sqrt(df['mag_g_Err'].values**2 + df['mag_r_Err'].values**2)
    df['r_i_Err'] = np.sqrt(df['mag_r_Err'].values**2 + df['mag_i_Err'].values**2)
    df['i_z_Err'] = np.sqrt(df['mag_i_Err'].values**2 + df['mag_z_Err'].values**2)
    df['z_w1_Err'] = np.sqrt(df['mag_z_Err'].values**2 + df['mag_w1_Err'].values**2)
    df['w1_w2_Err'] = np.sqrt(df['mag_w1_Err'].values**2 + df['mag_w2_Err'].values**2)
    
    df['z'] = df['phot_z'].values
    df['zErr'] = df['phot_zErr'].values
    specz_idx = (df['spec_z'] > 0)
    df.loc[specz_idx, 'z'] = df.loc[specz_idx, 'spec_z']
    df.loc[specz_idx, 'zErr'] = 0.
    return df


def find_cluster_bcg(
    cluster_df: pd.DataFrame,
    backgal_dir_path: Union[str, Path],
    backgal_file_pattern: str = 'mstar_{:02d}.fits',
    num_partitions: int = 72,
    partition_width: float = 5.0,
    boundary_width: float = 1.0,
    cluster_cols: Optional[List[str]] = None,
    backgal_cols: Optional[List[str]] = None,
    prop_cols: Optional[List[str]] = None,
    column_prefix: str = '',
    filter_matched: bool = True,
    deduplicate_by: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    为星系团查找BCG（Brightest Cluster Galaxy）

    此函数将天空按RA分成多个分区，对每个分区：
    1. 选择该分区的星系团数据
    2. 读取对应的背景星系（包括边界补充）
    3. 使用cpp_findBCG查找每个星系团的BCG
    4. 将BCG的属性添加到星系团数据中

    Args:
        cluster_df: 星系团DataFrame，需包含'ra','dec','z','ang_r500'等列
        backgal_dir_path: 背景星系文件所在目录
        backgal_file_pattern: 背景星系文件命名格式，使用{:02d}作为分区ID占位符
        num_partitions: RA分区数量（默认72，即每分区5度）
        partition_width: 每个分区的RA宽度（度）
        boundary_width: 左右边界各补充的RA宽度（度）
        cluster_cols: 传递给cpp_findBCG的星系团列名，默认为['ra','dec','z','ang_r500']
        backgal_cols: 传递给cpp_findBCG的背景星系列名，默认为['uid','ra','dec','z','mag_z']
        prop_cols: 要从背景星系复制的属性列名列表
        column_prefix: 为原始列添加的前缀（如'MCXCII_'），避免与BCG属性列冲突
        filter_matched: 是否过滤掉未匹配到BCG的星系团
        deduplicate_by: 去重时使用的误差列名（如'M500_err'），选择误差最小的星系团
        verbose: 是否打印处理进度

    Returns:
        包含BCG信息的星系团DataFrame，增加了以下列：
        - {prefix}BCG_uid: BCG的唯一标识符
        - {prefix}BCG_ra: BCG的赤经
        - {prefix}BCG_dec: BCG的赤纬
        - 以及prop_cols中指定的所有BCG属性列

    Example:
        >>> # 读取星系团数据
        >>> tab = cu.readfile('../mcxcii.fits')
        >>> tab['dist'] = cu.disofz(tab['z'], rad2ang=True)
        >>> tab['r500'] = cu.m500_to_r500(tab['M500'], tab['z'])
        >>> tab['ang_r500'] = tab['r500'] / tab['dist']
        >>> tab['M500_err'] = (tab['e_M500'] + tab['E_M500']) / 2
        >>> # 查找BCG
        >>> result = find_cluster_bcg(
        ...     cluster_df=tab,
        ...     backgal_dir_path='/path/to/mstar_wen',
        ...     column_prefix='MCXCII_',
        ...     deduplicate_by='M500_err'
        ... )
    """
    from . import cpp_findBCG

    # 设置默认参数
    if cluster_cols is None:
        cluster_cols = ['ra', 'dec', 'z', 'ang_r500']
    if backgal_cols is None:
        backgal_cols = ['uid', 'ra', 'dec', 'z', 'mag_z']
    if prop_cols is None:
        prop_cols = ['z', 'zErr', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_w1', 'mag_w2',
                     'g_r', 'r_i', 'i_z', 'z_w1', 'w1_w2',
                     'mag_g_Err', 'mag_r_Err', 'mag_i_Err', 'mag_z_Err', 'mag_w1_Err', 'mag_w2_Err',
                     'g_r_Err', 'r_i_Err', 'i_z_Err', 'z_w1_Err', 'w1_w2_Err',
                     'mstar']

    # 准备DataFrame
    df = cluster_df.copy()

    # 初始化BCG列
    df['BCG_uid'] = np.nan
    df['BCG_ra'] = np.nan
    df['BCG_dec'] = np.nan

    # 如果指定了列前缀，重命名原有列
    if column_prefix:
        df.columns = [column_prefix + col if col in cluster_df.columns else col
                     for col in df.columns]
        # 更新cluster_cols以匹配新的列名
        prefixed_cluster_cols = [column_prefix + col for col in cluster_cols]
    else:
        prefixed_cluster_cols = cluster_cols

    # 初始化所有BCG属性列
    for col in prop_cols:
        df[col] = -1.

    backgal_dir = Path(backgal_dir_path)
    if not backgal_dir.exists():
        raise FileNotFoundError(f"Background galaxy directory not found: {backgal_dir}")

    total_matched = 0

    # 遍历每个RA分区
    for i in range(num_partitions):
        ra_start = i * partition_width
        ra_end = (i + 1) * partition_width

        # 选择当前分区的星系团
        ra_col = prefixed_cluster_cols[0]  # 'ra' 列（可能带前缀）
        cluster_idx = df[ra_col].between(ra_start, ra_end)
        cluster_array = df[cluster_idx][prefixed_cluster_cols].values

        if len(cluster_array) == 0:
            if verbose:
                print(f'No clusters in R.A. range {ra_start:.1f} - {ra_end:.1f}.')
            continue

        # 读取主背景星系数据
        backgal_path = backgal_dir / backgal_file_pattern.format(i)
        if not backgal_path.exists():
            print(f"Warning: Background galaxy file not found: {backgal_path}")
            continue

        backgal = readfile(backgal_path)
        backgal = process_backgal(backgal)

        # 添加边界补充星系（左右各boundary_width度）
        boundaries = [
            ((i - 1) % num_partitions,
             ra_start - boundary_width if i > 0 else 360 - boundary_width,
             ra_start if i > 0 else 360),
            ((i + 1) % num_partitions,
             ra_end if i < num_partitions - 1 else 0,
             ra_end + boundary_width if i < num_partitions - 1 else boundary_width)
        ]

        for fid, ra_min, ra_max in boundaries:
            supp_backgal_path = backgal_dir / backgal_file_pattern.format(fid)
            if not supp_backgal_path.exists():
                continue

            supp_backgal = readfile(supp_backgal_path)
            idx = (supp_backgal['ra'] >= ra_min) & (supp_backgal['ra'] < ra_max)
            if idx.sum() > 0:
                supp_backgal = process_backgal(supp_backgal[idx])
                backgal = pd.concat([backgal, supp_backgal], ignore_index=True)

        backgal_array = backgal[backgal_cols].values

        # 调用C++函数查找BCG
        result = cpp_findBCG(cluster_array, backgal_array)

        # 更新BCG位置信息
        bcg_uid_col = column_prefix + 'BCG_uid' if column_prefix else 'BCG_uid'
        bcg_ra_col = column_prefix + 'BCG_ra' if column_prefix else 'BCG_ra'
        bcg_dec_col = column_prefix + 'BCG_dec' if column_prefix else 'BCG_dec'

        df.loc[cluster_idx, bcg_uid_col] = result[:, 0]
        df.loc[cluster_idx, bcg_ra_col] = result[:, 1]
        df.loc[cluster_idx, bcg_dec_col] = result[:, 2]

        # 匹配BCG属性
        cluster_indices = df[cluster_idx].index
        uids = result[:, 0]
        backgal_indexed = backgal.set_index('uid')

        matched_count = 0
        for df_idx, uid in zip(cluster_indices, uids):
            if uid in backgal_indexed.index:
                df.loc[df_idx, prop_cols] = backgal_indexed.loc[uid, prop_cols].values
                matched_count += 1

        total_matched += matched_count
        if verbose:
            print(f'Partition {i+1}/{num_partitions} done. Matched {matched_count} BCGs.')

    if verbose:
        print(f'All partitions processed. Total matched BCGs: {total_matched}')

    # 过滤未匹配的星系团
    if filter_matched:
        bcg_ra_col = column_prefix + 'BCG_ra' if column_prefix else 'BCG_ra'
        idx = df[bcg_ra_col] > 0

        # 如果有M500列，也进行过滤
        m500_col = column_prefix + 'M500' if column_prefix else 'M500'
        if m500_col in df.columns:
            idx &= df[m500_col] > 0

        df = df[idx]
        if verbose:
            print(f'After filtering, {len(df)} clusters remain.')

    # 去重处理
    if deduplicate_by is not None:
        bcg_uid_col = column_prefix + 'BCG_uid' if column_prefix else 'BCG_uid'
        dedup_col = column_prefix + deduplicate_by if column_prefix and deduplicate_by in cluster_df.columns else deduplicate_by

        if dedup_col in df.columns:
            df = df.loc[df.groupby(bcg_uid_col)[dedup_col].idxmin()]
            if verbose:
                print(f'After deduplication by {dedup_col}, {len(df)} clusters remain.')
        else:
            print(f"Warning: Deduplication column '{dedup_col}' not found. Skipping deduplication.")

    return df

#======================================================================
# Add survey photometry data functions:
# an template usage
#======================================================================
def process_backgal(df):
    df['g_r'] = df['mag_g'] - df['mag_r']
    df['r_i'] = df['mag_r'] - df['mag_i']
    df['i_z'] = df['mag_i'] - df['mag_z']
    df['z_w1'] = df['mag_z'] - df['mag_w1']
    df['w1_w2'] = df['mag_w1'] - df['mag_w2']
    df['g_r_Err'] = np.sqrt(df['mag_g_Err'].values**2 + df['mag_r_Err'].values**2)
    df['r_i_Err'] = np.sqrt(df['mag_r_Err'].values**2 + df['mag_i_Err'].values**2)
    df['i_z_Err'] = np.sqrt(df['mag_i_Err'].values**2 + df['mag_z_Err'].values**2)
    df['z_w1_Err'] = np.sqrt(df['mag_z_Err'].values**2 + df['mag_w1_Err'].values**2)
    df['w1_w2_Err'] = np.sqrt(df['mag_w1_Err'].values**2 + df['mag_w2_Err'].values**2)
    
    df['z'] = df['phot_z'].values
    df['zErr'] = df['phot_zErr'].values
    specz_idx = (df['spec_z'] > 0)
    df.loc[specz_idx, 'z'] = df.loc[specz_idx, 'spec_z']
    df.loc[specz_idx, 'zErr'] = 0.
    return df

def add_photometry_data(
    cluster_df,
    backgal_dir_path='/home/tiandc/Data/LegacySurveys/DR10/mstar_wen',
    backgal_file_pattern='mstar_{:02d}.fits',
    uid_col='uid',
    phot_cols=None,
    output_path=None
):
    """
    为星系团目录添加BCG的测光数据
    
    Parameters
    ----------
    cluster_df : pandas.DataFrame
        星系团目录数据框
    backgal_data_dir : str
        星系数据目录路径
    backgal_file_pattern : str
        星系数据文件名模式
    uid_col : str
        用于匹配的uid列名
    phot_cols : list, optional
        需要提取的测光列名。如果为None，使用默认列
    output_path : str, optional
        输出文件路径
    
    Returns
    -------
    pandas.DataFrame
        添加了测光数据的星系团目录
    """
    # 默认的测光列
    if phot_cols is None:
        phot_cols = ['z', 'zErr', 'mag_g', 'mag_r', 'mag_i', 'mag_z',
                     'mag_w1', 'mag_w2', 'g_r', 'r_i', 'i_z', 
                     'z_w1', 'w1_w2', 'mstar']
    
    # 读取星系团目录并设置索引
    df = cluster_df.set_index(uid_col)
    
    # 循环读取72个分区的星系数据
    total_updated = 0
    for i in range(72):
        backgal_path = os.path.join(backgal_dir_path, backgal_file_pattern.format(i))
        
        # 检查文件是否存在
        if not os.path.exists(backgal_path):
            print(f'Warning: File {backgal_path} not found, skipping...')
            continue
        
        backgal_df = cu.readfile(backgal_path).set_index(uid_col)
        backgal_df = process_backgal(backgal_df)
        
        # 找到共同的uid
        common_idx = backgal_df.index.intersection(df.index)
        
        if len(common_idx) > 0:
            df.loc[common_idx, phot_cols] = backgal_df.loc[common_idx, phot_cols].values
            total_updated += len(common_idx)
        
        print(f'File {i:02d} processed. Updated {len(common_idx)} records.')
    
    print(f'\nTotal records updated: {total_updated}')
    
    # 重置索引
    df = df.reset_index()
    
    # 保存文件
    if output_path is not None:
        cu.savefile(df, output_path)
        print(f'Saved to {output_path}')
    
    return df

#======================================================================
# Search nearby galaxies functions:
# an template usage
#======================================================================
def add_dis_range(df, search_radius_mpc=1.5):
    """
    添加距离相关列

    Parameters
    ----------
    df : pd.DataFrame
        包含星系团数据的DataFrame
    search_radius_mpc : float, optional
        搜索半径（单位：Mpc），默认为1.5

    Returns
    -------
    pd.DataFrame
        添加了距离列的DataFrame
    """
    df = df.copy()
    df.loc[:, 'dist_ang'] = disofz(df['z'].values, rad2ang=True)
    df['ang_r500'] = df['r500'] / df['dist_ang']
    df.loc[:, 'ang_1mpc'] = 1. / df['dist_ang'].values
    df.loc[:, 'ang_1.5mpc'] = 1.5 / df['dist_ang'].values
    df.loc[:, 'ang_2mpc'] = 2. / df['dist_ang'].values

    # 添加自定义搜索半径列
    if search_radius_mpc not in [1.0, 1.5, 2.0]:
        col_name = f'ang_{search_radius_mpc}mpc'
        df.loc[:, col_name] = search_radius_mpc / df['dist_ang'].values

    return df

def process_backgal(df):
    df['g_r'] = df['mag_g'] - df['mag_r']
    df['r_i'] = df['mag_r'] - df['mag_i']
    df['i_z'] = df['mag_i'] - df['mag_z']
    df['z_w1'] = df['mag_z'] - df['mag_w1']
    df['w1_w2'] = df['mag_w1'] - df['mag_w2']
    df['g_r_Err'] = np.sqrt(df['mag_g_Err']**2 + df['mag_r_Err']**2)
    df['r_i_Err'] = np.sqrt(df['mag_r_Err']**2 + df['mag_i_Err']**2)
    df['i_z_Err'] = np.sqrt(df['mag_i_Err']**2 + df['mag_z_Err']**2)
    df['z_w1_Err'] = np.sqrt(df['mag_z_Err']**2 + df['mag_w1_Err']**2)
    df['w1_w2_Err'] = np.sqrt(df['mag_w1_Err']**2 + df['mag_w2_Err']**2)
    
    df['z'] = df['phot_z']
    df['zErr'] = df['phot_zErr']
    specz_idx = (df['spec_z'] > 0)
    df.loc[specz_idx, 'z'] = df.loc[specz_idx, 'spec_z']
    df.loc[specz_idx, 'zErr'] = 0.
    return df

def process_member_galaxies(backgal_df, center_df, indices_1mpc, indices_r500):
    all_members = []
    
    for i, (mpc_idx, r500_idx) in enumerate(zip(indices_1mpc, indices_r500)):
        # Get cluster ID
        cluster_id = center_df.iloc[i]['cid']
        
        # Extract member galaxies
        nearbygals = backgal_df.iloc[mpc_idx].copy()
        
        # Add cluster ID
        nearbygals['cid'] = cluster_id.astype('int')
        
        # Add labels (1 for R500 members, 0 for others)
        nearbygals['ifInR500'] = 0
        
        # Convert r500 position indices to actual DataFrame indices
        r500_actual_idx = backgal_df.iloc[r500_idx].index
        nearbygals.loc[nearbygals.index.isin(r500_actual_idx), 'ifInR500'] = 1
        
        all_members.append(nearbygals)
    
    # Combine all members
    return pd.concat(all_members, ignore_index=True)


def _process_single_partition(args):
    """
    处理单个RA分区的辅助函数（用于并行处理）

    Parameters
    ----------
    args : tuple
        包含所有需要的参数

    Returns
    -------
    pd.DataFrame or None
        成员星系DataFrame，如果没有找到则返回None
    """
    (fid, tab, backgal_dir, backgal_file_pattern, num_partitions,
     partition_width, boundary_width, centergal_cols, verbose) = args

    from . import cpp_search

    ra_start = fid * partition_width
    ra_end = (fid + 1) * partition_width

    # 选择当前分区的星系团
    cluster_idx = tab['ra'].between(ra_start, ra_end)
    subtab = tab[cluster_idx].copy()

    if subtab.shape[0] == 0:
        if verbose:
            print(f'[Partition {fid:02d}] No clusters in R.A. range {ra_start:.1f} - {ra_end:.1f}.')
        return None

    # 读取主背景星系数据
    backgal_path = backgal_dir / backgal_file_pattern.format(fid)
    if not backgal_path.exists():
        print(f"[Partition {fid:02d}] Warning: Background galaxy file not found: {backgal_path}")
        return None

    backgal = readfile(backgal_path)
    backgal = process_backgal(backgal)

    # 添加边界补充星系（左右各boundary_width度）
    boundaries = [
        ((fid - 1) % num_partitions,
         ra_start - boundary_width if fid > 0 else 360 - boundary_width,
         ra_start if fid > 0 else 360),
        ((fid + 1) % num_partitions,
         ra_end if fid < num_partitions - 1 else 0,
         ra_end + boundary_width if fid < num_partitions - 1 else boundary_width)
    ]

    for supp_fid, ra_min, ra_max in boundaries:
        supp_backgal_path = backgal_dir / backgal_file_pattern.format(supp_fid)
        if not supp_backgal_path.exists():
            continue

        supp_backgal = readfile(supp_backgal_path)
        supp_idx = (supp_backgal['ra'] >= ra_min) & (supp_backgal['ra'] < ra_max)
        if supp_idx.sum() > 0:
            supp_backgal = process_backgal(supp_backgal[supp_idx])
            backgal = pd.concat([backgal, supp_backgal], ignore_index=True)

    # 准备搜索数组
    centergal_array = subtab[centergal_cols].values
    background_array = backgal[['ra', 'dec', 'z']].values

    # 搜索邻近星系
    indices_nearby, indices_r500 = cpp_search(
        centergal_array,
        background_array,
        mode='train'
    )

    # 处理成员星系
    member_galaxies = process_member_galaxies(backgal, subtab, indices_nearby, indices_r500)

    if member_galaxies.shape[0] > 0:
        if verbose:
            print(f'[Partition {fid:02d}] Done: {len(subtab)} clusters, '
                  f'{member_galaxies.shape[0]} member galaxies')
        return member_galaxies
    else:
        if verbose:
            print(f'[Partition {fid:02d}] No member galaxies in R.A. range {ra_start:.1f} - {ra_end:.1f}.')
        return None


def search_nearby_galaxies(
    cluster_df: pd.DataFrame,
    backgal_dir_path: Union[str, Path],
    backgal_file_pattern: str = 'mstar_{:02d}.fits',
    num_partitions: int = 72,
    partition_width: float = 5.0,
    boundary_width: float = 1.0,
    search_radius_mpc: float = 1.5,
    centergal_cols: Optional[List[str]] = None,
    compute_distances: bool = True,
    normalize_distances: bool = True,
    n_jobs: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    搜索星系团成员星系（支持并行处理）

    Parameters
    ----------
    cluster_df : pd.DataFrame
        星系团目录DataFrame
    backgal_dir_path : Union[str, Path]
        背景星系数据目录路径
    backgal_file_pattern : str
        背景星系文件名模式，默认为 'mstar_{:02d}.fits'
    num_partitions : int
        RA分区数量，默认为72
    partition_width : float
        每个分区的宽度（度），默认为5.0
    boundary_width : float
        边界补充宽度（度），默认为1.0
    search_radius_mpc : float
        搜索半径（Mpc），默认为1.5
    centergal_cols : Optional[List[str]]
        中心星系列名列表，如果为None则自动生成
    compute_distances : bool
        是否计算距离，默认为True
    normalize_distances : bool
        是否归一化距离，默认为True
    n_jobs : int
        并行处理的进程数。1表示串行处理，-1表示使用所有CPU核心，默认为1
    verbose : bool
        是否输出详细信息，默认为True

    Returns
    -------
    pd.DataFrame
        成员星系DataFrame
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # 根据搜索半径生成列名
    ang_col = f'ang_{search_radius_mpc}mpc'

    # 设置默认参数
    if centergal_cols is None:
        centergal_cols = ['ra', 'dec', 'z', ang_col, 'ang_r500']

    # 准备星系团数据
    tab = cluster_df.copy()

    # 检查是否已经包含必要的列，如果没有则添加
    required_cols = [ang_col, 'ang_r500', 'dist_ang']
    missing_cols = [col for col in required_cols if col not in tab.columns]
    if missing_cols:
        if verbose:
            print(f"Adding distance range columns: {missing_cols}")
        tab = add_dis_range(tab, search_radius_mpc=search_radius_mpc)

    backgal_dir = Path(backgal_dir_path)
    if not backgal_dir.exists():
        raise FileNotFoundError(f"Background galaxy directory not found: {backgal_dir}")

    # 确定使用的进程数
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    if verbose:
        if n_jobs > 1:
            print(f"Using {n_jobs} parallel processes to search {num_partitions} partitions")
        else:
            print(f"Processing {num_partitions} partitions sequentially")

    # 准备所有分区的参数
    partition_args = [
        (fid, tab, backgal_dir, backgal_file_pattern, num_partitions,
         partition_width, boundary_width, centergal_cols, verbose)
        for fid in range(num_partitions)
    ]

    # 并行或串行处理
    all_results = []

    if n_jobs > 1:
        # 并行处理
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_process_single_partition, args): args[0]
                      for args in partition_args}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.append(result)
    else:
        # 串行处理
        for args in partition_args:
            result = _process_single_partition(args)
            if result is not None:
                all_results.append(result)

    # 合并所有结果
    if len(all_results) == 0:
        if verbose:
            print("No member galaxies found")
        return pd.DataFrame()

    nearbygal = pd.concat(all_results, ignore_index=True)

    # 计算距离指标
    if compute_distances and len(nearbygal) > 0:
        if verbose:
            print("Computing distances to BCG...")

        ANG = np.pi / 180.0
        cluster_cols = ['cid', 'ra', 'dec', 'z', 'dist_ang']
        bcgs = tab[cluster_cols].set_index('cid')
        bcgs_data = bcgs.loc[nearbygal['cid'].values]

        # 计算角距离
        delta_ra = bcgs_data['ra'].values - nearbygal['ra'].values
        delta_dec = bcgs_data['dec'].values - nearbygal['dec'].values
        cos_dec = np.cos(ANG * bcgs_data['dec'].values)
        ang_sep_sq = delta_ra**2 * cos_dec**2 + delta_dec**2

        nearbygal['angSepToBCG_deg'] = np.sqrt(ang_sep_sq)
        nearbygal['projDistToBCG_mpc'] = bcgs_data['dist_ang'].values * nearbygal['angSepToBCG_deg']
        nearbygal['zdeltaToBCG'] = nearbygal['z'].values - bcgs_data['z'].values

        # 归一化距离
        if normalize_distances:
            max_angSep = nearbygal['angSepToBCG_deg'].max()
            max_projDist = nearbygal['projDistToBCG_mpc'].max()
            nearbygal['angSepToBCG_norm'] = nearbygal['angSepToBCG_deg'] / max_angSep
            nearbygal['projDistToBCG_norm'] = nearbygal['projDistToBCG_mpc'] / max_projDist

    if verbose:
        print(f"\nTotal member galaxies found: {len(nearbygal)}")

    return nearbygal


#======================================================================
# make SOM:
# an template usage
#======================================================================
def makeSOM(num_channels, som_r, som_sig, memgal_r, centergal, backgal, output_path):
    """ make h5 file for stellar mass map"""
    centergal_cols = ['cid', 'ra', 'dec', 'z', 'mag_z']
    backgal_cols = ['ra', 'dec', 'z', 'mstar', 'mag_z', 'mag_r']
    
    with h5py.File(output_path, 'w') as h5f:
        # 创建数据集
        h5f.create_dataset('mstar_map', shape=(0, num_channels, 200, 200), 
                            maxshape=(None, num_channels, 200, 200), 
                            dtype=np.float32, chunks=True)
        h5f.create_dataset('label', shape=(0,), maxshape=(None,), 
                            dtype=np.float32, chunks=True)
        h5f.create_dataset('redshift', shape=(0,), maxshape=(None,), 
                            dtype=np.float32, chunks=True)
        h5f.create_dataset('cid', shape=(0,), maxshape=(None,), 
                            dtype=np.int64, chunks=True)
    
    for i in range(72):
        # 从 centergal 中取出对应 RA 范围
        idx = (centergal['ra'] > 5*i) & (centergal['ra'] <= 5*(i+1))
        batch_centergal = centergal[idx]
        if batch_centergal.empty:
            print(f"Skip fid {i}: no cluster in this RA range.")
            continue  # 跳过空数据
        batch_centergal_array = batch_centergal[centergal_cols].values
        
        idx = (backgal['ra'] > 5*i) & (backgal['ra'] <= 5*(i+1))
        # 处理跨 0 度的边界情况
        if i == 0:
            idx |= (backgal['ra'] > 359) & (backgal['ra'] <= 360)
        elif i == 71:
            idx |= (backgal['ra'] > 0) & (backgal['ra'] <= 1)
        batch_backgal = backgal[idx]
        backgal_array = batch_backgal[backgal_cols].values
        
        # 调用 C++ 函数，result.shape = (num_cg, num_channels, 200, 200)
        mstar_map = cpp_som(backgal_array, 
                            batch_centergal_array,  
                            mode='MstarMap', 
                            som_channel=num_channels,
                            memgal_r=memgal_r,
                            som_r=som_r,
                            som_sig=som_sig,
                            use_log_mstar=False,
                            nthreads=None,
                            which_server='MCMC')

        # 追加数据到已存在的文件
        with h5py.File(output_path, 'a') as h5f:
            for dataset, data in zip(['mstar_map', 'label', 'redshift', 'cid'],
                                  [mstar_map,
                                   batch_centergal['eROSITA_M500'].values,
                                   batch_centergal['z'].values, 
                                   batch_centergal['cid'].values]):
                h5f[dataset].resize((h5f[dataset].shape[0] + data.shape[0]), axis=0)
                h5f[dataset][-data.shape[0]:] = data

    print(f"Finished and saved to {output_path}")


#======================================================================
# STILTS Cross-Match
#======================================================================
def stilts_crossmatch(
    input1: Union[str, Path],
    input2: Union[str, Path],
    output_file: Union[str, Path],
    stilts_path: Union[str, Path] = "/home/tiandc/download/stilts.jar",
    match_radius: float = 1.0,
    values1: str = "ra dec",
    values2: str = "ra dec",
    join: str = "1and2",
    find: str = "best",
    heap_size: str = "128G",
    tmp_dir: Optional[str] = "/home/tiandc/tmp",
    runner: str = "parallel16",
    progress: str = "log",
    verbose: bool = True
) -> bool:
    """
    使用STILTS进行两个FITS表格的天球坐标交叉匹配

    Parameters
    ----------
    input1 : Union[str, Path]
        第一个输入FITS文件路径
    input2 : Union[str, Path]
        第二个输入FITS文件路径
    output_file : Union[str, Path]
        输出FITS文件路径
    stilts_path : Union[str, Path], optional
        STILTS JAR文件路径，默认为 "/home/tiandc/download/stilts.jar"
    match_radius : float, optional
        匹配半径（角秒），默认为 1.0
    values1 : str, optional
        第一个表的坐标列名（空格分隔的RA和Dec列名），默认为 "ra dec"
    values2 : str, optional
        第二个表的坐标列名（空格分隔的RA和Dec列名），默认为 "ra dec"
    join : str, optional
        匹配类型，可选值：
        - "1and2": 内连接，只保留两表都有的匹配（默认）
        - "1or2": 外连接，保留所有记录
        - "all1": 左连接，保留第一个表的所有记录
        - "all2": 右连接，保留第二个表的所有记录
        - "1not2": 保留第一个表中未匹配的记录
        - "2not1": 保留第二个表中未匹配的记录
        - "1xor2": 保留两表中未匹配的记录
    find : str, optional
        匹配模式，可选值：
        - "best": 只保留最佳匹配（默认）
        - "best1": 每个input1记录的最佳匹配
        - "best2": 每个input2记录的最佳匹配
        - "all": 保留所有匹配
    heap_size : str, optional
        Java堆内存大小，默认为 "128G"
    tmp_dir : str, optional
        Java临时目录路径，默认为 "/home/tiandc/tmp"
        设置为 None 则使用系统默认临时目录
    runner : str, optional
        运行模式，例如 "parallel16" 表示使用16线程并行，默认为 "parallel16"
        可选值包括 "sequential"（串行）、"parallel"（自动并行）、"parallelN"（N线程）
    progress : str, optional
        进度显示模式，可选值：
        - "log": 显示日志进度（默认）
        - "none": 不显示进度
        - "time": 显示时间进度
    verbose : bool, optional
        是否打印详细信息，默认为 True

    Returns
    -------
    bool
        匹配是否成功

    Examples
    --------
    >>> # 基本用法：匹配两个FITS文件
    >>> stilts_crossmatch(
    ...     input1="catalog1.fits",
    ...     input2="catalog2.fits",
    ...     output_file="matched.fits",
    ...     match_radius=1.0,
    ...     values1="ra dec",
    ...     values2="RA DEC"
    ... )

    >>> # 自定义匹配参数
    >>> stilts_crossmatch(
    ...     input1="/path/to/ps1.fits",
    ...     input2="/path/to/legacy.fits",
    ...     output_file="/path/to/output.fits",
    ...     match_radius=2.0,
    ...     values1="raStack decStack",
    ...     values2="ra dec",
    ...     join="all1",  # 左连接
    ...     find="best",
    ...     heap_size="64G",
    ...     runner="parallel8"
    ... )

    Notes
    -----
    - 需要预先安装Java运行环境
    - STILTS下载地址：http://www.starlink.ac.uk/stilts/
    - 对于大文件，建议设置足够大的heap_size和专用的tmp_dir
    - 匹配使用的是天球距离（sky matcher）
    """
    import subprocess

    input1 = str(input1)
    input2 = str(input2)
    output_file = str(output_file)
    stilts_path = str(stilts_path)

    # 检查STILTS是否存在
    if not os.path.exists(stilts_path):
        raise FileNotFoundError(f"STILTS JAR文件不存在: {stilts_path}")

    # 检查输入文件是否存在
    if not os.path.exists(input1):
        raise FileNotFoundError(f"输入文件1不存在: {input1}")
    if not os.path.exists(input2):
        raise FileNotFoundError(f"输入文件2不存在: {input2}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 构建Java命令
    java_opts = [
        "java",
        f"-Xmx{heap_size}"
    ]

    if tmp_dir is not None:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)
        java_opts.append(f"-Djava.io.tmpdir={tmp_dir}")

    java_opts.extend(["-jar", stilts_path])

    # 构建STILTS tmatch2命令
    cmd = java_opts + [
        "tmatch2",
        f"runner={runner}",
        f"in1={input1}",
        f"in2={input2}",
        f"out={output_file}",
        "matcher=sky",
        f"params={match_radius}",
        f"values1={values1}",
        f"values2={values2}",
        f"join={join}",
        f"find={find}",
        f"progress={progress}"
    ]

    if verbose:
        print("正在开始交叉匹配，请耐心等待...")
        print(f"输入文件1: {input1}")
        print(f"输入文件2: {input2}")
        print(f"输出文件: {output_file}")
        print(f"匹配半径: {match_radius} arcsec")
        print(f"匹配类型: join={join}, find={find}")
        print(f"执行命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        if verbose:
            print(f"\n匹配成功！结果已保存至: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n匹配失败，错误代码: {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n错误: 未找到Java运行环境，请确保已安装Java")
        return False


# ============================================================
# SHAP Feature Importance Analysis
# ============================================================

def compute_shap_importance(
    inference,
    data_path: str,
    n_background: int = 200,
    n_samples: int = 2000,
    random_state: int = 42,
    save_dir: Optional[str] = None,
    plot: bool = True,
    plot_save_path: Optional[str] = None,
):
    """
    使用 SHAP (KernelExplainer) 计算模型的特征重要性，并可选保存结果和绘图。

    Parameters
    ----------
    inference : PhotozBinInference
        已加载模型的推理对象（需要 .model, .device, ._create_dataset 方法）。
    data_path : str
        数据文件路径（FITS 等），用于创建数据集。
    n_background : int
        KernelExplainer 的背景样本数量。
    n_samples : int
        计算 SHAP 值的样本数量。
    random_state : int
        随机种子。
    save_dir : str, optional
        若提供，将 SHAP 结果保存为 ``{save_dir}/shap_feature_importance.h5``。

    Returns
    -------
    dict
        包含以下键:
        - ``shap_values``  : np.ndarray [n_samples, n_features]
        - ``X_test``       : np.ndarray [n_samples, n_features]
        - ``feature_names``: list[str]
        - ``mean_abs_shap``: np.ndarray [n_features]
        - ``importance_df`` : pd.DataFrame（按重要性排序）
    """
    import shap
    import torch

    np.random.seed(random_state)

    # ---- 加载数据 ----
    ds = inference._create_dataset(data_path, mode='validation')
    X = ds.features
    feature_names = ds.feature_cols
    bin_centers = ds.bin_centers

    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"背景样本: {n_background}, 测试样本: {n_samples}")

    bg_idx = np.random.choice(len(X), min(n_background, len(X)), replace=False)
    test_idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_background = X[bg_idx]
    X_test = X[test_idx]

    # ---- 预测函数（输出红移期望值）----
    def predict_fn(x):
        x_tensor = torch.FloatTensor(x).to(inference.device)
        inference.model.eval()
        with torch.no_grad():
            logits = inference.model(x_tensor).cpu().numpy()
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        return (probs * bin_centers).sum(axis=1)

    # ---- 计算 SHAP 值 ----
    print("\n创建 SHAP 解释器...")
    explainer = shap.KernelExplainer(predict_fn, X_background)
    print("计算 SHAP 值（这可能需要几分钟）...")
    shap_values = explainer.shap_values(X_test, silent=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # ---- 打印排名 ----
    print("\n特征重要性 (Mean |SHAP|):")
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for rank, idx in enumerate(sorted_idx):
        print(f"  [{rank + 1:2d}] {feature_names[idx]:20s}: {mean_abs_shap[idx]:.6f}")

    # ---- 构建 DataFrame ----
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap,
        'importance_normalized': mean_abs_shap / mean_abs_shap.max(),
    })
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = importance_df.index + 1

    # ---- 保存 ----
    if save_dir is not None:
        file_path = os.path.join(save_dir, "shap_feature_importance.h5")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('shap_values', data=shap_values, compression='gzip')
            f.create_dataset('X_test', data=X_test, compression='gzip')
            f.create_dataset('feature_names', data=np.array(feature_names, dtype='S'))
            f.create_dataset('mean_abs_shap', data=mean_abs_shap)
            f.create_dataset('mean_abs_shap_normalized',
                             data=mean_abs_shap / mean_abs_shap.max())
        print(f"\nSHAP 结果已保存到: {file_path}")

    return {
        'shap_values': shap_values,
        'X_test': X_test,
        'feature_names': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'importance_df': importance_df,
    }
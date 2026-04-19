#!/usr/bin/env python
"""
测光红移预测脚本

支持 LSDR10 和 PS1DR2 两种巡天数据的批量预测。

Usage:
    python predict.py --survey lsdr10
    python predict.py --survey ps1dr2
    python predict.py --survey all
    python predict.py --survey merge
    python predict.py --survey publish
    python predict.py --survey check
"""

import numpy as np
import pandas as pd
import os
import glob
import gc
import argparse
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import h5py
from astropy.table import Table
from astropy.io import fits
from photoz_bin import PhotozBinInference
from galaxyClf_ann import GalaxyClassificationInference


# ========================== 文件读写函数 ========================== #

PDF_UINT16_SCALE = np.uint32(np.iinfo(np.uint16).max)
PDF_BINS = 40
PDF_BIN_EDGES = np.linspace(0.0, 2.0, PDF_BINS + 1, dtype=np.float32)

CATALOG_BASE_DIR = '/home/tiandc/photoz/catalog'
MODEL_BASE_DIR = os.path.join(CATALOG_BASE_DIR, 'models')
GALAXY_CLF_LS_DIR = os.path.join(MODEL_BASE_DIR, 'galaxy_clf_ls')
GALAXY_CLF_PS_DIR = os.path.join(MODEL_BASE_DIR, 'galaxy_clf_ps')

LSDR10_INPUT_DIR = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy_clean'
LSDR10_OUTPUT_DIR = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy_clean_photozClf'
PS1DR2_INPUT_DIR = '/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE_clean'
PS1DR2_OUTPUT_DIR = '/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE_clean_photozClf'

MERGE_DIR = os.path.join(CATALOG_BASE_DIR, 'merge')
MERGE_MAIN_DIR = os.path.join(MERGE_DIR, 'main')
MERGE_PDF_DIR = os.path.join(MERGE_DIR, 'pdf')
PUBLISH_DIR = os.path.join(CATALOG_BASE_DIR, 'publish')

LSDR10_OUTPUT_TEMPLATE = 'lsdr9x10_{fid:02d}_xPS1DR2i_clean_photoz.h5'
PS1DR2_OUTPUT_TEMPLATE = 'ps1dr2_{fid:02d}_xunWISE_clean_photoz.h5'
MERGE_MAIN_TEMPLATE = 'RA{ra_start:03d}_{ra_end:03d}_main.fits'
MERGE_PDF_TEMPLATE = 'RA{ra_start:03d}_{ra_end:03d}_pdf.h5'
PUBLISH_MAIN_TEMPLATE = 'catalog_main_RA{ra_start:03d}_{ra_end:03d}.fits.gz'
PUBLISH_PDF_TEMPLATE = 'catalog_pdf_RA{ra_start:03d}_{ra_end:03d}.h5'

N_RA_BINS = 72
MERGE_RA_STEP = 5
PUBLISH_RA_STEP = 30

CATALOG_MAIN_COLUMNS = [
    'uid',
    'ra', 'dec',
    'z_phot_mean',
    'z_phot_std',
    'z_phot_mode',
    'z_phot_median',
    'z_phot_l68', 'z_phot_u68',
    'z_phot_l95', 'z_phot_u95',
    'priority',
    'survey',
    'galaxy_prob',
]
CATALOG_SCALAR_COLUMNS = [col for col in CATALOG_MAIN_COLUMNS if col != 'uid']
CATALOG_PDF_COLUMNS = ['uid', 'z_phot_pdf']


def read_fits(filepath: str) -> pd.DataFrame:
    """读取 FITS 文件为 DataFrame"""
    table = Table.read(filepath, format='fits')
    return table.to_pandas()


def save_fits(df: pd.DataFrame, filepath: str) -> None:
    """保存 DataFrame 为 FITS 文件"""
    # 确保整数类型列使用 int16 而不是 int8（避免类型转换问题）
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == np.int8:
            df[col] = df[col].astype(np.int16)
        elif df[col].dtype == np.uint16:
            df[col] = df[col].astype(np.int32)
        elif df[col].dtype == object and len(df[col]) > 0:
            first_valid = next((v for v in df[col] if v is not None), None)
            if isinstance(first_valid, np.ndarray) and first_valid.dtype == np.uint16:
                df[col] = [np.asarray(v, dtype=np.int32) if v is not None else v for v in df[col]]

    table = Table.from_pandas(df)
    table.write(filepath, format='fits', overwrite=True)


def encode_pdf_to_uint16(probs: np.ndarray) -> np.ndarray:
    """
    将归一化 PDF 量化为 uint16，并保证每行整数和精确等于 65535。
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array")

    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each PDF row must have positive total probability")
    probs = probs / row_sums

    scaled = probs * PDF_UINT16_SCALE
    base = np.floor(scaled).astype(np.uint32)
    frac = scaled - base
    deficit = (PDF_UINT16_SCALE - base.sum(axis=1)).astype(np.int64)

    max_deficit = int(deficit.max())
    if max_deficit > 0:
        top_idx = np.argpartition(-frac, kth=max_deficit - 1, axis=1)[:, :max_deficit]
        add_mask = np.arange(max_deficit)[None, :] < deficit[:, None]
        row_idx = np.broadcast_to(np.arange(len(probs))[:, None], top_idx.shape)[add_mask]
        col_idx = top_idx[add_mask]
        base[row_idx, col_idx] += 1

    return base.astype(np.uint16)


def save_hdf5(df: pd.DataFrame, filepath: str) -> None:
    """保存 DataFrame 为 HDF5 文件。"""
    df = df.copy()
    with h5py.File(filepath, 'w') as f:
        meta = f.create_group('meta')
        data_group = f.create_group('data')

        meta.attrs['format'] = 'photoz_table_v2'
        meta.attrs['pdf_uint16_scale'] = int(PDF_UINT16_SCALE)
        meta.create_dataset('columns', data=np.array(df.columns.tolist(), dtype='S'))
        if 'z_phot_pdf' in df.columns:
            meta.attrs['pdf_bins'] = PDF_BINS
            meta.create_dataset('pdf_bin_edges', data=PDF_BIN_EDGES)

        for col in df.columns:
            values = df[col].values
            if col == 'z_phot_pdf':
                pdf_array = np.stack([np.asarray(v, dtype=np.uint16) for v in values], axis=0)
                data_group.create_dataset(col, data=pdf_array, compression='gzip', shuffle=True)
            else:
                data_group.create_dataset(col, data=values, compression='gzip', shuffle=True)


def _is_valid_prediction_hdf5(filepath: str, required_columns: list) -> bool:
    """检查输出 HDF5 是否结构完整，可用于决定是否跳过。"""
    if not os.path.exists(filepath):
        return False
    try:
        with h5py.File(filepath, 'r') as f:
            if 'meta' not in f or 'data' not in f:
                return False
            meta = f['meta']
            data = f['data']
            if 'columns' not in meta:
                return False
            cols = [c.decode() if isinstance(c, bytes) else c for c in meta['columns'][...]]
            for col in required_columns:
                if col not in cols or col not in data:
                    return False
            if 'z_phot_pdf' in data:
                if data['z_phot_pdf'].ndim != 2 or data['z_phot_pdf'].shape[1] != PDF_BINS:
                    return False
            n_rows = data[required_columns[0]].shape[0]
            return n_rows > 0
    except Exception:
        return False


def read_hdf5(filepath: str) -> pd.DataFrame:
    """读取 HDF5 文件为 DataFrame。"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        meta = f['meta']
        data_group = f['data']
        columns = [col.decode() if isinstance(col, bytes) else col for col in meta['columns'][...]]

        for col in columns:
            values = data_group[col][...]
            if col == 'z_phot_pdf':
                data[col] = [row for row in values]
            else:
                data[col] = values
    return pd.DataFrame(data)


def read_table(filepath: str) -> pd.DataFrame:
    """根据扩展名读取表。"""
    lower_path = filepath.lower()
    if lower_path.endswith(('.fits', '.fit', '.fz', '.fits.gz', '.fit.gz')):
        return read_fits(filepath)
    if lower_path.endswith(('.h5', '.hdf5')):
        return read_hdf5(filepath)
    raise ValueError(f"Unsupported input format: {filepath}")


def save_table(df: pd.DataFrame, filepath: str) -> None:
    """根据扩展名保存表。"""
    lower_path = filepath.lower()
    if lower_path.endswith(('.fits', '.fit', '.fz', '.fits.gz', '.fit.gz')):
        save_fits(df, filepath)
        return
    if lower_path.endswith(('.h5', '.hdf5')):
        save_hdf5(df, filepath)
        return
    raise ValueError(f"Unsupported output format: {filepath}")


# ========================== 统计量计算 ========================== #

def compute_photoz_stats_torch(probs: torch.Tensor, bin_centers: torch.Tensor,
                               to_cpu: bool = True) -> dict:
    """
    根据概率分布计算测光红移统计量（PyTorch/GPU 版本）。
    """
    z_mean = torch.sum(probs * bin_centers, dim=1)

    z_diff = bin_centers.unsqueeze(0) - z_mean.unsqueeze(1)
    z_std = torch.sqrt(torch.sum(probs * z_diff**2, dim=1))

    z_mode = bin_centers[torch.argmax(probs, dim=1)]

    cdf = torch.cumsum(probs, dim=1)
    rows = torch.arange(probs.shape[0], device=probs.device)

    quantiles_map = {
        'z_phot_l95': 0.025,
        'z_phot_l68': 0.16,
        'z_phot_median': 0.50,
        'z_phot_u68': 0.84,
        'z_phot_u95': 0.975
    }

    quantile_results = {}
    for col_name, q in quantiles_map.items():
        idx = torch.argmax((cdf >= q).to(torch.int64), dim=1)
        idx_prev = torch.clamp(idx - 1, min=0)

        z_high = bin_centers[idx]
        z_low = bin_centers[idx_prev]
        c_high = cdf[rows, idx]
        c_low = cdf[rows, idx_prev]

        denom = c_high - c_low
        valid = (idx > 0) & (denom > 0)

        result = z_high.clone()
        result[valid] = z_low[valid] + (q - c_low[valid]) / denom[valid] * (z_high[valid] - z_low[valid])
        quantile_results[col_name] = result

    stats = {
        'z_phot_mean': z_mean,
        'z_phot_std': z_std,
        'z_phot_mode': z_mode,
        **quantile_results
    }
    if to_cpu:
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in stats.items()}
    return stats


def resample_bins(prob: np.ndarray, bin_edges: np.ndarray, target_n: int = PDF_BINS):
    """
    将原始的高分辨率 bin 数据重采样为目标 bin 数量。
    保证概率质量守恒，输入 prob 视为概率质量（总和为 1）。
    """
    new_bin_edges = np.linspace(bin_edges[0], bin_edges[-1], target_n + 1)
    new_bin_centers = 0.5 * (new_bin_edges[:-1] + new_bin_edges[1:])

    original_cdf = np.concatenate(([0], np.cumsum(prob)))
    original_cdf /= original_cdf[-1]

    new_cdf_edges = np.interp(new_bin_edges, bin_edges, original_cdf)

    new_prob = np.diff(new_cdf_edges)
    new_prob = new_prob / np.sum(new_prob)

    return new_prob, new_bin_edges, new_bin_centers


def build_resample_matrix(bin_edges: np.ndarray, target_n: int = PDF_BINS) -> np.ndarray:
    """
    预计算从原始 bin 概率质量到目标 bin 概率质量的线性映射矩阵。
    返回形状为 (n_input_bins, target_n) 的矩阵，可直接做 probs @ matrix。
    """
    n_input_bins = len(bin_edges) - 1
    basis = np.eye(n_input_bins, dtype=np.float64)
    matrix = np.empty((n_input_bins, target_n), dtype=np.float32)

    for i in range(n_input_bins):
        new_prob, _, _ = resample_bins(basis[i], bin_edges, target_n=target_n)
        matrix[i] = new_prob.astype(np.float32)

    return matrix


# ========================== 预测器类 ========================== #

class PhotozPredictor:
    """
    测光红移预测器，支持模型预加载和大数据高效批量预测。
    可同时预测星系概率。
    """

    def __init__(self, model_base_dir: str = 'catalog/models', device: str = 'cuda:0',
                 galaxy_clf_dir: str = None):
        self.model_base_dir = model_base_dir
        self.device = device
        self._infers = {}
        self._bin_centers_torch_cache = {}  # 缓存 torch bin_centers
        self._resample_matrix_cache = {}  # 缓存 400->40 重采样矩阵

        # 星系分类器 (ANN)
        self.galaxy_clf_dir = galaxy_clf_dir
        self._galaxy_clf = None

    def _get_infer(self, model_name: str) -> PhotozBinInference:
        """获取或加载指定模型的推理器（懒加载）"""
        if model_name not in self._infers:
            model_dir = os.path.join(self.model_base_dir, model_name)
            calibrator_path = os.path.join(model_dir, 'TempScalingCalib', 'calibrator.pkl')

            infer = PhotozBinInference(model_dir=model_dir, device=self.device)
            infer.load_calibrator(calibrator_path)
            self._infers[model_name] = infer

            binning_config = infer.config.dataset_params['binning_config']
            z_min = binning_config.get('z_min', 0.0)
            z_max = binning_config.get('z_max', 2.0)
            num_bins = binning_config.get('num_bins', 400)
            bin_edges = np.linspace(z_min, z_max, num_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            self._bin_centers_torch_cache[model_name] = torch.from_numpy(
                bin_centers.astype(np.float32)
            ).to(self.device)
            self._resample_matrix_cache[model_name] = torch.from_numpy(
                build_resample_matrix(bin_edges, target_n=PDF_BINS)
            ).to(self.device)

        return self._infers[model_name]

    def preload_lsdr10_models(self):
        for name in ['p1', 'p2', 'p3']:
            self._get_infer(name)

    def preload_ps1dr2_models(self):
        for name in ['p4', 'p5']:
            self._get_infer(name)

    def _get_galaxy_clf(self) -> GalaxyClassificationInference:
        """获取或加载星系分类器（懒加载）"""
        if self._galaxy_clf is None:
            if self.galaxy_clf_dir is None:
                raise ValueError("galaxy_clf_dir not specified")
            self._galaxy_clf = GalaxyClassificationInference(
                model_dir=self.galaxy_clf_dir, device=self.device
            )
        return self._galaxy_clf

    def predict_galaxy_prob(self, df: pd.DataFrame, batch_size: int = 4096,
                            chunk_size: int = 1_000_000) -> np.ndarray:
        """
        预测星系概率

        Returns
        -------
        np.ndarray
            星系概率数组，值域 [0, 1]
        """
        clf = self._get_galaxy_clf()
        n_samples = len(df)
        probs = np.empty(n_samples, dtype=np.float32)
        feature_columns = clf.config.dataset_params.get('feature_columns')

        chunk_iterator = range(0, n_samples, chunk_size)

        with torch.no_grad():
            for chunk_start in chunk_iterator:
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_df = df.iloc[chunk_start:chunk_end]
                chunk_features = chunk_df[feature_columns].values.astype(np.float32)
                chunk_features = clf.scaler.transform(chunk_features)

                for local_start in range(0, len(chunk_features), batch_size):
                    local_end = min(local_start + batch_size, len(chunk_features))
                    start = chunk_start + local_start
                    end = chunk_start + local_end

                    inputs = torch.from_numpy(chunk_features[local_start:local_end]).to(clf.device)
                    outputs = torch.sigmoid(clf.model(inputs)).squeeze(1).cpu().numpy().astype(np.float32)
                    probs[start:end] = outputs

                del chunk_df, chunk_features

        return probs

    def _predict_subset_efficient(self, df: pd.DataFrame, indices: np.ndarray, model_name: str,
                                   batch_size: int, chunk_size: int) -> dict:
        """
        高效版本：一次性提取特征，直接使用模型推理。
        避免每个批次都创建 Dataset 的开销。
        """
        infer = self._get_infer(model_name)
        bin_centers_torch = self._bin_centers_torch_cache[model_name]
        resample_matrix = self._resample_matrix_cache[model_name]

        n_samples = len(indices)
        pred_cols = ['z_phot_mean', 'z_phot_std', 'z_phot_mode', 'z_phot_median',
                     'z_phot_l68', 'z_phot_u68', 'z_phot_l95', 'z_phot_u95']

        # 预分配结果数组
        results = {col: np.empty(n_samples, dtype=np.float32) for col in pred_cols}
        pdf_40 = np.empty((n_samples, PDF_BINS), dtype=np.float32)

        feature_columns = infer.config.dataset_params.get('feature_columns')

        temperature = infer.calibrator.temperature if infer.calibrator else 1.0

        chunk_iterator = range(0, n_samples, chunk_size)

        with torch.no_grad():
            for chunk_start in chunk_iterator:
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_indices = indices[chunk_start:chunk_end]
                chunk_df = df.iloc[chunk_indices]
                chunk_features = chunk_df[feature_columns].values.astype(np.float32)
                chunk_features = infer.scaler.transform(chunk_features)

                for local_start in range(0, len(chunk_features), batch_size):
                    local_end = min(local_start + batch_size, len(chunk_features))
                    start = chunk_start + local_start
                    end = chunk_start + local_end

                    batch_features = torch.from_numpy(
                        chunk_features[local_start:local_end]
                    ).to(infer.device)
                    if infer.device.type == 'cuda':
                        torch.cuda.synchronize(infer.device)

                    logits = infer.model(batch_features)

                    if temperature != 1.0:
                        logits = logits / temperature
                    probs_torch = torch.softmax(logits, dim=1)
                    stats_torch = compute_photoz_stats_torch(probs_torch, bin_centers_torch, to_cpu=False)
                    resampled_probs_torch = torch.matmul(probs_torch, resample_matrix)
                    if infer.device.type == 'cuda':
                        torch.cuda.synchronize(infer.device)

                    stats = {
                        k: v.detach().cpu().numpy().astype(np.float32)
                        for k, v in stats_torch.items()
                    }
                    resampled_probs = resampled_probs_torch.cpu().numpy()
                    if infer.device.type == 'cuda':
                        torch.cuda.synchronize(infer.device)

                    for col in pred_cols:
                        results[col][start:end] = stats[col]
                    pdf_40[start:end] = resampled_probs

                del chunk_df, chunk_features

        torch.cuda.empty_cache()
        z_phot_pdf_uint16 = encode_pdf_to_uint16(pdf_40)

        return {
            **results,
            'z_phot_pdf': z_phot_pdf_uint16,
        }

    def predict_lsdr10(
        self,
        df: pd.DataFrame,
        batch_size: int = 4096,
        chunk_size: int = 1_000_000,
    ) -> pd.DataFrame:
        """
        对 LSDR10 数据进行测光红移预测（高效版本）。

        优先级策略:
        - P1: 有原生 i 波段 (grizW1W2)
        - P2: 无原生 i 波段，但有 PS1 i 波段 (grzW1W2 + PS1 i)
        - P3: 无原生 i 波段，也无 PS1 i 波段 (grzW1W2)

        同时预测星系概率 galaxy_prob（若已配置星系分类器）。
        """
        ps1_i_col = 'iKronMag_dered'

        pred_cols = ['z_phot_mean', 'z_phot_std', 'z_phot_mode', 'z_phot_median',
                     'z_phot_l68', 'z_phot_u68', 'z_phot_l95', 'z_phot_u95']

        # 初始化结果数组
        n_samples = len(df)

        # 重置索引，确保索引是连续整数
        df = df.reset_index(drop=True)

        # 根据信噪比计算星等误差（星系分类器所需）
        snr_to_err = {'snr_g': 'mag_g_Err', 'snr_r': 'mag_r_Err', 'snr_i': 'mag_i_Err',
                       'snr_z': 'mag_z_Err', 'snr_w1': 'mag_w1_Err', 'snr_w2': 'mag_w2_Err'}
        for snr_col, err_col in snr_to_err.items():
            if snr_col in df.columns:
                df[err_col] = (1.0857 / df[snr_col]).astype(np.float32)

        result_arrays = {col: np.full(n_samples, np.nan, dtype=np.float32) for col in pred_cols}
        pdf_array = np.zeros((n_samples, PDF_BINS), dtype=np.uint16)
        priority_array = np.zeros(n_samples, dtype=np.int16)  # 使用 int16
        survey_array = np.zeros(n_samples, dtype=np.int16)    # LSDR10 = 0

        # 构建掩码（确保是 numpy boolean array）
        has_native_i = (df['dered_mag_i'].notna() & (df['dered_mag_i'] > 0)).values.astype(bool)
        has_ps1_i = ((df[ps1_i_col].notna() & (df[ps1_i_col] > 0)).values.astype(bool)
                     if ps1_i_col in df.columns else np.zeros(n_samples, dtype=bool))

        mask_p1 = has_native_i                      # 有原生 i
        mask_p2 = ~has_native_i & has_ps1_i         # 无原生 i，有 PS1 i
        mask_p3 = ~has_native_i & ~has_ps1_i        # 无原生 i，无 PS1 i

        model_configs = [
            ('p1', mask_p1, 1),
            ('p2', mask_p2, 2),
            ('p3', mask_p3, 3),
        ]

        t_photoz = time.time()
        for model_name, mask, priority in model_configs:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            stats = self._predict_subset_efficient(df, indices, model_name, batch_size, chunk_size)

            for col in pred_cols:
                result_arrays[col][indices] = stats[col]
            pdf_array[indices] = stats['z_phot_pdf']
            priority_array[indices] = priority
        photoz_time = time.time() - t_photoz

        galaxy_prob_array = None
        clf_time = 0.0
        if self.galaxy_clf_dir is not None:
            t_clf = time.time()
            galaxy_prob_array = self.predict_galaxy_prob(df, batch_size=batch_size, chunk_size=chunk_size)
            clf_time = time.time() - t_clf

        # 构建最终 DataFrame
        result = pd.DataFrame({
            'uid_ls': df['uid'].values,
            'ra': df['ra'].values,
            'dec': df['dec'].values,
            **result_arrays,
            'priority': priority_array,
            'survey': survey_array,
        })
        result['z_phot_pdf'] = list(pdf_array)

        if galaxy_prob_array is not None:
            result['galaxy_prob'] = galaxy_prob_array
        result.attrs['timing_photoz'] = photoz_time
        result.attrs['timing_clf'] = clf_time

        return result

    def predict_ps1dr2(
        self,
        df: pd.DataFrame,
        batch_size: int = 4096,
        chunk_size: int = 1_000_000,
    ) -> pd.DataFrame:
        """
        对 PS1DR2 数据进行测光红移预测（高效版本）。

        优先级策略:
        - P4: 有 unWISE 匹配 (grizy + W1W2)
        - P5: 无 unWISE 匹配 (grizy only)

        同时预测星系概率 galaxy_prob（若已配置星系分类器）。
        """
        w1_col = 'mag_w1'
        w2_col = 'mag_w2'

        pred_cols = ['z_phot_mean', 'z_phot_std', 'z_phot_mode', 'z_phot_median',
                     'z_phot_l68', 'z_phot_u68', 'z_phot_l95', 'z_phot_u95']

        # 初始化结果数组
        n_samples = len(df)

        # 重置索引，确保索引是连续整数
        df = df.reset_index(drop=True)

        result_arrays = {col: np.full(n_samples, np.nan, dtype=np.float32) for col in pred_cols}
        pdf_array = np.zeros((n_samples, PDF_BINS), dtype=np.uint16)
        priority_array = np.zeros(n_samples, dtype=np.int16)  # 使用 int16
        survey_array = np.ones(n_samples, dtype=np.int16)     # PS1DR2 = 1

        # 构建掩码（确保是 numpy boolean array）
        has_unwise = (
            (df[w1_col].notna() & (df[w1_col] > 0) &
             df[w2_col].notna() & (df[w2_col] > 0)).values.astype(bool)
        )

        mask_p4 = has_unwise
        mask_p5 = ~has_unwise

        model_configs = [
            ('p4', mask_p4, 4),
            ('p5', mask_p5, 5),
        ]

        t_photoz = time.time()
        for model_name, mask, priority in model_configs:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            stats = self._predict_subset_efficient(df, indices, model_name, batch_size, chunk_size)

            for col in pred_cols:
                result_arrays[col][indices] = stats[col]
            pdf_array[indices] = stats['z_phot_pdf']
            priority_array[indices] = priority
        photoz_time = time.time() - t_photoz

        galaxy_prob_array = None
        clf_time = 0.0
        if self.galaxy_clf_dir is not None:
            t_clf = time.time()
            galaxy_prob_array = self.predict_galaxy_prob(df, batch_size=batch_size, chunk_size=chunk_size)
            clf_time = time.time() - t_clf

        # 构建最终 DataFrame
        result = pd.DataFrame({
            'uid_ps': df['uid'].values,
            'ra': df['ra'].values,
            'dec': df['dec'].values,
            **result_arrays,
            'priority': priority_array,
            'survey': survey_array,
        })
        result['z_phot_pdf'] = list(pdf_array)

        if galaxy_prob_array is not None:
            result['galaxy_prob'] = galaxy_prob_array
        result.attrs['timing_photoz'] = photoz_time
        result.attrs['timing_clf'] = clf_time

        return result


# ========================== 批量预测函数 ========================== #

def _process_single_file_lsdr10(args, predictor: PhotozPredictor = None):
    """处理单个 LSDR10 文件的 worker 函数"""
    fid, filepath, output_path, model_base_dir, device, batch_size, chunk_size, galaxy_clf_dir = args

    try:
        t0 = time.time()
        required_columns = [
            'uid_ls', 'ra', 'dec',
            'z_phot_mean', 'z_phot_std', 'z_phot_mode', 'z_phot_median',
            'z_phot_l68', 'z_phot_u68', 'z_phot_l95', 'z_phot_u95',
            'priority', 'survey', 'z_phot_pdf'
        ]
        if galaxy_clf_dir is not None:
            required_columns.append('galaxy_prob')

        # 跳过已处理的有效文件；损坏文件自动删除重跑
        if _is_valid_prediction_hdf5(output_path, required_columns):
            return fid, 'skipped', None
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"  [{fid:02d}] Begin")

        # 每个进程独立加载模型；单进程模式下可复用外部 predictor
        if predictor is None:
            predictor = PhotozPredictor(model_base_dir=model_base_dir, device=device,
                                        galaxy_clf_dir=galaxy_clf_dir)
            predictor.preload_lsdr10_models()

        # 读取数据
        t_read = time.time()
        df = read_table(filepath)
        n_rows = len(df)
        t_read = time.time() - t_read

        # 预测
        result = predictor.predict_lsdr10(df, batch_size=batch_size, chunk_size=chunk_size)
        t_photoz = float(result.attrs.get('timing_photoz', 0.0))
        t_clf = float(result.attrs.get('timing_clf', 0.0))

        # 保存结果
        t_save = time.time()
        save_table(result, output_path)
        t_save = time.time() - t_save
        t_total = time.time() - t0
        print(
            f"  [{fid:02d}] Done, rows={n_rows:,} "
            f"read={t_read:.1f}s photoz={t_photoz:.1f}s clf={t_clf:.1f}s "
            f"write={t_save:.1f}s total={t_total:.1f}s"
        )

        # 清理内存
        del df, result
        gc.collect()
        torch.cuda.empty_cache()

        return fid, 'success', n_rows
    except Exception as e:
        return fid, 'error', str(e)


def _process_single_file_ps1dr2(args, predictor: PhotozPredictor = None):
    """处理单个 PS1DR2 文件的 worker 函数"""
    fid, filepath, output_path, model_base_dir, device, batch_size, chunk_size, galaxy_clf_dir = args

    try:
        t0 = time.time()
        required_columns = [
            'uid_ps', 'ra', 'dec',
            'z_phot_mean', 'z_phot_std', 'z_phot_mode', 'z_phot_median',
            'z_phot_l68', 'z_phot_u68', 'z_phot_l95', 'z_phot_u95',
            'priority', 'survey', 'z_phot_pdf'
        ]
        if galaxy_clf_dir is not None:
            required_columns.append('galaxy_prob')

        # 跳过已处理的有效文件；损坏文件自动删除重跑
        if _is_valid_prediction_hdf5(output_path, required_columns):
            return fid, 'skipped', None
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"  [{fid:02d}] Begin")

        # 每个进程独立加载模型；单进程模式下可复用外部 predictor
        if predictor is None:
            predictor = PhotozPredictor(model_base_dir=model_base_dir, device=device,
                                        galaxy_clf_dir=galaxy_clf_dir)
            predictor.preload_ps1dr2_models()

        # 读取数据
        t_read = time.time()
        df = read_table(filepath)
        n_rows = len(df)
        t_read = time.time() - t_read

        # 预测
        result = predictor.predict_ps1dr2(df, batch_size=batch_size, chunk_size=chunk_size)
        t_photoz = float(result.attrs.get('timing_photoz', 0.0))
        t_clf = float(result.attrs.get('timing_clf', 0.0))

        # 保存结果
        t_save = time.time()
        save_table(result, output_path)
        t_save = time.time() - t_save
        t_total = time.time() - t0
        print(
            f"  [{fid:02d}] Done, rows={n_rows:,} "
            f"read={t_read:.1f}s photoz={t_photoz:.1f}s clf={t_clf:.1f}s "
            f"write={t_save:.1f}s total={t_total:.1f}s"
        )

        # 清理内存
        del df, result
        gc.collect()
        torch.cuda.empty_cache()

        return fid, 'success', n_rows
    except Exception as e:
        return fid, 'error', str(e)


def predict_lsdr10_batch(
    data_dir: str,
    output_dir: str,
    model_base_dir: str,
    devices: list,
    batch_size: int = 4096,
    chunk_size: int = 1_000_000,
    num_workers: int = 1,
    galaxy_clf_dir: str = None,
    isolate_per_file: bool = False,
):
    """
    批量预测 LSDR10 文件（支持多进程并行）

    Parameters
    ----------
    data_dir : str
        输入数据目录
    output_dir : str
        输出目录
    model_base_dir : str
        模型目录
    devices : list
        GPU 设备列表，如 ['cuda:0', 'cuda:1']
    batch_size : int
        批处理大小
    num_workers : int
        并行进程数
    galaxy_clf_dir : str
        星系分类模型目录（可选）
    """
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(data_dir, '*.fits')))
    print(f"Found {len(file_list)} files in {data_dir}")

    if len(file_list) == 0:
        print("No files found. Exiting.")
        return

    # 准备任务列表
    tasks = []
    for fid, filepath in enumerate(file_list):
        output_path = os.path.join(output_dir, LSDR10_OUTPUT_TEMPLATE.format(fid=fid))
        device = devices[fid % len(devices)]  # 轮询分配 GPU
        tasks.append((fid, filepath, output_path, model_base_dir, device, batch_size, chunk_size, galaxy_clf_dir))

    if isolate_per_file:
        print("Using one fresh subprocess per LSDR10 file")
        for task in tasks:
            fid = task[0]
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_process_single_file_lsdr10, task)
                try:
                    fid, status, info = future.result()
                    if status == 'skipped':
                        print(f"  [{fid:02d}] Skipped (exists)")
                    elif status == 'error':
                        print(f"  [{fid:02d}] Error: {info}")
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")
    elif num_workers == 1:
        # 单进程模式
        predictor = PhotozPredictor(
            model_base_dir=model_base_dir,
            device=devices[0],
            galaxy_clf_dir=galaxy_clf_dir
        )
        predictor.preload_lsdr10_models()
        for task in tasks:
            fid, status, info = _process_single_file_lsdr10(task, predictor=predictor)
            if status == 'skipped':
                print(f"  [{fid:02d}] Skipped (exists)")
            elif status == 'error':
                print(f"  [{fid:02d}] Error: {info}")
    else:
        # 多进程模式
        print(f"Using {num_workers} workers with devices: {devices}")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_file_lsdr10, task): task[0] for task in tasks}

            for future in as_completed(futures):
                fid = futures[future]
                try:
                    fid, status, info = future.result()
                    if status == 'skipped':
                        print(f"  [{fid:02d}] Skipped (exists)")
                    elif status == 'error':
                        print(f"  [{fid:02d}] Error: {info}")
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")


def predict_ps1dr2_batch(
    data_dir: str,
    output_dir: str,
    model_base_dir: str,
    devices: list,
    batch_size: int = 4096,
    chunk_size: int = 1_000_000,
    num_workers: int = 1,
    galaxy_clf_dir: str = None
):
    """
    批量预测 PS1DR2 文件（支持多进程并行）

    Parameters
    ----------
    data_dir : str
        输入数据目录
    output_dir : str
        输出目录
    model_base_dir : str
        模型目录
    devices : list
        GPU 设备列表，如 ['cuda:0', 'cuda:1']
    batch_size : int
        批处理大小
    num_workers : int
        并行进程数
    galaxy_clf_dir : str
        星系分类模型目录（可选）
    """
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(data_dir, '*.fits')))
    print(f"Found {len(file_list)} files in {data_dir}")

    if len(file_list) == 0:
        print("No files found. Exiting.")
        return

    # 准备任务列表
    tasks = []
    for fid, filepath in enumerate(file_list):
        output_path = os.path.join(output_dir, PS1DR2_OUTPUT_TEMPLATE.format(fid=fid))
        device = devices[fid % len(devices)]  # 轮询分配 GPU
        tasks.append((fid, filepath, output_path, model_base_dir, device, batch_size, chunk_size, galaxy_clf_dir))

    if num_workers == 1:
        # 单进程模式
        predictor = PhotozPredictor(
            model_base_dir=model_base_dir,
            device=devices[0],
            galaxy_clf_dir=galaxy_clf_dir
        )
        predictor.preload_ps1dr2_models()
        for task in tasks:
            fid, status, info = _process_single_file_ps1dr2(task, predictor=predictor)
            if status == 'skipped':
                print(f"  [{fid:02d}] Skipped (exists)")
            elif status == 'error':
                print(f"  [{fid:02d}] Error: {info}")
    else:
        # 多进程模式
        print(f"Using {num_workers} workers with devices: {devices}")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_file_ps1dr2, task): task[0] for task in tasks}

            for future in as_completed(futures):
                fid = futures[future]
                try:
                    fid, status, info = future.result()
                    if status == 'skipped':
                        print(f"  [{fid:02d}] Skipped (exists)")
                    elif status == 'error':
                        print(f"  [{fid:02d}] Error: {info}")
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")


# ========================== 合并函数 ========================== #

def _catalog_output_dirs(base_dir: str):
    merge_main_dir = os.path.join(base_dir, os.path.relpath(MERGE_MAIN_DIR, CATALOG_BASE_DIR))
    merge_pdf_dir = os.path.join(base_dir, os.path.relpath(MERGE_PDF_DIR, CATALOG_BASE_DIR))
    publish_dir = os.path.join(base_dir, os.path.relpath(PUBLISH_DIR, CATALOG_BASE_DIR))
    os.makedirs(merge_main_dir, exist_ok=True)
    os.makedirs(merge_pdf_dir, exist_ok=True)
    os.makedirs(publish_dir, exist_ok=True)
    return merge_main_dir, merge_pdf_dir, publish_dir


def _intermediate_paths(merge_main_dir: str, merge_pdf_dir: str, ra_start: int, ra_end: int):
    return (
        os.path.join(merge_main_dir, MERGE_MAIN_TEMPLATE.format(ra_start=ra_start, ra_end=ra_end)),
        os.path.join(merge_pdf_dir, MERGE_PDF_TEMPLATE.format(ra_start=ra_start, ra_end=ra_end)),
    )


def _publish_paths(publish_dir: str, ra_start: int, ra_end: int):
    return (
        os.path.join(publish_dir, PUBLISH_MAIN_TEMPLATE.format(ra_start=ra_start, ra_end=ra_end)),
        os.path.join(publish_dir, PUBLISH_PDF_TEMPLATE.format(ra_start=ra_start, ra_end=ra_end)),
    )


def _remove_files_if_exist(*filepaths: str) -> None:
    """删除一组文件，忽略不存在的情况。"""
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


def _count_non_increasing_steps(values: np.ndarray) -> int:
    """统计数组中非严格递增的位置数。"""
    if len(values) <= 1:
        return 0
    return int(np.sum(values[1:] <= values[:-1]))


def _fits_uid_info(filepath: str, return_uid: bool = False) -> dict:
    """读取 FITS 主表的行数与 uid 边界，用于轻量校验。"""
    with fits.open(filepath, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"Missing binary table extension in {filepath}")
        hdu = hdul[1]
        columns = list(hdu.columns.names)
        if columns != CATALOG_MAIN_COLUMNS:
            raise ValueError(f"Unexpected main columns in {filepath}: {columns}")

        n_rows = int(hdu.header.get('NAXIS2', 0))
        if n_rows == 0:
            info = {'rows': 0, 'first_uid': None, 'last_uid': None}
            if return_uid:
                info['uid'] = np.array([], dtype=np.int64)
            return info

        uid = np.asarray(hdu.data['uid'])
        non_increasing = _count_non_increasing_steps(uid)
        if non_increasing > 0:
            raise ValueError(f"Main uid not strictly increasing in {filepath}: {non_increasing}")

        info = {
            'rows': n_rows,
            'first_uid': int(uid[0]),
            'last_uid': int(uid[-1]),
        }
        if return_uid:
            info['uid'] = uid
        return info


def _pdf_uid_info(filepath: str, return_uid: bool = False) -> dict:
    """读取 PDF HDF5 的行数与 uid 边界，并检查基本结构。"""
    with h5py.File(filepath, 'r') as f:
        if 'meta' not in f or 'data' not in f:
            raise ValueError(f"Invalid HDF5 structure: {filepath}")

        meta = f['meta']
        data = f['data']
        if 'columns' not in meta:
            raise ValueError(f"Missing HDF5 columns metadata: {filepath}")
        columns = [col.decode() if isinstance(col, bytes) else col for col in meta['columns'][...]]
        if columns != CATALOG_PDF_COLUMNS:
            raise ValueError(f"Unexpected PDF columns in {filepath}: {columns}")
        if 'uid' not in data or 'z_phot_pdf' not in data:
            raise ValueError(f"Missing uid/z_phot_pdf dataset: {filepath}")

        uid = data['uid']
        pdf = data['z_phot_pdf']
        n_rows = int(uid.shape[0])

        if pdf.shape != (n_rows, PDF_BINS):
            raise ValueError(f"Unexpected PDF shape in {filepath}: {pdf.shape}")
        if pdf.dtype != np.uint16:
            raise ValueError(f"Unexpected PDF dtype in {filepath}: {pdf.dtype}")
        if meta.attrs.get('format') != 'photoz_table_v2':
            raise ValueError(f"Unexpected HDF5 format in {filepath}: {meta.attrs.get('format')}")

        if n_rows == 0:
            info = {'rows': 0, 'first_uid': None, 'last_uid': None}
            if return_uid:
                info['uid'] = np.array([], dtype=np.int64)
            return info

        uid_values = uid[...]
        non_increasing = _count_non_increasing_steps(uid_values)
        if non_increasing > 0:
            raise ValueError(f"PDF uid not strictly increasing in {filepath}: {non_increasing}")

        info = {
            'rows': n_rows,
            'first_uid': int(uid_values[0]),
            'last_uid': int(uid_values[-1]),
        }
        if return_uid:
            info['uid'] = uid_values
        return info


def _validate_intermediate_shard(main_file: str, pdf_file: str) -> dict:
    """校验单个 merge 中间分片的主表/PDF 是否一致。"""
    if not os.path.exists(main_file) or not os.path.exists(pdf_file):
        raise FileNotFoundError(
            f"Missing intermediate shard files: {os.path.basename(main_file)}, {os.path.basename(pdf_file)}"
        )

    main_info = _fits_uid_info(main_file, return_uid=True)
    pdf_info = _pdf_uid_info(pdf_file, return_uid=True)

    if main_info['rows'] != pdf_info['rows']:
        raise ValueError(
            f"Row count mismatch in {os.path.basename(main_file)}: "
            f"main={main_info['rows']}, pdf={pdf_info['rows']}"
        )

    if not np.array_equal(main_info['uid'], pdf_info['uid']):
        raise ValueError(f"UID array mismatch in {os.path.basename(main_file)}")

    return {
        'rows': main_info['rows'],
        'first_uid': main_info['first_uid'],
        'last_uid': main_info['last_uid'],
    }


def _validate_intermediate_shards(
    merge_main_dir: str,
    merge_pdf_dir: str,
    require_global_uid: bool = False
) -> int:
    """
    校验 72 个 merge 中间分片是否完整。
    当 require_global_uid=True 时，还要求 uid 在全局上连续一致。
    """
    expected_next_uid = None
    total_rows = 0

    for fid in range(N_RA_BINS):
        ra_start = fid * MERGE_RA_STEP
        ra_end = (fid + 1) * MERGE_RA_STEP
        main_file, pdf_file = _intermediate_paths(merge_main_dir, merge_pdf_dir, ra_start, ra_end)
        main_info = _validate_intermediate_shard(main_file, pdf_file)
        total_rows += main_info['rows']

        if main_info['rows'] == 0:
            continue

        if require_global_uid and expected_next_uid is not None and main_info['first_uid'] != expected_next_uid:
            raise ValueError(
                f"Non-contiguous intermediate uid range before RA{ra_start:03d}_{ra_end:03d}: "
                f"expected {expected_next_uid}, got {main_info['first_uid']}"
            )

        expected_next_uid = main_info['last_uid'] + 1

    return total_rows


def _finalize_intermediate_uids(merge_main_dir: str, merge_pdf_dir: str) -> int:
    """
    根据 72 个分片的实际行数，统一回填全局连续 uid。
    """
    shard_infos = []
    uid_offset = 0

    for fid in range(N_RA_BINS):
        ra_start = fid * MERGE_RA_STEP
        ra_end = (fid + 1) * MERGE_RA_STEP
        main_file, pdf_file = _intermediate_paths(merge_main_dir, merge_pdf_dir, ra_start, ra_end)
        shard_info = _validate_intermediate_shard(main_file, pdf_file)
        shard_info['main_file'] = main_file
        shard_info['pdf_file'] = pdf_file
        shard_info['uid_start'] = uid_offset
        shard_infos.append(shard_info)
        uid_offset += shard_info['rows']

    for shard_info in shard_infos:
        rows = shard_info['rows']
        if rows == 0:
            continue

        uid_values = np.arange(
            shard_info['uid_start'],
            shard_info['uid_start'] + rows,
            dtype=np.int64
        )

        with fits.open(shard_info['main_file'], mode='update', memmap=True) as hdul:
            hdul[1].data['uid'][:] = uid_values
            hdul.flush()

        with h5py.File(shard_info['pdf_file'], 'r+') as f:
            f['data']['uid'][...] = uid_values

    _validate_intermediate_shards(merge_main_dir, merge_pdf_dir, require_global_uid=True)
    return uid_offset


def _combine_preferred_columns(df_matched: pd.DataFrame, prefer_1: pd.Series, columns: list) -> pd.DataFrame:
    result = pd.DataFrame(index=df_matched.index)
    for col in columns:
        col_1 = f'{col}_1'
        col_2 = f'{col}_2'
        if col_1 in df_matched.columns and col_2 in df_matched.columns:
            result[col] = np.where(prefer_1, df_matched[col_1], df_matched[col_2])
        elif col_1 in df_matched.columns:
            result[col] = df_matched[col_1]
        elif col_2 in df_matched.columns:
            result[col] = df_matched[col_2]
        elif col in df_matched.columns:
            result[col] = df_matched[col]
        else:
            raise KeyError(f"Column {col} not found in cross-match output")
    return result


def _select_preferred_pdf(df_matched: pd.DataFrame, prefer_1: pd.Series,
                          df_ls: pd.DataFrame, df_ps: pd.DataFrame) -> np.ndarray:
    n_rows = len(df_matched)
    pdf_array = np.zeros((n_rows, PDF_BINS), dtype=np.uint16)

    use_ls = prefer_1.to_numpy(dtype=bool)
    use_ps = ~use_ls

    if len(df_ls) > 0:
        ls_lookup = pd.Series(np.arange(len(df_ls), dtype=np.int64), index=df_ls['uid_ls'])
        ls_pdf = np.stack(df_ls['z_phot_pdf'].map(lambda v: np.asarray(v, dtype=np.uint16)), axis=0)
        if 'uid_ls' in df_matched.columns:
            uid_ls = df_matched['uid_ls']
        else:
            uid_ls = df_matched.get('uid_ls_1', pd.Series(index=df_matched.index, dtype=df_ls['uid_ls'].dtype))
        ls_rows = uid_ls.map(ls_lookup)
        if use_ls.any():
            if ls_rows[use_ls].isna().any():
                raise ValueError("Missing LSDR10 PDF rows after cross-match")
            pdf_array[use_ls] = ls_pdf[ls_rows[use_ls].to_numpy(dtype=np.int64)]
    elif use_ls.any():
        raise ValueError("Cross-match selected LSDR10 rows but LSDR10 input is empty")

    if len(df_ps) > 0:
        ps_lookup = pd.Series(np.arange(len(df_ps), dtype=np.int64), index=df_ps['uid_ps'])
        ps_pdf = np.stack(df_ps['z_phot_pdf'].map(lambda v: np.asarray(v, dtype=np.uint16)), axis=0)
        if 'uid_ps' in df_matched.columns:
            uid_ps = df_matched['uid_ps']
        else:
            uid_ps = df_matched.get('uid_ps_2', pd.Series(index=df_matched.index, dtype=df_ps['uid_ps'].dtype))
        ps_rows = uid_ps.map(ps_lookup)
        if use_ps.any():
            if ps_rows[use_ps].isna().any():
                raise ValueError("Missing PS1DR2 PDF rows after cross-match")
            pdf_array[use_ps] = ps_pdf[ps_rows[use_ps].to_numpy(dtype=np.int64)]
    elif use_ps.any():
        raise ValueError("Cross-match selected PS1DR2 rows but PS1DR2 input is empty")

    return pdf_array


def _concat_tables(filepaths: list) -> pd.DataFrame:
    frames = [read_table(filepath) for filepath in filepaths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def check_published_catalog() -> None:
    """
    检查 publish 目录下 12 对发布文件是否结构完整且数据一致。
    """
    print("=" * 60)
    print("Checking published catalog volumes")
    print("=" * 60)

    n_ok = 0
    n_bad = 0
    total_rows = 0
    expected_next_uid = None

    for ra_start in range(0, 360, PUBLISH_RA_STEP):
        ra_end = ra_start + PUBLISH_RA_STEP
        main_file, pdf_file = _publish_paths(PUBLISH_DIR, ra_start, ra_end)
        issues = []
        rows = 0
        next_expected_uid = None
        main_uid = None
        pdf_uid = None
        pdf_rows = 0

        if not os.path.exists(main_file):
            issues.append(f"missing main file: {os.path.basename(main_file)}")
        else:
            try:
                with fits.open(main_file, memmap=True) as hdul:
                    if len(hdul) < 2:
                        raise ValueError("missing binary table extension")

                    hdu = hdul[1]
                    columns = list(hdu.columns.names)
                    if columns != CATALOG_MAIN_COLUMNS:
                        issues.append(f"unexpected main columns: {columns}")

                    data = hdu.data
                    rows = int(hdu.header.get('NAXIS2', 0))
                    total_rows += rows

                    if rows > 0:
                        main_uid = np.asarray(data['uid'])
                        ra = np.asarray(data['ra'])
                        dec = np.asarray(data['dec'])
                        z_std = np.asarray(data['z_phot_std'])
                        z_l95 = np.asarray(data['z_phot_l95'])
                        z_l68 = np.asarray(data['z_phot_l68'])
                        z_med = np.asarray(data['z_phot_median'])
                        z_u68 = np.asarray(data['z_phot_u68'])
                        z_u95 = np.asarray(data['z_phot_u95'])
                        priority = np.asarray(data['priority'])
                        survey = np.asarray(data['survey'])
                        galaxy_prob = np.asarray(data['galaxy_prob'])

                        non_increasing = _count_non_increasing_steps(main_uid)
                        if non_increasing > 0:
                            issues.append(f"main uid not strictly increasing: {non_increasing}")

                        first_uid = int(main_uid[0])
                        last_uid = int(main_uid[-1])
                        if expected_next_uid is not None and first_uid != expected_next_uid:
                            issues.append(
                                f"main uid range break: expected {expected_next_uid}, got {first_uid}"
                            )
                        next_expected_uid = last_uid + 1

                        bad_ra = (ra < ra_start) | (ra >= ra_end)
                        if ra_end == 360:
                            bad_ra = (ra < ra_start) | (ra > ra_end)
                        if np.any(bad_ra):
                            issues.append(f"ra out of range rows: {int(np.sum(bad_ra))}")

                        bad_dec = (dec < -90.0) | (dec > 90.0)
                        if np.any(bad_dec):
                            issues.append(f"dec out of range rows: {int(np.sum(bad_dec))}")

                        if np.any(np.isnan(ra)):
                            issues.append(f"ra NaN rows: {int(np.isnan(ra).sum())}")
                        if np.any(np.isnan(dec)):
                            issues.append(f"dec NaN rows: {int(np.isnan(dec).sum())}")
                        if np.any(np.isnan(z_std)):
                            issues.append(f"z_phot_std NaN rows: {int(np.isnan(z_std).sum())}")
                        if np.any(np.isnan(galaxy_prob)):
                            issues.append(f"galaxy_prob NaN rows: {int(np.isnan(galaxy_prob).sum())}")
                        if np.any(z_std < 0):
                            issues.append(f"negative z_phot_std rows: {int(np.sum(z_std < 0))}")

                        quantile_bad = ~(
                            (z_l95 <= z_l68)
                            & (z_l68 <= z_med)
                            & (z_med <= z_u68)
                            & (z_u68 <= z_u95)
                        )
                        if np.any(quantile_bad):
                            issues.append(f"invalid quantile ordering rows: {int(np.sum(quantile_bad))}")

                        survey_bad = ~np.isin(survey, [0, 1])
                        if np.any(survey_bad):
                            issues.append(f"invalid survey rows: {int(np.sum(survey_bad))}")

                        ls_priority_bad = (survey == 0) & (~np.isin(priority, [1, 2, 3]))
                        if np.any(ls_priority_bad):
                            issues.append(
                                f"invalid LSDR10 priority rows: {int(np.sum(ls_priority_bad))}"
                            )

                        ps_priority_bad = (survey == 1) & (~np.isin(priority, [4, 5]))
                        if np.any(ps_priority_bad):
                            issues.append(
                                f"invalid PS1DR2 priority rows: {int(np.sum(ps_priority_bad))}"
                            )

                        galaxy_bad = galaxy_prob <= 0.5
                        if np.any(galaxy_bad):
                            issues.append(f"galaxy_prob <= 0.5 rows: {int(np.sum(galaxy_bad))}")
            except Exception as e:
                issues.append(f"main file read/check failed: {e}")

        if not os.path.exists(pdf_file):
            issues.append(f"missing pdf file: {os.path.basename(pdf_file)}")
        else:
            try:
                with h5py.File(pdf_file, 'r') as f:
                    if 'meta' not in f or 'data' not in f:
                        raise ValueError("missing /meta or /data group")

                    meta = f['meta']
                    data_group = f['data']

                    if 'columns' not in meta:
                        issues.append("pdf meta.columns missing")
                        columns = []
                    else:
                        columns = [
                            col.decode() if isinstance(col, bytes) else col
                            for col in meta['columns'][...]
                        ]
                        if columns != CATALOG_PDF_COLUMNS:
                            issues.append(f"unexpected pdf columns: {columns}")

                    if meta.attrs.get('format') != 'photoz_table_v2':
                        issues.append(f"unexpected pdf format: {meta.attrs.get('format')}")

                    if 'uid' not in data_group or 'z_phot_pdf' not in data_group:
                        raise ValueError("missing uid or z_phot_pdf dataset")

                    uid_ds = data_group['uid']
                    pdf_ds = data_group['z_phot_pdf']
                    pdf_rows = int(uid_ds.shape[0])

                    if uid_ds.ndim != 1:
                        issues.append(f"uid dataset dimension invalid: {uid_ds.shape}")
                    if pdf_ds.shape != (pdf_rows, PDF_BINS):
                        issues.append(f"pdf dataset shape invalid: {pdf_ds.shape}")
                    if pdf_ds.dtype != np.uint16:
                        issues.append(f"pdf dataset dtype invalid: {pdf_ds.dtype}")

                    if pdf_rows > 0:
                        pdf_uid = uid_ds[...]
                        non_increasing = _count_non_increasing_steps(pdf_uid)
                        if non_increasing > 0:
                            issues.append(f"pdf uid not strictly increasing: {non_increasing}")

                        bad_pdf_rows = 0
                        for start in range(0, pdf_rows, 200_000):
                            end = min(start + 200_000, pdf_rows)
                            row_sums = np.sum(pdf_ds[start:end], axis=1, dtype=np.uint32)
                            bad_pdf_rows += int(np.sum(row_sums != PDF_UINT16_SCALE))
                        if bad_pdf_rows > 0:
                            issues.append(f"pdf row sum != {int(PDF_UINT16_SCALE)} rows: {bad_pdf_rows}")
            except Exception as e:
                issues.append(f"pdf file read/check failed: {e}")

        if rows != pdf_rows:
            issues.append(f"main/pdf row mismatch: main={rows}, pdf={pdf_rows}")

        if main_uid is not None and pdf_uid is not None and not np.array_equal(main_uid, pdf_uid):
            issues.append("main/pdf uid arrays differ")

        label = "OK" if not issues else "BAD"
        print(f"  [{label}] RA{ra_start:03d}_{ra_end:03d} main+pdf rows={rows:,}")
        for issue in issues:
            print(f"    - {issue}")

        if issues:
            n_bad += 1
        else:
            n_ok += 1
            expected_next_uid = next_expected_uid

        del main_uid, pdf_uid
        gc.collect()

    print(
        f"Summary publish main+pdf: ok={n_ok}, bad={n_bad}, rows={total_rows:,}, "
        f"volumes={n_ok + n_bad}"
    )
    if n_bad > 0:
        raise SystemExit(1)


def merge_surveys():
    """
    生成 72 个中间 merge 分片：
    - 主表分片：FITS，仅包含标量列
    - PDF 分片：HDF5，仅包含 uid 和 z_phot_pdf
    - z_phot_pdf 不进入 pandas/stilts 主表匹配路径
    - 全局 uid 在全部分片齐全后统一回填
    """
    from cosmic.utils import stilts_crossmatch

    ls_dir = LSDR10_OUTPUT_DIR
    ps_dir = PS1DR2_OUTPUT_DIR
    merge_main_dir, merge_pdf_dir, _ = _catalog_output_dirs(CATALOG_BASE_DIR)

    n_bins = N_RA_BINS
    ra_step = MERGE_RA_STEP

    print("=" * 60)
    print("Merging LSDR10 + PS1DR2 into intermediate main/pdf shards")
    print("=" * 60)

    failed_shards = []

    for fid in range(n_bins):
        ra_start = fid * ra_step
        ra_end = (fid + 1) * ra_step
        main_output, pdf_output = _intermediate_paths(merge_main_dir, merge_pdf_dir, ra_start, ra_end)

        if os.path.exists(main_output) and os.path.exists(pdf_output):
            try:
                shard_info = _validate_intermediate_shard(main_output, pdf_output)
                print(
                    f"  [{fid:02d}] Skipped (exists) -> "
                    f"{shard_info['rows']:,} sources"
                )
                continue
            except Exception as e:
                print(f"  [{fid:02d}] Removing invalid existing shard -> {e}")
                _remove_files_if_exist(main_output, pdf_output)

        try:
            ls_file = os.path.join(ls_dir, LSDR10_OUTPUT_TEMPLATE.format(fid=fid))
            ps_file = os.path.join(ps_dir, PS1DR2_OUTPUT_TEMPLATE.format(fid=fid))

            # 读取并筛选 galaxy_prob > 0.5
            df_ls = read_table(ls_file)
            df_ls = df_ls[df_ls['galaxy_prob'] > 0.5].reset_index(drop=True)

            df_ps = read_table(ps_file)
            df_ps = df_ps[df_ps['galaxy_prob'] > 0.5].reset_index(drop=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_ls = os.path.join(tmpdir, 'ls.fits')
                tmp_ps = os.path.join(tmpdir, 'ps.fits')
                tmp_out = os.path.join(tmpdir, 'matched.fits')

                save_fits(df_ls[['uid_ls', *CATALOG_SCALAR_COLUMNS]], tmp_ls)
                save_fits(df_ps[['uid_ps', *CATALOG_SCALAR_COLUMNS]], tmp_ps)

                # stilts 交叉匹配
                success = stilts_crossmatch(
                    input1=tmp_ls,
                    input2=tmp_ps,
                    output_file=tmp_out,
                    join="1or2",
                    find="best",
                    match_radius=1.0,
                    verbose=False,
                )

                if not success:
                    raise RuntimeError(f"Cross-match failed for RA{ra_start:03d}_{ra_end:03d}")

                # 读取匹配结果并合并列（优先 table1=LSDR10）
                df_matched = read_fits(tmp_out)

            # 根据 priority 决定每行取哪个表的值（数值越小优先级越高）
            p1 = df_matched.get('priority_1', pd.Series(dtype='float64'))
            p2 = df_matched.get('priority_2', pd.Series(dtype='float64'))
            prefer_1 = p2.isna() | (p1.notna() & (p1 <= p2))

            main_result = _combine_preferred_columns(df_matched, prefer_1, CATALOG_SCALAR_COLUMNS)
            pdf_array = _select_preferred_pdf(df_matched, prefer_1, df_ls, df_ps)

            # 整数列恢复类型
            for col in ['priority', 'survey']:
                if col in main_result.columns:
                    main_result[col] = main_result[col].astype(np.int16)

            # 先写入分片内局部 uid，全部分片完成后再统一回填全局 uid
            n = len(main_result)
            uid = np.arange(n, dtype=np.int64)
            main_result.insert(0, 'uid', uid)
            pdf_result = pd.DataFrame({
                'uid': uid,
                'z_phot_pdf': list(pdf_array),
            })

            save_table(main_result[CATALOG_MAIN_COLUMNS], main_output)
            save_table(pdf_result[CATALOG_PDF_COLUMNS], pdf_output)
            print(
                f"  [{fid:02d}] {n:,} sources -> "
                f"{os.path.basename(main_output)}, {os.path.basename(pdf_output)}"
            )

            del df_ls, df_ps, df_matched, main_result, pdf_result, pdf_array
            gc.collect()
        except Exception as e:
            _remove_files_if_exist(main_output, pdf_output)
            failed_shards.append((ra_start, ra_end, str(e)))
            print(f"  [{fid:02d}] Failed: RA{ra_start:03d}_{ra_end:03d} -> {e}")
            gc.collect()

    if failed_shards:
        failed_text = ', '.join(
            f"RA{ra_start:03d}_{ra_end:03d}"
            for ra_start, ra_end, _ in failed_shards
        )
        raise RuntimeError(
            "Merge completed with failed shards; successful shards were kept. "
            f"Rerun merge after fixing inputs. Failed shards: {failed_text}"
        )

    total_rows = _finalize_intermediate_uids(merge_main_dir, merge_pdf_dir)
    print(f"\nIntermediate merge completed! Total UIDs assigned: {total_rows:,}")


def publish_catalog():
    """
    将 72 个 5-degree 中间分片 regroup 为 12 个 30-degree 发布分卷。
    """
    merge_main_dir = MERGE_MAIN_DIR
    merge_pdf_dir = MERGE_PDF_DIR
    _finalize_intermediate_uids(merge_main_dir, merge_pdf_dir)
    _, _, publish_dir = _catalog_output_dirs(CATALOG_BASE_DIR)

    ra_group_step = PUBLISH_RA_STEP
    shard_step = MERGE_RA_STEP

    print("=" * 60)
    print("Publishing merged catalog into 12 main FITS + 12 PDF HDF5 volumes")
    print("=" * 60)

    failed_groups = []

    for ra_start in range(0, 360, ra_group_step):
        ra_end = ra_start + ra_group_step
        main_output, pdf_output = _publish_paths(publish_dir, ra_start, ra_end)
        try:
            main_inputs = []
            pdf_inputs = []
            for shard_start in range(ra_start, ra_end, shard_step):
                shard_end = shard_start + shard_step
                main_file, pdf_file = _intermediate_paths(merge_main_dir, merge_pdf_dir, shard_start, shard_end)
                if not os.path.exists(main_file) or not os.path.exists(pdf_file):
                    raise FileNotFoundError(
                        f"Missing intermediate shard for RA{shard_start:03d}_{shard_end:03d}"
                    )
                main_inputs.append(main_file)
                pdf_inputs.append(pdf_file)

            main_result = _concat_tables(main_inputs).sort_values('uid').reset_index(drop=True)
            pdf_result = _concat_tables(pdf_inputs).sort_values('uid').reset_index(drop=True)

            if len(main_result) != len(pdf_result):
                raise ValueError(f"Row count mismatch in RA{ra_start:03d}_{ra_end:03d} publish group")
            if not np.array_equal(main_result['uid'].to_numpy(), pdf_result['uid'].to_numpy()):
                raise ValueError(f"UID mismatch in RA{ra_start:03d}_{ra_end:03d} publish group")

            if len(main_result) > 0:
                first_info = _fits_uid_info(main_inputs[0])
                last_info = _fits_uid_info(main_inputs[-1])
                first_uid = int(main_result['uid'].iloc[0])
                last_uid = int(main_result['uid'].iloc[-1])
                if first_uid != first_info['first_uid'] or last_uid != last_info['last_uid']:
                    raise ValueError(
                        f"Unexpected publish uid range in RA{ra_start:03d}_{ra_end:03d}: "
                        f"expected ({first_info['first_uid']}, {last_info['last_uid']}), "
                        f"got ({first_uid}, {last_uid})"
                    )

            save_table(main_result[CATALOG_MAIN_COLUMNS], main_output)
            save_table(pdf_result[CATALOG_PDF_COLUMNS], pdf_output)
            print(
                f"  RA{ra_start:03d}_{ra_end:03d}: "
                f"{len(main_result):,} sources -> "
                f"{os.path.basename(main_output)}, {os.path.basename(pdf_output)}"
            )

            del main_result, pdf_result
            gc.collect()
        except Exception as e:
            _remove_files_if_exist(main_output, pdf_output)
            failed_groups.append((ra_start, ra_end, str(e)))
            print(f"  Publish failed: RA{ra_start:03d}_{ra_end:03d} -> {e}")
            gc.collect()

    if failed_groups:
        failed_text = ', '.join(
            f"RA{ra_start:03d}_{ra_end:03d}"
            for ra_start, ra_end, _ in failed_groups
        )
        raise RuntimeError(
            "Publish completed with failed groups; successful volumes were kept. "
            f"Rerun publish after fixing the failed groups. Failed groups: {failed_text}"
        )


# ========================== 主函数 ========================== #

def run_lsdr10():
    """运行 LSDR10 预测"""
    # 配置路径
    model_base_dir = MODEL_BASE_DIR
    data_dir = LSDR10_INPUT_DIR
    output_dir = LSDR10_OUTPUT_DIR
    # galaxy_clf_dir 原目录： '/home/tiandc/photoz/galaxyClf/galaxyClf_ann/LSDR10/grzW1W2/0NoSpec'
    galaxy_clf_dir = GALAXY_CLF_LS_DIR
    devices = ['cuda:0']
    batch_size = 32768
    chunk_size = 200_000
    num_workers = 1

    print("=" * 60)
    print("LSDR10 Photo-z Prediction")
    if galaxy_clf_dir:
        print(f"Galaxy classification enabled: {galaxy_clf_dir}")
    print("=" * 60)

    # 批量预测
    predict_lsdr10_batch(
        data_dir=data_dir,
        output_dir=output_dir,
        model_base_dir=model_base_dir,
        devices=devices,
        batch_size=batch_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
        galaxy_clf_dir=galaxy_clf_dir,
        isolate_per_file=True,
    )

    print("\nLSDR10 prediction completed!")


def run_ps1dr2():
    """运行 PS1DR2 预测"""
    # 配置路径
    model_base_dir = MODEL_BASE_DIR
    data_dir = PS1DR2_INPUT_DIR
    output_dir = PS1DR2_OUTPUT_DIR
    # galaxy_clf_dir 原目录：'/home/tiandc/photoz/galaxyClf/galaxyClf_ann/PS1DR2/1Allgri'
    galaxy_clf_dir = GALAXY_CLF_PS_DIR
    devices = ['cuda:1']
    batch_size = 32768
    chunk_size = 1_000_000
    num_workers = 1

    print("=" * 60)
    print("PS1DR2 Photo-z Prediction")
    if galaxy_clf_dir:
        print(f"Galaxy classification enabled: {galaxy_clf_dir}")
    print("=" * 60)

    # 批量预测
    predict_ps1dr2_batch(
        data_dir=data_dir,
        output_dir=output_dir,
        model_base_dir=model_base_dir,
        devices=devices,
        batch_size=batch_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
        galaxy_clf_dir=galaxy_clf_dir
    )

    print("\nPS1DR2 prediction completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Photo-z prediction for LSDR10 and PS1DR2 surveys',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py --survey lsdr10
    python predict.py --survey ps1dr2
    python predict.py --survey all
    python predict.py --survey merge
    python predict.py --survey publish
    python predict.py --survey check
        """
    )

    parser.add_argument(
        '--survey',
        type=str,
        required=True,
        choices=['lsdr10', 'ps1dr2', 'all', 'merge', 'publish', 'check'],
        help='Survey to process: lsdr10, ps1dr2, all, merge, publish, or check'
    )

    args = parser.parse_args()

    if args.survey == 'lsdr10':
        run_lsdr10()
    elif args.survey == 'ps1dr2':
        run_ps1dr2()
    elif args.survey == 'all':
        run_lsdr10()
        print("\n" + "=" * 60 + "\n")
        run_ps1dr2()
    elif args.survey == 'merge':
        merge_surveys()
    elif args.survey == 'publish':
        publish_catalog()
    elif args.survey == 'check':
        check_published_catalog()


if __name__ == '__main__':
    main()

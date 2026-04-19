# AUTO-CONVERTED FROM clean.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [markdown] =====
# # 清理数据
# 
# 清理顺序：
# 1. 排除 grz 波段的 fracflux >= 1 的源
# 2. 对于有 GAIA 测量的源，排除自行信噪比 >= 3 的源
# 3. 对于有 PS 数据且 iPSFMag_dered < 21 的，排除 iPSFMag_dered - iKronMag_dered <= 0.05 的源（恒星排除）
# 4. grzW1W2 波段星等必须在 (0, 30) 范围内
# 5. grzW1 波段 SNR 必须大于 1
# 6. i波段：dered_mag_i < 0 的保留（缺失），snr_i > 1 的保留

# ===== CELL 001 [code] =====
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import time
import warnings

# 输入输出目录
input_dir = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy'
output_dir = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy_clean'
os.makedirs(output_dir, exist_ok=True)


def wait_for_file_exists(input_file, fid, sleep_seconds=10, label='输入文件'):
    """等待文件出现。"""
    while not os.path.exists(input_file):
        print(f'[{fid:02d}/71] {label}不存在，等待 {sleep_seconds} 秒: {os.path.basename(input_file)}')
        time.sleep(sleep_seconds)


def wait_for_input_file_complete(input_file, fid, sleep_seconds=10):
    """等待输入文件存在、大小稳定、且 FITS 结构完整后再返回。"""
    last_size = None

    while True:
        if not os.path.exists(input_file):
            print(f'[{fid:02d}/71] 输入文件不存在，等待 {sleep_seconds} 秒: {os.path.basename(input_file)}')
            time.sleep(sleep_seconds)
            continue

        try:
            size_now = os.path.getsize(input_file)
        except OSError as e:
            print(f'[{fid:02d}/71] 无法获取文件大小，等待 {sleep_seconds} 秒后重试: {e}')
            time.sleep(sleep_seconds)
            continue

        if last_size is None or size_now != last_size:
            print(
                f'[{fid:02d}/71] 输入文件仍在写入中，当前大小 {size_now:,} bytes，'
                f'等待 {sleep_seconds} 秒后再次确认'
            )
            last_size = size_now
            time.sleep(sleep_seconds)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with fits.open(input_file, memmap=True) as hdul:
                    if len(hdul) < 2:
                        raise ValueError('missing table HDU')
                    _ = hdul[1].columns
            return
        except Exception as e:
            print(
                f'[{fid:02d}/71] 文件大小已稳定，但 FITS 结构仍不完整，'
                f'等待 {sleep_seconds} 秒后重试: {e}'
            )
            time.sleep(sleep_seconds)


def clean_catalog(t, snr_threshold=1):
    """
    清理星表数据（逐步筛选，每步打印删除数量）

    清理顺序：
    1. fracflux: 排除 grz 波段 fracflux >= 1 的源
    2. 自行: 排除有Gaia测量且自行信噪比 >= 3 的源
    3. PS恒星: 排除PS数据中 iPSFMag < 21 且 iPSF - iKron <= 0.05 的恒星
    4a. 星等: grzW1W2 星等在 (0, 30) 范围内
    4b. 星等: dered_mag_i 不在 (0, 30) 内的赋值为 nan（不删行）
    4c. SNR: snr_i 不在正常范围 (>0) 的赋值为 nan（不删行）
    5. SNR: grzW1W2 波段 snr > threshold
    6. i波段: snr_i 为 nan 的保留（无观测），有正常 snr_i 的要求 > threshold
    """
    n_total = len(t)
    print(f"  初始源数: {n_total:,}")

    mask_fracflux = (t['fracflux_g'] < 1) & (t['fracflux_r'] < 1) & (t['fracflux_z'] < 1)
    n_removed = np.sum(~mask_fracflux)
    t_clean = t[mask_fracflux]
    print(f"  Step 1 - fracflux (grz < 1): 删除 {n_removed:,}, 剩余 {len(t_clean):,}")

    has_gaia = ~np.isnan(t_clean['pmra'])
    pm_snr = np.sqrt(t_clean['pmra']**2 * t_clean['pmra_ivar'] + t_clean['pmdec']**2 * t_clean['pmdec_ivar'])
    mask_pm = ~has_gaia | (pm_snr < 3)
    n_removed = np.sum(~mask_pm)
    t_clean = t_clean[mask_pm]
    print(f"  Step 2 - 自行 (pm_snr < 3): 删除 {n_removed:,}, 剩余 {len(t_clean):,}")

    has_ps = ~np.isnan(t_clean['objID'])
    i_bright = t_clean['iPSFMag_dered'] < 21
    psf_kron_diff = t_clean['iPSFMag_dered'] - t_clean['iKronMag_dered']
    mask_star = ~has_ps | ~i_bright | (psf_kron_diff > 0.05)
    n_removed = np.sum(~mask_star)
    t_clean = t_clean[mask_star]
    print(f"  Step 3 - PS恒星排除 (iPSF-iKron <= 0.05): 删除 {n_removed:,}, 剩余 {len(t_clean):,}")

    mag_cols_required = ['dered_mag_g', 'dered_mag_r', 'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']
    mask_mag = np.ones(len(t_clean), dtype=bool)
    for col in mag_cols_required:
        mask_mag &= (t_clean[col] > 0) & (t_clean[col] < 30) & np.isfinite(t_clean[col])
    n_removed = np.sum(~mask_mag)
    t_clean = t_clean[mask_mag]
    print(f"  Step 4a - 星等有效性 (grzW1W2 in 0-30): 删除 {n_removed:,}, 剩余 {len(t_clean):,}")

    valid_mag_i = (t_clean['dered_mag_i'] > 0) & (t_clean['dered_mag_i'] < 30) & np.isfinite(t_clean['dered_mag_i'])
    n_bad_mag_i = np.sum(~valid_mag_i)
    t_clean['dered_mag_i'][~valid_mag_i] = np.nan
    print(f"  Step 4b - dered_mag_i 清理 (不在0-30设为nan): {n_bad_mag_i:,} 个设为 nan")

    valid_snr_i = (t_clean['snr_i'] > 0) & np.isfinite(t_clean['snr_i'])
    n_bad_snr_i = np.sum(~valid_snr_i)
    t_clean['snr_i'][~valid_snr_i] = np.nan
    print(f"  Step 4c - snr_i 清理 (<=0或非有限设为nan): {n_bad_snr_i:,} 个设为 nan")

    mask_snr = (
        (t_clean['snr_g'] > snr_threshold) & (t_clean['snr_r'] > snr_threshold) &
        (t_clean['snr_z'] > snr_threshold) & (t_clean['snr_w1'] > snr_threshold) &
        (t_clean['snr_w2'] > snr_threshold)
    )
    n_removed = np.sum(~mask_snr)
    t_clean = t_clean[mask_snr]
    print(f"  Step 5 - SNR (grzW1W2 > {snr_threshold}): 删除 {n_removed:,}, 剩余 {len(t_clean):,}")

    mask_i = np.isnan(t_clean['snr_i']) | (t_clean['snr_i'] > snr_threshold)
    n_removed = np.sum(~mask_i)
    t_clean = t_clean[mask_i]
    print(f"  Step 6 - i波段 (snr_i为nan保留 或 snr_i>{snr_threshold}): 删除 {n_removed:,}, 剩余 {len(t_clean):,}")

    for band in ['g', 'r', 'i', 'z', 'w1', 'w2']:
        snr_col = f'snr_{band}'
        err_col = f'mag_{band}_err'
        snr = t_clean[snr_col].copy()
        snr[snr <= 0] = np.nan
        t_clean[err_col] = 1.0857 / snr

    check_cols = [f'dered_mag_{b}' for b in ['g', 'r', 'i', 'z', 'w1', 'w2']] +                  [f'mag_{b}_err' for b in ['g', 'r', 'i', 'z', 'w1', 'w2']]
    for col in check_cols:
        if col in t_clean.colnames:
            data = t_clean[col]
            n_inf = np.sum(np.isinf(data))
            if n_inf > 0:
                print(f"  WARNING: {col} 存在 {n_inf} 个 inf，已替换为 nan")
                t_clean[col][np.isinf(data)] = np.nan

    n_final = len(t_clean)
    print(f"  总计: {n_total:,} -> {n_final:,} ({n_final/n_total*100:.2f}%), 删除 {n_total - n_final:,}")

    stats = {
        'n_total': n_total,
        'n_final': n_final,
    }

    return t_clean, stats

print(f"输入目录: {input_dir}")
print(f"输出目录: {output_dir}")

# ===== CELL 002 [code] =====
# 处理所有72个文件
all_stats = []

for fid in range(72):
    input_file = f'{input_dir}/lsdr9x10_{fid:02d}_xPS1DR2grizy.fits'
    output_file = f'{output_dir}/lsdr9x10_{fid:02d}_xPS1DR2grizy_clean.fits'

    # 检查输出文件是否已存在
    if os.path.exists(output_file):
        print(f'[{fid:02d}/71] 已存在，跳过')
        continue

    if fid < 71:
        next_input_file = f'{input_dir}/lsdr9x10_{fid+1:02d}_xPS1DR2grizy.fits'
        print(f'[{fid:02d}/71] 等待下一文件出现后再清理当前文件: {os.path.basename(next_input_file)}')
        wait_for_file_exists(next_input_file, fid, label='下一输入文件')
    else:
        print(f'[{fid:02d}/71] 最后一个文件，等待当前输入文件完整写入: {os.path.basename(input_file)}')
        wait_for_input_file_complete(input_file, fid)

    t = Table.read(input_file)

    # 清理数据
    print(f"[{fid:02d}/71]")
    t_clean, stats = clean_catalog(t)
    stats['fid'] = fid
    all_stats.append(stats)

    # 保存
    t_clean.write(output_file, overwrite=True)

# 汇总统计
if all_stats:
    total_in = sum(s['n_total'] for s in all_stats)
    total_out = sum(s['n_final'] for s in all_stats)

    print(f'\n{"="*70}')
    print(f'总计: {total_in:,} -> {total_out:,} ({total_out/total_in*100:.2f}%)')


# AUTO-CONVERTED FROM process.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [markdown] =====
# # Recollect `download` -> `raw72`

# ===== CELL 001 [code] =====
import os
import time
import cosmic.utils as cu # type: ignore

def wait_for_enough_fits_files(target_dir: str, target_count: int = 1200) -> None:
    """ 等待目标目录下的.fits文件数量达到指定阈值（默认1200个）"""
    # 先检查目标目录是否存在，避免后续报错
    if not os.path.isdir(target_dir):
        raise ValueError(f"错误：目标目录 '{target_dir}' 不存在！")
    
    print(f"开始监控目录 '{target_dir}' 下的.fits文件，目标数量：{target_count}个")
    
    while True:
        # 统计目录下所有.fits文件（兼容大小写，且只统计文件，排除子目录）
        fits_count = 0
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            # 确保是文件 + 后缀是.fits（不区分大小写）
            if os.path.isfile(file_path) and filename.lower().endswith('.fits'):
                fits_count += 1
        
        # 检查是否达到目标数量
        if fits_count >= target_count:
            print(f"✅ 检测到{fits_count}个.fits文件，已达到目标数量，结束等待")
            break
        
        # 未达到目标数量，打印提示并等待10分钟（600秒）
        print(f"⚠️ 当前仅检测到{fits_count}个.fits文件（目标{target_count}个），将等待10分钟后重新检查...")
        time.sleep(600)  # 10分钟 = 600秒

# ===== CELL 002 [code] =====
target_directory = "/home/tiandc/Data/PanSTARRS/DR2All/download"
wait_for_enough_fits_files(target_directory)

input_dir = "/home/tiandc/Data/PanSTARRS/DR2All/download"
output_dir = "/home/tiandc/Data/PanSTARRS/DR2All/raw72"
cu.merge_fits_by_ra(
    input_dir=input_dir,
    output_dir=output_dir,
    output_filename="ps1dr2_{i:02d}.fits",
    temp_dir="./temp",
    num_processes=None,
    ra_column='raStack',
    fits_extension="*.fits",
    data_hdu_index=1,
    buffer_flush_size=2000000,
    bin_size=5.0,
    add_uid=True
)

# ===== CELL 003 [code] =====
# 收集 raw72 目录下所有文件的位置信息 [uid, raStack, decStack]
import os                                                                                                                                                                                            
import glob                                                                                                                                                                                          
import shutil                                                                                                                                                                                        
import pandas as pd                                                                                                                                                                                  
from tqdm import tqdm

raw72_dir = "/home/tiandc/Data/PanSTARRS/DR2All/raw72"
output_file = "/home/tiandc/Data/PanSTARRS/DR2All/ps1dr2_raw72_loc.fits"
temp_dir = "/home/tiandc/Data/PanSTARRS/DR2All/temp_loc"
save_interval = 10  # 每隔10个文件保存一次，释放内存

# 创建临时目录
os.makedirs(temp_dir, exist_ok=True)

# 获取所有 FITS 文件并按文件名排序
fits_files = sorted(glob.glob(os.path.join(raw72_dir, "*.fits")))
print(f"找到 {len(fits_files)} 个 FITS 文件")

# 要提取的列
cols_to_extract = ['uid', 'raStack', 'decStack']

# 读取并收集所有文件的指定列
subdfs = []
chunk_id = 0
temp_files = []

for i, fpath in enumerate(tqdm(fits_files, desc="读取文件"), start=1):
    df = cu.readfile(fpath)
    sub_df = df[cols_to_extract]
    subdfs.append(sub_df)
    
    # 每隔 save_interval 个文件保存一次并释放内存
    if i % save_interval == 0:
        merged = pd.concat(subdfs, ignore_index=True)
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_id:03d}.fits")
        cu.savefile(merged, temp_file)
        temp_files.append(temp_file)
        print(f"\n[chunk {chunk_id}] 已保存 {len(merged):,} 行到临时文件")
        
        # 释放内存
        del subdfs, merged
        subdfs = []
        chunk_id += 1

# 处理剩余不足 save_interval 的文件
if subdfs:
    merged = pd.concat(subdfs, ignore_index=True)
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_id:03d}.fits")
    cu.savefile(merged, temp_file)
    temp_files.append(temp_file)
    print(f"\n[chunk {chunk_id}] 已保存 {len(merged):,} 行到临时文件")
    del subdfs, merged

# 合并所有临时文件
print(f"\n正在合并 {len(temp_files)} 个临时文件...")
final_tables = []
for tf in tqdm(temp_files, desc="合并临时文件"):
    final_tables.append(cu.readfile(tf))

final_table = pd.concat(final_tables, ignore_index=True)
cu.savefile(final_table, output_file)
print(f"✅ 完成！已保存 {len(final_table):,} 行数据到 {output_file}")

# 清理临时文件
import shutil
shutil.rmtree(temp_dir)
print(f"已清理临时目录 {temp_dir}")

# ===== CELL 004 [markdown] =====
# # 去红化：`raw72` -> `raw72_dered`
# 
# 已在`/home/tiandc/Data/LegacySurveys/DR9x10/code/dataProcess.ipynb`中操作

# ===== CELL 005 [markdown] =====
# # 添加unWISE红外数据

# ===== CELL 006 [markdown] =====
# - 先用stilts交叉匹配

# ===== CELL 007 [code] =====
import subprocess
import os

# === 配置参数 ===
# 建议使用绝对路径，防止报错
stilts_path = "/home/tiandc/download/stilts.jar" 
input1 = "/home/tiandc/Data/PanSTARRS/DR2All/ps1dr2_raw72_loc.fits"       
input2 = "/home/tiandc/Data/unWISE/unwisedr1_clean_all_loc.fits"
output_file = "/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/ps1dr2_raw72loc_xunWISEloc.fits"
heap_size = "128G"

custom_tmp_dir = "/home/tiandc/tmp" 

# 构建命令列表
cmd = [
    "java", f"-Xmx{heap_size}", f"-Djava.io.tmpdir={custom_tmp_dir}", "-jar", stilts_path, 
    "tmatch2",
    "runner=parallel16",          # 多线程
    f"in1={input1}",
    f"in2={input2}",
    f"out={output_file}",
    "matcher=sky",
    "params=1.0",                 # 匹配半径 1 arcsec
    "values1=raStack decStack",             
    "values2=ra dec",
    "join=1and2",                 # 1and2 = inner join
    "find=best",                  # 只要最佳匹配
    "progress=log"                # 显示进度
]

print("正在开始交叉匹配，请耐心等待...")
print("执行命令:", " ".join(cmd))

# 调用系统命令
# check=True 表示如果 STILTS 报错，Python 也会抛出异常停止运行
try:
    subprocess.run(cmd, check=True)
    print(f"\n匹配成功！结果已保存至: {output_file}")
except subprocess.CalledProcessError as e:
    print(f"\n匹配失败，错误代码: {e.returncode}")

# ===== CELL 008 [markdown] =====
# - 添加交叉到的unWISE数据，未交叉到的为NaN

# ===== CELL 009 [code] =====
# ============ 多线程版本 ============
import cosmic.utils as cu
import pandas as pd
import numpy as np
import gc
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# ============ 星等转换函数 ============
def vega_to_ab(mag_vega, band):
    if band == 'w1':
        return mag_vega + 2.699
    elif band == 'w2':
        return mag_vega + 3.339
    else:
        raise ValueError('band must be w1 or w2')

def calc_ab_magerr(flux, flux_err):
    return 1.0857 * (flux_err / flux)

# Read the cross-match location file (shared, read-only)
# uid_1 = PS1DR2 uid, uid_2 = unWISE uid
path = "/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/ps1dr2_raw72loc_xunWISEloc.fits"
df_loc = cu.readfile(path)

# unWISE原始列（需要从unWISE文件中读取）
cols_unwise_raw = ['uid', 'mag_w1_vg', 'mag_w2_vg', 'flux_w1', 'flux_w2', 'dflux_w1', 'dflux_w2']

# 要添加到PS1DR2的新列（转换后）
cols_to_add = ['mag_w1', 'mag_w2', 'mag_w1_err', 'mag_w2_err']

# Create output directory
output_dir = '/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE/'
os.makedirs(output_dir, exist_ok=True)

# Thread-safe print lock
print_lock = threading.Lock()

def process_single_file(i):
    """Process a single file pair"""
    try:
        # Check if output file already exists, skip if so
        output_path = f'{output_dir}ps1dr2_{i:02d}_xunWISE.fits'
        if os.path.exists(output_path):
            with print_lock:
                print(f'[{i:02d}/71] Output file already exists, skipping.')
            return i, True, 'skipped'
        
        # Read current PS1DR2 raw72 file (ALL rows)
        ps_path = f'/home/tiandc/Data/PanSTARRS/DR2All/raw72_dered/ps1dr2_{i:02d}_dered.fits'
        df_ps = cu.readfile(ps_path)
        
        # Initialize new columns with NaN
        for col in cols_to_add:
            df_ps[col] = np.nan
        
        # Read current unWISE clean file
        unwise_path = f'/home/tiandc/Data/unWISE/clean/unwisedr1_clean_{i:02d}.fits'
        df_unwise = cu.readfile(unwise_path)
        
        # Select only needed columns and convert magnitudes
        df_unwise_subset = df_unwise[cols_unwise_raw].copy()
        del df_unwise
        
        # Convert Vega mag to AB mag and calculate errors
        df_unwise_subset['mag_w1'] = vega_to_ab(df_unwise_subset['mag_w1_vg'], 'w1')
        df_unwise_subset['mag_w2'] = vega_to_ab(df_unwise_subset['mag_w2_vg'], 'w2')
        df_unwise_subset['mag_w1_err'] = calc_ab_magerr(df_unwise_subset['flux_w1'], df_unwise_subset['dflux_w1'])
        df_unwise_subset['mag_w2_err'] = calc_ab_magerr(df_unwise_subset['flux_w2'], df_unwise_subset['dflux_w2'])
        
        # Keep only uid and converted columns
        df_unwise_subset = df_unwise_subset[['uid'] + cols_to_add]
        
        # Filter df_loc for current PS1DR2 file's uids
        # uid_1 is PS1DR2 uid, uid_2 is unWISE uid
        df_loc_subset = df_loc[df_loc['uid_1'].isin(df_ps['uid'])].copy()
        
        # Merge df_loc_subset with unWISE data to get mag columns
        df_match = df_loc_subset.merge(df_unwise_subset, left_on='uid_2', right_on='uid', how='left', suffixes=('', '_unwise'))
        del df_unwise_subset, df_loc_subset
        
        # Create a mapping from PS1DR2 uid to unWISE data
        df_match = df_match.set_index('uid_1')
        
        # Set PS1DR2 uid as index for efficient updating
        df_ps = df_ps.set_index('uid')
        
        # Update matched rows with unWISE data
        matched_uids = df_match.index.intersection(df_ps.index)
        for col in cols_to_add:
            df_ps.loc[matched_uids, col] = df_match.loc[matched_uids, col].values
        
        # Reset index to restore uid as a column
        df_ps = df_ps.reset_index()
        
        # Save to output directory
        cu.savefile(df_ps, output_path)
        
        # Report statistics
        n_total = len(df_ps)
        n_matched = len(matched_uids)
        
        with print_lock:
            print(f'[{i:02d}/71] Total: {n_total:,}, Matched: {n_matched:,} ({n_matched/n_total*100:.2f}%)')
        
        # Clean up memory
        del df_ps, df_match
        gc.collect()
        
        return i, True, None
    except Exception as e:
        with print_lock:
            print(f'[{i:02d}/71] Error: {e}')
        return i, False, str(e)

# Number of threads (adjust based on storage type)
# HDD: 2-4 threads, SSD: 6-10 threads
N_THREADS = 2

print(f'Processing 72 files with {N_THREADS} threads...')
print(f'Output: {output_dir}\n')

with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    results = list(executor.map(process_single_file, range(72)))

# Summary
success = sum(1 for _, ok, _ in results if ok)
failed = [(i, err) for i, ok, err in results if not ok]

print(f'\n{"="*50}')
print(f'Completed: {success}/72')
if failed:
    print(f'Failed: {[i for i, _ in failed]}')

# ===== CELL 010 [markdown] =====
# # 数据清理

# ===== CELL 011 [code] =====
import pandas as pd
import numpy as np 
import cosmic.utils as cu     
import os

def clean_panstarrs(df_raw):      
    """PanSTARRS 数据清理 - 保留高质量非恒星源"""   
                   
    df = df_raw.copy() 
    n_orig = len(df)
                              
    # 0. 先展平所有多维列     
    for col in df.columns:    
        if len(df[col].shape) > 1:                  
            df[col] = df[col][:, 0]                 
                   
    # 替换无效值为NaN
    invalid_vals = [-99.0, -99, -999.0, -9999.0, -999, -9999]
    df = df.replace(invalid_vals, np.nan)
    
    # 必须有效的核心列: grizy波段的PSF, Kron, Ap星等及其误差
    core_mag_cols = []
    core_err_cols = []
    for b in ['g', 'r', 'i', 'z', 'y']:
        core_mag_cols.extend([f'{b}PSFMag_dered', f'{b}KronMag_dered', f'{b}ApMag_dered'])
        core_err_cols.extend([f'{b}PSFMagErr', f'{b}KronMagErr', f'{b}ApMagErr'])
    
    existing_mag_cols = [c for c in core_mag_cols if c in df.columns]
    df = df.dropna(subset=existing_mag_cols)
    
    existing_err_cols = [c for c in core_err_cols if c in df.columns]
    df = df.dropna(subset=existing_err_cols)
                   
    # 1. nDetections >= 1 (至少有一次有效探测)
    df = df[np.array(df['nDetections']).flatten() >= 1]
                   
    # ============ objInfoFlag 过滤 ============    
    BAD_OBJ_MASK = 0x00000020 | 0x00000040 | 0x00080000 | 0x00100000      
    flags = np.nan_to_num(np.array(df['objInfoFlag']).flatten().astype(np.int64), nan=0)                   
    df = df[(flags & BAD_OBJ_MASK) == 0]
                   
    # ============ qualityFlag 过滤 ============    
    BAD_QUALITY_MASK = 0x00000040 | 0x00000080
    flags = np.nan_to_num(np.array(df['qualityFlag']).flatten().astype(np.int64), nan=0)
    df = df[(flags & BAD_QUALITY_MASK) == 0]
                   
    # ============ infoFlag 过滤 (每个波段) ============                  
    STAR_MASK = 0x400000
    CONTAMINATION_MASK = 0x8 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x10000
    EXCLUDE_MASK = STAR_MASK | CONTAMINATION_MASK   
                   
    mask = np.ones(len(df), dtype=bool)             
    for b in ['g', 'r', 'i', 'z', 'y']:             
        col = f'{b}infoFlag'  
        if col in df.columns: 
            flags = np.nan_to_num(np.array(df[col]).flatten().astype(np.int64), nan=0)                     
            mask &= (flags & EXCLUDE_MASK) == 0
    df = df[mask]
    
    # ============ 恒星排除 (基于PSF-Kron星等差) ============
    if 'iPSFMag_dered' in df.columns and 'iKronMag_dered' in df.columns:
        i_psf = df['iPSFMag_dered'].values
        i_kron = df['iKronMag_dered'].values
        psf_kron_diff = i_psf - i_kron
        is_bright = i_psf < 21
        is_star = psf_kron_diff <= 0.05
        df = df[~is_bright | ~is_star]
                   
    # 重命名列     
    df = df.rename(columns={'raStack': 'ra', 'decStack': 'dec'})                  
                   
    # 指定要保留的列          
    cols_to_keep = [          
        'objID', 'uid', 'ra', 'dec',             
        'gKronMag_dered', 'rKronMag_dered', 'iKronMag_dered', 'zKronMag_dered', 'yKronMag_dered',          
        'gPSFMag_dered', 'rPSFMag_dered', 'iPSFMag_dered', 'zPSFMag_dered', 'yPSFMag_dered',               
        'gApMag_dered', 'rApMag_dered', 'iApMag_dered', 'zApMag_dered', 'yApMag_dered',                    
        'gKronMagErr', 'rKronMagErr', 'iKronMagErr', 'zKronMagErr', 'yKronMagErr',   
        'gPSFMagErr', 'rPSFMagErr', 'iPSFMagErr', 'zPSFMagErr', 'yPSFMagErr',        
        'gApMagErr', 'rApMagErr', 'iApMagErr', 'zApMagErr', 'yApMagErr',  
        'mag_w1', 'mag_w2', 'mag_w1_err', 'mag_w2_err',
    ]              
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]           
    df_clean = df[cols_to_keep].reset_index(drop=True)                    
    
    print(f'Clean: {n_orig:,} -> {len(df_clean):,} ({len(df_clean)/n_orig*100:.2f}%)')        
                   
    return df_clean

# ===== CELL 012 [code] =====
import os 
output_dir = '/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE_clean/'
os.makedirs(output_dir, exist_ok=True)

for fid in range(72):
    path = f'/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE/ps1dr2_{fid:02d}_xunWISE.fits'
    output_path = os.path.join(output_dir, f'ps1dr2_{fid:02d}_xunWISE_clean.fits')
    if os.path.exists(output_path):
        print(f'[{fid}/71] exists: pass')
        continue
    
    df = cu.readfile(path)
    df_clean = clean_panstarrs(df)
    
    cu.savefile(df_clean, output_path)
    print(f'[{fid}/71] Saved cleaned file: {output_path}')

# ===== CELL 013 [markdown] =====
# # 测试

# ===== CELL 014 [code] =====
import cosmic.utils as cu
from astropy.table import Table
path = "/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE_clean/ps1dr2_00_xunWISE_clean.fits"
tab = Table.read(path)

# ===== CELL 015 [code] =====
tab

# ===== CELL 016 [code] =====
df.groupby('priority').size()

# ===== CELL 017 [code] =====
import numpy as np                                                                                                                                                                                                          
import pandas as pd                                                                                                                                                                                                         
                                                                                                                                                                                                                            
# 模拟 predict_lsdr9x10 的逻辑                                                                                                                                                                                              
n_samples = 10                                                                                                                                                                                                              
priority_array = np.zeros(n_samples, dtype=np.int8)                                                                                                                                                                         
survey_array = np.zeros(n_samples, dtype=np.int8)                                                                                                                                                                           
                                                                                                                                                                                                                            
# 模拟掩码                                                                                                                                                                                                                  
has_native_i = np.array([True, True, False, False, True, False, False, True, False, False])                                                                                                                                 
has_ps1_i = np.array([False, True, True, False, False, True, False, False, True, False])                                                                                                                                    
                                                                                                                                                                                                                            
mask_p1 = has_native_i                                                                                                                                                                                                      
mask_p2 = ~has_native_i & has_ps1_i                                                                                                                                                                                         
mask_p3 = ~has_native_i & ~has_ps1_i                                                                                                                                                                                        
                                                                                                                                                                                                                            
model_configs = [                                                                                                                                                                                                           
    ('p1', mask_p1, 1),                                                                                                                                                                                                     
    ('p2', mask_p2, 2),                                                                                                                                                                                                     
    ('p3', mask_p3, 3),                                                                                                                                                                                                     
]                                                                                                                                                                                                                           
                                                                                                                                                                                                                            
print("Before loop:")                                                                                                                                                                                                       
print(f"  priority_array dtype: {priority_array.dtype}")                                                                                                                                                                    
print(f"  priority_array: {priority_array}")                                                                                                                                                                                
                                                                                                                                                                                                                            
for model_name, mask, priority in model_configs:                                                                                                                                                                            
    indices = np.where(mask)[0]                                                                                                                                                                                             
    print(f"\n{model_name}: indices={indices}, priority={priority}, type(priority)={type(priority)}")                                                                                                                       
    priority_array[indices] = priority                                                                                                                                                                                      
    print(f"  After assignment: {priority_array}")                                                                                                                                                                          
                                                                                                                                                                                                                            
print(f"\nFinal priority_array: {priority_array}")                                                                                                                                                                          
print(f"Final dtype: {priority_array.dtype}")                                                                                                                                                                               
                                                                                                                                                                                                                            
# 构建 DataFrame                                                                                                                                                                                                            
result = pd.DataFrame({                                                                                                                                                                                                     
    'priority': priority_array,                                                                                                                                                                                             
    'survey': survey_array,                                                                                                                                                                                                 
})                                                                                                                                                                                                                          
print(f"\nDataFrame:\n{result}")                                                                                                                                                                                            
print(f"\nDataFrame dtypes:\n{result.dtypes}")

# ===== CELL 018 [code] =====
result

# ===== CELL 019 [code] =====
import pandas as pd
from astropy.table import Table
import os

def read_fits(filepath: str) -> pd.DataFrame:
    """读取 FITS 文件为 DataFrame"""
    table = Table.read(filepath, format='fits')
    return table.to_pandas()

def read_table(filepath: str) -> pd.DataFrame:
    """根据扩展名读取表。"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.fits', '.fit', '.fz']:
        return read_fits(filepath)
    raise ValueError(f"Unsupported input format: {filepath}")

path = '/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE/ps1dr2_00_xunWISE.fits'
df = read_table(path)

# ===== CELL 020 [code] =====
df.columns

# ===== CELL 021 [code] =====
print(df.uid.duplicated().sum())
print(df.uid.isna().sum())


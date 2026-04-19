# AUTO-CONVERTED FROM dataProcess.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [markdown] =====
# # 数据清理: `raw_seg/` --> `clean`

# ===== CELL 001 [code] =====
# Data clean for downloaded LS DR10 phot data
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import glob
import threading
import concurrent.futures
import cosmic.utils as cu

def get_file_path():
    return ['../raw_seg/lsdr9x10_gal_seg_%02d.fits'%(i) for i in range(72)]

def clean_star(df):
    idx = (df['dered_mag_r'] - df['dered_mag_z']) >= 1
    idx &= (df['dered_mag_z'] - df['dered_mag_w1']) < (1.2*(df['dered_mag_r'] - df['dered_mag_z']) - 1.5)
    return df[~idx]
    
# def clean_FRAC(df):
#     idx = (df['fracflux_g'] < 0.5) & (df['fracflux_r'] < 0.5) & (df['fracflux_z'] < 0.5)
#     idx &= (df['fracmasked_g'] < 0.4) & (df['fracmasked_r'] < 0.4) & (df['fracmasked_z'] < 0.4)
#     idx &= (df['fracin_g'] > 0.3) & (df['fracin_r'] > 0.3) & (df['fracin_z'] > 0.3)
#     return df[idx]

def clean_snr(df):
    idx = (df.snr_g > 5) & (df.snr_r > 5) & (df.snr_z > 5)
    idx &= (df.snr_w1 > 5) & (df.snr_w2 > 5)
    return df[idx]

def clean_mag(df):
    idx = (df['dered_mag_g'] > 0) & (df['dered_mag_r'] > 0) & (df['dered_mag_z'] > 0)
    idx = (df['dered_mag_w1'] > 0) & (df['dered_mag_w2'] > 0)
    return df[idx]

def assign_z_and_zErr(df):
    """
    Assign redshift values and errors based on priority:
    1. z_spec (if > 0) -> z = z_spec, zErr = 0
    2. z_phot_mean_i -> z = z_phot_mean_i, zErr = z_phot_std_i
    3. z_phot_mean_grz -> z = z_phot_mean_grz, zErr = z_phot_std_grz
    
    Returns dataframe with only rows where z > 0 and zErr > 0
    """
    # Create new columns
    df = df.copy()
    df["z"] = np.nan
    df["zErr"] = np.nan
    
    # Priority 1: z_spec (if > 0)
    spec_mask = df["z_spec"] > 0
    df.loc[spec_mask, "z"] = df.loc[spec_mask, "z_spec"]
    df.loc[spec_mask, "zErr"] = 0.0
    
    # Priority 2: z_phot_mean_i (if not already assigned and > 0)
    i_mask = (df["z"].isna()) & (df["z_phot_mean_i"] > 0) & (df["z_phot_std_i"] > 0)
    df.loc[i_mask, "z"] = df.loc[i_mask, "z_phot_mean_i"]
    df.loc[i_mask, "zErr"] = df.loc[i_mask, "z_phot_std_i"]
    
    # Priority 3: z_phot_mean_grz (if not already assigned and > 0)
    grz_mask = (df["z"].isna()) & (df["z_phot_mean_grz"] > 0) & (df["z_phot_std_grz"] > 0)
    df.loc[grz_mask, "z"] = df.loc[grz_mask, "z_phot_mean_grz"]
    df.loc[grz_mask, "zErr"] = df.loc[grz_mask, "z_phot_std_grz"]
    
    # Return only rows where z > 0 and zErr > 0
    result_mask = (df["z"] > 0) & (df["zErr"] > 0)
    return df[result_mask]

def process_single_file(args):
    """Process a single file and save it immediately after processing"""
    path, file_idx, total_files, uid_lock, uid_counter, stats_lock, stats = args
    fid = os.path.basename(path).split('_')[3].split('.')[0]
    new_path = '../clean/lsdr9x10_gal_clean_%s.fits'%(fid)
    
    if os.path.exists(new_path):
        print(f'[{file_idx}/{total_files}] File {fid} already exists, skipping.')
        return
    
    # Read and process file
    df = cu.readfile(path)
    raw_count = df.shape[0]
    
    # Apply cleaning filters
    df = clean_mag(df)
    df = assign_z_and_zErr(df)
    df = clean_star(df)
    # df = clean_FRAC(df)
    # df = clean_snr(df)
    
    clean_count = df.shape[0]
    
    # Thread-safe: assign uid
    with uid_lock:
        uid_start = uid_counter['value']
        df['uid'] = range(uid_start, uid_start + len(df))
        uid_counter['value'] += len(df)
        
    # Save file and print status (outside the lock)
    cu.savefile(df, new_path)
    print(f'[{file_idx}/{total_files}] ✓ Saved {fid}: {raw_count} -> {clean_count} galaxies (uid: {uid_start} ~ {uid_start + len(df) - 1})')
    
    # Thread-safe: update statistics
    with stats_lock:
        stats['raw'] += raw_count
        stats['clean'] += clean_count

def process():
    """Process all files using multi-threading with 10 threads - saves immediately after processing"""
    file_list = get_file_path()
    total_files = len(file_list)
    print(f'Total files to process: {total_files}')
    print(f'Using 10 threads for parallel processing...')
    print(f'Files will be saved immediately after processing.\n')
    
    # Shared variables with thread locks
    uid_lock = threading.Lock()
    uid_counter = {'value': 0}  # Use dict to allow modification in nested scope
    stats_lock = threading.Lock()
    stats = {'raw': 0, 'clean': 0}
    
    # Prepare arguments for each file
    file_args = [(path, idx+1, total_files, uid_lock, uid_counter, stats_lock, stats) 
                 for idx, path in enumerate(file_list)]
    
    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_file, arg) for arg in file_args]
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    print('\n' + '='*50)
    print(f'Processing complete!')
    print(f'Raw galaxy number: {stats["raw"]}')
    print(f'Clean galaxy number: {stats["clean"]}')
    if stats['raw'] > 0:
        print(f'Retention rate: {(stats["clean"]/stats["raw"])*100:.2f}%')
        print(f'Reduction rate: {(1 - stats["clean"]/stats["raw"])*100:.2f}%')
    print('='*50)

# ===== CELL 002 [code] =====
process()

# ===== CELL 003 [markdown] =====
# # collect position of all clean galaxies

# ===== CELL 004 [code] =====
import pandas as pd
import numpy as np
import os
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cosmic.utils as cu
from tqdm import tqdm

def optimized_read_files(cols=['uid', 'ra', 'dec'], max_workers=8):
    """
    优化版本的文件读取和合并
    """
    def read_single_file(i):
        """读取单个文件"""
        path = f'../clean/lsdr9x10_gal_clean_{i:02d}.fits'
        try:
            df = cu.readfile(path)
            return df[cols].copy()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return pd.DataFrame(columns=cols)
    
    # 并行读取所有文件
    print("Reading files in parallel...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(read_single_file, i) for i in range(72)]
        
        # 收集结果，显示进度条
        dataframes = []
        for future in tqdm(futures, desc="Reading files"):
            df = future.result()
            if not df.empty:
                dataframes.append(df)
    
    # 一次性合并所有数据
    print("Concatenating dataframes...")
    if dataframes:
        result_df = pd.concat(dataframes, ignore_index=True)
        # 清理中间数据
        del dataframes
        gc.collect()
    else:
        result_df = pd.DataFrame(columns=cols)
    
    return result_df

result_df = optimized_read_files()

# 保存结果
output_path = '../lsdr9x10_gal_clean_allLoc.fits'
cu.savefile(result_df, output_path)
print(f"Saved to {output_path}")

# ===== CELL 005 [markdown] =====
# # collect position of all galaxies

# ===== CELL 006 [code] =====
import pandas as pd
import numpy as np
import os
import gc
import cosmic.utils as cu
from tqdm import tqdm

def read_files_add_uid():
    """
    顺序读取每个文件，添加连续uid列
    保存完整文件（带uid）到 raw72/
    最后合并保存位置文件（uid, ra, dec）
    """
    os.makedirs('../raw72', exist_ok=True)
    uid_start = 0
    loc_dfs = []
    
    for i in tqdm(range(72), desc="Processing files"):
        path = f'../raw_seg/lsdr9x10_gal_seg_{i:02d}.fits'
        try:
            df = cu.readfile(path)
            
            # 添加uid列
            df.insert(0, 'uid', range(uid_start, uid_start + len(df)))
            uid_end = uid_start + len(df) - 1
            
            # 保存完整文件（带uid）
            full_path = f'../raw72/lsdr9x10_gal_seg_{i:02d}.fits'
            cu.savefile(df, full_path)
            
            # 收集位置数据
            loc_dfs.append(df[['uid', 'ra', 'dec']].copy())
            
            print(f"[{i:02d}/71] rows: {len(df)}, uid: {uid_start} ~ {uid_end}")
            uid_start = uid_end + 1
            
            # 释放原数据内存
            del df
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # 合并并保存所有位置数据
    print("\nConcatenating location data...")
    loc_df = pd.concat(loc_dfs, ignore_index=True)
    del loc_dfs
    gc.collect()
    
    loc_path = '../lsdr9x10_gal_raw72_allLoc.fits'
    cu.savefile(loc_df, loc_path)
    print(f"Saved location file: {loc_path}")
    print(f"Total rows: {len(loc_df)}, uid: 0 ~ {uid_start - 1}")

read_files_add_uid()

# ===== CELL 007 [markdown] =====
# # LSDR9x10 `raw72` 与PANSTARRS `raw72` 交叉匹配，获取i波段数据： `raw72`->`raw72_xPS1DR2`

# ===== CELL 008 [markdown] =====
# - 使用STILTS.jar进行交叉匹配

# ===== CELL 009 [code] =====
import subprocess
import os

# === 配置参数 ===
# 建议使用绝对路径，防止报错
stilts_path = "/home/tiandc/download/stilts.jar" 
input1 = "/home/tiandc/Data/PanSTARRS/DR2All/ps1dr2_raw72_loc.fits"       
input2 = "/home/tiandc/Data/LegacySurveys/DR9x10/lsdr9x10_gal_raw72_allLoc.fits"
output_file = "/home/tiandc/Data/LegacySurveys/DR9x10/lsdr9x10_gal_raw72_allLoc_xPS1DR2raw72Loc.fits"
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

# ===== CELL 010 [markdown] =====
# - add PS1DR2 i band photometric data

# ===== CELL 011 [code] =====
# ============ 多线程版本 ============
import cosmic.utils as cu
from cosmic.panstarrs_dr2.dataProcess import correct_extinction
import pandas as pd
import numpy as np
import gc
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Read the cross-match location file (shared, read-only)
path = '/home/tiandc/Data/LegacySurveys/DR9x10/xPS1DR2/lsdr9x10_gal_raw72_allLoc_xPS1DR2raw72Loc.fits'
df_loc = cu.readfile(path)

# Columns to add to LSDR9x10 (after extinction correction, with _dered suffix)
cols_ps_iband = [
    'objID', 'iPSFMag_dered', 'iKronMag_dered', 'iApMag_dered',
    'iPSFMagErr', 'iApMagErr', 'iKronMagErr',
]

# Columns to save for extinction-corrected PS1DR2 data
cols_ps_save = [
    'objID', 'raStack', 'decStack', 'l', 'b', 'nDetections', 'gPSFMag_dered',
    'gPSFMagErr', 'gApMag_dered', 'gApMagErr', 'gKronMag_dered', 'gKronMagErr',
    'rPSFMag_dered', 'rPSFMagErr', 'rApMag_dered', 'rApMagErr', 'rKronMag_dered',
    'rKronMagErr', 'iPSFMag_dered', 'iPSFMagErr', 'iApMag_dered', 'iApMagErr',
    'iKronMag_dered', 'iKronMagErr', 'zPSFMag_dered', 'zPSFMagErr', 'zApMag_dered',
    'zApMagErr', 'zKronMag_dered', 'zKronMagErr', 'yPSFMag_dered', 'yPSFMagErr',
    'yApMag_dered', 'yApMagErr', 'yKronMag_dered', 'yKronMagErr', 'ginfoFlag',
    'ginfoFlag2', 'ginfoFlag3', 'rinfoFlag', 'rinfoFlag2', 'rinfoFlag3',
    'iinfoFlag', 'iinfoFlag2', 'iinfoFlag3', 'zinfoFlag', 'zinfoFlag2',
    'zinfoFlag3', 'yinfoFlag', 'yinfoFlag2', 'yinfoFlag3', 'qualityFlag',
    'objInfoFlag', 'uid'
]

# Create output directories
output_dir = '/home/tiandc/Data/LegacySurveys/DR9x10/xPS1DR2/raw72_xPS1DR2/'
ps_dered_dir = '/home/tiandc/Data/PanSTARRS/DR2All/raw72_dered/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ps_dered_dir, exist_ok=True)

# Thread-safe print lock
print_lock = threading.Lock()

def process_single_file(i):
    """Process a single file pair"""
    try:
        # Read current LSDR9x10 raw72 file (ALL rows)
        ls_path = f'/home/tiandc/Data/LegacySurveys/DR9x10/raw72/lsdr9x10_gal_seg_{i:02d}.fits'
        df_ls = cu.readfile(ls_path)
        
        # Initialize new columns with NaN
        for col in cols_ps_iband:
            df_ls[col] = np.nan
        
        # Read current PS1DR2 raw72 file
        ps_path = f'/home/tiandc/Data/PanSTARRS/DR2All/raw72/ps1dr2_{i:02d}.fits'
        df_ps = cu.readfile(ps_path)
        
        # Apply extinction correction and select columns to save
        df_ps = correct_extinction(df_ps)[cols_ps_save]
        
        # Save extinction-corrected PS1DR2 data
        ps_dered_path = f'{ps_dered_dir}ps1dr2_{i:02d}_dered.fits'
        cu.savefile(df_ps, ps_dered_path)
        
        # Select only needed columns for merging
        df_ps_subset = df_ps[['uid'] + cols_ps_iband].copy()
        del df_ps
        
        # Filter df_loc for current LSDR9x10 file's uids
        # uid_1 is PS1DR2 uid, uid_2 is LSDR9x10 uid
        df_loc_subset = df_loc[df_loc['uid_2'].isin(df_ls['uid'])].copy()
        
        # Merge df_loc_subset with PS1DR2 data to get i-band columns
        df_match = df_loc_subset.merge(df_ps_subset, left_on='uid_1', right_on='uid', how='left', suffixes=('', '_ps'))
        del df_ps_subset, df_loc_subset
        
        # Create a mapping from LSDR9x10 uid to PS1DR2 i-band data
        df_match = df_match.set_index('uid_2')
        
        # Set LSDR9x10 uid as index for efficient updating
        df_ls = df_ls.set_index('uid')
        
        # Update matched rows with PS1DR2 i-band data
        matched_uids = df_match.index.intersection(df_ls.index)
        for col in cols_ps_iband:
            df_ls.loc[matched_uids, col] = df_match.loc[matched_uids, col].values
        
        # Reset index to restore uid as a column
        df_ls = df_ls.reset_index()
        
        # Save to output directory
        output_path = f'{output_dir}lsdr9x10_{i:02d}_xPS1DR2i.fits'
        cu.savefile(df_ls, output_path)
        
        # Report statistics
        n_total = len(df_ls)
        n_matched = len(matched_uids)
        
        with print_lock:
            print(f'[{i:02d}/71] Total: {n_total}, Matched: {n_matched} ({n_matched/n_total*100:.2f}%)')
        
        # Clean up memory
        del df_ls, df_match
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
print(f'Output: {output_dir}')
print(f'PS1DR2 dered: {ps_dered_dir}\n')

with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    results = list(executor.map(process_single_file, range(72)))

# Summary
success = sum(1 for _, ok, _ in results if ok)
failed = [(i, err) for i, ok, err in results if not ok]

print(f'\n{"="*50}')
print(f'Completed: {success}/72')
if failed:
    print(f'Failed: {[i for i, _ in failed]}')

# ===== CELL 012 [code] =====
import pandas as pd
import numpy as np
import cosmic.utils as cu

def clean_lsdr9x10(df_raw):               
    """                   
    清理LSDR9x10数据      

    Parameters            
    ----------            
    df : DataFrame        
        输入数据          
              
    Returns               
    -------               
    DataFrame: 清理后的数据      
    """                   
    import numpy as np
    df = df_raw.copy()

    # 重命名 uid -> uid_ls
    if 'uid' in df.columns:
        df['uid_ls'] = df['uid']
    
    # 从 SNR 计算星等误差 (mag_err ≈ 1.0857 / SNR)
    snr_to_err = {
        'snr_g': 'mag_g_Err',
        'snr_r': 'mag_r_Err',
        'snr_i': 'mag_i_Err',
        'snr_z': 'mag_z_Err',
        'snr_w1': 'mag_w1_Err',
        'snr_w2': 'mag_w2_Err',
    }
    
    for snr_col, err_col in snr_to_err.items():
        if snr_col in df.columns:
            # 避免除零，SNR <= 0 的设为 NaN
            snr = df[snr_col].values.astype(float)
            snr = np.where(snr > 0, snr, np.nan)
            df[err_col] = 1.0857 / snr

    mask = np.ones(len(df), dtype=bool)                  

    # (1) grz波段的SNR都要大于0  
    mask &= (df['snr_g'] > 1) & (df['snr_r'] > 1) & (df['snr_z'] > 1)            

    # (2) grzW1W2星等大于0且在合理范围内 (0, 30)         
    mag_cols = ['dered_mag_g', 'dered_mag_r', 'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']             
    for col in mag_cols:  
        mask &= (df[col] > 0) & (df[col] < 30) & np.isfinite(df[col])            

    # (3) W1W2的SNR>1以确保星等误差有效                  
    mask &= (df['snr_w1'] > 1) & (df['snr_w2'] > 1)    
    
    # (4) 如果 i 波段有数据（非 nan），必须 valid                                                                                                                                           
    i_is_nan = np.isnan(df['dered_mag_i'])                                                                                                                                                  
    i_band_valid = (                                                                                                                                                                        
        (df['snr_i'] > 1) &                                                                                                                                                                 
        (df['dered_mag_i'] > 0) &                                                                                                                                                           
        (df['dered_mag_i'] < 30) &                                                                                                                                                          
        np.isfinite(df['dered_mag_i'])                                                                                                                                                      
    )                                                                                                                                                                                       
    mask &= i_is_nan | i_band_valid
     
    # 应用过滤            
    df_clean = df[mask].reset_index(drop=True)           
        
    # 指定要保留的列
    cols_to_keep = [          
        'uid_ls', 'ra', 'dec',      
        'dered_mag_g', 'dered_mag_r', 'dered_mag_i', 'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2', 
        'mag_g_Err', 'mag_r_Err', 'mag_i_Err', 'mag_z_Err', 'mag_w1_Err', 'mag_w2_Err',
        'objID', 'iPSFMag_dered', 'iKronMag_dered', 'iPSFMagErr', 'iKronMagErr',  # PS1DR2 i-band   
        'iApMag_dered', 'iApMagErr',             
    ] 
    df_clean = df_clean[cols_to_keep]                

    print(f'Clean: {len(df):,} -> {len(df_clean):,} ({len(df_clean)/len(df)*100:.2f}%)')                 

    return df_clean

# ===== CELL 013 [code] =====
import os 
output_dir = '/home/tiandc/Data/LegacySurveys/DR9x10/xPS1DR2/raw72_xPS1DR2_clean_photozInput/'
os.makedirs(output_dir, exist_ok=True)

for fid in range(72):
    path = f'/home/tiandc/Data/LegacySurveys/DR9x10/xPS1DR2/raw72_xPS1DR2/lsdr9x10_{fid:02d}_xPS1DR2i.fits'
    output_path = os.path.join(output_dir, f'lsdr9x10_{fid:02d}_xPS1DR2i_clean_photozInput.fits')
    if os.path.exists(output_path):
        print(f'[{fid}/71] exists: pass')
        continue
    
    df = cu.readfile(path)
    df_clean = clean_lsdr9x10(df)
    
    cu.savefile(df_clean, output_path)
    print(f'[{fid}/71] Saved cleaned file: {output_path}')


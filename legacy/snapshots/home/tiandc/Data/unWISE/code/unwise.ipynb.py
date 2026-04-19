# AUTO-CONVERTED FROM unwise.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [markdown] =====
# # unWISE DR1 Data Download -> `download/`
# 从 NOIRLab DataLab 下载 unWISE DR1 目录数据，支持自动细分大区域

# ===== CELL 001 [code] =====
# unWISE DR1 数据下载 - 使用 NOIRLab DataLab
import warnings
warnings.simplefilter("ignore")
import os

# 确保代理设置被 Python requests 库使用
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

import numpy as np
from dl import authClient as ac, queryClient as qc
from concurrent.futures import ThreadPoolExecutor, as_completed

# 登录 DataLab
TOKEN = ac.login(user='tiandc1998', password='290923272Tdc.')
print("DataLab 登录成功")

# ===== CELL 002 [code] =====
# 测试 DataLab 连通性
def test_datalab_connectivity():
    """测试服务器连通性"""
    test_query = """
        SELECT COUNT(*) 
        FROM unwise_dr1.object
        WHERE ra >= 0 AND ra < 0.1 AND dec >= 0 AND dec < 0.1
    """
    try:
        print("正在测试 DataLab 连通性...")
        result = qc.query(TOKEN, sql=test_query, fmt='pandas', timeout=60)
        if result is not None:
            count = result.iloc[0, 0]
            print(f"✓ 连接成功! 测试区域数据量: {count:,}")
            return True
        else:
            print("✗ 连接失败: 无响应")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {type(e).__name__}: {e}")
        return False

test_datalab_connectivity()

# ===== CELL 003 [code] =====
import time
import random

def make_seg(output_dir, n_ra, n_dec):
    """使用向量化操作生成天区分割并保存到readme.txt"""
    ra_edges = np.linspace(0, 360, n_ra + 1)
    dec_edges = np.linspace(-90, 90, n_dec + 1)

    # 使用 meshgrid 向量化生成所有组合
    ra_grid, dec_grid = np.meshgrid(range(n_ra), range(n_dec), indexing='ij')

    # 展平并构建数组
    n_total = n_ra * n_dec
    arr = np.column_stack([
        np.arange(n_total),           # fid
        ra_edges[ra_grid.ravel()],    # ra_l
        ra_edges[ra_grid.ravel() + 1],# ra_u
        dec_edges[dec_grid.ravel()],  # dec_l
        dec_edges[dec_grid.ravel() + 1]# dec_u
    ])

    cols = ['fid', 'ra_l', 'ra_u', 'dec_l', 'dec_u']
    df = pd.DataFrame(arr, columns=cols)
    path = f'{output_dir}/readme.txt'
    df.to_csv(path, index=False)
    print(f"分割信息已保存到 {path}")
    return df


def count_region_data(ra_lo, ra_hi, dec_lo, dec_hi):
    """查询区域数据量"""
    # 随机延迟，避免并发请求过多导致代理断开
    time.sleep(random.uniform(2, 8))
    
    count_query = f"""
    SELECT COUNT(*) 
    FROM unwise_dr1.object
    WHERE 
        "primary" = 1
        AND ra >= {ra_lo} AND ra < {ra_hi}
        AND dec >= {dec_lo} AND dec < {dec_hi}
    """
    result = qc.query(TOKEN, sql=count_query, fmt='pandas', timeout=600)
    return result.iloc[0, 0]


def download_region(idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, count, sub_idx=None, max_retries=3):
    """下载单个区域的数据，支持自动重试"""
    # 随机延迟，避免并发请求过多导致SSL错误
    time.sleep(random.uniform(2, 8))
    
    if sub_idx is None:
        output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}.fits")
        file_id = f"{idx:04d}"
    else:
        output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}_{sub_idx}.fits")
        file_id = f"{idx:04d}_{sub_idx}"
    
    # 检查文件是否已存在
    if os.path.exists(output_file):
        return f"[{file_id}] (已存在)"
    
    # unWISE 数据查询 - 根据实际列名
    data_query = f"""
    SELECT
        unwise_objid, ra, dec, glon, glat,
        mag_w1_vg, mag_w2_vg, w1_w2_vg,
        
        -- 流量及误差
        flux_w1, flux_w2,
        dflux_w1, dflux_w2,
        fluxlbs_w1, fluxlbs_w2,
        dfluxlbs_w1, dfluxlbs_w2,
        
        -- 质量指标
        qf_w1, qf_w2,
        rchi2_w1, rchi2_w2,
        fracflux_w1, fracflux_w2,
        fwhm_w1, fwhm_w2,
        nm_w1, nm_w2,
        
        -- 标志位
        "primary",
        flags_unwise_w1, flags_unwise_w2,
        flags_info_w1, flags_info_w2
        
    FROM 
        unwise_dr1.object
    WHERE 
        "primary" = 1
        AND ra >= {ra_lo} AND ra < {ra_hi}
        AND dec >= {dec_lo} AND dec < {dec_hi}
    """
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # 直接查询并保存为FITS
            qc.query(TOKEN, sql=data_query, fmt='fits', out=output_file, timeout=86400)
            
            # 检查文件
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                return f"[{file_id}] {count:,} rows, {file_size_mb:.2f} MB"
            else:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return f"[{file_id}] 下载失败，文件为空"
                
        except Exception as e:
            # 删除可能的部分文件
            if os.path.exists(output_file):
                os.remove(output_file)
            
            if attempt < max_retries - 1:
                print(f"  [{file_id}] 重试 {attempt + 2}/{max_retries}")
                time.sleep(10)  # 等待10秒后重试
            else:
                return f"[{file_id}] 下载失败 ({max_retries}次重试): {str(e)[:50]}"


def process_region(idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, max_rows=5000000):
    """处理单个区域：先查询数据量，决定是否细分"""
    
    # 先检查文件是否已存在（避免不必要的count查询）
    output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}.fits")
    if os.path.exists(output_file):
        return [f"[{idx:04d}] (已存在)"]
    
    try:
        # 查询数据量
        count = count_region_data(ra_lo, ra_hi, dec_lo, dec_hi)
        
        results = []
        
        # 判断是否需要细分
        if count == 0:
            results.append(f"[{idx:04d}] 空区域，跳过")
        elif count <= max_rows:
            # 直接下载
            result = download_region(idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, count)
            results.append(result)
        else:
            # 需要细分 - 按RA方向细分
            n_subdivisions = int(np.ceil(count / max_rows))
            
            # 计算细分的RA边界
            sub_ra_bins = np.linspace(ra_lo, ra_hi, n_subdivisions + 1)
            
            # 下载每个子区域
            for j in range(n_subdivisions):
                sub_ra_lo = sub_ra_bins[j]
                sub_ra_hi = sub_ra_bins[j + 1]
                # 子区域需要重新查询数据量
                sub_count = count_region_data(sub_ra_lo, sub_ra_hi, dec_lo, dec_hi)
                result = download_region(idx, sub_ra_lo, sub_ra_hi, dec_lo, dec_hi, output_dir, sub_count, sub_idx=j)
                results.append(result)
        
        return results
        
    except Exception as e:
        return [f"[{idx:04d}] 处理出错: {str(e)}"]


# ============ 配置参数 ============
MAX_ROWS = 5000000       # 每次查询最大行数
N_RA_SEGMENTS = 180       # RA方向分割数
N_DEC_SEGMENTS = 60      # DEC方向分割数
MAX_WORKERS = 10          # 并行线程数

# 保存目录
output_dir = "../download"
os.makedirs(output_dir, exist_ok=True)

# 生成分割信息并保存
import pandas as pd
seg_df = make_seg(output_dir, N_RA_SEGMENTS, N_DEC_SEGMENTS)

# 生成所有区域任务
tasks = []
for idx, row in seg_df.iterrows():
    tasks.append((int(row['fid']), row['ra_l'], row['ra_u'], row['dec_l'], row['dec_u']))

print(f"开始使用 {MAX_WORKERS} 个线程下载数据...")
print(f"总共 {len(tasks)} 个区域 ({N_RA_SEGMENTS} x {N_DEC_SEGMENTS})")
print(f"每个文件最大行数: {MAX_ROWS:,}\n")

# 并行下载
all_results = []
completed = 0
total = len(tasks)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务
    futures = {}
    for task in tasks:
        idx, ra_lo, ra_hi, dec_lo, dec_hi = task
        future = executor.submit(process_region, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, MAX_ROWS)
        futures[future] = idx
    
    # 处理完成的任务
    for future in as_completed(futures):
        idx = futures[future]
        try:
            results = future.result()
            for result in results:
                print(result)
                all_results.append(result)
        except Exception as e:
            print(f"[{idx:04d}] 线程执行出错: {str(e)}")
        
        completed += 1
        if completed % 10 == 0:
            print(f"进度: {completed}/{total} ({100*completed/total:.1f}%)\n")

print("\n下载完成!")
print(f"总共生成了 {len(all_results)} 个文件")

# ===== CELL 004 [code] =====
table.info()

# ===== CELL 005 [markdown] =====
# # Data Recollected into 72 Files `download/` -> `raw/`

# ===== CELL 006 [code] =====
from cosmic.utils import merge_fits_by_ra
result = merge_fits_by_ra(
    input_dir="../download",
    output_dir="../raw",
    output_filename="unwisedr1_{i:02d}.fits",
    temp_dir="../temp",
    num_processes=20,
    ra_column='ra',
    fits_extension="*.fits",
    data_hdu_index=1,
    buffer_flush_size=1000000,
    bin_size=5.0
)

print(f"\n结果: 共处理 {result['total_rows']:,} 行，生成 {len(result['output_files'])} 个文件")

# ===== CELL 007 [markdown] =====
# # Data Clean `raw/ -> clean/`

# ===== CELL 008 [code] =====
# unWISE DR1 Data Cleaning
# 基于官方文档和论文推荐的数据清理标准
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import glob
import threading
import concurrent.futures
from astropy.table import Table
import cosmic.utils as cu

# unWISE波段
UNWISE_BANDS = ['w1', 'w2']

# ============ flags_unwise 位掩码定义 ============
# 所有位都标记亮星相关的伪影
FLAGS_UNWISE = {
    0: 'In core or wings',           # 在亮星核心或翼部
    1: 'In diffraction spike',       # 在衍射尖峰
    2: 'In ghost',                   # 在鬼像
    3: 'In first latent',            # 在第一潜影
    4: 'In second latent',           # 在第二潜影
    5: 'In circular halo',           # 在圆形光晕
    6: 'Saturated',                  # 饱和
    7: 'In geometric diffraction spike'  # 在几何衍射尖峰
}

# ============ flags_info 位掩码定义 ============
FLAGS_INFO = {
    0: 'In PSF of bright star falling off coadd',  # 在落到coadd边缘的亮星PSF中
    1: 'In HyperLeda large galaxy',                # 在HyperLeda大星系中
    2: 'In big object (e.g., Magellanic cloud)',   # 在大天体中（如麦哲伦云）
    3: 'May contain centroid of very bright star', # 可能包含非常亮星的质心
    4: 'Potentially affected by saturation',       # 可能受饱和影响
    5: 'May contain nebulosity',                   # 可能包含星云
    6: 'Will not be aggressively deblended',       # 不会被激进地分解
    7: 'Must be sharp to be optimized'             # 必须锐利才能优化
}


def get_file_path():
    """获取所有unWISE原始文件路径"""
    return sorted(glob.glob('../raw/unwisedr1_*.fits'))

def read_unwise(filepath):
    """读取unWISE FITS文件，返回DataFrame"""
    table = Table.read(filepath, hdu=1)
    df = table.to_pandas()
    
    # 转换字节序
    for col in df.columns:
        if hasattr(df[col].dtype, 'byteorder') and df[col].dtype.byteorder not in ('=', '|', '<'):
            df[col] = df[col].astype(df[col].dtype.newbyteorder('='))
    
    # 转换bytes类型的列（如unwise_objid）
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
    
    return df

def filter_by_flags(df):
    """
    根据flags清理unWISE数据
    
    Parameters
    ----------
    df : DataFrame
    strict : bool
        True: 严格模式，两个波段都必须通过所有检查
        False: 宽松模式，允许某些非关键位被设置
    
    Returns
    -------
    DataFrame: 过滤后的数据
    
    Notes
    -----
    flags_unwise位定义（所有位都标记亮星伪影）:
        bit 0: In core or wings
        bit 1: In diffraction spike
        bit 2: In ghost
        bit 3: In first latent
        bit 4: In second latent
        bit 5: In circular halo
        bit 6: Saturated
        bit 7: In geometric diffraction spike
    
    flags_info位定义:
        bit 0: In PSF of bright star falling off coadd
        bit 1: In HyperLeda large galaxy
        bit 2: In big object (e.g., Magellanic cloud)
        bit 3: May contain centroid of very bright star
        bit 4: Potentially affected by saturation
        bit 5: May contain nebulosity
        bit 6: Will not be aggressively deblended
        bit 7: Must be sharp to be optimized
    """
    
    # flags_unwise必须为0（无任何亮星伪影）
    mask_flags = (
        (df['flags_unwise_w1'] == 0) & 
        (df['flags_unwise_w2'] == 0)
    )
    
    # flags_info必须为0
    mask_info = (
        (df['flags_info_w1'] == 0) &
        (df['flags_info_w2'] == 0)
    )

    final_mask = mask_flags & mask_info
    return df[final_mask].reset_index(drop=True)

def filter_by_mag(df):
    """
    - mag_w1_vg > 0 且 mag_w2_vg > 0（有效星等）
    - mag 不为 inf（无穷大表示无有效测量）
    - dflux_w1 > 0 且 dflux_w2 > 0（有效流量误差）
    """
    mask = np.ones(len(df), dtype=bool)
    
    # 星等 > 0 且不为 inf
    if 'mag_w1_vg' in df.columns and 'mag_w2_vg' in df.columns:
        mask &= (df['mag_w1_vg'] > 0) & (df['mag_w2_vg'] > 0)
        mask &= np.isfinite(df['mag_w1_vg']) & np.isfinite(df['mag_w2_vg'])
    
    # 流量误差 > 0
    if 'dflux_w1' in df.columns and 'dflux_w2' in df.columns:
        mask &= (df['dflux_w1'] > 0) & (df['dflux_w2'] > 0)
    
    return df[mask].reset_index(drop=True)

def process_single_file(args):
    """处理单个文件并立即保存"""
    path, file_idx, total_files, uid_lock, uid_counter, stats_lock, stats = args
    fid = os.path.basename(path).split('_')[1].split('.')[0]
    new_path = f'../clean/unwisedr1_clean_{fid}.fits'
    
    if os.path.exists(new_path):
        print(f'[{file_idx}/{total_files}] File {fid} already exists, skipping.')
        return
    
    # 读取数据
    df = read_unwise(path)
    raw_count = df.shape[0]
    
    # 应用清理过滤
    df = filter_by_mag(df)
    df = filter_by_flags(df)
    
    clean_count = df.shape[0]
    
    # 线程安全：分配uid
    with uid_lock:
        uid_start = uid_counter['value']
        df['uid'] = range(uid_start, uid_start + len(df))
        uid_counter['value'] += len(df)
    
    # 保存文件
    cu.savefile(df, new_path)
    print(f'[{file_idx}/{total_files}] Saved {fid}: {raw_count:,} -> {clean_count:,} ({100*clean_count/raw_count:.1f}%) uid: {uid_start:,} ~ {uid_start + len(df) - 1:,}')
    
    # 线程安全：更新统计
    with stats_lock:
        stats['raw'] += raw_count
        stats['clean'] += clean_count

def process():
    """使用多线程处理所有文件"""
    file_list = get_file_path()
    total_files = len(file_list)
    print(f'Total files to process: {total_files}')
    print(f'Using 10 threads for parallel processing...')
    print(f'Output directory: ../clean/')
    print()
    print('Cleaning criteria:')
    print('  - mag > 0 and finite (valid magnitude)')
    print('  - dflux > 0 (valid flux error)')
    print('  - flags_unwise = 0 (no bright star artifacts)')
    print('  - flags_info = 0 (no contamination)')
    print()
    
    # 创建输出目录
    os.makedirs('../clean', exist_ok=True)
    
    # 共享变量和锁
    uid_lock = threading.Lock()
    uid_counter = {'value': 0}
    stats_lock = threading.Lock()
    stats = {'raw': 0, 'clean': 0}
    
    # 准备参数
    file_args = [(path, idx+1, total_files, uid_lock, uid_counter, stats_lock, stats) 
                 for idx, path in enumerate(file_list)]
    
    # 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_file, arg) for arg in file_args]
        concurrent.futures.wait(futures)
    
    print()
    print('='*60)
    print('Processing complete!')
    print(f'Raw source count:   {stats["raw"]:,}')
    print(f'Clean source count: {stats["clean"]:,}')
    if stats['raw'] > 0:
        print(f'Retention rate: {(stats["clean"]/stats["raw"])*100:.2f}%')
    print('='*60)

# ===== CELL 009 [code] =====
process()

# ===== CELL 010 [code] =====
# make loc file
import cosmic.utils as cu
dir_path = '../clean'
df_loc = pd.DataFrame(columns=['uid', 'ra', 'dec'])
for i in range(72):
    path = os.path.join(dir_path, 'unwisedr1_clean_%02d.fits'%(i))
    if not os.path.exists(path):
        continue
    df = cu.readfile(path)
    sub_df = df[['uid', 'ra', 'dec']]
    df_loc = pd.concat([df_loc, sub_df], axis=0)
path = '../unwisedr1_clean_all_loc.fits'
cu.savefile(df_loc, path)


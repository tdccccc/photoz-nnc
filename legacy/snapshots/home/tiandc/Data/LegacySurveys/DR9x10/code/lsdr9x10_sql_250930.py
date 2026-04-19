# this code used for download DESI Legacy Surveys DR10
# for better download efficiency, please use TOPCAT TAP query

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from dl import authClient as ac, queryClient as qc, storeClient as sc
import os
import multiprocessing
from filelock import FileLock
import time

# Configuration constants
N_RA_SEGMENTS = 180
N_DEC_SEGMENTS = 30
MAX_ROWS_PER_REGION = 5000000
MAX_THREADS = 4
QUERY_TIMEOUT = 3600
DOWNLOAD_TIMEOUT = 86400
DEC_BOUNDARY = 32.375  # DESI南北天区分界线 (dec < 32.375为南天区DR10，>= 32.375为北天区DR9)

TOKEN = ac.login(
            user='tiandc1998',
            password='290923272Tdc.'
        )


def make_seg(output_dir, n_ra=N_RA_SEGMENTS, n_dec=N_DEC_SEGMENTS):
    """使用向量化操作生成天区分割"""
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


def load_region_list(path):
    """加载区域索引列表，消除重复代码"""
    if not os.path.exists(path):
        return []
    try:
        data = np.loadtxt(path, dtype=int).tolist()
        return [data] if not isinstance(data, list) else data
    except Exception:
        return []


def append_to_file_with_lock(path, idx):
    """使用文件锁安全地追加索引到文件"""
    lock_path = path + '.lock'
    with FileLock(lock_path, timeout=10):
        with open(path, 'a') as f:
            f.write(f'{idx}\n')


def download_cat(output_dir, idx, ra_l, ra_u, dec_l, dec_u):
    # 构建数据量检查查询
    count_query = f"""
        SELECT COUNT(*)
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND t.type <> 'PSF'
            AND t.ra >= {ra_l:.1f} AND t.ra < {ra_u:.1f}
            AND t.dec >= {dec_l:.1f} AND t.dec < {dec_u:.1f}
            """

    # 根据赤纬范围选择查询策略
    if dec_u <= DEC_BOUNDARY:
        # 纯南天区 - 只JOIN DR10 photo_z
        query = f"""
            SELECT
                t.ra, t.dec, t.glat, t.glon,

                p10.z_phot_mean AS z_phot_mean_grz,
                p10.z_phot_median AS z_phot_median_grz,
                p10.z_phot_std AS z_phot_std_grz,
                p10.z_spec AS z_spec,

                p10.z_phot_mean_i AS z_phot_mean_i,
                p10.z_phot_median_i AS z_phot_median_i,
                p10.z_phot_std_i AS z_phot_std_i,

                CASE WHEN p10.ls_id IS NOT NULL THEN 'DR10' ELSE 'None' END AS photo_z_source,

                t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
                t.dered_mag_w1, t.dered_mag_w2, t.dered_mag_w3, t.dered_mag_w4,
                t.g_r, t.r_z, t.z_w1, t.w1_w2, t.w2_w3, t.w3_w4,
                t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2, t.snr_w3, t.snr_w4,
                t.fracflux_g, t.fracflux_r, t.fracflux_i, t.fracflux_z,
                t.fracmasked_g, t.fracmasked_r, t.fracmasked_i, t.fracmasked_z,
                t.fracin_g, t.fracin_r, t.fracin_i, t.fracin_z,
                t.allmask_g, t.allmask_r, t.allmask_i, t.allmask_z
            FROM
                ls_dr10.tractor AS t
            LEFT JOIN
                ls_dr10.photo_z AS p10 ON t.ls_id = p10.ls_id
            WHERE
                t.brick_primary = 1
                AND t.type <> 'PSF'
                AND t.ra >= {ra_l:.1f} AND t.ra < {ra_u:.1f}
                AND t.dec >= {dec_l:.1f} AND t.dec < {dec_u:.1f}
        """
    elif dec_l >= DEC_BOUNDARY:
        # 纯北天区 - 只JOIN DR9 photo_z
        query = f"""
            SELECT
                t.ra, t.dec, t.glat, t.glon,

                p9.z_phot_mean AS z_phot_mean_grz,
                p9.z_phot_median AS z_phot_median_grz,
                p9.z_phot_std AS z_phot_std_grz,
                p9.z_spec AS z_spec,

                NULL AS z_phot_mean_i,
                NULL AS z_phot_median_i,
                NULL AS z_phot_std_i,

                CASE WHEN p9.ls_id IS NOT NULL THEN 'DR9' ELSE 'None' END AS photo_z_source,

                t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
                t.dered_mag_w1, t.dered_mag_w2, t.dered_mag_w3, t.dered_mag_w4,
                t.g_r, t.r_z, t.z_w1, t.w1_w2, t.w2_w3, t.w3_w4,
                t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2, t.snr_w3, t.snr_w4,
                t.fracflux_g, t.fracflux_r, t.fracflux_i, t.fracflux_z,
                t.fracmasked_g, t.fracmasked_r, t.fracmasked_i, t.fracmasked_z,
                t.fracin_g, t.fracin_r, t.fracin_i, t.fracin_z,
                t.allmask_g, t.allmask_r, t.allmask_i, t.allmask_z
            FROM
                ls_dr10.tractor AS t
            LEFT JOIN
                ls_dr9.photo_z AS p9 ON t.ls_id = p9.ls_id
            WHERE
                t.brick_primary = 1
                AND t.type <> 'PSF'
                AND t.ra >= {ra_l:.1f} AND t.ra < {ra_u:.1f}
                AND t.dec >= {dec_l:.1f} AND t.dec < {dec_u:.1f}
        """
    else:
        # 跨越分界线 - 使用双LEFT JOIN
        query = f"""
            SELECT
                t.ra, t.dec, t.glat, t.glon,

                COALESCE(p10.z_phot_mean, p9.z_phot_mean) AS z_phot_mean_grz,
                COALESCE(p10.z_phot_median, p9.z_phot_median) AS z_phot_median_grz,
                COALESCE(p10.z_phot_std, p9.z_phot_std) AS z_phot_std_grz,
                COALESCE(p10.z_spec, p9.z_spec) AS z_spec,

                p10.z_phot_mean_i AS z_phot_mean_i,
                p10.z_phot_median_i AS z_phot_median_i,
                p10.z_phot_std_i AS z_phot_std_i,

                CASE
                    WHEN p10.ls_id IS NOT NULL THEN 'DR10'
                    WHEN p9.ls_id IS NOT NULL THEN 'DR9'
                    ELSE 'None'
                END AS photo_z_source,

                t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
                t.dered_mag_w1, t.dered_mag_w2, t.dered_mag_w3, t.dered_mag_w4,
                t.g_r, t.r_z, t.z_w1, t.w1_w2, t.w2_w3, t.w3_w4,
                t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2, t.snr_w3, t.snr_w4,
                t.fracflux_g, t.fracflux_r, t.fracflux_i, t.fracflux_z,
                t.fracmasked_g, t.fracmasked_r, t.fracmasked_i, t.fracmasked_z,
                t.fracin_g, t.fracin_r, t.fracin_i, t.fracin_z,
                t.allmask_g, t.allmask_r, t.allmask_i, t.allmask_z
            FROM
                ls_dr10.tractor AS t
            LEFT JOIN
                ls_dr10.photo_z AS p10 ON t.ls_id = p10.ls_id
            LEFT JOIN
                ls_dr9.photo_z AS p9 ON t.ls_id = p9.ls_id
            WHERE
                t.brick_primary = 1
                AND t.type <> 'PSF'
                AND t.ra >= {ra_l:.1f} AND t.ra < {ra_u:.1f}
                AND t.dec >= {dec_l:.1f} AND t.dec < {dec_u:.1f}
        """

    # 定义文件路径
    empty_region_idx_path = os.path.join(output_dir, 'empty_region_idx.txt')
    failed_region_idx_path = os.path.join(output_dir, 'failed_region_idx.txt')
    large_region_idx_path = os.path.join(output_dir, 'large_region_idx.txt')

    # 加载区域列表
    empty_region_idx = load_region_list(empty_region_idx_path)
    large_region_idx = load_region_list(large_region_idx_path)

    # Define path for the download file
    path = os.path.join(output_dir, f'lsdr9x10_gal_download_{idx}.fits')

    # Proceed if the file does not exist and index is not in empty or large regions
    # Note: failed regions are allowed to retry
    if (not os.path.exists(path)) and (idx not in empty_region_idx) and (idx not in large_region_idx):
        print(f'{idx} starting download...')

        try:
            # First, check data volume
            print(f'{idx} checking data volume...')
            count_result = qc.query(TOKEN, sql=count_query, fmt='pandas', timeout=QUERY_TIMEOUT)

            if count_result is not None and len(count_result) > 0:
                row_count = count_result.iloc[0, 0]

                # If no data found, mark as empty region and skip
                if row_count == 0:
                    append_to_file_with_lock(empty_region_idx_path, idx)
                    print(f'{idx} marked as empty region (0 rows), skipping download.')
                    return

                # If more than MAX_ROWS_PER_REGION, save to large_region_idx.txt and skip
                if row_count > MAX_ROWS_PER_REGION:
                    append_to_file_with_lock(large_region_idx_path, idx)
                    print(f'{idx} marked as large region ({row_count} rows), skipping download.')
                    return

                print(f'{idx} data volume check passed ({row_count} rows), executing download query...')

                # Query and save directly as FITS format
                qc.query(TOKEN, sql=query, fmt='fits', out=path, timeout=DOWNLOAD_TIMEOUT)
                print(f'{idx} query completed, checking file...')

                # Check if the file was created and has content
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    file_size_mb = os.path.getsize(path) / (1024 * 1024)  # Convert bytes to MB
                    print(f'{idx} done. File size: {file_size_mb:.2f} MB')
                else:
                    # This shouldn't happen since we checked data volume beforehand
                    print(f'{idx} unexpected: file not created despite volume check, marking as failed...')
                    append_to_file_with_lock(failed_region_idx_path, idx)
                    if os.path.exists(path):
                        os.remove(path)
            else:
                # Count query failed, mark as failed region
                print(f'{idx} data volume check failed, marking as failed region...')
                append_to_file_with_lock(failed_region_idx_path, idx)
                return

        except KeyboardInterrupt:
            print(f'{idx} interrupted by user.')
            raise  # Re-raise to stop the program

        except Exception as e:
            print(f"{idx} error occurred: {type(e).__name__}: {e}")
            print(f'{idx} marking as failed region...')
            append_to_file_with_lock(failed_region_idx_path, idx)
            # Remove any partial/empty file
            if os.path.exists(path):
                os.remove(path)
            print(f'{idx} marked as failed.')
    else:
        if os.path.exists(path):
            print(f'{idx} already exists, skipping.')
        elif idx in empty_region_idx:
            print(f'{idx} in empty region list, skipping.')
        elif idx in large_region_idx:
            print(f'{idx} in large region list (>{MAX_ROWS_PER_REGION} rows), skipping.')
        else:
            print(f'{idx} skipped for unknown reason.')


def test_server_connectivity():
    """测试服务器连通性"""
    test_query = f"""
        SELECT COUNT(*)
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND t.type <> 'PSF'
            AND t.ra >= 0 AND t.ra < 0.1
            AND t.dec >= 0 AND t.dec < 0.1
    """
    try:
        print("Testing server connectivity...")
        result = qc.query(TOKEN, sql=test_query, fmt='pandas', timeout=30)
        if result is not None:
            print("✓ Server connection successful")
            return True
        else:
            print("✗ Server connection failed: No response")
            return False
    except Exception as e:
        print(f"✗ Server connection failed: {type(e).__name__}: {e}")
        return False


def main():
    # 测试服务器连通性
    while not test_server_connectivity():
        print("Waiting 5 minutes before retry...")
        time.sleep(300)  # 等待5分钟

    target_dir = '../download_lsdr10'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    make_seg(target_dir)

    # Retrieve segmentation info from the file
    df = pd.read_csv(f'{target_dir}/readme.txt').astype('int').set_index('fid')

    # 准备所有任务参数
    tasks = [
        (target_dir, idx, row['ra_l'], row['ra_u'], row['dec_l'], row['dec_u'])
        for idx, row in df.iterrows()
    ]

    # 使用进程池管理下载任务
    with multiprocessing.Pool(processes=MAX_THREADS) as pool:
        pool.starmap(download_cat, tasks)

    print("All downloads completed.")




if __name__=='__main__':

    main()

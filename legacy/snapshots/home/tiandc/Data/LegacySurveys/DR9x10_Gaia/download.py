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
N_RA_SEGMENTS = 10
N_DEC_SEGMENTS = 120
MAX_ROWS_PER_REGION = 5000000
MAX_THREADS = 8
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


def download_cat(output_dir, idx, ra_l, ra_u, dec_l, dec_u, sub_idx=None):
    # 构建数据量检查查询 (需要与主查询的WHERE条件一致)
    count_query = f"""
        SELECT COUNT(*)
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND t.type <> 'PSF'
            AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
            AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}
            AND t.dered_mag_g > 0 AND t.dered_mag_g < 30 AND t.snr_g > 0
            AND t.dered_mag_r > 0 AND t.dered_mag_r < 30 AND t.snr_r > 0
            AND t.dered_mag_z > 0 AND t.dered_mag_z < 30 AND t.snr_z > 0
            AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 30 AND t.snr_w1 > 0
            AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 30 AND t.snr_w2 > 0
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
                t.allmask_g, t.allmask_r, t.allmask_i, t.allmask_z,

                -- Gaia 星等及误差
                t.gaia_phot_g_mean_mag,
                t.gaia_phot_g_mean_flux_over_error,
                t.gaia_phot_bp_mean_mag,
                t.gaia_phot_bp_mean_flux_over_error,
                t.gaia_phot_rp_mean_mag,
                t.gaia_phot_rp_mean_flux_over_error,

                -- Gaia 自行及误差
                t.pmra,
                t.pmra_ivar,
                t.pmdec,
                t.pmdec_ivar,

                -- Gaia 视差及误差
                t.parallax,
                t.parallax_ivar
            FROM
                ls_dr10.tractor AS t
            LEFT JOIN
                ls_dr10.photo_z AS p10 ON t.ls_id = p10.ls_id
            WHERE
                t.brick_primary = 1
                AND t.type <> 'PSF'
                AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
                AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}

                -- 光学波段星等有效 (有检测且误差有效)
                AND t.dered_mag_g > 0 AND t.dered_mag_g < 30 AND t.snr_g > 0
                AND t.dered_mag_r > 0 AND t.dered_mag_r < 30 AND t.snr_r > 0
                AND t.dered_mag_z > 0 AND t.dered_mag_z < 30 AND t.snr_z > 0

                -- WISE 波段星等有效
                AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 30 AND t.snr_w1 > 0
                AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 30 AND t.snr_w2 > 0
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
                t.allmask_g, t.allmask_r, t.allmask_i, t.allmask_z,

                -- Gaia 星等及误差
                t.gaia_phot_g_mean_mag,
                t.gaia_phot_g_mean_flux_over_error,
                t.gaia_phot_bp_mean_mag,
                t.gaia_phot_bp_mean_flux_over_error,
                t.gaia_phot_rp_mean_mag,
                t.gaia_phot_rp_mean_flux_over_error,

                -- Gaia 自行及误差
                t.pmra,
                t.pmra_ivar,
                t.pmdec,
                t.pmdec_ivar,

                -- Gaia 视差及误差
                t.parallax,
                t.parallax_ivar
            FROM
                ls_dr10.tractor AS t
            LEFT JOIN
                ls_dr9.photo_z AS p9 ON t.ls_id = p9.ls_id
            WHERE
                t.brick_primary = 1
                AND t.type <> 'PSF'
                AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
                AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}

                -- 光学波段星等有效 (有检测且误差有效)
                AND t.dered_mag_g > 0 AND t.dered_mag_g < 30 AND t.snr_g > 0
                AND t.dered_mag_r > 0 AND t.dered_mag_r < 30 AND t.snr_r > 0
                AND t.dered_mag_z > 0 AND t.dered_mag_z < 30 AND t.snr_z > 0

                -- WISE 波段星等有效
                AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 30 AND t.snr_w1 > 0
                AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 30 AND t.snr_w2 > 0
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
                t.allmask_g, t.allmask_r, t.allmask_i, t.allmask_z,

                -- Gaia 星等及误差
                t.gaia_phot_g_mean_mag,
                t.gaia_phot_g_mean_flux_over_error,
                t.gaia_phot_bp_mean_mag,
                t.gaia_phot_bp_mean_flux_over_error,
                t.gaia_phot_rp_mean_mag,
                t.gaia_phot_rp_mean_flux_over_error,

                -- Gaia 自行及误差
                t.pmra,
                t.pmra_ivar,
                t.pmdec,
                t.pmdec_ivar,

                -- Gaia 视差及误差
                t.parallax,
                t.parallax_ivar
            FROM
                ls_dr10.tractor AS t
            LEFT JOIN
                ls_dr10.photo_z AS p10 ON t.ls_id = p10.ls_id
            LEFT JOIN
                ls_dr9.photo_z AS p9 ON t.ls_id = p9.ls_id
            WHERE
                t.brick_primary = 1
                AND t.type <> 'PSF'
                AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
                AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}

                -- 光学波段星等有效 (有检测且误差有效)
                AND t.dered_mag_g > 0 AND t.dered_mag_g < 30 AND t.snr_g > 0
                AND t.dered_mag_r > 0 AND t.dered_mag_r < 30 AND t.snr_r > 0
                AND t.dered_mag_z > 0 AND t.dered_mag_z < 30 AND t.snr_z > 0

                -- WISE 波段星等有效
                AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 30 AND t.snr_w1 > 0
                AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 30 AND t.snr_w2 > 0
        """

    # 定义文件路径
    empty_region_idx_path = os.path.join(output_dir, 'empty_region_idx.txt')
    failed_region_idx_path = os.path.join(output_dir, 'failed_region_idx.txt')

    # 加载区域列表
    empty_region_idx = load_region_list(empty_region_idx_path)

    # Define path for the download file
    if sub_idx is not None:
        path = os.path.join(output_dir, f'lsdr9x10_gal_download_{idx}_part{sub_idx}.fits')
        region_id = f'{idx}_part{sub_idx}'
    else:
        path = os.path.join(output_dir, f'lsdr9x10_gal_download_{idx}.fits')
        region_id = str(idx)

    # Proceed if the file does not exist and index is not in empty regions
    # Note: failed regions are allowed to retry, large regions will be split automatically
    if (not os.path.exists(path)) and (idx not in empty_region_idx):
        print(f'{region_id} starting download...')

        try:
            # First, check data volume
            print(f'{region_id} checking data volume...')
            count_result = qc.query(TOKEN, sql=count_query, fmt='pandas', timeout=QUERY_TIMEOUT)

            if count_result is not None and len(count_result) > 0:
                row_count = count_result.iloc[0, 0]

                # If no data found, mark as empty region and skip
                if row_count == 0:
                    append_to_file_with_lock(empty_region_idx_path, idx)
                    print(f'{region_id} marked as empty region (0 rows), skipping download.')
                    return

                # If more than MAX_ROWS_PER_REGION, split into sub-regions
                if row_count > MAX_ROWS_PER_REGION:
                    # 计算需要分割的份数
                    n_splits = int(np.ceil(row_count / MAX_ROWS_PER_REGION))
                    print(f'{region_id} is large region ({row_count} rows), splitting into {n_splits} parts by RA...')

                    # 按 RA 细分
                    ra_edges = np.linspace(ra_l, ra_u, n_splits + 1)
                    for i in range(n_splits):
                        sub_ra_l = ra_edges[i]
                        sub_ra_u = ra_edges[i + 1]
                        # 递归调用，使用子区域索引
                        if sub_idx is not None:
                            new_sub_idx = f'{sub_idx}_{i}'
                        else:
                            new_sub_idx = i
                        download_cat(output_dir, idx, sub_ra_l, sub_ra_u, dec_l, dec_u, sub_idx=new_sub_idx)
                    return

                print(f'{region_id} data volume check passed ({row_count} rows), executing download query...')

                # Query and save directly as FITS format
                qc.query(TOKEN, sql=query, fmt='fits', out=path, timeout=DOWNLOAD_TIMEOUT)
                print(f'{region_id} query completed, checking file...')

                # Check if the file was created and has content
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    file_size_mb = os.path.getsize(path) / (1024 * 1024)  # Convert bytes to MB
                    print(f'{region_id} done. File size: {file_size_mb:.2f} MB')
                else:
                    # This shouldn't happen since we checked data volume beforehand
                    print(f'{region_id} unexpected: file not created despite volume check, marking as failed...')
                    append_to_file_with_lock(failed_region_idx_path, region_id)
                    if os.path.exists(path):
                        os.remove(path)
            else:
                # Count query failed, mark as failed region
                print(f'{region_id} data volume check failed, marking as failed region...')
                append_to_file_with_lock(failed_region_idx_path, region_id)
                return

        except KeyboardInterrupt:
            print(f'{region_id} interrupted by user.')
            raise  # Re-raise to stop the program

        except Exception as e:
            print(f"{region_id} error occurred: {type(e).__name__}: {e}")
            print(f'{region_id} marking as failed region...')
            append_to_file_with_lock(failed_region_idx_path, region_id)
            # Remove any partial/empty file
            if os.path.exists(path):
                os.remove(path)
            print(f'{region_id} marked as failed.')
    else:
        if os.path.exists(path):
            print(f'{region_id} already exists, skipping.')
        elif idx in empty_region_idx:
            print(f'{region_id} in empty region list, skipping.')
        else:
            print(f'{region_id} skipped for unknown reason.')


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

    target_dir = './download'
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

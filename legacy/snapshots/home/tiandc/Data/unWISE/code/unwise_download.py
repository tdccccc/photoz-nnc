# unWISE DR1 数据下载 - 使用 NOIRLab DataLab
import warnings
warnings.simplefilter("ignore")

import os
import time
import random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_proxy():
    """设置代理"""
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'


def login_datalab(user, password):
    """登录 DataLab 并返回 TOKEN"""
    from dl import authClient as ac
    token = ac.login(user=user, password=password)
    print("DataLab 登录成功")
    return token


def test_datalab_connectivity(token):
    """测试服务器连通性"""
    from dl import queryClient as qc

    test_query = """
        SELECT COUNT(*)
        FROM unwise_dr1.object
        WHERE ra >= 0 AND ra < 0.1 AND dec >= 0 AND dec < 0.1
    """
    try:
        print("正在测试 DataLab 连通性...")
        result = qc.query(token, sql=test_query, fmt='pandas', timeout=60)
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


def count_region_data(token, ra_lo, ra_hi, dec_lo, dec_hi):
    """查询区域数据量"""
    from dl import queryClient as qc

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
    result = qc.query(token, sql=count_query, fmt='pandas', timeout=600)
    return result.iloc[0, 0]


def download_region(token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, count, sub_idx=None, max_retries=3):
    """下载单个区域的数据，支持自动重试"""
    from dl import queryClient as qc

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
            qc.query(token, sql=data_query, fmt='fits', out=output_file, timeout=86400)

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


def process_region(token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, max_rows=5000000):
    """处理单个区域：先查询数据量，决定是否细分"""

    # 先检查文件是否已存在（避免不必要的count查询）
    output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}.fits")
    if os.path.exists(output_file):
        return [f"[{idx:04d}] (已存在)"]

    try:
        # 查询数据量
        count = count_region_data(token, ra_lo, ra_hi, dec_lo, dec_hi)

        results = []

        # 判断是否需要细分
        if count == 0:
            results.append(f"[{idx:04d}] 空区域，跳过")
        elif count <= max_rows:
            # 直接下载
            result = download_region(token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, count)
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
                sub_count = count_region_data(token, sub_ra_lo, sub_ra_hi, dec_lo, dec_hi)
                result = download_region(token, idx, sub_ra_lo, sub_ra_hi, dec_lo, dec_hi, output_dir, sub_count, sub_idx=j)
                results.append(result)

        return results

    except Exception as e:
        return [f"[{idx:04d}] 处理出错: {str(e)}"]


def run_download(token, output_dir, n_ra_segments=180, n_dec_segments=60, max_rows=5000000, max_workers=10):
    """执行下载任务"""
    os.makedirs(output_dir, exist_ok=True)

    # 生成分割信息并保存
    seg_df = make_seg(output_dir, n_ra_segments, n_dec_segments)

    # 生成所有区域任务
    tasks = []
    for idx, row in seg_df.iterrows():
        tasks.append((int(row['fid']), row['ra_l'], row['ra_u'], row['dec_l'], row['dec_u']))

    print(f"开始使用 {max_workers} 个线程下载数据...")
    print(f"总共 {len(tasks)} 个区域 ({n_ra_segments} x {n_dec_segments})")
    print(f"每个文件最大行数: {max_rows:,}\n")

    # 并行下载
    all_results = []
    completed = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {}
        for task in tasks:
            idx, ra_lo, ra_hi, dec_lo, dec_hi = task
            future = executor.submit(process_region, token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, max_rows)
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
    return all_results


def main():
    """主函数"""
    # 配置参数
    MAX_ROWS = 5000000       # 每次查询最大行数
    N_RA_SEGMENTS = 180      # RA方向分割数
    N_DEC_SEGMENTS = 60      # DEC方向分割数
    MAX_WORKERS = 10         # 并行线程数
    OUTPUT_DIR = "../download"

    # 设置代理
    setup_proxy()

    # 登录 DataLab
    token = login_datalab(user='tiandc1998', password='290923272Tdc.')

    # 测试连通性
    if not test_datalab_connectivity(token):
        print("连接测试失败，退出")
        return

    # 执行下载
    run_download(
        token=token,
        output_dir=OUTPUT_DIR,
        n_ra_segments=N_RA_SEGMENTS,
        n_dec_segments=N_DEC_SEGMENTS,
        max_rows=MAX_ROWS,
        max_workers=MAX_WORKERS
    )


if __name__ == "__main__":
    main()

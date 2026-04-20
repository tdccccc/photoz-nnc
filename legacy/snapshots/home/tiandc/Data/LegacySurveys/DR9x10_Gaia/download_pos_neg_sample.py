#!/usr/bin/env python3
"""
下载星系分类器训练样本

正样本（延展源）: type != 'PSF' AND shape_r > 0.5 arcsec
负样本（点源）: type == 'PSF' OR (has_gaia AND PM_SNR >= 3)

使用方法:
    python download_pos_neg_sample.py --type positive --limit 5000000
    python download_pos_neg_sample.py --type negative --limit 5000000
    python download_pos_neg_sample.py --type both --limit 5000000
"""

import os
import argparse
import numpy as np
import pandas as pd
from dl import authClient as ac, queryClient as qc
from astropy.table import Table
import time

# 配置
QUERY_TIMEOUT = 3600
OUTPUT_DIR = './galaxyClf'

# 登录
TOKEN = ac.login('anonymous')


def get_positive_query(ra_l, ra_u, dec_l, dec_u, limit=None):
    """
    正样本查询：展源（星系）
    条件：type != 'PSF' AND shape_r > 0.5 arcsec
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            t.ls_id,
            t.ra, t.dec,
            t.type,
            t.shape_r,
            t.shape_e1, t.shape_e2,
            t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
            t.dered_mag_w1, t.dered_mag_w2,
            t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2,
            t.gaia_phot_g_mean_mag,
            t.gaia_phot_g_mean_flux_over_error,
            t.gaia_phot_bp_mean_mag,
            t.gaia_phot_bp_mean_flux_over_error,
            t.gaia_phot_rp_mean_mag,
            t.gaia_phot_rp_mean_flux_over_error,
            t.pmra, t.pmdec,
            t.pmra_ivar, t.pmdec_ivar,
            t.parallax
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND t.type != 'PSF'
            AND t.shape_r > 0.5
            AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
            AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}
            -- 星等有效
            AND t.dered_mag_g > 0 AND t.dered_mag_g < 23
            AND t.dered_mag_r > 0 AND t.dered_mag_r < 23
            AND t.dered_mag_z > 0 AND t.dered_mag_z < 23
            AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 23
            AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 23
            -- 质量标志: 流量占比高、掩蔽少、在图像内、无掩蔽位
            AND t.fracflux_g < 0.5 AND t.fracflux_r < 0.5 AND t.fracflux_z < 0.5
            AND t.fracmasked_g < 0.3 AND t.fracmasked_r < 0.3 AND t.fracmasked_z < 0.3
            AND t.fracin_g > 0.8 AND t.fracin_r > 0.8 AND t.fracin_z > 0.8
            AND t.allmask_g = 0 AND t.allmask_r = 0 AND t.allmask_z = 0
        {limit_clause}
    """
    return query


def get_negative_query(ra_l, ra_u, dec_l, dec_u, limit=None):
    """
    负样本查询：点源（恒星/QSO）
    条件：type == 'PSF' OR (has_gaia AND PM_SNR >= 3)
    PM_SNR = sqrt(pmra^2 + pmdec^2) / sqrt(1/pmra_ivar + 1/pmdec_ivar)
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            t.ls_id,
            t.ra, t.dec,
            t.type,
            t.shape_r,
            t.shape_e1, t.shape_e2,
            t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
            t.dered_mag_w1, t.dered_mag_w2,
            t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2,
            t.gaia_phot_g_mean_mag,
            t.gaia_phot_g_mean_flux_over_error,
            t.gaia_phot_bp_mean_mag,
            t.gaia_phot_bp_mean_flux_over_error,
            t.gaia_phot_rp_mean_mag,
            t.gaia_phot_rp_mean_flux_over_error,
            t.pmra, t.pmdec,
            t.pmra_ivar, t.pmdec_ivar,
            t.parallax
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND (
                t.type = 'PSF'
                OR (
                    t.gaia_phot_g_mean_mag > 0
                    AND t.pmra_ivar > 0 AND t.pmdec_ivar > 0
                    AND (t.pmra * t.pmra + t.pmdec * t.pmdec) >= 9.0 * (1.0/t.pmra_ivar + 1.0/t.pmdec_ivar)
                )
            )
            AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
            AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}
            AND t.dered_mag_g > 0 AND t.dered_mag_g < 23
            AND t.dered_mag_r > 0 AND t.dered_mag_r < 23
            AND t.dered_mag_z > 0 AND t.dered_mag_z < 23
            AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 23
            AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 23
        {limit_clause}
    """
    return query


def count_samples(query_func, n_regions=10):
    """估算样本总数"""
    print("估算样本总数...")

    # 在几个随机区域采样估算
    total_estimate = 0
    ra_size = 360 / n_regions
    dec_size = 180 / n_regions

    sample_regions = [
        (i * ra_size, (i + 1) * ra_size, -90 + j * dec_size, -90 + (j + 1) * dec_size)
        for i in range(n_regions) for j in range(n_regions)
    ]

    # 随机选几个区域
    np.random.seed(42)
    selected = np.random.choice(len(sample_regions), min(5, len(sample_regions)), replace=False)

    counts = []
    for idx in selected:
        ra_l, ra_u, dec_l, dec_u = sample_regions[idx]
        query = query_func(ra_l, ra_u, dec_l, dec_u)
        count_query = f"SELECT COUNT(*) FROM ({query}) AS subq"
        try:
            result = qc.query(TOKEN, sql=count_query, fmt='pandas', timeout=120)
            if result is not None and len(result) > 0:
                counts.append(result.iloc[0, 0])
        except:
            pass

    if counts:
        avg_per_region = np.mean(counts)
        total_estimate = int(avg_per_region * len(sample_regions))
        print(f"  采样区域平均: {avg_per_region:.0f}")
        print(f"  估算总数: ~{total_estimate:,}")

    return total_estimate


def download_samples(sample_type, total_limit, output_dir):
    """
    分区域下载样本

    sample_type: 'positive' 或 'negative'
    total_limit: 总样本数限制
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    if sample_type == 'positive':
        query_func = get_positive_query
        output_file = os.path.join(output_dir, 'LSDR10_extendSource.fits')
        desc = "正样本（延展源）"
    else:
        query_func = get_negative_query
        output_file = os.path.join(output_dir, 'LSDR10_pointSource.fits')
        desc = "负样本（点源）"

    print(f"\n{'='*60}")
    print(f"下载{desc}")
    print(f"目标数量: {total_limit:,}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}\n")

    # 检查是否已存在
    if os.path.exists(output_file):
        existing = Table.read(output_file)
        print(f"已存在文件，包含 {len(existing):,} 条记录")
        if len(existing) >= total_limit:
            print("已达到目标数量，跳过下载")
            return existing
        print(f"继续下载，还需 {total_limit - len(existing):,} 条")
        all_data = [existing.to_pandas()]
        current_count = len(existing)
    else:
        all_data = []
        current_count = 0

    # 分区域下载，不限制每区域数量
    n_ra = 20  
    n_dec = 60 

    ra_edges = np.linspace(0, 360, n_ra + 1)
    dec_edges = np.linspace(-90, 90, n_dec + 1)

    n_regions = n_ra * n_dec

    # 构建区域列表
    if sample_type == 'positive':
        # 正样本：只选高银纬区域 (|b| > 20°)，减少恒星污染
        # 北银冠和南银冠
        regions = []
        for i in range(n_ra):
            for j in range(n_dec):
                ra_c = (ra_edges[i] + ra_edges[i + 1]) / 2
                dec_c = (dec_edges[j] + dec_edges[j + 1]) / 2
                # 计算银纬 (近似: 银极 ra~192.85, dec~27.13)
                ra_rad = np.radians(ra_c)
                dec_rad = np.radians(dec_c)
                ra_ngp = np.radians(192.85)
                dec_ngp = np.radians(27.13)
                sin_b = (np.sin(dec_rad) * np.sin(dec_ngp)
                         + np.cos(dec_rad) * np.cos(dec_ngp) * np.cos(ra_rad - ra_ngp))
                glat = np.degrees(np.arcsin(sin_b))
                if abs(glat) > 20:
                    regions.append((i, j))
        print(f"正样本区域筛选: |b| > 20°, {len(regions)}/{n_regions} 个区域")
    else:
        # 负样本：全天
        regions = [(i, j) for i in range(n_ra) for j in range(n_dec)]
        print(f"负样本区域: 全天 {len(regions)} 个区域")

    # 随机打乱区域顺序
    np.random.seed(42)
    np.random.shuffle(regions)

    print(f"分区策略: {n_ra}x{n_dec}={len(regions)}个区域 (每区域30°x30°)\n")

    for region_idx, (i, j) in enumerate(regions):
        if current_count >= total_limit:
            print(f"\n已达到目标数量 {total_limit:,}，停止下载")
            break

        ra_l, ra_u = ra_edges[i], ra_edges[i + 1]
        dec_l, dec_u = dec_edges[j], dec_edges[j + 1]

        query = query_func(ra_l, ra_u, dec_l, dec_u)

        try:
            print(f"[{region_idx+1}/{len(regions)}] RA=[{ra_l:.0f},{ra_u:.0f}] Dec=[{dec_l:.0f},{dec_u:.0f}]", end=" ")

            result = qc.query(TOKEN, sql=query, fmt='pandas', timeout=QUERY_TIMEOUT)

            if result is not None and len(result) > 0:
                all_data.append(result)
                current_count += len(result)
                print(f"-> {len(result):,} 条 (累计: {current_count:,})")
            else:
                print("-> 0 条")

        except KeyboardInterrupt:
            print("\n用户中断，保存已下载的数据...")
            break
        except Exception as e:
            print(f"-> 错误: {e}")
            continue

        # 每50个区域保存一次
        if (region_idx + 1) % 50 == 0 and all_data:
            print(f"\n  中间保存... ", end="")
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=['ls_id'])
            table = Table.from_pandas(combined)
            table.write(output_file, format='fits', overwrite=True)
            print(f"已保存 {len(table):,} 条\n")

    # 最终合并和保存
    if all_data:
        print(f"\n合并数据...")
        combined = pd.concat(all_data, ignore_index=True)

        # 去重
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['ls_id'])
        print(f"  去重: {before_dedup:,} -> {len(combined):,}")

        # 如果超过限制，随机采样
        if len(combined) > total_limit:
            combined = combined.sample(n=total_limit, random_state=42)
            print(f"  随机采样: -> {len(combined):,}")

        # 保存
        table = Table.from_pandas(combined)
        table.write(output_file, format='fits', overwrite=True)

        print(f"\n{'='*60}")
        print(f"下载完成!")
        print(f"  样本数: {len(table):,}")
        print(f"  文件: {output_file}")
        print(f"  大小: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
        print(f"{'='*60}")

        return table
    else:
        print("没有下载到任何数据")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='下载星系分类器训练样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--type', '-t', choices=['positive', 'negative', 'both'],
                        default='both', help='样本类型 (默认: both)')
    parser.add_argument('--limit', '-n', type=int, default=5000000,
                        help='每类样本数量限制 (默认: 5000000)')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("星系分类器训练样本下载")
    print(f"{'#'*60}")
    print(f"\n正样本定义: type != 'PSF' AND shape_r > 0.5\"")
    print(f"负样本定义: type == 'PSF' OR (has_gaia AND PM_SNR >= 3)")
    print(f"每类限制: {args.limit:,}")
    print(f"输出目录: {args.output}\n")

    if args.type in ['positive', 'both']:
        download_samples('positive', args.limit, args.output)

    if args.type in ['negative', 'both']:
        download_samples('negative', args.limit, args.output)

    print("\n全部完成!")


if __name__ == '__main__':
    main()

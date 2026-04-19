# AUTO-CONVERTED FROM crossmatch.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [markdown] =====
# # LSDR9x10_Gaia `raw72` 与 PS1DR2 `raw72_dered` 交叉匹配，获取grizy波段数据
# 
# - 使用 `stilts_crossmatch` 进行交叉匹配
# - 匹配半径: 1 arcsec
# - 保留所有LS数据 (join="all1")，添加PS的grizy波段星等和误差

# ===== CELL 001 [code] =====
import os
import cosmic.utils as cu
from cosmic.utils import stilts_crossmatch

# 输入目录
ls_dir = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/raw72'
ps_dir = '/home/tiandc/Data/PanSTARRS/DR2All/raw72_dered'

# 输出目录
output_dir = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy'
os.makedirs(output_dir, exist_ok=True)

# PS1DR2 grizy波段: PSF, Kron, Ap 星等和误差
cols_ps_grizy = [
    'objID',
    # g波段
    'gPSFMag_dered', 'gPSFMagErr', 'gKronMag_dered', 'gKronMagErr', 'gApMag_dered', 'gApMagErr',
    # r波段
    'rPSFMag_dered', 'rPSFMagErr', 'rKronMag_dered', 'rKronMagErr', 'rApMag_dered', 'rApMagErr',
    # i波段
    'iPSFMag_dered', 'iPSFMagErr', 'iKronMag_dered', 'iKronMagErr', 'iApMag_dered', 'iApMagErr',
    # z波段
    'zPSFMag_dered', 'zPSFMagErr', 'zKronMag_dered', 'zKronMagErr', 'zApMag_dered', 'zApMagErr',
    # y波段
    'yPSFMag_dered', 'yPSFMagErr', 'yKronMag_dered', 'yKronMagErr', 'yApMag_dered', 'yApMagErr',
]

print(f"LS目录: {ls_dir}")
print(f"PS目录: {ps_dir}")
print(f"输出目录: {output_dir}")
print(f"将添加的PS列 ({len(cols_ps_grizy)}列): {cols_ps_grizy}")

# ===== CELL 002 [code] =====
import numpy as np
import tempfile
from astropy.io import fits
from astropy.table import Table

def crossmatch_and_merge(fid, ls_dir, ps_dir, output_dir, cols_ps_grizy, match_radius=1.0):
    """
    交叉匹配单个文件对，并将PS的grizy波段数据添加到LS数据中

    Parameters
    ----------
    fid : int
        文件编号 (0-71)
    ls_dir : str
        LS目录路径
    ps_dir : str
        PS目录路径
    output_dir : str
        输出目录路径
    cols_ps_grizy : list
        要添加的PS列名
    match_radius : float
        匹配半径 (arcsec)

    Returns
    -------
    dict
        包含统计信息的字典
    """
    # 文件路径
    ls_file = f'{ls_dir}/lsdr9x10_{fid:02d}.fits'
    ps_file = f'{ps_dir}/ps1dr2_{fid:02d}_dered.fits'
    output_file = f'{output_dir}/lsdr9x10_{fid:02d}_xPS1DR2grizy.fits'

    # 检查输出文件是否已存在
    if os.path.exists(output_file):
        print(f'[{fid:02d}/71] 已存在，跳过')
        return {'fid': fid, 'status': 'skipped'}

    # 创建临时文件用于stilts输出
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        tmp_match_file = tmp.name

    try:
        # 使用stilts进行交叉匹配
        # join="all1" 保留所有LS数据
        success = stilts_crossmatch(
            input1=ls_file,
            input2=ps_file,
            output_file=tmp_match_file,
            match_radius=match_radius,
            values1="ra dec",
            values2="raStack decStack",
            join="all1",  # 保留所有LS数据
            find="best",  # 只保留最佳匹配
            progress='none',
            verbose=False
        )

        if not success:
            print(f'[{fid:02d}/71] STILTS匹配失败')
            return {'fid': fid, 'status': 'failed', 'error': 'STILTS failed'}

        # 读取匹配结果
        t_matched = Table.read(tmp_match_file)

        # STILTS 可能将同名 uid 列重命名为 uid_1 / uid_2；保留LS原始 uid
        if 'uid_1' in t_matched.colnames and 'uid' not in t_matched.colnames:
            t_matched.rename_column('uid_1', 'uid')
        if 'uid_2' in t_matched.colnames:
            t_matched.remove_column('uid_2')

        n_total = len(t_matched)

        # 统计匹配数量 (有objID的即为匹配到的)
        n_matched = np.sum(~np.isnan(t_matched['objID']))

        # 只读取FITS header获取LS原始列名 (不加载数据)
        with fits.open(ls_file, memmap=True) as hdu:
            ls_original_cols = [col.name for col in hdu[1].columns]

        # 构建最终需要的列：原始LS列 + PS grizy列
        final_cols = ls_original_cols + cols_ps_grizy

        # 选择这些列 (处理可能缺失的列)
        available_cols = [c for c in final_cols if c in t_matched.colnames]
        t_final = t_matched[available_cols]

        # 保存结果
        t_final.write(output_file, overwrite=True)

        match_rate = n_matched / n_total * 100 if n_total > 0 else 0
        print(f'[{fid:02d}/71] Total: {n_total:,}, Matched: {n_matched:,} ({match_rate:.2f}%)')

        return {
            'fid': fid,
            'status': 'success',
            'n_total': n_total,
            'n_matched': n_matched,
            'match_rate': match_rate
        }

    except Exception as e:
        print(f'[{fid:02d}/71] 错误: {e}')
        return {'fid': fid, 'status': 'failed', 'error': str(e)}

    finally:
        # 清理临时文件
        if os.path.exists(tmp_match_file):
            os.remove(tmp_match_file)

# ===== CELL 003 [code] =====
# 处理所有72个文件
results = []

for fid in range(72):
    result = crossmatch_and_merge(
        fid=fid,
        ls_dir=ls_dir,
        ps_dir=ps_dir,
        output_dir=output_dir,
        cols_ps_grizy=cols_ps_grizy,
        match_radius=1.0
    )
    results.append(result)

# 统计结果
n_success = sum(1 for r in results if r['status'] == 'success')
n_skipped = sum(1 for r in results if r['status'] == 'skipped')
n_failed = sum(1 for r in results if r['status'] == 'failed')

print(f'\n{"="*50}')
print(f'完成: {n_success}/72 成功, {n_skipped}/72 跳过, {n_failed}/72 失败')

if n_failed > 0:
    print('失败的文件:')
    for r in results:
        if r['status'] == 'failed':
            print(f"  [{r['fid']:02d}] {r.get('error', 'Unknown error')}")


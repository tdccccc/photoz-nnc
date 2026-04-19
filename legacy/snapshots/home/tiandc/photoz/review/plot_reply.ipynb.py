# AUTO-CONVERTED FROM plot_reply.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [code] =====
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cosmic.utils as cu

plt.rcParams.update({
      # 字体
      'font.family': 'sans-serif',
      'font.sans-serif': ['Arial'],
      'mathtext.fontset': 'stix', # 'dejavusans'
      # 'font.weight': 'bold',
      # 字号
      'font.size': 18,
      'axes.labelsize': 18,
      'axes.titlesize': 14,
      'legend.fontsize': 15,
      'xtick.labelsize': 12,
      'ytick.labelsize': 12,
      # 线宽
      'axes.linewidth': 1.2,
      'xtick.major.width': 1.2,
      'ytick.major.width': 1.2,
      # 刻度朝内
      'xtick.direction': 'in',
      'ytick.direction': 'in',
      'xtick.top': True,
      'ytick.right': True,
      # 图例
      'legend.frameon': False,
      # 保存
      'savefig.dpi': 300,
      'savefig.bbox': 'tight',
  })


print('Current working directory:', os.getcwd())
FIGURE_DIR = '/home/tiandc/photoz/review/figure'
os.makedirs(FIGURE_DIR, exist_ok=True)
print('Figure directory:', FIGURE_DIR)

# Priority 1 - LSDR9x10 grizW1W2
P1_EXPERIMENT_DIR = "/home/tiandc/photoz/catalog/models/p1" 
P1_CALIBRATOR_DIR = f"{P1_EXPERIMENT_DIR}/TempScalingCalib"
P1_TRAINSET_RES_PATH = f"{P1_CALIBRATOR_DIR}/predictions_train.fits"
P1_VALSET_RES_PATH = f"{P1_CALIBRATOR_DIR}/predictions_val.fits"
P1_TESTSET_RES_PATH = f"{P1_CALIBRATOR_DIR}/predictions_test.fits"
P1_TRAINSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train.fits'
P1_VALSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_val.fits'
P1_TESTSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_test.fits'

# Priority 2 - LSDR9x10 grzW1W2 + PS1DR2 i
P2_EXPERIMENT_DIR = "/home/tiandc/photoz/catalog/models/p2"
P2_CALIBRATOR_DIR = f'{P2_EXPERIMENT_DIR}/TempScalingCalib'
P2_TRAINSET_RES_PATH = f'{P2_CALIBRATOR_DIR}/predictions_train.fits'
P2_VALSET_RES_PATH = f'{P2_CALIBRATOR_DIR}/predictions_val.fits'
P2_TESTSET_RES_PATH = f'{P2_CALIBRATOR_DIR}/predictions_test.fits'
P2_TESTSET_PATH = "/home/tiandc/photoz/lsdr9x10/data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_grz_test.fits"

# Priority 3 - LSDR9x10 grzW1W2
P3_EXPERIMENT_DIR = "/home/tiandc/photoz/catalog/models/p3"
P3_CALIBRATOR_DIR = f'{P3_EXPERIMENT_DIR}/TempScalingCalib'
P3_TRAINSET_RES_PATH = f'{P3_CALIBRATOR_DIR}/predictions_train.fits'
P3_VALSET_RES_PATH = f'{P3_CALIBRATOR_DIR}/predictions_val.fits'
P3_TESTSET_RES_PATH = f'{P3_CALIBRATOR_DIR}/predictions_test.fits'
P3_TESTSET_PATH = "/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_train.fits"

# Priority 4 - PanSTARRS x unWISE
P4_EXPERIMENT_DIR = "/home/tiandc/photoz/catalog/models/p4"
P4_CALIBRATOR_DIR = f'{P4_EXPERIMENT_DIR}/TempScalingCalib'
P4_TRAINSET_RES_PATH = f'{P4_CALIBRATOR_DIR}/predictions_train.fits'
P4_VALSET_RES_PATH = f'{P4_CALIBRATOR_DIR}/predictions_val.fits'
P4_TESTSET_RES_PATH = f'{P4_CALIBRATOR_DIR}/predictions_test.fits'
P4_TRAINSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEphot.fits'
P4_VALSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_val_xunWISEphot.fits'
P4_TESTSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_test_xunWISEphot.fits'

# Priority 5 - PanSTARRS
P5_EXPERIMENT_DIR = "/home/tiandc/photoz/catalog/models/p5"
P5_CALIBRATOR_DIR = f'{P5_EXPERIMENT_DIR}/TempScalingCalib'
P5_TRAINSET_RES_PATH = f'{P5_CALIBRATOR_DIR}/predictions_train.fits'
P5_VALSET_RES_PATH = f'{P5_CALIBRATOR_DIR}/predictions_val.fits'
P5_TESTSET_RES_PATH = f'{P5_CALIBRATOR_DIR}/predictions_test.fits'
P5_TRAINSET_PATH = '/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train.fits'
P5_VALSET_PATH = '/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2phot_clean_val.fits'
P5_TESTSET_PATH = '/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2phot_clean_test.fits'

# Priority 1 - DESI only
P1_DESIONLY_EXPERIMENT_DIR = "/home/tiandc/photoz/review/models/p1_DESIOnly"
P1_DESIONLY_CALIBRATOR_DIR = f'{P1_DESIONLY_EXPERIMENT_DIR}/TempScalingCalib'
P1_DESIONLY_TRAINSET_RES_PATH = f'{P1_DESIONLY_CALIBRATOR_DIR}/predictions_train.fits'
P1_DESIONLY_VALSET_RES_PATH = f'{P1_DESIONLY_CALIBRATOR_DIR}/predictions_val.fits'
P1_DESIONLY_TESTSET_RES_PATH = f'{P1_DESIONLY_CALIBRATOR_DIR}/predictions_test.fits'
P1_DESIONLY_TRAINSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_DESIonly_train.fits'
P1_DESIONLY_VALSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_DESIonly_val.fits'
P1_DESIONLY_TESTSET_PATH = P1_TESTSET_PATH

# Priority 1 - SDSS only
P1_SDSSONLY_EXPERIMENT_DIR = "/home/tiandc/photoz/review/models/p1_SDSSOnly"
P1_SDSSONLY_CALIBRATOR_DIR = f'{P1_SDSSONLY_EXPERIMENT_DIR}/TempScalingCalib'
P1_SDSSONLY_TRAINSET_RES_PATH = f'{P1_SDSSONLY_CALIBRATOR_DIR}/predictions_train.fits'
P1_SDSSONLY_VALSET_RES_PATH = f'{P1_SDSSONLY_CALIBRATOR_DIR}/predictions_val.fits'
P1_SDSSONLY_TESTSET_RES_PATH = f'{P1_SDSSONLY_CALIBRATOR_DIR}/predictions_test.fits'
P1_SDSSONLY_TRAINSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_SDSSonly_train.fits'
P1_SDSSONLY_VALSET_PATH = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_SDSSonly_val.fits'
P1_SDSSONLY_TESTSET_PATH = P1_TESTSET_PATH

# P4 - DESI only 
P4_DESIONLY_EXPERIMENT_DIR = "/home/tiandc/photoz/review/models/p4_DESIOnly"
P4_DESIONLY_CALIBRATOR_DIR = f'{P4_DESIONLY_EXPERIMENT_DIR}/TempScalingCalib'
P4_DESIONLY_TRAINSET_RES_PATH = f'{P4_DESIONLY_CALIBRATOR_DIR}/predictions_train.fits'
P4_DESIONLY_VALSET_RES_PATH = f'{P4_DESIONLY_CALIBRATOR_DIR}/predictions_val.fits'
P4_DESIONLY_TESTSET_RES_PATH = f'{P4_DESIONLY_CALIBRATOR_DIR}/predictions_test.fits'
P4_DESIONLY_TRAINSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEphot_onlyDESI_train.fits'
P4_DESIONLY_VALSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEphot_onlyDESI_val.fits'
P4_DESIONLY_TESTSET_PATH = P4_TESTSET_PATH

# P4 - SDSS only
P4_SDSSONLY_EXPERIMENT_DIR = "/home/tiandc/photoz/review/models/p4_SDSSOnly"
P4_SDSSONLY_CALIBRATOR_DIR = f'{P4_SDSSONLY_EXPERIMENT_DIR}/TempScalingCalib'
P4_SDSSONLY_TRAINSET_RES_PATH = f'{P4_SDSSONLY_CALIBRATOR_DIR}/predictions_train.fits'
P4_SDSSONLY_VALSET_RES_PATH = f'{P4_SDSSONLY_CALIBRATOR_DIR}/predictions_val.fits'
P4_SDSSONLY_TESTSET_RES_PATH = f'{P4_SDSSONLY_CALIBRATOR_DIR}/predictions_test.fits'
P4_SDSSONLY_TRAINSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEphot_onlySDSS_train.fits'
P4_SDSSONLY_VALSET_PATH = '/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEphot_onlySDSS_val.fits'
P4_SDSSONLY_TESTSET_PATH = P4_TESTSET_PATH

# ===== CELL 001 [markdown] =====
# # Table & Figure

# ===== CELL 002 [markdown] =====
# - Figure 1: redshift distribution

# ===== CELL 003 [code] =====
# 读取数据
path = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean.fits'
LSDR10_data_df = cu.readfile(path)
mask = (LSDR10_data_df['dered_mag_i'] > 0) & (LSDR10_data_df['dered_mag_i'] < 30)
mask &= (LSDR10_data_df['z'] > 0.) & (LSDR10_data_df['z'] <= 2.)
LSDR10_data_df = LSDR10_data_df[mask]

# PS1DR2
path_raw = '/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2loc.fits'
df_raw = cu.readfile(path_raw)

path = '/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2phot_clean.fits'
PS1DR2_data_df = cu.readfile(path)

cols = ['DESI_DESI_TARGET', 'DESI_BGS_TARGET', 'DESI_PRIORITY']
PS1DR2_data_df = PS1DR2_data_df.merge(df_raw[['objID'] + cols], on='objID', how='left')

mask = (PS1DR2_data_df['z'] > 0.) & (PS1DR2_data_df['z'] <= 2.)
PS1DR2_data_df = PS1DR2_data_df[mask]

# 统计各子样本数量
for name, df in [('LSDR10', LSDR10_data_df), ('PS1DR2', PS1DR2_data_df)]:
    n_all = len(df)
    n_sdss = (df['SDSS_specobjid'].fillna(0) > 0).sum()
    desi_mask = df['DESI_TARGETID'].fillna(0) > 0
    n_desi = desi_mask.sum()
    desi_sub = df[desi_mask]
    n_bgs = ((desi_sub['DESI_DESI_TARGET'] & 2**60) != 0).sum()
    n_lrg = ((desi_sub['DESI_DESI_TARGET'] & 2**0) != 0).sum()
    n_elg = ((desi_sub['DESI_DESI_TARGET'] & 2**1) != 0).sum()
    print(f"{name}: All={n_all:,}  SDSS={n_sdss:,}  DESI={n_desi:,}  (BGS={n_bgs:,}  LRG={n_lrg:,}  ELG={n_elg:,})")

# ===== CELL 004 [code] =====
bins = np.linspace(0.0, 2.0, 101)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, df, title in [(ax1, LSDR10_data_df, 'LSDR10'), 
                       (ax2, PS1DR2_data_df, 'PS1DR2')]:
    # Total
    hist_all = np.histogram(df['z'].values, bins=bins)[0]
    
    # SDSS
    sdss_mask = df['SDSS_specobjid'].fillna(0) > 0
    hist_sdss = np.histogram(df.loc[sdss_mask, 'z'].values, bins=bins)[0]
    
    # DESI subtypes
    desi_mask = df['DESI_TARGETID'].fillna(0) > 0
    desi_sub = df[desi_mask]
    
    bgs_mask = (desi_sub['DESI_DESI_TARGET'] & 2**60) != 0
    hist_bgs = np.histogram(desi_sub.loc[bgs_mask, 'z'].values, bins=bins)[0]
    
    lrg_mask = (desi_sub['DESI_DESI_TARGET'] & 2**0) != 0
    hist_lrg = np.histogram(desi_sub.loc[lrg_mask, 'z'].values, bins=bins)[0]
    
    elg_mask = (desi_sub['DESI_DESI_TARGET'] & 2**1) != 0
    hist_elg = np.histogram(desi_sub.loc[elg_mask, 'z'].values, bins=bins)[0]
    
    n_sdss = sdss_mask.sum()
    n_bgs = bgs_mask.sum()
    n_lrg = lrg_mask.sum()
    n_elg = elg_mask.sum()
    
    ax.plot(bin_centers, hist_all, 'k-', linewidth=1.5, label=f'All')
    ax.plot(bin_centers, hist_sdss, color='blue', linestyle='--', linewidth=1.5, 
            label=f'SDSS')
    ax.plot(bin_centers, hist_bgs, color='green', linestyle='-', linewidth=1.5, 
            label=f'DESI BGS')
    ax.plot(bin_centers, hist_lrg, color='red', linestyle='-.', linewidth=1.5, 
            label=f'DESI LRG')
    ax.plot(bin_centers, hist_elg, color='purple', linestyle=':', linewidth=3.0, 
            label=f'DESI ELG')
    
    ax.set_xlabel(r'$z_{\mathrm{spec}}$')
    ax.set_ylabel('Count')
    
    if title == 'PS1DR2':
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    else:
        ax.set_xlim(0.0, 1.8)
        ax.set_xticks(np.arange(0.0, 2.0, 0.2))
    
    ax.set_ylim(10, hist_all.max() * 1.15)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.legend(loc='upper right')
    
    ax.text(0.96, 0.04, title, transform=ax.transAxes, fontsize=20,
            ha='right', va='bottom')

plt.tight_layout()

path = f"{FIGURE_DIR}/figure1.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 005 [markdown] =====
# - Figure 2: Color space comparison

# ===== CELL 006 [code] =====
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter

# ---------- 数据准备 ----------
# 总光谱样本
spec_path = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean.fits'
spec = cu.readfile(spec_path)
mask = (spec['dered_mag_i'] > 0) & (spec['dered_mag_i'] < 30)
mask &= (spec['z'] > 0.) & (spec['z'] <= 2.)
spec = spec[mask]

spec['g-r'] = spec['dered_mag_g'] - spec['dered_mag_r']
spec['r-z'] = spec['dered_mag_r'] - spec['dered_mag_z']

sdss_mask = spec['SDSS_specobjid'].fillna(0) > 0
desi_mask = spec['DESI_TARGETID'].fillna(0) > 0
sdss_df = spec[sdss_mask]
desi_df = spec[desi_mask]
BGS_df = desi_df[(desi_df['DESI_DESI_TARGET'] & 2**60) != 0]
LRG_df = desi_df[(desi_df['DESI_DESI_TARGET'] & 2**0) != 0]
ELG_df = desi_df[(desi_df['DESI_DESI_TARGET'] & 2**1) != 0]

# 测光样本
phot_path = './p4-9/lsdr9x10_gal_clean_all_simpleClean_sample10M.fits'
phot = cu.readfile(phot_path)

phot['g-r'] = phot['dered_mag_g'] - phot['dered_mag_r']
phot['r-z'] = phot['dered_mag_r'] - phot['dered_mag_z']

print(f"Spec total: {len(spec):,}, Phot sample: {len(phot):,}")
print(f"SDSS: {len(sdss_df):,}, BGS: {len(BGS_df):,}, LRG: {len(LRG_df):,}, ELG: {len(ELG_df):,}")

# ---------- 绘图函数 ----------
def plot_smooth_density(ax, x_data, y_data, x_range, y_range, 
                        cmap_name='Blues', sigma=1.5, alpha_scale=1.0):
    if len(x_data) < 10: return None
    bins = 300
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins,
                                        range=[x_range, y_range])
    H_log = np.log10(H.T + 1)
    H_smooth = gaussian_filter(H_log, sigma=sigma)
    H_masked = np.ma.masked_where(H_smooth < 0.001, H_smooth)

    cmap_base = plt.get_cmap(cmap_name)
    my_cmap_data = cmap_base(np.arange(cmap_base.N))
    my_cmap_data[:,-1] = np.linspace(0., 1, cmap_base.N) * alpha_scale
    my_cmap = mcolors.ListedColormap(my_cmap_data)
    my_cmap.set_bad(color='none')

    im = ax.imshow(H_masked,
                    extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                    origin='lower', cmap=my_cmap, aspect='auto',
                    interpolation='nearest')
    return im

def plot_contour(ax, x_data, y_data, x_range, y_range, 
                 color='black', levels_pct=[30, 60, 80, 95], nbins=200, sigma=1.5):
    """画高斯平滑后的等高线"""
    mask = (np.isfinite(x_data) & np.isfinite(y_data) 
            & (x_data >= x_range[0]) & (x_data <= x_range[1])
            & (y_data >= y_range[0]) & (y_data <= y_range[1]))
    x, y = x_data[mask], y_data[mask]
    if len(x) < 100: return
    
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[x_range, y_range])
    H_smooth = gaussian_filter(H.T, sigma=sigma)
    
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    
    lvls = np.percentile(H_smooth[H_smooth > 0], levels_pct)
    ax.contour(xc, yc, H_smooth, levels=lvls, colors=color, 
               linewidths=1.0, alpha=0.8)


# ---------- 3×2 布局 ----------
fig, axes = plt.subplots(3, 2, figsize=(12, 13.5),
                         gridspec_kw={'height_ratios': [1, 1, 1]})

# === 坐标范围 ===
z_xlims = [0, 1.8]     # 上面两行：z_spec
z_ylims = [-0.5, 3]
mag_xlims = [15, 23.5]  # 下面一行：r mag
mag_ylims = [-0.5, 3]

color_col_1 = 'g-r'
color_label_1 = r'$g - r$'
color_col_2 = 'r-z'
color_label_2 = r'$r - z_{\mathrm{m}}$'

# === Row 0: SDSS color-redshift ===
ax = axes[0, 0]
plot_smooth_density(ax, sdss_df['z'].values, sdss_df[color_col_1].values, 
                    z_xlims, z_ylims, cmap_name='Blues')
ax.set_ylabel(color_label_1)
ax.text(0.96, 0.95, 'SDSS', transform=ax.transAxes, fontsize=20,
        ha='right', va='top')

ax = axes[0, 1]
plot_smooth_density(ax, sdss_df['z'].values, sdss_df[color_col_2].values, 
                    z_xlims, z_ylims, cmap_name='Blues')
ax.set_ylabel(color_label_2)
ax.text(0.96, 0.95, 'SDSS', transform=ax.transAxes, fontsize=20,
        ha='right', va='top')

# === Row 1: DESI color-redshift ===
type_configs = [
    {'data': BGS_df, 'label': 'BGS', 'cmap': 'Greens', 'color': 'green'},
    {'data': LRG_df, 'label': 'LRG', 'cmap': 'Reds',   'color': 'red'},
    {'data': ELG_df, 'label': 'ELG', 'cmap': 'Purples','color': 'purple'}
]

ax = axes[1, 0]
for cfg in type_configs:
    plot_smooth_density(ax, cfg['data']['z'].values, cfg['data'][color_col_1].values, 
                        z_xlims, z_ylims, cmap_name=cfg['cmap'], alpha_scale=1)
ax.set_ylabel(color_label_1)
ax.text(0.96, 0.95, 'DESI', transform=ax.transAxes, fontsize=20,
        ha='right', va='top')

ax = axes[1, 1]
for cfg in type_configs:
    plot_smooth_density(ax, cfg['data']['z'].values, cfg['data'][color_col_2].values, 
                        z_xlims, z_ylims, cmap_name=cfg['cmap'], alpha_scale=1)
ax.set_ylabel(color_label_2)
ax.text(0.96, 0.95, 'DESI', transform=ax.transAxes, fontsize=20,
        ha='right', va='top')

# DESI legend
patches = [mpatches.Patch(color=cfg['color'], label=cfg['label']) for cfg in type_configs]
axes[1, 1].legend(handles=patches, loc='upper left', fontsize=16,
                  frameon=False, facecolor='white', framealpha=0.9)

# === Row 2: Color-Magnitude (spec density + phot contour) ===
# 左图：r mag vs g-r
ax = axes[2, 0]
# 光谱样本：密度背景（用全部光谱样本，橙色系）
plot_smooth_density(ax, spec['dered_mag_r'].values, spec[color_col_1].values,
                    mag_xlims, mag_ylims, cmap_name='Oranges', sigma=1.5)
# 测光样本：等高线叠加
plot_contour(ax, phot['dered_mag_r'].values, phot[color_col_1].values,
             mag_xlims, mag_ylims, color='#1565C0', 
             levels_pct=[20, 50, 80, 95], nbins=200, sigma=2.0)
ax.set_xlabel(r'$r$')
ax.set_ylabel(color_label_1)
ax.set_xlim(mag_xlims)
ax.set_ylim(mag_ylims)

# 右图：r mag vs r-z
ax = axes[2, 1]
plot_smooth_density(ax, spec['dered_mag_r'].values, spec[color_col_2].values,
                    mag_xlims, mag_ylims, cmap_name='Oranges', sigma=1.5)
plot_contour(ax, phot['dered_mag_r'].values, phot[color_col_2].values,
             mag_xlims, mag_ylims, color='#1565C0',
             levels_pct=[20, 50, 80, 95], nbins=200, sigma=2.0)
ax.set_xlabel(r'$r$')
ax.set_ylabel(color_label_2)
ax.set_xlim(mag_xlims)
ax.set_ylim(mag_ylims)

# Row 2 legend
from matplotlib.lines import Line2D
legend_row2 = [
    mpatches.Patch(color='#FB8C00', alpha=0.7, label='Spectroscopic'),
    Line2D([0], [0], color='#1565C0', lw=1.5, label='Photometric'),
]
axes[2, 1].legend(handles=legend_row2, loc='upper left', fontsize=14,
                  frameon=False, facecolor='white', framealpha=0.9)

# === 统一刻度设置 ===
# Row 0, 1: z_spec x轴
z_ticks = np.arange(0, 2., 0.2)
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        ax.set_xlim(z_xlims)
        ax.set_ylim(z_ylims)
        ax.set_xticks(z_ticks)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i != 2:
            ax.set_xticklabels([f'{x:.1f}' for x in z_ticks])
            ax.set_xlabel(r'$z_{\mathrm{spec}}$')
        else:
            ax.set_xticklabels([])

# Row 2: r mag x轴（已在上面设置）
for j in range(2):
    axes[2, j].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# 保存
path = f"{FIGURE_DIR}/figure2.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 007 [markdown] =====
# - Figure 3: overall Performance: LSDR10 + PS1DR2 + PS1DR2+unWISE

# ===== CELL 008 [code] =====
# Overall Performance: LSDR10 + PS1DR2 + PS1DR2+unWISE (Test set only)
# Three surveys in one figure for paper

# Load data
# LSDR10
p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_labels = p1_test_res_df['label'].values
p1_preds = p1_test_res_df['pred'].values
p1_metrics = cu.evaluate_redshift_quality(p1_labels, p1_preds)

# PS1DR2 optical only
p5_test_res_df = cu.readfile(P5_TESTSET_RES_PATH)
p5_labels = p5_test_res_df['label'].values
p5_preds = p5_test_res_df['pred'].values
p5_metrics = cu.evaluate_redshift_quality(p5_labels, p5_preds)

# PS1DR2 + unWISE
p4_test_res_df = cu.readfile(P4_TESTSET_RES_PATH)
p4_labels = p4_test_res_df['label'].values
p4_preds = p4_test_res_df['pred'].values
p4_metrics = cu.evaluate_redshift_quality(p4_labels, p4_preds)

# Create figure (single row)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

label_font = 20

# Different axis ranges for each survey
datasets = [
    (p1_labels, p1_preds, 'LSDR10', p1_metrics, 1.8),
    (p5_labels, p5_preds, 'PS1DR2', p5_metrics, 1.2),
    (p4_labels, p4_preds, 'PS1DR2 + unWISE', p4_metrics, 1.2)
]

def format_sci_text(x):
    """Format number in LaTeX scientific notation for plot text"""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coef = x / (10 ** exp)
    return fr"{coef:.2f} \times 10^{{{exp}}}"

hb_list = []

for col, (labels, preds, title, metrics, z_max) in enumerate(datasets):
    ax = axes[col]
    
    # Set ticks based on z_max
    ticks = np.arange(0, z_max + 0.1, 0.2)
    
    # Main plot (hexbin)
    hb = ax.hexbin(labels, preds, gridsize=100, bins='log', cmap='Grays', mincnt=1)
    hb_list.append(hb)
    
    # 1:1 line
    ax.plot([0, z_max], [0, z_max], 'r--', lw=1.5)

    # Outlier boundaries (|Δz_norm| = 0.15)
    z_line = np.linspace(0., z_max, 100)
    upper_outlier = 1.15 * z_line + 0.15
    lower_outlier = 0.85 * z_line - 0.15
    ax.plot(z_line, upper_outlier, 'r:', lw=1)
    ax.plot(z_line, lower_outlier, 'r:', lw=1)

    ax.set_xlabel(r'$z_\mathrm{spec}$', fontsize=label_font)
    if col == 0:
        ax.set_ylabel(r'$z_\mathrm{phot}$', fontsize=label_font)
    
    ax.set_xlim(0, z_max)
    ax.set_ylim(0, z_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{x:.1f}' for x in ticks])
    ax.set_yticks(ticks[1:])
    ax.set_yticklabels([f'{x:.1f}' for x in ticks[1:]])
    ax.set_aspect('equal')
    
    # Title
    ax.text(0.96, 0.04, title, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=18)

    # Metrics text
    text = (
        r"$\mathrm{Bias}$ = $" + format_sci_text(metrics['mean_bias']) + "$\n" +
        r"$\sigma$ = " + f"{metrics['std_dev']:.4f}" + "\n" +
        fr"$\sigma_{{\mathrm{{NMAD}}}}$ = " + f"{metrics['mad']:.4f}" + "\n" +
        r"$\eta$ = " + f"{metrics['outlier_fraction']*100:.2f}" + r"%"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=16)

plt.tight_layout()

# Add colorbars after tight_layout
for col, ax in enumerate(axes):
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.005, pos.y0, 0.008, pos.height])
    fig.colorbar(hb_list[col], cax=cbar_ax)

path = f"{FIGURE_DIR}/figure3.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 009 [markdown] =====
# - Table 1: Comprehensive Performance Metrics Table

# ===== CELL 010 [code] =====
# Comprehensive Performance Metrics Table
# All configurations in one table for the paper

from cosmic.panstarrs_dr2.dataProcess import get_mstar_params, calculate_mstar

def format_sci(x):
    """Format number in LaTeX scientific notation"""
    exp = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    coef = x / (10 ** exp)
    sign = "-" if coef < 0 else ""
    return f"${sign}{abs(coef):.2f} \\times 10^{{{exp}}}$"

# Load all data
p1_test_df = cu.readfile(P1_TESTSET_PATH)
p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_test_df['z_pred'] = p1_test_res_df['pred'].values

# Calculate stellar mass for mass bins
params = get_mstar_params()
p1_test_df['mstar'] = calculate_mstar(p1_test_df['z'].values, 
                                    p1_test_df['dered_mag_r'].values, 
                                    p1_test_df['dered_mag_z'].values, params)

# Helper function to get metrics
def get_metrics(labels, preds):
    return cu.evaluate_redshift_quality(labels, preds)

# Collect all metrics: (section, config, metrics, bold)
results = []

# ===== LSDR10 Section =====
# 1. LSDR10 Overall (Test set)
p1_metrics = get_metrics(p1_test_res_df['label'].values, p1_test_res_df['pred'].values)
results.append(('LSDR10', 'Full (testset)', p1_metrics, True))

# 2. LSDR10 by magnitude (2 bins: <21 and >=21)
mag_bins = [(None, 21., '$z_{\\mathrm{m}} < 21$'), (21., None, '$z_{\\mathrm{m}} \\geq 21$')]
for mag_min, mag_max, label in mag_bins:
    if mag_min is None:
        mask = p1_test_df['dered_mag_z'] < mag_max
    elif mag_max is None:
        mask = p1_test_df['dered_mag_z'] >= mag_min
    else:
        mask = (p1_test_df['dered_mag_z'] >= mag_min) & (p1_test_df['dered_mag_z'] < mag_max)
    metrics = get_metrics(p1_test_df.loc[mask, 'z'].values, p1_test_df.loc[mask, 'z_pred'].values)
    results.append(('LSDR10', label, metrics, False))

# 3. LSDR10 by stellar mass (3 bins)
mstar_bins = [(11., None, '$\\log M_\\star \\geq 11$'), 
              (10., 11., '$10 \\leq \\log M_\\star < 11$'),
              (None, 10., '$\\log M_\\star < 10$')]
for mstar_min, mstar_max, label in mstar_bins:
    if mstar_min is None:
        mask = p1_test_df['mstar'] < mstar_max
    elif mstar_max is None:
        mask = p1_test_df['mstar'] >= mstar_min
    else:
        mask = (p1_test_df['mstar'] >= mstar_min) & (p1_test_df['mstar'] < mstar_max)
    metrics = get_metrics(p1_test_df.loc[mask, 'z'].values, p1_test_df.loc[mask, 'z_pred'].values)
    results.append(('LSDR10', label, metrics, False))

# 4. LSDR10 by galaxy type (BGS/LRG/ELG)
BGS_mask = (p1_test_df['DESI_DESI_TARGET'] & 2**60) != 0
LRG_mask = (p1_test_df['DESI_DESI_TARGET'] & 2**0) != 0
ELG_mask = (p1_test_df['DESI_DESI_TARGET'] & 2**1) != 0
for mask, label in [(BGS_mask, 'BGS'), (LRG_mask, 'LRG'), (ELG_mask, 'ELG')]:
    metrics = get_metrics(p1_test_df.loc[mask, 'z'].values, p1_test_df.loc[mask, 'z_pred'].values)
    results.append(('LSDR10', label, metrics, False))

# 5. Priority 2 - LSDR10 grzW1W2 + PS1DR2 i
p2_test_res_df = cu.readfile(P2_TESTSET_RES_PATH)
p2_metrics = get_metrics(p2_test_res_df['label'].values, p2_test_res_df['pred'].values)
results.append(('LSDR10', '$g,r,z_{\mathrm{m}},W1,W2$ + PS1DR2 $i$', p2_metrics, False))

# 6. Priority 3 - LSDR10 grzW1W2
p3_test_res_df = cu.readfile(P3_TESTSET_RES_PATH)
p3_metrics = get_metrics(p3_test_res_df['label'].values, p3_test_res_df['pred'].values)
results.append(('LSDR10', '$g,r,z_{\mathrm{m}},W1,W2$ (no $i$)', p3_metrics, False))

# 7. LSDR10 SDSS-only and DESI-only training
p1_SDSS_test_res_df = cu.readfile(P1_SDSSONLY_TESTSET_RES_PATH)
p1_DESI_test_res_df = cu.readfile(P1_DESIONLY_TESTSET_RES_PATH)
p1_SDSS_metrics = get_metrics(p1_SDSS_test_res_df['label'].values, p1_SDSS_test_res_df['pred'].values)
p1_DESI_metrics = get_metrics(p1_DESI_test_res_df['label'].values, p1_DESI_test_res_df['pred'].values)
results.append(('LSDR10', 'SDSS-only train', p1_SDSS_metrics, False))
results.append(('LSDR10', 'DESI-only train', p1_DESI_metrics, False))

# ===== ML Methods Section =====
# 11. ML Methods comparison (LSDR10)
ML_METHODS = {
    'RF': '/home/tiandc/photoz/lsdr9x10/photoz_rf/grizW1W2/Mag&MagErr',
    'XGBoost': '/home/tiandc/photoz/lsdr9x10/photoz_xgb/grizW1W2/Mag&MagErr',
    'ANN': '/home/tiandc/photoz/lsdr9x10/photoz_ann/grizW1W2/Mag&MagErr',
}
for method, path in ML_METHODS.items():
    try:
        ml_df = cu.readfile(f"{path}/predictions_test.fits")
        if method in ['RF', 'XGBoost']:
            metrics = get_metrics(ml_df['z'].values, ml_df['z_pred'].values)
        else:
            metrics = get_metrics(ml_df['label'].values, ml_df['pred'].values)
        results.append(('Other ML Methods on LSDR10', method, metrics, False))
    except Exception as e:
        print(f"Error loading {method}: {e}")

# ===== PS1DR2 Section =====
# 8. PS1DR2 optical only
p5_test_res_df = cu.readfile(P5_TESTSET_RES_PATH)
p5_metrics = get_metrics(p5_test_res_df['label'].values, p5_test_res_df['pred'].values)
results.append(('PS1DR2', 'Optical-only', p5_metrics, False))

# 9. PS1DR2 + unWISE
p4_test_res_df = cu.readfile(P4_TESTSET_RES_PATH)
p4_metrics = get_metrics(p4_test_res_df['label'].values, p4_test_res_df['pred'].values)
results.append(('PS1DR2', 'Optical + unWISE', p4_metrics, True))

# 10. PS1DR2+unWISE SDSS-only and DESI-only training
p4_SDSS_test_res_df = cu.readfile(P4_SDSSONLY_TESTSET_RES_PATH)
p4_DESI_test_res_df = cu.readfile(P4_DESIONLY_TESTSET_RES_PATH)
p4_SDSS_metrics = get_metrics(p4_SDSS_test_res_df['label'].values, p4_SDSS_test_res_df['pred'].values)
p4_DESI_metrics = get_metrics(p4_DESI_test_res_df['label'].values, p4_DESI_test_res_df['pred'].values)
results.append(('PS1DR2', 'SDSS-only train', p4_SDSS_metrics, False))
results.append(('PS1DR2', 'DESI-only train', p4_DESI_metrics, False))

# Output LaTeX table
print(r"\begin{deluxetable*}{lcccc}")
print(r"\tablecaption{Comprehensive Photometric Redshift Performance Summary \label{tab:table1}}")
print(r"\tablehead{")
print(r"\colhead{Configuration} & \colhead{Bias} & \colhead{$\sigma$} & \colhead{$\sigma_{\mathrm{NMAD}}$} & \colhead{$\eta$ (\%)}")
print(r"}")
print(r"\startdata")

current_section = None
for section, config, metrics, bold in results:
    # Print section header when section changes
    if section != current_section:
        if current_section is not None:
            print(r"\hline")
        print(f"\\multicolumn{{5}}{{c}}{{{section}}} \\\\")
        print(r"\hline")
        current_section = section

    # Format row
    if bold:
        print(f"\\textbf{{{config}}} & \\textbf{{{format_sci(metrics['mean_bias'])}}} & \\textbf{{{metrics['std_dev']:.4f}}} & \\textbf{{{metrics['mad']:.4f}}} & \\textbf{{{metrics['outlier_fraction']*100:.2f}}} \\\\")
    else:
        print(f"{config} & {format_sci(metrics['mean_bias'])} & {metrics['std_dev']:.4f} & {metrics['mad']:.4f} & {metrics['outlier_fraction']*100:.2f} \\\\")

print(r"\enddata")
print(r"\end{deluxetable*}")

# ===== CELL 011 [markdown] =====
# - Figure 4: σ_NMAD vs Redshift (All Configurations)

# ===== CELL 012 [code] =====
# Merged σ_NMAD vs Redshift - All configurations in one figure
# Left panel: LSDR10 (Combined, SDSS-only, DESI-only)
# Right panel: PS1DR2+unWISE (Combined, SDSS-only, DESI-only) + PS1DR2 optical + LSDR10 reference

# Load LSDR10 data
p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_SDSS_test_res_df = cu.readfile(P1_SDSSONLY_TESTSET_RES_PATH)
p1_DESI_test_res_df = cu.readfile(P1_DESIONLY_TESTSET_RES_PATH)

p1_test_labels = p1_test_res_df['label'].values
p1_test_preds = p1_test_res_df['pred'].values
p1_SDSS_test_preds = p1_SDSS_test_res_df['pred'].values
p1_DESI_test_preds = p1_DESI_test_res_df['pred'].values

# Load PS1DR2 data
p5_test_res_df = cu.readfile(P5_TESTSET_RES_PATH)  # optical only
p4_test_res_df = cu.readfile(P4_TESTSET_RES_PATH)
p4_SDSS_test_res_df = cu.readfile(P4_SDSSONLY_TESTSET_RES_PATH)
p4_DESI_test_res_df = cu.readfile(P4_DESIONLY_TESTSET_RES_PATH)

p5_test_labels = p5_test_res_df['label'].values
p5_test_preds = p5_test_res_df['pred'].values
p4_test_labels = p4_test_res_df['label'].values
p4_test_preds = p4_test_res_df['pred'].values
p4_SDSS_test_preds = p4_SDSS_test_res_df['pred'].values
p4_DESI_test_preds = p4_DESI_test_res_df['pred'].values

def calculate_bin_std(labels, preds, bins):
    """Calculate σ_NMAD for each redshift bin"""
    bin_indices = np.digitize(labels, bins)
    std_values = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 10:  # Minimum samples per bin
            bin_preds = preds[mask]
            bin_labels = labels[mask]
            std_val = cu.evaluate_redshift_quality(bin_labels, bin_preds)['mad']
            std_values.append(std_val)
        else:
            std_values.append(np.nan)
    return np.array(std_values)

# LSDR10 bins (wider range)
z_min, z_max, z_step = 0.0, 1.6, 0.1
z_bins = np.arange(z_min, z_max + z_step, z_step)
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

# PS1DR2 bins (shorter range)
PS_z_min, PS_z_max, PS_z_step = 0.0, 1.2, 0.1
PS_z_bins = np.arange(PS_z_min, PS_z_max + PS_z_step, PS_z_step)
PS_z_centers = (PS_z_bins[:-1] + PS_z_bins[1:]) / 2

# Calculate σ_NMAD for LSDR10
p1_std = calculate_bin_std(p1_test_labels, p1_test_preds, z_bins)
p1_SDSS_std = calculate_bin_std(p1_test_labels, p1_SDSS_test_preds, z_bins)
p1_DESI_std = calculate_bin_std(p1_test_labels, p1_DESI_test_preds, z_bins)

# Calculate σ_NMAD for PS1DR2
p5_std = calculate_bin_std(p5_test_labels, p5_test_preds, PS_z_bins)
p4_std = calculate_bin_std(p4_test_labels, p4_test_preds, PS_z_bins)
p4_SDSS_std = calculate_bin_std(p4_test_labels, p4_SDSS_test_preds, PS_z_bins)
p4_DESI_std = calculate_bin_std(p4_test_labels, p4_DESI_test_preds, PS_z_bins)

# Also calculate LSDR10 on PS range for reference
p1_ref_std = calculate_bin_std(p1_test_labels, p1_test_preds, PS_z_bins)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: LSDR10
ax1.plot(z_centers, p1_std, '-', label='All', linewidth=2.5, color='black')
ax1.plot(z_centers, p1_SDSS_std, ':', label='SDSS-only', linewidth=2, color='blue')
ax1.plot(z_centers, p1_DESI_std, '--', label='DESI-only', linewidth=2, color='red')

ax1.set_xlabel(r'$z_\mathrm{spec}$')
ax1.set_ylabel(r'$\sigma_{\mathrm{NMAD}}$')
ax1.set_xlim(-0.02, 1.7)
ax1.set_ylim(0.01, 0.08)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1.set_xticks(np.arange(0.0, 1.8, 0.2))
ax1.text(0.96, 0.04, 'LSDR10', transform=ax1.transAxes,
        verticalalignment='bottom', horizontalalignment='right', fontsize=18)

# Right panel: PS1DR2 comparison
ax2.plot(PS_z_centers, p4_std, '-', label='PS1DR2 + unWISE (All)', linewidth=2.5, color='black')
ax2.plot(PS_z_centers, p4_SDSS_std, ':', label='PS1DR2 + unWISE (SDSS-only)', linewidth=2, color='blue')
ax2.plot(PS_z_centers, p4_DESI_std, '--', label='PS1DR2 + unWISE (DESI-only)', linewidth=2, color='red')
ax2.plot(PS_z_centers, p5_std, '-.', label='PS1DR2 optical-only', linewidth=2, color='green')
ax2.plot(PS_z_centers, p1_ref_std, '-', label='LSDR10 (All)', linewidth=1.5, color='gray', alpha=0.7)

ax2.set_xlabel(r'$z_\mathrm{spec}$')
ax2.set_ylabel(r'$\sigma_{\mathrm{NMAD}}$')
ax2.set_xlim(-0.02, 1.2)
ax2.set_ylim(0.01, 0.10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=13)
ax2.set_xticks(np.arange(0.0, 1.4, 0.2))
ax2.text(0.96, 0.04, 'PS1DR2', transform=ax2.transAxes,
        verticalalignment='bottom', horizontalalignment='right', fontsize=18)

plt.tight_layout()

path = f"{FIGURE_DIR}/figure4.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 013 [markdown] =====
# - Figure 5: Single PDF samples: LSDR10

# ===== CELL 014 [code] =====
def resample_bins(prob, bin_edges, target_n=40):                  
    """     
    将原始的高分辨率 bin 数据重采样为目标 bin 数量 (通用插值法)。              
    保证概率质量绝对守恒。适用于任意 N 到任意 N 的转换。
    假设输入的 prob 是概率质量 (总和为1)。
    """     
    # 1. 构造新的 bin edges 和 centers
    new_bin_edges = np.linspace(bin_edges[0], bin_edges[-1], target_n + 1)                   
    new_bin_centers = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
  
    # 2. 计算原始的累积分布函数 (CDF)
    original_cdf = np.concatenate(([0], np.cumsum(prob)))              
    original_cdf /= original_cdf[-1]   
         
    # 3. 在新的 edges 上对 CDF 进行线性插值
    new_cdf_edges = np.interp(new_bin_edges, bin_edges, original_cdf)                        
    
    # 4. 新的概率就是新 CDF 的差分
    new_prob = np.diff(new_cdf_edges)    
    new_prob = new_prob / np.sum(new_prob)   	
   	  	  
    return new_prob, new_bin_edges, new_bin_centers 			
 		      	
def redshift_std(prob, bin_edges):     
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expectation = np.sum(prob * bin_centers)                      
    zVar = np.sum(prob * (bin_centers - expectation)**2)          
    zStd = np.sqrt(zVar)               
    return zStd   					
     	
def redshift_confidence(prob, bin_edges, alpha=0.03):             
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2            
    expectation = np.sum(prob * bin_centers)                      
    z_lower = expectation - alpha * (1 + expectation)             
    z_upper = expectation + alpha * (1 + expectation)             
    mask = (bin_centers >= z_lower) & (bin_centers <= z_upper)    
    zConf = np.sum(prob[mask])         
    return zConf   	 				
      	     
def plot_single_pdf_bar(ax, prob, expect, label, bin_centers, bin_edges, 
                        text_fontsize=12, legend_fontsize=12, info_text_y=0.95,
                        z_std_override=None, z_conf_override=None):             
    """绘制单个 PDF 柱状图的辅助函数"""      
    z_std = z_std_override if z_std_override is not None else redshift_std(prob, bin_edges)
    z_conf = z_conf_override if z_conf_override is not None else redshift_confidence(prob, bin_edges)
    z_lower = expect - z_std           
    z_upper = expect + z_std   		  

    bin_widths = np.diff(bin_edges)
    
    ax.axvspan(z_lower, z_upper, color='lightcoral', alpha=0.3,
                label=r'Confidence Interval (1$\sigma$)')   			
 			
    bars = ax.bar(bin_centers, prob, width=bin_widths[0],
                  color='steelblue', edgecolor='none',
                  label='Probability', align='center')
   
    ax.axvline(expect, color='red', linestyle='-', linewidth=2,   
                label=f'Expectation: {expect:.3f}')               
    ax.axvline(label, color='blue', linestyle='--', linewidth=2,  
                label=f'Label: {label:.3f}')   	 	 		
     		
    info_text = (                      
        fr'$\sigma$ : {z_std:.3f}' '\n'
        fr'$zConf$ : {z_conf:.3f}' '\n'
    )
    ax.text(0.98, info_text_y, info_text, transform=ax.transAxes,        
            verticalalignment='top', horizontalalignment='right', 
            fontsize=text_fontsize)    
  		
    ax.set_xlabel(r'$z$', fontsize=20) 
    ax.set_ylabel('Probability', fontsize=18)                     
    ax.set_xlim(0., 1.8)               
    ax.set_ylim(bottom=0)              
    ax.set_xticks(np.arange(0, 1.9, 0.2))                         
    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.9, 0.2)])
    ax.legend(fontsize=legend_fontsize)
             
# --- 数据加载 ---             
p1_res_df = cu.readfile(P1_TESTSET_RES_PATH)             
p1_test_labels = p1_res_df['label'].values   
p1_test_preds = p1_res_df['pred'].values   	  		
       
p1_test_df = cu.readfile(P1_TESTSET_PATH)    
p1_test_prob_path = f"{P1_CALIBRATOR_DIR}/predictions_test_prob.h5"   	 		
       
with h5py.File(p1_test_prob_path, 'r') as f:                         
    orig_bin_centers = f['bin_centers'][:]                        
    orig_bin_edges = f['bin_edges'][:] 
  		 	  
target_n = 40                          
print(f"Re-binning from {len(orig_bin_edges)-1} to {target_n} bins...")   	 				
     		
fig, axes = plt.subplots(2, 3, figsize=(18, 10))                  
text_fontsize = 18                     
legend_fontsize = 12    		     	 		
     		 	  
upper_idxs = [5, 6, 241]               
lower_idxs = [2501, 40585, 17399]   	 		
 	 	    
for idx, row_list in enumerate([upper_idxs, lower_idxs]):
    for col, n in enumerate(row_list):
        with h5py.File(p1_test_prob_path, 'r') as f:
            prob_400 = f['probabilities'][n]
            expect_400 = f['expectation'][n]
        label = p1_test_labels[n]

        # 用 N=400 原始概率计算 σ 和 zConf
        z_std_400 = redshift_std(prob_400, orig_bin_edges)
        z_conf_400 = redshift_confidence(prob_400, orig_bin_edges)

        # rebin 到 N=40 仅用于可视化
        prob_40, bin_edges_40, bin_centers_40 = resample_bins(prob_400, orig_bin_edges, target_n=target_n)
        # (f) 子图 info_text 下移避免和 legend 重叠
        ity = 0.65 if (idx == 1 and col == 2) else 0.95
        plot_single_pdf_bar(axes[idx, col], prob_40, expect_400, label,
                            bin_centers_40, bin_edges_40,
                            text_fontsize, legend_fontsize, info_text_y=ity,
                            z_std_override=z_std_400, z_conf_override=z_conf_400)   		
     		
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  
for i, ax in enumerate(axes.flat):     
    ax.text(0.95, 0.05, labels[i], transform=ax.transAxes,        
            fontsize=20, fontweight='bold',                       
            verticalalignment='bottom', horizontalalignment='right')   	
       	    		    
      	  
plt.tight_layout()             

path = f"{FIGURE_DIR}/figure5.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 015 [markdown] =====
# - Figure 6: PIT distribution check: Before vs After Temperature Scaling Calibration

# ===== CELL 016 [code] =====
def calculate_pit(probs, labels, bin_edges):
    """
    Calculate PIT (Probability Integral Transform) values.
    PIT = F(z_true), i.e., the predicted CDF at the true redshift.
    
    Args:
        probs: probability distributions [n_samples, n_bins]
        labels: true redshift values [n_samples]
        bin_edges: bin edges [n_bins + 1]
    
    Returns:
        pit_values: PIT values [n_samples]
    """
    n_samples = len(labels)
    pit_values = np.zeros(n_samples)

    # Calculate CDF from PDF
    cdf = np.cumsum(probs, axis=1)

    for i in range(n_samples):
        z_true = labels[i]
        # Find which bin z_true falls into
        bin_idx = np.searchsorted(bin_edges[1:], z_true)
        bin_idx = min(bin_idx, len(bin_edges) - 2)

        if bin_idx == 0:
            # Interpolate within first bin
            frac = (z_true - bin_edges[0]) / (bin_edges[1] - bin_edges[0])
            pit_values[i] = frac * cdf[i, 0]
        else:
            # Interpolate between bins
            frac = (z_true - bin_edges[bin_idx]) / (bin_edges[bin_idx + 1] - bin_edges[bin_idx])
            pit_values[i] = cdf[i, bin_idx - 1] + frac * (cdf[i, bin_idx] - cdf[i, bin_idx - 1])

    return pit_values

path_before = f"{P1_EXPERIMENT_DIR}/predictions_test_prob.h5"
with h5py.File(path_before, 'r') as f:
    probs_before = f['probabilities'][:]
    bin_centers = f['bin_centers'][:]
    bin_edges = f['bin_edges'][:]

path_after = f"{P1_CALIBRATOR_DIR}/predictions_test_prob.h5"
with h5py.File(path_after, 'r') as f:
    probs_after = f['probabilities'][:]

p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_test_labels = p1_test_res_df['label'].values

# Calculate PIT values
pit_before = calculate_pit(probs_before, p1_test_labels, bin_edges)
pit_after = calculate_pit(probs_after, p1_test_labels, bin_edges)

# Plot
plt.figure(figsize=(11, 5))
n_pit_bins = 20
pit_bins = np.linspace(0, 1, n_pit_bins + 1)
expected_freq = 1.0  # 均匀分布的density在[0,1]上为1

# Before calibration
plt.subplot(121)
counts_before, _, _ = plt.hist(pit_before, bins=pit_bins, density=True,
                                color='dimgray', alpha=0.7,
                                edgecolor='white', linewidth=0.5)
plt.axhline(expected_freq, color='red', linestyle='--', linewidth=2,
            label='Ideal')
plt.xlabel('PIT')
plt.ylabel('Relative Frequency')
plt.xlim(0, 1)
plt.ylim(0, 1.5)
plt.text(0.05, 0.95, 'Before Calibration', transform=plt.gca().transAxes,
        verticalalignment='top', horizontalalignment='left')
plt.legend()
plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)

# After calibration
plt.subplot(122)
counts_after, _, _ = plt.hist(pit_after, bins=pit_bins, density=True,
                                color='dimgray', alpha=0.7,
                                edgecolor='white', linewidth=0.5)
plt.axhline(expected_freq, color='red', linestyle='--', linewidth=2,
            label='Ideal')
plt.xlabel('PIT')
plt.ylabel('Relative Frequency')
plt.xlim(0, 1)
plt.ylim(0, 1.5)
plt.text(0.05, 0.95, 'After Calibration', transform=plt.gca().transAxes,
        verticalalignment='top', horizontalalignment='left')
plt.legend()
plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)

plt.tight_layout()

path = f"{FIGURE_DIR}/figure6.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 017 [markdown] =====
# - Figure 7: stack PDF for testset

# ===== CELL 018 [code] =====
from scipy.ndimage import gaussian_filter1d

p1_test_df = cu.readfile(P1_TESTSET_PATH)
p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_test_labels = p1_test_res_df['label'].values

path = f"{P1_EXPERIMENT_DIR}/predictions_test_prob.h5"
with h5py.File(path, 'r') as f:
    p1_test_probs = f['probabilities'][:]
    bin_centers = f['bin_centers'][:]
    bin_edges = f['bin_edges'][:]

# bin_width = bin_edges[1] - bin_edges[0]
# sigma = 4
sigma_z = 0.02  # 物理平滑尺度 (redshift)
bin_width = bin_edges[1] - bin_edges[0]
sigma = sigma_z / bin_width  # N=400时 sigma=4
  
plt.figure(figsize=(7, 5))

# All test set
stack_prob_all = np.sum(p1_test_probs, axis=0)
hist_all, _ = np.histogram(p1_test_labels, bins=bin_edges)
hist_all_smooth = gaussian_filter1d(hist_all.astype(float), sigma=sigma)
stack_prob_all_smooth = gaussian_filter1d(stack_prob_all, sigma=sigma)
# 归一化使 reconstructed 的总数与 true 一致
stack_prob_all_scaled = stack_prob_all_smooth * (np.sum(hist_all) / np.sum(stack_prob_all_smooth))

# DESI
desi_idx = np.where(p1_test_df['DESI_TARGETID'].fillna(0) > 0)[0]
desi_labels = p1_test_labels[desi_idx]
desi_probs = p1_test_probs[desi_idx]
desi_stack_prob = np.sum(desi_probs, axis=0)
hist_desi, _ = np.histogram(desi_labels, bins=bin_edges)
hist_desi_smooth = gaussian_filter1d(hist_desi.astype(float), sigma=sigma)
desi_stack_prob_smooth = gaussian_filter1d(desi_stack_prob, sigma=sigma)
desi_stack_prob_scaled = desi_stack_prob_smooth * (np.sum(hist_desi) / np.sum(desi_stack_prob_smooth))

# SDSS
sdss_idx = np.where(p1_test_df['SDSS_specobjid'].fillna(0) > 0)[0]
sdss_labels = p1_test_labels[sdss_idx]
sdss_probs = p1_test_probs[sdss_idx]
sdss_stack_prob = np.sum(sdss_probs, axis=0)
hist_sdss, _ = np.histogram(sdss_labels, bins=bin_edges)
hist_sdss_smooth = gaussian_filter1d(hist_sdss.astype(float), sigma=sigma)
sdss_stack_prob_smooth = gaussian_filter1d(sdss_stack_prob, sigma=sigma)
sdss_stack_prob_scaled = sdss_stack_prob_smooth * (np.sum(hist_sdss) / np.sum(sdss_stack_prob_smooth))

# All
plt.fill_between(bin_centers, hist_all_smooth, alpha=0.3, color='black', label=r'All - $N(z)$')
plt.plot(bin_centers, stack_prob_all_scaled, 'k-', linewidth=2, label=r'All - Stacked PDFs')
# DESI
plt.fill_between(bin_centers, hist_desi_smooth, alpha=0.3, color='blue', label=r'DESI - $N(z)$')
plt.plot(bin_centers, desi_stack_prob_scaled, 'b-', linewidth=2, label=r'DESI - Stacked PDFs')
# SDSS
plt.fill_between(bin_centers, hist_sdss_smooth, alpha=0.3, color='red', label=r'SDSS - $N(z)$')
plt.plot(bin_centers, sdss_stack_prob_scaled, 'r-', linewidth=2, label=r'SDSS - Stacked PDFs')

plt.xlabel(r'$z$')
plt.ylabel('Count')
plt.xlim(0, 1.8)
ymax = max(hist_all_smooth.max(), hist_desi_smooth.max(), hist_sdss_smooth.max(),
            stack_prob_all_scaled.max(), desi_stack_prob_scaled.max(), sdss_stack_prob_scaled.max())
plt.ylim(0, ymax * 1.2)
plt.legend(loc='upper right', ncol=1)  # ncol=1 纵向排列
plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)

plt.tight_layout()

path = f"{FIGURE_DIR}/figure7.pdf"
plt.savefig(path, format='pdf')

# ===== CELL 019 [markdown] =====
# - Figure 8: SHAP Analysis

# ===== CELL 020 [code] =====
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 特征名称映射
LSDR10_RENAME = {
    'dered_mag_g': r'$g$',
    'dered_mag_r': r'$r$',
    'dered_mag_i': r'$i$',
    'dered_mag_z': r'$z_{\mathrm{m}}$',
    'dered_mag_w1': r'$W1$',
    'dered_mag_w2': r'$W2$',
    'mag_g_Err': r'$\sigma_g$',
    'mag_r_Err': r'$\sigma_r$',
    'mag_i_Err': r'$\sigma_i$',
    'mag_z_Err': r'$\sigma_{z_{\mathrm{m}}}$',
    'mag_w1_Err': r'$\sigma_{W1}$',
    'mag_w2_Err': r'$\sigma_{W2}$',
    'g_r': r'$g-r$',
    'r_i': r'$r-i$',
    'i_z': r'$i-z_{\mathrm{m}}$',
    'z_w1': r'$z_{\mathrm{m}}-W1$',
    'w1_w2': r'$W1-W2$',
}

PS1DR2_RENAME = {
    'gKronMag_dered': r'$g^{\mathrm{Kron}}$',
    'rKronMag_dered': r'$r^{\mathrm{Kron}}$',
    'iKronMag_dered': r'$i^{\mathrm{Kron}}$',
    'zKronMag_dered': r'$z_{\mathrm{m}}^{\mathrm{Kron}}$',
    'yKronMag_dered': r'$y^{\mathrm{Kron}}$',
    'gPSFMag_dered': r'$g^{\mathrm{PSF}}$',
    'rPSFMag_dered': r'$r^{\mathrm{PSF}}$',
    'iPSFMag_dered': r'$i^{\mathrm{PSF}}$',
    'zPSFMag_dered': r'$z_{\mathrm{m}}^{\mathrm{PSF}}$',
    'yPSFMag_dered': r'$y^{\mathrm{PSF}}$',
    'gApMag_dered': r'$g^{\mathrm{Ap}}$',
    'rApMag_dered': r'$r^{\mathrm{Ap}}$',
    'iApMag_dered': r'$i^{\mathrm{Ap}}$',
    'zApMag_dered': r'$z_{\mathrm{m}}^{\mathrm{Ap}}$',
    'yApMag_dered': r'$y^{\mathrm{Ap}}$',
    'gKronMagErr': r'$\sigma_g^{\mathrm{Kron}}$',
    'rKronMagErr': r'$\sigma_r^{\mathrm{Kron}}$',
    'iKronMagErr': r'$\sigma_i^{\mathrm{Kron}}$',
    'zKronMagErr': r'$\sigma_{z_{\mathrm{m}}}^{\mathrm{Kron}}$',
    'yKronMagErr': r'$\sigma_y^{\mathrm{Kron}}$',
    'gPSFMagErr': r'$\sigma_g^{\mathrm{PSF}}$',
    'rPSFMagErr': r'$\sigma_r^{\mathrm{PSF}}$',
    'iPSFMagErr': r'$\sigma_i^{\mathrm{PSF}}$',
    'zPSFMagErr': r'$\sigma_{z_{\mathrm{m}}}^{\mathrm{PSF}}$',
    'yPSFMagErr': r'$\sigma_y^{\mathrm{PSF}}$',
    'gApMagErr': r'$\sigma_g^{\mathrm{Ap}}$',
    'rApMagErr': r'$\sigma_r^{\mathrm{Ap}}$',
    'iApMagErr': r'$\sigma_i^{\mathrm{Ap}}$',
    'zApMagErr': r'$\sigma_{z_{\mathrm{m}}}^{\mathrm{Ap}}$',
    'yApMagErr': r'$\sigma_y^{\mathrm{Ap}}$',
    'Kron_gr': r'$(g-r)^{\mathrm{Kron}}$',
    'Kron_ri': r'$(r-i)^{\mathrm{Kron}}$',
    'Kron_iz': r'$(i-z_{\mathrm{m}})^{\mathrm{Kron}}$',
    'Kron_zy': r'$(z_{\mathrm{m}}-y)^{\mathrm{Kron}}$',
    'Ap_gr': r'$(g-r)^{\mathrm{Ap}}$',
    'Ap_ri': r'$(r-i)^{\mathrm{Ap}}$',
    'Ap_iz': r'$(i-z_{\mathrm{m}})^{\mathrm{Ap}}$',
    'Ap_zy': r'$(z_{\mathrm{m}}-y)^{\mathrm{Ap}}$',
    'mag_w1': r'$W1$',
    'mag_w2': r'$W2$',
    'mag_w1_err': r'$\sigma_{W1}$',
    'mag_w2_err': r'$\sigma_{W2}$',
}

def rename_features(feature_names, rename_dict):
    """重命名特征"""
    return [rename_dict.get(name, name) for name in feature_names]

def plot_shap_dual_panel(shap_values, X_test, feature_names_orig, feature_names_renamed,
                         title, ax_bar, ax_summary, n_top=10, cmap='coolwarm'):
    """
    绘制SHAP双面板图：左图为Bar plot（明确特征重要性），右图为Summary plot（分布）
    """
    # 计算mean(|SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:n_top]

    # ========== 左图：Bar plot ==========
    y_pos = np.arange(len(sorted_idx))
    importances = mean_abs_shap[sorted_idx]

    bars = ax_bar.barh(y_pos, importances, color='steelblue', alpha=0.8, 
                       edgecolor='black', linewidth=0.8)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([feature_names_renamed[i] for i in sorted_idx], fontsize=TICKSIZE)
    ax_bar.set_xlabel('mean|SHAP|', fontsize=LABELSIZE)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, importances.max() * 1.15)
    ax_bar.tick_params(axis='x', labelsize=TICKSIZE)
    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')

    # ========== 右图：Summary plot ==========

    # 计算每个特征的SHAP值排序索引（用于排序scatter点）
    for plot_idx, feat_idx in enumerate(sorted_idx):
        shap_feat = shap_values[:, feat_idx]
        feature_values = X_test[:, feat_idx]

        # 标准化特征值用于颜色编码
        vmin = np.percentile(feature_values, 1)
        vmax = np.percentile(feature_values, 99)
        normalized_values = np.clip((feature_values - vmin) / (vmax - vmin), 0, 1)

        # 创建颜色映射
        colors = plt.cm.get_cmap(cmap)(normalized_values)

        # 添加jitter以避免重叠
        y = np.random.normal(plot_idx, 0.08, size=len(shap_feat))

        # 绘制scatter
        ax_summary.scatter(shap_feat, y, c=colors, alpha=0.5, s=12, edgecolors='none')

    ax_summary.set_yticks(y_pos)
    ax_summary.set_yticklabels([feature_names_renamed[i] for i in sorted_idx], fontsize=TICKSIZE)
    ax_summary.set_xlabel('SHAP value', fontsize=LABELSIZE)
    ax_summary.invert_yaxis()
    ax_summary.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_summary.tick_params(axis='x', labelsize=TICKSIZE)
    ax_summary.grid(axis='x', alpha=0.3, linestyle='--')

    return sorted_idx, mean_abs_shap[sorted_idx]


# ============ Main plotting code ============

# Load LSDR10 SHAP data
path_lsdr10 = f"{P1_CALIBRATOR_DIR}/shap_feature_importance.h5"
with h5py.File(path_lsdr10, 'r') as f:
    shap_values_lsdr10 = f['shap_values'][:]
    X_test_lsdr10 = f['X_test'][:]
    feature_names_lsdr10 = [s.decode() for s in f['feature_names'][:]]

# Load PANSTARRS SHAP data
path_ps1 = f"{P4_CALIBRATOR_DIR}/shap_feature_importance.h5"
with h5py.File(path_ps1, 'r') as f:
    shap_values_ps1 = f['shap_values'][:]
    X_test_ps1 = f['X_test'][:]
    feature_names_ps1 = [s.decode() for s in f['feature_names'][:]]

# 重命名特征
feature_names_lsdr10_renamed = rename_features(feature_names_lsdr10, LSDR10_RENAME)
feature_names_ps1_renamed = rename_features(feature_names_ps1, PS1DR2_RENAME)

# ========== 创建5子图布局 ==========
fig = plt.figure(figsize=(14, 10))

TICKSIZE = 12
LABELSIZE = 16

# 定义Each survey占据的空间 (修改为上下子图高度一致，均为0.40)
# 上层：主要的双面板图 (2列)
ax_bar_lsdr10 = fig.add_axes([0.08, 0.55, 0.35, 0.40])
ax_summary_lsdr10 = fig.add_axes([0.48, 0.55, 0.35, 0.40])

# 下层
ax_bar_ps1 = fig.add_axes([0.08, 0.08, 0.35, 0.40])
ax_summary_ps1 = fig.add_axes([0.48, 0.08, 0.35, 0.40])

# Colorbar轴 (同步对齐上下图的总高度: 0.08起，0.87高)
cbar_ax = fig.add_axes([0.86, 0.08, 0.012, 0.87])

# ========== Plot LSDR10 ==========
sorted_idx_lsdr10, importances_lsdr10 = plot_shap_dual_panel(
    shap_values_lsdr10, X_test_lsdr10, feature_names_lsdr10, feature_names_lsdr10_renamed,
    'LSDR10', ax_bar_lsdr10, ax_summary_lsdr10, n_top=10, cmap='RdYlBu_r'
)

# ========== Plot PS1DR2 ==========
sorted_idx_ps1, importances_ps1 = plot_shap_dual_panel(
    shap_values_ps1, X_test_ps1, feature_names_ps1, feature_names_ps1_renamed,
    'PS1DR2 + unWISE', ax_bar_ps1, ax_summary_ps1, n_top=10, cmap='RdYlBu_r'
)

# ========== 追加定制化设置 (左侧添加ylabel，上方删除xlabel) ==========
ax_bar_lsdr10.set_ylabel('Importance', fontsize=LABELSIZE)
ax_bar_ps1.set_ylabel('Importance', fontsize=LABELSIZE)

# ax_bar_lsdr10.set_xlabel('')
# ax_summary_lsdr10.set_xlabel('')

# ========== Add colorbar ==========
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'], fontsize=TICKSIZE) 
cbar.ax.set_ylabel('Feature Value', fontsize=LABELSIZE, rotation=270, labelpad=20)

# 右下角标注标题
ax_bar_lsdr10.text(0.96, 0.04, 'LSDR10', transform=ax_bar_lsdr10.transAxes,
         fontsize=20, verticalalignment='bottom', horizontalalignment='right');
ax_bar_ps1.text(0.96, 0.04, 'PS1DR2 + unWISE', transform=ax_bar_ps1.transAxes,
         fontsize=20, verticalalignment='bottom', horizontalalignment='right');

# 保存
path = f"{FIGURE_DIR}/figure8.pdf"
plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.2)

# ===== CELL 021 [markdown] =====
# - Priority level

# ===== CELL 022 [code] =====
p1_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p2_res_df = cu.readfile(P2_TESTSET_RES_PATH)
p3_res_df = cu.readfile(P3_TESTSET_RES_PATH)
p4_res_df = cu.readfile(P4_TESTSET_RES_PATH)
p5_res_df = cu.readfile(P5_TESTSET_RES_PATH)

metrics = {
    'p1': cu.evaluate_redshift_quality(p1_res_df['pred'], p1_res_df['label']),
    'p2': cu.evaluate_redshift_quality(p2_res_df['pred'], p2_res_df['label']),
    'p3': cu.evaluate_redshift_quality(p3_res_df['pred'], p3_res_df['label']),
    'p4': cu.evaluate_redshift_quality(p4_res_df['pred'], p4_res_df['label']),
    'p5': cu.evaluate_redshift_quality(p5_res_df['pred'], p5_res_df['label']),
}

# 创建DataFrame
metrics_list = []
for i in range(1, 6):
    key = f'p{i}'
    if key in metrics:
        m = metrics[key]
        metrics_list.append({
            'Priority': f'Priority {i}',
            'Bias': float(m['mean_bias']),
            r'$\sigma$': float(m['std_dev']),
            r'$\sigma_{NMAD}$': float(m['mad']),
            r'$\eta$': float(m['outlier_fraction'])
        })

# 创建DataFrame并打印
df = pd.DataFrame(metrics_list)
print(df.to_string(index=False))

# ===== CELL 023 [markdown] =====
# # 测光红移表
# - 优先级1：LSDR10 全波段 (grizW1W2)     → LSDR10模型
# - 优先级2：LSDR9 (grzW1W2) + PS1 i波段  → 混合模型
# - 优先级3：LSDR10（grzW1W2）            → LSDR10（仅grzW1W2）模型
# - 优先级4：PS1 (grizy) + unWISE (W1W2)  → PS1+unWISE模型
# - 优先级5：PS1 (grizy) 仅光学           → PS1光学模型

# ===== CELL 024 [markdown] =====
# # Compare to Zhou21

# ===== CELL 025 [code] =====
p1_test_df = cu.readfile(P1_TESTSET_PATH)
p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_test_df['z_pred'] = p1_test_res_df['pred'].values

metrics_ours = cu.evaluate_redshift_quality(p1_test_df['z'], p1_test_df['z_pred'])
mask = p1_test_df['z_phot_mean_i'] > 0
metrics_zhou23_i = cu.evaluate_redshift_quality(p1_test_df[mask]['z'], p1_test_df[mask]['z_phot_mean_i'])
mask = p1_test_df['z_phot_mean_grz'] > 0
metrics_zhou23_grz = cu.evaluate_redshift_quality(p1_test_df[mask]['z'], p1_test_df[mask]['z_phot_mean_grz'])


import pandas as pd

masks = {
    'z_{\\mathrm{m}} < 21': p1_test_df['dered_mag_z'] < 21,
    'z_{\\mathrm{m}} > 21': p1_test_df['dered_mag_z'] > 21,
    'All': pd.Series([True] * len(p1_test_df))
}

methods = {
    'Zhou23_i': 'z_phot_mean_i',
    'Zhou23_grz': 'z_phot_mean_grz',
    'This work': 'z_pred'
}

# 构建结果表
rows = []
for mask_name, mask in masks.items():
    subset = p1_test_df[mask]
    for method_name, col in methods.items():
        m = cu.evaluate_redshift_quality(subset['z'], subset[col])
        rows.append([
            mask_name, len(subset), method_name,
            m['mean_bias'], m['std_dev'], m['mad'], m['outlier_fraction']
        ])

results_df = pd.DataFrame(rows, columns=[
    'Sample', 'N', 'Method', 'Bias', r'$\sigma$', r'$\sigma_{NMAD}$', 'η_out'
])

# 格式化打印
pd.set_option('display.float_format', '{:.5f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(results_df.to_string(index=False))

# ===== CELL 026 [markdown] =====
# # Compare to Wen2024

# ===== CELL 027 [code] =====
# p1_test_df = cu.readfile(P1_TESTSET_PATH)
# p1_test_res_df = cu.readfile(P1_TESTSET_RES_PATH)
# p1_test_df['z_pred'] = p1_test_res_df['pred'].values

# output_path = f"{P1_CALIBRATOR_DIR}/testset_with_pred.fits"
# cu.savefile(p1_test_df, output_path)

# ===== CELL 028 [code] =====
# def phot_spec_cm(df):
#     df = df.set_index('uid')
#     dir_path = '/home/tiandc/Data/LegacySurveys/DR10/mstar_clean'
    
#     cols = [
#         'phot_z', 'phot_zErr', 'spec_z', 'mag_g', 'mag_g_Err',
#         'mag_r', 'mag_r_Err', 'mag_i', 'mag_i_Err', 'mag_z', 'mag_z_Err',
#         'mag_w1', 'mag_w1_Err', 'mag_w2', 'mag_w2_Err', 'mstar', 'BCG_flag',
#     ]

#     for i in range(72):
#         path = os.path.join(dir_path, f'mstar_clean_{i:02d}.fits')
#         backgal_df = cu.readfile(path).set_index('uid')

#         common_idx = backgal_df.index.intersection(df.index)
#         df.loc[common_idx, cols] = backgal_df.loc[common_idx, cols].values

#         print(f'File {i:02d} processed. Updated {len(common_idx)} records.')

#     return df

# path = f"{P1_CALIBRATOR_DIR}/testset_with_pred_xlsdr10loc.fits"
# df = cu.readfile(path)

# cols_to_rename = {'uid_1': 'uid', 'ra_1': 'ra', 'dec_1': 'dec'}
# df = df.rename(columns=cols_to_rename)

# df_save = phot_spec_cm(df).reset_index().apply(pd.to_numeric, errors='coerce')

# path = f"{P1_CALIBRATOR_DIR}/testset_with_pred_xlsdr10Phot.fits"
# cu.savefile(df_save, path)

# ===== CELL 029 [code] =====
path = f"{P1_CALIBRATOR_DIR}/testset_with_pred_xlsdr10Phot.fits"
p1_test_df = cu.readfile(path)
mask = p1_test_df['phot_z'] > 0
p1_test_df = p1_test_df[mask]


import pandas as pd

masks = {
    'z_{\\mathrm{m}} < 21': p1_test_df['dered_mag_z'] < 21,
    'z_{\\mathrm{m}} > 21': p1_test_df['dered_mag_z'] > 21,
    'All': pd.Series(True, index=p1_test_df.index)
}

methods = {
    'Wen2024': 'phot_z',
    'This work': 'z_pred'
}

# 构建结果表
rows = []
for mask_name, mask in masks.items():
    subset = p1_test_df[mask]
    for method_name, col in methods.items():
        m = cu.evaluate_redshift_quality(subset['z'], subset[col])
        rows.append([
            mask_name, len(subset), method_name,
            m['mean_bias'], m['std_dev'], m['mad'], m['outlier_fraction']
        ])

results_df = pd.DataFrame(rows, columns=[
    'Sample', 'N', 'Method', 'Bias', r'$\sigma$', r'$\sigma_{NMAD}$', 'η_out'
])

# 格式化打印
pd.set_option('display.float_format', '{:.5f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(results_df.to_string(index=False))

# ===== CELL 030 [markdown] =====
# # ELG poor performance reason

# ===== CELL 031 [code] =====
import pandas as pd

# Use the clean LSDR10 test set, then restrict to the primary six-band subset
path = '/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_test.fits'
test_df = cu.readfile(path)
p1_test_df = test_df[(test_df['dered_mag_i'] > 0) & (test_df['dered_mag_i'] < 30)].copy()

# DESI target-type masks
BGS_mask = (p1_test_df['DESI_DESI_TARGET'] & 2**60) != 0
LRG_mask = (p1_test_df['DESI_DESI_TARGET'] & 2**0) != 0
ELG_mask = (p1_test_df['DESI_DESI_TARGET'] & 2**1) != 0

rows = []
for sample, mask in [('BGS', BGS_mask), ('LRG', LRG_mask), ('ELG', ELG_mask)]:
    sub = p1_test_df[mask]
    rows.append({
        'Sample': sample,
        'N': len(sub),
        'median_zm': sub['dered_mag_z'].median(),
        'median_snr_g': sub['snr_g'].median(),
        'median_snr_r': sub['snr_r'].median(),
        'median_snr_i': sub['snr_i'].median(),
        'median_snr_z': sub['snr_z'].median(),
        'median_snr_w1': sub['snr_w1'].median(),
        'median_snr_w2': sub['snr_w2'].median(),
    })

compare_df = pd.DataFrame(rows)
pd.set_option('display.float_format', '{:.3f}'.format)
print(compare_df)

# ===== CELL 032 [markdown] =====
# # 以下非论文内容

# ===== CELL 033 [markdown] =====
# # 计算星系列表覆盖的天区面积

# ===== CELL 034 [code] =====
# import numpy as np
# import healpy as hp
# from astropy import units as u
# from astropy.coordinates import SkyCoord

# def calculate_coverage_area(ra_deg, dec_deg, nside=256):
#     """
#     计算星系列表覆盖的天区面积
    
#     参数
#     ----------
#     ra_deg : array_like
#         赤经（度）
#     dec_deg : array_like
#         赤纬（度）
#     nside : int
#         HEALPix NSIDE 参数，推荐 256 或 512
    
#     返回
#     -------
#     area_sqdeg : float
#         覆盖面积（平方度）
#     npix_used : int
#         使用的像素数
#     """
    
#     # 1. 将赤经赤纬转换为 HEALPix 像素索引
#     # healpy 的 ang2pix 需要 theta(0,π), phi(0,2π)，其中 theta=π/2 - dec, phi=ra
#     theta = np.radians(90.0 - dec_deg)  # 转换为余纬 (colatitude)
#     phi = np.radians(ra_deg)
    
#     # 获取每个星系对应的像素索引
#     pixel_indices = hp.ang2pix(nside, theta, phi, nest=False)  # RING 排序更常用
    
#     # 2. 统计唯一像素
#     unique_pixels = np.unique(pixel_indices)
#     npix_used = len(unique_pixels)
    
#     # 3. 计算每个像素的面积（平方度）
#     npix_total = 12 * nside * nside
#     pixel_area_sr = 4 * np.pi / npix_total  # 每个像素的立体角（球面度）
#     pixel_area_sqdeg = pixel_area_sr * (180.0 / np.pi) ** 2  # 转换为平方度
    
#     # 4. 总面积
#     total_area = npix_used * pixel_area_sqdeg
    
#     return total_area, npix_used, pixel_area_sqdeg

# ===== CELL 035 [code] =====
# path = '/home/tiandc/Data/LegacySurveys/DR9x10/lsdr9x10_gal_rawseg_allLoc.fits'
# df = cu.readfile(path)

# # 提取赤经赤纬
# ra = df['ra'].values
# dec = df['dec'].values

# # 计算覆盖面积
# area, npix, pix_area = calculate_coverage_area(ra, dec, nside=256)

# print(f"NSIDE = {256}")
# print(f"每个像素面积: {pix_area:.6f} 平方度")
# print(f"使用的唯一像素数: {npix}")
# print(f"覆盖面积: {area:.2f} 平方度")
# print(f"占全天比例: {area/41252.96*100:.2f}%")

# ===== CELL 036 [code] =====
# # 测试不同 NSIDE 的效果
# nsides = [64, 128, 256, 512, 1024, 2048, 4096]

# for nside in nsides:
#     area, npix, _ = calculate_coverage_area(ra, dec, nside=nside)
#     print(f"NSIDE={nside:4d}, 像素数={npix:6d}, 面积={area:8.2f} 平方度")

# ===== CELL 037 [markdown] =====
# # 比较不同点估计方法

# ===== CELL 038 [code] =====
import h5py
import numpy as np

# 加载PS1DR2的概率分布
# path = f"{P4_EXPERIMENT_DIR}/predictions_test_prob.h5"
# path = f"{P4_CALIBRATOR_DIR}/predictions_test_prob.h5"

dir_path = '/home/tiandc/photoz/review/p14/PS1DR2/Bin400_CRPS1.0+KL0.1'

path = os.path.join(dir_path, 'TempScalingCalib/predictions_test_prob.h5')
with h5py.File(path, 'r') as f:
    probs = f['probabilities'][:]
    bin_centers = f['bin_centers'][:]
    bin_edges = f['bin_edges'][:]

path = os.path.join(dir_path, 'TempScalingCalib/predictions_test.fits')
p4_res = cu.readfile(path)
labels = p4_res['label'].values

# 找低红移样本
low_z_mask = labels < 0.03
low_z_indices = np.where(low_z_mask)[0]

# 对比 expectation vs mode
for idx in low_z_indices[:10]:
    exp_z = np.sum(probs[idx] * bin_centers)  # 期望
    mode_z = bin_centers[np.argmax(probs[idx])]  # 众数
    med_idx = np.searchsorted(np.cumsum(probs[idx]), 0.5)  # 中位数
    med_z = bin_centers[min(med_idx, len(bin_centers)-1)]
    print(f"z_spec={labels[idx]:.4f}  expectation={exp_z:.4f}  mode={mode_z:.4f}  median={med_z:.4f}")

# ===== CELL 039 [markdown] =====
# # rebin effect

# ===== CELL 040 [code] =====
# ============================================================
# Rebin Effect: 评估从 N=80 rebin 到更少 bin 对点估计和校准的影响
# ============================================================
import h5py
import numpy as np
from scipy import stats

# --- 加载 P1 LSDR10 的原始概率分布 (N=80) ---
p1_prob_path = f"{P1_CALIBRATOR_DIR}/predictions_test_prob.h5"
with h5py.File(p1_prob_path, 'r') as f:
    p1_probs_orig = f['probabilities'][:]
    p1_bin_centers_orig = f['bin_centers'][:]
    p1_bin_edges_orig = f['bin_edges'][:]

p1_res_df = cu.readfile(P1_TESTSET_RES_PATH)
p1_labels = p1_res_df['label'].values

# --- 加载 P4 PS1DR2+unWISE 的原始概率分布 (N=80) ---
p4_prob_path = f"{P4_CALIBRATOR_DIR}/predictions_test_prob.h5"
with h5py.File(p4_prob_path, 'r') as f:
    p4_probs_orig = f['probabilities'][:]
    p4_bin_centers_orig = f['bin_centers'][:]
    p4_bin_edges_orig = f['bin_edges'][:]

p4_res_df = cu.readfile(P4_TESTSET_RES_PATH)
p4_labels = p4_res_df['label'].values

# --- Rebin 函数 (批量版本) ---
def rebin_all(probs, bin_edges, target_n):
    """将整个概率矩阵从原始bin数重采样到target_n"""
    n_samples = probs.shape[0]
    new_bin_edges = np.linspace(bin_edges[0], bin_edges[-1], target_n + 1)
    new_bin_centers = 0.5 * (new_bin_edges[:-1] + new_bin_edges[1:])
    
    new_probs = np.zeros((n_samples, target_n), dtype=np.float32)
    for i in range(n_samples):
        new_probs[i], _, _ = resample_bins(probs[i], bin_edges, target_n=target_n)
    
    return new_probs, new_bin_edges, new_bin_centers

# --- 计算点估计 ---
def compute_estimates(probs, bin_centers):
    expectation = (probs * bin_centers[None, :]).sum(axis=1)
    mode = bin_centers[np.argmax(probs, axis=1)]
    return expectation, mode

# --- 计算 PIT ---
def compute_pit_simple(probs, labels, bin_edges):
    n = len(labels)
    num_bins = probs.shape[1]
    cdf = np.cumsum(probs, axis=1)
    bin_idx = np.digitize(labels, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    
    pit = np.zeros(n)
    for i in range(n):
        idx = bin_idx[i]
        cdf_left = cdf[i, idx - 1] if idx > 0 else 0.0
        frac = (labels[i] - bin_edges[idx]) / (bin_edges[idx + 1] - bin_edges[idx])
        frac = np.clip(frac, 0, 1)
        pit[i] = cdf_left + probs[i, idx] * frac
    return pit

# --- 评估不同 target_n ---
target_ns = [40, 80, 100]  # 80 = 原始，不需要rebin

print("=" * 100)
print(f"{'Survey':<12} {'N_bin':<8} {'Δ_exp_max':<12} {'Δ_exp_mean':<14} {'Δ_exp_std':<12} {'KS_stat':<10} {'σ_NMAD':<10} {'η(%)':<8}")
print("=" * 100)

for name, probs_orig, bin_edges_orig, bin_centers_orig, labels in [
    ('P1_LSDR10', p1_probs_orig, p1_bin_edges_orig, p1_bin_centers_orig, p1_labels),
    ('P4_PS1DR2', p4_probs_orig, p4_bin_edges_orig, p4_bin_centers_orig, p4_labels),
]:
    # 原始 N=80 的结果
    expect_orig, mode_orig = compute_estimates(probs_orig, bin_centers_orig)
    pit_orig = compute_pit_simple(probs_orig, labels, bin_edges_orig)
    ks_orig, _ = stats.kstest(pit_orig, 'uniform')
    metrics_orig = cu.evaluate_redshift_quality(labels, expect_orig)
    
    print(f"{name:<12} {len(bin_centers_orig):<8} {'--':<12} {'--':<14} {'--':<12} {ks_orig:<10.4f} {metrics_orig['mad']:<10.4f} {metrics_orig['outlier_fraction']*100:<8.2f}")
    
    for tn in target_ns:
        # Rebin
        probs_rb, edges_rb, centers_rb = rebin_all(probs_orig, bin_edges_orig, tn)
        
        # 点估计对比
        expect_rb, _ = compute_estimates(probs_rb, centers_rb)
        delta_expect = np.abs(expect_rb - expect_orig)
        
        # PIT
        pit_rb = compute_pit_simple(probs_rb, labels, edges_rb)
        ks_rb, _ = stats.kstest(pit_rb, 'uniform')
        
        # σ_NMAD (用 rebin 后的 expectation)
        metrics_rb = cu.evaluate_redshift_quality(labels, expect_rb)
        
        print(f"{'':<12} {tn:<8} {delta_expect.max():<12.6f} {delta_expect.mean():<14.6f} {delta_expect.std():<12.6f} {ks_rb:<10.4f} {metrics_rb['mad']:<10.4f} {metrics_rb['outlier_fraction']*100:<8.2f}")

    print("-" * 100)

print()
print("Δ_exp = |expectation_rebinned - expectation_original|")
print("如果 Δ_exp_mean << σ_NMAD 且 KS/σ_NMAD 变化很小, 说明 rebin 对结果影响可忽略")

# ===== CELL 041 [code] =====
# ============================================================
# 公平对比 N=80 vs N=400 的 stacked N(z)
# 使用统一的 bin_edges 和按物理红移尺度的 sigma
# ============================================================
from scipy.ndimage import gaussian_filter1d

# --- 物理平滑尺度 ---
sigma_z = 0.03  # 平滑窗口 = 0.03 in redshift

# --- 统一的 histogram bin edges (独立于模型 bin 数) ---
common_bin_edges = np.linspace(0.0, 2.0, 201)  # Δz = 0.01, 200 bins
common_bin_centers = 0.5 * (common_bin_edges[:-1] + common_bin_edges[1:])
common_dz = common_bin_edges[1] - common_bin_edges[0]  # 0.01
sigma_bins = sigma_z / common_dz  # sigma in bin units

# --- 加载 N=80 数据 ---
p1_test_df_80 = cu.readfile(P1_TESTSET_PATH)
p1_res_df_80 = cu.readfile(P1_TESTSET_RES_PATH)
labels_80 = p1_res_df_80['label'].values

path_80 = f"{P1_CALIBRATOR_DIR}/predictions_test_prob.h5"
with h5py.File(path_80, 'r') as f:
    probs_80 = f['probabilities'][:]
    bin_centers_80 = f['bin_centers'][:]
    bin_edges_80 = f['bin_edges'][:]

# --- 加载 N=400 数据 (需要改为 N=400 路径) ---
# TODO: 替换为 N=400 的路径
P1_N400_CALIBRATOR_DIR = '/home/tiandc/photoz/catalog/models/p1/TempScalingCalib'
P1_N400_TESTSET_RES_PATH = f"{P1_N400_CALIBRATOR_DIR}/predictions_test.fits"

p1_res_df_400 = cu.readfile(P1_N400_TESTSET_RES_PATH)
labels_400 = p1_res_df_400['label'].values

path_400 = f"{P1_N400_CALIBRATOR_DIR}/predictions_test_prob.h5"
with h5py.File(path_400, 'r') as f:
    probs_400 = f['probabilities'][:]
    bin_centers_400 = f['bin_centers'][:]
    bin_edges_400 = f['bin_edges'][:]

# --- 辅助函数: 将模型概率投射到统一 bin 上 ---
def project_to_common_bins(probs, model_bin_edges, common_bin_edges):
    """
    将模型概率分布投射到统一的 common bins 上。
    对每个样本，用 CDF 插值将概率质量重新分配到 common bins。
    返回 stacked (sum) 结果。
    """
    n_samples = probs.shape[0]
    n_common = len(common_bin_edges) - 1
    stacked = np.zeros(n_common)
    
    for i in range(n_samples):
        # 原始 CDF
        orig_cdf = np.concatenate(([0], np.cumsum(probs[i])))
        orig_cdf /= orig_cdf[-1]  # 归一化
        # 插值到 common edges
        new_cdf = np.interp(common_bin_edges, model_bin_edges, orig_cdf)
        # 差分得到概率质量
        new_prob = np.diff(new_cdf)
        stacked += new_prob
    
    return stacked

# --- 计算 stacked PDFs (投射到统一 bin) ---
print("Computing stacked PDFs on common bins...")
stack_80 = project_to_common_bins(probs_80, bin_edges_80, common_bin_edges)
stack_400 = project_to_common_bins(probs_400, bin_edges_400, common_bin_edges)

# --- 真实 N(z) 直方图 (统一 bin) ---
hist_true_80, _ = np.histogram(labels_80, bins=common_bin_edges)
hist_true_400, _ = np.histogram(labels_400, bins=common_bin_edges)

# --- 高斯平滑 (统一的物理尺度 sigma) ---
hist_true_80_smooth = gaussian_filter1d(hist_true_80.astype(float), sigma=sigma_bins)
hist_true_400_smooth = gaussian_filter1d(hist_true_400.astype(float), sigma=sigma_bins)
stack_80_smooth = gaussian_filter1d(stack_80, sigma=sigma_bins)
stack_400_smooth = gaussian_filter1d(stack_400, sigma=sigma_bins)

# --- 归一化: reconstructed 总数 = true 总数 ---
stack_80_scaled = stack_80_smooth * (np.sum(hist_true_80) / np.sum(stack_80_smooth))
stack_400_scaled = stack_400_smooth * (np.sum(hist_true_400) / np.sum(stack_400_smooth))

# --- 绘图: 1x2 对比 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# Left: N=80
ax1.fill_between(common_bin_centers, hist_true_80_smooth, alpha=0.3, color='gray',
                 label=r'True $N(z)$')
ax1.plot(common_bin_centers, stack_80_scaled, 'k-', linewidth=2,
         label=r'Reconstructed $N(z)$')
ax1.set_xlabel(r'$z$')
ax1.set_ylabel('Count')
ax1.set_xlim(0, 1.8)
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right')
ax1.set_title(r'$N_\mathrm{bin} = 80$')
ax1.set_xticks(np.arange(0, 1.9, 0.2))

# Right: N=400
ax2.fill_between(common_bin_centers, hist_true_400_smooth, alpha=0.3, color='gray',
                 label=r'True $N(z)$')
ax2.plot(common_bin_centers, stack_400_scaled, 'k-', linewidth=2,
         label=r'Reconstructed $N(z)$')
ax2.set_xlabel(r'$z$')
ax2.set_ylabel('Count')
ax2.set_xlim(0, 1.8)
ax2.set_ylim(bottom=0)
ax2.legend(loc='upper right')
ax2.set_title(r'$N_\mathrm{bin} = 400$')
ax2.set_xticks(np.arange(0, 1.9, 0.2))

plt.tight_layout()

path = f"{FIGURE_DIR}/stackpdf_N80_vs_N400_fair.png"
plt.savefig(path, format='png')

print(f"Smoothing: sigma_z = {sigma_z}, sigma_bins = {sigma_bins:.1f}")
print(f"Common bins: {len(common_bin_centers)} bins, Δz = {common_dz}")


# AUTO-CONVERTED FROM process.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [code] =====
import cosmic.utils as cu
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

# ===== CELL 001 [markdown] =====
# # 只用DESI DR1的BGS和LRG

# ===== CELL 002 [code] =====
path = './desi_dr1_BGS_LRG_xsdssdr19.fits'
df_raw = cu.readfile(path)

# ===== CELL 003 [code] =====
df_raw.columns

# ===== CELL 004 [code] =====
df = df_raw.copy()
df['ra'] = np.nan
df['dec'] = np.nan
df['z'] = np.nan
df['zErr'] = np.nan

# SDSS only
idx = (df['SDSS_ra'] > 0) & (df['DESI_RA'].isna())
df.loc[idx, 'ra'] = df.loc[idx, 'SDSS_ra']
df.loc[idx, 'dec'] = df.loc[idx, 'SDSS_dec']
df.loc[idx, 'z'] = df.loc[idx, 'SDSS_z']
df.loc[idx, 'zErr'] = df.loc[idx, 'SDSS_zErr']

# DESI only
idx = (df['SDSS_ra'].isna()) & (df['DESI_RA'] > 0)
df.loc[idx, 'ra'] = df.loc[idx, 'DESI_RA']
df.loc[idx, 'dec'] = df.loc[idx, 'DESI_DEC']
df.loc[idx, 'z'] = df.loc[idx, 'DESI_Z']
df.loc[idx, 'zErr'] = df.loc[idx, 'DESI_ZERR']

# Overlap
idx = (df['SDSS_ra'] > 0) & (df['DESI_RA'] > 0)
overlap = df[idx]
desi_better = overlap['DESI_ZERR'] <= overlap['SDSS_zErr']
sdss_better = overlap['DESI_ZERR'] > overlap['SDSS_zErr']

df.loc[overlap[desi_better].index, 'ra'] = overlap[desi_better]['DESI_RA']
df.loc[overlap[desi_better].index, 'dec'] = overlap[desi_better]['DESI_DEC']
df.loc[overlap[desi_better].index, 'z'] = overlap[desi_better]['DESI_Z']
df.loc[overlap[desi_better].index, 'zErr'] = overlap[desi_better]['DESI_ZERR']

df.loc[overlap[sdss_better].index, 'ra'] = overlap[sdss_better]['SDSS_ra']
df.loc[overlap[sdss_better].index, 'dec'] = overlap[sdss_better]['SDSS_dec']
df.loc[overlap[sdss_better].index, 'z'] = overlap[sdss_better]['SDSS_z']
df.loc[overlap[sdss_better].index, 'zErr'] = overlap[sdss_better]['SDSS_zErr']

cols = ['ra', 'dec', 'z', 'zErr', 
        'DESI_TARGETID', 'DESI_Z', 'DESI_ZERR', 'DESI_ZWARN', 'DESI_DELTACHI2',
        'DESI_RA', 'DESI_DEC', 'DESI_MASKBITS', 'DESI_DESI_TARGET',
        'DESI_BGS_TARGET', 'DESI_PRIORITY', 
        'SDSS_specobjid', 'SDSS_ra', 'SDSS_dec', 'SDSS_z', 'SDSS_zErr']
df = df[cols]
df = df.fillna(np.nan)
print(df.shape)

# ===== CELL 005 [code] =====
# check Nan 
print(df[~(df['ra'] > 0)].shape)
print(df[~(df['z'] > 0)].shape)
print(df[~(df['zErr'] > 0)].shape)

# ===== CELL 006 [code] =====
# remove NaN
mask = (df['ra'] > 0)
mask &= (df['z'] > 0) 
mask &= (df['zErr'] > 0) & (df['zErr'] < 0.001)
df_save = df[mask]
print(df_save.shape)

# ===== CELL 007 [code] =====
output_path = './DESIDR1BGSLRG_xSDSSDR19.fits'
cu.savefile(df_save, output_path)

# ===== CELL 008 [markdown] =====
# # 重新清理的DESI DR1

# ===== CELL 009 [code] =====
path = './desidr1_xsdssdr19.fits'
df_raw = cu.readfile(path)

# ===== CELL 010 [code] =====
df_raw.columns

# ===== CELL 011 [code] =====
df = df_raw.copy()
df['ra'] = np.nan
df['dec'] = np.nan
df['z'] = np.nan
df['zErr'] = np.nan

# SDSS only
idx = (df['SDSS_ra'] > 0) & (df['DESI_RA'].isna())
df.loc[idx, 'ra'] = df.loc[idx, 'SDSS_ra']
df.loc[idx, 'dec'] = df.loc[idx, 'SDSS_dec']
df.loc[idx, 'z'] = df.loc[idx, 'SDSS_z']
df.loc[idx, 'zErr'] = df.loc[idx, 'SDSS_zErr']

# DESI only
idx = (df['SDSS_ra'].isna()) & (df['DESI_RA'] > 0)
df.loc[idx, 'ra'] = df.loc[idx, 'DESI_RA']
df.loc[idx, 'dec'] = df.loc[idx, 'DESI_DEC']
df.loc[idx, 'z'] = df.loc[idx, 'DESI_Z']
df.loc[idx, 'zErr'] = df.loc[idx, 'DESI_ZERR']

# Overlap
idx = (df['SDSS_ra'] > 0) & (df['DESI_RA'] > 0)
overlap = df[idx]
desi_better = overlap['DESI_ZERR'] <= overlap['SDSS_zErr']
sdss_better = overlap['DESI_ZERR'] > overlap['SDSS_zErr']

df.loc[overlap[desi_better].index, 'ra'] = overlap[desi_better]['DESI_RA']
df.loc[overlap[desi_better].index, 'dec'] = overlap[desi_better]['DESI_DEC']
df.loc[overlap[desi_better].index, 'z'] = overlap[desi_better]['DESI_Z']
df.loc[overlap[desi_better].index, 'zErr'] = overlap[desi_better]['DESI_ZERR']

df.loc[overlap[sdss_better].index, 'ra'] = overlap[sdss_better]['SDSS_ra']
df.loc[overlap[sdss_better].index, 'dec'] = overlap[sdss_better]['SDSS_dec']
df.loc[overlap[sdss_better].index, 'z'] = overlap[sdss_better]['SDSS_z']
df.loc[overlap[sdss_better].index, 'zErr'] = overlap[sdss_better]['SDSS_zErr']

cols = ['ra', 'dec', 'z', 'zErr', 
        'DESI_TARGETID', 'DESI_RA', 'DESI_DEC', 'DESI_Z', 'DESI_ZERR',
        'DESI_DELTACHI2', 'DESI_DESI_TARGET', 'DESI_BGS_TARGET', 'DESI_PRIORITY', 
        'SDSS_specobjid', 'SDSS_ra', 'SDSS_dec', 'SDSS_z', 'SDSS_zErr',]
df = df[cols]
df = df.fillna(np.nan)
print(df.shape)

# ===== CELL 012 [code] =====
# check Nan 
print(df[~(df['ra'] > 0)].shape)
print(df[~(df['z'] > 0)].shape)
print(df[~(df['zErr'] > 0)].shape)

# ===== CELL 013 [code] =====
# remove NaN
mask = (df['ra'] > 0)
mask &= (df['z'] > 0) 
mask &= (df['zErr'] > 0) & (df['zErr'] < 0.001)
df= df[mask]
print(f'After remove NaN: {len(df)}')

# ===== CELL 014 [code] =====
# 排除DESI与SDSS光谱红移差异大的星系
overlap_idx = (df['SDSS_z'] > 0) & (df['DESI_Z'] > 0)
overlap_data = df[overlap_idx]
z_diff_mask = abs(overlap_data['DESI_Z'] - overlap_data['SDSS_z']) > 0.005
exclude_idx = overlap_data[z_diff_mask].index
df_save = df.drop(exclude_idx)
print(f'After exclude DESI-SDSS z diff > 0.005: {len(df_save)}')

# ===== CELL 015 [code] =====
output_path = './DESIDR1_xSDSSDR19.fits'
cu.savefile(df_save, output_path)


# AUTO-CONVERTED FROM makeTrainSample.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [code] =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import cosmic.utils as cu

# ===== CELL 001 [markdown] =====
# # LSDR9x10 grizW1W2

# ===== CELL 002 [markdown] =====
# ### Add LSDR9x10 photometry

# ===== CELL 003 [code] =====
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Loc.fits'  
df_raw = cu.readfile(path)

df = df_raw.copy()

cols_to_rename = {'ra_1': 'ra', 'dec_1': 'dec'}
df = df.rename(columns=cols_to_rename)

cols = ['ra', 'dec', 'z', 'zErr', 
        'DESI_TARGETID', 'DESI_RA', 'DESI_DEC', 'DESI_Z', 'DESI_ZERR',
        'DESI_DELTACHI2', 'DESI_DESI_TARGET', 'DESI_BGS_TARGET', 'DESI_PRIORITY', 
        'SDSS_specobjid', 'SDSS_ra', 'SDSS_dec', 'SDSS_z', 'SDSS_zErr', 'uid']
df = df[cols]

# ===== CELL 004 [code] =====
def phot_spec_cm(df):
    df = df.set_index('uid')
    dir_path = '/home/tiandc/Data/LegacySurveys/DR9x10/raw_seg_uid'
    
    cols = [
        'z_phot_mean_grz','z_phot_median_grz', 'z_phot_std_grz', 
        'z_phot_mean_i', 'z_phot_median_i', 'z_phot_std_i', 'photo_z_source', 
        'dered_mag_g', 'dered_mag_r', 'dered_mag_i', 'dered_mag_z', 
        'dered_mag_w1', 'dered_mag_w2', 'dered_mag_w3', 'dered_mag_w4', 
        'snr_g', 'snr_r', 'snr_i', 'snr_z', 
        'snr_w1', 'snr_w2', 'snr_w3', 'snr_w4', 
        'fracflux_g', 'fracflux_r', 'fracflux_i', 'fracflux_z', 
        'fracmasked_g', 'fracmasked_r', 'fracmasked_i', 'fracmasked_z', 
        'fracin_g', 'fracin_r', 'fracin_i', 'fracin_z', 
        'allmask_g', 'allmask_r', 'allmask_i', 'allmask_z',
    ]

    for i in range(72):
        path = os.path.join(dir_path, f'lsdr9x10_gal_seg_{i:02d}.fits')
        backgal_df = cu.readfile(path).set_index('uid')

        common_idx = backgal_df.index.intersection(df.index)
        df.loc[common_idx, cols] = backgal_df.loc[common_idx, cols].values

        print(f'File {i:02d} processed. Updated {len(common_idx)} records.')

    return df

df_save = phot_spec_cm(df).reset_index().apply(pd.to_numeric, errors='coerce')

path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot.fits'
cu.savefile(df_save, path)

# ===== CELL 005 [code] =====
def snr_to_mag_err(snr):
    return 1.0857 / snr 

path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot.fits'
df_raw = cu.readfile(path)
df = df_raw.copy()

df['mag_g_Err'] = snr_to_mag_err(df['snr_g'])
df['mag_r_Err'] = snr_to_mag_err(df['snr_r'])
df['mag_i_Err'] = snr_to_mag_err(df['snr_i'])
df['mag_z_Err'] = snr_to_mag_err(df['snr_z'])
df['mag_w1_Err'] = snr_to_mag_err(df['snr_w1'])
df['mag_w2_Err'] = snr_to_mag_err(df['snr_w2'])

df['g_r'] = df['dered_mag_g'] - df['dered_mag_r']
df['r_i'] = df['dered_mag_r'] - df['dered_mag_i']
df['i_z'] = df['dered_mag_i'] - df['dered_mag_z']
df['z_w1'] = df['dered_mag_z'] - df['dered_mag_w1']
df['w1_w2'] = df['dered_mag_w1'] - df['dered_mag_w2']

# 星等大于0
mask = df[['dered_mag_g', 'dered_mag_r', # 这里先不区分i波段
           'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']].gt(0).all(axis=1)
df = df[mask]
print('After mag > 0:', len(df))

# 星等保证正常
mask = df[['dered_mag_g', 'dered_mag_r', # 这里先不区分i波段
           'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']].lt(30).all(axis=1)
df = df[mask]
print('After mag < 30:', len(df))

# 清理frac
def clean_FRAC(df):
    idx = (df['fracflux_g'] < 0.5) & (df['fracflux_r'] < 0.5) & (df['fracflux_z'] < 0.5)
    idx &= (df['fracmasked_g'] < 0.4) & (df['fracmasked_r'] < 0.4) & (df['fracmasked_z'] < 0.4)
    idx &= (df['fracin_g'] > 0.3) & (df['fracin_r'] > 0.3) & (df['fracin_z'] > 0.3)
    return df[idx]
df = clean_FRAC(df)
print('After clean FRAC:', len(df))

# 限制snr
def clean_snr(df):
    idx = (df.snr_g > 5) & (df.snr_r > 5) & (df.snr_z > 5)
    idx &= (df.snr_w1 > 5) & (df.snr_w2 > 5)
    return df[idx]
df = clean_snr(df)
print('After clean snr:', len(df))

output_path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean.fits'
cu.savefile(df, output_path)

# ===== CELL 006 [markdown] =====
# - grizW1W2

# ===== CELL 007 [code] =====
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean.fits'
df_raw = cu.readfile(path)

mask = (df_raw['dered_mag_i'] > 0) & (df_raw['dered_mag_i'] < 30)
mask &= (df_raw['z'] > 0.) & (df_raw['z'] <= 2.)
df_i = df_raw[mask]
print('grizW1W2:', len(df_i))

# split into train, val, test with 8:1:1 ratio
from sklearn.model_selection import train_test_split
df_trainval, df_test = train_test_split(df_i, test_size=0.1, random_state=42, shuffle=True)
df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=42, shuffle=True)

print(f"train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train.fits'
cu.savefile(df_train, path)
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_val.fits'
cu.savefile(df_val, path)
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_test.fits'
cu.savefile(df_test, path)

# ===== CELL 008 [markdown] =====
# - grzW1W2

# ===== CELL 009 [code] =====
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean.fits'
df_raw = cu.readfile(path)

mask = (df_raw['z'] > 0.) & (df_raw['z'] <= 2.)
df = df_raw[mask]
print('grzW1W2:', len(df))

# split into train, val, test with 8:1:1 ratio
from sklearn.model_selection import train_test_split
df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=42, shuffle=True)

print(f"train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_train.fits'
cu.savefile(df_train, path)
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_val.fits'
cu.savefile(df_val, path)
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_test.fits'
cu.savefile(df_test, path)

# ===== CELL 010 [markdown] =====
# ### 使用grizW1W2 训练集划分SDSS/DESI后重新分别划分训练集和验证集

# ===== CELL 011 [code] =====
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train.fits'
df_raw = cu.readfile(path)

# DESI only
desi_df = df_raw[df_raw['DESI_Z']>0]
print('DESI training sample size:', desi_df.shape)
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(desi_df, test_size=0.1, random_state=42, shuffle=True)
print(f"DESI only:  train: {len(df_train)}, val: {len(df_val)}")

path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_DESIonly_train.fits'
cu.savefile(df_train, path)
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_DESIonly_val.fits'
cu.savefile(df_val, path)

# SDSS only
sdss_df = df_raw[df_raw['SDSS_z']>0]
print('SDSS training sample size:', sdss_df.shape)
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(sdss_df, test_size=0.1, random_state=42, shuffle=True)
print(f"SDSS only:  train: {len(df_train)}, val: {len(df_val)}")

path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_SDSSonly_train.fits'
cu.savefile(df_train, path)
path = './data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_i_train_SDSSonly_val.fits'
cu.savefile(df_val, path)

# ===== CELL 012 [code] =====
df_desi.columns

# ===== CELL 013 [markdown] =====
# # LSDR9x10 grzW1W2 + PS1DR2 i

# ===== CELL 014 [code] =====
# Add LSDR9x10xPS1DR2 photometry
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Loc.fits'  
df_raw = cu.readfile(path)

df = df_raw.copy()

cols_to_rename = {'ra_1': 'ra', 'dec_1': 'dec'}
df = df.rename(columns=cols_to_rename)

# ===== CELL 015 [code] =====
def phot_spec_cm(df):
    df = df.set_index('objID')
    dir_path = '/home/tiandc/Data/PanSTARRS/DR2/final'
    
    cols = [
        'iPSFMag_dered', 'iKronMag_dered', 'iApMag_dered', 
        'iPSFMagErr', 'iApMagErr', 'iKronMagErr',
    ]

    for i in range(72):
        path = os.path.join(dir_path, f'ps1dr2_final_{i:d}.fits')
        backgal_df = cu.readfile(path).set_index('objID')

        common_idx = backgal_df.index.intersection(df.index)
        df.loc[common_idx, cols] = backgal_df.loc[common_idx, cols].values

        print(f'File {i:02d} processed. Updated {len(common_idx)} records.')

    return df

df_save = phot_spec_cm(df).reset_index().apply(pd.to_numeric, errors='coerce')

path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi.fits'
cu.savefile(df_save, path)

# ===== CELL 016 [markdown] =====
# - 用所有样本的grz

# ===== CELL 017 [code] =====
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi.fits'
df_raw = cu.readfile(path)

idx = (df_raw['z'] > 0.) & (df_raw['z'] <= 2.)
df_allgrz = df_raw[idx]

# split into train, val, test with 8:1:1 ratio
from sklearn.model_selection import train_test_split
df_trainval, df_test = train_test_split(df_allgrz, test_size=0.1, random_state=42, shuffle=True)
df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=42, shuffle=True)

print(f"train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_train.fits'
cu.savefile(df_train, path)
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_val.fits'
cu.savefile(df_val, path)
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_test.fits'
cu.savefile(df_test, path)

# ===== CELL 018 [markdown] =====
# - 只使用没有i波段数据的样本

# ===== CELL 019 [code] =====
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi.fits'
df_raw = cu.readfile(path)

idx = (df_raw['z'] > 0.) & (df_raw['z'] <= 2.)
idx &= np.isinf(df_raw['dered_mag_i']) | (df_raw['dered_mag_i'] == -9999)
df_grz = df_raw[idx]

# split into train, val, test with 8:1:1 ratio
from sklearn.model_selection import train_test_split
df_trainval, df_test = train_test_split(df_grz, test_size=0.1, random_state=42, shuffle=True)
df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=42, shuffle=True)

print(f"train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_grz_train.fits'
cu.savefile(df_train, path)
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_grz_val.fits'
cu.savefile(df_val, path)
path = './data/LSDR9x10xPS1DR2/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean_xPS1DR2Photi_grz_test.fits'
cu.savefile(df_test, path)

# ===== CELL 020 [code] =====
df_grz.groupby(df_grz['dered_mag_i']).size()


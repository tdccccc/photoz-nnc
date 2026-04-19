# AUTO-CONVERTED FROM process.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [code] =====
import cosmic.utils as cu
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

# ===== CELL 001 [markdown] =====
# # LRG

# ===== CELL 002 [code] =====
path = './LRG_full_noveto.dat.fits'
with fits.open(path) as hdul:
    header = hdul[1].header
    data = hdul[1].data
    tab = Table(data)

# ===== CELL 003 [code] =====
tab.columns

# ===== CELL 004 [code] =====
cols = ['TARGETID', 'RA', 'DEC', 'Z', 'ZERR', 'ZWARN',
        'SPECTYPE', 'DELTACHI2', 'MASKBITS', 'DESI_TARGET', 'BGS_TARGET',
        'PRIORITY', 'MORPHTYPE',
        'FLUX_G','FLUX_R','FLUX_Z', 'FLUX_W1','FLUX_W2',
        'FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2']
df = tab[cols].to_pandas()

# ===== CELL 005 [code] =====
# clean
# ref: Zhou2024
mask = df['SPECTYPE'] == 'GALAXY'
mask &= df['ZWARN'] == 0
mask &= df['MASKBITS'] == 0
mask &= df['MORPHTYPE'] != 'PSF'
mask &= (df['DESI_TARGET'] & 2**0) != 0 # LRG
mask &= (df['FLUX_G']>0) & (df['FLUX_R']>0) & (df['FLUX_Z']>0) 
mask &= (df['FLUX_W1']>0) & (df['FLUX_W2']>0)
mask &= (df['FLUX_IVAR_G']>0) & (df['FLUX_IVAR_R']>0) & (df['FLUX_IVAR_Z']>0) 
mask &= (df['FLUX_IVAR_W1']>0) & (df['FLUX_IVAR_W2']>0)
mask &= df['DELTACHI2'] > 25

cols_save = ['TARGETID', 'RA', 'DEC', 'Z', 'ZERR',
            'DELTACHI2', 'DESI_TARGET', 'BGS_TARGET',
            'PRIORITY']

df_save = df[mask][cols_save]
print(df_save.shape)
df_save.head()

# ===== CELL 006 [code] =====
output_path = './LRG_clean.fits'
cu.savefile(df_save, output_path)

# ===== CELL 007 [markdown] =====
# # BGS

# ===== CELL 008 [code] =====
path = './BGS_ANY_full_noveto.dat.fits'
with fits.open(path) as hdul:
    header = hdul[1].header
    data = hdul[1].data
    tab = Table(data)

# ===== CELL 009 [code] =====
tab.columns

# ===== CELL 010 [code] =====
cols = ['TARGETID', 'RA', 'DEC', 'Z', 'ZERR', 'ZWARN',
        'SPECTYPE', 'DELTACHI2', 'MASKBITS', 'DESI_TARGET', 'BGS_TARGET',
        'PRIORITY', 'MORPHTYPE',
        'FLUX_G','FLUX_R','FLUX_Z', 'FLUX_W1','FLUX_W2',
        'FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2']
df = tab[cols].to_pandas()

# ===== CELL 011 [code] =====
# clean
# ref: Zhou2024
mask = df['SPECTYPE'] == 'GALAXY'
mask &= df['ZWARN'] == 0
mask &= df['MASKBITS'] == 0
mask &= df['MORPHTYPE'] != 'PSF'
mask &= (df['DESI_TARGET'] & 2**60) != 0 # BGS
mask &= (df['FLUX_G']>0) & (df['FLUX_R']>0) & (df['FLUX_Z']>0) 
mask &= (df['FLUX_W1']>0) & (df['FLUX_W2']>0)
mask &= (df['FLUX_IVAR_G']>0) & (df['FLUX_IVAR_R']>0) & (df['FLUX_IVAR_Z']>0) 
mask &= (df['FLUX_IVAR_W1']>0) & (df['FLUX_IVAR_W2']>0)
mask &= df['DELTACHI2'] > 25

cols_save = ['TARGETID', 'RA', 'DEC', 'Z', 'ZERR',
            'DELTACHI2', 'DESI_TARGET', 'BGS_TARGET',
            'PRIORITY']

df_save = df[mask][cols_save]
print(df_save.shape)
df_save.head()

# ===== CELL 012 [code] =====
output_path = './BGS_clean.fits'
cu.savefile(df_save, output_path)

# ===== CELL 013 [markdown] =====
# # ELG

# ===== CELL 014 [code] =====
path = './ELG_LOPnotqso_full_noveto.dat.fits'
with fits.open(path) as hdul:
    header = hdul[1].header
    data = hdul[1].data
    tab = Table(data)

# ===== CELL 015 [code] =====
tab.columns

# ===== CELL 016 [code] =====
cols = ['TARGETID', 'RA', 'DEC', 'Z', 'ZERR', 'ZWARN',
        'SPECTYPE', 'DELTACHI2', 'MASKBITS', 'DESI_TARGET', 'BGS_TARGET',
        'PRIORITY', 'MORPHTYPE',
        'FLUX_G','FLUX_R','FLUX_Z', 'FLUX_W1','FLUX_W2',
        'FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2']
df = tab[cols].to_pandas()

# ===== CELL 017 [code] =====
df.head()

# ===== CELL 018 [code] =====
# clean
# ref: Zhou2024
mask = df['SPECTYPE'] == 'GALAXY'
mask &= df['ZWARN'] == 0
mask &= df['MASKBITS'] == 0
mask &= df['MORPHTYPE'] != 'PSF'
mask &= (df['DESI_TARGET'] & 2**1) != 0 # ELG
mask &= (df['FLUX_G']>0) & (df['FLUX_R']>0) & (df['FLUX_Z']>0) 
mask &= (df['FLUX_W1']>0) & (df['FLUX_W2']>0)
mask &= (df['FLUX_IVAR_G']>0) & (df['FLUX_IVAR_R']>0) & (df['FLUX_IVAR_Z']>0) 
mask &= (df['FLUX_IVAR_W1']>0) & (df['FLUX_IVAR_W2']>0)
mask &= df['DELTACHI2'] > 25

cols_save = ['TARGETID', 'RA', 'DEC', 'Z', 'ZERR',
            'DELTACHI2', 'DESI_TARGET', 'BGS_TARGET',
            'PRIORITY']

df_save = df[mask][cols_save]
print(df_save.shape)
df_save.head()

# ===== CELL 019 [code] =====
output_path = './ELG_clean.fits'
cu.savefile(df_save, output_path)

# ===== CELL 020 [markdown] =====
# # 合并LRG和BGS

# ===== CELL 021 [code] =====
bgs = cu.readfile('./BGS_clean.fits')
lrg = cu.readfile('./LRG_clean.fits')
elg = cu.readfile('./ELG_clean.fits')

# ===== CELL 022 [code] =====
# check duplicated
print(bgs.TARGETID.duplicated().sum())
print(lrg.TARGETID.duplicated().sum())
print(elg.TARGETID.duplicated().sum())

# ===== CELL 023 [code] =====
df_save = pd.concat([bgs, lrg, elg], axis=0)
df_save.columns = ['DESI_' + col for col in df_save.columns]

# ===== CELL 024 [code] =====
df_save.columns

# ===== CELL 025 [code] =====
output_path = 'desidr1.fits'
cu.savefile(df_save, output_path)


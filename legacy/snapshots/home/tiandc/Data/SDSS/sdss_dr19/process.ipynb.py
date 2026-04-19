# AUTO-CONVERTED FROM process.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [code] =====
import cosmic.utils as cu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== CELL 001 [code] =====
path = './MyDB_DachuanTian.csv'
df_raw = pd.read_csv(path)
df_raw.head()

# ===== CELL 002 [code] =====
df_raw.survey.unique()

# ===== CELL 003 [code] =====
df = df_raw.copy()

idx = (df['survey'] == 'boss') | (df['survey'] == 'eboss')
mask_boss_eboss = (df['z_noqso'] > 0.) 
mask_boss_eboss &= (df['zErr_noqso'] > 0)
mask_boss_eboss &= (df['zwarning_noqso'] == 0)
mask_boss_eboss &= (df['class_noqso'] == 'GALAXY')
df_boss_eboss = df.loc[idx & mask_boss_eboss]

mask_others = (df['z'] > 0.) 
mask_others &= (df['zErr'] > 0)
mask_others &= (df['zwarning'] == 0)
mask_others &= (df['class'] == 'GALAXY')
df_others = df.loc[~idx & mask_others]

df_save = pd.concat([df_boss_eboss, df_others], axis=0)
print(df_save.shape)

# ===== CELL 004 [code] =====
idx = (df_save['survey'] == 'boss') | (df_save['survey'] == 'eboss')
df_save[idx].head()

# ===== CELL 005 [code] =====
df_save.specobjid.duplicated().sum()

# ===== CELL 006 [code] =====
df_save.columns

# ===== CELL 007 [code] =====
cols_save = ['specobjid', 'ra', 'dec', 'z', 'zErr']
df_save = df_save[cols_save]
df_save.columns = ['SDSS_' + col for col in df_save.columns]
df_save.head()

# ===== CELL 008 [code] =====
output_path = './sdss_dr19.fits'
cu.savefile(df_save, output_path)

# ===== CELL 009 [markdown] =====
# # SDSS DR19 STAR

# ===== CELL 010 [code] =====
path = './SDSSDR19_STARQSO.csv'
df_raw = pd.read_csv(path)
df_raw.head()

# ===== CELL 011 [code] =====
df_starqso = df_raw[['ra','dec']]

output_path = './SDSSDR19_STARQSO_loc.fits'
cu.savefile(df_starqso, output_path)


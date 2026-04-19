# AUTO-CONVERTED FROM LSDR10_predict.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [code] =====
import cosmic.utils as cu
from galaxyClf_ann import GalaxyClassificationInference
import numpy as np
import matplotlib.pyplot as plt

model_dir = './LSDR10/grzW1W2/0NoSpec'
inference = GalaxyClassificationInference(model_dir)

# ===== CELL 001 [code] =====
data_path = '/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/raw72/lsdr9x10_01.fits'
df = cu.readfile(data_path)

# 计算 mag_Err（模型需要）
def snr_to_error(snr):
    return 2.5 / np.log(10) / snr

df['mag_g_Err'] = snr_to_error(df['snr_g'])
df['mag_r_Err'] = snr_to_error(df['snr_r'])
df['mag_z_Err'] = snr_to_error(df['snr_z'])
df['mag_w1_Err'] = snr_to_error(df['snr_w1'])
df['mag_w2_Err'] = snr_to_error(df['snr_w2'])

df['bp_g'] = df['gaia_phot_bp_mean_mag'] - df['gaia_phot_g_mean_mag']
df['g_rp'] = df['gaia_phot_g_mean_mag'] - df['gaia_phot_rp_mean_mag']

# ===== CELL 002 [code] =====
result = inference.predict(df)
pred_proba = result['probabilities']
df['pred_proba'] = pred_proba
print(f"预测完成，共 {len(df)} 行")
print(f"预测为星系 (prob >= 0.5): {(pred_proba >= 0.5).sum()}")
print(f"预测为非星系 (prob < 0.5): {(pred_proba < 0.5).sum()}")

# ===== CELL 003 [code] =====
# Use hexbin instead of scatter (better for large datasets)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: Galaxy (prob >= 0.5)
galaxy_mask = df['pred_proba'] >= 0.5
ax = axes[0]
hb = ax.hexbin(df.loc[galaxy_mask, 'r_z'], df.loc[galaxy_mask, 'z_w1'],
               gridsize=100, cmap='Blues', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('r - z')
ax.set_ylabel('z - W1')
ax.set_title(f'Galaxy (prob >= 0.5, N={galaxy_mask.sum()})')
ax.set_xlim(-1, 3)
ax.set_ylim(-2, 5)

# Right: Non-Galaxy (prob < 0.5)
star_mask = df['pred_proba'] < 0.5
ax = axes[1]
hb = ax.hexbin(df.loc[star_mask, 'r_z'], df.loc[star_mask, 'z_w1'],
               gridsize=100, cmap='Reds', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('r - z')
ax.set_ylabel('z - W1')
ax.set_title(f'Non-Galaxy (prob < 0.5, N={star_mask.sum()})')
ax.set_xlim(-1, 3)
ax.set_ylim(-2, 5)

plt.tight_layout()
plt.show()

# ===== CELL 004 [code] =====
# BP-G vs G-RP (Gaia colors)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
mask = (df['gaia_phot_bp_mean_mag'] > 0) & (df['gaia_phot_rp_mean_mag'] > 0)

# Left: Galaxy
galaxy_gaia = mask & (df['pred_proba'] >= 0.5)
ax = axes[0]
hb = ax.hexbin(df.loc[galaxy_gaia, 'bp_g'], df.loc[galaxy_gaia, 'g_rp'],
               gridsize=100, cmap='Blues', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('BP - G')
ax.set_ylabel('G - RP')
ax.set_title(f'Galaxy (N={galaxy_gaia.sum()})')

# Right: Non-Galaxy
star_gaia = mask & (df['pred_proba'] < 0.5)
ax = axes[1]
hb = ax.hexbin(df.loc[star_gaia, 'bp_g'], df.loc[star_gaia, 'g_rp'],
               gridsize=100, cmap='Reds', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('BP - G')
ax.set_ylabel('G - RP')
ax.set_title(f'Non-Galaxy (N={star_gaia.sum()})')

plt.tight_layout()
plt.show()

# ===== CELL 005 [code] =====
# Proper motion analysis
# Filter sources with valid Gaia photometry (to exclude sources without Gaia data)
has_gaia = df['gaia_phot_g_mean_mag'] > 0
print(f"Sources with Gaia photometry: {has_gaia.sum()} / {len(df)}")

# Calculate total proper motion (mas/yr)
df['pm_total'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)

# Define PM threshold (significant PM -> likely star)
PM_THRESHOLD = 3.0  # mas/yr

# Only analyze sources with valid Gaia data
pm_low = has_gaia & (df['pm_total'] < PM_THRESHOLD)   # PM < 3
pm_high = has_gaia & (df['pm_total'] >= PM_THRESHOLD)  # PM >= 3 (likely stars)

print(f"\nPM selection for Gaia sources (threshold = {PM_THRESHOLD} mas/yr):")
print(f"  PM < {PM_THRESHOLD}: {pm_low.sum()}")
print(f"  PM >= {PM_THRESHOLD} (Star): {pm_high.sum()}")

# ===== CELL 006 [code] =====
# Compare ANN prediction vs PM selection (only for sources with Gaia data)
ann_galaxy = df['pred_proba'] >= 0.5
ann_star = df['pred_proba'] < 0.5

# Cross-tabulation (only Gaia sources)
print("=" * 60)
print("ANN Prediction vs PM Selection (Gaia sources only)")
print("=" * 60)
print(f"Total Gaia sources: {has_gaia.sum()}")
print(f"                      | PM<3       | PM>=3 Star |")
print(f"ANN=Galaxy (prob>=0.5)| {(ann_galaxy & pm_low).sum():>10} | {(ann_galaxy & pm_high).sum():>10} |")
print(f"ANN=Non-Gal (prob<0.5)| {(ann_star & pm_low).sum():>10} | {(ann_star & pm_high).sum():>10} |")

# Agreement analysis (only Gaia sources)
agree = ((ann_galaxy & pm_low) | (ann_star & pm_high)).sum()
total_gaia = has_gaia.sum()
print(f"\nAgreement rate (Gaia): {agree}/{total_gaia} = {agree/total_gaia*100:.2f}%")

# Key: ANN rejects but PM low (among Gaia sources)
ann_reject_pm_low = ann_star & pm_low
print(f"\nANN rejects but PM<3: {ann_reject_pm_low.sum()}")
print("  (Possibly distant stars with insignificant PM)")

# Sources without Gaia data
no_gaia = ~has_gaia
print(f"\nSources without Gaia data: {no_gaia.sum()}")
print(f"  ANN=Galaxy: {(ann_galaxy & no_gaia).sum()}")
print(f"  ANN=Non-Galaxy: {(ann_star & no_gaia).sum()}")

# ===== CELL 007 [code] =====
# Visualization: ANN rejected but PM<3 sources (Gaia sources only)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Left: r-z vs z-W1
ax = axes[0]
# Background: all Gaia sources
ax.hexbin(df.loc[has_gaia, 'r_z'], df.loc[has_gaia, 'z_w1'], 
          gridsize=80, cmap='Greys', mincnt=1, alpha=0.5)
# Foreground: ANN rejected but PM<3
if ann_reject_pm_low.sum() > 0:
    ax.hexbin(df.loc[ann_reject_pm_low, 'r_z'], df.loc[ann_reject_pm_low, 'z_w1'],
              gridsize=80, cmap='Oranges', mincnt=1, alpha=0.8)
ax.set_xlabel('r - z')
ax.set_ylabel('z - W1')
ax.set_title(f'ANN=Non-Gal but PM<3 (N={ann_reject_pm_low.sum()})')
ax.set_xlim(-1, 3)
ax.set_ylim(-2, 5)

# Right: PM distribution comparison (Gaia sources only)
ax = axes[1]
bins = np.logspace(-2, 3, 100)
ax.hist(df.loc[ann_galaxy & has_gaia, 'pm_total'], bins=bins, alpha=0.5, 
        label=f'ANN=Galaxy (N={(ann_galaxy & has_gaia).sum()})', density=True)
ax.hist(df.loc[ann_star & has_gaia, 'pm_total'], bins=bins, alpha=0.5,
        label=f'ANN=Non-Galaxy (N={(ann_star & has_gaia).sum()})', density=True)
ax.axvline(PM_THRESHOLD, color='red', linestyle='--', label=f'PM threshold={PM_THRESHOLD}')
ax.set_xscale('log')
ax.set_xlabel('Total Proper Motion (mas/yr)')
ax.set_ylabel('Density')
ax.set_title('PM Distribution (Gaia sources)')
ax.legend()

plt.tight_layout()
plt.show()

# ===== CELL 008 [code] =====
# 4-quadrant color-color plot: ANN x PM classification (Gaia sources only)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

categories = [
    (ann_galaxy & pm_low,  'ANN=Galaxy & PM<3', 'Blues'),
    (ann_galaxy & pm_high, 'ANN=Galaxy & PM>=3 Star', 'Purples'),
    (ann_star & pm_low,    'ANN=Non-Gal & PM<3', 'Oranges'),
    (ann_star & pm_high,   'ANN=Non-Gal & PM>=3 Star', 'Reds'),
]

for ax, (mask, title, cmap) in zip(axes.flat, categories):
    n = mask.sum()
    if n > 0:
        ax.hexbin(df.loc[mask, 'r_z'], df.loc[mask, 'z_w1'],
                  gridsize=80, cmap=cmap, mincnt=1)
    ax.set_xlabel('r - z')
    ax.set_ylabel('z - W1')
    ax.set_title(f'{title} (N={n})')
    ax.set_xlim(-1, 3)
    ax.set_ylim(-2, 5)

plt.suptitle('Gaia Sources Only', y=1.02)
plt.tight_layout()
plt.show()

# ===== CELL 009 [code] =====
# Gaia color plot: 4-quadrant comparison
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
gaia_mask = (df['gaia_phot_bp_mean_mag'] > 0) & (df['gaia_phot_rp_mean_mag'] > 0)

categories = [
    (ann_galaxy & pm_low & gaia_mask,  'ANN=Galaxy & PM<3', 'Blues'),
    (ann_galaxy & pm_high & gaia_mask, 'ANN=Galaxy & PM>=3 Star', 'Purples'),
    (ann_star & pm_low & gaia_mask,    'ANN=Non-Gal & PM<3', 'Oranges'),
    (ann_star & pm_high & gaia_mask,   'ANN=Non-Gal & PM>=3 Star', 'Reds'),
]

for ax, (mask, title, cmap) in zip(axes.flat, categories):
    n = mask.sum()
    if n > 0:
        ax.hexbin(df.loc[mask, 'bp_g'], df.loc[mask, 'g_rp'],
                  gridsize=80, cmap=cmap, mincnt=1)
    ax.set_xlabel('BP - G')
    ax.set_ylabel('G - RP')
    ax.set_title(f'{title} (N={n})')

plt.tight_layout()
plt.show()

# ===== CELL 010 [code] =====
# Check sources with PM>=3 (Star) but ANN=Galaxy
ann_keep_pm_star = ann_galaxy & pm_high
print(f"PM>=3 Star but ANN=Galaxy: {ann_keep_pm_star.sum()}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Left: r-z vs z-W1
ax = axes[0]
ax.hexbin(df.loc[has_gaia, 'r_z'], df.loc[has_gaia, 'z_w1'], 
          gridsize=80, cmap='Greys', mincnt=1, alpha=0.5)
if ann_keep_pm_star.sum() > 0:
    ax.hexbin(df.loc[ann_keep_pm_star, 'r_z'], df.loc[ann_keep_pm_star, 'z_w1'],
              gridsize=80, cmap='Purples', mincnt=1, alpha=0.8)
ax.set_xlabel('r - z')
ax.set_ylabel('z - W1')
ax.set_title(f'PM>=3 Star but ANN=Galaxy (N={ann_keep_pm_star.sum()})')
ax.set_xlim(-1, 3)
ax.set_ylim(-2, 5)

# Middle: BP-G vs G-RP
ax = axes[1]
gaia_color_mask = (df['gaia_phot_bp_mean_mag'] > 0) & (df['gaia_phot_rp_mean_mag'] > 0)
mask_plot = ann_keep_pm_star & gaia_color_mask
if mask_plot.sum() > 0:
    ax.hexbin(df.loc[mask_plot, 'bp_g'], df.loc[mask_plot, 'g_rp'],
              gridsize=80, cmap='Purples', mincnt=1)
ax.set_xlabel('BP - G')
ax.set_ylabel('G - RP')
ax.set_title(f'Gaia Color (N={mask_plot.sum()})')

# Right: PM distribution
ax = axes[2]
bins = np.logspace(-1, 3, 50)
ax.hist(df.loc[ann_keep_pm_star, 'pm_total'], bins=bins, alpha=0.7, color='purple')
ax.axvline(PM_THRESHOLD, color='red', linestyle='--', label=f'PM threshold={PM_THRESHOLD}')
ax.set_xscale('log')
ax.set_xlabel('Total Proper Motion (mas/yr)')
ax.set_ylabel('Count')
ax.set_title('PM Distribution')
ax.legend()

plt.tight_layout()
plt.show()

# Statistics
print(f"\nStatistics for PM>=3 Star but ANN=Galaxy:")
print(f"  Median PM: {df.loc[ann_keep_pm_star, 'pm_total'].median():.2f} mas/yr")
print(f"  Mean pred_proba: {df.loc[ann_keep_pm_star, 'pred_proba'].mean():.3f}")

# ===== CELL 011 [code] =====
idx = ann_galaxy & pm_high
idx &= df['pred_proba'] > 0.6
df[idx].sort_values(by='fracflux_r', ascending=False)[
    ['ra','dec','pred_proba', 'pm_total', 'fracflux_r', 'fracin_r', 'fracmasked_r']]

# ===== CELL 012 [code] =====
tab = df[df['pred_proba'] > 0.6].sort_values(by='pred_proba', ascending=True)[
        ['ra','dec','pred_proba', 'pm_total', 'fracflux_r', 'fracin_r', 'fracmasked_r']].head(100)

# ===== CELL 013 [code] =====
tab.iloc[[97]]

# ===== CELL 014 [code] =====
tab[['ra','dec']].to_csv('check.csv', index=False)

# ===== CELL 015 [markdown] =====
# # 用光谱样本测试

# ===== CELL 016 [markdown] =====
# - NoSpec模型

# ===== CELL 017 [code] =====
import cosmic.utils as cu
from galaxyClf_ann import GalaxyClassificationInference
import numpy as np
import matplotlib.pyplot as plt

model_dir = './LSDR10/grzW1W2/0NoSpec'
inference = GalaxyClassificationInference(model_dir)

path = '../data/LSDR10/LSDR10.fits'
df = cu.readfile(path)
print('galaxy: ', df[df['label'] == 1].shape[0])
print('star&QSO: ', df[df['label'] == 0].shape[0])

result = inference.predict(df)
pred_proba = result['probabilities']
df['pred_proba'] = pred_proba
print(f"预测完成，共 {len(df)} 行")
print(f"预测为星系 (prob >= 0.5): {(pred_proba >= 0.5).sum()}")
print(f"预测为非星系 (prob < 0.5): {(pred_proba < 0.5).sum()}")

# ===== CELL 018 [code] =====
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

pred = (df['pred_proba'] >= 0.5).astype(int)
cm = confusion_matrix(df['label'], pred)

# Print rates
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Galaxy recall (TP / (TP+FN)): {tp/(tp+fn)*100:.2f}%")
print(f"Galaxy precision (TP / (TP+FP)): {tp/(tp+fp)*100:.2f}%")
print(f"Star/QSO recall (TN / (TN+FP)): {tn/(tn+fp)*100:.2f}%")
print(f"Star/QSO precision (TN / (TN+FN)): {tn/(tn+fn)*100:.2f}%")

# 污染率
print(f"点源对星系的污染率 (FP/(TP+FP)): {fp/(tp+fp)*100:.2f}%")
print(f"星系对点源的污染率 (FN/(TN+FN)): {fn/(tn+fn)*100:.2f}%")

fig, ax = plt.subplots(figsize=(4, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Star/QSO', 'Galaxy'])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title('NoSpec Model (ANN)')
plt.tight_layout()
plt.show()

# ===== CELL 019 [markdown] =====
# - NoSpec_NoErr模型

# ===== CELL 020 [code] =====
import cosmic.utils as cu
from galaxyClf_ann import GalaxyClassificationInference
import numpy as np
import matplotlib.pyplot as plt

model_dir = './LSDR10/grzW1W2/0NoSpec_NoErr'
inference = GalaxyClassificationInference(model_dir)

path = '../data/LSDR10/LSDR10.fits'
df = cu.readfile(path)
print('galaxy: ', df[df['label'] == 1].shape[0])
print('star&QSO: ', df[df['label'] == 0].shape[0])

result = inference.predict(df)
pred_proba = result['probabilities']
df['pred_proba'] = pred_proba
print(f"预测完成，共 {len(df)} 行")
print(f"预测为星系 (prob >= 0.5): {(pred_proba >= 0.5).sum()}")
print(f"预测为非星系 (prob < 0.5): {(pred_proba < 0.5).sum()}")

# ===== CELL 021 [code] =====
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

pred = (df['pred_proba'] >= 0.5).astype(int)
cm = confusion_matrix(df['label'], pred)

# Print rates
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Galaxy recall (TP / (TP+FN)): {tp/(tp+fn)*100:.2f}%")
print(f"Galaxy precision (TP / (TP+FP)): {tp/(tp+fp)*100:.2f}%")
print(f"Star/QSO recall (TN / (TN+FP)): {tn/(tn+fp)*100:.2f}%")
print(f"Star/QSO precision (TN / (TN+FN)): {tn/(tn+fn)*100:.2f}%")

# 污染率
print(f"点源对星系的污染率 (FP/(TP+FP)): {fp/(tp+fp)*100:.2f}%")
print(f"星系对点源的污染率 (FN/(TN+FN)): {fn/(tn+fn)*100:.2f}%")

fig, ax = plt.subplots(figsize=(4, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Star/QSO', 'Galaxy'])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title('NoSpec_NoErr Model (ANN)')
plt.tight_layout()
plt.show()

# ===== CELL 022 [markdown] =====
# # 小结
# - 使用grzW1W2波段预测，能够保留大部分星系，同时排除大量点源。
# - 可以设置 `fracflux_r > 1`，确保曝光正常。
# - 结合机器学习和GAIA自行应该能排除大量点源


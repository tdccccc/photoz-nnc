#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import cosmic.utils as cu
from photoz_ann import PhotozRegressionInference, Config
print(os.getcwd())

#%% model and config (binned classifier inference)
experiment_dir = "./grizW1W2/Mag&MagErr" 

RANDOM_SEED = 42
config_path = f"{experiment_dir}/config.yaml"

config = Config()
config.update_from_yaml(config_path)
print(f"Config loaded from: {config_path}")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"Dataset params: {config.dataset_params}")
print(f"Model params: {config.model_params}")

# Initialize inference helper (loads scaler and model internally)
infer = PhotozRegressionInference(model_dir=experiment_dir, device=device, batch_size=4096)

# Predict directly from file paths; labels from 'z' column
paths = config.dataset_mode.get('paths', {})
if not all(paths.get(split) for split in ['train', 'val', 'test']):
    raise ValueError("When using 'files' mode, all paths (train, val, test) must be specified")

train_df = cu.readfile(paths['train']); train_labels = train_df['z'].values
val_df = cu.readfile(paths['val']); val_labels = val_df['z'].values
test_df = cu.readfile(paths['test']); test_labels = test_df['z'].values

# Expectation for metrics/hexbin
train_preds = infer.predict(paths['train'])['predictions']
val_preds = infer.predict(paths['val'])['predictions']
test_preds = infer.predict(paths['test'])['predictions']

print(f"Inference completed")
print(f"  Trainset predictions range: [{train_preds.min():.4f}, {train_preds.max():.4f}]")
print(f"  Trainset labels range: [{train_labels.min():.4f}, {train_labels.max():.4f}]")
print(f"  Valset predictions range: [{val_preds.min():.4f}, {val_preds.max():.4f}]")
print(f"  Valset labels range: [{val_labels.min():.4f}, {val_labels.max():.4f}]")
print(f"  Testset predictions range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print(f"  Testset labels range: [{test_labels.min():.4f}, {test_labels.max():.4f}]")

prediction_train = np.vstack((train_labels, train_preds)).T
prediction_train_df = pd.DataFrame(prediction_train, columns=['label', 'pred'])
path = f"{experiment_dir}/predictions_train.fits"
cu.savefile(prediction_train_df, path)

prediction_val = np.vstack((val_labels, val_preds)).T
prediction_val_df = pd.DataFrame(prediction_val, columns=['label', 'pred'])
path = f"{experiment_dir}/predictions_val.fits"
cu.savefile(prediction_val_df, path)

prediction_test = np.vstack((test_labels, test_preds)).T
prediction_test_df = pd.DataFrame(prediction_test, columns=['label', 'pred'])
path = f"{experiment_dir}/predictions_test.fits"
cu.savefile(prediction_test_df, path)


# metrics
train_path = f"{experiment_dir}/predictions_train.fits"
val_path = f"{experiment_dir}/predictions_val.fits"
test_path = f"{experiment_dir}/predictions_test.fits"
train_df = cu.readfile(train_path)
val_df = cu.readfile(val_path)
test_df = cu.readfile(test_path)

train_labels = train_df['label'].values
train_preds = train_df['pred'].values
val_labels = val_df['label'].values
val_preds = val_df['pred'].values
test_labels = test_df['label'].values
test_preds = test_df['pred'].values

train_metrics = cu.evaluate_redshift_quality(train_labels, train_preds)
val_metrics = cu.evaluate_redshift_quality(val_labels, val_preds)
test_metrics = cu.evaluate_redshift_quality(test_labels, test_preds)

print("\n" + "="*60)
print(f"{'Metric':<15} {'Trainset':<12} {'Valset':<12} {'Testset':<12}")
print("="*60)
print(f"{'<Δz_norm>':<15} {train_metrics['mean_bias']:<12.6f} {val_metrics['mean_bias']:<12.6f} {test_metrics['mean_bias']:<12.6f}")
print(f"{'std':<15} {train_metrics['std_dev']:<12.6f} {val_metrics['std_dev']:<12.6f} {test_metrics['std_dev']:<12.6f}")
print(f"{'mad':<15} {train_metrics['mad']:<12.6f} {val_metrics['mad']:<12.6f} {test_metrics['mad']:<12.6f}")
print(f"{'outliers':<15} {train_metrics['outlier_fraction']:<12.6f} {val_metrics['outlier_fraction']:<12.6f} {test_metrics['outlier_fraction']:<12.6f}")
print("="*60)


# plot
plt.figure(figsize=(16, 5))

plt.subplot(131)
plt.hexbin(train_labels, train_preds, gridsize=100, bins='log', cmap='Blues', mincnt=1)
plt.plot([0, 1.2], [0, 1.2], 'r--', lw=2)
plt.xlabel(r'$z_\mathrm{spec}$')
plt.ylabel(r'$\mathrm{Predicted}\ z_\mathrm{phot}$')
plt.title('Trainset comparison')
text = f'<Δz_norm> = {train_metrics['mean_bias']:.5f}\nstd = {train_metrics['std_dev']:.5f}\nMAD = {train_metrics['mad']:.5f}\noutliers = {train_metrics['outlier_fraction']:.5f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='left',
         fontsize=12)

plt.subplot(132)
plt.hexbin(val_labels, val_preds, gridsize=100, bins='log', cmap='Blues', mincnt=1)
plt.plot([0, 1.2], [0, 1.2], 'r--', lw=2)
plt.xlabel(r'$z_\mathrm{spec}$')
plt.ylabel(r'$\mathrm{Predicted}\ z_\mathrm{phot}$')
plt.title('Valset comparison')
text = f'<Δz_norm> = {val_metrics['mean_bias']:.5f}\nstd = {val_metrics['std_dev']:.5f}\nMAD = {val_metrics['mad']:.5f}\noutliers = {val_metrics['outlier_fraction']:.5f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='left',
         fontsize=12)

plt.subplot(133)
plt.hexbin(test_labels, test_preds, gridsize=100, bins='log', cmap='Blues', mincnt=1)
plt.plot([0, 1.2], [0, 1.2], 'r--', lw=2)
plt.xlabel(r'$z_\mathrm{spec}$')
plt.ylabel(r'$\mathrm{Predicted}\ z_\mathrm{phot}$')
plt.title('Testset comparison')
text = f'<Δz_norm> = {test_metrics['mean_bias']:.5f}\nstd = {test_metrics['std_dev']:.5f}\nMAD = {test_metrics['mad']:.5f}\noutliers = {test_metrics['outlier_fraction']:.5f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='left',
         fontsize=12)

plt.tight_layout()

path = f"{experiment_dir}/results.jpg"
plt.savefig(path)


# redshift distribution
plt.figure(figsize=(18, 5))
# trainset
bins_train = np.linspace(
    min(train_labels.min(), train_preds.min()),
    max(train_labels.max(), train_preds.max()),
    101)
plt.subplot(131)
plt.hist(train_labels, bins=bins_train, alpha=0.75, label='Spec z', color='blue')
plt.hist(train_preds, bins=bins_train, alpha=0.75, label='Photo z', color='red')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Trainset')
plt.legend()

# valset
bins_val = np.linspace(
    min(val_labels.min(), val_preds.min()),
    max(val_labels.max(), val_preds.max()),
    101)
plt.subplot(132)
plt.hist(val_labels, bins=bins_val, alpha=0.75, label='Spec z', color='blue')
plt.hist(val_preds, bins=bins_val, alpha=0.75, label='Photo z', color='red')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Valset')
plt.legend()

# testset
bins_test = np.linspace(
    min(test_labels.min(), test_preds.min()),
    max(test_labels.max(), test_preds.max()),
    101)
plt.subplot(133)
plt.hist(test_labels, bins=bins_test, alpha=0.75, label='Spec z', color='blue')
plt.hist(test_preds, bins=bins_test, alpha=0.75, label='Photo z', color='red')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Testset')
plt.legend()
plt.tight_layout()

path = f"{experiment_dir}/results_hist.jpg"
plt.savefig(path)



# deviation distribution
plt.figure(figsize=(16, 5))

plt.subplot(131)
dev = train_preds - train_labels
plt.hist(dev, bins=100, alpha=0.5, label='Deviation', color='blue')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Trainset')
plt.legend()

plt.subplot(132)
dev = val_preds - val_labels
plt.hist(dev, bins=100, alpha=0.5, label='Deviation', color='blue')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Valset')
plt.legend()

plt.subplot(133)
dev = test_preds - test_labels
plt.hist(dev, bins=100, alpha=0.5, label='Deviation', color='blue')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('Testset')
plt.legend()

plt.tight_layout()

path = f"{experiment_dir}/results_dev.jpg"
plt.savefig(path)


# std vs redshift
z_min, z_max, z_step = 0.0, 1.2, 0.05
z_bins = np.arange(z_min, z_max + z_step, z_step)
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

def calculate_bin_std(labels, preds, bins):
    """Calculate standard deviation for each redshift bin"""
    bin_indices = np.digitize(labels, bins)
    std_values = []
    counts = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_preds = preds[mask]
            bin_labels = labels[mask]
            std_val = cu.evaluate_redshift_quality(bin_labels, bin_preds)['std_dev']
            std_values.append(std_val)
            counts.append(np.sum(mask))
        else:
            std_values.append(np.nan)
            counts.append(0)
    
    return np.array(std_values), np.array(counts)

train_std, train_counts = calculate_bin_std(train_labels, train_preds, z_bins)
val_std, val_counts = calculate_bin_std(val_labels, val_preds, z_bins)
test_std, test_counts = calculate_bin_std(test_labels, test_preds, z_bins)

plt.figure(figsize=(8, 6))
plt.plot(z_centers, train_std, 'o-', label='Train', linewidth=2, markersize=4)
plt.plot(z_centers, val_std, 'o-', label='Val', linewidth=2, markersize=4, color='orange')
plt.plot(z_centers, test_std, 'o-', label='Test', linewidth=2, markersize=4, color='green')
plt.xlabel('Redshift')
plt.ylabel('Standard Deviation')
plt.title('Std vs Redshift')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

path = f"{experiment_dir}/std_vs_redshift.jpg"
plt.savefig(path, dpi=150, bbox_inches='tight')


#%% predict probabilities
experiment_dir = "./5deltachi2>1000"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
infer = PhotozBinInference(model_dir=experiment_dir, 
                           device=device, batch_size=4096)

test_path = infer.config.dataset_mode.get('paths', {})['test']
save_path = os.path.join(experiment_dir, 'predictions_test_prob.h5')
infer.predict_probabilities_chunked(test_path, save_path,
                                    chunk_size=1000000, 
                                    batch_size=4096,
                                    compression='gzip')



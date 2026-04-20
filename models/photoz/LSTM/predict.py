#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import cosmic.utils as cu
from photoz_lstm import Dataset_Photz, DeepPhotozLSTM, DeepPhotozLSTMWithUncertainty, Config

#%% model and config
experiment_dir = "./0LSTMwithUncertainty"
config_path = f"{experiment_dir}/config.yaml"
model_path = f"{experiment_dir}/best_model.pkl"
scaler_path = f"{experiment_dir}/scaler.pkl"

config = Config()
config.update_from_yaml(config_path)
print(f"Config loaded from: {config_path}")
config.batch_size = 4096

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Dataset params: {config.dataset_params}")
print(f"Model params: {config.model_params}")

scaler = joblib.load(scaler_path)
print(f"Scaler loaded from: {scaler_path}")

if hasattr(scaler, 'n_features_in_'):
    input_dim = scaler.n_features_in_
else:
    raise ValueError("Cannot determine input dimension from scaler")

model_params = config.model_params.copy()
model_params['input_dim'] = input_dim

# Dynamically select model class based on config
model_class_name = config.model_class_name
model_class = globals().get(model_class_name)
if model_class is None:
    raise ValueError(f"Model class '{model_class_name}' not found. Make sure it's imported in predict.py.")

model = model_class(**model_params)
model.load_state_dict(torch.load(model_path, 
                                 map_location=device, 
                                 weights_only=True))
model.to(device)
model.eval()

print(f"Model loaded from: {model_path}")
print(f"Model input dimension: {input_dim}")
print(f"Model architecture: {model_params}")

mode_type = config.dataset_mode.get('type', 'ratio')
mag_types = config.dataset_params.get('mag_types')
color_types = config.dataset_params.get('color_types')
use_mag = config.dataset_params.get('use_mag')
use_magerr = config.dataset_params.get('use_magerr')
use_colorerr = config.dataset_params.get('use_colorerr')

if mode_type == 'files':
    # Load datasets directly from file paths
    paths = config.dataset_mode.get('paths', {})
    if not all(paths.get(split) for split in ['train', 'val', 'test']):
        raise ValueError("When using 'files' mode, all paths (train, val, test) must be specified")
    
    print("Loading datasets from specified file paths...")
    
    # paths = {
    #     'train': '/home/tiandc/panstarrs/photoz_ANN/trainset.fits',
    #     'val': '/home/tiandc/panstarrs/photoz_ANN/valset.fits',
    #     'test': '/home/tiandc/panstarrs/photoz_ANN/testset.fits'
    # }
    
    # Create datasets directly from file paths
    train_dataset = Dataset_Photz(
        file_path=paths['train'],
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=scaler,
        mode='train',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    
    val_dataset = Dataset_Photz(
        file_path=paths['val'],
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=scaler,
        mode='validation',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    
    test_dataset = Dataset_Photz(
        file_path=paths['test'],
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=scaler,
        mode='test',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    
    trainloader = DataLoader(train_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    valloader = DataLoader(val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    testloader = DataLoader(test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Data loaded from paths: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

elif mode_type == 'ratio':
    # Use ratio-based splitting (original logic)
    ratios = config.dataset_mode.get('ratios')
    dataset_path = config.dataset_params.get('file_path')
    
    all_dataset = Dataset_Photz(
        file_path=dataset_path,
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=None,
        mode='train',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    
    indices = list(range(len(all_dataset)))
    val_test_ratio = ratios['val'] + ratios['test']
    train_indices, temp_indices = train_test_split(
        indices, test_size=val_test_ratio, 
        random_state=config.random_seed
    )
    val_size_corrected = ratios['val'] / val_test_ratio
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_size_corrected), 
        random_state=config.random_seed
    )
    
    train_dataset = Dataset_Photz(
        file_path=dataset_path,
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=scaler,
        mode='train',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    
    train_subset = Subset(train_dataset, train_indices)
    
    val_dataset = Dataset_Photz(
        file_path=dataset_path,
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=scaler,
        mode='validation',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    val_subset = Subset(val_dataset, val_indices)
    
    test_dataset = Dataset_Photz(
        file_path=dataset_path,
        mag_types=mag_types,
        use_mag=use_mag,
        scaler_X=scaler,
        mode='test',
        use_magerr=use_magerr,
        use_colorerr=use_colorerr,
        color_types=color_types
    )
    test_subset = Subset(test_dataset, test_indices)
    
    trainloader = DataLoader(train_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    valloader = DataLoader(val_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    testloader = DataLoader(test_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

else:
    raise ValueError(f"Unknown dataset_mode type: {mode_type}. Supported types are 'files' and 'ratio'.")


# inference
train_preds = []
train_labels = []
train_sigmas = []  # To store uncertainty if available
val_preds = []
val_labels = []
val_sigmas = []
test_preds = []
test_labels = []
test_sigmas = []

print("Starting inference...")
with torch.no_grad():
    for batch in trainloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(inputs)
        
        if config.model_class_name == 'DeepPhotozLSTMWithUncertainty':
            mu, sigma = outputs
            train_preds.extend(mu.cpu().numpy().flatten())
            train_sigmas.extend(sigma.cpu().numpy().flatten())
        else:
            train_preds.extend(outputs.cpu().numpy().flatten())
            
        train_labels.extend(labels.cpu().numpy().flatten())
    print("  Trainset done")

    for batch in valloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(inputs)
        
        if config.model_class_name == 'DeepPhotozLSTMWithUncertainty':
            mu, sigma = outputs
            val_preds.extend(mu.cpu().numpy().flatten())
            val_sigmas.extend(sigma.cpu().numpy().flatten())
        else:
            val_preds.extend(outputs.cpu().numpy().flatten())

        val_labels.extend(labels.cpu().numpy().flatten())
    print("  Valset done")
    
    for batch in testloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(inputs)
        
        if config.model_class_name == 'DeepPhotozLSTMWithUncertainty':
            mu, sigma = outputs
            test_preds.extend(mu.cpu().numpy().flatten())
            test_sigmas.extend(sigma.cpu().numpy().flatten())
        else:
            test_preds.extend(outputs.cpu().numpy().flatten())
            
        test_labels.extend(labels.cpu().numpy().flatten())
    print("  Testset done")

train_preds = np.array(train_preds)
train_labels = np.array(train_labels)
val_preds = np.array(val_preds)
val_labels = np.array(val_labels)
test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# Also convert sigmas to numpy arrays if they exist
train_sigmas = np.array(train_sigmas) if train_sigmas else None
val_sigmas = np.array(val_sigmas) if val_sigmas else None
test_sigmas = np.array(test_sigmas) if test_sigmas else None


# Apply inverse transformation to get original scale if label transformation was used
if config.label_transform != 'none':
    print(f"Applying inverse transformation: {config.label_transform}")
    train_preds = Dataset_Photz.inverse_transform_labels(
        train_preds, 
        config.label_transform, 
        config.label_transform_params
    )
    
    val_preds = Dataset_Photz.inverse_transform_labels(
        val_preds, 
        config.label_transform, 
        config.label_transform_params
    )
    
    test_preds = Dataset_Photz.inverse_transform_labels(
        test_preds, 
        config.label_transform, 
        config.label_transform_params
    )

    # IMPORTANT: Also transform sigma back. For y=sqrt(z), dy = 1/(2*sqrt(z))*dz => dz = 2*sqrt(z)*dy = 2*y*dy
    # So, sigma_z approx 2 * y_pred * sigma_y
    if train_sigmas is not None:
        if config.label_transform == 'sqrt':
            # Inverse transform sigma. The predictions are still in the transformed space here.
            train_sigmas = 2 * train_preds * train_sigmas
            val_sigmas = 2 * val_preds * val_sigmas
            test_sigmas = 2 * test_preds * test_sigmas
        elif config.label_transform == 'log1p':
            # For y=log(1+z), dy = 1/(1+z)*dz => dz = (1+z)*dy. 1+z = exp(y).
            # So, sigma_z approx exp(y_pred) * sigma_y
            train_sigmas = np.exp(train_preds) * train_sigmas
            val_sigmas = np.exp(val_preds) * val_sigmas
            test_sigmas = np.exp(test_preds) * test_sigmas
    
    print("Inverse transformation completed")

print(f"Inference completed")
print(f"  Trainset predictions range: [{train_preds.min():.4f}, {train_preds.max():.4f}]")
print(f"  Trainset labels range: [{train_labels.min():.4f}, {train_labels.max():.4f}]")
print(f"  Valset predictions range: [{val_preds.min():.4f}, {val_preds.max():.4f}]")
print(f"  Valset labels range: [{val_labels.min():.4f}, {val_labels.max():.4f}]")
print(f"  Testset predictions range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print(f"  Testset labels range: [{test_labels.min():.4f}, {test_labels.max():.4f}]")

prediction_train_df = pd.DataFrame({'label': train_labels, 'pred': train_preds})
if train_sigmas is not None:
    prediction_train_df['pred_sigma'] = train_sigmas
path = f"{experiment_dir}/predictions_train.fits"
cu.savefile(prediction_train_df, path)

prediction_val_df = pd.DataFrame({'label': val_labels, 'pred': val_preds})
if val_sigmas is not None:
    prediction_val_df['pred_sigma'] = val_sigmas
path = f"{experiment_dir}/predictions_val.fits"
cu.savefile(prediction_val_df, path)

prediction_test_df = pd.DataFrame({'label': test_labels, 'pred': test_preds})
if test_sigmas is not None:
    prediction_test_df['pred_sigma'] = test_sigmas
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




# %% save datasets
# deltachi2_threshold = 100
# experiment_dir = f"./0Sqrt_UseMag_chi2>{deltachi2_threshold}_NoColorErr"
# config_path = f"{experiment_dir}/config.yaml"
# model_path = f"{experiment_dir}/best_model.pkl"
# scaler_path = f"{experiment_dir}/scaler.pkl"

# path = f'../data/ps1dr2_loc_xDESIDR1_xSDSSDR18_phot_clean_chi2>{deltachi2_threshold}.fits'
# all_dataset = cu.readfile(path)

# config = Config()
# config.update_from_yaml(config_path)
# print(f"Config loaded from: {config_path}")

# ratios = config.dataset_mode.get('ratios')

# indices = list(range(len(all_dataset)))
# val_test_ratio = ratios['val'] + ratios['test']
# train_indices, temp_indices = train_test_split(
#     indices, test_size=val_test_ratio, 
#     random_state=config.random_seed
# )
# val_size_corrected = ratios['val'] / val_test_ratio
# val_indices, test_indices = train_test_split(
#     temp_indices, test_size=(1 - val_size_corrected), 
#     random_state=config.random_seed
# )

# train_df = cu.readfile(f"{experiment_dir}/predictions_train.fits")
# val_df = cu.readfile(f"{experiment_dir}/predictions_val.fits")
# test_df = cu.readfile(f"{experiment_dir}/predictions_test.fits")

# train_preds = train_df['pred'].values
# val_preds = val_df['pred'].values
# test_preds = test_df['pred'].values

# trainset = all_dataset.loc[train_indices]
# trainset['z_pred'] = train_preds
# cu.savefile(trainset, f'./{experiment_dir}/trainset.fits')

# valset = all_dataset.loc[val_indices]
# valset['z_pred'] = val_preds
# cu.savefile(valset, f'./{experiment_dir}/valset.fits')

# testset = all_dataset.loc[test_indices]
# testset['z_pred'] = test_preds
# cu.savefile(testset, f'./{experiment_dir}/testset.fits')


"""LSDR10 catalog cleaning recipes.

This file holds the cleaning routines applied to DESI Legacy Surveys
DR10 (+ Gaia + PS1DR2 cross-match) tile catalogs. There are two
distinct recipes that target different downstream uses:

* **Release cleaning** (:func:`clean_for_release`, implemented below) —
  the cleaning applied to the public catalog shipped with the paper.
  Strict quality cuts on fracflux, proper motion, star rejection,
  magnitude validity, and SNR thresholds.

* **Training cleaning** (:func:`clean_for_training`) — the cleaning
  used to build the model training set. Tighter SNR thresholds and
  stricter fracflux / fracmasked / fracin cuts, plus computes
  adjacent-band colors as model features. Lives next to the release
  recipe so the two can be read and diffed in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.table import Table

__all__ = ["clean_for_release", "clean_for_training"]


def clean_for_release(t: Table, snr_threshold: int = 1) -> tuple[Table, dict]:
    """Cleaning pipeline for the public LSDR10 release catalog.

    Input is a per-tile astropy Table from the LSDR10 × Gaia × PS1DR2
    cross-match. Each step prints how many sources are removed so that
    batch runs are easy to monitor.

    Cleaning steps (in order):

    1. ``fracflux``: reject sources with ``fracflux >= 1`` in any of
       g, r, z (flux contamination from neighbors).
    2. Proper motion: reject sources that have a Gaia match **and**
       proper-motion SNR >= 3 (likely stars).
    3. PS1 star rejection: reject sources with a PS1 match where
       ``iPSFMag < 21`` and ``iPSF - iKron <= 0.05`` (point-source
       morphology in PS1).
    4a. Magnitude validity: require g, r, z, W1, W2 dereddened mags in
        ``(0, 30)``.
    4b. ``dered_mag_i`` outside ``(0, 30)`` is set to ``NaN`` (row
        kept — i-band coverage is incomplete).
    4c. ``snr_i <= 0`` or non-finite is set to ``NaN`` (row kept).
    5. SNR: require ``snr > threshold`` in g, r, z, W1, W2.
    6. i-band SNR: sources with ``snr_i = NaN`` (no observation) are
       kept; sources with a valid ``snr_i`` must exceed ``threshold``.
    7. Compute magnitude errors from SNR (``1.0857 / snr``) for all
       six bands and replace any resulting ``inf`` with ``NaN``.

    Args:
        t: Per-tile astropy Table. Not modified; cleaning runs on a
            copy.
        snr_threshold: Minimum signal-to-noise ratio for the mandatory
            bands. Defaults to ``1``.

    Returns:
        Tuple of ``(t_clean, stats)`` where ``t_clean`` is the cleaned
        Table and ``stats`` is a dict with ``n_total`` / ``n_final``
        counts.
    """
    n_total = len(t)
    print(f"  Initial sources: {n_total:,}")

    # Step 1 — fracflux: reject sources with neighbor contamination.
    mask_fracflux = (t['fracflux_g'] < 1) & (t['fracflux_r'] < 1) & (t['fracflux_z'] < 1)
    n_removed = np.sum(~mask_fracflux)
    t_clean = t[mask_fracflux]
    print(f"  Step 1 - fracflux (grz < 1): removed {n_removed:,}, remaining {len(t_clean):,}")

    # Step 2 — proper motion: reject Gaia-matched sources with high pm SNR.
    has_gaia = ~np.isnan(t_clean['pmra'])
    pm_snr = np.sqrt(t_clean['pmra']**2 * t_clean['pmra_ivar'] + t_clean['pmdec']**2 * t_clean['pmdec_ivar'])
    mask_pm = ~has_gaia | (pm_snr < 3)
    n_removed = np.sum(~mask_pm)
    t_clean = t_clean[mask_pm]
    print(f"  Step 2 - proper motion (pm_snr < 3): removed {n_removed:,}, remaining {len(t_clean):,}")

    # Step 3 — PS1 star rejection via iPSF - iKron morphology cut.
    has_ps = ~np.isnan(t_clean['objID'])
    i_bright = t_clean['iPSFMag_dered'] < 21
    psf_kron_diff = t_clean['iPSFMag_dered'] - t_clean['iKronMag_dered']
    mask_star = ~has_ps | ~i_bright | (psf_kron_diff > 0.05)
    n_removed = np.sum(~mask_star)
    t_clean = t_clean[mask_star]
    print(f"  Step 3 - PS1 star rejection (iPSF-iKron <= 0.05): removed {n_removed:,}, remaining {len(t_clean):,}")

    # Step 4a — magnitude validity for mandatory bands (grzW1W2).
    mag_cols_required = ['dered_mag_g', 'dered_mag_r', 'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']
    mask_mag = np.ones(len(t_clean), dtype=bool)
    for col in mag_cols_required:
        mask_mag &= (t_clean[col] > 0) & (t_clean[col] < 30) & np.isfinite(t_clean[col])
    n_removed = np.sum(~mask_mag)
    t_clean = t_clean[mask_mag]
    print(f"  Step 4a - mag validity (grzW1W2 in 0-30): removed {n_removed:,}, remaining {len(t_clean):,}")

    # Step 4b — i-band mag: set invalid values to NaN (keep row).
    valid_mag_i = (t_clean['dered_mag_i'] > 0) & (t_clean['dered_mag_i'] < 30) & np.isfinite(t_clean['dered_mag_i'])
    n_bad_mag_i = np.sum(~valid_mag_i)
    t_clean['dered_mag_i'][~valid_mag_i] = np.nan
    print(f"  Step 4b - dered_mag_i cleanup (invalid -> NaN): {n_bad_mag_i:,} set to NaN")

    # Step 4c — i-band SNR: set invalid values to NaN (keep row).
    valid_snr_i = (t_clean['snr_i'] > 0) & np.isfinite(t_clean['snr_i'])
    n_bad_snr_i = np.sum(~valid_snr_i)
    t_clean['snr_i'][~valid_snr_i] = np.nan
    print(f"  Step 4c - snr_i cleanup (<=0 or non-finite -> NaN): {n_bad_snr_i:,} set to NaN")

    # Step 5 — SNR threshold for mandatory bands (grzW1W2).
    mask_snr = (
        (t_clean['snr_g'] > snr_threshold) & (t_clean['snr_r'] > snr_threshold) &
        (t_clean['snr_z'] > snr_threshold) & (t_clean['snr_w1'] > snr_threshold) &
        (t_clean['snr_w2'] > snr_threshold)
    )
    n_removed = np.sum(~mask_snr)
    t_clean = t_clean[mask_snr]
    print(f"  Step 5 - SNR (grzW1W2 > {snr_threshold}): removed {n_removed:,}, remaining {len(t_clean):,}")

    # Step 6 — i-band SNR: keep NaN (no observation); require > threshold
    # for sources with a valid measurement.
    mask_i = np.isnan(t_clean['snr_i']) | (t_clean['snr_i'] > snr_threshold)
    n_removed = np.sum(~mask_i)
    t_clean = t_clean[mask_i]
    print(f"  Step 6 - i-band (NaN kept or snr_i>{snr_threshold}): removed {n_removed:,}, remaining {len(t_clean):,}")

    # Compute magnitude errors from SNR: mag_err = 1.0857 / snr.
    for band in ['g', 'r', 'i', 'z', 'w1', 'w2']:
        snr_col = f'snr_{band}'
        err_col = f'mag_{band}_err'
        snr = t_clean[snr_col].copy()
        snr[snr <= 0] = np.nan
        t_clean[err_col] = 1.0857 / snr

    # Replace any inf with NaN in magnitude and error columns.
    check_cols = [f'dered_mag_{b}' for b in ['g', 'r', 'i', 'z', 'w1', 'w2']] + \
                 [f'mag_{b}_err' for b in ['g', 'r', 'i', 'z', 'w1', 'w2']]
    for col in check_cols:
        if col in t_clean.colnames:
            data = t_clean[col]
            n_inf = np.sum(np.isinf(data))
            if n_inf > 0:
                print(f"  WARNING: {col} has {n_inf} inf values, replaced with NaN")
                t_clean[col][np.isinf(data)] = np.nan

    n_final = len(t_clean)
    print(f"  Total: {n_total:,} -> {n_final:,} ({n_final/n_total*100:.2f}%), removed {n_total - n_final:,}")

    stats = {
        'n_total': n_total,
        'n_final': n_final,
    }

    return t_clean, stats


# ---------------------------------------------------------------------------
# Training cleaning
# ---------------------------------------------------------------------------


def clean_for_training(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for the LSDR10 photo-z training sample.

    Input is expected to be an LSDR10 catalog already cross-matched with
    a spectroscopic redshift source (e.g. DESI DR1 × SDSS DR19) and
    carrying dereddened magnitudes and SNR columns. Relative to
    :func:`clean_for_release`, this recipe uses tighter SNR thresholds
    (> 5), stricter fracflux / fracmasked / fracin cuts, and appends
    adjacent-band colors and magnitude errors as model features.

    This is the *training* recipe, not the release recipe — see the
    module docstring for the distinction.

    Args:
        df_raw: LSDR10 × spec-z catalog as a DataFrame. Must contain
            dereddened magnitudes (``dered_mag_*``), SNR columns
            (``snr_*``), and flux-quality columns (``fracflux_*``,
            ``fracmasked_*``, ``fracin_*``). Not modified; cleaning
            runs on a copy.

    Returns:
        Cleaned DataFrame with magnitude-error and adjacent-band color
        columns appended. Per-step source counts are printed to stdout.
    """
    df = df_raw.copy()

    # Compute magnitude errors from SNR: mag_err = 1.0857 / snr.
    for band in ['g', 'r', 'i', 'z', 'w1', 'w2']:
        df[f'mag_{band}_Err'] = 1.0857 / df[f'snr_{band}']

    # Adjacent-band colors.
    df['g_r'] = df['dered_mag_g'] - df['dered_mag_r']
    df['r_i'] = df['dered_mag_r'] - df['dered_mag_i']
    df['i_z'] = df['dered_mag_i'] - df['dered_mag_z']
    df['z_w1'] = df['dered_mag_z'] - df['dered_mag_w1']
    df['w1_w2'] = df['dered_mag_w1'] - df['dered_mag_w2']

    # Require mandatory-band magnitudes > 0 (i-band excluded here
    # because its coverage is incomplete).
    mask = df[['dered_mag_g', 'dered_mag_r',
               'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']].gt(0).all(axis=1)
    df = df[mask]
    print('After mag > 0:', len(df))

    # Require mandatory-band magnitudes < 30.
    mask = df[['dered_mag_g', 'dered_mag_r',
               'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2']].lt(30).all(axis=1)
    df = df[mask]
    print('After mag < 30:', len(df))

    # Flux-quality cuts: fracflux, fracmasked, fracin for grz bands.
    idx = (df['fracflux_g'] < 0.5) & (df['fracflux_r'] < 0.5) & (df['fracflux_z'] < 0.5)
    idx &= (df['fracmasked_g'] < 0.4) & (df['fracmasked_r'] < 0.4) & (df['fracmasked_z'] < 0.4)
    idx &= (df['fracin_g'] > 0.3) & (df['fracin_r'] > 0.3) & (df['fracin_z'] > 0.3)
    df = df[idx]
    print('After clean FRAC:', len(df))

    # SNR > 5 for mandatory bands (grzW1W2).
    idx = (df.snr_g > 5) & (df.snr_r > 5) & (df.snr_z > 5)
    idx &= (df.snr_w1 > 5) & (df.snr_w2 > 5)
    df = df[idx]
    print('After clean snr:', len(df))

    return df

"""PS1DR2 catalog cleaning recipes.

This file holds the cleaning routines applied to PS1DR2 tile catalogs.
There are two distinct recipes that target different downstream uses:

* **Release cleaning** (:func:`clean_for_release`, implemented below) —
  the cleaning applied to the public catalog shipped with the paper.
  Strict quality cuts, star rejection, and projection onto a fixed
  column schema so the release product is predictable.

* **Training cleaning** (:func:`clean_for_training`) — the cleaning
  used to build the model training set. Tighter per-band magnitude-
  error thresholds, a bright-end (PSF > Kron) star cut applied to
  every band, redshift-range selection on the spec-z label, and
  appends adjacent-band color columns used as model features. Lives
  next to the release recipe so the two can be read and diffed in one
  place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["clean_for_release", "clean_for_training"]


def clean_for_release(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for the public PS1DR2 release catalog.

    Keeps high-quality, non-stellar sources and projects onto the fixed
    release schema. The input is expected to already carry dereddened
    PS1 grizy magnitudes (``<band><mag_type>_dered`` from
    :mod:`surveys.PS1DR2.dered`) and the xunWISE W1/W2 photometry from
    the cross-match step.

    This is the *release* recipe, not the training-set recipe — see the
    module docstring for the distinction.

    Args:
        df_raw: PS1DR2 tile DataFrame. Not modified; cleaning runs on a
            copy.

    Returns:
        Cleaned DataFrame projected onto the release column list, with
        a fresh ``RangeIndex``. Per-tile yield is printed to stdout as
        ``Clean: <before> -> <after> (<fraction>%)``.
    """
    df = df_raw.copy()
    n_orig = len(df)

    # 0. Flatten any 2-D columns some FITS readers return as (N, 1).
    for col in df.columns:
        if len(df[col].shape) > 1:
            df[col] = df[col][:, 0]

    # Replace PS1 sentinel values with NaN so dropna() works uniformly.
    invalid_vals = [-99.0, -99, -999.0, -9999.0, -999, -9999]
    df = df.replace(invalid_vals, np.nan)

    # Require all core columns to be valid: grizy PSF / Kron / Ap
    # magnitudes and their errors.
    core_mag_cols = []
    core_err_cols = []
    for b in ['g', 'r', 'i', 'z', 'y']:
        core_mag_cols.extend([f'{b}PSFMag_dered', f'{b}KronMag_dered', f'{b}ApMag_dered'])
        core_err_cols.extend([f'{b}PSFMagErr', f'{b}KronMagErr', f'{b}ApMagErr'])

    existing_mag_cols = [c for c in core_mag_cols if c in df.columns]
    df = df.dropna(subset=existing_mag_cols)

    existing_err_cols = [c for c in core_err_cols if c in df.columns]
    df = df.dropna(subset=existing_err_cols)

    # 1. nDetections >= 1 (at least one valid detection).
    df = df[np.array(df['nDetections']).flatten() >= 1]

    # ============ objInfoFlag filter ============
    # Reject blended / bad-astrometry / poor-quality / masked sources.
    BAD_OBJ_MASK = 0x00000020 | 0x00000040 | 0x00080000 | 0x00100000
    flags = np.nan_to_num(np.array(df['objInfoFlag']).flatten().astype(np.int64), nan=0)
    df = df[(flags & BAD_OBJ_MASK) == 0]

    # ============ qualityFlag filter ============
    BAD_QUALITY_MASK = 0x00000040 | 0x00000080
    flags = np.nan_to_num(np.array(df['qualityFlag']).flatten().astype(np.int64), nan=0)
    df = df[(flags & BAD_QUALITY_MASK) == 0]

    # ============ infoFlag filter (per band) ============
    # STAR_MASK: pipeline-flagged likely star.
    # CONTAMINATION_MASK: diffraction spikes, ghosts, saturation, poor fit, etc.
    STAR_MASK = 0x400000
    CONTAMINATION_MASK = 0x8 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x10000
    EXCLUDE_MASK = STAR_MASK | CONTAMINATION_MASK

    mask = np.ones(len(df), dtype=bool)
    for b in ['g', 'r', 'i', 'z', 'y']:
        col = f'{b}infoFlag'
        if col in df.columns:
            flags = np.nan_to_num(np.array(df[col]).flatten().astype(np.int64), nan=0)
            mask &= (flags & EXCLUDE_MASK) == 0
    df = df[mask]

    # ============ Star rejection (PSF - Kron magnitude difference) ============
    # Only applied at i < 21 where the PSF/Kron separation is reliable;
    # fainter sources are kept regardless.
    if 'iPSFMag_dered' in df.columns and 'iKronMag_dered' in df.columns:
        i_psf = df['iPSFMag_dered'].values
        i_kron = df['iKronMag_dered'].values
        psf_kron_diff = i_psf - i_kron
        is_bright = i_psf < 21
        is_star = psf_kron_diff <= 0.05
        df = df[~is_bright | ~is_star]

    # Rename position columns to the release schema.
    df = df.rename(columns={'raStack': 'ra', 'decStack': 'dec'})

    # Release column list — columns not present are silently dropped.
    cols_to_keep = [
        'objID', 'uid', 'ra', 'dec',
        'gKronMag_dered', 'rKronMag_dered', 'iKronMag_dered', 'zKronMag_dered', 'yKronMag_dered',
        'gPSFMag_dered', 'rPSFMag_dered', 'iPSFMag_dered', 'zPSFMag_dered', 'yPSFMag_dered',
        'gApMag_dered', 'rApMag_dered', 'iApMag_dered', 'zApMag_dered', 'yApMag_dered',
        'gKronMagErr', 'rKronMagErr', 'iKronMagErr', 'zKronMagErr', 'yKronMagErr',
        'gPSFMagErr', 'rPSFMagErr', 'iPSFMagErr', 'zPSFMagErr', 'yPSFMagErr',
        'gApMagErr', 'rApMagErr', 'iApMagErr', 'zApMagErr', 'yApMagErr',
        'mag_w1', 'mag_w2', 'mag_w1_err', 'mag_w2_err',
    ]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df_clean = df[cols_to_keep].reset_index(drop=True)

    print(f'Clean: {n_orig:,} -> {len(df_clean):,} ({len(df_clean)/n_orig*100:.2f}%)')

    return df_clean


# ---------------------------------------------------------------------------
# Training cleaning
# ---------------------------------------------------------------------------


def clean_for_training(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for the PS1DR2 photo-z training sample.

    Input is expected to be a PS1DR2 catalog already cross-matched with
    a spectroscopic redshift source (e.g. DESI DR1 × SDSS DR19) and
    carrying dereddened grizy magnitudes. Relative to
    :func:`clean_for_release`, this recipe swaps the object-level
    quality bitmask filters for tighter per-band magnitude-error
    thresholds, enforces the (PSF - Kron) > 0.1 star cut on every
    band, applies the spec-z range of interest, and finally appends
    adjacent-band color columns used as features by the photo-z
    model.

    This is the *training* recipe, not the release recipe — see the
    module docstring for the distinction.

    Args:
        df_raw: PS1DR2 × spec-z catalog. Must contain grizy PSF / Kron
            / Ap dereddened magnitudes and their errors, plus ``z`` and
            ``zErr`` columns. Not modified; cleaning runs on a copy.

    Returns:
        Cleaned DataFrame with adjacent-band color columns appended
        (``PSF_gr``, ``PSF_ri``, ``PSF_iz``, ``PSF_zy`` and the same
        for ``Ap`` / ``Kron``). Per-step source counts are printed to
        stdout as ``<step>: <count>``; the final line is the count of
        rows with any NaN across the feature columns.
    """
    df = df_raw.copy()

    # Require all bands' magnitudes and errors to be positive.
    bands = ['g', 'r', 'i', 'z', 'y']
    mag_types = ['KronMag_dered', 'PSFMag_dered', 'ApMag_dered',
                 'KronMagErr', 'PSFMagErr', 'ApMagErr']

    idx = pd.Series(True, index=df.index)
    for band in bands:
        for mag_type in mag_types:
            col = f'{band}{mag_type}'
            idx &= df[col] > 0
    df = df[idx]
    print(f'1: {len(df)}')

    # Per-band Kron magnitude-error thresholds.
    err_thresholds = {'g': 0.2, 'r': 0.1, 'i': 0.05, 'z': 0.1, 'y': 0.2}
    idx = pd.Series(True, index=df.index)
    for band, threshold in err_thresholds.items():
        idx &= df[f'{band}KronMagErr'] < threshold
    df = df[idx]
    print(f'2: {len(df)}')

    # # Per-band PSF magnitude-error thresholds (disabled).
    # err_thresholds = {'g': 0.1, 'r': 0.04, 'i': 0.03, 'z': 0.04, 'y': 0.1}
    # idx = pd.Series(True, index=df.index)
    # for band, threshold in err_thresholds.items():
    #     idx &= df[f'{band}PSFMagErr'] < threshold
    # df = df[idx]
    # print('3: ', len(df))

    # Per-band aperture magnitude-error thresholds.
    err_thresholds = {'g': 0.02, 'r': 0.01, 'i': 0.006, 'z': 0.006, 'y': 0.006}
    idx = pd.Series(True, index=df.index)
    for band, threshold in err_thresholds.items():
        idx &= df[f'{band}ApMagErr'] < threshold
    df = df[idx]
    print(f'4: {len(df)}')

    # Reject likely point sources: for stars, PSF and Kron magnitudes
    # are close; require a clear PSF > Kron separation in every band.
    idx = pd.Series(True, index=df.index)
    for band in bands:
        idx &= df[f'{band}PSFMag_dered'] - df[f'{band}KronMag_dered'] > 0.1
    df = df[idx]
    print(f'5: {len(df)}')

    # Redshift range selection for the training label.
    idx = df['zErr'] < 0.001
    idx &= df['z'] > 0.01
    idx &= df['z'] < 1.2
    df = df[idx]
    print(f'7: {len(df)}')

    # Adjacent-band colors, one set per magnitude type. These columns
    # feed the photo-z model as features.
    band_pairs = [('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y')]
    for mag_type in ['PSF', 'Ap', 'Kron']:
        for b1, b2 in band_pairs:
            df[f'{mag_type}_{b1}{b2}'] = (
                df[f'{b1}{mag_type}Mag_dered'] - df[f'{b2}{mag_type}Mag_dered']
            )

    # Diagnostic: how many rows have any NaN across the feature columns.
    cols = [
        'gPSFMag_dered', 'rPSFMag_dered', 'iPSFMag_dered', 'zPSFMag_dered', 'yPSFMag_dered',
        'gApMag_dered', 'rApMag_dered', 'iApMag_dered', 'zApMag_dered', 'yApMag_dered',
        'gKronMag_dered', 'rKronMag_dered', 'iKronMag_dered', 'zKronMag_dered', 'yKronMag_dered',
        'PSF_gr', 'PSF_ri', 'PSF_iz', 'PSF_zy',
        'Ap_gr', 'Ap_ri', 'Ap_iz', 'Ap_zy',
        'Kron_gr', 'Kron_ri', 'Kron_iz', 'Kron_zy',
    ]
    print(df[cols].isna().any(axis=1).sum())

    return df

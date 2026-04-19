"""unWISE DR1 catalog cleaning recipes.

This file holds the cleaning routines applied to unWISE DR1 tile
catalogs. Rejects sources with invalid magnitudes / flux errors, and
sources flagged by any ``flags_unwise`` or ``flags_info`` bit
(bright-star artifacts, contamination, etc.).

Cleaning criteria (in order):

1. Magnitude validity: require ``mag_w1_vg > 0``, ``mag_w2_vg > 0``,
   and both finite (not ``inf``/``NaN``).
2. Flux error validity: require ``dflux_w1 > 0`` and ``dflux_w2 > 0``.
3. ``flags_unwise`` filter: require ``flags_unwise_w1 == 0`` and
   ``flags_unwise_w2 == 0`` (no bright-star artifacts in either band).
4. ``flags_info`` filter: require ``flags_info_w1 == 0`` and
   ``flags_info_w2 == 0`` (no contamination in either band).

flags_unwise bit definitions (all bits mark bright-star artifacts)::

    bit 0: In core or wings
    bit 1: In diffraction spike
    bit 2: In ghost
    bit 3: In first latent
    bit 4: In second latent
    bit 5: In circular halo
    bit 6: Saturated
    bit 7: In geometric diffraction spike

flags_info bit definitions::

    bit 0: In PSF of bright star falling off coadd
    bit 1: In HyperLeda large galaxy
    bit 2: In big object (e.g., Magellanic cloud)
    bit 3: May contain centroid of very bright star
    bit 4: Potentially affected by saturation
    bit 5: May contain nebulosity
    bit 6: Will not be aggressively deblended
    bit 7: Must be sharp to be optimized
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["clean"]


def filter_by_mag(df: pd.DataFrame) -> pd.DataFrame:
    """Keep sources with valid W1/W2 magnitudes and flux errors.

    Requires ``mag_w1_vg > 0``, ``mag_w2_vg > 0``, both finite, and
    ``dflux_w1 > 0``, ``dflux_w2 > 0``.
    """
    mask = np.ones(len(df), dtype=bool)

    # Magnitudes must be positive and finite.
    if 'mag_w1_vg' in df.columns and 'mag_w2_vg' in df.columns:
        mask &= (df['mag_w1_vg'] > 0) & (df['mag_w2_vg'] > 0)
        mask &= np.isfinite(df['mag_w1_vg']) & np.isfinite(df['mag_w2_vg'])

    # Flux errors must be positive.
    if 'dflux_w1' in df.columns and 'dflux_w2' in df.columns:
        mask &= (df['dflux_w1'] > 0) & (df['dflux_w2'] > 0)

    return df[mask].reset_index(drop=True)


def filter_by_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Reject sources with any flags_unwise or flags_info bit set.

    ``flags_unwise == 0`` means no bright-star artifacts in either band.
    ``flags_info == 0`` means no contamination in either band.
    """
    mask_flags = (
        (df['flags_unwise_w1'] == 0) &
        (df['flags_unwise_w2'] == 0)
    )

    mask_info = (
        (df['flags_info_w1'] == 0) &
        (df['flags_info_w2'] == 0)
    )

    return df[mask_flags & mask_info].reset_index(drop=True)


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for unWISE DR1 catalogs.

    Applies magnitude/flux-error validity checks and flag filters in
    sequence. See the module docstring for the full cleaning criteria
    and flag bit definitions.

    Args:
        df_raw: unWISE DR1 tile DataFrame. Not modified; cleaning runs
            on a copy.

    Returns:
        Cleaned DataFrame with a fresh ``RangeIndex``. Per-tile yield
        is printed to stdout as
        ``Clean: <before> -> <after> (<fraction>%)``.
    """
    df = df_raw.copy()
    n_orig = len(df)

    df = filter_by_mag(df)
    df = filter_by_flags(df)

    print(f'Clean: {n_orig:,} -> {len(df):,} ({len(df)/n_orig*100:.1f}%)')

    return df

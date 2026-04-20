"""Utilities to clean and merge DESI DR1 spectroscopic catalogs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.io import readfile, savefile

RAW_COLUMNS = [
    "TARGETID",
    "RA",
    "DEC",
    "Z",
    "ZERR",
    "ZWARN",
    "SPECTYPE",
    "DELTACHI2",
    "MASKBITS",
    "DESI_TARGET",
    "BGS_TARGET",
    "PRIORITY",
    "MORPHTYPE",
    "FLUX_G",
    "FLUX_R",
    "FLUX_Z",
    "FLUX_W1",
    "FLUX_W2",
    "FLUX_IVAR_G",
    "FLUX_IVAR_R",
    "FLUX_IVAR_Z",
    "FLUX_IVAR_W1",
    "FLUX_IVAR_W2",
]

OUTPUT_COLUMNS = [
    "TARGETID",
    "RA",
    "DEC",
    "Z",
    "ZERR",
    "DELTACHI2",
    "DESI_TARGET",
    "BGS_TARGET",
    "PRIORITY",
]


def clean_bgs(
    input_path: str | Path = "BGS_ANY_full_noveto.dat.fits",
    output_path: str | Path = "TODO_BGS_CLEAN_OUTPUT.fits",
) -> pd.DataFrame:
    """Clean the DESI BGS catalog and save the filtered table."""
    df = readfile(input_path)[RAW_COLUMNS]

    # Keep galaxy targets with reliable redshifts and valid photometry.
    mask = df["SPECTYPE"] == "GALAXY"
    mask &= df["ZWARN"] == 0
    mask &= df["MASKBITS"] == 0
    mask &= df["MORPHTYPE"] != "PSF"
    mask &= (df["DESI_TARGET"] & 2**60) != 0
    mask &= (df["FLUX_G"] > 0) & (df["FLUX_R"] > 0) & (df["FLUX_Z"] > 0)
    mask &= (df["FLUX_W1"] > 0) & (df["FLUX_W2"] > 0)
    mask &= (df["FLUX_IVAR_G"] > 0) & (df["FLUX_IVAR_R"] > 0)
    mask &= (df["FLUX_IVAR_Z"] > 0) & (df["FLUX_IVAR_W1"] > 0)
    mask &= df["FLUX_IVAR_W2"] > 0
    mask &= df["DELTACHI2"] > 25

    df_clean = df.loc[mask, OUTPUT_COLUMNS].copy()
    savefile(df_clean, output_path)
    return df_clean


def clean_elg(
    input_path: str | Path = "ELG_LOPnotqso_full_noveto.dat.fits",
    output_path: str | Path = "TODO_ELG_CLEAN_OUTPUT.fits",
) -> pd.DataFrame:
    """Clean the DESI ELG catalog and save the filtered table."""
    df = readfile(input_path)[RAW_COLUMNS]

    # Keep galaxy targets with reliable redshifts and valid photometry.
    mask = df["SPECTYPE"] == "GALAXY"
    mask &= df["ZWARN"] == 0
    mask &= df["MASKBITS"] == 0
    mask &= df["MORPHTYPE"] != "PSF"
    mask &= (df["DESI_TARGET"] & 2**1) != 0
    mask &= (df["FLUX_G"] > 0) & (df["FLUX_R"] > 0) & (df["FLUX_Z"] > 0)
    mask &= (df["FLUX_W1"] > 0) & (df["FLUX_W2"] > 0)
    mask &= (df["FLUX_IVAR_G"] > 0) & (df["FLUX_IVAR_R"] > 0)
    mask &= (df["FLUX_IVAR_Z"] > 0) & (df["FLUX_IVAR_W1"] > 0)
    mask &= df["FLUX_IVAR_W2"] > 0
    mask &= df["DELTACHI2"] > 25

    df_clean = df.loc[mask, OUTPUT_COLUMNS].copy()
    savefile(df_clean, output_path)
    return df_clean


def clean_lrg(
    input_path: str | Path = "LRG_full_noveto.dat.fits",
    output_path: str | Path = "TODO_LRG_CLEAN_OUTPUT.fits",
) -> pd.DataFrame:
    """Clean the DESI LRG catalog and save the filtered table."""
    df = readfile(input_path)[RAW_COLUMNS]

    # Keep galaxy targets with reliable redshifts and valid photometry.
    mask = df["SPECTYPE"] == "GALAXY"
    mask &= df["ZWARN"] == 0
    mask &= df["MASKBITS"] == 0
    mask &= df["MORPHTYPE"] != "PSF"
    mask &= (df["DESI_TARGET"] & 2**0) != 0
    mask &= (df["FLUX_G"] > 0) & (df["FLUX_R"] > 0) & (df["FLUX_Z"] > 0)
    mask &= (df["FLUX_W1"] > 0) & (df["FLUX_W2"] > 0)
    mask &= (df["FLUX_IVAR_G"] > 0) & (df["FLUX_IVAR_R"] > 0)
    mask &= (df["FLUX_IVAR_Z"] > 0) & (df["FLUX_IVAR_W1"] > 0)
    mask &= df["FLUX_IVAR_W2"] > 0
    mask &= df["DELTACHI2"] > 25

    df_clean = df.loc[mask, OUTPUT_COLUMNS].copy()
    savefile(df_clean, output_path)
    return df_clean


def merge_clean_catalogs(
    bgs_path: str | Path = "TODO_BGS_CLEAN_PATH.fits",
    elg_path: str | Path = "TODO_ELG_CLEAN_PATH.fits",
    lrg_path: str | Path = "TODO_LRG_CLEAN_PATH.fits",
    output_path: str | Path = "TODO_DESIDR1_OUTPUT.fits",
) -> pd.DataFrame:
    """Merge cleaned DESI catalogs, prefix columns, and save the table."""
    bgs = readfile(bgs_path)
    elg = readfile(elg_path)
    lrg = readfile(lrg_path)

    df_merged = pd.concat([bgs, lrg, elg], axis=0, ignore_index=True)
    df_merged.columns = [f"DESI_{col}" for col in df_merged.columns]

    savefile(df_merged, output_path)
    return df_merged

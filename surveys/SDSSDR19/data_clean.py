"""Utilities to clean the SDSS DR19 spectroscopic catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.io import readfile, savefile


def clean_sdss_dr19(
    input_path: str | Path = "TODO_SDSSDR19_INPUT.csv",
    output_path: str | Path = "TODO_SDSSDR19_OUTPUT.fits",
) -> pd.DataFrame:
    """Clean the SDSS DR19 spectroscopic catalog and save the result."""
    df_raw = readfile(input_path)
    df = df_raw.copy()

    # Apply the BOSS/eBOSS quality cuts from the original notebook.
    idx = (df["survey"] == "boss") | (df["survey"] == "eboss")
    mask_boss_eboss = df["z_noqso"] > 0.0
    mask_boss_eboss &= df["zErr_noqso"] > 0
    mask_boss_eboss &= df["zwarning_noqso"] == 0
    mask_boss_eboss &= df["class_noqso"] == "GALAXY"
    df_boss_eboss = df.loc[idx & mask_boss_eboss]

    # Apply the standard quality cuts to the remaining surveys.
    mask_others = df["z"] > 0.0
    mask_others &= df["zErr"] > 0
    mask_others &= df["zwarning"] == 0
    mask_others &= df["class"] == "GALAXY"
    df_others = df.loc[~idx & mask_others]

    df_clean = pd.concat([df_boss_eboss, df_others], axis=0)
    df_clean = df_clean[["specobjid", "ra", "dec", "z", "zErr"]].copy()
    df_clean.columns = [f"SDSS_{col}" for col in df_clean.columns]

    savefile(df_clean, output_path)
    return df_clean

"""Shared helpers for the catalog pipeline.

Loads ``config.yaml`` once at import time (with ``${dotted.key}``
interpolation) and exposes:

* Derived constants: :data:`PDF_BINS`, :data:`PDF_BIN_EDGES`,
  :data:`PDF_UINT16_SCALE`, :data:`N_RA_BINS`,
  :data:`MERGE_RA_STEP`, :data:`PUBLISH_RA_STEP`,
  :data:`CATALOG_MAIN_COLUMNS`, :data:`CATALOG_SCALAR_COLUMNS`,
  :data:`CATALOG_PDF_COLUMNS`.
* Table I/O: :func:`read_table` / :func:`save_table` dispatch on
  extension, using :mod:`lib.io` for FITS and a project-specific
  ``photoz_table_v2`` HDF5 format (:func:`save_hdf5` /
  :func:`read_hdf5`) for anything carrying the quantised PDF.
* Photo-z statistics on GPU:
  :func:`compute_photoz_stats_torch`, :func:`resample_bins`,
  :func:`build_resample_matrix`.

The HDF5 format intentionally stays inside this package because it
stores the PDF as a ``uint16`` matrix with renormalised row sums and
a dedicated ``meta`` group — this can't round-trip through the
generic pandas ``to_hdf`` path used by :mod:`lib.io`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from lib.io import readfile, savefile

__all__ = [
    "load_config",
    "PDF_UINT16_SCALE",
    "PDF_BINS",
    "PDF_BIN_EDGES",
    "N_RA_BINS",
    "MERGE_RA_STEP",
    "PUBLISH_RA_STEP",
    "CATALOG_MAIN_COLUMNS",
    "CATALOG_SCALAR_COLUMNS",
    "CATALOG_PDF_COLUMNS",
    "read_table",
    "save_table",
    "save_fits",
    "encode_pdf_to_uint16",
    "save_hdf5",
    "read_hdf5",
    "is_valid_prediction_hdf5",
    "compute_photoz_stats_torch",
    "resample_bins",
    "build_resample_matrix",
]


# ---------------------------------------------------------------------------
# Config loader with ${dotted.key} interpolation
# ---------------------------------------------------------------------------


_CONFIG_CACHE: dict[str, dict] = {}
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_INTERP_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _lookup_dotted(dotted: str, root: dict) -> Any:
    """Resolve ``a.b.c`` against ``root``; raise ``KeyError`` if missing."""
    obj: Any = root
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def _resolve_value(value: Any, root: dict) -> Any:
    """Recursively substitute ``${dotted.key}`` references from ``root``.

    Strings are re-expanded until a fixed point is reached so that
    chained references (``${a}`` -> ``${b}`` -> literal) resolve in
    one pass.
    """
    if isinstance(value, str):
        prev: Optional[str] = None
        while prev != value:
            prev = value
            value = _INTERP_PATTERN.sub(
                lambda m: str(_lookup_dotted(m.group(1), root)), value
            )
        return value
    if isinstance(value, dict):
        return {k: _resolve_value(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, root) for v in value]
    return value


def load_config(path: Optional[Union[str, Path]] = None) -> dict:
    """Load ``config.yaml`` with ``${...}`` interpolation, cached per path.

    Resolution order for the config path:

    1. Explicit ``path`` argument.
    2. ``PHOTOZ_CATALOG_CONFIG`` environment variable.
    3. ``catalog/config.yaml`` next to this module.
    """
    if path is None:
        path = os.environ.get("PHOTOZ_CATALOG_CONFIG") or _DEFAULT_CONFIG_PATH
    cache_key = str(Path(path).resolve())
    if cache_key not in _CONFIG_CACHE:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        _CONFIG_CACHE[cache_key] = _resolve_value(raw, raw)
    return _CONFIG_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Derived constants (populated from config at import time).
#
# PDF_UINT16_SCALE is a true constant (max value of a uint16). All the
# others are pulled from config.yaml so that users can tweak binning /
# RA partitioning without touching the code.
# ---------------------------------------------------------------------------


_cfg = load_config()

PDF_UINT16_SCALE = np.uint32(np.iinfo(np.uint16).max)
PDF_BINS: int = int(_cfg["bins"]["pdf_bins"])
PDF_BIN_EDGES = np.linspace(
    float(_cfg["bins"]["pdf_z_min"]),
    float(_cfg["bins"]["pdf_z_max"]),
    PDF_BINS + 1,
    dtype=np.float32,
)
N_RA_BINS: int = int(_cfg["bins"]["n_ra_bins"])
MERGE_RA_STEP: int = int(_cfg["bins"]["merge_ra_step"])
PUBLISH_RA_STEP: int = int(_cfg["bins"]["publish_ra_step"])

CATALOG_MAIN_COLUMNS: list[str] = list(_cfg["columns"]["main"])
CATALOG_SCALAR_COLUMNS: list[str] = [c for c in CATALOG_MAIN_COLUMNS if c != "uid"]
CATALOG_PDF_COLUMNS: list[str] = list(_cfg["columns"]["pdf"])


# ---------------------------------------------------------------------------
# Table I/O — FITS delegates to lib.io, HDF5 uses the photoz_table_v2 format.
# ---------------------------------------------------------------------------


_FITS_SUFFIXES = (".fits", ".fit", ".fz", ".fits.gz", ".fit.gz")
_HDF5_SUFFIXES = (".h5", ".hdf5")


def read_table(filepath: str) -> pd.DataFrame:
    """Read a catalog table; dispatched by extension.

    ``.fits`` / ``.fit`` (optionally ``.gz`` / ``.fz``) go through
    :func:`lib.io.readfile`; ``.h5`` / ``.hdf5`` use the project's
    :func:`read_hdf5` which understands the ``photoz_table_v2``
    layout (including the ``z_phot_pdf`` uint16 matrix).
    """
    lower = filepath.lower()
    if lower.endswith(_FITS_SUFFIXES):
        return readfile(filepath, data_format="dataframe")
    if lower.endswith(_HDF5_SUFFIXES):
        return read_hdf5(filepath)
    raise ValueError(f"Unsupported input format: {filepath}")


def save_table(df: pd.DataFrame, filepath: str) -> None:
    """Write a catalog table; dispatched by extension.

    Mirrors :func:`read_table`. ``z_phot_pdf`` columns can only be
    written to HDF5 because FITS binary tables can't represent the
    per-row ``uint16[PDF_BINS]`` array used by the project format.
    """
    lower = filepath.lower()
    if lower.endswith(_FITS_SUFFIXES):
        save_fits(df, filepath)
        return
    if lower.endswith(_HDF5_SUFFIXES):
        save_hdf5(df, filepath)
        return
    raise ValueError(f"Unsupported output format: {filepath}")


def save_fits(df: pd.DataFrame, filepath: str) -> None:
    """Save a photoz DataFrame to FITS via :func:`lib.io.savefile`.

    Widens narrow integer dtypes first because astropy's FITS writer
    refuses ``int8`` and ``uint16``: ``int8`` -> ``int16`` and
    ``uint16`` -> ``int32``. Object columns holding ``uint16``
    ndarrays (e.g. stray PDF columns in sanity-check paths) get the
    same widening applied element-wise.

    ``auto_convert_objects=False`` is passed through so that
    lib.io does not coerce genuine array/object columns via
    ``pd.to_numeric``.
    """
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == np.int8:
            df[col] = df[col].astype(np.int16)
        elif dtype == np.uint16:
            df[col] = df[col].astype(np.int32)
        elif dtype == object and len(df[col]) > 0:
            first_valid = next((v for v in df[col] if v is not None), None)
            if isinstance(first_valid, np.ndarray) and first_valid.dtype == np.uint16:
                df[col] = [
                    np.asarray(v, dtype=np.int32) if v is not None else v
                    for v in df[col]
                ]
    savefile(df, filepath, auto_convert_objects=False)


# ---------------------------------------------------------------------------
# Custom HDF5 format ("photoz_table_v2"):
#
#   /meta    attrs: format = "photoz_table_v2"
#                   pdf_uint16_scale = 65535
#                   pdf_bins (when z_phot_pdf present)
#            datasets: columns[str], pdf_bin_edges[float32]
#   /data    one dataset per column; z_phot_pdf is uint16[N, PDF_BINS]
#
# Every dataset under /data is gzip + shuffle compressed.
# ---------------------------------------------------------------------------


def encode_pdf_to_uint16(probs: np.ndarray) -> np.ndarray:
    """Quantise a normalised 2D PDF to ``uint16`` with row sums == 65535.

    Floors ``probs * 65535``, then distributes the per-row deficit
    to the bins with the largest fractional remainder so that every
    row's integer total exactly equals :data:`PDF_UINT16_SCALE`.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array")

    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each PDF row must have positive total probability")
    probs = probs / row_sums

    scaled = probs * PDF_UINT16_SCALE
    base = np.floor(scaled).astype(np.uint32)
    frac = scaled - base
    deficit = (PDF_UINT16_SCALE - base.sum(axis=1)).astype(np.int64)

    max_deficit = int(deficit.max())
    if max_deficit > 0:
        top_idx = np.argpartition(-frac, kth=max_deficit - 1, axis=1)[:, :max_deficit]
        add_mask = np.arange(max_deficit)[None, :] < deficit[:, None]
        row_idx = np.broadcast_to(np.arange(len(probs))[:, None], top_idx.shape)[add_mask]
        col_idx = top_idx[add_mask]
        base[row_idx, col_idx] += 1

    return base.astype(np.uint16)


def save_hdf5(df: pd.DataFrame, filepath: str) -> None:
    """Write a DataFrame to HDF5 in the ``photoz_table_v2`` format."""
    df = df.copy()
    with h5py.File(filepath, "w") as f:
        meta = f.create_group("meta")
        data_group = f.create_group("data")

        meta.attrs["format"] = "photoz_table_v2"
        meta.attrs["pdf_uint16_scale"] = int(PDF_UINT16_SCALE)
        meta.create_dataset(
            "columns", data=np.array(df.columns.tolist(), dtype="S")
        )
        if "z_phot_pdf" in df.columns:
            meta.attrs["pdf_bins"] = PDF_BINS
            meta.create_dataset("pdf_bin_edges", data=PDF_BIN_EDGES)

        for col in df.columns:
            values = df[col].values
            if col == "z_phot_pdf":
                pdf_array = np.stack(
                    [np.asarray(v, dtype=np.uint16) for v in values], axis=0
                )
                data_group.create_dataset(
                    col, data=pdf_array, compression="gzip", shuffle=True
                )
            else:
                data_group.create_dataset(
                    col, data=values, compression="gzip", shuffle=True
                )


def read_hdf5(filepath: str) -> pd.DataFrame:
    """Read a ``photoz_table_v2`` HDF5 file into a DataFrame.

    ``z_phot_pdf`` is returned as a Python list of uint16 arrays
    (one per row) so it round-trips through pandas without being
    flattened to object-of-scalars.
    """
    data: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        meta = f["meta"]
        data_group = f["data"]
        columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in meta["columns"][...]
        ]
        for col in columns:
            values = data_group[col][...]
            if col == "z_phot_pdf":
                data[col] = [row for row in values]
            else:
                data[col] = values
    return pd.DataFrame(data)


def is_valid_prediction_hdf5(filepath: str, required_columns: list) -> bool:
    """Quick structural sanity check for a prediction-stage HDF5 file.

    Used to decide whether an existing output can be skipped vs.
    overwritten. Returns ``True`` only when the file has a sane
    ``/meta`` + ``/data`` layout, every ``required_columns`` entry
    is present in both, the PDF dataset (if any) matches the
    expected shape, and the file is non-empty.
    """
    if not os.path.exists(filepath):
        return False
    try:
        with h5py.File(filepath, "r") as f:
            if "meta" not in f or "data" not in f:
                return False
            meta = f["meta"]
            data = f["data"]
            if "columns" not in meta:
                return False
            cols = [
                c.decode() if isinstance(c, bytes) else c
                for c in meta["columns"][...]
            ]
            for col in required_columns:
                if col not in cols or col not in data:
                    return False
            if "z_phot_pdf" in data:
                pdf = data["z_phot_pdf"]
                if pdf.ndim != 2 or pdf.shape[1] != PDF_BINS:
                    return False
            n_rows = data[required_columns[0]].shape[0]
            return n_rows > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Photo-z statistics on GPU.
# ---------------------------------------------------------------------------


def compute_photoz_stats_torch(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    to_cpu: bool = True,
) -> dict:
    """Per-row mean / std / mode / (l95, l68, median, u68, u95) on GPU.

    Quantiles are linearly interpolated across the CDF so they do
    not snap to bin edges. Returning ``to_cpu=False`` keeps tensors
    on GPU for subsequent torch operations.
    """
    z_mean = torch.sum(probs * bin_centers, dim=1)

    z_diff = bin_centers.unsqueeze(0) - z_mean.unsqueeze(1)
    z_std = torch.sqrt(torch.sum(probs * z_diff ** 2, dim=1))

    z_mode = bin_centers[torch.argmax(probs, dim=1)]

    cdf = torch.cumsum(probs, dim=1)
    rows = torch.arange(probs.shape[0], device=probs.device)

    quantiles_map = {
        "z_phot_l95": 0.025,
        "z_phot_l68": 0.16,
        "z_phot_median": 0.50,
        "z_phot_u68": 0.84,
        "z_phot_u95": 0.975,
    }

    quantile_results: dict[str, torch.Tensor] = {}
    for col_name, q in quantiles_map.items():
        idx = torch.argmax((cdf >= q).to(torch.int64), dim=1)
        idx_prev = torch.clamp(idx - 1, min=0)

        z_high = bin_centers[idx]
        z_low = bin_centers[idx_prev]
        c_high = cdf[rows, idx]
        c_low = cdf[rows, idx_prev]

        denom = c_high - c_low
        valid = (idx > 0) & (denom > 0)

        result = z_high.clone()
        result[valid] = (
            z_low[valid]
            + (q - c_low[valid]) / denom[valid] * (z_high[valid] - z_low[valid])
        )
        quantile_results[col_name] = result

    stats = {
        "z_phot_mean": z_mean,
        "z_phot_std": z_std,
        "z_phot_mode": z_mode,
        **quantile_results,
    }
    if to_cpu:
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in stats.items()
        }
    return stats


def resample_bins(
    prob: np.ndarray,
    bin_edges: np.ndarray,
    target_n: Optional[int] = None,
):
    """Resample a high-resolution PDF to ``target_n`` bins, preserving mass.

    ``prob`` is treated as probability mass (expected to sum to 1).
    Mass is remapped through the CDF so ``new_prob`` also sums to 1
    regardless of bin-edge alignment. When ``target_n`` is ``None``,
    the published :data:`PDF_BINS` is used.
    """
    if target_n is None:
        target_n = PDF_BINS
    new_bin_edges = np.linspace(bin_edges[0], bin_edges[-1], target_n + 1)
    new_bin_centers = 0.5 * (new_bin_edges[:-1] + new_bin_edges[1:])

    original_cdf = np.concatenate(([0], np.cumsum(prob)))
    original_cdf /= original_cdf[-1]

    new_cdf_edges = np.interp(new_bin_edges, bin_edges, original_cdf)

    new_prob = np.diff(new_cdf_edges)
    new_prob = new_prob / np.sum(new_prob)

    return new_prob, new_bin_edges, new_bin_centers


def build_resample_matrix(
    bin_edges: np.ndarray,
    target_n: Optional[int] = None,
) -> np.ndarray:
    """Pre-compute the ``n_input_bins x target_n`` resample matrix.

    Once built, downsampling a batch is a single ``probs @ matrix``
    call, which runs much faster than looping :func:`resample_bins`
    row by row. Each column of the matrix is
    :func:`resample_bins` applied to a one-hot basis vector.
    """
    if target_n is None:
        target_n = PDF_BINS
    n_input_bins = len(bin_edges) - 1
    basis = np.eye(n_input_bins, dtype=np.float64)
    matrix = np.empty((n_input_bins, target_n), dtype=np.float32)

    for i in range(n_input_bins):
        new_prob, _, _ = resample_bins(basis[i], bin_edges, target_n=target_n)
        matrix[i] = new_prob.astype(np.float32)

    return matrix

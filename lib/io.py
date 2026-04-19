"""Unified table I/O for FITS, HDF5, CSV, and DAT files.

Two entry points:

* :func:`readfile` — read any supported format into a
  :class:`pandas.DataFrame` (or :class:`astropy.table.Table`). Can
  optionally return the FITS primary header alongside the data.
* :func:`savefile` — write a DataFrame / Table / ndarray back to disk.

Supported extensions (case-insensitive):

* ``.fits`` / ``.fit`` — binary FITS tables
* ``.h5`` — pandas HDF5 (``format='table'``)
* ``.csv`` — comma-separated values
* ``.dat`` — whitespace-delimited text loaded via :func:`numpy.loadtxt`

Example:
    >>> df = readfile("catalog.fits")
    >>> df, header = readfile("catalog.fits", return_header=True)
    >>> savefile(df, "out.h5")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

__all__ = ["readfile", "savefile"]


PathLike = Union[str, Path]
DataLike = Union[pd.DataFrame, pd.Series, Table, np.ndarray]

# ``.fit`` is a legacy alias for ``.fits``.
_FITS_SUFFIXES = {".fits", ".fit"}


# ---------------------------------------------------------------------------
# Private readers — one per file format. They return the "natural" container
# for that format (DataFrame for the pandas-backed formats, Table for FITS).
# ---------------------------------------------------------------------------


def _read_hdf5(path: Path, *, hdf5_key: str = "data", **kw: Any) -> pd.DataFrame:
    return pd.read_hdf(path, key=hdf5_key, mode="r", **kw)


def _read_csv(path: Path, **kw: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def _read_dat(
    path: Path, *, delimiter: Optional[str] = None, **kw: Any
) -> pd.DataFrame:
    arr = np.loadtxt(path, delimiter=delimiter or " ", **kw)
    return pd.DataFrame(arr)


def _read_fits(path: Path, *, hdu_index: int = 1, **kw: Any) -> Table:
    return Table.read(path, format="fits", hdu=hdu_index, **kw)


# ---------------------------------------------------------------------------
# Private writers — one per file format. Each expects the input container
# type most natural for that format and handles the conversion itself.
# ---------------------------------------------------------------------------


def _write_hdf5(
    df: pd.DataFrame,
    path: Path,
    *,
    hdf5_key: str = "data",
    compression: Optional[str] = None,
    **kw: Any,
) -> None:
    df.to_hdf(
        path,
        key=hdf5_key,
        mode="w",
        format="table",
        complib=compression,
        **kw,
    )


def _write_csv(df: pd.DataFrame, path: Path, **kw: Any) -> None:
    df.to_csv(path, index=False, **kw)


def _write_dat(
    data: DataLike,
    path: Path,
    *,
    delimiter: Optional[str] = None,
    **kw: Any,
) -> None:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        arr = data.to_numpy()
    else:
        arr = np.asarray(data)
    np.savetxt(path, arr, delimiter=delimiter or "\t", **kw)


def _write_fits(data: DataLike, path: Path, **kw: Any) -> None:
    if isinstance(data, pd.DataFrame):
        table = Table.from_pandas(data)
    elif isinstance(data, Table):
        table = data
    else:
        table = Table(data)
    table.write(path, format="fits", overwrite=True, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_object_columns(
    df: pd.DataFrame, *, verbose: bool = False
) -> pd.DataFrame:
    """Coerce ``object``-dtype columns to numeric where possible.

    Non-numeric values become ``NaN``. This avoids ragged dtypes when
    writing to FITS / HDF5 but may silently drop genuine string
    columns — callers that want to preserve strings should pass
    ``auto_convert_objects=False`` to :func:`savefile`.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) == 0:
        return df

    for col in object_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if verbose:
        converted = [c for c in object_cols if df[c].dtype != object]
        if converted:
            print(
                f"Coerced {len(converted)} object column(s) to numeric: "
                f"{', '.join(map(str, converted))}"
            )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def readfile(
    path: PathLike,
    *,
    data_format: Literal["dataframe", "table"] = "dataframe",
    delimiter: Optional[str] = None,
    hdf5_key: str = "data",
    hdu_index: int = 1,
    return_header: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, Table, tuple]:
    """Read a FITS / HDF5 / CSV / DAT table.

    Dispatches on the file extension: ``.fits`` / ``.fit`` go through
    :class:`astropy.table.Table`, all other formats go through pandas.
    The result is then converted to the container requested via
    ``data_format``.

    Args:
        path: File path. Extension is case-insensitive and selects the
            underlying reader.
        data_format: ``"dataframe"`` (default) returns a
            :class:`pandas.DataFrame`; ``"table"`` returns an
            :class:`astropy.table.Table`.
        delimiter: Delimiter for ``.dat`` files. ``None`` falls back
            to whitespace.
        hdf5_key: HDF5 dataset key. Only used for ``.h5`` files.
        hdu_index: FITS HDU index. Defaults to ``1`` (the first
            extension; HDU 0 is conventionally the primary header).
        return_header: If ``True`` and the file is FITS, return the
            tuple ``(data, header_dict)`` where ``header_dict`` is
            the primary (HDU 0) header as a plain dict. Must be
            ``False`` for non-FITS formats.
        verbose: If ``True``, print shape and column info after
            reading, plus ``hdul.info()`` for FITS.
        **kwargs: Forwarded to the underlying reader
            (:func:`pandas.read_csv`, :meth:`Table.read`, ...).

    Returns:
        The data as a DataFrame or Table, or a
        ``(data, header_dict)`` tuple if ``return_header=True``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: Extension is not supported, ``return_header=True``
            with a non-FITS file, or ``data_format`` is neither
            ``"dataframe"`` nor ``"table"``.

    Example:
        >>> df = readfile("catalog.fits")
        >>> df, hdr = readfile("catalog.fits", return_header=True)
        >>> df = readfile("catalog.h5", hdf5_key="data")
        >>> tbl = readfile("catalog.fits", data_format="table")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    header: Optional[dict] = None

    if suffix in _FITS_SUFFIXES:
        data = _read_fits(path, hdu_index=hdu_index, **kwargs)
        if return_header or verbose:
            with fits.open(str(path)) as hdul:
                if verbose:
                    print(f"FITS info ({path.name}):")
                    hdul.info()
                if return_header:
                    header = dict(hdul[0].header)
    else:
        if return_header:
            raise ValueError("return_header=True is only valid for FITS files.")
        if suffix == ".h5":
            data = _read_hdf5(path, hdf5_key=hdf5_key, **kwargs)
        elif suffix == ".csv":
            data = _read_csv(path, **kwargs)
        elif suffix == ".dat":
            data = _read_dat(path, delimiter=delimiter, **kwargs)
        else:
            raise ValueError(f"Unsupported extension: {suffix!r}")

    if data_format == "dataframe":
        if isinstance(data, Table):
            data = data.to_pandas()
    elif data_format == "table":
        if isinstance(data, pd.DataFrame):
            data = Table.from_pandas(data)
    else:
        raise ValueError(
            f"data_format must be 'dataframe' or 'table', got {data_format!r}"
        )

    if verbose and isinstance(data, pd.DataFrame):
        cols = list(data.columns)
        head = ", ".join(map(str, cols[:10]))
        tail = "..." if len(cols) > 10 else ""
        print(f"shape: {data.shape}, columns: {head}{tail}")

    if return_header:
        return data, header
    return data


def savefile(
    data: DataLike,
    path: PathLike,
    *,
    auto_convert_objects: bool = True,
    delimiter: Optional[str] = None,
    compression: Optional[str] = None,
    hdf5_key: str = "data",
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Write a DataFrame / Table / ndarray to FITS / HDF5 / CSV / DAT.

    Parent directories are created automatically. The writer is
    selected from the destination file extension (case-insensitive).

    Args:
        data: The payload. DataFrame, Series, Table, or ndarray.
        path: Destination file path.
        auto_convert_objects: If ``True`` and ``data`` is a DataFrame,
            coerce ``object``-dtype columns to numeric before writing.
            Non-numeric values become ``NaN``. Set to ``False`` when
            the DataFrame contains genuine string columns that must
            be preserved.
        delimiter: Delimiter for ``.dat`` files. ``None`` falls back
            to tab.
        compression: Compression library for ``.h5`` (passed as
            ``complib`` to :meth:`pandas.DataFrame.to_hdf`). ``None``
            writes uncompressed.
        hdf5_key: HDF5 dataset key.
        verbose: If ``True``, print conversion info when object
            columns are coerced.
        **kwargs: Forwarded to the underlying writer
            (:meth:`DataFrame.to_hdf` / :meth:`DataFrame.to_csv` /
            :meth:`Table.write` / :func:`numpy.savetxt`).

    Raises:
        ValueError: Extension is not supported.

    Example:
        >>> savefile(df, "out.h5")
        >>> savefile(df, "out.fits")
        >>> savefile(df, "out.csv", sep=";")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if auto_convert_objects and isinstance(data, pd.DataFrame):
        data = _coerce_object_columns(data, verbose=verbose)

    if suffix == ".h5":
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        _write_hdf5(
            df, path, hdf5_key=hdf5_key, compression=compression, **kwargs
        )
    elif suffix == ".csv":
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        _write_csv(df, path, **kwargs)
    elif suffix in _FITS_SUFFIXES:
        _write_fits(data, path, **kwargs)
    elif suffix == ".dat":
        _write_dat(data, path, delimiter=delimiter, **kwargs)
    else:
        raise ValueError(f"Unsupported extension: {suffix!r}")

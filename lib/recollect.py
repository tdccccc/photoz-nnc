"""Re-bucket a directory of FITS tables into fixed-width RA slices.

Large sky-survey catalogs are typically delivered as many per-tile FITS
shards whose source coordinates span the whole sky. Downstream pipelines
(cross-match, photo-z prediction, catalog publish) are much easier to
parallelize when the shards are instead organized by RA slice: every
output file holds all sources with RA inside
``[bin_idx * bin_size, (bin_idx + 1) * bin_size)``.

This module provides :func:`recollect_fits_by_ra`, which does that
re-bucketing in parallel with a bounded memory footprint:

1. A worker pool scans the input shards, bins rows by RA, and
   periodically flushes per-bin buffers to intermediate FITS files
   (``bin_XX_worker_YY_part_ZZZZ.fits``) under a scratch directory.
2. The main process concatenates the intermediate files for each bin
   into a single output FITS shard and (optionally) stamps a global
   monotonically-increasing ``uid`` column across all output files.

The intermediate spill step lets the implementation handle input
catalogs that are much larger than memory.

Example:
    >>> stats = recollect_fits_by_ra(
    ...     input_dir="survey/raw/",
    ...     output_dir="survey/ra_bins/",
    ...     ra_column="ra",
    ...     bin_size=5.0,
    ... )
    >>> stats["total_rows"], len(stats["output_files"])
"""

from __future__ import annotations

import glob
import multiprocessing
import os
import shutil
import tempfile
import time
import warnings
from typing import Any, Optional

import numpy as np
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

__all__ = ["recollect_fits_by_ra"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_table_dtypes(tables: list[Table]) -> list[Table]:
    """Align column dtypes across a list of tables so :func:`vstack` works.

    :func:`astropy.table.vstack` requires compatible dtypes for every
    shared column. Whenever a column appears with more than one dtype
    across the input tables, this helper casts all copies of that
    column to ``float32``: bytes / unicode values are decoded and
    parsed as floats (empty / ``"nan"``-like placeholders become
    ``NaN``), plain numeric columns are cast with ``.astype``.

    Args:
        tables: List of astropy :class:`~astropy.table.Table` objects
            with (largely) overlapping column names.

    Returns:
        New list of tables with dtypes normalized where needed. The
        original tables are never mutated.
    """
    if not tables:
        return tables

    col_types: dict[str, set[str]] = {}
    for t in tables:
        for col in t.colnames:
            col_types.setdefault(col, set()).add(str(t[col].dtype))

    problematic_cols = {c for c, s in col_types.items() if len(s) > 1}
    if not problematic_cols:
        return tables

    normalized: list[Table] = []
    for t in tables:
        t = t.copy()
        for col in problematic_cols:
            if col not in t.colnames:
                continue
            dtype_str = str(t[col].dtype)
            if dtype_str.startswith(("|S", "S", "bytes", "U", "<U")):
                # Decode / parse byte or unicode columns as float32.
                try:
                    coerced = []
                    for val in t[col]:
                        try:
                            if isinstance(val, bytes):
                                val = val.decode("utf-8").strip()
                            else:
                                val = str(val).strip()
                            if val == "" or val.lower() in (
                                "nan",
                                "null",
                                "none",
                                "--",
                            ):
                                coerced.append(np.nan)
                            else:
                                coerced.append(float(val))
                        except Exception:
                            coerced.append(np.nan)
                    t[col] = np.array(coerced, dtype=np.float32)
                except Exception:
                    # Give up on this column; vstack may still succeed
                    # or surface a clearer error downstream.
                    pass
            else:
                try:
                    t[col] = t[col].astype(np.float32)
                except Exception:
                    pass
        normalized.append(t)

    return normalized


def _process_chunk(args: tuple[list[str], int, dict[str, Any]]) -> bool:
    """Worker: bin one chunk of input files and spill per-bin buffers.

    Runs in a subprocess spawned by :func:`recollect_fits_by_ra`.
    Reads each file in ``file_chunk``, computes a bin index per row,
    and appends the per-bin mini-tables into an in-memory buffer. When
    a bin's buffer exceeds ``buffer_flush_size`` rows, the buffer is
    vstacked and written to
    ``{temp_dir}/bin_XX_worker_YY_part_ZZZZ.fits``; after every input
    file is consumed the remaining non-empty buffers are flushed too.

    Errors reading a single file are logged to stdout but do not halt
    the worker or other workers.

    Args:
        args: Tuple ``(file_chunk, worker_id, config)`` where
            ``config`` carries ``temp_dir``, ``ra_column``,
            ``data_hdu_index``, ``buffer_flush_size``, ``bin_size``,
            ``num_bins``.

    Returns:
        ``True`` on completion (the return value is unused by the
        parent — it only waits for all workers to finish).
    """
    # Silence astropy warnings in the subprocess as well.
    warnings.filterwarnings("ignore", category=AstropyWarning)

    file_chunk, worker_id, config = args
    temp_dir = config["temp_dir"]
    ra_column = config["ra_column"]
    data_hdu_index = config["data_hdu_index"]
    buffer_flush_size = config["buffer_flush_size"]
    bin_size = config["bin_size"]
    num_bins = config["num_bins"]

    buffers: list[list[Table]] = [[] for _ in range(num_bins)]
    part_counters = [0] * num_bins

    def _flush(bin_idx: int) -> None:
        normalized = _normalize_table_dtypes(buffers[bin_idx])
        merged = vstack(normalized, metadata_conflicts="silent")
        out_path = os.path.join(
            temp_dir,
            f"bin_{bin_idx:02d}_worker_{worker_id:02d}"
            f"_part_{part_counters[bin_idx]:04d}.fits",
        )
        merged.write(out_path, format="fits", overwrite=True)
        part_counters[bin_idx] += 1
        buffers[bin_idx] = []

    for filepath in file_chunk:
        try:
            table = Table.read(filepath, hdu=data_hdu_index)
            if ra_column not in table.colnames:
                continue

            ra_values = table[ra_column] % 360.0
            bin_indices = (ra_values // bin_size).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            for bin_idx in np.unique(bin_indices):
                mask = bin_indices == bin_idx
                buffers[bin_idx].append(table[mask])

                total_rows = sum(len(t) for t in buffers[bin_idx])
                if total_rows >= buffer_flush_size:
                    _flush(bin_idx)

            del table, ra_values, bin_indices
        except Exception as e:
            print(
                f"[Worker {worker_id}] Warning: failed to process "
                f"{os.path.basename(filepath)}: {e}"
            )

    # Final flush for all non-empty buffers.
    for bin_idx in range(num_bins):
        if buffers[bin_idx]:
            _flush(bin_idx)

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recollect_fits_by_ra(
    input_dir: str,
    output_dir: str,
    *,
    output_filename: str = "ra_bin_{i:02d}.fits",
    temp_dir: Optional[str] = None,
    num_processes: Optional[int] = None,
    ra_column: str = "ra",
    fits_extension: str = "*.fits",
    data_hdu_index: int = 1,
    buffer_flush_size: int = 1_000_000,
    bin_size: float = 5.0,
    add_uid: bool = True,
) -> dict[str, Any]:
    """Re-bucket a directory of FITS tables into fixed-width RA slices.

    The function partitions all sources across the input shards into
    ``360 / bin_size`` slices in right ascension and writes one FITS
    shard per slice. See the module docstring for the two-stage
    (spill-then-consolidate) algorithm and why it bounds memory usage.

    Args:
        input_dir: Directory holding the input FITS shards.
        output_dir: Destination directory for the RA-binned shards.
            Created if missing.
        output_filename: Output filename template. Must contain the
            ``{i}`` placeholder that receives the bin index. Defaults
            to ``"ra_bin_{i:02d}.fits"``.
        temp_dir: Scratch directory for intermediate per-worker shards.
            ``None`` (default) allocates a fresh directory under the
            system temp location (``tempfile.mkdtemp``). If a path is
            given, any existing contents are wiped at the start and
            the whole directory is removed at the end.
        num_processes: Worker pool size. ``None`` defaults to
            ``max(1, cpu_count() - 2)``.
        ra_column: Name of the right-ascension column in the input
            FITS tables. Defaults to ``"ra"``.
        fits_extension: Glob pattern used to discover input files
            under ``input_dir``. Defaults to ``"*.fits"``.
        data_hdu_index: HDU index that holds the binary table in each
            input FITS. Defaults to 1 (first extension).
        buffer_flush_size: Per-bin buffer threshold in rows. Once a
            worker accumulates this many rows for a bin, the buffer is
            flushed to a temp FITS file. Defaults to 1,000,000.
        bin_size: Width of each RA slice in degrees. Defaults to 5.0
            (which yields 72 output shards).
        add_uid: If ``True``, stamp a global monotonically-increasing
            ``"uid"`` column (``int64``, starting at 0) across the
            consolidated output files in bin order.

    Returns:
        Dict with keys:

        * ``"total_rows"``: total number of rows written across all
          output shards.
        * ``"duration_minutes"``: wall time for the whole run.
        * ``"output_files"``: list of paths for the output shards
          (only the bins that received at least one row).

    Example:
        >>> stats = recollect_fits_by_ra(
        ...     input_dir="survey/raw/",
        ...     output_dir="survey/ra_bins/",
        ...     ra_column="ra",
        ...     bin_size=5.0,
        ... )
        >>> stats["total_rows"]
        42000000
    """
    warnings.filterwarnings("ignore", category=AstropyWarning)
    start_time = time.time()

    if num_processes is None:
        num_processes = max(1, (os.cpu_count() or 1) - 2)

    num_bins = int(360 / bin_size)

    # Resolve the scratch directory: either ours via tempfile, or the
    # path provided by the caller (which we clear upfront).
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="recollect_ra_")
    else:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "temp_dir": temp_dir,
        "ra_column": ra_column,
        "data_hdu_index": data_hdu_index,
        "buffer_flush_size": buffer_flush_size,
        "bin_size": bin_size,
        "num_bins": num_bins,
    }

    print(
        f"Step 1: {num_processes} workers will bin into {num_bins} "
        f"RA slices of {bin_size}°."
    )

    source_files = sorted(glob.glob(os.path.join(input_dir, fits_extension)))
    if not source_files:
        print(f"No files matching {fits_extension!r} under {input_dir!r}.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "total_rows": 0,
            "duration_minutes": 0.0,
            "output_files": [],
        }

    total_size_gb = sum(os.path.getsize(f) for f in source_files) / (1024 ** 3)
    print(
        f"Step 2: Found {len(source_files)} source files "
        f"({total_size_gb:.2f} GB total)."
    )

    chunks = np.array_split(source_files, num_processes)
    tasks = [(list(chunk), i, config) for i, chunk in enumerate(chunks)]

    print("Step 3: Parallel binning and buffer flushing ...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(_process_chunk, tasks)
    print(
        f"Parallel stage finished in "
        f"{(time.time() - start_time) / 60:.2f} minutes."
    )

    print("Step 4: Consolidating per-bin temp files ...")
    total_rows = 0
    output_files: list[str] = []
    uid_counter = 0

    for i in tqdm(range(num_bins), desc="Consolidating bins"):
        temp_files_for_bin = sorted(
            glob.glob(os.path.join(temp_dir, f"bin_{i:02d}_worker_*.fits"))
        )
        if not temp_files_for_bin:
            continue

        try:
            tables = [Table.read(f, hdu=1) for f in temp_files_for_bin]
            tables = _normalize_table_dtypes(tables)
            final_table = vstack(tables, metadata_conflicts="silent")

            if add_uid:
                n_rows = len(final_table)
                final_table["uid"] = np.arange(
                    uid_counter, uid_counter + n_rows, dtype=np.int64
                )
                uid_counter += n_rows

            out_path = os.path.join(output_dir, output_filename.format(i=i))
            final_table.write(out_path, format="fits", overwrite=True)

            total_rows += len(final_table)
            output_files.append(out_path)
            del tables, final_table
        except Exception as e:
            print(f"\nError consolidating bin {i}: {e}")
            print(f"Problematic files might be: {temp_files_for_bin}")

    print("Step 5: Removing temp directory ...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    total_duration = time.time() - start_time
    print(
        f"\nDone. {total_rows:,} rows written in "
        f"{total_duration / 60:.2f} minutes. Output at {output_dir}."
    )

    return {
        "total_rows": total_rows,
        "duration_minutes": total_duration / 60,
        "output_files": output_files,
    }

"""Cross-survey helpers for catalog-shard aggregates.

Currently exposes:

* :func:`survey_source_count` — parallel-counts how many rows live
  across every shard in a directory (useful for "how big is this
  dataset really?" sanity checks).
* :func:`collect_columns` — gathers a subset of columns from every
  shard into a single output file, flushing intermediates to a temp
  directory to keep peak memory bounded.

Expect more shared survey-statistics helpers — coverage maps,
per-RA-bin counts, completeness diagnostics — to land here later.
Anything that is specific to a single survey belongs under
``surveys/<NAME>/`` instead.
"""

from __future__ import annotations

import concurrent.futures
import shutil
import threading
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd
from tqdm import tqdm

from .io import readfile, savefile

__all__ = ["survey_source_count", "collect_columns"]


def survey_source_count(
    dir_path: Union[str, Path],
    *,
    file_pattern: str = "*.fits",
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Count total source rows across every file in a directory.

    Walks ``dir_path`` for files matching ``file_pattern``, reads
    each through :func:`lib.io.readfile`, and sums the row counts in
    parallel using a thread pool. Per-file read errors are logged
    and counted as zero, so a single bad file cannot abort a
    whole-survey audit.

    Args:
        dir_path: Directory to scan (non-recursive).
        file_pattern: Glob pattern relative to ``dir_path``. Defaults
            to ``"*.fits"``.
        max_workers: Thread pool size. ``None`` lets
            :class:`concurrent.futures.ThreadPoolExecutor` pick the
            default (typically ``min(32, cpu_count + 4)``).
        show_progress: If ``True``, render a tqdm progress bar.

    Returns:
        Dict with keys:

        * ``"total_sources"`` — sum of row counts across all matching
          files.
        * ``"n_files"`` — number of files successfully processed.
        * ``"files_processed"`` — sorted list of processed paths
          (strings).
        * ``"average_sources_per_file"`` — mean row count, or ``0``
          when no file was processed.

    Raises:
        FileNotFoundError: ``dir_path`` does not exist.

    Example:
        >>> from lib.survey_stats import survey_source_count
        >>> stats = survey_source_count("survey/raw72/", file_pattern="*.fits")
        >>> stats["total_sources"]
        12345678
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    file_list = list(dir_path.glob(file_pattern))
    if not file_list:
        return {
            "total_sources": 0,
            "n_files": 0,
            "files_processed": [],
            "average_sources_per_file": 0,
        }

    total_sources = 0
    processed_files: list[str] = []
    lock = threading.Lock()

    def _count_one(file_path: Path) -> int:
        try:
            df = readfile(file_path)
            n = len(df)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return 0
        with lock:
            nonlocal total_sources
            total_sources += n
            processed_files.append(str(file_path))
        return n

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {executor.submit(_count_one, f): f for f in file_list}
        iterator = concurrent.futures.as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator, total=len(futures), desc="Counting sources"
            )
        for future in iterator:
            future.result()

    n_files = len(processed_files)
    return {
        "total_sources": total_sources,
        "n_files": n_files,
        "files_processed": sorted(processed_files),
        "average_sources_per_file": (
            total_sources / n_files if n_files else 0
        ),
    }


def collect_columns(
    src_dir: Union[str, Path],
    output_path: Union[str, Path],
    columns: Sequence[str],
    *,
    file_pattern: str = "*.fits",
    chunk_size: int = 10,
    temp_dir: Optional[Union[str, Path]] = None,
    cleanup_temp: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Gather a subset of columns from every shard in a directory.

    Walks ``src_dir`` for files matching ``file_pattern``, reads each
    one through :func:`lib.io.readfile`, slices it down to ``columns``,
    and concatenates the pieces into a single table written to
    ``output_path``. To keep peak memory bounded on many-shard surveys,
    intermediate buffers are flushed to a temp directory every
    ``chunk_size`` files; the chunks are re-read and merged at the end.

    The function is survey-agnostic — change ``file_pattern`` /
    ``columns`` to target a different dataset. It is the right primitive
    for building a shared "object locations" (``uid``, ``ra``, ``dec``)
    table, or any other narrow projection across a directory of shards.

    Args:
        src_dir: Directory to scan (non-recursive).
        output_path: Destination for the merged table. The extension
            selects the writer (see :func:`lib.io.savefile`) and is also
            reused for the per-chunk temp files so that each chunk can
            be read back with the same machinery.
        columns: Column names to retain from every shard. A shard that
            is missing any requested column causes a ``KeyError``.
        file_pattern: Glob relative to ``src_dir``. Defaults to
            ``"*.fits"``.
        chunk_size: Number of shards to accumulate in memory before
            flushing to a temp file. Larger values trade memory for
            fewer merge-time reads.
        temp_dir: Where to place intermediate chunks. ``None`` creates a
            hidden sibling of ``output_path`` (``.<stem>_tmp``).
        cleanup_temp: If ``True`` (default), remove ``temp_dir`` after
            the final merge. Set to ``False`` when debugging a crash
            mid-run.
        show_progress: If ``True``, render tqdm bars for the per-shard
            read pass and the final merge pass.

    Returns:
        pandas.DataFrame: The merged table, identical in content to
        what was written to ``output_path``.

    Raises:
        FileNotFoundError: ``src_dir`` does not exist, or no files in
            it match ``file_pattern``.

    Example:
        >>> from lib.survey_stats import collect_columns
        >>> loc = collect_columns(
        ...     "/data/PanSTARRS/DR2All/raw72",
        ...     "/data/PanSTARRS/DR2All/ps1dr2_raw72_loc.fits",
        ...     columns=["uid", "raStack", "decStack"],
        ...     chunk_size=10,
        ... )
    """
    src_dir = Path(src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"Directory not found: {src_dir}")

    file_list = sorted(src_dir.glob(file_pattern))
    if not file_list:
        raise FileNotFoundError(
            f"No files matching {file_pattern!r} in {src_dir}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_suffix = output_path.suffix or ".fits"

    if temp_dir is None:
        temp_dir = output_path.parent / f".{output_path.stem}_tmp"
    else:
        temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    columns = list(columns)
    chunk_paths: list[Path] = []
    buffer: list[pd.DataFrame] = []
    chunk_id = 0

    def _flush(buf: list[pd.DataFrame], idx: int) -> Path:
        merged = pd.concat(buf, ignore_index=True)
        chunk_path = temp_dir / f"chunk_{idx:03d}{chunk_suffix}"
        savefile(merged, chunk_path)
        tqdm.write(f"[chunk {idx}] saved {len(merged):,} rows -> {chunk_path.name}")
        return chunk_path

    read_iter = enumerate(file_list, start=1)
    if show_progress:
        read_iter = tqdm(read_iter, total=len(file_list), desc="Reading shards")

    for i, fpath in read_iter:
        df = readfile(fpath)
        buffer.append(df[columns])
        if i % chunk_size == 0:
            chunk_paths.append(_flush(buffer, chunk_id))
            buffer = []
            chunk_id += 1

    if buffer:
        chunk_paths.append(_flush(buffer, chunk_id))
        buffer = []

    merge_iter: Any = chunk_paths
    if show_progress:
        merge_iter = tqdm(chunk_paths, desc="Merging chunks")

    merged_chunks = [readfile(cp) for cp in merge_iter]
    result = pd.concat(merged_chunks, ignore_index=True)
    savefile(result, output_path)
    print(f"Saved {len(result):,} rows to {output_path}")

    if cleanup_temp:
        shutil.rmtree(temp_dir)

    return result

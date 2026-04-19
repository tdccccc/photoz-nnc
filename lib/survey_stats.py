"""Per-directory source-count statistics for survey catalog dumps.

Currently exposes a single helper, :func:`survey_source_count`, that
parallel-counts how many rows live across every shard in a directory
(useful for "how big is this dataset really?" sanity checks). Expect
more shared survey-statistics helpers — coverage maps, per-RA-bin
counts, completeness diagnostics — to land here later. Anything that
is specific to a single survey belongs under
``surveys/<NAME>/`` instead.
"""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import Any, Optional, Union

from tqdm import tqdm

from .io import readfile

__all__ = ["survey_source_count"]


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

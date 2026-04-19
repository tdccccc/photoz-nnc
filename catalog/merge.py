"""Merge stage: combine per-survey prediction shards into a unified catalog.

This module takes the 72 per-RA prediction HDF5 files produced by each
survey (LSDR10 and PS1DR2), cross-matches them with STILTS, resolves
duplicates by priority, and writes 72 intermediate 5-degree merge
shards (one main FITS + one PDF HDF5 per shard). A final pass assigns
globally contiguous UIDs across all shards.

The intermediate shards are the input for the publish stage
(:mod:`catalog.publish`).

Entry point: :func:`merge_surveys` (reads all paths from
``config.yaml``).
"""

from __future__ import annotations

import gc
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

from lib.crossmatch import stilts_crossmatch

from .common import (
    CATALOG_MAIN_COLUMNS,
    CATALOG_PDF_COLUMNS,
    CATALOG_SCALAR_COLUMNS,
    MERGE_RA_STEP,
    N_RA_BINS,
    PDF_BINS,
    load_config,
    read_hdf5,
    read_table,
    save_fits,
    save_hdf5,
    save_table,
)

__all__ = ["merge_surveys"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _catalog_output_dirs(cfg: dict):
    """Ensure merge/publish directories exist and return their paths."""
    merge_main = cfg["paths"]["merge"]["main"]
    merge_pdf = cfg["paths"]["merge"]["pdf"]
    publish = cfg["paths"]["publish"]
    os.makedirs(merge_main, exist_ok=True)
    os.makedirs(merge_pdf, exist_ok=True)
    os.makedirs(publish, exist_ok=True)
    return merge_main, merge_pdf, publish


def _intermediate_paths(
    merge_main_dir: str,
    merge_pdf_dir: str,
    ra_start: int,
    ra_end: int,
    cfg: dict,
):
    """Return (main_fits_path, pdf_hdf5_path) for one 5-degree shard."""
    tmpl_main = cfg["templates"]["merge_main"]
    tmpl_pdf = cfg["templates"]["merge_pdf"]
    return (
        os.path.join(
            merge_main_dir, tmpl_main.format(ra_start=ra_start, ra_end=ra_end)
        ),
        os.path.join(
            merge_pdf_dir, tmpl_pdf.format(ra_start=ra_start, ra_end=ra_end)
        ),
    )


def _remove_files_if_exist(*filepaths: str) -> None:
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _count_non_increasing_steps(values: np.ndarray) -> int:
    if len(values) <= 1:
        return 0
    return int(np.sum(values[1:] <= values[:-1]))


def _fits_uid_info(filepath: str, return_uid: bool = False) -> dict:
    """Read row count and UID range from a merge main FITS shard."""
    with fits.open(filepath, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"Missing binary table extension in {filepath}")
        hdu = hdul[1]
        columns = list(hdu.columns.names)
        if columns != CATALOG_MAIN_COLUMNS:
            raise ValueError(
                f"Unexpected main columns in {filepath}: {columns}"
            )

        n_rows = int(hdu.header.get("NAXIS2", 0))
        if n_rows == 0:
            info: dict = {"rows": 0, "first_uid": None, "last_uid": None}
            if return_uid:
                info["uid"] = np.array([], dtype=np.int64)
            return info

        uid = np.asarray(hdu.data["uid"])
        non_increasing = _count_non_increasing_steps(uid)
        if non_increasing > 0:
            raise ValueError(
                f"Main uid not strictly increasing in {filepath}: "
                f"{non_increasing}"
            )

        info = {
            "rows": n_rows,
            "first_uid": int(uid[0]),
            "last_uid": int(uid[-1]),
        }
        if return_uid:
            info["uid"] = uid
        return info


def _pdf_uid_info(filepath: str, return_uid: bool = False) -> dict:
    """Read row count and UID range from a merge PDF HDF5 shard."""
    with h5py.File(filepath, "r") as f:
        if "meta" not in f or "data" not in f:
            raise ValueError(f"Invalid HDF5 structure: {filepath}")

        meta = f["meta"]
        data = f["data"]
        if "columns" not in meta:
            raise ValueError(f"Missing HDF5 columns metadata: {filepath}")
        columns = [
            col.decode() if isinstance(col, bytes) else col
            for col in meta["columns"][...]
        ]
        if columns != CATALOG_PDF_COLUMNS:
            raise ValueError(
                f"Unexpected PDF columns in {filepath}: {columns}"
            )
        if "uid" not in data or "z_phot_pdf" not in data:
            raise ValueError(f"Missing uid/z_phot_pdf dataset: {filepath}")

        uid_ds = data["uid"]
        pdf_ds = data["z_phot_pdf"]
        n_rows = int(uid_ds.shape[0])

        if pdf_ds.shape != (n_rows, PDF_BINS):
            raise ValueError(
                f"Unexpected PDF shape in {filepath}: {pdf_ds.shape}"
            )
        if pdf_ds.dtype != np.uint16:
            raise ValueError(
                f"Unexpected PDF dtype in {filepath}: {pdf_ds.dtype}"
            )
        if meta.attrs.get("format") != "photoz_table_v2":
            raise ValueError(
                f"Unexpected HDF5 format in {filepath}: "
                f"{meta.attrs.get('format')}"
            )

        if n_rows == 0:
            info: dict = {"rows": 0, "first_uid": None, "last_uid": None}
            if return_uid:
                info["uid"] = np.array([], dtype=np.int64)
            return info

        uid_values = uid_ds[...]
        non_increasing = _count_non_increasing_steps(uid_values)
        if non_increasing > 0:
            raise ValueError(
                f"PDF uid not strictly increasing in {filepath}: "
                f"{non_increasing}"
            )

        info = {
            "rows": n_rows,
            "first_uid": int(uid_values[0]),
            "last_uid": int(uid_values[-1]),
        }
        if return_uid:
            info["uid"] = uid_values
        return info


def _validate_intermediate_shard(main_file: str, pdf_file: str) -> dict:
    """Validate that a single shard's main/PDF UIDs match."""
    if not os.path.exists(main_file) or not os.path.exists(pdf_file):
        raise FileNotFoundError(
            f"Missing intermediate shard files: "
            f"{os.path.basename(main_file)}, {os.path.basename(pdf_file)}"
        )

    main_info = _fits_uid_info(main_file, return_uid=True)
    pdf_info = _pdf_uid_info(pdf_file, return_uid=True)

    if main_info["rows"] != pdf_info["rows"]:
        raise ValueError(
            f"Row count mismatch in {os.path.basename(main_file)}: "
            f"main={main_info['rows']}, pdf={pdf_info['rows']}"
        )

    if not np.array_equal(main_info["uid"], pdf_info["uid"]):
        raise ValueError(
            f"UID array mismatch in {os.path.basename(main_file)}"
        )

    return {
        "rows": main_info["rows"],
        "first_uid": main_info["first_uid"],
        "last_uid": main_info["last_uid"],
    }


def _validate_intermediate_shards(
    merge_main_dir: str,
    merge_pdf_dir: str,
    cfg: dict,
    require_global_uid: bool = False,
) -> int:
    """Validate all 72 intermediate shards; return total row count."""
    expected_next_uid = None
    total_rows = 0

    for fid in range(N_RA_BINS):
        ra_start = fid * MERGE_RA_STEP
        ra_end = (fid + 1) * MERGE_RA_STEP
        main_file, pdf_file = _intermediate_paths(
            merge_main_dir, merge_pdf_dir, ra_start, ra_end, cfg
        )
        main_info = _validate_intermediate_shard(main_file, pdf_file)
        total_rows += main_info["rows"]

        if main_info["rows"] == 0:
            continue

        if (
            require_global_uid
            and expected_next_uid is not None
            and main_info["first_uid"] != expected_next_uid
        ):
            raise ValueError(
                f"Non-contiguous intermediate uid range before "
                f"RA{ra_start:03d}_{ra_end:03d}: expected "
                f"{expected_next_uid}, got {main_info['first_uid']}"
            )

        expected_next_uid = main_info["last_uid"] + 1

    return total_rows


def _finalize_intermediate_uids(
    merge_main_dir: str,
    merge_pdf_dir: str,
    cfg: dict,
) -> int:
    """Assign globally contiguous UIDs across all 72 intermediate shards.

    Reads each shard's row count, computes a cumulative offset, then
    overwrites the ``uid`` column in every main FITS and PDF HDF5
    file in-place.  Finishes with a full validation pass that
    asserts global contiguity.
    """
    shard_infos = []
    uid_offset = 0

    for fid in range(N_RA_BINS):
        ra_start = fid * MERGE_RA_STEP
        ra_end = (fid + 1) * MERGE_RA_STEP
        main_file, pdf_file = _intermediate_paths(
            merge_main_dir, merge_pdf_dir, ra_start, ra_end, cfg
        )
        shard_info = _validate_intermediate_shard(main_file, pdf_file)
        shard_info["main_file"] = main_file
        shard_info["pdf_file"] = pdf_file
        shard_info["uid_start"] = uid_offset
        shard_infos.append(shard_info)
        uid_offset += shard_info["rows"]

    for shard_info in shard_infos:
        rows = shard_info["rows"]
        if rows == 0:
            continue

        uid_values = np.arange(
            shard_info["uid_start"],
            shard_info["uid_start"] + rows,
            dtype=np.int64,
        )

        with fits.open(shard_info["main_file"], mode="update", memmap=True) as hdul:
            hdul[1].data["uid"][:] = uid_values
            hdul.flush()

        with h5py.File(shard_info["pdf_file"], "r+") as f:
            f["data"]["uid"][...] = uid_values

    _validate_intermediate_shards(
        merge_main_dir, merge_pdf_dir, cfg, require_global_uid=True
    )
    return uid_offset


# ---------------------------------------------------------------------------
# Cross-match column resolution
# ---------------------------------------------------------------------------


def _combine_preferred_columns(
    df_matched: pd.DataFrame,
    prefer_1: pd.Series,
    columns: list,
) -> pd.DataFrame:
    """Pick each column value from table 1 or table 2 based on ``prefer_1``.

    STILTS ``1or2`` join suffixes matched columns with ``_1`` /
    ``_2``. For each column in ``columns`` we select the table-1
    value when ``prefer_1`` is ``True``, table-2 otherwise. Falls
    back to the unsuffixed column name when only one table contributed
    a column (i.e. non-overlapping schemas).
    """
    result = pd.DataFrame(index=df_matched.index)
    for col in columns:
        col_1 = f"{col}_1"
        col_2 = f"{col}_2"
        if col_1 in df_matched.columns and col_2 in df_matched.columns:
            result[col] = np.where(prefer_1, df_matched[col_1], df_matched[col_2])
        elif col_1 in df_matched.columns:
            result[col] = df_matched[col_1]
        elif col_2 in df_matched.columns:
            result[col] = df_matched[col_2]
        elif col in df_matched.columns:
            result[col] = df_matched[col]
        else:
            raise KeyError(f"Column {col} not found in cross-match output")
    return result


def _select_preferred_pdf(
    df_matched: pd.DataFrame,
    prefer_1: pd.Series,
    df_ls: pd.DataFrame,
    df_ps: pd.DataFrame,
) -> np.ndarray:
    """Build the merged PDF array by picking from LSDR10 or PS1DR2 per row.

    Uses the ``uid_ls`` / ``uid_ps`` columns in ``df_matched`` to
    look up the corresponding row in the original per-survey
    DataFrames and fetch its uint16 PDF vector.
    """
    n_rows = len(df_matched)
    pdf_array = np.zeros((n_rows, PDF_BINS), dtype=np.uint16)

    use_ls = prefer_1.to_numpy(dtype=bool)
    use_ps = ~use_ls

    if len(df_ls) > 0:
        ls_lookup = pd.Series(
            np.arange(len(df_ls), dtype=np.int64), index=df_ls["uid_ls"]
        )
        ls_pdf = np.stack(
            df_ls["z_phot_pdf"].map(
                lambda v: np.asarray(v, dtype=np.uint16)
            ),
            axis=0,
        )
        if "uid_ls" in df_matched.columns:
            uid_ls = df_matched["uid_ls"]
        else:
            uid_ls = df_matched.get(
                "uid_ls_1",
                pd.Series(index=df_matched.index, dtype=df_ls["uid_ls"].dtype),
            )
        ls_rows = uid_ls.map(ls_lookup)
        if use_ls.any():
            if ls_rows[use_ls].isna().any():
                raise ValueError(
                    "Missing LSDR10 PDF rows after cross-match"
                )
            pdf_array[use_ls] = ls_pdf[
                ls_rows[use_ls].to_numpy(dtype=np.int64)
            ]
    elif use_ls.any():
        raise ValueError(
            "Cross-match selected LSDR10 rows but LSDR10 input is empty"
        )

    if len(df_ps) > 0:
        ps_lookup = pd.Series(
            np.arange(len(df_ps), dtype=np.int64), index=df_ps["uid_ps"]
        )
        ps_pdf = np.stack(
            df_ps["z_phot_pdf"].map(
                lambda v: np.asarray(v, dtype=np.uint16)
            ),
            axis=0,
        )
        if "uid_ps" in df_matched.columns:
            uid_ps = df_matched["uid_ps"]
        else:
            uid_ps = df_matched.get(
                "uid_ps_2",
                pd.Series(index=df_matched.index, dtype=df_ps["uid_ps"].dtype),
            )
        ps_rows = uid_ps.map(ps_lookup)
        if use_ps.any():
            if ps_rows[use_ps].isna().any():
                raise ValueError(
                    "Missing PS1DR2 PDF rows after cross-match"
                )
            pdf_array[use_ps] = ps_pdf[
                ps_rows[use_ps].to_numpy(dtype=np.int64)
            ]
    elif use_ps.any():
        raise ValueError(
            "Cross-match selected PS1DR2 rows but PS1DR2 input is empty"
        )

    return pdf_array


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def merge_surveys() -> None:
    """Build 72 intermediate 5-degree merge shards from per-survey outputs.

    Each shard consists of a main FITS file (scalar columns only) and
    a PDF HDF5 file (``uid`` + ``z_phot_pdf``). The PDF is kept out
    of the STILTS cross-match path to avoid blowing up the temporary
    FITS tables. After all shards are written, UIDs are back-filled
    in a single sequential pass to guarantee global contiguity.
    """
    cfg = load_config()
    ls_dir = cfg["paths"]["lsdr10"]["output"]
    ps_dir = cfg["paths"]["ps1dr2"]["output"]
    merge_main_dir, merge_pdf_dir, _ = _catalog_output_dirs(cfg)

    ls_template = cfg["templates"]["lsdr10_output"]
    ps_template = cfg["templates"]["ps1dr2_output"]

    print("=" * 60)
    print("Merging LSDR10 + PS1DR2 into intermediate main/pdf shards")
    print("=" * 60)

    failed_shards = []

    for fid in range(N_RA_BINS):
        ra_start = fid * MERGE_RA_STEP
        ra_end = (fid + 1) * MERGE_RA_STEP
        main_output, pdf_output = _intermediate_paths(
            merge_main_dir, merge_pdf_dir, ra_start, ra_end, cfg
        )

        # Skip existing valid shards.
        if os.path.exists(main_output) and os.path.exists(pdf_output):
            try:
                shard_info = _validate_intermediate_shard(
                    main_output, pdf_output
                )
                print(
                    f"  [{fid:02d}] Skipped (exists) -> "
                    f"{shard_info['rows']:,} sources"
                )
                continue
            except Exception as e:
                print(
                    f"  [{fid:02d}] Removing invalid existing shard -> {e}"
                )
                _remove_files_if_exist(main_output, pdf_output)

        try:
            ls_file = os.path.join(ls_dir, ls_template.format(fid=fid))
            ps_file = os.path.join(ps_dir, ps_template.format(fid=fid))

            # Filter to galaxy_prob > 0.5.
            df_ls = read_table(ls_file)
            df_ls = df_ls[df_ls["galaxy_prob"] > 0.5].reset_index(drop=True)

            df_ps = read_table(ps_file)
            df_ps = df_ps[df_ps["galaxy_prob"] > 0.5].reset_index(drop=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_ls = os.path.join(tmpdir, "ls.fits")
                tmp_ps = os.path.join(tmpdir, "ps.fits")
                tmp_out = os.path.join(tmpdir, "matched.fits")

                save_fits(
                    df_ls[["uid_ls", *CATALOG_SCALAR_COLUMNS]], tmp_ls
                )
                save_fits(
                    df_ps[["uid_ps", *CATALOG_SCALAR_COLUMNS]], tmp_ps
                )

                success = stilts_crossmatch(
                    input1=tmp_ls,
                    input2=tmp_ps,
                    output_file=tmp_out,
                    join="1or2",
                    find="best",
                    match_radius=1.0,
                    verbose=False,
                )

                if not success:
                    raise RuntimeError(
                        f"Cross-match failed for "
                        f"RA{ra_start:03d}_{ra_end:03d}"
                    )

                df_matched = read_table(tmp_out)

            # Lower priority number wins; prefer LSDR10 (table 1).
            p1 = df_matched.get("priority_1", pd.Series(dtype="float64"))
            p2 = df_matched.get("priority_2", pd.Series(dtype="float64"))
            prefer_1 = p2.isna() | (p1.notna() & (p1 <= p2))

            main_result = _combine_preferred_columns(
                df_matched, prefer_1, CATALOG_SCALAR_COLUMNS
            )
            pdf_array = _select_preferred_pdf(
                df_matched, prefer_1, df_ls, df_ps
            )

            for col in ("priority", "survey"):
                if col in main_result.columns:
                    main_result[col] = main_result[col].astype(np.int16)

            # Write shard-local UIDs; globally contiguous UIDs are
            # assigned after all shards succeed.
            n = len(main_result)
            uid = np.arange(n, dtype=np.int64)
            main_result.insert(0, "uid", uid)
            pdf_result = pd.DataFrame({
                "uid": uid,
                "z_phot_pdf": list(pdf_array),
            })

            save_table(main_result[CATALOG_MAIN_COLUMNS], main_output)
            save_table(pdf_result[CATALOG_PDF_COLUMNS], pdf_output)
            print(
                f"  [{fid:02d}] {n:,} sources -> "
                f"{os.path.basename(main_output)}, "
                f"{os.path.basename(pdf_output)}"
            )

            del df_ls, df_ps, df_matched, main_result, pdf_result, pdf_array
            gc.collect()
        except Exception as e:
            _remove_files_if_exist(main_output, pdf_output)
            failed_shards.append((ra_start, ra_end, str(e)))
            print(
                f"  [{fid:02d}] Failed: "
                f"RA{ra_start:03d}_{ra_end:03d} -> {e}"
            )
            gc.collect()

    if failed_shards:
        failed_text = ", ".join(
            f"RA{ra_start:03d}_{ra_end:03d}"
            for ra_start, ra_end, _ in failed_shards
        )
        raise RuntimeError(
            "Merge completed with failed shards; successful shards were "
            "kept. Rerun merge after fixing inputs. Failed shards: "
            + failed_text
        )

    total_rows = _finalize_intermediate_uids(
        merge_main_dir, merge_pdf_dir, cfg
    )
    print(f"\nIntermediate merge completed! Total UIDs assigned: {total_rows:,}")

"""Publish stage: regroup intermediate shards into release volumes and verify.

Two entry points:

* :func:`publish_catalog` — concatenates six consecutive 5-degree
  intermediate merge shards into one 30-degree publish volume (12
  volumes total covering the full 0–360 degree RA range). Each
  volume is a main FITS file and a PDF HDF5 file.
* :func:`check_published_catalog` — validates every publish volume
  for structural integrity, column correctness, UID contiguity,
  coordinate range, quantile ordering, and PDF row-sum fidelity.
"""

from __future__ import annotations

import gc
import os

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

from .common import (
    CATALOG_MAIN_COLUMNS,
    CATALOG_PDF_COLUMNS,
    MERGE_RA_STEP,
    N_RA_BINS,
    PDF_BINS,
    PDF_UINT16_SCALE,
    PUBLISH_RA_STEP,
    load_config,
    read_table,
    save_table,
)
from .merge import (
    _catalog_output_dirs,
    _count_non_increasing_steps,
    _finalize_intermediate_uids,
    _fits_uid_info,
    _intermediate_paths,
    _remove_files_if_exist,
)

__all__ = ["publish_catalog", "check_published_catalog"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _publish_paths(publish_dir: str, ra_start: int, ra_end: int, cfg: dict):
    """Return (main_fits_path, pdf_hdf5_path) for a publish volume."""
    tmpl_main = cfg["templates"]["publish_main"]
    tmpl_pdf = cfg["templates"]["publish_pdf"]
    return (
        os.path.join(
            publish_dir, tmpl_main.format(ra_start=ra_start, ra_end=ra_end)
        ),
        os.path.join(
            publish_dir, tmpl_pdf.format(ra_start=ra_start, ra_end=ra_end)
        ),
    )


def _concat_tables(filepaths: list) -> pd.DataFrame:
    frames = [read_table(fp) for fp in filepaths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


def publish_catalog() -> None:
    """Regroup 72 intermediate 5-degree shards into 12 publish volumes.

    Before regrouping, UIDs are finalized so they are globally
    contiguous. Each publish volume is sorted by UID. A row-count
    and UID-range sanity check runs at the end of each group.
    """
    cfg = load_config()
    merge_main_dir = cfg["paths"]["merge"]["main"]
    merge_pdf_dir = cfg["paths"]["merge"]["pdf"]
    _finalize_intermediate_uids(merge_main_dir, merge_pdf_dir, cfg)
    _, _, publish_dir = _catalog_output_dirs(cfg)

    print("=" * 60)
    print("Publishing merged catalog into 12 main FITS + 12 PDF HDF5 volumes")
    print("=" * 60)

    failed_groups = []

    for ra_start in range(0, 360, PUBLISH_RA_STEP):
        ra_end = ra_start + PUBLISH_RA_STEP
        main_output, pdf_output = _publish_paths(
            publish_dir, ra_start, ra_end, cfg
        )
        try:
            main_inputs = []
            pdf_inputs = []
            for shard_start in range(ra_start, ra_end, MERGE_RA_STEP):
                shard_end = shard_start + MERGE_RA_STEP
                main_file, pdf_file = _intermediate_paths(
                    merge_main_dir, merge_pdf_dir,
                    shard_start, shard_end, cfg,
                )
                if not os.path.exists(main_file) or not os.path.exists(pdf_file):
                    raise FileNotFoundError(
                        f"Missing intermediate shard for "
                        f"RA{shard_start:03d}_{shard_end:03d}"
                    )
                main_inputs.append(main_file)
                pdf_inputs.append(pdf_file)

            main_result = (
                _concat_tables(main_inputs)
                .sort_values("uid")
                .reset_index(drop=True)
            )
            pdf_result = (
                _concat_tables(pdf_inputs)
                .sort_values("uid")
                .reset_index(drop=True)
            )

            if len(main_result) != len(pdf_result):
                raise ValueError(
                    f"Row count mismatch in "
                    f"RA{ra_start:03d}_{ra_end:03d} publish group"
                )
            if not np.array_equal(
                main_result["uid"].to_numpy(), pdf_result["uid"].to_numpy()
            ):
                raise ValueError(
                    f"UID mismatch in "
                    f"RA{ra_start:03d}_{ra_end:03d} publish group"
                )

            if len(main_result) > 0:
                first_info = _fits_uid_info(main_inputs[0])
                last_info = _fits_uid_info(main_inputs[-1])
                first_uid = int(main_result["uid"].iloc[0])
                last_uid = int(main_result["uid"].iloc[-1])
                if (
                    first_uid != first_info["first_uid"]
                    or last_uid != last_info["last_uid"]
                ):
                    raise ValueError(
                        f"Unexpected publish uid range in "
                        f"RA{ra_start:03d}_{ra_end:03d}: expected "
                        f"({first_info['first_uid']}, "
                        f"{last_info['last_uid']}), got "
                        f"({first_uid}, {last_uid})"
                    )

            save_table(main_result[CATALOG_MAIN_COLUMNS], main_output)
            save_table(pdf_result[CATALOG_PDF_COLUMNS], pdf_output)
            print(
                f"  RA{ra_start:03d}_{ra_end:03d}: "
                f"{len(main_result):,} sources -> "
                f"{os.path.basename(main_output)}, "
                f"{os.path.basename(pdf_output)}"
            )

            del main_result, pdf_result
            gc.collect()
        except Exception as e:
            _remove_files_if_exist(main_output, pdf_output)
            failed_groups.append((ra_start, ra_end, str(e)))
            print(
                f"  Publish failed: RA{ra_start:03d}_{ra_end:03d} -> {e}"
            )
            gc.collect()

    if failed_groups:
        failed_text = ", ".join(
            f"RA{ra_start:03d}_{ra_end:03d}"
            for ra_start, ra_end, _ in failed_groups
        )
        raise RuntimeError(
            "Publish completed with failed groups; successful volumes "
            "were kept. Rerun publish after fixing the failed groups. "
            f"Failed groups: {failed_text}"
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check_published_catalog() -> None:
    """Validate all 12 publish volumes for structural integrity.

    Checks performed per volume:

    * Main FITS: column schema, UID strict increase, RA/dec range,
      NaN absence, z_phot_std >= 0, quantile ordering, survey and
      priority validity, galaxy_prob > 0.5.
    * PDF HDF5: ``photoz_table_v2`` structure, UID strict increase,
      PDF shape and dtype, each row sums to :data:`PDF_UINT16_SCALE`.
    * Cross-file: main/PDF row count and UID array equality.
    * Cross-volume: UID contiguity (last UID + 1 == first UID of next
      volume).
    """
    cfg = load_config()
    publish_dir = cfg["paths"]["publish"]

    print("=" * 60)
    print("Checking published catalog volumes")
    print("=" * 60)

    n_ok = 0
    n_bad = 0
    total_rows = 0
    expected_next_uid = None

    for ra_start in range(0, 360, PUBLISH_RA_STEP):
        ra_end = ra_start + PUBLISH_RA_STEP
        main_file, pdf_file = _publish_paths(
            publish_dir, ra_start, ra_end, cfg
        )
        issues: list[str] = []
        rows = 0
        next_expected_uid = None
        main_uid = None
        pdf_uid = None
        pdf_rows = 0

        # ----- main FITS -----
        if not os.path.exists(main_file):
            issues.append(
                f"missing main file: {os.path.basename(main_file)}"
            )
        else:
            try:
                with fits.open(main_file, memmap=True) as hdul:
                    if len(hdul) < 2:
                        raise ValueError(
                            "missing binary table extension"
                        )

                    hdu = hdul[1]
                    columns = list(hdu.columns.names)
                    if columns != CATALOG_MAIN_COLUMNS:
                        issues.append(
                            f"unexpected main columns: {columns}"
                        )

                    data = hdu.data
                    rows = int(hdu.header.get("NAXIS2", 0))
                    total_rows += rows

                    if rows > 0:
                        main_uid = np.asarray(data["uid"])
                        ra = np.asarray(data["ra"])
                        dec = np.asarray(data["dec"])
                        z_std = np.asarray(data["z_phot_std"])
                        z_l95 = np.asarray(data["z_phot_l95"])
                        z_l68 = np.asarray(data["z_phot_l68"])
                        z_med = np.asarray(data["z_phot_median"])
                        z_u68 = np.asarray(data["z_phot_u68"])
                        z_u95 = np.asarray(data["z_phot_u95"])
                        priority = np.asarray(data["priority"])
                        survey = np.asarray(data["survey"])
                        galaxy_prob = np.asarray(data["galaxy_prob"])

                        non_inc = _count_non_increasing_steps(main_uid)
                        if non_inc > 0:
                            issues.append(
                                f"main uid not strictly increasing: "
                                f"{non_inc}"
                            )

                        first_uid = int(main_uid[0])
                        last_uid = int(main_uid[-1])
                        if (
                            expected_next_uid is not None
                            and first_uid != expected_next_uid
                        ):
                            issues.append(
                                f"main uid range break: expected "
                                f"{expected_next_uid}, got {first_uid}"
                            )
                        next_expected_uid = last_uid + 1

                        bad_ra = (ra < ra_start) | (ra >= ra_end)
                        if ra_end == 360:
                            bad_ra = (ra < ra_start) | (ra > ra_end)
                        if np.any(bad_ra):
                            issues.append(
                                f"ra out of range rows: "
                                f"{int(np.sum(bad_ra))}"
                            )

                        bad_dec = (dec < -90.0) | (dec > 90.0)
                        if np.any(bad_dec):
                            issues.append(
                                f"dec out of range rows: "
                                f"{int(np.sum(bad_dec))}"
                            )

                        for name, arr in [
                            ("ra", ra), ("dec", dec),
                            ("z_phot_std", z_std),
                            ("galaxy_prob", galaxy_prob),
                        ]:
                            nan_count = int(np.isnan(arr).sum())
                            if nan_count > 0:
                                issues.append(
                                    f"{name} NaN rows: {nan_count}"
                                )

                        if np.any(z_std < 0):
                            issues.append(
                                f"negative z_phot_std rows: "
                                f"{int(np.sum(z_std < 0))}"
                            )

                        quantile_bad = ~(
                            (z_l95 <= z_l68)
                            & (z_l68 <= z_med)
                            & (z_med <= z_u68)
                            & (z_u68 <= z_u95)
                        )
                        if np.any(quantile_bad):
                            issues.append(
                                f"invalid quantile ordering rows: "
                                f"{int(np.sum(quantile_bad))}"
                            )

                        survey_bad = ~np.isin(survey, [0, 1])
                        if np.any(survey_bad):
                            issues.append(
                                f"invalid survey rows: "
                                f"{int(np.sum(survey_bad))}"
                            )

                        ls_prio_bad = (survey == 0) & (
                            ~np.isin(priority, [1, 2, 3])
                        )
                        if np.any(ls_prio_bad):
                            issues.append(
                                f"invalid LSDR10 priority rows: "
                                f"{int(np.sum(ls_prio_bad))}"
                            )

                        ps_prio_bad = (survey == 1) & (
                            ~np.isin(priority, [4, 5])
                        )
                        if np.any(ps_prio_bad):
                            issues.append(
                                f"invalid PS1DR2 priority rows: "
                                f"{int(np.sum(ps_prio_bad))}"
                            )

                        galaxy_bad = galaxy_prob <= 0.5
                        if np.any(galaxy_bad):
                            issues.append(
                                f"galaxy_prob <= 0.5 rows: "
                                f"{int(np.sum(galaxy_bad))}"
                            )
            except Exception as e:
                issues.append(f"main file read/check failed: {e}")

        # ----- PDF HDF5 -----
        if not os.path.exists(pdf_file):
            issues.append(
                f"missing pdf file: {os.path.basename(pdf_file)}"
            )
        else:
            try:
                with h5py.File(pdf_file, "r") as f:
                    if "meta" not in f or "data" not in f:
                        raise ValueError("missing /meta or /data group")

                    meta = f["meta"]
                    data_group = f["data"]

                    if "columns" not in meta:
                        issues.append("pdf meta.columns missing")
                        pdf_columns: list[str] = []
                    else:
                        pdf_columns = [
                            col.decode() if isinstance(col, bytes) else col
                            for col in meta["columns"][...]
                        ]
                        if pdf_columns != CATALOG_PDF_COLUMNS:
                            issues.append(
                                f"unexpected pdf columns: {pdf_columns}"
                            )

                    if meta.attrs.get("format") != "photoz_table_v2":
                        issues.append(
                            f"unexpected pdf format: "
                            f"{meta.attrs.get('format')}"
                        )

                    if (
                        "uid" not in data_group
                        or "z_phot_pdf" not in data_group
                    ):
                        raise ValueError(
                            "missing uid or z_phot_pdf dataset"
                        )

                    uid_ds = data_group["uid"]
                    pdf_ds = data_group["z_phot_pdf"]
                    pdf_rows = int(uid_ds.shape[0])

                    if uid_ds.ndim != 1:
                        issues.append(
                            f"uid dataset dimension invalid: "
                            f"{uid_ds.shape}"
                        )
                    if pdf_ds.shape != (pdf_rows, PDF_BINS):
                        issues.append(
                            f"pdf dataset shape invalid: {pdf_ds.shape}"
                        )
                    if pdf_ds.dtype != np.uint16:
                        issues.append(
                            f"pdf dataset dtype invalid: {pdf_ds.dtype}"
                        )

                    if pdf_rows > 0:
                        pdf_uid = uid_ds[...]
                        non_inc = _count_non_increasing_steps(pdf_uid)
                        if non_inc > 0:
                            issues.append(
                                f"pdf uid not strictly increasing: "
                                f"{non_inc}"
                            )

                        bad_pdf_rows = 0
                        for start in range(0, pdf_rows, 200_000):
                            end = min(start + 200_000, pdf_rows)
                            row_sums = np.sum(
                                pdf_ds[start:end],
                                axis=1,
                                dtype=np.uint32,
                            )
                            bad_pdf_rows += int(
                                np.sum(row_sums != PDF_UINT16_SCALE)
                            )
                        if bad_pdf_rows > 0:
                            issues.append(
                                f"pdf row sum != "
                                f"{int(PDF_UINT16_SCALE)} rows: "
                                f"{bad_pdf_rows}"
                            )
            except Exception as e:
                issues.append(f"pdf file read/check failed: {e}")

        # ----- cross-file consistency -----
        if rows != pdf_rows:
            issues.append(
                f"main/pdf row mismatch: main={rows}, pdf={pdf_rows}"
            )

        if (
            main_uid is not None
            and pdf_uid is not None
            and not np.array_equal(main_uid, pdf_uid)
        ):
            issues.append("main/pdf uid arrays differ")

        label = "OK" if not issues else "BAD"
        print(
            f"  [{label}] RA{ra_start:03d}_{ra_end:03d} "
            f"main+pdf rows={rows:,}"
        )
        for issue in issues:
            print(f"    - {issue}")

        if issues:
            n_bad += 1
        else:
            n_ok += 1
            expected_next_uid = next_expected_uid

        del main_uid, pdf_uid
        gc.collect()

    print(
        f"Summary publish main+pdf: ok={n_ok}, bad={n_bad}, "
        f"rows={total_rows:,}, volumes={n_ok + n_bad}"
    )
    if n_bad > 0:
        raise SystemExit(1)

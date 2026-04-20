#!/usr/bin/env python3
"""
Download LSDR10 positive / negative samples for galaxy-classifier training.

Two sample definitions are used:

* positive sample (extended-source / galaxy proxy):
  ``type != 'PSF' AND shape_r > 0.5``
* negative sample (point-source proxy):
  ``type == 'PSF' OR (Gaia match with PM_SNR >= 3)``

The script downloads samples region by region from NOIRLab Data Lab,
merges the partial results, de-duplicates on ``ls_id``, and optionally
down-samples to the requested cap.

Dependencies and setup
----------------------
This script uses the NOIRLab Data Lab client distributed through the
``astro-datalab`` package, which provides the ``dl`` Python module::

    pip install astro-datalab

To use an authenticated Data Lab account, export credentials before
running the script::

    export DATALAB_USER=<YOUR_DATALAB_USERNAME>
    export DATALAB_PASSWORD=<YOUR_DATALAB_PASSWORD>

If credentials are not provided, the script falls back to anonymous
login.

Typical usage::

    python download_pos_neg_sample.py --type positive --limit 5000000
    python download_pos_neg_sample.py --type negative --limit 5000000
    python download_pos_neg_sample.py --type both --limit 5000000
"""

import os
import argparse
import numpy as np
import pandas as pd
from dl import authClient as ac, queryClient as qc
from astropy.table import Table

# Configuration constants
QUERY_TIMEOUT = 3600
OUTPUT_DIR = "./galaxyClf"

DATALAB_USER = os.environ.get("DATALAB_USER", "<YOUR_DATALAB_USERNAME>")
DATALAB_PASSWORD = os.environ.get(
    "DATALAB_PASSWORD", "<YOUR_DATALAB_PASSWORD>"
)


def login_datalab():
    """Log in to NOIRLab Data Lab and return an auth token.

    When no credentials are configured, fall back to anonymous access.
    """
    if (
        DATALAB_USER.startswith("<")
        or DATALAB_PASSWORD.startswith("<")
    ):
        print("No Data Lab credentials configured; using anonymous login.")
        return ac.login("anonymous")

    print(f"Logging in to Data Lab as {DATALAB_USER!r} ...")
    return ac.login(user=DATALAB_USER, password=DATALAB_PASSWORD)


TOKEN = login_datalab()


def get_positive_query(ra_l, ra_u, dec_l, dec_u, limit=None):
    """
    Query the positive sample: extended-source / galaxy proxy.

    Selection: ``type != 'PSF' AND shape_r > 0.5`` arcsec.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            t.ls_id,
            t.ra, t.dec,
            t.type,
            t.shape_r,
            t.shape_e1, t.shape_e2,
            t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
            t.dered_mag_w1, t.dered_mag_w2,
            t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2,
            t.gaia_phot_g_mean_mag,
            t.gaia_phot_g_mean_flux_over_error,
            t.gaia_phot_bp_mean_mag,
            t.gaia_phot_bp_mean_flux_over_error,
            t.gaia_phot_rp_mean_mag,
            t.gaia_phot_rp_mean_flux_over_error,
            t.pmra, t.pmdec,
            t.pmra_ivar, t.pmdec_ivar,
            t.parallax
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND t.type != 'PSF'
            AND t.shape_r > 0.5
            AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
            AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}
            -- Valid magnitudes
            AND t.dered_mag_g > 0 AND t.dered_mag_g < 23
            AND t.dered_mag_r > 0 AND t.dered_mag_r < 23
            AND t.dered_mag_z > 0 AND t.dered_mag_z < 23
            AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 23
            AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 23
            -- Quality cuts: low neighbour contamination, limited masking,
            -- source mostly inside the image footprint, and no allmask bits
            AND t.fracflux_g < 0.5 AND t.fracflux_r < 0.5 AND t.fracflux_z < 0.5
            AND t.fracmasked_g < 0.3 AND t.fracmasked_r < 0.3 AND t.fracmasked_z < 0.3
            AND t.fracin_g > 0.8 AND t.fracin_r > 0.8 AND t.fracin_z > 0.8
            AND t.allmask_g = 0 AND t.allmask_r = 0 AND t.allmask_z = 0
        {limit_clause}
    """
    return query


def get_negative_query(ra_l, ra_u, dec_l, dec_u, limit=None):
    """
    Query the negative sample: point-source proxy (star / QSO-like).

    Selection: ``type == 'PSF'`` or a Gaia-matched source with
    ``PM_SNR >= 3``, where::

        PM_SNR = sqrt(pmra^2 + pmdec^2) /
                 sqrt(1/pmra_ivar + 1/pmdec_ivar)
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            t.ls_id,
            t.ra, t.dec,
            t.type,
            t.shape_r,
            t.shape_e1, t.shape_e2,
            t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
            t.dered_mag_w1, t.dered_mag_w2,
            t.snr_g, t.snr_r, t.snr_i, t.snr_z, t.snr_w1, t.snr_w2,
            t.gaia_phot_g_mean_mag,
            t.gaia_phot_g_mean_flux_over_error,
            t.gaia_phot_bp_mean_mag,
            t.gaia_phot_bp_mean_flux_over_error,
            t.gaia_phot_rp_mean_mag,
            t.gaia_phot_rp_mean_flux_over_error,
            t.pmra, t.pmdec,
            t.pmra_ivar, t.pmdec_ivar,
            t.parallax
        FROM ls_dr10.tractor AS t
        WHERE
            t.brick_primary = 1
            AND (
                t.type = 'PSF'
                OR (
                    t.gaia_phot_g_mean_mag > 0
                    AND t.pmra_ivar > 0 AND t.pmdec_ivar > 0
                    AND (t.pmra * t.pmra + t.pmdec * t.pmdec) >= 9.0 * (1.0/t.pmra_ivar + 1.0/t.pmdec_ivar)
                )
            )
            AND t.ra >= {ra_l:.6f} AND t.ra < {ra_u:.6f}
            AND t.dec >= {dec_l:.6f} AND t.dec < {dec_u:.6f}
            AND t.dered_mag_g > 0 AND t.dered_mag_g < 23
            AND t.dered_mag_r > 0 AND t.dered_mag_r < 23
            AND t.dered_mag_z > 0 AND t.dered_mag_z < 23
            AND t.dered_mag_w1 > 0 AND t.dered_mag_w1 < 23
            AND t.dered_mag_w2 > 0 AND t.dered_mag_w2 < 23
        {limit_clause}
    """
    return query


def count_samples(query_func, n_regions=10):
    """Estimate the total sample count by querying a few random regions."""
    print("Estimating total sample count...")

    total_estimate = 0
    ra_size = 360 / n_regions
    dec_size = 180 / n_regions

    sample_regions = [
        (i * ra_size, (i + 1) * ra_size, -90 + j * dec_size, -90 + (j + 1) * dec_size)
        for i in range(n_regions) for j in range(n_regions)
    ]

    # Randomly sample a few regions for a rough extrapolation.
    np.random.seed(42)
    selected = np.random.choice(len(sample_regions), min(5, len(sample_regions)), replace=False)

    counts = []
    for idx in selected:
        ra_l, ra_u, dec_l, dec_u = sample_regions[idx]
        query = query_func(ra_l, ra_u, dec_l, dec_u)
        count_query = f"SELECT COUNT(*) FROM ({query}) AS subq"
        try:
            result = qc.query(TOKEN, sql=count_query, fmt='pandas', timeout=120)
            if result is not None and len(result) > 0:
                counts.append(result.iloc[0, 0])
        except:
            pass

    if counts:
        avg_per_region = np.mean(counts)
        total_estimate = int(avg_per_region * len(sample_regions))
        print(f"  Mean count per sampled region: {avg_per_region:.0f}")
        print(f"  Estimated total: ~{total_estimate:,}")

    return total_estimate


def download_samples(sample_type, total_limit, output_dir):
    """
    Download one sample class region by region.

    Args:
        sample_type: Either ``"positive"`` or ``"negative"``.
        total_limit: Maximum number of rows to keep after merge.
        output_dir: Output directory for the merged FITS file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sample_type == 'positive':
        query_func = get_positive_query
        output_file = os.path.join(output_dir, 'LSDR10_extendSource.fits')
        desc = "positive sample (extended-source proxy)"
    else:
        query_func = get_negative_query
        output_file = os.path.join(output_dir, 'LSDR10_pointSource.fits')
        desc = "negative sample (point-source proxy)"

    print(f"\n{'='*60}")
    print(f"Downloading {desc}")
    print(f"Target size: {total_limit:,}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    # Resume from an existing merged file when possible.
    if os.path.exists(output_file):
        existing = Table.read(output_file)
        print(f"Existing file found with {len(existing):,} rows")
        if len(existing) >= total_limit:
            print("Target already reached; skipping download.")
            return existing
        print(f"Continuing download; still need {total_limit - len(existing):,} rows")
        all_data = [existing.to_pandas()]
        current_count = len(existing)
    else:
        all_data = []
        current_count = 0

    # Download region by region without a per-region cap.
    n_ra = 20
    n_dec = 60

    ra_edges = np.linspace(0, 360, n_ra + 1)
    dec_edges = np.linspace(-90, 90, n_dec + 1)

    n_regions = n_ra * n_dec

    # Build the region list.
    if sample_type == 'positive':
        # Positive sample: use only high-Galactic-latitude regions to
        # reduce stellar contamination.
        regions = []
        for i in range(n_ra):
            for j in range(n_dec):
                ra_c = (ra_edges[i] + ra_edges[i + 1]) / 2
                dec_c = (dec_edges[j] + dec_edges[j + 1]) / 2
                # Approximate Galactic latitude using the north Galactic
                # pole coordinates.
                ra_rad = np.radians(ra_c)
                dec_rad = np.radians(dec_c)
                ra_ngp = np.radians(192.85)
                dec_ngp = np.radians(27.13)
                sin_b = (np.sin(dec_rad) * np.sin(dec_ngp)
                         + np.cos(dec_rad) * np.cos(dec_ngp) * np.cos(ra_rad - ra_ngp))
                glat = np.degrees(np.arcsin(sin_b))
                if abs(glat) > 20:
                    regions.append((i, j))
        print(
            f"Positive-sample region filter: |b| > 20 deg, "
            f"{len(regions)}/{n_regions} regions kept"
        )
    else:
        # Negative sample: use the full sky.
        regions = [(i, j) for i in range(n_ra) for j in range(n_dec)]
        print(f"Negative-sample region set: full sky, {len(regions)} regions")

    # Shuffle region order so early interruption still gives broad coverage.
    np.random.seed(42)
    np.random.shuffle(regions)

    ra_width = 360 / n_ra
    dec_width = 180 / n_dec
    print(
        f"Tiling strategy: {n_ra}x{n_dec} -> {len(regions)} regions "
        f"({ra_width:.1f} deg x {dec_width:.1f} deg each)\n"
    )

    for region_idx, (i, j) in enumerate(regions):
        if current_count >= total_limit:
            print(f"\nTarget {total_limit:,} reached; stopping download")
            break

        ra_l, ra_u = ra_edges[i], ra_edges[i + 1]
        dec_l, dec_u = dec_edges[j], dec_edges[j + 1]

        query = query_func(ra_l, ra_u, dec_l, dec_u)

        try:
            print(f"[{region_idx+1}/{len(regions)}] RA=[{ra_l:.0f},{ra_u:.0f}] Dec=[{dec_l:.0f},{dec_u:.0f}]", end=" ")

            result = qc.query(TOKEN, sql=query, fmt='pandas', timeout=QUERY_TIMEOUT)

            if result is not None and len(result) > 0:
                all_data.append(result)
                current_count += len(result)
                print(f"-> {len(result):,} rows (running total: {current_count:,})")
            else:
                print("-> 0 rows")

        except KeyboardInterrupt:
            print("\nInterrupted by user; saving downloaded rows so far ...")
            break
        except Exception as e:
            print(f"-> error: {e}")
            continue

        # Save an intermediate merged file every 50 regions.
        if (region_idx + 1) % 50 == 0 and all_data:
            print("\n  Saving intermediate result ... ", end="")
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=["ls_id"])
            table = Table.from_pandas(combined)
            table.write(output_file, format='fits', overwrite=True)
            print(f"saved {len(table):,} rows\n")

    # Final merge and write.
    if all_data:
        print("\nMerging downloaded chunks ...")
        combined = pd.concat(all_data, ignore_index=True)

        # De-duplicate on the unique Legacy Surveys source id.
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=["ls_id"])
        print(f"  De-duplicate: {before_dedup:,} -> {len(combined):,}")

        # Randomly down-sample if the merged table exceeds the requested cap.
        if len(combined) > total_limit:
            combined = combined.sample(n=total_limit, random_state=42)
            print(f"  Random down-sample: -> {len(combined):,}")

        # Write the merged table to disk.
        table = Table.from_pandas(combined)
        table.write(output_file, format='fits', overwrite=True)

        print(f"\n{'='*60}")
        print("Download complete")
        print(f"  Rows: {len(table):,}")
        print(f"  File: {output_file}")
        print(f"  Size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
        print(f"{'='*60}")

        return table
    else:
        print("No data downloaded")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download LSDR10 galaxy-classifier training samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--type", "-t",
        choices=["positive", "negative", "both"],
        default="both",
        help="Sample type to download (default: both).",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=5000000,
        help="Maximum number of rows to keep per class (default: 5000000).",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("LSDR10 galaxy-classifier sample download")
    print(f"{'#'*60}")
    print("\nPositive definition: type != 'PSF' AND shape_r > 0.5")
    print("Negative definition: type == 'PSF' OR (has_gaia AND PM_SNR >= 3)")
    print(f"Per-class cap: {args.limit:,}")
    print(f"Output directory: {args.output}\n")

    if args.type in ["positive", "both"]:
        download_samples("positive", args.limit, args.output)

    if args.type in ["negative", "both"]:
        download_samples("negative", args.limit, args.output)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()

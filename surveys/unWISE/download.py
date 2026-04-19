"""Bulk downloader for unWISE DR1 photometry via NOIRLab Datalab.

The sky is tiled into an RA/Dec grid (see :func:`make_seg`). For each
tile the row count is checked first; tiles exceeding ``max_rows`` are
automatically sub-split along RA. Downloads run in parallel using a
thread pool.

Selected columns include W1/W2 Vega magnitudes, fluxes with errors
(both regular and local-sky-subtracted), quality metrics (``qf``,
``rchi2``, ``fracflux``, ``fwhm``, ``nm``), and flag columns
(``flags_unwise``, ``flags_info``). Only primary detections
(``"primary" = 1``) are queried.

Dependencies and setup
----------------------
Requires the NOIRLab Datalab client library::

    pip install astro-datalab

Create a Datalab account at https://datalab.noirlab.edu/ and pass your
credentials to :func:`login_datalab`.

Typical usage::

    python download.py
"""

import warnings
warnings.simplefilter("ignore")

import os
import time
import random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_proxy():
    """Set HTTP(S) proxy environment variables.

    Replace the placeholder URLs with your own proxy address, or remove
    this call entirely if no proxy is needed.
    """
    os.environ['HTTPS_PROXY'] = '<YOUR_PROXY_URL>'
    os.environ['HTTP_PROXY'] = '<YOUR_PROXY_URL>'


def login_datalab(user, password):
    """Log in to NOIRLab Datalab and return an auth token."""
    from dl import authClient as ac
    token = ac.login(user=user, password=password)
    print("DataLab login successful")
    return token


def test_datalab_connectivity(token):
    """Run a small COUNT query to verify the Datalab connection."""
    from dl import queryClient as qc

    test_query = """
        SELECT COUNT(*)
        FROM unwise_dr1.object
        WHERE ra >= 0 AND ra < 0.1 AND dec >= 0 AND dec < 0.1
    """
    try:
        print("Testing DataLab connectivity...")
        result = qc.query(token, sql=test_query, fmt='pandas', timeout=60)
        if result is not None:
            count = result.iloc[0, 0]
            print(f"Connection OK, test region row count: {count:,}")
            return True
        else:
            print("Connection failed: no response")
            return False
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")
        return False


def make_seg(output_dir, n_ra, n_dec):
    """Build a uniform RA/Dec grid and save it as the download manifest."""
    ra_edges = np.linspace(0, 360, n_ra + 1)
    dec_edges = np.linspace(-90, 90, n_dec + 1)

    # Vectorised grid generation via meshgrid.
    ra_grid, dec_grid = np.meshgrid(range(n_ra), range(n_dec), indexing='ij')

    n_total = n_ra * n_dec
    arr = np.column_stack([
        np.arange(n_total),           # fid
        ra_edges[ra_grid.ravel()],    # ra_l
        ra_edges[ra_grid.ravel() + 1],# ra_u
        dec_edges[dec_grid.ravel()],  # dec_l
        dec_edges[dec_grid.ravel() + 1]# dec_u
    ])

    cols = ['fid', 'ra_l', 'ra_u', 'dec_l', 'dec_u']
    df = pd.DataFrame(arr, columns=cols)
    path = f'{output_dir}/readme.txt'
    df.to_csv(path, index=False)
    print(f"Segmentation manifest saved to {path}")
    return df


def count_region_data(token, ra_lo, ra_hi, dec_lo, dec_hi):
    """Query row count for a single RA/Dec tile."""
    from dl import queryClient as qc

    # Random delay to avoid overwhelming the proxy with concurrent requests.
    time.sleep(random.uniform(2, 8))

    count_query = f"""
    SELECT COUNT(*)
    FROM unwise_dr1.object
    WHERE
        "primary" = 1
        AND ra >= {ra_lo} AND ra < {ra_hi}
        AND dec >= {dec_lo} AND dec < {dec_hi}
    """
    result = qc.query(token, sql=count_query, fmt='pandas', timeout=600)
    return result.iloc[0, 0]


def download_region(token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, count, sub_idx=None, max_retries=3):
    """Download one tile (or sub-tile) to a FITS file, with retry logic."""
    from dl import queryClient as qc

    # Random delay to avoid concurrent SSL errors.
    time.sleep(random.uniform(2, 8))

    if sub_idx is None:
        output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}.fits")
        file_id = f"{idx:04d}"
    else:
        output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}_{sub_idx}.fits")
        file_id = f"{idx:04d}_{sub_idx}"

    if os.path.exists(output_file):
        return f"[{file_id}] (exists, skipped)"

    data_query = f"""
    SELECT
        unwise_objid, ra, dec, glon, glat,
        mag_w1_vg, mag_w2_vg, w1_w2_vg,

        -- Fluxes and errors
        flux_w1, flux_w2,
        dflux_w1, dflux_w2,
        fluxlbs_w1, fluxlbs_w2,
        dfluxlbs_w1, dfluxlbs_w2,

        -- Quality metrics
        qf_w1, qf_w2,
        rchi2_w1, rchi2_w2,
        fracflux_w1, fracflux_w2,
        fwhm_w1, fwhm_w2,
        nm_w1, nm_w2,

        -- Flags
        "primary",
        flags_unwise_w1, flags_unwise_w2,
        flags_info_w1, flags_info_w2

    FROM
        unwise_dr1.object
    WHERE
        "primary" = 1
        AND ra >= {ra_lo} AND ra < {ra_hi}
        AND dec >= {dec_lo} AND dec < {dec_hi}
    """

    for attempt in range(max_retries):
        try:
            qc.query(token, sql=data_query, fmt='fits', out=output_file, timeout=86400)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                return f"[{file_id}] {count:,} rows, {file_size_mb:.2f} MB"
            else:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return f"[{file_id}] download failed, empty file"

        except Exception as e:
            # Remove partial file before retrying.
            if os.path.exists(output_file):
                os.remove(output_file)

            if attempt < max_retries - 1:
                print(f"  [{file_id}] retry {attempt + 2}/{max_retries}")
                time.sleep(10)
            else:
                return f"[{file_id}] download failed ({max_retries} retries): {str(e)[:50]}"


def process_region(token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, max_rows=5000000):
    """Process one tile: check row count, sub-split if too large, then download."""

    # Skip the count query if the output file already exists.
    output_file = os.path.join(output_dir, f"unwise_dr1_{idx:04d}.fits")
    if os.path.exists(output_file):
        return [f"[{idx:04d}] (exists, skipped)"]

    try:
        count = count_region_data(token, ra_lo, ra_hi, dec_lo, dec_hi)

        results = []

        if count == 0:
            results.append(f"[{idx:04d}] empty region, skipped")
        elif count <= max_rows:
            result = download_region(token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, count)
            results.append(result)
        else:
            # Sub-split along RA when the tile exceeds max_rows.
            n_subdivisions = int(np.ceil(count / max_rows))

            sub_ra_bins = np.linspace(ra_lo, ra_hi, n_subdivisions + 1)

            for j in range(n_subdivisions):
                sub_ra_lo = sub_ra_bins[j]
                sub_ra_hi = sub_ra_bins[j + 1]
                sub_count = count_region_data(token, sub_ra_lo, sub_ra_hi, dec_lo, dec_hi)
                result = download_region(token, idx, sub_ra_lo, sub_ra_hi, dec_lo, dec_hi, output_dir, sub_count, sub_idx=j)
                results.append(result)

        return results

    except Exception as e:
        return [f"[{idx:04d}] error: {str(e)}"]


def run_download(token, output_dir, n_ra_segments=180, n_dec_segments=60, max_rows=5000000, max_workers=10):
    """Run the full download pipeline with a thread pool."""
    os.makedirs(output_dir, exist_ok=True)

    seg_df = make_seg(output_dir, n_ra_segments, n_dec_segments)

    tasks = []
    for idx, row in seg_df.iterrows():
        tasks.append((int(row['fid']), row['ra_l'], row['ra_u'], row['dec_l'], row['dec_u']))

    print(f"Starting download with {max_workers} threads...")
    print(f"Total regions: {len(tasks)} ({n_ra_segments} x {n_dec_segments})")
    print(f"Max rows per file: {max_rows:,}\n")

    all_results = []
    completed = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task in tasks:
            idx, ra_lo, ra_hi, dec_lo, dec_hi = task
            future = executor.submit(process_region, token, idx, ra_lo, ra_hi, dec_lo, dec_hi, output_dir, max_rows)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results = future.result()
                for result in results:
                    print(result)
                    all_results.append(result)
            except Exception as e:
                print(f"[{idx:04d}] thread error: {str(e)}")

            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)\n")

    print("\nDownload complete!")
    print(f"Total files generated: {len(all_results)}")
    return all_results


def main():
    """Entry point: configure parameters, log in, and run download."""
    MAX_ROWS = 5000000       # Max rows per query
    N_RA_SEGMENTS = 180      # Number of RA bins
    N_DEC_SEGMENTS = 60      # Number of Dec bins
    MAX_WORKERS = 10         # Thread pool size
    OUTPUT_DIR = "../download"

    setup_proxy()

    token = login_datalab(user='<YOUR_USERNAME>', password='<YOUR_PASSWORD>')

    if not test_datalab_connectivity(token):
        print("Connectivity test failed, exiting")
        return

    run_download(
        token=token,
        output_dir=OUTPUT_DIR,
        n_ra_segments=N_RA_SEGMENTS,
        n_dec_segments=N_DEC_SEGMENTS,
        max_rows=MAX_ROWS,
        max_workers=MAX_WORKERS
    )


if __name__ == "__main__":
    main()

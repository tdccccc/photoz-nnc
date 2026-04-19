#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bulk downloader for PanSTARRS DR2 stack photometry via MAST CasJobs.

The sky is tiled into an RA/Dec grid (see :func:`make_seg`). For each tile a
CasJobs SQL job is submitted against the ``PanSTARRS_DR2`` context, the result
is pulled with :meth:`mastcasjobs.MastCasJobs.fast_table`, and written to
``./download/ps1dr2_<fid>.fits``. Existing output files are skipped, which makes
the script safe to restart after interruptions.

Only the five-band stack photometry columns needed downstream (PSF / Aperture /
Kron mags with errors, info flags, and object-level quality flags) are
selected, and every magnitude is required to be positive with a positive error
so that obviously bad rows never reach disk.

Typical usage:
    python download.py -u <user> -p <password> [-r] [-s <start_fid>]

Notes:
    Any HTTP(S) proxy environment variables are cleared at import time because
    the MAST CasJobs endpoints are not reachable through common corporate
    proxies in our setup. Remove this block if you need the opposite behavior.
"""

import os
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = '*'

import pandas as pd
import numpy as np
import mastcasjobs
import time
import argparse


def make_seg(n_ra=10, n_dec=120, output_dir='./download'):
    """Build a uniform RA/Dec grid and persist it as the download manifest.

    The sky is split into ``n_ra * n_dec`` rectangular tiles. RA is divided
    evenly over ``[0, 360)`` and Dec over ``[-30, 90)``, which is the PS1 3pi
    survey footprint. Each tile gets a unique ``fid`` that later drives the
    output filename ``ps1dr2_<fid>.fits``.

    Args:
        n_ra: Number of RA bins.
        n_dec: Number of Dec bins. Kept much larger than ``n_ra`` so that each
            tile holds a tractable number of rows for CasJobs.
        output_dir: Directory to create if missing; the manifest is written to
            ``<output_dir>/readme.txt``.

    Returns:
        pandas.DataFrame: Manifest with columns ``fid``, ``ra_l``, ``ra_u``,
        ``dec_l``, ``dec_u`` (lower-inclusive / upper-exclusive bounds).
    """
    ra_lst = np.linspace(0, 360, n_ra + 1)
    dec_lst = np.linspace(-30, 90, n_dec + 1)

    records = []
    fid = 0
    for i in range(n_ra):
        for j in range(n_dec):
            records.append({
                'fid': fid,
                'ra_l': ra_lst[i],
                'ra_u': ra_lst[i + 1],
                'dec_l': dec_lst[j],
                'dec_u': dec_lst[j + 1]
            })
            fid += 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'readme.txt'), index=False)
    print(f"n_ra={n_ra}, n_dec={n_dec}, total tiles: {len(df)}")
    return df

class PS1DR2_download:
    """Driver that walks the tile manifest and downloads one FITS per tile.

    The class owns a single :class:`mastcasjobs.MastCasJobs` session that is
    recreated on transient failures (see :meth:`_init_casjobs`). The public
    entry point is :meth:`execute`, which reads ``./download/readme.txt`` and
    dispatches each row to :meth:`download_one`.

    Attributes:
        user: CasJobs account name.
        pwd: CasJobs password.
        max_jobs: Reserved for future parallel submission; currently unused
            (jobs are processed serially so a single connection is enough).
        reverse: If True, iterate tiles from highest ``fid`` to lowest. Useful
            when running two workers from opposite ends of the manifest.
        max_retries: Per-tile retry budget inside :meth:`download_one` and
            status polling inside :meth:`wait_job`.
        start_fid: If set, skip every tile with ``fid < start_fid`` so the
            caller can resume from a known checkpoint.
        casjobs: The active CasJobs client, lazily (re)created.
    """

    def __init__(self, user, pwd, max_jobs=1, reverse=False, max_retries=2, start_fid=None):
        self.user = user
        self.pwd = pwd
        self.max_jobs = max_jobs
        self.reverse = reverse
        self.max_retries = max_retries
        self.start_fid = start_fid
        self.casjobs = None
        self._init_casjobs()

    def _init_casjobs(self):
        """(Re)open the CasJobs client, retrying forever with exponential backoff.

        MAST CasJobs occasionally rejects new sessions during maintenance
        windows. We would rather block here than let the outer loop abort a
        long download, so the retry is unbounded. Wait time grows linearly and
        caps at five minutes.
        """
        attempt = 0
        while True:
            try:
                self.casjobs = mastcasjobs.MastCasJobs(
                    username=self.user, password=self.pwd, context='PanSTARRS_DR2'
                )
                return
            except Exception as e:
                attempt += 1
                wait_time = min(30 * attempt, 300)  # cap backoff at 5 minutes
                print(f"  Connection failed: {str(e)[:60]}, waiting {wait_time}s (attempt {attempt})...")
                time.sleep(wait_time)

    def make_query(self, ra_l, ra_u, dec_l, dec_u, table_name):
        """Render the CasJobs SQL for one RA/Dec tile.

        The SELECT joins ``ObjectThin`` with ``StackObjectThin`` on ``objID``
        and writes into the caller's MyDB under ``table_name``. Rows are kept
        only when the primary detection flag is set and every PSF / Kron /
        Aperture magnitude (and its error) is strictly positive in all five
        bands, which removes the CasJobs ``-999`` sentinels at the source.

        Args:
            ra_l: Lower RA bound in degrees (inclusive).
            ra_u: Upper RA bound in degrees (exclusive).
            dec_l: Lower Dec bound in degrees (inclusive).
            dec_u: Upper Dec bound in degrees (exclusive).
            table_name: MyDB table name to receive the results.

        Returns:
            str: SQL statement ready for :meth:`mastcasjobs.MastCasJobs.submit`.
        """
        return f"""
            SELECT
                o.objID, o.raStack, o.decStack, o.l, o.b, o.nDetections,
                s.gPSFMag, s.gPSFMagErr, s.gApMag, s.gApMagErr, s.gKronMag, s.gKronMagErr,
                s.rPSFMag, s.rPSFMagErr, s.rApMag, s.rApMagErr, s.rKronMag, s.rKronMagErr,
                s.iPSFMag, s.iPSFMagErr, s.iApMag, s.iApMagErr, s.iKronMag, s.iKronMagErr,
                s.zPSFMag, s.zPSFMagErr, s.zApMag, s.zApMagErr, s.zKronMag, s.zKronMagErr,
                s.yPSFMag, s.yPSFMagErr, s.yApMag, s.yApMagErr, s.yKronMag, s.yKronMagErr,
                s.ginfoFlag, s.ginfoFlag2, s.ginfoFlag3,
                s.rinfoFlag, s.rinfoFlag2, s.rinfoFlag3,
                s.iinfoFlag, s.iinfoFlag2, s.iinfoFlag3,
                s.zinfoFlag, s.zinfoFlag2, s.zinfoFlag3,
                s.yinfoFlag, s.yinfoFlag2, s.yinfoFlag3,
                o.qualityFlag, o.objInfoFlag
            INTO mydb.[{table_name}]
            FROM ObjectThin o
            INNER JOIN StackObjectThin s ON o.objID = s.objID
            WHERE o.raStack >= {ra_l:.5f} AND o.raStack < {ra_u:.5f}
                AND o.decStack >= {dec_l:.5f} AND o.decStack < {dec_u:.5f}
                AND s.primaryDetection = 1
                AND s.gPSFMag > 0 AND s.gPSFMagErr > 0
                AND s.rPSFMag > 0 AND s.rPSFMagErr > 0
                AND s.iPSFMag > 0 AND s.iPSFMagErr > 0
                AND s.zPSFMag > 0 AND s.zPSFMagErr > 0
                AND s.yPSFMag > 0 AND s.yPSFMagErr > 0
                AND s.gKronMag > 0 AND s.gKronMagErr > 0
                AND s.rKronMag > 0 AND s.rKronMagErr > 0
                AND s.iKronMag > 0 AND s.iKronMagErr > 0
                AND s.zKronMag > 0 AND s.zKronMagErr > 0
                AND s.yKronMag > 0 AND s.yKronMagErr > 0
                AND s.gApMag > 0 AND s.gApMagErr > 0
                AND s.rApMag > 0 AND s.rApMagErr > 0
                AND s.iApMag > 0 AND s.iApMagErr > 0
                AND s.zApMag > 0 AND s.zApMagErr > 0
                AND s.yApMag > 0 AND s.yApMagErr > 0
        """

    def safe_drop_table(self, table_name):
        """Drop a MyDB table, swallowing errors.

        Used both before submitting a new job (to reclaim the name if a
        previous run crashed mid-way) and after downloading the results.
        Failures here are not actionable, so we silently ignore them.
        """
        try:
            self.casjobs.drop_table_if_exists(table_name)
        except:
            pass

    def wait_job(self, jobid, fid=None):
        """Poll a CasJobs job until it reaches a terminal state.

        A status query that raises counts as a transient failure and is
        retried up to :attr:`max_retries` times, re-opening the CasJobs client
        between attempts. A successful query resets the retry counter so that
        a long-running job never times out from occasional flaky polls.

        Args:
            jobid: Job id returned by :meth:`mastcasjobs.MastCasJobs.submit`.
            fid: Tile id used only in log messages.

        Returns:
            str: One of ``'finished'``, ``'failed'`` or ``'cancelled'``.

        Raises:
            Exception: Re-raises the last status-query exception after the
                retry budget is exhausted.
        """
        retry = 0
        while True:
            try:
                status = self.casjobs.status(jobid)[1]
                if status in ['finished', 'failed', 'cancelled']:
                    return status
                retry = 0  # reset after a successful poll
            except Exception as e:
                retry += 1
                if retry > self.max_retries:
                    raise e
                print(f"  FID {fid}: status query failed, retrying {retry}/{self.max_retries}...")
                time.sleep(10)
                self._init_casjobs()  # reopen the session before the next poll
            time.sleep(5)

    def download_one(self, fid, ra_l, ra_u, dec_l, dec_u):
        """Download one tile to ``./download/ps1dr2_<fid>.fits``.

        If the output file already exists, the tile is skipped so the whole
        run can be restarted safely. Otherwise a CasJobs job is submitted,
        awaited and its MyDB table is pulled with ``fast_table``. Transient
        errors (network, timeouts, MAST hiccups) trigger a backoff and a
        session reset, up to :attr:`max_retries` attempts.

        Args:
            fid: Tile id from the manifest.
            ra_l, ra_u, dec_l, dec_u: Tile bounds in degrees.

        Returns:
            One of:
                - ``int``: Row count on success.
                - ``'skip'``: Output file already present.
                - ``'empty'``: Job finished but the table was empty.
                - ``'failed'``: Job reached a ``failed``/``cancelled`` state.
                - ``'error: <msg>'``: All retries exhausted; ``<msg>`` is the
                  truncated exception text.
        """
        path = f'./download/ps1dr2_{fid}.fits'
        if os.path.exists(path):
            return 'skip'

        table_name = f"tmp_{fid}"

        for attempt in range(self.max_retries):
            try:
                self.safe_drop_table(table_name)
                query = self.make_query(ra_l, ra_u, dec_l, dec_u, table_name)
                jobid = self.casjobs.submit(query, task_name=table_name)
                status = self.wait_job(jobid, fid)

                if status == 'finished':
                    tab = self.casjobs.fast_table(table=table_name)
                    self.safe_drop_table(table_name)
                    if tab is not None and len(tab) > 0:
                        tab.write(path, format='fits', overwrite=True)
                        return len(tab)
                    return 'empty'
                else:
                    self.safe_drop_table(table_name)
                    return 'failed'

            except Exception as e:
                self.safe_drop_table(table_name)
                err_msg = str(e)[:80]

                if attempt < self.max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Error: {err_msg}, waiting {wait_time}s before retry ({attempt+1}/{self.max_retries})...")
                    time.sleep(wait_time)
                    self._init_casjobs()  # reopen the session before retrying
                else:
                    return f'error: {err_msg}'

        return 'error: max retries exceeded'

    def execute(self):
        """Iterate the manifest and download every tile sequentially.

        Reads ``./download/readme.txt`` produced by :func:`make_seg`, applies
        the ``start_fid`` / ``reverse`` filters, and delegates each row to
        :meth:`download_one`. Per-tile outcomes are printed with a running
        ``done/total`` counter, where ``done`` counts only tiles that were
        actually attempted (skipped ones do not increment it).
        """
        readme = pd.read_csv('./download/readme.txt')
        if self.start_fid is not None:
            readme = readme[readme['fid'] >= self.start_fid]
        if self.reverse:
            readme = readme.iloc[::-1]

        total = len(readme)
        done = 0
        print(f"Total: {total}, start_fid: {self.start_fid}, reverse: {self.reverse}, max_retries: {self.max_retries}")

        for _, row in readme.iterrows():
            fid = int(row['fid'])
            result = self.download_one(fid, row['ra_l'], row['ra_u'], row['dec_l'], row['dec_u'])
            if result != 'skip':
                done += 1
            print(f"FID {fid}: {result} ({done}/{total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanSTARRS DR2 bulk downloader')
    parser.add_argument('-u', '--user', required=True, help='CasJobs username')
    parser.add_argument('-p', '--password', required=True, help='CasJobs password')
    parser.add_argument('-j', '--max-jobs', type=int, default=1, help='max concurrent jobs (default: 1)')
    parser.add_argument('-r', '--reverse', action='store_true', help='iterate tiles in reverse order')
    parser.add_argument('-t', '--max-retries', type=int, default=2, help='max retries per tile (default: 2)')
    parser.add_argument('-s', '--start-fid', type=int, default=None, help='start from this FID (resume checkpoint)')

    args = parser.parse_args()

    make_seg()

    worker = PS1DR2_download(
        user=args.user,
        pwd=args.password,
        max_jobs=args.max_jobs,
        reverse=args.reverse,
        max_retries=args.max_retries,
        start_fid=args.start_fid
    )
    worker.execute()

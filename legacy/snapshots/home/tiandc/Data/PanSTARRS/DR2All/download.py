#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PanSTARRS DR2 数据下载脚本
用法: python download.py -u 用户名 -p 密码 [-r] [-s 起始FID]
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
    """
    简单的 RA/Dec 网格分区
    n_ra: RA 方向分区数
    n_dec: Dec 方向分区数
    总分区数 = n_ra * n_dec
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
    print(f"n_ra={n_ra}, n_dec={n_dec}, 总文件数: {len(df)}")
    return df

class PS1DR2_download:
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
        """初始化或重新初始化 casjobs 连接，失败时无限重试"""
        attempt = 0
        while True:
            try:
                self.casjobs = mastcasjobs.MastCasJobs(
                    username=self.user, password=self.pwd, context='PanSTARRS_DR2'
                )
                return
            except Exception as e:
                attempt += 1
                wait_time = min(30 * attempt, 300)  # 最多等待5分钟
                print(f"  Connection failed: {str(e)[:60]}, waiting {wait_time}s (attempt {attempt})...")
                time.sleep(wait_time)

    def make_query(self, ra_l, ra_u, dec_l, dec_u, table_name):
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
        """安全删除表，忽略错误"""
        try:
            self.casjobs.drop_table_if_exists(table_name)
        except:
            pass

    def wait_job(self, jobid, fid=None):
        """等待任务完成，带重试"""
        retry = 0
        while True:
            try:
                status = self.casjobs.status(jobid)[1]
                if status in ['finished', 'failed', 'cancelled']:
                    return status
                retry = 0  # 重置重试计数
            except Exception as e:
                retry += 1
                if retry > self.max_retries:
                    raise e
                print(f"  FID {fid}: status query failed, retrying {retry}/{self.max_retries}...")
                time.sleep(10)
                self._init_casjobs()  # 重新初始化连接
            time.sleep(5)

    def download_one(self, fid, ra_l, ra_u, dec_l, dec_u):
        """下载单个区域，带重试"""
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
                    self._init_casjobs()  # 重新初始化连接
                else:
                    return f'error: {err_msg}'

        return 'error: max retries exceeded'

    def execute(self):
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
    parser = argparse.ArgumentParser(description='PanSTARRS DR2 数据下载')
    parser.add_argument('-u', '--user', required=True, help='CasJobs 用户名')
    parser.add_argument('-p', '--password', required=True, help='CasJobs 密码')
    parser.add_argument('-j', '--max-jobs', type=int, default=1, help='最大任务数 (默认: 1)')
    parser.add_argument('-r', '--reverse', action='store_true', help='倒序下载')
    parser.add_argument('-t', '--max-retries', type=int, default=2, help='最大重试次数 (默认: 3)')
    parser.add_argument('-s', '--start-fid', type=int, default=None, help='起始FID (从此FID开始下载)')

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

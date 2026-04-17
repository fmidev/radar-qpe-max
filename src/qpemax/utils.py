# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

import datetime
import logging
import re
from glob import glob
from typing import List

import pandas as pd

from qpemax.constants import ACC_CACHE_FMT, DATEFMT, QPE_CACHE_FMT


logger = logging.getLogger('airflow.task')


def corr_suffix(dbz_field: str) -> str:
    """Return attenuation-correction suffix for filenames."""
    return '_c' if 'C' in dbz_field else ''


def qpe_cache_fname(
        ts: str, nod: str, size: int, resolution: int,
        corr: str, p_chunksize: int) -> str:
    """Build QPE cache netCDF filename (basename only)."""
    return QPE_CACHE_FMT.format(
        ts=ts, nod=nod, size=size, resolution=resolution,
        corr=corr, chunksize=p_chunksize)


def acc_cache_fname(
        date: datetime.date, nod: str, size: int, resolution: int,
        corr: str, acc_chunksize: int, win: str) -> str:
    """Build accumulation cache netCDF filename (basename only)."""
    return ACC_CACHE_FMT.format(
        ts=date.strftime(DATEFMT), nod=nod, size=size, resolution=resolution,
        corr=corr, chunksize=acc_chunksize, win=win).lower()


def two_day_glob(
        date: datetime.date,
        globfmt: str = '{date}*.h5',
        **kws) -> tuple[List[str], List[str]]:
    """List paths matching a glob pattern for given and previous date.

    The returned list includes paths matching the given date and one day before
    it. Variables {yyyy}, {mm}, {dd} and {date} are available in the `globfmt`
    search pattern. The returned list is sorted.

    Additional keyword arguments are passed to `glob.glob`."""
    def fmtglob(d: datetime.date) -> str:
        return globfmt.format(yyyy=d.strftime('%Y'),
                              mm=d.strftime('%m'), dd=d.strftime('%d'),
                              date=d.strftime(DATEFMT))
    date0 = date - datetime.timedelta(days=1)
    ls = glob(fmtglob(date0), **kws)
    ls0 = ls.copy()
    ls.extend(glob(fmtglob(date), **kws))
    return sorted(ls), ls0


def tstep_from_fpaths(fpaths: List[str]) -> datetime.datetime:
    """timestep length from filenames"""
    tstep_default = pd.to_timedelta('5min')
    # file paths like /path/to/202405280005_radar.polar.filuo.h5
    # search for 12 digits in 2 consecutive file paths
    tstep = pd.to_datetime(re.search(r'\d{12}', fpaths[1]).group())
    tstep -= pd.to_datetime(re.search(r'\d{12}', fpaths[0]).group())
    # double check with the last two files
    tstep2 = pd.to_datetime(re.search(r'\d{12}', fpaths[-1]).group())
    tstep2 -= pd.to_datetime(re.search(r'\d{12}', fpaths[-2]).group())
    if tstep != tstep2:
        logger.error(
            'Inconsistent timestep lengths in file paths. Assuming 5min.')
        return tstep_default
    if tstep != tstep_default:
        logger.warning(f'Unusual timestep length {tstep}.')
    return tstep

# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

import datetime
import logging
import os
from typing import List

import pandas as pd
import xarray as xr

from radproc.aliases.fmi import LWE

from qpemax.callbacks import ProgressLogging
from qpemax.constants import (
    ACC, DATEFMT, DEFAULT_ACC_CHUNKSIZE, DEFAULT_CACHE_DIR,
    DEFAULT_ENCODING, DEFAULT_P_CHUNKSIZE, DEFAULT_RESOLUTION, DEFAULT_XY_SIZE,
    DEFAULT_COMPUTE_CHUNKSIZE, EPSG_TARGET, QPE_CACHE_FMT, ZH,
)
from qpemax.utils import acc_cache_fname, corr_suffix


logger = logging.getLogger('airflow.task')


def _combine_rds(
        ncfiles: List[str],
        ncpath: str,
        chunksize: int,
        ignore_cache: bool) -> str:
    """Prepare precip rate dataset.

    Load cached precipitation rasters from netcdf files and write them to a
    single chunked netcdf file."""
    if not os.path.isfile(ncpath) or ignore_cache: # then create it
        logger.info('Loading cached individual precipitation rasters.')
        # combine all files into a single dataset
        chunks = {'x': chunksize, 'y': chunksize}
        rds = xr.open_mfdataset(
            ncfiles,
            data_vars='minimal',
            parallel=False,
            engine='h5netcdf',
            phony_dims='sort',
            chunks=chunks,
        )
        # at least one million elements per chunk is recommended
        encoding = DEFAULT_ENCODING.copy()
        encoding[LWE]['chunksizes'] = (24, chunksize, chunksize)
        logger.info(f'Writing chunked dataset {ncpath}')
        rds.to_netcdf(ncpath, encoding=encoding, engine='h5netcdf')
    return ncpath


def load_chunked_dataset(ncpath: str) -> xr.Dataset:
    """Load chunked dataset.

    Return the dataset with time rounded to minutes."""
    # open the dataset in chunks
    logger.info(f'Loading chunked dataset {ncpath}')
    rds = xr.open_dataset(ncpath, chunks={},
                          engine='h5netcdf')
    logger.info('Rasters loaded.')
    rds['time'] = rds.indexes['time'].round('min')
    return rds.convert_calendar(calendar='standard', use_cftime=True)


def combine_rasters(
        date: datetime.date, nod: str, cachedir: str = DEFAULT_CACHE_DIR,
        size: int = DEFAULT_XY_SIZE, resolution: int = DEFAULT_RESOLUTION,
        p_chunksize: int = DEFAULT_P_CHUNKSIZE,
        ignore_cache: bool = False,
        dbz_field: str = ZH) -> tuple[str, List[str]]:
    """Combine individual rasters into a single chunked netcdf file."""
    from qpemax.utils import two_day_glob
    corr = corr_suffix(dbz_field)
    globfmt = os.path.join(cachedir, QPE_CACHE_FMT.format(
        ts='{date}????', nod=nod, size=size, resolution=resolution,
        corr=corr, chunksize=p_chunksize))
    ncpath = os.path.join(cachedir, QPE_CACHE_FMT.format(
        ts=date.strftime(DATEFMT), nod=nod, size=size, resolution=resolution,
        corr=corr, chunksize=p_chunksize))
    ncfiles, ncfiles_obsolete = two_day_glob(date, globfmt=globfmt)
    ncfile = _combine_rds(ncfiles, ncpath, p_chunksize, ignore_cache)
    return ncfile, ncfiles_obsolete


def _accu_time_bounds(
        rds: 'xr.Dataset', date: datetime.date, win: str,
    ) -> tuple[pd.Timestamp, pd.Timestamp, int, pd.Timedelta]:
    """Compute rolling-window time bounds and step size for accumulation.

    Returns (tstep_pre, tstep_last, iwin, tdelta) where:
      tstep_pre  — first timestep needed to fill the window ending at date
      tstep_last — last timestep of date
      iwin       — number of timesteps per window
      tdelta     — dataset timestep length
    """
    win_trim = win.replace(' ', '')
    # number of timesteps in window (e.g. 288 5min steps in a day)
    iwin = rds.time.groupby(rds.time.dt.floor(win_trim)).sizes['time']
    dwin = pd.to_timedelta(win)
    tind = rds.indexes['time']
    # timestep length as timedelta
    # CFTimeIndex after convert_calendar may have freq=None;
    # pd.to_timedelta(None) returns NaT which is truthy, so use explicit check.
    freq = tind.freq
    tdelta = pd.to_timedelta(freq) if freq is not None else pd.Series(tind).diff().median()
    tstep_last = pd.to_datetime(date + datetime.timedelta(days=1)) - tdelta
    tstep_pre = pd.to_datetime(date) - dwin + tdelta
    return tstep_pre, tstep_last, iwin, tdelta


def _write_accums(accums, accumsfile, acc_chunksize: int,
                  ignore_cache: bool = False):
    """Write accumulation dataset to file."""
    # check if the file exists already
    if os.path.isfile(accumsfile) and not ignore_cache:
        logger.info(f'File {accumsfile} exists.')
        return
    encoding = DEFAULT_ENCODING.copy()
    encoding[LWE]['chunksizes'] = (
        accums.shape[0], acc_chunksize, acc_chunksize)
    # Align dask chunks to output encoding to avoid implicit shuffle on write
    accums = accums.chunk({'time': -1, 'x': acc_chunksize, 'y': acc_chunksize})
    with ProgressLogging(logger, dt=5):
        accums.to_netcdf(accumsfile, encoding=encoding, engine='h5netcdf')


def accu(
        date: datetime.date, nod: str,
        cachedir: str = DEFAULT_CACHE_DIR,
        size: int = DEFAULT_XY_SIZE,
        resolution: int = DEFAULT_RESOLUTION,
        p_chunksize: int = DEFAULT_P_CHUNKSIZE,
        compute_chunksize: int = DEFAULT_COMPUTE_CHUNKSIZE,
        acc_chunksize: int = DEFAULT_ACC_CHUNKSIZE,
        dbz_field: str = ZH,
        win: str = '1D', **kws
    ):
    """Rolling window precipitation accumulation."""
    corr = corr_suffix(dbz_field)
    ncpath = os.path.join(cachedir, QPE_CACHE_FMT.format(
        ts=date.strftime(DATEFMT), nod=nod, size=size, resolution=resolution,
        corr=corr, chunksize=p_chunksize))
    accpath = os.path.join(
        cachedir, acc_cache_fname(date, nod, size, resolution, corr, acc_chunksize, win))
    rds = load_chunked_dataset(ncpath)
    tstep_pre, tstep_last, iwin, tdelta = _accu_time_bounds(rds, date, win)
    rollsel = rds.sel(time=slice(tstep_pre, tstep_last))
    # Consolidate time into one chunk so rolling is intra-chunk;
    # use large spatial chunks for compute, _write_accums rechunks to acc_chunksize.
    compute_chunksize = min(p_chunksize, compute_chunksize)
    rollsel = rollsel.chunk({'time': -1, 'x': compute_chunksize, 'y': compute_chunksize})
    # The data is still precip rate, so scale to mm
    acc_scaling = datetime.timedelta(hours=1) / tdelta # 12 for 5min steps
    accums = (
        rollsel[LWE].rolling({'time': iwin}).sum() / acc_scaling
    )
    logger.info(f'Processing accumulation dataset to {accpath}')
    _write_accums(accums, accpath, acc_chunksize, **kws)
    return accpath, rds.attrs


def aggmax(
        accfile: str, attrs, p_chunksize: int = DEFAULT_P_CHUNKSIZE,
        acc_chunksize: int = DEFAULT_ACC_CHUNKSIZE
    ) -> tuple[xr.DataArray, xr.DataArray]:
    """maximum precipitation accumulation for each pixel"""
    from qpemax.output import _write_dat_attrs, _write_dattime_attrs
    logger.info('Loading accumulation dataset.')
    accums = xr.open_dataarray(accfile, engine='h5netcdf', chunks={})
    accums = accums.convert_calendar(calendar='standard', use_cftime=True)
    accums = accums.rename(ACC)
    dat = accums.max('time').rio.write_crs(EPSG_TARGET)
    dattime = accums.idxmax(
        dim='time', keep_attrs=True
    ).rio.write_crs(EPSG_TARGET)
    dat = _write_dat_attrs(dat, attrs)
    dattime = _write_dattime_attrs(dattime, attrs)
    dattime = dattime.chunk((acc_chunksize, acc_chunksize))
    return dat, dattime

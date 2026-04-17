# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

import datetime
import logging
import os

import rioxarray
import xarray as xr

from qpemax.callbacks import ProgressLogging
from qpemax.constants import (
    ACC, ATTRS, COG_COMPRESS, DATEFMT, DEFAULT_RESOLUTION, DEFAULT_XY_SIZE,
)


logger = logging.getLogger('airflow.task')


def _write_dat_attrs(data: xr.Dataset, rdattrs: dict) -> xr.Dataset:
    """Write attributes to precipitation maximum data."""
    dat = data.copy()
    dat.attrs.update(ATTRS[ACC])
    dat.attrs.update({'long_name': 'maximum precipitation accumulation'})
    dat.attrs.update(rdattrs)
    return dat.rio.write_coordinate_system()


def _write_dattime_attrs(data: xr.DataArray, rdattrs: dict) -> xr.DataArray:
    """Write attributes to precipitation maximum time data."""
    dattime = data.copy()
    dattime.attrs.update(rdattrs)
    dattime.attrs.update(ATTRS['time'])
    return dattime.rio.write_coordinate_system()


def _write_dattime_tif(
        dattime: xr.DataArray, tift: str, blocksize: int = 512) -> None:
    """timestamp geotiff"""
    logger.info(f'Processing geotiff product {tift}')
    with ProgressLogging(logger, dt=5):
        dattime.load()
    tunits = 'minutes since ' + str(dattime.min().item())
    enc = {'units': tunits, 'calendar': 'gregorian'}
    dattime.rio.update_encoding(enc, inplace=True)
    dattime.rio.to_raster(tift, dtype='uint16', compress=COG_COMPRESS)
    unidat = rioxarray.open_rasterio(tift).rio.update_attrs({'units': tunits})
    unidat.rio.to_raster(
        tift, compress=COG_COMPRESS,
        driver='COG', dtype='uint16',
        blocksize=blocksize,
    )


def _write_dat_tif(dat: xr.DataArray, tifp: str, blocksize: int = 512) -> None:
    """main geotiff"""
    logger.info(f'Processing geotiff product {tifp}')
    with ProgressLogging(logger, dt=5):
        dat.rio.to_raster(
            tifp, driver='COG',
            dtype='uint16', compress=COG_COMPRESS,
            blocksize=blocksize,
        )


def write_max_tifs(
        dat: xr.DataArray, dattime: xr.DataArray, date: datetime.date,
        resultsdir: str, nod: str, win: str, corr: str = '',
        size: int = DEFAULT_XY_SIZE,
        resolution: int = DEFAULT_RESOLUTION) -> None:
    """Write maximum precipitation accumulation and time to geotiffs."""
    win = win.lower()
    tstamp = date.strftime(DATEFMT)
    tifp = os.path.join(
        resultsdir,
        f'{nod}{tstamp}max{win}{size}px{resolution}m{corr}.tif')
    tift = os.path.join(
        resultsdir,
        f'{nod}{tstamp}maxtime{win}{size}px{resolution}m{corr}.tif')
    _write_dat_tif(dat, tifp)
    _write_dattime_tif(dattime, tift)

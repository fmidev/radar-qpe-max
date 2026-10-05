# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

import datetime
import logging
import os

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from qpemax.callbacks import ProgressLogging
from qpemax.constants import (
    ACC, ATTRS, COG_COMPRESS, DEFAULT_RESOLUTION, LWE_SCALE_FACTOR,
    UINT16_FILLVAL,
)
from qpemax.utils import corr_suffix


logger = logging.getLogger('airflow.task')

MAX_TIF_FMT = (
    '{ts}_{site}_{product}_{param}_acrr_finrad{resolution}_{filt}.tif'
)


def _max_tif_segments(
        date: datetime.date, win: str, resolution: int, dbz_field: str,
) -> dict[str, str]:
    """Filename segments shared by all daily max products of a date."""
    hours = pd.to_timedelta(win) / pd.Timedelta(hours=1)
    if hours <= 0 or not hours.is_integer():
        raise ValueError(f'window {win!r} is not a positive whole number of hours')
    end = pd.Timestamp(date).normalize() + pd.Timedelta(days=1)
    return dict(
        ts=end.strftime('%Y%m%d%H%M'), param=f'{int(hours)}h',
        resolution=str(int(resolution)),
        filt='rawac' if corr_suffix(dbz_field) else 'raw',
    )


def max_tif_name(
        date: datetime.date, nod: str, product: str, win: str,
        resolution: int, dbz_field: str = 'DBZH') -> str:
    """Daily max product filename following the FMI radar GeoTIFF convention.

    `{timestamp}_{site}_{product}_{parameter}_acrr_finrad{resolution}_{filter}.tif`

    - timestamp: end of the UTC `date` (`date` + 1 day, 00:00) as `%Y%m%d%H%M`.
      The max is taken over all sliding windows whose last time step is on `date`.
    - site: radar node name `nod` or `composite`.
    - product: `max` (accumulation) or `maxtime` (time of maximum).
    - parameter: window length `win` in whole hours, e.g. `1 D` -> `24h`.
    - filter: `rawac` for attenuation corrected `dbz_field` (e.g. DBZHC), else `raw`.

    Raises ValueError on non-whole-hour windows or segments containing `_` or `.`."""
    segments = _max_tif_segments(date, win, resolution, dbz_field)
    segments.update(site=nod, product=product)
    for key, value in segments.items():
        if not value or '_' in value or '.' in value:
            raise ValueError(f'invalid filename segment {key}={value!r}')
    return MAX_TIF_FMT.format(**segments)


def max_tif_glob(
        date: datetime.date, win: str, resolution: int,
        dbz_field: str = 'DBZH') -> str:
    """Glob pattern for the `max` files of `date` for all sites.

    Matches `max` but not `maxtime` products. Note that the pattern also
    matches the `composite` max file; exclude paths containing `_composite_`
    if it is written to the same directory."""
    segments = _max_tif_segments(date, win, resolution, dbz_field)
    return MAX_TIF_FMT.format(site='*', product='max', **segments)


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
    # Coerce to datetime64[ns]; idxmax on a cftime-indexed dim returns
    # object dtype (cftime objects interleaved with NaN fills), which
    # doesn't support arithmetic or reductions cleanly.
    raw = np.asarray(dattime.values).ravel()
    # cftime objects are not recognized by pd.to_datetime with utc=True;
    # convert via isoformat strings which pandas parses correctly.
    if raw.dtype == object and len(raw) > 0:
        strs = [v.isoformat() if hasattr(v, 'isoformat') else None for v in raw]
        flat = pd.to_datetime(strs, errors='coerce')
    else:
        flat = pd.to_datetime(raw, errors='coerce', utc=True).tz_localize(None)
    times = flat.to_numpy(dtype='datetime64[ns]').reshape(dattime.shape)
    tmin = pd.Timestamp(np.nanmin(times.astype('int64'))).to_datetime64()
    tunits = 'minutes since ' + str(pd.Timestamp(tmin))
    # Manually encode datetime64 -> uint16 minutes since tmin, bypassing
    # xarray CF encoding (which tangles with rioxarray over _FillValue and
    # AlwaysGreaterThan sentinels on datetime data). NaT -> 0 = nodata;
    # valid times start at 1 (offset +1 so tmin itself isn't nodata).
    delta_min = (times - tmin) / np.timedelta64(1, 'm')
    delta_min = np.where(np.isnan(delta_min), -1, delta_min) + 1
    encoded = xr.DataArray(
        delta_min.astype('uint16'),
        coords=dattime.coords, dims=dattime.dims,
        attrs={k: v for k, v in dattime.attrs.items() if k != '_FillValue'},
    )
    encoded.attrs['units'] = tunits
    encoded.rio.write_nodata(0, inplace=True)
    encoded.rio.to_raster(tift, dtype='uint16', compress=COG_COMPRESS)
    unidat = rioxarray.open_rasterio(tift).rio.update_attrs({'units': tunits})
    unidat.rio.to_raster(
        tift, compress=COG_COMPRESS,
        driver='COG', dtype='uint16',
        blocksize=blocksize,
    )


def _write_dat_tif(dat: xr.DataArray, tifp: str, blocksize: int = 512) -> None:
    """main geotiff"""
    logger.info(f'Processing geotiff product {tifp}')
    dat.attrs.pop('_FillValue', None)
    dat.encoding.update({
        'scale_factor': LWE_SCALE_FACTOR,
        '_FillValue': UINT16_FILLVAL,
        'dtype': 'uint16',
    })
    with ProgressLogging(logger, dt=5):
        dat.rio.to_raster(
            tifp, driver='COG',
            dtype='uint16', compress=COG_COMPRESS,
            blocksize=blocksize,
        )


def write_max_tifs(
        dat: xr.DataArray, dattime: xr.DataArray, date: datetime.date,
        resultsdir: str, nod: str, win: str, dbz_field: str = 'DBZH',
        resolution: int = DEFAULT_RESOLUTION) -> None:
    """Write maximum precipitation accumulation and time to geotiffs.

    File names are given by `max_tif_name`."""
    tifp, tift = (
        os.path.join(
            resultsdir,
            max_tif_name(date, nod, product, win, resolution, dbz_field))
        for product in ('max', 'maxtime')
    )
    _write_dat_tif(dat, tifp)
    _write_dattime_tif(dattime, tift)

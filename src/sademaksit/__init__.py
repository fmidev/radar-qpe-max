# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT
import os
import logging
import datetime
from glob import glob

import xarray as xr
import rioxarray
import pyart
from pyart.graph.common import generate_radar_time_begin
import numpy as np
import pandas as pd
from pyproj import Transformer

from radproc.aliases import lwe
from radproc.radar import z_r_qpe, source2dict
from sademaksit._version import __version__


ZH = 'DBZH'
ACC = 'lwe_accum'
PYART_AEQD_FMT = '+proj={proj} +lon_0={lon_0} +lat_0={lat_0} +R={R}'
qpefmt = '{ts}{nod}{size}px{resolution}m{corr}'
QPE_CACHE_FMT = qpefmt + '.nc'
QPE_TIF_FMT = qpefmt + '.tif'
LWE_SCALE_FACTOR = 0.01
DATEFMT = '%Y%m%d'
UINT16_FILLVAL = np.iinfo(np.uint16).max
DEFAULT_ENCODING = {lwe: {'zlib': True,
                          '_FillValue': UINT16_FILLVAL,
                          'dtype': 'u2',
                          'scale_factor': LWE_SCALE_FACTOR}}
ATTRS = {ACC: {'units': 'mm',
               'standard_name': 'lwe_thickness_of_precipitation_amount',
               '_FillValue': UINT16_FILLVAL},
         'time': {'long_name': 'end time of maximum precipitation accumulation period',
                  '_FillValue': UINT16_FILLVAL}}
CACHE_DIR = os.path.expanduser('~/.cache/sademaksit')
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger('maksit')


def clear_cache():
    for path in glob(os.path.join(CACHE_DIR, '*')):
        os.remove(path)


def basic_gatefilter(radar, field=ZH):
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(field)
    return gatefilter


def pyart_aeqd(radar):
    lat = radar.latitude['data'][0]
    lon = radar.longitude['data'][0]
    if isinstance(lat, np.ndarray):
        lat = lat[0]
        lon = lon[0]
    return dict(proj='aeqd', lat_0=lat, lon_0=lon, R=6370997)


def save_precip_grid(radar, cachefile, tiffile=None, size=2048, resolution=250):
    gf = basic_gatefilter(radar)
    r_m = size*resolution/2
    grid_shape = (1, size, size)
    grid_limits = ((0, 5000), (-r_m, r_m), (-r_m, r_m))
    grid = pyart.map.grid_from_radars(radar, gatefilters=gf, grid_shape=grid_shape,
                                      grid_limits=grid_limits, fields=[lwe],
                                      grid_projection=pyart_aeqd(radar))
    rds = grid.to_xarray().isel(z=0).reset_coords(drop=True)
    transproj = Transformer.from_crs(pyart_aeqd(radar), 'EPSG:3067')
    x, y = transproj.transform(rds.x, rds.y)
    rds['x'] = x
    rds['y'] = y
    rda = rds[lwe].fillna(0)
    rda.rio.write_crs(3067, inplace=True)
    rda = rda.to_dataset()
    rda.attrs.update(source2dict(radar))
    rda.to_netcdf(cachefile, encoding=DEFAULT_ENCODING)
    if isinstance(tiffile, str):
        # TODO: hardcoded scan frequency assumption
        acc = (rda.isel(time=0)[lwe]/12).rename(ACC)
        acc.attrs.update(ATTRS[ACC])
        acc.rio.update_encoding({'scale_factor': LWE_SCALE_FACTOR}, inplace=True)
        acc.rio.to_raster(tiffile, dtype='uint16', compress='deflate')


def qpe_grids_caching(h5paths, size, resolution, ignore_cache, resultsdir=None,
                      dbz_field=ZH):
    corr = '_c' if 'C' in dbz_field else ''
    if isinstance(resultsdir, str):
        tifdir = os.path.join(resultsdir, 'scan_accums')
        os.makedirs(tifdir, exist_ok=True)
    for fpath in h5paths:
        # read only the lowest elevation
        radar = pyart.aux_io.read_odim_h5(fpath, include_datasets=['dataset1'],
                                          file_field_names=True)
        t = generate_radar_time_begin(radar)
        ts = t.strftime('%Y%m%d%H%M')
        nod = source2dict(radar)['NOD']
        cachefname = QPE_CACHE_FMT.format(ts=ts, nod=nod, size=size,
                                          resolution=resolution, corr=corr)
        cachefile = os.path.join(CACHE_DIR, cachefname)
        if isinstance(resultsdir, str):
            tifname = QPE_TIF_FMT.format(ts=ts, nod=nod, size=size,
                                         resolution=resolution, corr=corr)
            tiffile = os.path.join(tifdir, tifname)
        if os.path.isfile(cachefile) and not ignore_cache:
            continue
        z_r_qpe(radar, dbz_field=dbz_field)
        save_precip_grid(radar, cachefile, tiffile=tiffile, size=size,
                         resolution=resolution)
    return nod


def ls_low_elev(date, site='', globfmt='{date}*{site}*.h5'):
    def fmtglob(d):
        return globfmt.format(yyyy=d.strftime('%Y'),
                              mm=d.strftime('%m'), dd=d.strftime('%d'),
                              date=d.strftime(DATEFMT), site=site)
    date0 = date - datetime.timedelta(days=1)
    ls = glob(fmtglob(date0))
    ls.extend(glob(fmtglob(date)))
    return sorted(ls)


def maxit(date, h5paths, resultsdir, size=2048, resolution=250, win='1 D',
          chunksize=None, ignore_cache=False, dbz_field=ZH):
    # takes forever with small chunksize
    if chunksize is None:
        if size > 1500:
            chunksize = 128 # to limit memory usage
        elif size > 250:
            chunksize = 256
        else:
            chunksize = size
    chunks = {'x': chunksize, 'y': chunksize}
    spatialchuncks = chunks.copy()
    corr = '_c' if 'C' in dbz_field else ''
    logger.info('Updating precipitation raster cache.')
    nod = qpe_grids_caching(h5paths, size, resolution, ignore_cache,
                            resultsdir=resultsdir, dbz_field=dbz_field)
    globstr = QPE_CACHE_FMT.format(ts='*', nod=nod, size=size,
                                   resolution=resolution, corr=corr)
    ncglob = os.path.join(CACHE_DIR, globstr)
    logger.info('Loading cached precipitation rasters.')
    rds = xr.open_mfdataset(ncglob, chunks=chunks, data_vars='minimal',
                            engine='h5netcdf', parallel=True)
    logger.info('Loaded.')
    chunks.update({'time': rds.dims['time']})
    rds = rds.chunk(chunks)
    logger.debug(rds.chunks)
    iwin = rds.time.groupby(rds.time.dt.floor(win)).sizes['time']
    dwin = pd.to_timedelta(win)
    win_trim = win.replace(' ', '').lower()
    t_round = rds.indexes['time'].round('min')
    rds['time'] = t_round
    rds = rds.convert_calendar(calendar='standard', use_cftime=True)
    tind = rds.indexes['time']
    tdelta = pd.to_timedelta(tind.freq) or pd.Series(tind).diff().median()
    tstep_last = pd.to_datetime(date+datetime.timedelta(days=1))-tdelta
    tstep_pre = pd.to_datetime(date)-dwin+tdelta
    rollsel = rds.sel(time=slice(tstep_pre, tstep_last))
    # TODO: document why divide by 12
    accums = (rollsel[lwe].rolling({'time': iwin}).sum()/12).to_dataset()
    accums = accums.rename({lwe: ACC})
    dat = accums.max('time').rio.write_crs(3067)
    dat['time'] = accums[ACC].idxmax(dim='time', keep_attrs=True).chunk(spatialchuncks)
    logger.debug(dat)
    logger.debug(rds)
    logger.debug(accums)
    logger.debug(dat.chunks)
    dat = dat.compute()
    tstamp = accums.time[-1].dt.strftime(DATEFMT).item()
    dat[ACC].attrs.update(ATTRS[ACC])
    dat[ACC].attrs.update({'long_name': 'maximum precipitation accumulation'})
    acc_cell = {'cell_methods': f'time: maximum (interval: {win.lower()})'}
    dat[ACC].attrs.update(acc_cell)
    dat[ACC].attrs.update(rds.attrs)
    dat['time'].attrs.update(rds.attrs)
    dat['time'].attrs.update(ATTRS['time'])
    dat.rio.write_coordinate_system(inplace=True)
    nod = rds.attrs['NOD']
    tifp = os.path.join(resultsdir, f'{nod}{tstamp}max{win_trim}{size}px{resolution}m{corr}.tif')
    tift = os.path.join(resultsdir, f'{nod}{tstamp}maxtime{win_trim}{size}px{resolution}m{corr}.tif')
    tunits = 'minutes since ' + str(dat.time.min().item())
    enc = {'time': {'units': tunits, 'calendar': 'proleptic_gregorian'}}
    dat.rio.update_encoding(enc, inplace=True)
    dat[ACC].rio.to_raster(tifp, dtype='uint16', compress='deflate')
    dat['time'].rio.to_raster(tift, dtype='uint16', compress='deflate')
    unidat = rioxarray.open_rasterio(tift).rio.update_attrs({'units': tunits})
    unidat.rio.to_raster(tift, compress='deflate')

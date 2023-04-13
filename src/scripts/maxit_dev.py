import os
import datetime
from glob import glob

import xarray as xr
import rioxarray
import pyart
from pyart.graph.common import generate_radar_time_begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

from radproc.aliases import zh, lwe
from radproc.radar import z_r_qpe


PYART_AEQD_FMT='+proj=aeqd +lon_0={lon} +lat_0={lat} +R=6370997'
LWE_SCALE_FACTOR = 0.01
UINT16_FILLVAL = np.iinfo(np.uint16).max
DEFAULT_ENCODING = {lwe: {'zlib': True, 'dtype': 'u2', 'scale_factor': LWE_SCALE_FACTOR}}
ATTRS = {lwe: {'units': 'mm h-1',
               '_FillValue': UINT16_FILLVAL},
         'time': {'_FillValue': UINT16_FILLVAL}}
CACHE_DIR = os.path.expanduser('~/.cache/sademaksit')
os.makedirs(CACHE_DIR, exist_ok=True)


def clear_cache():
    for path in glob(os.path.join(CACHE_DIR, '*')):
        os.remove(path)


def basic_gatefilter(radar, field=zh):
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(field)
    return gatefilter


def select_center(da, size=2048):
    xstart = int((da.x.size-size)/2)
    ystart = int((da.y.size-size)/2)
    return da[:, ystart:ystart+size, xstart:xstart+size].copy()


def pyart_aeqd(radar):
    lat = radar.latitude['data'][0]
    lon = radar.longitude['data'][0]
    return PYART_AEQD_FMT.format(lat=lat, lon=lon)


def save_precip_grid(radar, outfile, size=2048, resolution=250):
    gf = basic_gatefilter(radar)
    intermsize = int(size*2)
    grid_shape = (1, intermsize, intermsize)
    r_m = size*resolution*0.625 # includes reprojection margin
    grid_limits = ((0, 5000), (-r_m, r_m), (-r_m, r_m))
    grid = pyart.map.grid_from_radars(radar, gatefilters=gf,
                                      grid_shape=grid_shape,
                                      grid_limits=grid_limits,
                                      fields=[lwe])
    rds = grid.to_xarray().isel(z=0).set_index(x='lon', y='lat').reset_coords(drop=True)
    rda = rds.lwe_precipitation_rate
    rda.rio.write_crs(4326, inplace=True)
    rda3067 = rda.rio.reproject("epsg:3067", resolution=resolution, nodata=UINT16_FILLVAL)
    rda3067 = select_center(rda3067, size=size)
    rda3067.to_dataset().to_netcdf(outfile, encoding=DEFAULT_ENCODING)


def save_precip_grid2(radar, outfile, size=2048, resolution=250):
    gf = basic_gatefilter(radar)
    r_m = size*resolution/2
    grid_shape = (1, size, size)
    grid_limits = ((0, 5000), (-r_m, r_m), (-r_m, r_m))
    grid = pyart.map.grid_from_radars(radar, gatefilters=gf, grid_shape=grid_shape,
                                      grid_limits=grid_limits, fields=[lwe])
    rds = grid.to_xarray().isel(z=0).reset_coords(drop=True)
    transproj = Transformer.from_crs(pyart_aeqd(radar), 'EPSG:3067')
    x, y = transproj.transform(rds.x, rds.y)
    rds['x'] = x
    rds['y'] = y
    rda = rds[lwe]
    rda.rio.write_crs(3067, inplace=True)
    rda.to_dataset().to_netcdf(outfile, encoding=DEFAULT_ENCODING)


if __name__ == '__main__':
    plt.close('all')
    size = 512
    resolution = 1000
    win = '1D'
    date = datetime.date(2022, 6, 4)
    chunksize = 128
    ignore_cache = True
    #
    chunks = {'x': chunksize, 'y': chunksize}
    datadir = os.path.expanduser('~/data/alakulma')
    resultsdir = os.path.expanduser('~/results/sademaksit')
    h5paths = sorted(glob(os.path.join(datadir, '*-A.h5')))
    for fpath in h5paths:
        radar = pyart.aux_io.read_odim_h5(fpath)
        t = generate_radar_time_begin(radar)
        ts = t.strftime('%Y%m%d%H%M')
        outfile = os.path.join(CACHE_DIR, f'tstep{ts}_{size}x{resolution}m.nc')
        if os.path.isfile(outfile) and not ignore_cache:
            continue
        z_r_qpe(radar)
        save_precip_grid2(radar, outfile, size=size, resolution=resolution)
    ncglob = os.path.join(CACHE_DIR, f'tstep*_{size}x{resolution}m.nc')
    rds = xr.open_mfdataset(ncglob, chunks=chunks, data_vars='minimal')
    iwin = rds.time.groupby(rds.time.dt.floor(win)).sizes['time']
    dwin = pd.to_timedelta(win)
    t_round = rds.indexes['time'].round('min')
    rds['time'] = t_round
    rds = rds.convert_calendar(calendar='standard', use_cftime=True)
    tdelta = pd.to_timedelta(rds.indexes['time'].freq)
    tstep_last = pd.to_datetime(date+datetime.timedelta(days=1))-tdelta
    tstep_pre = pd.to_datetime(date)-dwin+tdelta
    rollsel = rds.sel(time=slice(tstep_pre, tstep_last))
    accums = (rollsel[lwe].rolling({'time': iwin}).sum()/12).to_dataset()
    dat = accums.max('time').rio.write_crs(3067)
    dat['time'] = accums[lwe].idxmax('time')
    tstamp = accums.time[-1].dt.strftime('%Y%m%d').item()
    dat[lwe].attrs.update(ATTRS[lwe])
    dat['time'].attrs.update(ATTRS['time'])
    dat.rio.write_coordinate_system(inplace=True)
    tif1h = os.path.join(resultsdir, f'{tstamp}_max{win}.tif')
    tif1htime = os.path.join(resultsdir, f'{tstamp}_max{win}_time.tif')
    dat = dat.compute()
    tunits = 'minutes since ' + str(dat.time.min().item())
    tenc = {'time': {'units': tunits, 'calendar': 'proleptic_gregorian'}}
    dat.rio.update_encoding(tenc, inplace=True)
    dat[lwe].rio.to_raster(tif1h, dtype='uint16')
    dat['time'].rio.to_raster(tif1htime, dtype='uint16')
    rioxarray.open_rasterio(tif1htime).rio.update_attrs({'units': tunits}).rio.to_raster(tif1htime)

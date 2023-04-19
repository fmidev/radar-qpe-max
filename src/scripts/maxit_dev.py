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
from radproc.radar import z_r_qpe, source2dict


ACC = 'lwe_accum'
PYART_AEQD_FMT = '+proj={proj} +lon_0={lon_0} +lat_0={lat_0} +R={R}'
QPE_CACHE_FMT = 'tstep{ts}_{size}px{resolution}m.nc'
LWE_SCALE_FACTOR = 0.01
UINT16_FILLVAL = np.iinfo(np.uint16).max
DEFAULT_ENCODING = {lwe: {'zlib': True,
                          '_FillValue': UINT16_FILLVAL,
                          'dtype': 'u2',
                          'scale_factor': LWE_SCALE_FACTOR}}
ATTRS = {ACC: {'units': 'mm',
               'standard_name': 'lwe_thickness_of_precipitation_amount',
               'long_name': 'maximum precipitation accumulation',
               '_FillValue': UINT16_FILLVAL},
         'time': {'long_name': 'end time of maximum precipitation accumulation period',
                  '_FillValue': UINT16_FILLVAL}}
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
    return dict(proj='aeqd', lat_0=lat, lon_0=lon, R=6370997)


def save_precip_grid(radar, outfile, size=2048, resolution=250):
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
    rda.to_netcdf(outfile, encoding=DEFAULT_ENCODING)


def qpe_grids_caching(h5paths, size, resolution, ignore_cache):
    for fpath in h5paths:
        radar = pyart.aux_io.read_odim_h5(fpath)
        t = generate_radar_time_begin(radar)
        ts = t.strftime('%Y%m%d%H%M')
        outfname = QPE_CACHE_FMT.format(ts=ts, size=size, resolution=resolution)
        outfile = os.path.join(CACHE_DIR, outfname)
        if os.path.isfile(outfile) and not ignore_cache:
            continue
        z_r_qpe(radar)
        save_precip_grid(radar, outfile, size=size, resolution=resolution)


def maxit(h5paths, resultsdir, size=2048, resolution=250, win='1 D',
          chunksize=128, ignore_cache=False):
    # takes forever with small chunksize
    chunks = {'x': chunksize, 'y': chunksize}
    qpe_grids_caching(h5paths, size, resolution, ignore_cache)
    globstr = QPE_CACHE_FMT.format(ts='*', size=size, resolution=resolution)
    ncglob = os.path.join(CACHE_DIR, globstr)
    rds = xr.open_mfdataset(ncglob, chunks=chunks, data_vars='minimal',
                            engine='h5netcdf', parallel=True)
    iwin = rds.time.groupby(rds.time.dt.floor(win)).sizes['time']
    dwin = pd.to_timedelta(win)
    win_trim = win.replace(' ', '').lower()
    t_round = rds.indexes['time'].round('min')
    rds['time'] = t_round
    rds = rds.convert_calendar(calendar='standard', use_cftime=True)
    tdelta = pd.to_timedelta(rds.indexes['time'].freq)
    tstep_last = pd.to_datetime(date+datetime.timedelta(days=1))-tdelta
    tstep_pre = pd.to_datetime(date)-dwin+tdelta
    rollsel = rds.sel(time=slice(tstep_pre, tstep_last))
    accums = (rollsel[lwe].rolling({'time': iwin}).sum()/12).to_dataset()
    accums = accums.rename({lwe: ACC})
    dat = accums.max('time').rio.write_crs(3067)
    dat['time'] = accums[ACC].idxmax(dim='time', keep_attrs=True)
    tstamp = accums.time[-1].dt.strftime('%Y%m%d').item()
    dat[ACC].attrs.update(ATTRS[ACC])
    acc_cell = {'cell_methods': f'time: maximum (interval: {win.lower()})'}
    dat[ACC].attrs.update(acc_cell)
    dat[ACC].attrs.update(rds.attrs)
    dat['time'].attrs.update(rds.attrs)
    dat['time'].attrs.update(ATTRS['time'])
    dat.rio.write_coordinate_system(inplace=True)
    tif1h = os.path.join(resultsdir, f"{tstamp}_max{win_trim}.tif")
    tif1htime = os.path.join(resultsdir, f'{tstamp}_max{win_trim}_time.tif')
    dat = dat.compute()
    tunits = 'minutes since ' + str(dat.time.min().item())
    enc = {'time': {'units': tunits, 'calendar': 'proleptic_gregorian'}}
    dat.rio.update_encoding(enc, inplace=True)
    dat[ACC].rio.to_raster(tif1h, dtype='uint16', compress='deflate')
    dat['time'].rio.to_raster(tif1htime, dtype='uint16', compress='deflate')
    unidat = rioxarray.open_rasterio(tif1htime).rio.update_attrs({'units': tunits})
    unidat.rio.to_raster(tif1htime, compress='deflate')


if __name__ == '__main__':
    plt.close('all')
    clear_cache()
    date = datetime.date(2022, 6, 4)
    #
    resultsdir = os.path.expanduser('~/results/sademaksit')
    datadir = os.path.expanduser('~/data/alakulma')
    h5paths = sorted(glob(os.path.join(datadir, '*-A.h5')))
    maxit(h5paths, resultsdir, size=256, resolution=2000, ignore_cache=True)

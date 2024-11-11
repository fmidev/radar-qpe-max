# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

# builtin
import os
import re
import time
import logging
import datetime
import warnings
from glob import glob
from typing import Optional, List

# pypi
import h5py
import xarray as xr
import rioxarray
import pyart
from pyart.aux_io.odim_h5 import _to_str
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS

# local
from radproc.aliases.fmi import LWE
from radproc.radar import z_r_qpe
from radproc.tools import source2dict
from qpemax._version import __version__


EPSG_TARGET = 3067
ZH = 'DBZH'
ACC = 'lwe_accum'
qpefmt = '{ts}{nod}{size}px{resolution}m{corr}'
DATEGLOB = '????????????'
QPE_CACHE_FMT = qpefmt + '.nc'
QPE_TIF_FMT = qpefmt + '.tif'
LWE_SCALE_FACTOR = 0.01
DATEFMT = '%Y%m%d'
UINT16_FILLVAL = np.iinfo(np.uint16).max
DEFAULT_ENCODING = {LWE: {'zlib': True,
                          'complevel': 9,
                          '_FillValue': UINT16_FILLVAL,
                          'dtype': 'uint16',
                          'scale_factor': LWE_SCALE_FACTOR},
                    'time': {'dtype': 'int32',}
                    }
ATTRS = {ACC: {'units': 'mm',
               'standard_name': 'lwe_thickness_of_precipitation_amount',
               '_FillValue': UINT16_FILLVAL},
         'time': {'long_name': 'end time of maximum precipitation accumulation period',
                  '_FillValue': UINT16_FILLVAL}}
DEFAULT_CACHE_DIR = '/tmp/radar-qpe-max'
SINGLE_SCAN_SUBDIR = 'scan-accums'
COG_COMPRESS = 'LZW'

logger = logging.getLogger('airflow.task')


def read_odim_h5(h5path: str, **kws) -> pyart.core.Radar:
    """Read radar data from ODIM H5 file."""
    radar = pyart.aux_io.read_odim_h5(h5path, **kws)
    # workaround for pyart bug
    radar.altitude['data'] = radar.altitude['data'].flatten()
    radar.latitude['data'] = radar.latitude['data'].flatten()
    radar.longitude['data'] = radar.longitude['data'].flatten()
    return radar


def basic_gatefilter(radar: pyart.core.Radar, field: str = ZH) -> pyart.filters.GateFilter:
    """basic gatefilter based on examples in pyart documentation"""
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(field)
    return gatefilter


def create_grid(radar: pyart.core.Radar, size: int = 2048,
                resolution: int = 250) -> pyart.core.Grid:
    """
    Create a grid from radar data.

    Args:
        radar (pyart.core.Radar): The radar object containing the data.
        size (int, optional): The size of the grid. Defaults to 2048.
        resolution (int, optional): The resolution of the grid. Defaults to 250.

    Returns:
        pyart.core.Grid: The grid object containing the gridded data.
    """
    gf = basic_gatefilter(radar)
    crs_target = CRS(EPSG_TARGET)
    with warnings.catch_warnings():
        # "you might lose some information blah blah"
        warnings.filterwarnings("ignore", category=UserWarning)
        projd_target = crs_target.to_dict()
    transp = Transformer.from_crs('WGS84', crs_target)
    radar_y, radar_x = transp.transform(radar.latitude['data'][0],
                                        radar.longitude['data'][0])
    r_m = size*resolution/2
    radar_alt = radar.altitude['data'][0]
    h_factor_xy = 1.0
    grid_shape = (1, size, size)
    grid_limits = ((0, 10000), # upper limit does not seem to matter
                   (radar_x-r_m, radar_x+r_m),
                   (radar_y-r_m, radar_y+r_m))
    grid = pyart.map.grid_from_radars(radar, gatefilters=gf,
                                      gridding_algo='map_gates_to_grid',
                                      grid_shape=grid_shape,
                                      grid_limits=grid_limits, fields=[LWE],
                                      grid_projection=projd_target,
                                      grid_origin=(0, 0),
                                      grid_origin_alt=radar_alt,
                                      h_factor=(50, h_factor_xy, h_factor_xy),
                                      min_radius=330,
                                      roi_func='dist_beam')
    grid.x['data'] = grid.x['data'].flatten()
    grid.y['data'] = grid.y['data'].flatten()
    return grid


def save_precip_grid(radar: pyart.core.Radar, cachefile: str,
                     tiffile: Optional[str] = None, size: int = 2048,
                     resolution: int = 250, scans_per_hour: int = 12,
                     blocksize: int = 512) -> None:
    """Save precipitation products from Radar objects to files.

    Precipitation rate is saved to netcdf `cachefile`, and optionally 5-minute
    accumulation to `tiffile`."""
    grid = create_grid(radar, size=size, resolution=resolution)
    rds = grid.to_xarray().isel(z=0).reset_coords(drop=True)
    rda = rds[LWE].fillna(0)
    rda.rio.write_crs(EPSG_TARGET, inplace=True)
    rda = rda.to_dataset()
    rda['time'] = rda.time.dt.round('min')  # round to minutes
    try:
        rda.attrs.update(source2dict(radar.metadata['source']))
    except KeyError:
        logger.warning('No source metadata found.')
    # TODO: retain existing history if any
    rda.attrs.update({'history': __version__})
    # netcdf4 engine causes HDF error on some machines
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            rda.to_netcdf(cachefile, encoding=DEFAULT_ENCODING, engine='h5netcdf')
            break
        except BlockingIOError as e:
            logger.error(f'Error writing {cachefile}: {e}')
            if e.errno == 11: # unable to lock file
                retries += 1
                time.sleep(1)
                if os.path.isfile(cachefile):
                    logger.warning('File was created by another process.')
                    break
                logger.error('Retrying after delay.')
            else:
                raise
        except Exception as e:
            logger.error(f'Error writing {cachefile}: {e}')
            raise
    if isinstance(tiffile, str):
        acc = (rda.isel(time=0)[LWE]/scans_per_hour).rename(ACC)
        acc.attrs.update(ATTRS[ACC])
        acc.rio.update_encoding({'scale_factor': LWE_SCALE_FACTOR}, inplace=True)
        logger.info(f'Writing geotiff {tiffile}')
        acc.rio.to_raster(
            tiffile, driver='COG',
            dtype='uint16', compress=COG_COMPRESS,
            blocksize=blocksize,
        )


def sweep_start_datetime(hfile: h5py.File, dset: str) -> datetime.datetime:
    """Get the starting time of the sweep defined by the dataset."""
    dset_what = hfile[dset]["what"].attrs
    start_str = _to_str(dset_what["startdate"] + dset_what["starttime"])
    return datetime.datetime.strptime(start_str, "%Y%m%d%H%M%S")


def qpe_grid_caching(h5path: str, size: int, resolution: int,
                     ignore_cache: bool, resultsdir: Optional[str] = None,
                     cachedir: str = DEFAULT_CACHE_DIR, dbz_field: str = ZH,
                     **kws) -> str:
    """Create precipitation grid cache file and optionally geotiff."""
    dset = 'dataset1' # lowest elevation
    corr = '_c' if 'C' in dbz_field else ''
    if isinstance(resultsdir, str):
        tifdir = os.path.join(resultsdir, SINGLE_SCAN_SUBDIR)
        os.makedirs(tifdir, exist_ok=True)
    # read ts and NOD using h5py for increased performance
    with h5py.File(h5path, 'r') as h5f:
        t = sweep_start_datetime(h5f, f'/{dset}')
        ts = t.strftime('%Y%m%d%H%M')
        source = _to_str(h5f['/what'].attrs['source'])
        nod = source.split('NOD:')[1].split(',')[0]
    cachefname = QPE_CACHE_FMT.format(ts=ts, nod=nod, size=size,
                                      resolution=resolution, corr=corr)
    cachefile = os.path.join(cachedir, cachefname)
    if os.path.isfile(cachefile) and not ignore_cache:
        logger.info(f'Cache file {cachefile} exists.')
        return nod
    logger.info(f'Creating cache file {cachefile}')
    radar = read_odim_h5(h5path, include_datasets=[dset], file_field_names=True)
    if isinstance(resultsdir, str):
        tifname = QPE_TIF_FMT.format(ts=ts, nod=nod, size=size,
                                     resolution=resolution, corr=corr)
        tiffile = os.path.join(tifdir, tifname)
    else:
        tiffile = None
    z_r_qpe(radar, dbz_field=dbz_field)
    save_precip_grid(radar, cachefile, tiffile=tiffile, size=size,
                     resolution=resolution, **kws)
    return nod


def two_day_glob(date: datetime.date,
                 globfmt: str = '{date}*.h5', **kws) -> tuple[List[str], List[str]]:
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


def _write_attrs(data: xr.Dataset, rdattrs: dict, win: str) -> xr.Dataset:
    """Write attributes to precipitation maximum data."""
    dat = data.copy()
    dat[ACC].attrs.update(ATTRS[ACC])
    dat[ACC].attrs.update({'long_name': 'maximum precipitation accumulation'})
    acc_cell = {'cell_methods': f'time: maximum (interval: {win.lower()})'}
    dat[ACC].attrs.update(acc_cell)
    dat[ACC].attrs.update(rdattrs)
    dat['time'].attrs.update(rdattrs)
    dat['time'].attrs.update(ATTRS['time'])
    return dat.rio.write_coordinate_system()


def _write_tifs(dat: xr.Dataset, tifp: str, tift: str, blocksize: int = 512) -> None:
    """Write main geotiff products to files."""
    tunits = 'minutes since ' + str(dat.time.min().item())
    enc = {'time': {'units': tunits, 'calendar': 'proleptic_gregorian'}}
    dat.rio.update_encoding(enc, inplace=True)
    dat[ACC].rio.to_raster(
        tifp, driver='COG',
        dtype='uint16', compress=COG_COMPRESS,
        blocksize=blocksize,
    )
    dat['time'].rio.to_raster(tift, dtype='uint16', compress=COG_COMPRESS)
    unidat = rioxarray.open_rasterio(tift).rio.update_attrs({'units': tunits})
    unidat.rio.to_raster(
        tift, compress=COG_COMPRESS,
        driver='COG', dtype='uint16',
        blocksize=blocksize,
    )


def combine_rds(ncfiles: List[str], chunksize: int, date: datetime.date, ignore_cache: bool) -> str:
    """Prepare precip rate dataset.

    Load cached precipitation rasters from netcdf files and write them to a
    single chunked netcdf file."""
    ncpath = re.sub(r'\d{12}', date.strftime(DATEFMT), ncfiles[0])
    if not os.path.isfile(ncpath) or ignore_cache: # then create it
        logger.info('Loading cached individual precipitation rasters.')
        # combine all files into a single dataset
        rds = xr.open_mfdataset(ncfiles, data_vars='minimal',
                                engine='h5netcdf', phony_dims='sort')
        # write dataset chunked by horizontal dimensions
        encoding = DEFAULT_ENCODING.copy()
        encoding[LWE]['chunksizes'] = (1, chunksize, chunksize)
        logger.info(f'Writing chunked dataset {ncpath}')
        rds.to_netcdf(ncpath, encoding=encoding, engine='h5netcdf')
    return ncpath


def load_chunked_dataset(ncpath: str, chunksize: int) -> xr.Dataset:
    """Load chunked dataset.

    Return the dataset with time rounded to minutes."""
    # open the dataset in chunks
    logger.info(f'Loading chunked dataset {ncpath}')
    rds = xr.open_dataset(ncpath, chunks={'x': chunksize, 'y': chunksize},
                          engine='h5netcdf')
    logger.info('Rasters loaded.')
    rds['time'] = rds.indexes['time'].round('min')
    return rds.convert_calendar(calendar='standard', use_cftime=True)


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
        logger.error('Inconsistent timestep lengths in file paths. Assuming 5min.')
        return tstep_default
    if tstep != tstep_default:
        logger.warning(f'Unusual timestep length {tstep}.')
    return tstep


def maxit(date: datetime.date, h5paths: List[str], resultsdir: str,
          cachedir: str = DEFAULT_CACHE_DIR, size: int = 2048,
          resolution: int = 250, win: str = '1 D',
          chunksize: int = 256, ignore_cache: bool = False,
          dbz_field: str = ZH) -> List[str]:
    """Moving window maximum precipitation accumulation."""
    chunks = {'x': chunksize, 'y': chunksize}
    corr = '_c' if 'C' in dbz_field else '' # mark attenuation correction
    os.makedirs(cachedir, exist_ok=True)
    # the guess is used only for geotiff accumulation scaling
    tstep_guess = tstep_from_fpaths(h5paths)
    acc_scaling_guess = datetime.timedelta(hours=1)/tstep_guess
    logger.info('Updating precipitation raster cache.')
    for fpath in h5paths:
        nod = qpe_grid_caching(fpath, size, resolution, ignore_cache,
                               resultsdir=resultsdir,
                               cachedir=cachedir, dbz_field=dbz_field,
                               scans_per_hour=acc_scaling_guess)
    globfmt = QPE_CACHE_FMT.format(ts='{date}????', nod=nod, size=size,
                                   resolution=resolution, corr=corr)
    ncfiles, ncfiles_obsolete = two_day_glob(date, globfmt=os.path.join(cachedir, globfmt))
    ncfile = combine_rds(ncfiles, chunksize, date, ignore_cache)
    rds = load_chunked_dataset(ncfile, chunksize)
    win_trim = win.replace(' ', '')
    winlow = win_trim.lower()
    # number of timesteps in window (e.g. 288 5min steps in a day)
    iwin = rds.time.groupby(rds.time.dt.floor(win_trim)).sizes['time']
    dwin = pd.to_timedelta(win)
    tind = rds.indexes['time']
    # timestep length as timedelta
    tdelta = pd.to_timedelta(tind.freq) or pd.Series(tind).diff().median()
    tstep_last = pd.to_datetime(date+datetime.timedelta(days=1))-tdelta
    tstep_pre = pd.to_datetime(date)-dwin+tdelta
    rollsel = rds.sel(time=slice(tstep_pre, tstep_last))
    # The data is still precip rate, so scale to mm
    acc_scaling = datetime.timedelta(hours=1)/tdelta # 12 for 5min steps
    accums = (rollsel[LWE].rolling({'time': iwin}).sum()/acc_scaling).to_dataset()
    accums = accums.rename({LWE: ACC})
    dat = accums.max('time').rio.write_crs(EPSG_TARGET)
    dat['time'] = accums[ACC].idxmax(dim='time', keep_attrs=True).chunk(chunks)
    dat = dat.compute()
    tstamp = accums.time[-1].dt.strftime(DATEFMT).item()
    dat = _write_attrs(dat, rds.attrs, win)
    tifp = os.path.join(resultsdir, f'{nod}{tstamp}max{winlow}{size}px{resolution}m{corr}.tif')
    tift = os.path.join(resultsdir, f'{nod}{tstamp}maxtime{winlow}{size}px{resolution}m{corr}.tif')
    _write_tifs(dat, tifp, tift)
    return ncfiles_obsolete
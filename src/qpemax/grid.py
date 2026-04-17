# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

import datetime
import logging
import os
import time
import warnings
from typing import List, Optional

import h5py
import pyart
import rioxarray
from pyart.aux_io.odim_h5 import _to_str
from pyproj import CRS, Transformer

from radproc.aliases.fmi import LWE
from radproc.radar import z_r_qpe
from radproc.tools import source2dict

from qpemax._version import __version__
from qpemax.constants import (
    ATTRS, COG_COMPRESS, DEFAULT_CACHE_DIR, DEFAULT_P_CHUNKSIZE,
    DEFAULT_RESOLUTION, DEFAULT_XY_SIZE, DEFAULT_ENCODING, EPSG_TARGET,
    LWE_SCALE_FACTOR, QPE_TIF_FMT, SINGLE_SCAN_SUBDIR, ZH, ACC,
)
from qpemax.utils import corr_suffix, qpe_cache_fname


logger = logging.getLogger('airflow.task')


def read_odim_h5(h5path: str, **kws) -> pyart.core.Radar:
    """Read radar data from ODIM H5 file."""
    try:
        radar = pyart.aux_io.read_odim_h5(h5path, **kws)
    except Exception as e:
        logger.error(f'Reading {h5path}: {e}')
        raise
    # workaround for pyart bug
    radar.altitude['data'] = radar.altitude['data'].flatten()
    radar.latitude['data'] = radar.latitude['data'].flatten()
    radar.longitude['data'] = radar.longitude['data'].flatten()
    return radar


def basic_gatefilter(
        radar: pyart.core.Radar,
        field: str = ZH) -> pyart.filters.GateFilter:
    """basic gatefilter based on examples in pyart documentation"""
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(field)
    return gatefilter


def create_grid(
        radar: pyart.core.Radar, size: int = DEFAULT_XY_SIZE,
        resolution: int = DEFAULT_RESOLUTION) -> pyart.core.Grid:
    """
    Create a grid from radar data.

    Args:
        radar (pyart.core.Radar): The radar object containing the data.
        size (int, optional): The size of the grid.
        resolution (int, optional): The resolution of the grid.

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
    grid = pyart.map.grid_from_radars(
        radar, gatefilters=gf,
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


def _grid_to_dataset(
        radar: pyart.core.Radar,
        grid: pyart.core.Grid) -> 'xr.Dataset':
    """Convert a pyart Grid to a labelled xarray Dataset with CRS and metadata."""
    import xarray as xr
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
    return rda


def save_precip_grid(
        radar: pyart.core.Radar, cachefile: str,
        tiffile: Optional[str] = None, size: int = DEFAULT_XY_SIZE,
        resolution: int = DEFAULT_RESOLUTION, scans_per_hour: int = 12,
        blocksize: int = 512, p_chunksize: int = DEFAULT_P_CHUNKSIZE) -> None:
    """Save precipitation products from Radar objects to files.

    Precipitation rate is saved to netcdf `cachefile`, and optionally per scan
    accumulation to `tiffile`."""
    grid = create_grid(radar, size=size, resolution=resolution)
    rda = _grid_to_dataset(radar, grid)
    # netcdf4 engine causes HDF error on some machines
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            encoding = DEFAULT_ENCODING.copy()
            encoding[LWE]['chunksizes'] = (1, p_chunksize, p_chunksize)
            rda.to_netcdf(cachefile, encoding=encoding, engine='h5netcdf')
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
        acc.rio.update_encoding(
            {'scale_factor': LWE_SCALE_FACTOR}, inplace=True)
        logger.info(f'Writing geotiff {tiffile}')
        acc.rio.to_raster(
            tiffile, driver='COG',
            dtype='uint16', compress=COG_COMPRESS,
            blocksize=blocksize,
        )


def sweep_start_datetime(
        hfile: h5py.File, dset: str) -> datetime.datetime:
    """Get the starting time of the sweep defined by the dataset."""
    dset_what = hfile[dset]["what"].attrs
    start_str = _to_str(dset_what["startdate"] + dset_what["starttime"])
    return datetime.datetime.strptime(start_str, "%Y%m%d%H%M%S")


def get_nod(h5file: h5py.File) -> str:
    """Get NOD from source metadata."""
    source = _to_str(h5file['/what'].attrs['source'])
    return source.split('NOD:')[1].split(',')[0]


def qpe_grid_caching(
        h5path: str, size: int, resolution: int,
        ignore_cache: bool = False, resultsdir: Optional[str] = None,
        cachedir: str = DEFAULT_CACHE_DIR, dbz_field: str = ZH,
        p_chunksize: int = DEFAULT_P_CHUNKSIZE, **kws) -> str:
    """Create precipitation grid cache file and optionally geotiff."""
    dset = 'dataset1' # lowest elevation
    corr = corr_suffix(dbz_field)
    if isinstance(resultsdir, str):
        tifdir = os.path.join(resultsdir, SINGLE_SCAN_SUBDIR)
        os.makedirs(tifdir, exist_ok=True)
    # read ts and NOD using h5py for increased performance
    logger.debug(f'Reading {h5path}')
    with h5py.File(h5path, 'r') as h5f:
        t = sweep_start_datetime(h5f, f'/{dset}')
        ts = t.strftime('%Y%m%d%H%M')
        nod = get_nod(h5f)
    cachefile = os.path.join(
        cachedir, qpe_cache_fname(ts, nod, size, resolution, corr, p_chunksize))
    if os.path.isfile(cachefile) and not ignore_cache:
        logger.info(f'Cache file {cachefile} exists.')
        return nod
    logger.info(f'Creating cache file {cachefile}')
    radar = read_odim_h5(
        h5path, include_datasets=[dset], file_field_names=True)
    if isinstance(resultsdir, str):
        tifname = QPE_TIF_FMT.format(ts=ts, nod=nod, size=size,
                                     resolution=resolution, corr=corr)
        tiffile = os.path.join(tifdir, tifname)
    else:
        tiffile = None
    z_r_qpe(radar, dbz_field=dbz_field)
    save_precip_grid(radar, cachefile, tiffile=tiffile, size=size,
                     resolution=resolution, p_chunksize=p_chunksize, **kws)
    return nod


def generate_individual_rasters(
        h5paths: List[str], resultsdir: str, cachedir: str = DEFAULT_CACHE_DIR,
        size: int = DEFAULT_XY_SIZE,
        resolution: int = DEFAULT_RESOLUTION,
        p_chunksize: int = DEFAULT_P_CHUNKSIZE,
        ignore_cache: bool = False, dbz_field: str = ZH) -> str:
    """Generate individual precipitation rasters and return the nod."""
    from qpemax.utils import tstep_from_fpaths
    tstep_guess = tstep_from_fpaths(h5paths)
    acc_scaling_guess = datetime.timedelta(hours=1) / tstep_guess
    for fpath in h5paths:
        nod = qpe_grid_caching(
            fpath, size, resolution, ignore_cache,
            resultsdir=resultsdir,
            cachedir=cachedir,
            dbz_field=dbz_field,
            scans_per_hour=acc_scaling_guess,
            p_chunksize=p_chunksize,
        )
    return nod

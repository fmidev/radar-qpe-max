"""command line interface"""
import datetime
import logging
import json

import click

from qpemax import (
    DATEFMT, DEFAULT_CACHE_DIR,
    two_day_glob,
    accu,
    qpe_grid_caching,
    generate_individual_rasters,
    combine_rasters,
    aggmax,
    write_max_tifs,
    __version__)
from qpemax.logs import streamlogger_setup


logger = logging.getLogger('airflow.task')

# multi-line help strings
CHUNKSIZE_HELP = "Horizontal chunksize PX*PX. Larger chunksize speeds up the processing but " \
    "requires more memory."


def autoresolution(size):
    """Determine resolution in meters based on size in pixels."""
    if size>1999:
        return 250
    elif size>999:
        return 500
    elif size>499:
        return 1000
    return 2000


@click.group()
@click.option('-v', '--verbose', is_flag=True, help='debug logging')
@click.version_option()
def cli(verbose):
    """command line interface"""
    parent_logger = logging.getLogger('airflow.task')
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    streamlogger_setup(parent_logger, log_level)
    logger.info(f'qpe, version {__version__}')


@cli.command()
@click.argument('h5file')
@click.option('-s', '--size', metavar='PX', help='output raster size PX*PX', default=1024)
@click.option('-r', '--resolution', metavar='METRE', help='spatial resolution in meters')
@click.option('-f', '--force', is_flag=True, help='Ignore existing netCDF files.')
@click.option('-t', '--tif-dir', type=str, default=None, help='Store optional geotiff output in DIR.')
@click.option('-o', '--out-dir', metavar='DIR', help='netCDF output directory', default=DEFAULT_CACHE_DIR)
@click.option('-z', '--dbz-field', metavar='FIELD', help='use FIELD for DBZ', default='DBZH')
def grid(h5file, size, resolution, force, tif_dir, out_dir, dbz_field):
    """Generate netCDF precipitation grid from ODIM radar data.

    H5FILE is a radar data file in ODIM HDF5 format.

    The output is a netCDF file with precipitation grid. These files are used as cache
    files for the winmax command allowing precomputing."""
    if resolution is None:
        resolution = autoresolution(size)
    qpe_grid_caching(h5file, size, resolution, force, cachedir=out_dir, resultsdir=tif_dir,
                     dbz_field=dbz_field)


@cli.command()
@click.argument('yyyymmdd')
@click.option('-i', '--input-glob', metavar='PATTERN', required=True,
              help='glob pattern of the input data. Available variables: {yyyy}, {mm}, {dd}, {date}')
@click.option('-o', '--output-dir', metavar='DIR', help='destination of geotiff output', required=True)
@click.option('-c', '--cache-dir', metavar='DIR', help='cache directory', default=DEFAULT_CACHE_DIR)
@click.option('-s', '--size', metavar='PX', help='output raster size PX*PX', default=1024)
@click.option('-x', '--chunksize', metavar='PX', help=CHUNKSIZE_HELP, default=256)
@click.option('-r', '--resolution', metavar='METRE', help='spatial resolution in meters')
@click.option('-w', '--window', metavar='WIN', help='length of the time window, e.g. 1D for 1 day', default='1 D')
@click.option('-z', '--dbz-field', metavar='FIELD', help='use FIELD for DBZ', default='DBZH')
@click.option('-l', '--list-obsolete', is_flag=True, help='json list of obsolete cache files')
def winmax(
        yyyymmdd, input_glob, output_dir, cache_dir, size, chunksize,
        resolution, dbz_field, window, list_obsolete):
    """Maximum precipitation accumulation over moving temporal window.

    YYYYMMDD is the date over which the end of the time window moves."""
    if chunksize > size:
        logger.warning(f'chunksize {chunksize} is larger than size {size}.')
        logger.warning('Using size as chunksize.')
        chunksize = size
    if resolution is None:
        resolution = autoresolution(size)
    date = datetime.datetime.strptime(yyyymmdd, DATEFMT)
    h5paths, _ = two_day_glob(date, globfmt=input_glob)
    nod = generate_individual_rasters(
        h5paths, resultsdir=output_dir, cachedir=cache_dir,
        size=size, resolution=resolution, p_chunksize=chunksize,
        dbz_field=dbz_field
    )
    _, nc_obsolete = combine_rasters(
        date, nod, cachedir=cache_dir, size=size, resolution=resolution,
        p_chunksize=chunksize, dbz_field=dbz_field
    )
    accfile, attrs = accu(
        date, nod, cachedir=cache_dir, size=size, resolution=resolution,
        p_chunksize=chunksize, win=window, dbz_field=dbz_field
    )
    dat, dattime = aggmax(accfile, attrs, p_chunksize=chunksize)
    write_max_tifs(
        dat, dattime, date, resultsdir=output_dir, nod=nod, win=window,
        size=size, resolution=resolution
    )
    if list_obsolete:
        print(json.dumps(nc_obsolete))

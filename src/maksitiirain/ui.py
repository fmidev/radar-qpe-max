"""User interface"""
import datetime
import logging

import click

from maksitiirain import DATEFMT, DEFAULT_CACHE_DIR, two_day_glob, maxit, __version__
from maksitiirain.logging import streamlogger_setup


logger = logging.getLogger(__name__)


@click.command()
@click.argument('yyyymmdd')
@click.option('-i', '--input-glob', metavar='PATTERN', required=True,
              help='glob pattern of the input data. Available variables: {yyyy}, {mm}, {dd}, {date}')
@click.option('-o', '--output-dir', metavar='DIR', help='write output here', required=True)
@click.option('-c', '--cache-dir', metavar='DIR', help='cache directory', default=DEFAULT_CACHE_DIR)
@click.option('-s', '--size', metavar='PX', help='output raster size PX*PX', default=1024)
@click.option('-r', '--resolution', metavar='METRE', help='spatial resolution in meters')
@click.option('-w', '--window', metavar='WIN', help='length of the time window, e.g. 1D for 1 day', default='1 D')
@click.option('-z', '--dbz-field', metavar='FIELD', help='use FIELD for DBZ', default='DBZH')
@click.version_option()
def cli(yyyymmdd, input_glob, output_dir, cache_dir, size, resolution, dbz_field, window):
    """Max precipitation accumulation over moving window integration period."""
    parent_logger = logging.getLogger('maksitiirain')
    streamlogger_setup(parent_logger, logging.INFO)
    logger.info(f'sademaksit, version {__version__}')
    if resolution is None:
        if size>1999:
            resolution = 250
        elif size>999:
            resolution = 500
        elif size>499:
            resolution = 1000
        else:
            resolution = 2000
    date = datetime.datetime.strptime(yyyymmdd, DATEFMT)
    h5paths = two_day_glob(date, globfmt=input_glob)
    maxit(date, h5paths, output_dir, cache_dir=cache_dir, size=size, resolution=resolution,
          dbz_field=dbz_field, win=window)

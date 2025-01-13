import os
import logging
import datetime

from qpemax import get_nod, maxit, generate_individual_rasters, combine_rasters, write_max_tifs, two_day_glob
from qpemax.logs import streamlogger_setup


logger = logging.getLogger('airflow.task')


if __name__ == '__main__':
    streamlogger_setup(logger, loglevel=logging.DEBUG)
    date = datetime.date(2022, 6, 4)
    #date = datetime.date(2022, 8, 5)
    date = datetime.date(2024, 5, 29)
    size = 64  # 2048
    resolution = 8000  # 250
    chunksize = 64  # 256
    win = '1D'
    #
    resultsdir = os.path.expanduser('~/results/radar-qpe-max')
    datadir = os.path.expanduser('~/data/alakulma')
    #datadir = os.path.expanduser('~/data/polar/fivih')
    datadir = os.path.expanduser('~/data/polar/filuo')
    h5paths, _ = two_day_glob(date, globfmt=os.path.join(datadir, '{date}*.h5'))
    cachedir = os.path.expanduser('~/.cache/radar-qpe-max')
    # size 1024 or 2048 can still be run on a laptop with chunksize 256

    # Generate individual rasters
    nod = generate_individual_rasters(
        h5paths, resultsdir,
        cachedir=cachedir,
        size=size,
        resolution=resolution,
        chunksize=chunksize,
    )

    # Combine rasters
    ncfile, _ = combine_rasters(
        date, nod, cachedir=cachedir,
        size=size,
        resolution=resolution,
        chunksize=chunksize,
    )

    # Calculate maximum precipitation accumulation
    dat, tstamp = maxit(date, ncfile, win=win, chunksize=chunksize)

    # Write results to GeoTIFF files
    write_max_tifs(dat, tstamp=tstamp, resultsdir=resultsdir, nod=nod, win=win)

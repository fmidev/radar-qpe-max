import os
import logging
import datetime

from qpemax import accu, aggmax, generate_individual_rasters, combine_rasters, write_max_tifs, two_day_glob
from qpemax.logs import streamlogger_setup


logger = logging.getLogger('airflow.task')


if __name__ == '__main__':
    streamlogger_setup(logger, loglevel=logging.DEBUG)
    #date = datetime.date(2022, 6, 4)
    #date = datetime.date(2022, 8, 5)
    date = datetime.date(2024, 5, 29)
    quickrun = True
    if quickrun:
        size = 64
        resolution = 8000
        chunksize = 64
    else:
        size = 2048  # 2048
        resolution = 250  # 250
        chunksize = 128  # 256
    win = '1D'  # '1h' or '1D'
    #
    resultsdir = os.path.expanduser('~/results/radar-qpe-max')
    #datadir = os.path.expanduser('~/data/alakulma')
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
        ignore_cache=False,
    )

    # maximum precipitation accumulation logic
    accfile, attrs = accu(
        date, nod,
        cachedir=cachedir,
        size=size,
        resolution=resolution,
        chunksize=chunksize,
        win=win)
    dat, dattime = aggmax(accfile, attrs)

    # Process geotiff products
    write_max_tifs(dat, dattime, date, resultsdir=resultsdir, nod=nod, win=win,
                   size=size, resolution=resolution)

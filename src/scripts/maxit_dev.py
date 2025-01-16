import os
import logging
import datetime

from qpemax import accu, aggmax, generate_individual_rasters, combine_rasters, write_max_tifs, two_day_glob
from qpemax.logs import streamlogger_setup


logger = logging.getLogger('airflow.task')


if __name__ == '__main__':
    streamlogger_setup(logger, loglevel=logging.DEBUG)

    #from dask.distributed import Client
    #client = Client(n_workers=1, threads_per_worker=1, memory_limit='4GB')
    #logger.info(client.dashboard_link)

    #date = datetime.date(2022, 6, 4)
    #date = datetime.date(2022, 8, 5)
    date = datetime.date(2024, 5, 29)
    quickrun = True
    if quickrun:
        size = 512
        resolution = 1000
        p_chunksize = 256
    else:
        size = 2048  # 2048
        resolution = 250  # 250
        p_chunksize = 512  # 256
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
        p_chunksize=p_chunksize,
    )

    # Combine rasters
    ncfile, _ = combine_rasters(
        date, nod, cachedir=cachedir,
        size=size,
        resolution=resolution,
        p_chunksize=p_chunksize,
        ignore_cache=False,
    )

    # maximum precipitation accumulation logic
    accfile, attrs = accu(
        date, nod,
        cachedir=cachedir,
        size=size,
        resolution=resolution,
        p_chunksize=p_chunksize,
        win=win)
    dat, dattime = aggmax(accfile, attrs, p_chunksize=p_chunksize)

    # Process geotiff products
    write_max_tifs(dat, dattime, date, resultsdir=resultsdir, nod=nod, win=win,
                   size=size, resolution=resolution)

import os
import logging
import datetime

from maksitiirain import maxit, two_day_glob
from maksitiirain.logs import streamlogger_setup


logger = logging.getLogger('maksitiirain')


if __name__ == '__main__':
    streamlogger_setup(logger, loglevel=logging.DEBUG)
    date = datetime.date(2022, 6, 4)
    #date = datetime.date(2022, 8, 5)
    date = datetime.date(2024, 5, 29)
    #
    resultsdir = os.path.expanduser('~/results/sademaksit')
    datadir = os.path.expanduser('~/data/alakulma')
    #datadir = os.path.expanduser('~/data/polar/fivih')
    datadir = os.path.expanduser('~/data/polar/filuo')
    h5paths = two_day_glob(date, globfmt=os.path.join(datadir, '{date}*.h5'))
    # size 1024 can still be run on a laptop with chunksize 256
    maxit(date, h5paths, resultsdir, size=1024, resolution=500, chunksize=256,
          ignore_cache=False, debug=True)

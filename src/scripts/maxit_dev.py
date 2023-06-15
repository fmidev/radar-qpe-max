import os
import logging
import datetime

from maksitiirain import maxit, two_day_glob, streamlogger_setup


logger = logging.getLogger('maksit')


if __name__ == '__main__':
    streamlogger_setup(logger, loglevel=logging.DEBUG)
    #plt.close('all')
    #date = datetime.date(2022, 6, 4)
    date = datetime.date(2022, 8, 5)
    #
    resultsdir = os.path.expanduser('~/results/sademaksit')
    #datadir = os.path.expanduser('~/data/alakulma')
    datadir = os.path.expanduser('~/data/polar/fivih')
    h5paths = two_day_glob(date, globfmt=os.path.join(datadir, '{date}*.h5'))
    maxit(date, h5paths, resultsdir, size=512, resolution=1000)#, ignore_cache=True)

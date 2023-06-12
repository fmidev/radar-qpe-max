import os
import logging
import datetime

from maksitiirain import maxit, two_day_glob


logger = logging.getLogger('maksit')


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    #plt.close('all')
    #date = datetime.date(2022, 6, 4)
    date = datetime.date(2022, 8, 5)
    #
    resultsdir = os.path.expanduser('~/results/sademaksit')
    #datadir = os.path.expanduser('~/data/alakulma')
    datadir = os.path.expanduser('~/data/polar/fivih')
    h5paths = two_day_glob(date, globfmt=os.path.join(datadir, '{date}*.h5'))
    maxit(date, h5paths, resultsdir, size=512, resolution=1000)#, ignore_cache=True)

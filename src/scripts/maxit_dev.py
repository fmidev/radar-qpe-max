import os
import datetime

from sademaksit import maxit, ls_low_elev


if __name__ == '__main__':
    #plt.close('all')
    #clear_cache()
    date = datetime.date(2022, 6, 4)
    #
    resultsdir = os.path.expanduser('~/results/sademaksit')
    datadir = os.path.expanduser('~/data/alakulma')
    h5paths = ls_low_elev(date, datadir)
    maxit(date, h5paths, resultsdir, size=512, resolution=1000)#, ignore_cache=True)

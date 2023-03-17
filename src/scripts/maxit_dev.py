import os

import pyart
import numpy as np

from radproc.aliases import zh, lwe


def db2lin(db):
    """decibels to linear scale"""
    return np.power(10, db/10)


def lin2db(lin):
    """linear to decibel scale"""
    return 10*np.log10(lin)


def rainrate(z):
    return 0.0292*z**(0.6536)


def z_r_qpe(radar):
    dbz = radar.get_field(0, zh)
    z = db2lin(dbz)
    r = rainrate(z)
    rfield = {'units': 'mm h-1', 'data': r}
    radar.add_field(lwe, rfield)


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/alakulma')
    fpath = os.path.join(datadir, '202206031905_radar.raw.fmi_SITE=fivih_TASKS=PPI2-A.h5')
    radar = pyart.aux_io.read_odim_h5(fpath)
    z_r_qpe(radar)

import os

import pyart
import numpy as np

from radproc.aliases import zh, lwe
from radproc.visual import plot_ppi
from radproc.radar import z_r_qpe


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/alakulma')
    fpath = os.path.join(datadir, '202206031905_radar.raw.fmi_SITE=fivih_TASKS=PPI2-A.h5')
    radar = pyart.aux_io.read_odim_h5(fpath)
    z_r_qpe(radar)
    axr = plot_ppi(radar, sweep=0, what=lwe)

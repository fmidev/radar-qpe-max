import os

import pyart
import numpy as np

from radproc.radar import z_r_qpe, ZH
import qpemax


datadir = os.path.expanduser('~/data/polar/filuo')
fpath = os.path.join(datadir, '202405281100_radar.polar.filuo.h5')

# read in the radar data
radar = pyart.aux_io.read_odim_h5(fpath, include_datasets=['dataset1'],
                                  file_field_names=True)

for field in radar.fields:
    radar.fields[field]['data'].mask = False
radar.fields[ZH]['data'] = np.ones_like(radar.fields[ZH]['data'])*40
z_r_qpe(radar, dbz_field=ZH)

print(qpemax.DEFAULT_ENCODING)
qpemax.save_precip_grid(radar, '/tmp/precip_grid.nc', tiffile='/tmp/precip_grid.tif', size=2048, resolution=250)
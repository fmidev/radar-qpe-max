# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

import numpy as np

from radproc.aliases.fmi import LWE


EPSG_TARGET = 3067
ZH = 'DBZH'
ACC = 'lwe_accum'
BASENAME_FMT = '{ts}{nod}{size}px{resolution}m{corr}'
DATEGLOB = '????????????'
QPE_CACHE_FMT = BASENAME_FMT + '{chunksize}ch.nc'
ACC_CACHE_FMT = BASENAME_FMT + '{chunksize}ch_acc{win}.nc'
QPE_TIF_FMT = BASENAME_FMT + '.tif'
LWE_SCALE_FACTOR = 0.01
DATEFMT = '%Y%m%d'
UINT16_FILLVAL = np.iinfo(np.uint16).max
DEFAULT_ENCODING = {
    LWE: {
        'zlib': True,
        'complevel': 4,
        '_FillValue': UINT16_FILLVAL,
        'dtype': 'uint16',
        'scale_factor': LWE_SCALE_FACTOR
    },
    'time': {'dtype': 'int32',}
}
ATTRS = {
    ACC: {
        'units': 'mm',
        'standard_name': 'lwe_thickness_of_precipitation_amount',
        '_FillValue': UINT16_FILLVAL
    },
    'time': {
        'long_name': 'end time of maximum precipitation accumulation period',
        '_FillValue': UINT16_FILLVAL
    }
}
DEFAULT_P_CHUNKSIZE = 512
DEFAULT_COMPUTE_CHUNKSIZE = 256
DEFAULT_ACC_CHUNKSIZE = 32
DEFAULT_RESOLUTION = 250
DEFAULT_XY_SIZE = 2048
DEFAULT_CACHE_DIR = '/tmp/radar-qpe-max'
SINGLE_SCAN_SUBDIR = 'scan-accums'
COG_COMPRESS = 'LZW'

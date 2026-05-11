# SPDX-FileCopyrightText: 2023-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

from qpemax._version import __version__
from qpemax.constants import (
    ACC,
    ACC_CACHE_FMT,
    ATTRS,
    BASENAME_FMT,
    COG_COMPRESS,
    DATEGLOB,
    DATEFMT,
    DEFAULT_ACC_CHUNKSIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_ENCODING,
    DEFAULT_P_CHUNKSIZE,
    DEFAULT_RESOLUTION,
    DEFAULT_XY_SIZE,
    EPSG_TARGET,
    LWE_SCALE_FACTOR,
    QPE_CACHE_FMT,
    QPE_TIF_FMT,
    SINGLE_SCAN_SUBDIR,
    UINT16_FILLVAL,
    ZH,
)
from qpemax.utils import two_day_glob, tstep_from_fpaths, corr_suffix, acc_cache_fname
from qpemax.grid import (
    basic_gatefilter,
    create_grid,
    generate_individual_rasters,
    get_nod,
    qpe_grid_caching,
    read_odim_h5,
    save_precip_grid,
    sweep_start_datetime,
)
from qpemax.accumulate import (
    accu,
    aggmax,
    combine_rasters,
    load_chunked_dataset,
)
from qpemax.composite import composite_max
from qpemax.output import write_max_tifs

__all__ = [
    '__version__',
    # constants
    'ACC', 'ACC_CACHE_FMT', 'ATTRS', 'BASENAME_FMT', 'COG_COMPRESS',
    'DATEGLOB', 'DATEFMT', 'DEFAULT_ACC_CHUNKSIZE', 'DEFAULT_CACHE_DIR',
    'DEFAULT_ENCODING', 'DEFAULT_P_CHUNKSIZE', 'DEFAULT_RESOLUTION',
    'DEFAULT_XY_SIZE', 'EPSG_TARGET', 'LWE_SCALE_FACTOR', 'QPE_CACHE_FMT',
    'QPE_TIF_FMT', 'SINGLE_SCAN_SUBDIR', 'UINT16_FILLVAL', 'ZH',
    # utils
    'two_day_glob', 'tstep_from_fpaths',
    'corr_suffix', 'acc_cache_fname',
    # grid
    'basic_gatefilter', 'create_grid', 'generate_individual_rasters',
    'get_nod', 'qpe_grid_caching', 'read_odim_h5', 'save_precip_grid',
    'sweep_start_datetime',
    # accumulate
    'accu', 'aggmax', 'combine_rasters', 'load_chunked_dataset',
    # output
    'write_max_tifs',
    # composite
    'composite_max',
]

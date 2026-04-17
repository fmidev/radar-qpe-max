import datetime

import numpy as np
import pandas as pd
import pyart
import xarray as xr

from qpemax import basic_gatefilter, ZH, tstep_from_fpaths
from qpemax.utils import acc_cache_fname, corr_suffix
from qpemax.grid import _grid_to_dataset
from qpemax.accumulate import _accu_time_bounds


def test_basic_gatefilter():
    radar = pyart.testing.make_target_radar()
    # rename the "reflectivity" field to ZH
    radar.add_field(ZH, radar.fields.pop('reflectivity'))
    gatefilter = basic_gatefilter(radar)
    assert isinstance(gatefilter, pyart.filters.GateFilter)
    assert np.all(gatefilter.gate_excluded == 0)
    assert np.all(gatefilter.gate_included == 1)


def test_tstep_from_fpaths():
    fpaths = [
        "/path/to/202405280000_radar.polar.filuo.h5",
        "/path/to/202405280005_radar.polar.filuo.h5",
        "/path/to/202405280010_radar.polar.filuo.h5",
    ]
    expected_tstep = datetime.timedelta(minutes=5)
    assert tstep_from_fpaths(fpaths) == expected_tstep

    fpaths = [
        "/path/to/202405280000_radar.polar.filuo.h5",
        "/path/to/202405280010_radar.polar.filuo.h5",
        "/path/to/202405280020_radar.polar.filuo.h5",
    ]
    expected_tstep = datetime.timedelta(minutes=10)
    assert tstep_from_fpaths(fpaths) == expected_tstep

    fpaths = [
        "/path/to/202405280000_radar.polar.filuo.h5",
        "/path/to/202405280015_radar.polar.filuo.h5",
        "/path/to/202405280030_radar.polar.filuo.h5",
    ]
    expected_tstep = datetime.timedelta(minutes=15)
    assert tstep_from_fpaths(fpaths) == expected_tstep


def test_corr_suffix():
    assert corr_suffix('DBZH') == ''
    assert corr_suffix('DBZHC') == '_c'


def test_acc_cache_fname():
    date = datetime.date(2024, 5, 28)
    fname = acc_cache_fname(date, 'filuo', 2048, 250, '', 32, '1D')
    assert fname == '20240528filuo2048px250m32ch_acc1d.nc'
    fname_corr = acc_cache_fname(date, 'filuo', 2048, 250, '_c', 32, '1D')
    assert fname_corr == '20240528filuo2048px250m_c32ch_acc1d.nc'


def _make_rds(freq='5min', date='2024-05-28', n_days=2):
    """Synthetic precipitation-rate dataset for testing."""
    times = pd.date_range(date, periods=288 * n_days, freq=freq)
    data = np.zeros((len(times), 4, 4), dtype='float32')
    return xr.Dataset(
        {'lwe_accum': xr.DataArray(data, dims=['time', 'y', 'x'],
                                   coords={'time': times})})


def test_accu_time_bounds_5min():
    date = datetime.date(2024, 5, 28)
    rds = _make_rds(freq='5min', date='2024-05-27')
    tstep_pre, tstep_last, iwin, tdelta = _accu_time_bounds(rds, date, '1D')

    assert tdelta == pd.to_timedelta('5min')
    assert iwin == 288  # 24h / 5min
    assert tstep_last == pd.Timestamp('2024-05-28 23:55')
    assert tstep_pre == pd.Timestamp('2024-05-27 00:05')


def test_accu_time_bounds_10min():
    date = datetime.date(2024, 5, 28)
    rds = _make_rds(freq='10min', date='2024-05-27')
    tstep_pre, tstep_last, iwin, tdelta = _accu_time_bounds(rds, date, '1D')

    assert tdelta == pd.to_timedelta('10min')
    assert iwin == 144  # 24h / 10min
    assert tstep_last == pd.Timestamp('2024-05-28 23:50')
    assert tstep_pre == pd.Timestamp('2024-05-27 00:10')


def test_grid_to_dataset():
    from radproc.radar import z_r_qpe
    from qpemax.constants import ZH as ZH_FIELD
    radar = pyart.testing.make_target_radar()
    radar.add_field(ZH_FIELD, radar.fields.pop('reflectivity'))
    z_r_qpe(radar, dbz_field=ZH_FIELD)
    from qpemax.grid import create_grid
    grid = create_grid(radar, size=16, resolution=5000)
    from radproc.aliases.fmi import LWE
    rda = _grid_to_dataset(radar, grid)
    assert LWE in rda
    assert rda[LWE].isnull().sum() == 0  # fillna(0) applied
    assert rda.rio.crs is not None
    assert 'history' in rda.attrs
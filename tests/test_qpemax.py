import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyart
import pytest
import rasterio
import xarray as xr

from radproc.aliases.fmi import LWE

from qpemax import basic_gatefilter, ZH, tstep_from_fpaths
from qpemax.accumulate import _accu_time_bounds, load_chunked_dataset
from qpemax.cli import autoresolution
from qpemax.composite import composite_max
from qpemax.constants import LWE_SCALE_FACTOR, UINT16_FILLVAL
from qpemax.grid import (
    _grid_to_dataset, _z_r_qpe, create_grid, get_nod, qpe_grid_caching,
    read_odim_h5, sweep_start_datetime,
)
from qpemax.output import _write_dat_attrs, _write_dat_tif, _write_dattime_attrs
from qpemax.utils import acc_cache_fname, corr_suffix, two_day_glob


DATA_DIR = Path(__file__).parent / 'data'
TEST_H5 = DATA_DIR / '202604170000_radar.polar.fivih.h5'


# --- Fixtures ---

@pytest.fixture(scope='module')
def radar():
    r = read_odim_h5(str(TEST_H5), include_datasets=['dataset1'], file_field_names=True)
    _z_r_qpe(r, dbz_field=ZH)
    return r


# --- Helpers ---

def _make_spatial_da():
    """Minimal 2D DataArray with CRS for attribute tests."""
    da = xr.DataArray(
        np.zeros((4, 4)), dims=['y', 'x'],
        coords={'x': np.arange(4) * 250., 'y': np.arange(4) * 250.})
    return da.rio.write_crs(3067)


def _make_rds(freq='5min', date='2024-05-28', n_days=2):
    """Synthetic precipitation-rate dataset for accumulation tests."""
    times = pd.date_range(date, periods=288 * n_days, freq=freq)
    data = np.zeros((len(times), 4, 4), dtype='float32')
    return xr.Dataset(
        {LWE: xr.DataArray(data, dims=['time', 'y', 'x'], coords={'time': times})})


# --- Simple unit tests (no files) ---

def test_basic_gatefilter():
    radar = pyart.testing.make_target_radar()
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
    assert tstep_from_fpaths(fpaths) == datetime.timedelta(minutes=5)

    fpaths = [
        "/path/to/202405280000_radar.polar.filuo.h5",
        "/path/to/202405280010_radar.polar.filuo.h5",
        "/path/to/202405280020_radar.polar.filuo.h5",
    ]
    assert tstep_from_fpaths(fpaths) == datetime.timedelta(minutes=10)

    fpaths = [
        "/path/to/202405280000_radar.polar.filuo.h5",
        "/path/to/202405280015_radar.polar.filuo.h5",
        "/path/to/202405280030_radar.polar.filuo.h5",
    ]
    assert tstep_from_fpaths(fpaths) == datetime.timedelta(minutes=15)


def test_autoresolution():
    assert autoresolution(2000) == 250
    assert autoresolution(1999) == 500
    assert autoresolution(1000) == 500
    assert autoresolution(999) == 1000
    assert autoresolution(500) == 1000
    assert autoresolution(499) == 2000


def test_corr_suffix():
    assert corr_suffix('DBZH') == ''
    assert corr_suffix('DBZHC') == '_c'


def test_acc_cache_fname():
    date = datetime.date(2024, 5, 28)
    assert acc_cache_fname(date, 'filuo', 2048, 250, '', 32, '1D') == \
        '20240528filuo2048px250m32ch_acc1d.nc'
    assert acc_cache_fname(date, 'filuo', 2048, 250, '_c', 32, '1D') == \
        '20240528filuo2048px250m_c32ch_acc1d.nc'


def test_write_dat_attrs():
    result = _write_dat_attrs(_make_spatial_da(), {'NOD': 'fivih'})
    assert result.attrs['units'] == 'mm'
    assert result.attrs['long_name'] == 'maximum precipitation accumulation'
    assert result.attrs['NOD'] == 'fivih'


def test_write_dattime_attrs():
    result = _write_dattime_attrs(_make_spatial_da(), {'NOD': 'fivih'})
    assert 'end time' in result.attrs['long_name']
    assert result.attrs['NOD'] == 'fivih'


def test_accu_time_bounds_5min():
    date = datetime.date(2024, 5, 28)
    rds = _make_rds(freq='5min', date='2024-05-27')
    tstep_pre, tstep_last, iwin, tdelta = _accu_time_bounds(rds, date, '1D')
    assert tdelta == pd.to_timedelta('5min')
    assert iwin == 288
    assert tstep_last == pd.Timestamp('2024-05-28 23:55')
    assert tstep_pre == pd.Timestamp('2024-05-27 00:05')


def test_accu_time_bounds_10min():
    date = datetime.date(2024, 5, 28)
    rds = _make_rds(freq='10min', date='2024-05-27')
    tstep_pre, tstep_last, iwin, tdelta = _accu_time_bounds(rds, date, '1D')
    assert tdelta == pd.to_timedelta('10min')
    assert iwin == 144
    assert tstep_last == pd.Timestamp('2024-05-28 23:50')
    assert tstep_pre == pd.Timestamp('2024-05-27 00:10')


def test_accu_time_bounds_cftime():
    """tdelta must not be NaT after convert_calendar (CFTimeIndex has freq=None)."""
    date = datetime.date(2024, 5, 28)
    rds = _make_rds(freq='5min', date='2024-05-27')
    rds = rds.convert_calendar(calendar='standard', use_cftime=True)
    tstep_pre, tstep_last, iwin, tdelta = _accu_time_bounds(rds, date, '1D')
    assert tdelta == pd.to_timedelta('5min')
    assert iwin == 288
    assert tstep_pre is not pd.NaT
    assert tstep_last is not pd.NaT


# --- Tests needing tmp_path ---

def test_two_day_glob(tmp_path):
    date = datetime.date(2024, 5, 28)
    for fname in ['202405270000_radar.h5', '202405270005_radar.h5',
                  '202405280000_radar.h5', '202405280005_radar.h5']:
        (tmp_path / fname).touch()
    globfmt = str(tmp_path / '{date}*_radar.h5')
    all_files, prev_files = two_day_glob(date, globfmt=globfmt)
    assert len(all_files) == 4
    assert all_files == sorted(all_files)
    assert len(prev_files) == 2
    assert all('20240527' in f for f in prev_files)


def test_load_chunked_dataset(tmp_path):
    # Sub-minute timestamps to verify rounding
    times = pd.date_range('2024-05-28 00:00:30', periods=5, freq='5min')
    ds = xr.Dataset({LWE: xr.DataArray(
        np.zeros((5, 4, 4)), dims=['time', 'y', 'x'], coords={'time': times})})
    ncpath = str(tmp_path / 'test.nc')
    ds.to_netcdf(ncpath, engine='h5netcdf')
    result = load_chunked_dataset(ncpath)
    assert LWE in result
    assert (result.time.dt.second == 0).all()


# --- Integration tests (real ODIM H5 file) ---

def test_read_odim_h5():
    r = read_odim_h5(str(TEST_H5), include_datasets=['dataset1'], file_field_names=True)
    assert isinstance(r, (pyart.core.Radar, pyart.xradar.Xradar))
    assert r.altitude['data'].ndim == 1
    assert r.latitude['data'].ndim == 1
    assert r.longitude['data'].ndim == 1
    assert ZH in r.fields


def test_sweep_start_datetime():
    with h5py.File(str(TEST_H5), 'r') as h5f:
        t = sweep_start_datetime(h5f, '/dataset1')
    assert t == datetime.datetime(2026, 4, 17, 0, 0, 2)


def test_get_nod():
    with h5py.File(str(TEST_H5), 'r') as h5f:
        nod = get_nod(h5f)
    assert nod == 'fivih'


def test_create_grid(radar):
    grid = create_grid(radar, size=16, resolution=5000)
    assert isinstance(grid, pyart.core.Grid)
    assert grid.x['data'].shape == (16,)
    assert grid.y['data'].shape == (16,)
    assert LWE in grid.fields


def test_grid_to_dataset(radar):
    grid = create_grid(radar, size=16, resolution=5000)
    rda = _grid_to_dataset(radar, grid)
    assert LWE in rda
    assert rda[LWE].isnull().sum() == 0   # fillna(0) applied
    assert rda.rio.crs is not None
    assert 'history' in rda.attrs


def test_qpe_grid_caching(tmp_path):
    kws = dict(size=16, resolution=5000, p_chunksize=16, cachedir=str(tmp_path))
    nod = qpe_grid_caching(str(TEST_H5), **kws)
    assert nod == 'fivih'
    cache_files = list(tmp_path.glob('*.nc'))
    assert len(cache_files) == 1
    # second call must use cache — no new files
    assert qpe_grid_caching(str(TEST_H5), **kws) == 'fivih'
    assert len(list(tmp_path.glob('*.nc'))) == 1


# --- COG scale/offset integration tests ---

def _make_cog_da(values=None):
    """Minimal float DataArray simulating post-aggmax output (no encoding)."""
    if values is None:
        values = np.array([[1.23, 4.56], [0.0, np.nan]], dtype='float64')
    da = xr.DataArray(values, dims=['y', 'x'],
                      coords={'x': [0.0, 250.0], 'y': [250.0, 0.0]})
    da = da.rio.set_spatial_dims('x', 'y')
    da.rio.write_crs(3067, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    return da


def test_write_dat_tif_scale_offset(tmp_path):
    """_write_dat_tif must embed GDAL band scale/offset in the COG."""
    da = _make_cog_da()
    tifp = str(tmp_path / 'test_dat.tif')
    _write_dat_tif(da, tifp, blocksize=128)
    with rasterio.open(tifp) as ds:
        assert ds.scales == (LWE_SCALE_FACTOR,)
        assert ds.offsets == (0.0,)
        assert ds.dtypes == ('uint16',)
        assert ds.nodata == UINT16_FILLVAL
        raw = ds.read(1)
    # 1.23 mm -> round(1.23/0.01) = 123
    assert raw[0, 0] == 123
    assert raw[0, 1] == 456
    assert raw[1, 0] == 0
    assert raw[1, 1] == UINT16_FILLVAL


def test_composite_max_scale_offset(tmp_path):
    """composite_max must embed GDAL band scale/offset in the output COG."""
    # Write two single-radar COGs with _write_dat_tif
    da1 = _make_cog_da(np.array([[100, 200], [300, np.nan]], dtype='float64'))
    da2 = _make_cog_da(np.array([[150, 100], [np.nan, np.nan]], dtype='float64'))
    tif1 = str(tmp_path / 'radar1.tif')
    tif2 = str(tmp_path / 'radar2.tif')
    _write_dat_tif(da1, tif1, blocksize=128)
    _write_dat_tif(da2, tif2, blocksize=128)
    # Composite
    out = str(tmp_path / 'composite.tif')
    composite_max([tif1, tif2], out)
    with rasterio.open(out) as ds:
        assert ds.scales == (LWE_SCALE_FACTOR,)
        assert ds.offsets == (0.0,)
        assert ds.dtypes == ('uint16',)
        raw = ds.read(1)
    # pixel-wise max: [max(10000,15000), max(20000,10000)] = [15000, 20000]
    assert raw[0, 0] == 15000
    assert raw[0, 1] == 20000
    assert raw[1, 0] == 30000

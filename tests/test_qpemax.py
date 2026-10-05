import datetime
from glob import glob
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
    read_odim_h5, save_precip_grid, sweep_start_datetime,
)
from qpemax.output import (
    _write_dat_attrs, _write_dat_tif, _write_dattime_attrs, max_tif_glob, max_tif_name,
)
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


@pytest.mark.parametrize('nod, product, win, resolution, dbz_field, expected', [
    ('fikor', 'max', '1h', 250, 'DBZH',
     '202610050000_fikor_max_1h_acrr_finrad250_raw.tif'),
    ('fikor', 'maxtime', '1 D', 250, 'DBZHC',
     '202610050000_fikor_maxtime_24h_acrr_finrad250_rawac.tif'),
    ('composite', 'max', '1H', 250, 'DBZH',
     '202610050000_composite_max_1h_acrr_finrad250_raw.tif'),
    ('composite', 'maxtime', '1D', 500.0, 'DBZHC',
     '202610050000_composite_maxtime_24h_acrr_finrad500_rawac.tif'),
    ('fivih', 'max', '24h', 500, 'DBZH',
     '202610050000_fivih_max_24h_acrr_finrad500_raw.tif'),
])
def test_max_tif_name(nod, product, win, resolution, dbz_field, expected):
    for date in (datetime.date(2026, 10, 4), datetime.datetime(2026, 10, 4)):
        name = max_tif_name(date, nod, product, win, resolution, dbz_field)
        assert name == expected
        assert len(name.removesuffix('.tif').split('_')) == 7


@pytest.mark.parametrize('date, ts', [
    (datetime.date(2026, 1, 31), '202602010000'),
    (datetime.date(2025, 12, 31), '202601010000'),
    (datetime.date(2028, 2, 28), '202802290000'),
])
def test_max_tif_name_end_of_day(date, ts):
    assert max_tif_name(date, 'fikor', 'max', '1h', 250).startswith(ts + '_')


@pytest.mark.parametrize('kws', [
    dict(win='90min'), dict(win='0h'), dict(nod='fi_kor'), dict(nod='fi.kor'),
    dict(nod=''), dict(product='max_x'),
])
def test_max_tif_name_invalid(kws):
    args = dict(nod='fikor', product='max', win='1h') | kws
    with pytest.raises(ValueError):
        max_tif_name(datetime.date(2026, 10, 4), resolution=250, **args)


def test_max_tif_glob(tmp_path):
    date = datetime.date(2026, 10, 4)
    names = {
        (nod, product, win, dbz): max_tif_name(date, nod, product, win, 250, dbz)
        for nod in ('fikor', 'fivih', 'composite')
        for product in ('max', 'maxtime')
        for win in ('1h', '24h')
        for dbz in ('DBZH', 'DBZHC')
    }
    for name in names.values():
        (tmp_path / name).touch()
    pattern = max_tif_glob(date, '1 D', 250, 'DBZH')
    assert pattern == '202610050000_*_max_24h_acrr_finrad250_raw.tif'
    found = {
        Path(p).name for p in glob(str(tmp_path / pattern))
        if '_composite_' not in p
    }
    assert found == {names[(nod, 'max', '24h', 'DBZH')] for nod in ('fikor', 'fivih')}


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


# --- Grid geometry ---

FINRAD_BOUNDS = (-208000, 6390000, 1072000, 7926000)


def _radar_xy(radar):
    from pyproj import Transformer
    return Transformer.from_crs('WGS84', 3067, always_xy=True).transform(
        radar.longitude['data'][0], radar.latitude['data'][0])


def test_create_grid_odd_size(radar):
    with pytest.raises(ValueError):
        create_grid(radar, size=15, resolution=5000)


def test_site_tif_geometry(radar, tmp_path):
    """Per-site raster is north-up, exactly `resolution` and on the lattice."""
    size, res = 16, 5000
    grid = create_grid(radar, size=size, resolution=res)
    tif = str(tmp_path / 'site.tif')
    save_precip_grid(radar, str(tmp_path / 'site.nc'), tiffile=tif,
                     size=size, resolution=res, p_chunksize=size)
    with rasterio.open(tif) as ds:
        t = ds.transform
        assert ds.shape == (size, size)
    assert (t.a, t.e) == (res, -res)
    assert t.b == t.d == 0
    assert t.c % res == 0 and t.f % res == 0
    rx, ry = _radar_xy(radar)
    col, row = ~t * (rx, ry)
    assert int(col) in (size//2 - 1, size//2)
    assert int(row) in (size//2 - 1, size//2)
    # netCDF cache is north-up as well
    rda = _grid_to_dataset(radar, grid)
    assert rda.rio.transform() == t
    assert np.all(np.diff(rda.y.values) == -res)
    assert np.all(np.diff(rda.x.values) == res)


def _write_lattice_tif(path, data_mm, x0, y1, res=250):
    """Write a north-up single-site max COG with top-left corner (x0, y1)."""
    ny, nx = data_mm.shape
    da = xr.DataArray(
        data_mm, dims=['y', 'x'],
        coords={'x': x0 + res*(np.arange(nx) + 0.5),
                'y': y1 - res*(np.arange(ny) + 0.5)})
    da = da.rio.set_spatial_dims('x', 'y').rio.write_crs(3067)
    da.rio.write_nodata(np.nan, inplace=True)
    _write_dat_tif(da, str(path), blocksize=512)


def test_composite_finrad_no_resampling(tmp_path):
    """Aligned 2048 px / 250 m inputs composite onto finrad without resampling."""
    res, size = 250, 2048
    rng = np.random.default_rng(0)
    # top-left corners on the lattice, overlapping inputs
    corners = [(-100000, 7500000), (150000, 7300000)]
    paths, raws = [], []
    for i, (x0, y1) in enumerate(corners):
        data = rng.integers(0, 50000, (size, size)) / 100
        data[:10, :10] = np.nan
        p = tmp_path / f'site{i}.tif'
        _write_lattice_tif(p, data, x0, y1, res)
        with rasterio.open(p) as ds:
            raws.append(ds.read(1))
        paths.append(p)
    out = tmp_path / 'composite.tif'
    composite_max(paths, out, bounds=FINRAD_BOUNDS)
    with rasterio.open(out) as ds:
        assert ds.shape == (6144, 5120)
        t = ds.transform
        result = ds.read(1)
    assert (t.c, t.f) == (-208000, 7926000)
    assert (t.a, t.e) == (res, -res)
    expected = np.zeros((6144, 5120), dtype=np.uint16)
    covered = np.zeros_like(expected, dtype=bool)
    xmin, _, _, ymax = FINRAD_BOUNDS
    for raw, (x0, y1) in zip(raws, corners):
        r0, c0 = (ymax - y1)//res, (x0 - xmin)//res
        win = np.s_[r0:r0 + size, c0:c0 + size]
        valid = raw != UINT16_FILLVAL
        expected[win] = np.where(valid, np.maximum(expected[win], raw),
                                 expected[win])
        covered[win] |= valid
    expected[~covered] = UINT16_FILLVAL
    np.testing.assert_array_equal(result, expected)


def test_composite_finest_resolution(tmp_path):
    """Without `resolution`, the finest input resolution is used."""
    coarse, fine = tmp_path / 'coarse.tif', tmp_path / 'fine.tif'
    _write_lattice_tif(coarse, np.ones((4, 4)), 0, 2000, res=500)
    _write_lattice_tif(fine, np.ones((4, 4)), 0, 1000, res=250)
    out = tmp_path / 'composite.tif'
    composite_max([coarse, fine], out)
    with rasterio.open(out) as ds:
        assert ds.res == (250, 250)
        assert ds.shape == (8, 8)

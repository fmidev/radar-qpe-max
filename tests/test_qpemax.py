import datetime

import pyart
import numpy as np

from qpemax import basic_gatefilter, ZH, tstep_from_fpaths


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
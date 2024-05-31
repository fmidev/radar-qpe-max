import pyart
import numpy as np

from maksitiirain import basic_gatefilter, ZH


def test_basic_gatefilter():
    radar = pyart.testing.make_target_radar()
    # rename the "reflectivity" field to ZH
    radar.add_field(ZH, radar.fields.pop('reflectivity'))
    gatefilter = basic_gatefilter(radar)
    assert isinstance(gatefilter, pyart.filters.GateFilter)
    assert np.all(gatefilter.gate_excluded == 0)
    assert np.all(gatefilter.gate_included == 1)
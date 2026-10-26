import numpy as np
import pytest

from .v1_clock_joint_activity import physical_windows


def test_native_clock_conversion_preserves_physical_bins_and_silent_units():
    state=dict(source_id=np.array([10,20,30]))
    labels=np.array(['E','E','I'])
    units=np.array([0,0,1,1,0,0])
    a=physical_windows(state,dict(unit_index=units,time_ms=np.array([.25,500.25,501.,501.75,2999.75,3000.])),labels,.25)
    b=physical_windows(state,dict(unit_index=units,time_ms=np.array([.125,500.125,501.,501.875,2999.875,3000.])),labels,.125)
    for name in a:
        left,x=a[name];right,y=b[name]
        assert left['bins']==right['bins'] and left['units']==right['units']==3
        for key in x:np.testing.assert_array_equal(x[key],y[key])
    assert a['full'][0]['events']==5 and a['pre_command'][0]['events']==2 and a['command_response'][0]['events']==3
    assert a['full'][1]['unit_totals'].tolist()==[3,2,0]


def test_fractional_native_time_and_unmatched_unit_map_are_rejected():
    state=dict(source_id=np.array([10,20]));events=dict(unit_index=np.array([0]),time_ms=np.array([.125]))
    with pytest.raises(ValueError,match='NATIVE_CLOCK'):
        physical_windows(state,events,np.array(['E','I']),.25)
    with pytest.raises(ValueError,match='UNIT_MAP'):
        physical_windows(state,events,np.array(['E']),.125)

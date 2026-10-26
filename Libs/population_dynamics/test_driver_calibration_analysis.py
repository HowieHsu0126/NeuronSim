import pytest

from Libs.population_dynamics.driver_calibration_analysis import source_support


def test_native_source_onset_is_first_update_plus_physical_delay():
    r=dict(backend_delay_ticks=17,dt_ms=.1,configuration={"duration_ms":2000.},expected_arrival_support=[15,19999])
    assert source_support(r)==(16,19999)
    assert r["expected_arrival_support"]==[15,19999]
    r["expected_arrival_support"]=[16,19999]
    assert source_support(r)==(16,19999)
    r["expected_arrival_support"]=[14,19999]
    with pytest.raises(ValueError,match="incompatible"):
        source_support(r)

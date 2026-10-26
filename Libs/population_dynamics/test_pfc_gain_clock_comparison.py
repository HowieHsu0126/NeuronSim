import numpy as np
import pytest
from pfc_gain_clock_comparison import normalize_response,gain_rows,CLOCKS,CONDITIONS,PHASES,METRICS,BOUNDS,SEEDS


def source():
    z=dict(source_rng_seeds=np.array(SEEDS),condition_names=np.array(CONDITIONS),phase_names=np.array(PHASES),
        metric_names=np.array(METRICS),phase_start_ms=BOUNDS[:,0],phase_stop_ms=BOUNDS[:,1],units=np.array("Hz"))
    for i,k in enumerate(CLOCKS):z[k+"_values"]=np.arange(3*32*5*8).reshape(3,32,5,8)+i
    return z


def test_named_axes_reorder_without_changing_values():
    z=source();expected=normalize_response(z)
    z["condition_names"]=z["condition_names"][[2,0,1]]
    for k in CLOCKS:z[k+"_values"]=z[k+"_values"][[2,0,1]]
    np.testing.assert_array_equal(normalize_response(z),expected)


@pytest.mark.parametrize("change",["seed","phase_time","units","nonfinite"])
def test_invalid_semantics_rejected(change):
    z=source()
    if change=="seed":z["source_rng_seeds"][0]=95002
    if change=="phase_time":z["phase_stop_ms"]=np.array([101,300,500,2000,5000])
    if change=="units":z["units"]=np.array("kHz")
    if change=="nonfinite":z[CLOCKS[0]+"_values"]=np.full((3,32,5,8),np.nan)
    with pytest.raises(ValueError):normalize_response(z)


def test_paired_gain_and_clock_difference_direction():
    before=np.zeros((32,3,3,5,8));after=before.copy();after[:,0,1]=3;after[:,1,1]=1;after[:,2,1]=2
    rows,interactions=gain_rows(before,after)
    assert len(rows)==len(interactions)==23040
    selected=[r for r in interactions if r["source_rng_seed"]==95001 and r["response"]=="cue_minus_omission" and
        r["phase"]=="late_delay" and r["metric"]=="cue_selectivity_hz" and r["left_clock"]==CLOCKS[0] and r["right_clock"]==CLOCKS[1]]
    assert selected[0]["gain_by_clock_difference"]==2

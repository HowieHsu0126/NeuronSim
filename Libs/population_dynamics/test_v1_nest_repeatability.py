from .v1_nest_repeatability import compare_input_plans
import numpy as np
import pytest

from .v1_nest_repeatability import compare_activity


def activity():
    return dict(spike_source_id=np.array([0,1,0,0]),spike_time_ms=np.array([.25,.25,.5,.5]),
                spike_counts=np.array([3,1,0]),final_vm_mV=np.array([-60.,-61.,-70.]))


def test_global_recorder_permutation_retains_per_unit_time_and_multiplicity():
    a=activity();b={k:v.copy() for k,v in a.items()}
    for key in ['spike_source_id','spike_time_ms']:b[key]=b[key][[3,1,0,2]]
    result=compare_activity(a,b,3,1.)
    assert all(result['checks'].values()) and not result['global_recorder_row_order_equal']


@pytest.mark.parametrize('change',['one_ulp_voltage','event_time','reassigned_event'])
def test_source_unit_or_float_changes_are_not_hidden_by_equal_totals(change):
    a=activity();b={k:v.copy() for k,v in a.items()}
    if change=='one_ulp_voltage':b['final_vm_mV'][0]=np.nextafter(b['final_vm_mV'][0],np.inf)
    elif change=='event_time':b['spike_time_ms'][0]=np.nextafter(.25,np.inf)
    else:
        b['spike_source_id'][0]=1;b['spike_counts']=np.array([2,2,0])
    assert not all(compare_activity(a,b,3,1.)['checks'].values())


def test_missing_duplicate_with_stale_counts_is_invalid():
    a=activity();b={k:v.copy() for k,v in a.items()}
    for key in ['spike_source_id','spike_time_ms']:b[key]=b[key][:-1]
    with pytest.raises(ValueError,match='IDENTITIES_OR_COUNTS'):
        compare_activity(a,b,3,1.)


def test_generated_input_timestamp_change_is_visible_when_counts_match():
    first = dict(population=np.array([0, 1]), source_id=np.array([0, 0]), source_counts=np.array([1, 1]),
                 source_event_unit=np.array([0, 1]), source_event_time_ms=np.array([10.0, 20.0]))
    second = {key: value.copy() for key, value in first.items()}
    second['source_event_time_ms'][0] = 10.125
    result = compare_input_plans(first, second)
    assert result['source_counts'] and not result['source_event_time_ms']

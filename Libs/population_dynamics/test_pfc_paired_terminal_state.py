"""Stored-state equality, future refractory state and tiny private residuals."""
import copy
import subprocess
import sys

import numpy as np
import pytest
from .pfc_paired_terminal_state import (
    difference_summary,remaining_refractory,compare_terminal_states,validate_terminal,
    _STATE_FIELDS,_IDENTITY_FIELDS,
)


def fixture():
    ne,ni=8,2
    config=dict(excitatory_neurons=ne,inhibitory_neurons=ni,dt_ms=.05,duration_ms=5000.,weight_scaling_factor=2.)
    state={}
    for population,n,offset in [('E',ne,0),('I',ni,ne)]:
        for name in _STATE_FIELDS[population]:
            state[population+'__'+name]=np.zeros(n,dtype=bool if name=='not_refractory' else np.float64)
        state[population+'__v'][:]=-.06
        state[population+'__lastspike'][:]=-10000.
        state[population+'__not_refractory'][:]=True
        state[population+'__source_indices']=np.arange(n,dtype=np.int32)
        state[population+'__archive_indices']=np.arange(n,dtype=np.int32)+offset
        state[population+'__spike_counts']=np.zeros(n,dtype=np.int32)
    return state,config


def test_history_counters_and_expired_spike_times_are_not_future_refractory_state():
    a,config=fixture();b=copy.deepcopy(a)
    b['E__lastspike'][:]=0.;b['E__spike_counts'][:]=1
    rows,summary=compare_terminal_states(a,b,config)
    assert summary['dynamic_state_exactly_equal'] and not summary['cumulative_spike_counters_equal']
    assert next(row for row in rows if row['population']=='E' and row['variable']=='lastspike')['different_values']==8
    np.testing.assert_array_equal(remaining_refractory(b,'E',config),np.zeros(8,dtype=np.int64))


def test_recent_spike_has_remaining_refractory_ticks_at_next_tick():
    a,config=fixture();b=copy.deepcopy(a)
    b['E__lastspike'][0]=5.-.00005
    remaining=remaining_refractory(b,'E',config)
    assert remaining[0]==39 and np.all(remaining[1:]==0)
    assert not compare_terminal_states(a,b,config)[1]['dynamic_state_exactly_equal']


def test_identical_projections_do_not_erase_private_nmda_residual():
    a,config=fixture();a['E__s_NMDA'][1:]=1.;b=copy.deepcopy(a)
    b['E__s_NMDA'][0]=1e-30
    _,summary=compare_terminal_states(a,b,config)
    assert summary['voltage_exactly_equal'] and summary['next_nmda_inputs_exactly_equal']
    assert not summary['private_nmda_state_exactly_equal'] and not summary['dynamic_state_exactly_equal']


def test_rms_preserves_subnormal_squared_residual_without_underflow():
    result=difference_summary(np.array([0.,0.]),np.array([1e-250,0.]))
    assert result['different_values']==1 and not result['exact_equal']
    assert result['root_mean_square_difference']==pytest.approx(1e-250/np.sqrt(2),rel=1e-14,abs=0)


def test_voltage_difference_stays_in_volts():
    result=difference_summary(np.array([-.060]),np.array([-.059]))
    assert result['maximum_absolute_difference']==pytest.approx(.001)


def test_full_population_identity_and_schema_are_required():
    state,_=fixture();state['I__archive_indices'][0]=0
    with pytest.raises(ValueError,match='TERMINAL_UNIT_MAPPING_CHANGED'):validate_terminal(state,8,2)
    state,_=fixture();del state['E__x']
    with pytest.raises(ValueError,match='COMPLETE_TERMINAL_STATE_SCHEMA_REQUIRED'):validate_terminal(state,8,2)


def test_real_cli_rejects_wrong_scope_without_outputs(tmp_path):
    config=tmp_path/'wrong.yaml';config.write_text('schema: wrong\n')
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_paired_terminal_state',
        '--config',str(config),'--project-root',str(tmp_path),'--output',str(tmp_path/'output'),
        '--scratch-parent',str(tmp_path/'scratch')],capture_output=True,text=True)
    assert result.returncode!=0 and 'BOUNDED_CALIBRATION_TERMINAL_DIAGNOSTIC_REQUIRED' in result.stderr
    assert not (tmp_path/'output').exists()

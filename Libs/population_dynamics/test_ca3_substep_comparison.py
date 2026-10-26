"""Actual AER/observer formats on bounded synthetic clocks; no native claims."""
import csv
import json

import numpy as np
import pytest

from . import ca3_substep_comparison as comparison
from . import ca3_native_observation as observer
from .test_ca3_cue_contrast import make_run


def numeric_runs(tmp_path, *, current_changed=False, counter_changed=False, stop_ms=1001, include_forty=False):
    groups = [observer.Group(0, 0, 1650, 0, 0, 0, 1650, 1, 'CA3_Pyramidal')]
    for index, name in enumerate(('CA3_QuadD_LM','CA3_Axo_Axonic','CA3_Basket','CA3_BC_CCK',
                                   'CA3_Bistratified','CA3_Ivy','CA3_MFA_ORDEN'), 1):
        start = 1650 + 2*(index-1)
        groups.append(observer.Group(index, start, start+2, 0, index, start, start+2, 1, name))
    groups = tuple(groups)
    currents = np.zeros((1001,1664), dtype='<f4')
    events = {
        5: np.asarray([[0,1],[519,276],[519,276],[520,550],[1000,1663]], dtype=np.int64),
        10: np.asarray([[0,1],[519,276],[519,276],[521,550],[1000,1663]], dtype=np.int64),
        20: np.asarray([[0,1],[519,276],[521,550]], dtype=np.int64),
    }
    events[40] = np.asarray([[0,1],[519,276],[521,550],[521,550]], dtype=np.int64)
    runs = {}
    counts = (5,10,20,40) if include_forty else (5,10,20)
    for count in counts:
        actual = currents.copy()
        if current_changed and count == counts[-1]:
            actual[-1,-1] = np.float32(-0.0)  # numerically equal, bitwise different
        trace_events = events[count].copy()
        if counter_changed and count == counts[-1]:
            trace_events[-1,0] += 1
        runs[count] = make_run(tmp_path/str(count), groups, events[count], actual,
                              trace_events=trace_events, stop_ms=stop_ms if count == counts[-1] else 1001)
        runs[count].native.update(integration='RUNGE_KUTTA4_'+str(count), native_elapsed_seconds=count,
                                  child_peak_rss_bytes=count*1024)
    return runs


def test_complete_numeric_cli_preserves_units_clocks_counts_and_nonzero_differences(tmp_path, monkeypatch):
    runs = numeric_runs(tmp_path)
    monkeypatch.setattr(comparison, 'bind_runs', lambda *args, **kwargs: (runs, {'synthetic_binding': True}))
    output = tmp_path/'result'
    monkeypatch.setattr('sys.argv', ['compare','--reference','fixture','--ten','fixture','--twenty','fixture',
        '--source-analysis','fixture','--source-rng','fixture','--output',str(output)])
    comparison.main()
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'MEASURED_CA3_NEURON_SUBSTEP_SENSITIVITY'
    assert report['input_comparison']['current_values_per_run'] == 1664*1001
    assert report['phase_rows'] == 3*7*14
    assert report['scientific_fidelity_qualified'] is False
    assert report['numerical_precision_qualified'] is False
    assert report['quantitative_tolerances_applied'] is False
    assert report['whole_schedule_differences'][0]['changed_unit_totals'] == 0
    assert report['whole_schedule_differences'][0]['population_1ms_count_rmse'] > 0
    assert report['whole_schedule_differences'][1]['changed_unit_totals'] == 2
    assert report['whole_schedule_differences'][1]['all_unit_spike_count_mae'] == pytest.approx(2/1664)
    assert all(row['unit_millisecond_counts_compared'] == 1664*1001 for row in report['counter_checks'].values())
    with np.load(output/'counts.npz', allow_pickle=False) as saved:
        assert saved['unit_totals'].shape == (3,1664)
        assert saved['field_counts'].shape == (3,1001,14)
        assert list(saved['unit_totals'].sum(axis=1)) == [5,5,3]
    with (output/'phase_differences.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    last = next(r for r in rows if r['coarse_substeps']=='10' and r['fine_substeps']=='20'
                and r['field']=='CA3_MFA_ORDEN' and r['phase']=='background_after_cue_3')
    assert last['fine_minus_coarse_count'] == '-1'
    assert float(last['fine_minus_coarse_rate_hz']) == -500


@pytest.mark.parametrize('option,error', [('current_changed','INPUT_BITS_DIFFER'),
    ('counter_changed','PER_UNIT_MILLISECOND_COUNTER_MISMATCH'), ('stop_ms','DURATION_OR_FOOTER_INVALID')])
def test_input_counter_or_clock_failure_stays_failed(tmp_path, monkeypatch, option, error):
    runs = numeric_runs(tmp_path, **{option:1000 if option=='stop_ms' else True})
    monkeypatch.setattr(comparison, 'bind_runs', lambda *args, **kwargs: (runs, {}))
    with pytest.raises(ValueError, match=error):
        comparison.analyze('fixture','fixture','fixture','fixture','fixture',tmp_path/'result')
    report = json.loads((tmp_path/'result/report.json').read_text())
    assert report['status']=='FAILED' and report['numerical_precision_qualified'] is False


def test_reusing_output_cannot_overwrite_evidence(tmp_path):
    output=tmp_path/'result'; output.mkdir(); (output/'report.json').write_text('original')
    with pytest.raises(FileExistsError):
        comparison.analyze('fixture','fixture','fixture','fixture','fixture',output)
    assert (output/'report.json').read_text()=='original'


def test_four_setting_cli_keeps_original_pairs_and_all_outputs(tmp_path, monkeypatch):
    runs = numeric_runs(tmp_path, include_forty=True)
    requests = []
    def bind(*args, **kwargs):
        requests.append(kwargs['forty'])
        return runs, {'synthetic_binding': True}
    monkeypatch.setattr(comparison, 'bind_runs', bind)
    output = tmp_path/'result'
    monkeypatch.setattr('sys.argv', ['compare','--reference','fixture','--ten','fixture','--twenty','fixture',
        '--forty','forty_fixture','--source-analysis','fixture','--source-rng','fixture','--output',str(output)])
    comparison.main()
    assert str(requests[0]) == 'forty_fixture'
    report = json.loads((output/'report.json').read_text())
    assert report['input_comparison']['runs_compared'] == 4
    assert report['input_comparison']['current_values_per_run'] == 1664*1001
    assert report['phase_rows'] == 6*7*14
    assert list(report['counter_checks']) == ['5','10','20','40']
    assert [(r['coarse_substeps'],r['fine_substeps']) for r in report['whole_schedule_differences']] == [
        (5,10),(10,20),(5,20),(20,40),(10,40),(5,40)]
    assert report['whole_schedule_differences'][3]['changed_unit_totals'] == 1
    assert report['whole_schedule_differences'][3]['all_unit_spike_count_mae'] == pytest.approx(1/1664)
    assert report['numerical_precision_qualified'] is False
    assert report['whole_simulator_convergence_qualified'] is False
    assert report['input_origins']['forty'] == 'forty_fixture'
    with np.load(output/'counts.npz', allow_pickle=False) as saved:
        assert saved['substeps'].tolist() == [5,10,20,40]
        assert saved['unit_totals'].shape == (4,1664)
        assert saved['field_counts'].shape == (4,1001,14)
        assert saved['unit_totals'].sum(axis=1).tolist() == [5,5,3,4]


@pytest.mark.parametrize('option,error', [('current_changed','INPUT_BITS_DIFFER'),
    ('counter_changed','PER_UNIT_MILLISECOND_COUNTER_MISMATCH'), ('stop_ms','DURATION_OR_FOOTER_INVALID')])
def test_optional_forty_is_checked_for_inputs_counters_and_complete_clock(tmp_path, monkeypatch, option, error):
    runs = numeric_runs(tmp_path, include_forty=True, **{option:1000 if option=='stop_ms' else True})
    monkeypatch.setattr(comparison, 'bind_runs', lambda *args, **kwargs: (runs, {}))
    with pytest.raises(ValueError, match=error):
        comparison.analyze('fixture','fixture','fixture','fixture','fixture',tmp_path/'result',forty='fixture')
    assert json.loads((tmp_path/'result/report.json').read_text())['status'] == 'FAILED'

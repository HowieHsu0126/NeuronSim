"""Identity, raw-input and endpoint contracts for the archived clock contrast."""
import json
from pathlib import Path

import numpy as np
import pytest

from . import v1_clock_comparison as clock
from .provenance import sha256_file


def _state():
    return dict(source_id=np.array([9, 1, 5]), inverse_source_id=np.array([9, 1, 5]),
        native_id=np.array([31, 22, 13]), node_type_id=np.array([200, 100, 200]),
        initial_vm_mV=np.array([-65., -60., -65.]), terminal_vm_mV=np.array([-64., -59., -65.]),
        spike_count=np.array([2, 1, 0]))


def test_all_unit_alignment_population_mapping_and_terminal_event_inclusion(tmp_path):
    values = _state()
    state = clock.align_state(values)
    assert state['source_id'].tolist() == [1, 5, 9]
    assert state['spike_count'].tolist() == [1, 0, 2]
    spikes = clock.align_spikes(dict(source_id=np.array([9, 1, 9]), native_id=np.array([31, 22, 31]),
        time_ms=np.array([.125, 0., 3000.])), state, .125)
    types = tmp_path/'types.csv'
    types.write_text('node_type_id pop_name population\n100 excitatory v1\n200 inhibitory v1\n300 empty v1\n')
    names, codes = clock.source_populations(types, state['node_type_id'])
    assert names.tolist() == ['excitatory', 'inhibitory', 'empty']
    assert codes.tolist() == [0, 1, 1]  # Includes the silent inhibitory unit.
    grid, curves = clock.cumulative_counts(spikes, codes, len(names))
    assert grid.shape == (12001,) and grid[-1] == 3000.
    assert curves[:, 0].tolist() == [1, 0, 0]
    assert curves[:, 1].tolist() == [1, 1, 0]
    assert curves[:, -2].tolist() == [1, 1, 0]
    assert curves[:, -1].tolist() == [1, 2, 0]
    summary = clock.summarize_units(state, state, codes == 1)
    assert summary['units'] == 2 and summary['reference_silent_units'] == 1
    assert summary['spike_counts']['difference']['maximum_absolute'] == 0.


def test_duplicate_and_unknown_source_ids_fail_without_dropping_units():
    values = _state(); values['source_id'][0] = 1
    with pytest.raises(ValueError, match='DUPLICATE_IDENTITY'):
        clock.align_state(values)
    state = clock.align_state(_state())
    with pytest.raises(ValueError, match='SPIKE_SOURCE_IDENTITY_DIFFERS'):
        clock.align_spikes(dict(source_id=np.array([999]), native_id=np.array([31]), time_ms=np.array([1.])), state, .25)


def _inputs(dt, packed, order=(0, 1, 2), changed=False):
    order = np.asarray(order)
    inverse_order = np.argsort(order)
    ids, populations = np.array([2, 7, 0]), np.array(['lgn', 'lgn', 'bkg'])
    values = dict(source_id=ids[order], inverse_source_id=ids[order], population=populations[order],
                  native_id=np.array([11, 12, 13])[order], precise_times=np.ones(3, dtype=bool))
    if packed:
        requested = np.array([.125, .125, 1.])
        values.update(source_event_counts=np.array([2, 0, 1])[order],
                      source_event_unit=inverse_order[np.array([0, 0, 2])])
    else:
        requested = np.array([[.125, .5], [.25, 1.], [.75, 2.]])[order]
        values['source_event_counts'] = np.full(3, 2)
    if changed:
        requested = requested.copy(); requested.reshape(-1)[0] += .125
    steps = np.ceil(requested / dt).astype(np.int64)
    offsets = .001 * (steps * int(1000 * dt)) - requested
    values.update(requested_times_ms=requested, native_times_ms=requested.copy(), encoded_steps=steps, encoded_offsets_ms=offsets)
    return values


@pytest.mark.parametrize('packed', [False, True])
def test_requested_input_comparison_joins_generator_identity_and_preserves_silence(packed):
    schema = 'packed_source_events_v1' if packed else 'diagnostic_pulse_matrix_v1'
    reference = clock.canonical_inputs(_inputs(.25, packed), schema, .25)
    fine = clock.canonical_inputs(_inputs(.125, packed, (2, 0, 1)), schema, .125)
    clock.require_same_requested_inputs(reference, fine)
    assert fine['population'].tolist() == ['bkg', 'lgn', 'lgn']
    assert fine['source_id'].tolist() == [0, 2, 7]
    if packed:
        assert fine['counts'].tolist() == [1, 2, 0]
        assert fine['requested_times_ms'].tolist() == [1., .125, .125]
    changed = clock.canonical_inputs(_inputs(.125, packed, changed=True), schema, .125)
    with pytest.raises(ValueError, match='REQUESTED_INPUTS_DIFFER'):
        clock.require_same_requested_inputs(reference, changed)


def test_native_time_readback_cannot_be_nonfinite_or_inconsistent():
    values = _inputs(.125, True); values['native_times_ms'][0] = np.nan
    with pytest.raises(ValueError, match='INPUT_TIME_DOMAIN_DIFFERS'):
        clock.canonical_inputs(values, 'packed_source_events_v1', .125)
    values = _inputs(.125, True); values['encoded_steps'][0] += 1
    with pytest.raises(ValueError, match='INPUT_DECODING_DIFFERS'):
        clock.canonical_inputs(values, 'packed_source_events_v1', .125)


def _archive(path: Path, *, fine=False, reference=None, defect=None):
    path.mkdir()
    csv_path = path/clock.SOURCE_TYPE_CSV; csv_path.parent.mkdir(parents=True)
    csv_path.write_text('node_type_id pop_name population\n100 excitatory v1\n200 inhibitory v1\n')
    table = dict(bytes=csv_path.stat().st_size, sha256=sha256_file(csv_path))
    job = 'test-fine' if fine else 'test-reference'
    protocol = path/'clock_protocol.resolved.yaml'; protocol.write_text('execution_resolution_ms: 0.125\n')
    for phase, pid in [('primary', 1), ('replay', 1 if defect == 'process' else 2)]:
        directory = path/phase/'native'; directory.mkdir(parents=True)
        for name in clock.NATIVE_ARRAYS:
            np.savez(path/phase/name, fixture=np.array([1]))
        (directory/'parameters.json').write_text('{}')
        report = dict(status=clock.FINE_STATUS if fine else 'POINTNET_NATIVE_DIAGNOSTIC_COMPLETE', exit_code=0,
            input_mode='diagnostic_pulses', input_event_schema='diagnostic_pulse_matrix_v1', mode='canary',
            native_nest_simulation=True, native_parameters_and_initial_state_exact=True,
            scientific_qualification=False, production_episodes=0, simulated_ms=3000.,
            runtime=dict(slurm=dict(SLURM_JOB_ID=job)), hostname='fixture', process_id=pid,
            input_readback=dict(tics_per_ms=1000.), array_artifacts=list(clock.NATIVE_ARRAYS),
            bindings=dict(point_config=dict(source_clock=dict(duration_ms=3000., resolution_ms=.25)),
                          recurrent=dict(files={'primary/glif/v1_node_types.csv': table})),
            connections=dict(full_native_tuple_readback=False, source_batch_sha256='a'*64),
            source_denominators={}, original_config_sha256='source-config', cortical_units=3, realized_glif_types=2,
            kernel_rng_seed=1, nest_version='3.10.0', local_num_threads=1, wheel=dict(sha256='wheel'))
        if fine:
            report.update(execution_clock=dict(duration_ms=3000., resolution_ms=.125, tics_per_ms=1000., tics_per_step=125),
                clock_protocol=dict(sha256=sha256_file(protocol)))
            if reference is not None:
                report['temporal_sensitivity'] = dict(reference_primary_report_sha256=reference['consumed']['primary/report.json']['sha256'],
                    reference_delivery_sha256=reference['consumed']['delivery.json']['sha256'],
                    reference_job_id=reference['delivery']['job_id'], unchanged_upstream_bindings=True,
                    reference_clock_source_batch_sha256='a'*64)
                report['connections'].update(reference_clock_source_batch_sha256='a'*64, reference_clock_source_batch_exact=True,
                    reference_clock_resolution_ms=.25, source_batch_resolution_ms=.125)
        (path/phase/'report.json').write_text(json.dumps(report))
    comparison = dict(status='POINTNET_NATIVE_REPLAY_EQUAL', exit_code=0, all_native_outputs_equal=True,
        independent_processes=True, binding_or_runtime_differences=[],
        left_report_sha256='wrong' if defect == 'comparison' else sha256_file(path/'primary/report.json'),
        right_report_sha256=sha256_file(path/'replay/report.json'),
        files=[dict(path=name, equal=True) for name in (*clock.NATIVE_ARRAYS, 'native/parameters.json')])
    (path/'comparison').mkdir(); (path/'comparison/report.json').write_text(json.dumps(comparison))
    files = [dict(path=name, bytes=(path/name).stat().st_size, sha256=sha256_file(path/name))
             for name in clock.required_files(report) if name != 'delivery.json']
    (path/'delivery.json').write_text(json.dumps(dict(exit_code=0, job_id=job, mode='canary', files=files,
        status='PASS_POINTNET_NATIVE_TEMPORAL_SENSITIVITY_DEVELOPMENT' if fine else 'PASS_POINTNET_NATIVE_DIAGNOSTIC')))
    return path


@pytest.mark.parametrize('defect', ['comparison', 'process', 'changed_array'])
def test_incomplete_replay_or_tampered_delivered_input_is_rejected(tmp_path, defect):
    path = _archive(tmp_path/'reference', defect=defect)
    if defect == 'changed_array':
        with (path/'primary/native/input_events.npz').open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError, match='INDEPENDENT_REPLAY_REQUIRED|DELIVERED_FILE_CHANGED'):
        clock.admit_archive(path, fine=False)


def test_pair_requires_consumed_reference_digest_and_source_bindings(tmp_path):
    reference = clock.admit_archive(_archive(tmp_path/'reference'), fine=False)
    fine = clock.admit_archive(_archive(tmp_path/'fine', fine=True, reference=reference), fine=True)
    clock.require_pair(reference, fine)
    fine['report']['connections']['reference_clock_source_batch_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='CONSUMED_REFERENCE_IDENTITY_DIFFERS'):
        clock.require_pair(reference, fine)
    fine['report']['connections']['reference_clock_source_batch_sha256'] = 'a'*64
    fine['report']['bindings']['source_identity'] = 'different'
    with pytest.raises(ValueError, match='SOURCE_CONTEXT_DIFFERS'):
        clock.require_pair(reference, fine)


def test_benchmark_windows_preserve_source_command_and_terminal_boundaries():
    events = dict(time_ms=np.array([0., 500., 500.125, 500.875, 501., 2999.875, 3000.]),
                  unit_index=np.zeros(7, dtype=np.int64))
    counts = clock.benchmark_window_counts(events, 2)
    assert {key: value.tolist() for key, value in counts.items()} == dict(
        full=[7, 0], source_baseline=[2, 0], source_response=[5, 0],
        pre_command=[4, 0], command_response=[2, 0], terminal=[1, 0])
    np.testing.assert_array_equal(counts['full'], counts['source_baseline'] + counts['source_response'])
    np.testing.assert_array_equal(counts['full'], counts['pre_command'] + counts['command_response'] + counts['terminal'])


def test_benchmark_windows_keep_empty_event_population():
    counts = clock.benchmark_window_counts(dict(time_ms=np.array([], dtype=float),
        unit_index=np.array([], dtype=np.int64)), 3)
    assert all(value.tolist() == [0, 0, 0] for value in counts.values())

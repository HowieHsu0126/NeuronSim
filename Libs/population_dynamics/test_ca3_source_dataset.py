"""Synthetic source receipts test CA3 conversion; no native physiology is asserted."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from . import ca3_source_dataset as exporter
from . import ca3_control_trace as control
from . import ca3_native_observation_replay as reconciliation
from .ca3_retrieval_readouts import _bound_inputs
from .carlsim_records import spike_blocks
from .provenance import sha256_file
from .test_ca3_coupled_readouts import fixture, add_realization
from .test_ca3_native_observation_replay import refresh_bindings
from . import ca3_realization as realization_protocol
from .ca3_realization import SourceRealization, contract, program


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def delivery(root, status):
    files = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256_file(path))
             for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'delivery.json']
    write_json(root/'delivery.json', dict(status=status, exit_code=0, copied_files_verified=True,
                                         synthetic_fixtures_only=True, files=files))


@pytest.fixture
def source(tmp_path):
    primary = tmp_path/'primary'; fixture(primary)
    _, native, _, groups, _, pattern, hashes = _bound_inputs(primary, coupled_extension=True)
    delivery(primary, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    analysis = tmp_path/'analysis'; (analysis/'control').mkdir(parents=True)
    proof = dict(status='PASS_COUPLED_EXTENSION_CA3_OBSERVATION_REPLAY', units=168106, duration_ms=1001,
        event_multisets_equal=True, actual_current_trace_verified=True, native_internal_spike_counter_verified=True,
        terminal_native_clock_independently_verified=True, observation_source_execution_equivalence_qualified=True,
        original_report_sha256=hashes['native_report.json'], original_recording_sha256=hashes['observations/recording.json'],
        observed_report_sha256='1'*64, native_observation_sha256='2'*64,
        implementation_sha256=sha256_file(Path(reconciliation.__file__)),
        coupled_extension=native['coupled_extension'], coupled_extension_config=native['coupled_extension_config'])
    proof['coupled_extension_config'] = dict(native['coupled_extension_config'],
        path=str(analysis/'observed_build/coupled_extension.yaml'))
    write_json(analysis/'reconciliation.json', proof)
    pyr = next(group for group in groups if group.name == exporter.PYRAMIDAL)
    commands = control.source_commands(pattern, pyr.start, coupled_extension=True)
    np.savez(analysis/'control/cue_command_events.npz', **commands)
    control._write_csv(analysis/'control/background_schedule.csv', control.background_schedule(groups))
    report = dict(status='VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY',
        implementation_sha256=sha256_file(Path(control.__file__)),
        current_units=dict(model_specific_physical_unit='pA'), numeric_conversion_applied=False,
        input_sha256={'reconciliation':sha256_file(analysis/'reconciliation.json'),
                      'native_report.json':proof['observed_report_sha256'],
                      'native_observation.bin':proof['native_observation_sha256']},
        coupled_extension=native['coupled_extension'], coupled_extension_config=proof['coupled_extension_config'],
        artifacts={name:sha256_file(analysis/'control'/name) for name in ('cue_command_events.npz','background_schedule.csv')})
    write_json(analysis/'control/report.json', report)
    write_json(analysis/'analysis.json', dict(status='PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL', exit_code=0))
    delivery(analysis, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    return primary, analysis, native, groups, pattern, commands


def test_real_cli_roundtrips_all_units_fields_controls_and_ordered_provenance(source, tmp_path, capsys):
    primary, analysis, native, groups, pattern, commands = source
    output = tmp_path/'export'
    staged_primary, staged_analysis = tmp_path/'staged_primary', tmp_path/'staged_analysis'
    shutil.copytree(primary, staged_primary); shutil.copytree(analysis, staged_analysis)
    origins = tmp_path/'origins.json'
    write_json(origins, {role:dict(archive=str(root), delivery_sha256=sha256_file(root/'delivery.json'))
                         for role, root in [('primary', primary), ('observation_analysis', analysis)]})
    assert exporter.main(['--primary', str(staged_primary), '--observation-analysis', str(staged_analysis),
                          '--input-origins', str(origins), '--output', str(output)]) == 0
    capsys.readouterr()
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'EXPORTED_VERIFIED_CA3_DEVELOPMENT_DATASET'
    assert report['complete_episodes'] == 1 and report['production_episodes'] == 0
    assert report['source_episodes_added_to_campaign'] == 0 and report['new_source_simulations'] == 0
    assert report['verification']['all_unit_bin_values_checked'] == 168274106
    assert report['verification']['event_records'] == 35
    assert report['control_features'] == 838
    dataset = exporter.CircuitDataset(output/'dataset', splits=('development',))
    assert not dataset.catalog['source_fidelity_qualified']
    assert dataset.catalog['binding']['scope'] == 'source_reproduction'
    assert dataset.catalog['binding']['primary_archive'] == str(primary.resolve())
    assert dataset.catalog['binding']['observation_analysis_archive'] == str(analysis.resolve())
    assert len(dataset.episodes) == len(dataset.catalog['circuits']) == 1
    row = dataset.episodes[report['episode_id']]; shard = dataset.shards[row['shard']]
    with h5py.File(dataset._verified(shard['path'], shard['sha256']), 'r') as archive:
        stored = archive[report['episode_id']]
        metadata = json.loads(stored.attrs['metadata_json'])
        assert metadata['seeds'] == {'source_carlsim_seed':10, 'source_host_srand_seed':3}
        assert not metadata['source_record']['observed_replay_counted_as_episode']
        for key, value in commands.items():
            np.testing.assert_array_equal(stored['privileged/raw_command_'+key][:], value)
        order = stored['privileged/source_event_ordinal'][:]
        restored_ticks = stored['spike_ticks'][:][np.argsort(order)]
        restored_ids = stored['spike_unit_indices'][:][np.argsort(order)]
        raw = [(block[:,0], block[:,1]+group.start) for group in groups
               for block in spike_blocks(primary/('native_run/results/spk_'+group.name+'.dat'), units=group.units, stop_ms=1001)]
        np.testing.assert_array_equal(restored_ticks, np.concatenate([item[0] for item in raw]))
        np.testing.assert_array_equal(restored_ids, np.concatenate([item[1] for item in raw]))
    # The common reader explicitly averages coarser bins. Only native1ms is
    # the lossless command view; a mean is not an additive impulse amplitude.
    coarse = dataset.window(report['episode_id'], boundary_tick=500, history_ticks=500, horizon_ticks=500,
                            bin_ticks=5, unit_start=0, unit_stop=4)
    assert np.any((coarse['known_controls'][:,:828] > 0) & (coarse['known_controls'][:,:828] < 1))
    assert 'privileged' not in coarse and 'raw_command_set_current' not in coarse


def test_source_identity_uses_global_header_offsets_and_keeps_silent_units(source):
    _, _, native, groups, _, _ = source
    units = exporter.source_units(native, groups)
    pyr = next(group for group in groups if group.name == exporter.PYRAMIDAL)
    assert pyr.start != 0
    assert len(units.unit_ids) == 168106 and units.coordinates is None
    assert len(set(units.populations)) == 8 and len(units.field_names) == 11
    for i in range(3):
        assert np.all(units.field_index[pyr.start+i*550:pyr.start+(i+1)*550] == i)
    assert np.all(units.field_index[pyr.start+1650:pyr.stop] == 3)
    repeat = deepcopy(native); repeat.update(native_pid=999, slurm_job_id='different-replay')
    assert exporter.source_units(repeat, groups).circuit_id == units.circuit_id


def test_commands_are_idempotent_sets_with_native_background_refresh_and_hold(source):
    _, _, _, groups, pattern, _ = source
    # The pinned pattern repeats targets across times but has no duplicate
    # (time,unit) pair. Introduce one only in this projection unit fixture.
    pattern = pattern.copy(); pattern[:2] = 0
    pyr = next(group for group in groups if group.name == exporter.PYRAMIDAL)
    commands = control.source_commands(pattern, pyr.start, coupled_extension=True)
    ticks, values, features = exporter.source_controls(groups, pattern, commands)
    pairs = set(zip(commands['time_ms'].tolist(), commands['unit_ids'].tolist()))
    assert len(pairs) < len(commands['time_ms'])
    assert values[:,:828].sum() == len(pairs) and values[:,:828].max() == 1
    np.testing.assert_array_equal(ticks, np.arange(1001))
    for i, group in enumerate(groups):
        assert values[519,828+i] == 1
        assert values[520,828+i] == (group.name == exporter.PYRAMIDAL)
        assert values[539,828+i] == (group.name == exporter.PYRAMIDAL)
        assert values[540,828+i] == 1
    np.testing.assert_array_equal(values[[519,520,539,540],-2:], [[1.5,1],[1,0],[1,0],[1.5,1]])
    assert all(feature['operation'] == 'set' and feature['assignment_value'] == '10000' for feature in features[:828])


@pytest.mark.parametrize('kind', ['incomplete_reconciliation','command_reordering','changed_raw_event','unit_relabeling',
                                        'config_hash','config_resolved','config_extra','coupled_identity'])
def test_changed_source_or_incomplete_proof_is_rejected_before_export(source, tmp_path, kind):
    primary, analysis, _, _, _, commands = source
    if kind == 'incomplete_reconciliation':
        path = analysis/'reconciliation.json'; proof = json.loads(path.read_text()); proof['event_multisets_equal'] = False
        write_json(path, proof)
    elif kind == 'unit_relabeling':
        path = analysis/'control/report.json'; report = json.loads(path.read_text())
        report['current_units']['model_specific_physical_unit'] = 'mA'; write_json(path, report)
    elif kind.startswith('config_') or kind == 'coupled_identity':
        path = analysis/'control/report.json'; report = json.loads(path.read_text())
        if kind == 'config_hash': report['coupled_extension_config']['sha256'] = '9'*64
        elif kind == 'config_resolved': report['coupled_extension_config']['resolved']['population_multiplier'] = 3
        elif kind == 'config_extra': report['coupled_extension_config']['extra'] = True
        else: report['coupled_extension']['graph_or_output_cloning'] = True
        write_json(path, report)
    elif kind == 'command_reordering':
        np.savez(analysis/'control/cue_command_events.npz', **{key:value[::-1] for key,value in commands.items()})
        path = analysis/'control/report.json'; report = json.loads(path.read_text())
        report['artifacts']['cue_command_events.npz'] = sha256_file(analysis/'control/cue_command_events.npz'); write_json(path, report)
    else:
        path = primary/'native_run/results/spk_CA3_Pyramidal.dat'; data = path.read_bytes()
        path.write_bytes(data[:-1]+bytes([data[-1]^1]))
    delivery(analysis, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    output = tmp_path/'rejected'
    with pytest.raises(ValueError):
        exporter.export(primary, analysis, output)
    assert not output.exists()


def test_failed_delivery_cannot_be_exported_and_existing_output_is_preserved(source, tmp_path):
    primary, analysis, *_ = source
    path = analysis/'delivery.json'; data = json.loads(path.read_text()); data['status'] = 'FAILED'; write_json(path, data)
    with pytest.raises(ValueError, match='COMPLETED_DELIVERY_REQUIRED'):
        exporter.export(primary, analysis, tmp_path/'failed')
    existing = tmp_path/'existing'; existing.mkdir(); (existing/'sentinel').write_text('retain')
    with pytest.raises(FileExistsError):
        exporter.export(primary, analysis, existing)
    assert (existing/'sentinel').read_text() == 'retain'


def test_changed_durable_origin_is_rejected(source, tmp_path):
    primary, analysis, *_ = source
    origins = tmp_path/'origins.json'
    write_json(origins, {role:dict(archive=str(root), delivery_sha256='0'*64)
                        for role, root in [('primary', primary), ('observation_analysis', analysis)]})
    with pytest.raises(ValueError, match='DURABLE_ORIGIN_BINDING_CHANGED'):
        exporter.export(primary, analysis, tmp_path/'invalid_origin', input_origins=origins)


@pytest.fixture
def cue_pair(source, tmp_path):
    """Synthetic completed-proof inputs; this fixture executes no native replay."""
    primary, analysis, _, _, _, _ = source
    omission = tmp_path/'omission'; shutil.copytree(primary, omission)
    # The measured-pair receipt is a prerequisite fixture, not a claim that
    # these marker bytes contain a native 1001-ms current/counter stream.
    trace = omission/'native_observation.bin'
    trace.write_bytes(b'synthetic prerequisite proof; never a model input')
    path = omission/'native_report.json'; native = json.loads(path.read_text())
    native.update(retrieval_condition='cue_omission', observed_native_duration_ms=1001,
        host_request_runtime_current_equal=True, native_observation_stream_validated=True,
        native_observation_sha256=sha256_file(trace), native_pid=2,
        overlay_source_sha256='b'*64)
    # A condition-specific main must not become a different circuit/block.
    native['source_file_byte_checks'][0]['staged_sha256'] = 'b'*64
    write_json(path, native)
    path = omission/'observations/recording.json'; recording = json.loads(path.read_text())
    recording['native_report_sha256'] = sha256_file(omission/'native_report.json')
    write_json(path, recording)
    delivery(omission, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    proof = json.loads((analysis/'reconciliation.json').read_text())
    contrast = tmp_path/'contrast'; (contrast/'analysis').mkdir(parents=True)
    measured = dict(status='MEASURED_CA3_CUE_OMISSION_CONTRAST', units=168106, duration_ms=1001,
        scope='development_whole_schedule_contrast', pre_cue_events_equal=True,
        pre_cue_interval_ms=[0, 520], configured_initialization_identity_equal=True,
        backend_library_bytes_equal=True, source_counter_and_current_qualification_reused=True,
        omission_current_checks=dict(intervals_compared=1001, current_values_compared=168274106,
            mismatched_values=0, exact_float32_bits_equal=True, tolerance=0),
        omission_counter_checks=dict(unit_millisecond_counts_compared=168274106, native_intervals_compared=1001),
        input_sha256=dict(source_reconciliation=sha256_file(analysis/'reconciliation.json'),
            source={'native_report.json':proof['observed_report_sha256'],
                    'native_observation.bin':proof['native_observation_sha256']},
            omission={name:sha256_file(omission/name) for name in (
                'native_report.json', 'native_observation.bin',
                'observations/recording.json', 'observations/population_counts.npz')}))
    write_json(contrast/'analysis/report.json', measured)
    delivery(contrast, 'MEASURED_CA3_CUE_OMISSION_CONTRAST')
    return omission, contrast


def test_omission_cli_keeps_source_circuit_block_and_full_native_roundtrip(source, cue_pair, tmp_path, capsys):
    primary, analysis, native, groups, pattern, commands = source
    omission, contrast = cue_pair
    source_output, omission_output = tmp_path/'source_export', tmp_path/'omission_export'
    original = exporter.export(primary, analysis, source_output)
    assert exporter.main(['--primary', str(primary), '--observation-analysis', str(analysis),
        '--cue-omission', str(omission), '--cue-contrast', str(contrast),
        '--output', str(omission_output)]) == 0
    capsys.readouterr()
    omitted = json.loads((omission_output/'report.json').read_text())
    assert omitted['status'] == 'EXPORTED_VERIFIED_CA3_DEVELOPMENT_DATASET'
    assert omitted['retrieval_condition'] == 'cue_omission'
    assert omitted['complete_episodes'] == 1 and omitted['production_episodes'] == 0
    assert omitted['new_source_simulations'] == omitted['source_episodes_added_to_campaign'] == 0
    assert omitted['circuit_id'] == original['circuit_id']
    assert omitted['block_id'] == original['block_id']
    assert omitted['episode_id'] != original['episode_id']
    assert omitted['episode_id'].startswith('ca3_cue_omission_')
    assert original['episode_id'].startswith('ca3_source_')
    assert omitted['block_id'].startswith('ca3_source_block_')
    assert omitted['units'] == 168106 and omitted['control_features'] == 838
    assert omitted['verification']['all_unit_bin_values_checked'] == 1001*168106
    assert omitted['verification']['event_records'] == 35
    assert omitted['verification']['future_realized_currents_in_model_inputs'] is False
    assert not omitted['scientific_fidelity_qualified']
    observed_native = json.loads((omission/'native_report.json').read_text())
    assert exporter.source_units(observed_native, groups).circuit_id != omitted['circuit_id']
    dataset = exporter.CircuitDataset(omission_output/'dataset', splits=('development',))
    binding = dataset.catalog['binding']
    assert binding['primary_archive'] == str(omission.resolve())
    assert binding['source_reference_archive'] == str(primary.resolve())
    assert binding['cue_contrast_archive'] == str(contrast.resolve())
    row = dataset.episodes[omitted['episode_id']]; shard = dataset.shards[row['shard']]
    _, source_values, _ = exporter.source_controls(groups, pattern, commands)
    private_names = {'source_event_ordinal'} | {'raw_command_'+key for key in commands}
    with h5py.File(dataset._verified(shard['path'], shard['sha256']), 'r') as archive:
        stored = archive[omitted['episode_id']]
        metadata = json.loads(stored.attrs['metadata_json'])
        record = metadata['source_record']
        assert record['source_reference_native_report_sha256'] == sha256_file(primary/'native_report.json')
        assert record['primary_native_report_sha256'] == sha256_file(omission/'native_report.json')
        assert record['shared_source_circuit_and_configured_block']
        assert not record['independent_noise_realization_claimed']
        assert metadata['seeds'] == {'source_carlsim_seed':10, 'source_host_srand_seed':3}
        assert set(stored['privileged']) == private_names
        for key, value in commands.items():
            np.testing.assert_array_equal(stored['privileged/raw_command_'+key][:], value[:0])
        values = stored['control_values'][:]
        assert values.shape == (1001, 838) and source_values[:,:828].sum() > 0
        assert np.count_nonzero(values[:,:828]) == 0
        np.testing.assert_array_equal(values[:,828:], source_values[:,828:])
        np.testing.assert_array_equal(stored['control_ticks'][:], np.arange(1001))
        ticks, ids = stored['spike_ticks'][:], stored['spike_unit_indices'][:]
    # The unchanged synthetic event records are a legal zero contrast; no
    # beneficial physiological effect is manufactured for this conversion test.
    assert np.count_nonzero(ticks == 1000) > 0
    tail = dataset.window(omitted['episode_id'], boundary_tick=1000,
        history_ticks=1, horizon_ticks=1, bin_ticks=1)
    assert tail['targets'].shape == (1, 168106)
    np.testing.assert_array_equal(tail['targets'][0],
        np.bincount(ids[ticks == 1000], minlength=168106))
    np.testing.assert_array_equal(tail['known_controls'], values[999:1001])
    assert not private_names.intersection(tail)
    assert not {'privileged', 'native_observation', 'actual_currents', 'future_realized_currents'}.intersection(tail)


def test_omission_controls_retain_all_background_laws_and_feature_identities(source):
    _, _, _, groups, pattern, commands = source
    ticks, source_values, features = exporter.source_controls(groups, pattern, commands)
    omission_ticks, omitted, omission_features = exporter.source_controls(
        groups, pattern, {key:value[:0] for key,value in commands.items()}, cue_omission=True)
    np.testing.assert_array_equal(omission_ticks, ticks)
    assert omission_features == features and len(features) == 838
    assert np.count_nonzero(omitted[:,:828]) == 0
    np.testing.assert_array_equal(omitted[:,828:], source_values[:,828:])
    assert np.count_nonzero(omitted[:,828:]) > 0
    for t in (520, 539, 750, 769, 980, 999):
        for i, group in enumerate(groups):
            assert omitted[t,828+i] == (group.name == exporter.PYRAMIDAL)
        np.testing.assert_array_equal(omitted[t,-2:], [1, 0])
    assert np.all(omitted[1000,828:836] == 1)
    np.testing.assert_array_equal(omitted[1000,-2:], [1.5, 1])


@pytest.mark.parametrize('mutation', ['source_assignments', 'nonempty_array', 'missing_key', 'extra_key'])
def test_omission_controls_require_exact_empty_source_command_arrays(source, mutation):
    _, _, _, groups, pattern, source_commands = source
    commands = {key:value[:0] for key,value in source_commands.items()}
    if mutation == 'source_assignments':
        commands = source_commands
    elif mutation == 'nonempty_array':
        commands['unit_ids'] = source_commands['unit_ids'][:1]
    elif mutation == 'missing_key':
        del commands['unit_ids']
    else:
        commands['realized_future_current'] = np.empty(0)
    with pytest.raises(ValueError, match='ORDERED_SOURCE_COMMANDS_CHANGED'):
        exporter.source_controls(groups, pattern, commands, cue_omission=True)


@pytest.mark.parametrize('missing', ['omission', 'contrast'])
def test_omission_and_contrast_must_both_be_explicit(source, cue_pair, tmp_path, missing):
    primary, analysis, *_ = source
    omission, contrast = cue_pair
    arguments = dict(omission_archive=omission, cue_contrast_archive=contrast)
    del arguments['omission_archive' if missing == 'omission' else 'cue_contrast_archive']
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='OMISSION_AND_CONTRAST_REQUIRED_TOGETHER'):
        exporter.export(primary, analysis, output, **arguments)
    assert not output.exists()


@pytest.mark.parametrize('field,value', [
    (('status',), 'FAILED'),
    (('duration_ms',), 1000),
    (('units',), 168105),
    (('pre_cue_events_equal',), False),
    (('omission_current_checks','intervals_compared'), 1000),
    (('omission_current_checks','current_values_compared'), 168106*1000),
    (('omission_current_checks','mismatched_values'), 1),
    (('omission_counter_checks','native_intervals_compared'), 1000),
    (('omission_counter_checks','unit_millisecond_counts_compared'), 168106*1000),
])
def test_omission_rejects_failed_or_incomplete_paired_measurement(source, cue_pair, tmp_path, field, value):
    primary, analysis, *_ = source
    omission, contrast = cue_pair
    path = contrast/'analysis/report.json'; report = json.loads(path.read_text())
    target = report
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    write_json(path, report); delivery(contrast, 'MEASURED_CA3_CUE_OMISSION_CONTRAST')
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='COMPLETE_CUE_CONTRAST_REQUIRED'):
        exporter.export(primary, analysis, output, omission_archive=omission, cue_contrast_archive=contrast)
    assert not output.exists()


@pytest.mark.parametrize('role,name,error', [
    ('root', 'source_reconciliation', 'CONTRAST_SOURCE_BINDING_CHANGED'),
    ('source', 'native_report.json', 'CONTRAST_SOURCE_BINDING_CHANGED'),
    ('source', 'native_observation.bin', 'CONTRAST_SOURCE_BINDING_CHANGED'),
    ('omission', 'native_report.json', 'CONTRAST_OMISSION_BINDING_CHANGED'),
    ('omission', 'observations/recording.json', 'CONTRAST_OMISSION_BINDING_CHANGED'),
    ('omission', 'observations/population_counts.npz', 'CONTRAST_OMISSION_BINDING_CHANGED'),
    ('omission', 'native_observation.bin', 'CUE_OMISSION_IDENTITY_CHANGED'),
])
def test_omission_rejects_contrast_bound_to_other_inputs(source, cue_pair, tmp_path, role, name, error):
    primary, analysis, *_ = source
    omission, contrast = cue_pair
    path = contrast/'analysis/report.json'; report = json.loads(path.read_text())
    target = report['input_sha256'] if role == 'root' else report['input_sha256'][role]
    target[name] = '0'*64
    write_json(path, report); delivery(contrast, 'MEASURED_CA3_CUE_OMISSION_CONTRAST')
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match=error):
        exporter.export(primary, analysis, output, omission_archive=omission, cue_contrast_archive=contrast)
    assert not output.exists()


@pytest.mark.parametrize('name,field,value', [
    ('native_report.json', 'observed_native_duration_ms', 1000),
    ('native_report.json', 'normalized_input_sha256', '0'*64),
    ('observations/recording.json', 'native_report_sha256', '0'*64),
    ('observations/recording.json', 'observation_sha256', '0'*64),
])
def test_omission_requires_native_and_recording_internal_identity(source, cue_pair, tmp_path, name, field, value):
    primary, analysis, *_ = source
    omission, contrast = cue_pair
    path = omission/name; document = json.loads(path.read_text()); document[field] = value
    write_json(path, document)
    # Rebind the synthetic outer receipt so the rejection has to come from
    # the native/recording identity contract, not a stale delivery checksum.
    path = contrast/'analysis/report.json'; report = json.loads(path.read_text())
    report['input_sha256']['omission'][name] = sha256_file(omission/name)
    if name == 'native_report.json':
        recording_path = omission/'observations/recording.json'
        recording = json.loads(recording_path.read_text())
        recording['native_report_sha256'] = sha256_file(omission/name)
        write_json(recording_path, recording)
        report['input_sha256']['omission']['observations/recording.json'] = sha256_file(recording_path)
    write_json(path, report)
    delivery(omission, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    delivery(contrast, 'MEASURED_CA3_CUE_OMISSION_CONTRAST')
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='CUE_OMISSION_IDENTITY_CHANGED'):
        exporter.export(primary, analysis, output, omission_archive=omission, cue_contrast_archive=contrast)
    assert not output.exists()


def test_failed_contrast_delivery_never_exports_an_episode(source, cue_pair, tmp_path):
    primary, analysis, *_ = source
    omission, contrast = cue_pair
    path = contrast/'delivery.json'; receipt = json.loads(path.read_text()); receipt['status'] = 'FAILED'
    write_json(path, receipt)
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='COMPLETED_DELIVERY_REQUIRED'):
        exporter.export(primary, analysis, output, omission_archive=omission, cue_contrast_archive=contrast)
    assert not output.exists()


def test_omission_does_not_discard_source_command_order_validation(source, cue_pair, tmp_path):
    primary, analysis, _, _, _, commands = source
    omission, contrast = cue_pair
    np.savez(analysis/'control/cue_command_events.npz',
        **{key:value[::-1] for key,value in commands.items()})
    path = analysis/'control/report.json'; report = json.loads(path.read_text())
    report['artifacts']['cue_command_events.npz'] = sha256_file(analysis/'control/cue_command_events.npz')
    write_json(path, report)
    delivery(analysis, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='ORDERED_SOURCE_COMMANDS_CHANGED'):
        exporter.export(primary, analysis, output, omission_archive=omission, cue_contrast_archive=contrast)
    assert not output.exists()


def realized_copy(source, root, *, uniform_seed=1001, engine_seed=2001, block_id='noise-01', circuit_seed=10):
    """Use the existing pinned-source fixture; this runs no native simulation."""
    original_primary, original_analysis, *_ = source
    primary, analysis = root/'primary', root/'analysis'
    shutil.copytree(original_primary, primary); shutil.copytree(original_analysis, analysis)
    original = (primary/'main.cpp').read_bytes()
    config = add_realization(primary)
    native = json.loads((primary/'native_report.json').read_text())
    circuit_id = ('normalized-'+native['normalized_input_sha256'] if circuit_seed == 10
                  else 'fixture-formed-circuit-'+str(circuit_seed))
    spec = SourceRealization(circuit_id, block_id,
                             'retrieval', 'development', circuit_seed, uniform_seed, engine_seed)
    value = json.loads(config.read_text()); value['ca3_source_realization'] = spec.record(); write_json(config, value)
    effective = program(original, spec)
    for name in ('main.cpp', 'realization_source.cpp'):
        (primary/name).write_bytes(effective)
    for name in ('build_report.json', 'native_report.json'):
        path = primary/name; value = json.loads(path.read_text())
        value.update(source_realization=contract(spec), overlay_source_sha256=sha256_file(primary/'realization_source.cpp'))
        value['source_realization_config'].update(sha256=sha256_file(config), resolved=spec.record())
        # A real new build has a different staging path, not different parameters.
        value['coupled_extension_config']['path'] = str(root/'another_build/coupled_extension.yaml')
        row = next(row for row in value['source_file_byte_checks'] if row['path'].startswith('src/main_ca3_'))
        row['staged_sha256'] = value['overlay_source_sha256']
        if name == 'native_report.json':
            value['realization_circuit_binding'].update(circuit_id=spec.circuit_id, block_id=spec.block_id,
                                                        circuit_seed=spec.circuit_seed)
        write_json(path, value)
    native = json.loads((primary/'native_report.json').read_text())
    path = primary/'observations/recording.json'; recording = json.loads(path.read_text())
    recording.update(source_realization=native['source_realization'], realization_circuit_binding=native['realization_circuit_binding'])
    write_json(path, recording); refresh_bindings(primary)
    native = json.loads((primary/'native_report.json').read_text())
    path = analysis/'reconciliation.json'; proof = json.loads(path.read_text())
    proof.update(original_report_sha256=sha256_file(primary/'native_report.json'),
                 original_recording_sha256=sha256_file(primary/'observations/recording.json'),
                 source_realization=native['source_realization'], realization_circuit_binding=native['realization_circuit_binding'])
    write_json(path, proof)
    path = analysis/'control/report.json'; delivered = json.loads(path.read_text())
    delivered.update(source_realization=native['source_realization'], realization_circuit_binding=native['realization_circuit_binding'])
    delivered['input_sha256']['reconciliation'] = sha256_file(analysis/'reconciliation.json')
    write_json(path, delivered)
    delivery(primary, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    delivery(analysis, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    return primary, analysis, config, spec


def realization_cli(primary, analysis, config, output, reference=None):
    args = [sys.executable, '-B', '-m', 'population_dynamics.ca3_source_dataset',
            '--primary', str(primary), '--observation-analysis', str(analysis), '--output', str(output)]
    if config is not None:
        args += ['--source-realization-config', str(config)]
    if reference is not None:
        args += ['--circuit-reference-primary', str(reference)]
    return subprocess.run(args, capture_output=True, text=True)


def test_realization_cli_keeps_legacy_circuit_and_distinct_noise_blocks(source, tmp_path):
    reference, _, native, groups, *_ = source
    legacy = exporter.source_units(native, groups)
    identities = []
    # The third run repeats the first realized process under another job/path.
    for name, uniform, engine, block in [('one',1001,2001,'noise-01'),
                                        ('two',1002,2002,'noise-02'),
                                        ('repeat',1001,2001,'noise-01')]:
        primary, analysis, config, spec = realized_copy(source, tmp_path/name,
            uniform_seed=uniform, engine_seed=engine, block_id=block)
        output = tmp_path/(name+'_export')
        completed = realization_cli(primary, analysis, config, output, reference)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report['circuit_id'] == legacy.circuit_id
        assert report['verification']['all_unit_bin_values_checked'] == 168274106
        assert report['complete_episodes'] == 1
        assert report['new_source_simulations'] == report['production_episodes'] == 0
        assert report['scientific_fidelity_qualified'] is False
        identities.append((report['episode_id'], report['block_id']))
        dataset = exporter.CircuitDataset(output/'dataset', splits=('development',))
        row = dataset.episodes[report['episode_id']]; shard = dataset.shards[row['shard']]
        with h5py.File(dataset._verified(shard['path'], shard['sha256']), 'r') as archive:
            metadata = json.loads(archive[report['episode_id']].attrs['metadata_json'])
        assert metadata['seeds'] == dict(source_carlsim_seed=10, source_host_srand_seed=3,
                                        source_uniform_seed=uniform, source_engine_seed=engine)
        record = metadata['source_record']
        assert record['source_realization'] == contract(spec)
        assert record['source_realization_config'] == dict(sha256=sha256_file(config), resolved=spec.record())
        assert record['circuit_provenance'] == legacy.circuit_provenance
        assert record['realization_circuit_binding']['block_id'] == block
        assert record['realization_circuit_binding']['circuit_id'] == spec.circuit_id
        assert record['circuit_reference_native_report_sha256'] == sha256_file(reference/'native_report.json')
        assert not record['observed_replay_counted_as_episode']
        assert not record['independent_noise_qualified']
        assert not record['independent_microscopic_states_qualified']
        view = dataset.window(report['episode_id'], boundary_tick=520, history_ticks=1, horizon_ticks=1, bin_ticks=1)
        assert not {'privileged','native_observation','actual_currents','future_realized_currents'}.intersection(view)
        assert not any(key.startswith('raw_command_') for key in view)
        assert all(feature['assignment_unit'] == 'pA' for feature in view['control_features'][:828])
    assert identities[0] != identities[1]
    assert identities[0] == identities[2]


def test_realization_without_legacy_reference_has_path_independent_parameters(source, tmp_path):
    units = []
    for name, seed in [('first',1001), ('second',1002)]:
        primary, _, config, _ = realized_copy(source, tmp_path/name, uniform_seed=seed)
        _, native, _, groups, *_ = _bound_inputs(primary, coupled_extension=True, source_realization_config=config)
        units.append(exporter.source_units(native, groups, realization_source=primary/'realization_source.cpp'))
    assert units[0].circuit_id == units[1].circuit_id
    assert units[0].circuit_provenance == units[1].circuit_provenance


@pytest.mark.parametrize('mutation', ['missing_config', 'config_seed', 'training_seed', 'source_bytes',
    'source_pre_hash', 'proof_noise', 'control_block', 'native_circuit', 'numeric_conversion', 'future_control'])
def test_realization_cli_rejects_changed_source_seed_or_private_control(source, tmp_path, mutation):
    primary, analysis, config, _ = realized_copy(source, tmp_path/'realized')
    if mutation == 'missing_config':
        config = None
    elif mutation in ('config_seed', 'training_seed'):
        wrong = tmp_path/'wrong.yaml'; value = json.loads(config.read_text())
        if mutation == 'config_seed': value['ca3_source_realization']['uniform_seed'] = 1002
        else: value['training_seeds'] = [41,42,44]
        write_json(wrong, value); config = wrong
    elif mutation == 'source_bytes':
        path = primary/'realization_source.cpp'; path.write_bytes(path.read_bytes()+b'\n// changed source\n')
    elif mutation == 'source_pre_hash':
        path = primary/'build_report.json'; value = json.loads(path.read_text())
        value['source_file_byte_checks'][0]['pre_realization_staged_sha256'] = '0'*64
        write_json(path, value)
    elif mutation == 'native_circuit':
        for name in ('native_report.json', 'observations/recording.json'):
            path = primary/name; value = json.loads(path.read_text())
            value['realization_circuit_binding']['circuit_seed'] = 11; write_json(path, value)
        refresh_bindings(primary)
        path = analysis/'reconciliation.json'; value = json.loads(path.read_text())
        value['original_report_sha256'] = sha256_file(primary/'native_report.json')
        value['original_recording_sha256'] = sha256_file(primary/'observations/recording.json')
        value['realization_circuit_binding']['circuit_seed'] = 11; write_json(path, value)
        path = analysis/'control/report.json'; value = json.loads(path.read_text())
        value['realization_circuit_binding']['circuit_seed'] = 11
        value['input_sha256']['reconciliation'] = sha256_file(analysis/'reconciliation.json'); write_json(path, value)
    elif mutation == 'proof_noise':
        path = analysis/'reconciliation.json'; value = json.loads(path.read_text())
        value['source_realization']['realization']['engine_seed'] = 2002; write_json(path, value)
        path = analysis/'control/report.json'; value = json.loads(path.read_text())
        value['input_sha256']['reconciliation'] = sha256_file(analysis/'reconciliation.json'); write_json(path, value)
    else:
        path = analysis/'control/report.json'; value = json.loads(path.read_text())
        if mutation == 'control_block': value['realization_circuit_binding']['block_id'] = 'another-block'
        elif mutation == 'numeric_conversion': value['numeric_conversion_applied'] = True
        else:
            path_commands = analysis/'control/cue_command_events.npz'
            with np.load(path_commands) as archive: commands = dict(archive)
            commands['future_realized_current'] = np.array([10000.0])
            np.savez(path_commands, **commands)
            value['artifacts']['cue_command_events.npz'] = sha256_file(path_commands)
        write_json(path, value)
    delivery(primary, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    delivery(analysis, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    output = tmp_path/'rejected'
    completed = realization_cli(primary, analysis, config, output, source[0])
    assert completed.returncode != 0 and ('CA3_' in completed.stderr or 'TRAINING_SEEDS_' in completed.stderr), completed.stdout
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['graph', 'parameter', 'source', 'unit_map', 'clock'])
def test_realization_rejects_changed_circuit_reference(source, tmp_path, mutation):
    primary, analysis, config, _ = realized_copy(source, tmp_path/'realized')
    reference = tmp_path/'reference'; shutil.copytree(source[0], reference)
    path = reference/'native_report.json'; native = json.loads(path.read_text())
    if mutation == 'graph': native['normalized_input_sha256'] = '0'*64
    elif mutation == 'parameter': native['conductance_memory_overlay'] = {'different_kernel':True}
    elif mutation == 'source': native['source_file_byte_checks'][0]['staged_sha256'] = '0'*64
    write_json(path, native)
    path = reference/'observations/recording.json'; recording = json.loads(path.read_text())
    recording['native_report_sha256'] = sha256_file(reference/'native_report.json')
    if mutation == 'unit_map': recording['groups'][0]['name'] = 'different-population'
    elif mutation == 'clock': recording['time_resolution_ms'] = 2
    write_json(path, recording); delivery(reference, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    output = tmp_path/'rejected'
    completed = realization_cli(primary, analysis, config, output, reference)
    assert completed.returncode != 0, completed.stdout
    assert not output.exists()


@pytest.mark.parametrize('mode,node,partition,gpus,visible,allowed', [
    ('export','cpu03','gpu','','',True),
    ('export','cpu04','cpu','','',True),
    ('export-omission','cpu03','gpu','','',True),
    ('export-omission','cpu04','cpu','','',True),
    ('export','cpu05','cpu','','',False),
    ('export','cpu03','cpu','','',False),
    ('export','cpu04','gpu','','',False),
    ('export-omission','gpu03','gpu1','','',False),
    ('validate','cpu03','gpu','','',False),
    ('export','cpu04','cpu','0','',False),
    ('export','cpu04','cpu','','0',False),
])
def test_realization_launcher_stages_config_and_circuit_reference(
        source, cue_pair, tmp_path, mode, node, partition, gpus, visible, allowed):
    """Run the embedded launcher on synthetic inputs; no Slurm job is submitted."""
    if mode == 'export':
        primary, analysis, config, spec = realized_copy(source, tmp_path/'realized')
    else:
        primary, analysis, *_ = source
        config, spec = None, None
    repo = tmp_path/'launcher_repo'; software = tmp_path/'software'
    modules = Path(exporter.__file__).parent
    target = software/'Libs/population_dynamics'; target.mkdir(parents=True)
    for path in modules.glob('*.py'):
        if not path.name.startswith('test_'): shutil.copy2(path, target/path.name)
    delivery(software, 'PASS_CA3_SOURCE_DATASET_TESTS')
    # Model the canonical header link that real retrieval archives preserve.
    relative = 'native_run/network65p_assem550_divnorm.dat'
    header = primary/relative; canonical = tmp_path/'normalization'/header.name
    canonical.parent.mkdir(); header.replace(canonical)
    delivery(canonical.parent, 'fixture_normalized_header')
    header.symlink_to(os.path.relpath(canonical, header.parent))
    receipt_path = primary/'delivery.json'; receipt = json.loads(receipt_path.read_text())
    row = next(row for row in receipt['files'] if row['path'] == relative)
    row.update(canonical_source=str(canonical), symlink_target=os.readlink(header),
               source_delivery=str(canonical.parent/'delivery.json'),
               source_delivery_sha256=sha256_file(canonical.parent/'delivery.json'))
    write_json(receipt_path, receipt)
    launcher = Path(os.environ.get('NEURONSIM_CA3_DATASET_LAUNCHER', str(
        Path(__file__).resolve().parents[2]/'Output/neural_population_dynamics/operations/ca3_source_dataset.sbatch')))
    syntax = subprocess.run(['bash','-n',str(launcher)],capture_output=True,text=True)
    assert syntax.returncode == 0, syntax.stderr
    worker = launcher.read_text().split("<<'PYTHON'\n",1)[1].rsplit('\nPYTHON',1)[0]
    env = dict(os.environ, SLURM_JOB_ID='fixture-export', SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition,
        SLURM_JOB_GPUS=gpus, CUDA_VISIBLE_DEVICES=visible, CA3_SOURCE_REALIZATION_CONFIG=str(config) if config else '',
        CA3_CIRCUIT_REFERENCE_PRIMARY=str(source[0]) if config else '')
    inputs = [str(primary),str(analysis)]
    if mode == 'export-omission': inputs += [str(path) for path in cue_pair]
    completed = subprocess.run([sys.executable,'-B','-',str(repo),mode,*inputs,
        str(software),str(launcher)],input=worker,env=env,capture_output=True,text=True)
    if not allowed:
        assert completed.returncode != 0
        assert 'CA3_DATASET_ALLOWED_CPU_NODE_WITHOUT_GPU_REQUIRED' in completed.stderr
        assert not (repo/'Output').exists()
        return
    assert completed.returncode == 0, completed.stdout+completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1]); archive = Path(result['archive'])
    receipt = json.loads((archive/'delivery.json').read_text())
    assert receipt['status'] == 'EXPORTED_VERIFIED_CA3_DEVELOPMENT_DATASET'
    assert receipt['node'] == node
    assert receipt['copied_files_verified'] is True
    assert receipt['export']['verification']['all_unit_bin_values_checked'] == 168274106
    assert receipt['export']['circuit_id'] == exporter.source_units(source[2], source[3]).circuit_id
    dataset = exporter.CircuitDataset(archive/'export/dataset', splits=('development',))
    if config is not None:
        assert receipt['source_realization_config']['sha256'] == sha256_file(config)
        assert '--source-realization-config' in receipt['command']
        assert '--circuit-reference-primary' in receipt['command']
        assert {row['role'] for row in receipt['inputs']} == {'primary','observation_analysis','circuit_reference_primary'}
        assert {'realization_source.cpp','source_realization.yaml'} <= {row['path'] for row in receipt['inputs']}
        assert dataset.catalog['binding']['source_realization'] == contract(spec)
        assert dataset.catalog['binding']['circuit_reference_archive'] == str(source[0].resolve())
    else:
        assert receipt['export']['retrieval_condition'] == 'cue_omission'
        assert {row['role'] for row in receipt['inputs']} == {'primary','observation_analysis','omission','cue_contrast'}
        assert '--cue-omission' in receipt['command'] and '--cue-contrast' in receipt['command']


def test_source_bundle_controls_support_for_nonlegacy_circuit_seed(source, tmp_path):
    """Current source proofs support new circuits; older proofs must still reject."""
    primary, analysis, config, spec = realized_copy(source, tmp_path/'seed11', circuit_seed=11)
    output = tmp_path/'export11'
    completed = realization_cli(primary, analysis, config, output)
    if not hasattr(realization_protocol, 'bind_circuit_input'):
        assert completed.returncode != 0
        assert 'CA3_REALIZATION_PINNED_CIRCUIT_RETRIEVAL_REQUIRED' in completed.stderr
        assert not output.exists()
        return
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    dataset = exporter.CircuitDataset(output/'dataset', splits=('development',))
    row = dataset.episodes[report['episode_id']]; shard = dataset.shards[row['shard']]
    with h5py.File(dataset._verified(shard['path'], shard['sha256']), 'r') as archive:
        metadata = json.loads(archive[report['episode_id']].attrs['metadata_json'])
    assert metadata['seeds']['source_carlsim_seed'] == 11
    assert metadata['source_record']['source_realization'] == contract(spec)
    assert report['new_source_simulations'] == report['production_episodes'] == 0

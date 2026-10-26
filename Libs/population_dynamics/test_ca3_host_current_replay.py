"""Synthetic tests of streaming, binding and failure handling; native replay is separate."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from . import ca3_host_current_replay as replay
from . import ca3_native_observation as observation
from .provenance import sha256_file
from .ca3_realization import SourceRealization


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    supplied = os.environ.get('CA3_REPLAY_ENGINE')
    if supplied:
        return Path(supplied)
    target = tmp_path_factory.mktemp('host-current')/'engine'
    subprocess.run(['g++', '-std=c++11', '-O3', str(replay.SOURCE_FILE), '-o', str(target)], check=True)
    return target


@pytest.fixture(scope='module')
def sample(engine):
    groups, offset = [], 0
    for i, name in enumerate(reversed(replay.SOURCE_ORDER)):
        size = 5 if name == replay.PYRAMIDAL else 2+i % 2
        groups.append(observation.Group(i, offset, offset+size, 0, i, offset, offset+size, 0, name))
        offset += size
    pyr = next(g for g in groups if g.name == replay.PYRAMIDAL)
    commands = dict(time_ms=np.array([520, 520, 750, 980], dtype=np.int32),
                    unit_ids=np.array([pyr.start, pyr.start, pyr.start+1, pyr.start+2], dtype=np.int64))
    result = subprocess.run([str(engine)], input=replay.engine_input(groups, commands),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    values = np.frombuffer(result.stdout, dtype='<f4').reshape(1001, offset).copy()
    return tuple(groups), commands, values


def intervals(values):
    for t, row in enumerate(values):
        yield observation.Interval(t, t+1, np.zeros(len(row), dtype=np.int64), row, row)


def test_persistent_schedule_and_idempotent_assignments(engine, sample):
    groups, commands, values = sample
    pyr = next(g for g in groups if g.name == replay.PYRAMIDAL)
    empty = dict(time_ms=np.array([], dtype=np.int32), unit_ids=np.array([], dtype=np.int64))
    result = subprocess.run([str(engine)], input=replay.engine_input(groups, empty),
                            stdout=subprocess.PIPE, check=True)
    baseline = np.frombuffer(result.stdout, dtype='<f4').reshape(values.shape)
    expected = baseline.copy()
    for t, unit in zip(commands['time_ms'], commands['unit_ids']):
        expected[t, unit] = 10000
    np.testing.assert_array_equal(values.view('<u4'), expected.view('<u4'))
    for start in (520, 750, 980):
        for group in groups:
            if group.name != replay.PYRAMIDAL:
                hold = values[start-1, group.start:group.stop]
                np.testing.assert_array_equal(values[start:start+20, group.start:group.stop],
                                              np.broadcast_to(hold, (20, group.units)))
        assert not np.array_equal(values[start, pyr.start:pyr.stop], values[start+1, pyr.start:pyr.stop])
    assert np.all(values >= 0)
    assert not np.array_equal(values[0], values[1])
    assert not np.array_equal(values[999], values[1000])


def test_complete_stream_and_header_order(engine, sample):
    groups, commands, values = sample
    result = replay.compare_currents(engine, groups[::-1], commands, intervals(values))
    assert result['exact_float32_bits_equal']
    assert result['current_values_compared'] == values.size
    assert result['intervals_compared'] == 1001
    assert result['mismatched_values'] == 0


def test_extended_terminal_background_preserves_every_prefix_float(engine, sample):
    groups, commands, original = sample
    run = subprocess.run([str(engine), '--duration-ms', '1210'],
                         input=replay.engine_input(groups, commands), stdout=subprocess.PIPE, check=True)
    extended = np.frombuffer(run.stdout, dtype='<f4').reshape(1210, original.shape[1]).copy()
    np.testing.assert_array_equal(extended[:1001].view('<u4'), original.view('<u4'))
    assert not np.array_equal(extended[1001], extended[1002])
    checked = replay.compare_currents(engine, groups, commands, intervals(extended), expected_stop_ms=1210)
    assert checked['exact_float32_bits_equal'] and checked['intervals_compared'] == 1210
    changed = extended.copy()
    changed[-1, -1] = np.nextafter(changed[-1, -1], np.float32(np.inf))
    checked = replay.compare_currents(engine, groups, commands, intervals(changed), expected_stop_ms=1210)
    assert checked['mismatched_values'] == 1 and checked['first_mismatches'][0]['time_ms'] == 1209
    with pytest.raises(ValueError, match='COMPLETE_ORDERED_CLOCK'):
        replay.compare_currents(engine, groups, commands, intervals(extended[:-1]), expected_stop_ms=1210)
    failed = subprocess.run([str(engine), '--duration-ms', '1209'], stdout=subprocess.PIPE)
    assert failed.returncode != 0 and not failed.stdout
    realization = SourceRealization('candidate', 'block', 'retrieval', 'development',
                                    300000000, 320000000, 310000000)
    def seeded(duration):
        output = subprocess.check_output([str(engine), *realization.replay_arguments(),
                                          '--duration-ms', str(duration)],
                                         input=replay.engine_input(groups, commands))
        return np.frombuffer(output, dtype='<f4').reshape(duration, original.shape[1]).copy()
    original_seeded, extended_seeded = seeded(1001), seeded(1210)
    np.testing.assert_array_equal(extended_seeded[:1001].view('<u4'), original_seeded.view('<u4'))
    checked = replay.compare_currents(engine, groups, commands, intervals(extended_seeded),
                                     realization=realization, expected_stop_ms=1210)
    assert checked['exact_float32_bits_equal'] and checked['current_values_compared'] == extended_seeded.size


def test_last_value_single_bit_mismatch_is_reported(engine, sample):
    groups, commands, values = sample
    changed = values.copy()
    changed[-1, -1] = np.nextafter(changed[-1, -1], np.float32(np.inf))
    result = replay.compare_currents(engine, groups, commands, intervals(changed))
    assert not result['exact_float32_bits_equal']
    assert result['mismatched_values'] == 1
    assert result['current_values_compared'] == values.size
    assert result['first_mismatches'][0]['time_ms'] == 1000
    assert result['first_mismatches'][0]['unit_id'] == values.shape[1]-1
    assert result['tolerance'] == 0


def test_short_or_disordered_clock_rejected(engine, sample):
    groups, commands, values = sample
    with pytest.raises(ValueError, match='COMPLETE_ORDERED_CLOCK'):
        replay.compare_currents(engine, groups, commands, intervals(values[:-1]))
    first = next(intervals(values))
    with pytest.raises(ValueError, match='COMPLETE_ORDERED_CLOCK'):
        replay.compare_currents(engine, groups, commands, [replace(first, start_ms=1, stop_ms=2)])


def test_population_overlap_and_non_cue_command_rejected(engine, sample):
    groups, commands, _ = sample
    bad = (replace(groups[0], start=1),)+groups[1:]
    with pytest.raises(ValueError, match='COMPLETE_SOURCE_POPULATIONS'):
        replay.engine_input(bad, commands)
    bad_commands = dict(commands, time_ms=np.array([0, 520, 750, 980], dtype=np.int32))
    result = subprocess.run([str(engine)], input=replay.engine_input(groups, bad_commands),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0
    assert not result.stdout
    assert b'invalid ordered cue command' in result.stderr


def write_trace(path, groups, values):
    with path.open('wb') as stream:
        stream.write(observation.HEADER.pack(observation.MAGIC, len(groups), values.shape[1]))
        for g in groups:
            name = g.name.encode()
            stream.write(observation.GROUP.pack(g.group_id, g.start, g.stop, g.net_id,
                g.local_group, g.local_start, g.local_stop, g.gpu, len(name)))
            stream.write(name)
        for t, row in enumerate(values):
            for kind in (b'R', b'A', b'C'):
                for g in groups:
                    stream.write(observation.RECORD.pack(kind, t, t if kind == b'R' else t+1,
                                                        g.group_id, g.units))
                    payload = np.zeros(g.units, dtype='<i4') if kind == b'C' else row[g.start:g.stop]
                    stream.write(payload.tobytes())
        stream.write(observation.RECORD.pack(b'E', 1001, 1001, -1, 0))


@pytest.fixture
def bound_sample(tmp_path, monkeypatch, sample):
    groups, commands, values = sample
    observed = tmp_path/'observed'
    observed.mkdir()
    write_trace(observed/'native_observation.bin', groups, values)
    (observed/'trainPat.csv').write_text('0\n')
    native = dict(native_observation_sha256=sha256_file(observed/'native_observation.bin'),
                  native_observation_bytes=(observed/'native_observation.bin').stat().st_size)
    (observed/'native_report.json').write_text(json.dumps(native))
    identity = dict(core_units=values.shape[1], populations={g.name: g.units for g in groups},
                    overlay_sha256='fixture-main', config_sha256='fixture-config',
                    pattern_sha256=sha256_file(observed/'trainPat.csv'))
    monkeypatch.setattr(replay, 'source_identity', lambda **kwargs: identity)
    monkeypatch.setattr(replay, 'source_commands', lambda *args, **kwargs: commands)
    hashes = {name: sha256_file(observed/name) for name in
              ('native_report.json', 'native_observation.bin', 'trainPat.csv')}
    control = dict(status='VERIFIED_SOURCE_CA3_CONTROL_DELIVERY', units=values.shape[1],
        intervals_verified=1001, actual_current_values_checked=values.size,
        ordered_assignments_verified=len(commands['unit_ids']), numeric_conversion_applied=False,
        current_units={'model_specific_physical_unit': 'pA'},
        observer_implementation_sha256=sha256_file(Path(observation.__file__)),
        input_sha256=dict(hashes, main_source='fixture-main', config_source='fixture-config'))
    report = tmp_path/'control.json'
    report.write_text(json.dumps(control))
    return observed, report, control


def test_bound_observation_reader_to_engine(engine, bound_sample, tmp_path):
    observed, control, _ = bound_sample
    result = replay.analyze(observed, control, engine, tmp_path/'result')
    assert result['status'] == 'VERIFIED_CA3_HOST_CURRENT_REPLAY'
    assert result['source_host_current_rng_reconstructed']
    assert result['rng_contract']['fitted_state_or_parameters'] is False
    assert result['independent_rng_streams_qualified'] is False
    assert result['production_episodes'] == 0


def test_changed_bound_trace_rejected_before_engine(bound_sample, tmp_path):
    observed, control, _ = bound_sample
    with (observed/'native_observation.bin').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='CONTROL_INPUT_CHANGED'):
        replay.analyze(observed, control, tmp_path/'missing-engine', tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_incomplete_or_wrong_source_control_rejected(bound_sample):
    observed, path, control = bound_sample
    control['status'] = 'NOT_RUN'
    path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match='VERIFIED_CONTROL_DELIVERY'):
        replay.bound_inputs(observed, path, coupled_extension=False)
    control['status'] = 'VERIFIED_SOURCE_CA3_CONTROL_DELIVERY'
    control['input_sha256']['main_source'] = 'different-source'
    path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match='PINNED_SOURCE_REQUIRED'):
        replay.bound_inputs(observed, path, coupled_extension=False)


def test_explicit_noise_arguments_reach_compiled_engine_and_bound_trace(
        engine, sample, bound_sample, tmp_path, monkeypatch):
    """Test seed dispatch; actual source-program binding is tested separately."""
    from . import ca3_realization
    from .config import load_config

    groups, commands, _ = sample
    observed, control_path, control = bound_sample
    spec = ca3_realization.SourceRealization('fixture-circuit', 'fixture-block', 'retrieval',
                                            'development', 10, 1001, 2001)
    campaign = Path(os.environ.get('NEURONSIM_CA3_CAMPAIGN_CONFIG', str(
        Path(__file__).resolve().parents[1]/'configs/neural_population_dynamics.yaml')))
    config = observed/'source_realization.yaml'
    config.write_text(json.dumps({**load_config(campaign), 'ca3_source_realization': spec.record()}))
    (observed/'realization_source.cpp').write_text('isolated program-binding fixture; not compiled')
    raw = subprocess.check_output([str(engine), *spec.replay_arguments()],
                                  input=replay.engine_input(groups, commands))
    values = np.frombuffer(raw, dtype='<f4').reshape(1001, -1)
    write_trace(observed/'native_observation.bin', groups, values)
    native = json.loads((observed/'native_report.json').read_text())
    native.update(native_observation_sha256=sha256_file(observed/'native_observation.bin'),
                  native_observation_bytes=(observed/'native_observation.bin').stat().st_size,
                  source_realization=ca3_realization.contract(spec), overlay_source_sha256='fixture-main',
                  realization_circuit_binding=dict(circuit_id=spec.circuit_id,block_id=spec.block_id))
    (observed/'native_report.json').write_text(json.dumps(native))
    control.update(status='VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY',
                   source_realization=native['source_realization'],
                   realization_circuit_binding=native['realization_circuit_binding'])
    control['input_sha256'].update({name:sha256_file(observed/name) for name in
        ('native_report.json','native_observation.bin','trainPat.csv')})
    control_path.write_text(json.dumps(control))
    calls=[]
    def bound_program(root, report, requested):
        assert root==observed and requested==config
        assert report['source_realization']==ca3_realization.contract(spec)
        calls.append(True)
        return spec
    monkeypatch.setattr(ca3_realization,'bind_runtime',bound_program)
    result=replay.analyze(observed,control_path,engine,tmp_path/'explicit-result',
                          coupled_extension=True,source_realization_config=config)
    assert calls
    assert result['exact_float32_bits_equal']
    assert result['current_values_compared']==values.size
    assert result['source_realization']==ca3_realization.contract(spec)
    assert result['rng_contract']['pre_current_drand48_draws']==0
    assert result['rng_contract']['common_cpp_engine_seed']==2001
    assert not result['independent_rng_streams_qualified']

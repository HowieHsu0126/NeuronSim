"""Compare explicit dispatch with the actual source samplers and RNG transitions."""
import os
import json
from types import SimpleNamespace
from pathlib import Path
import random
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import yaml

from . import v1_lgn_spikes as spikes_module
from .provenance import sha256_file
from .v1_lgn_source import CONSTRUCTION_FILES

from .v1_lgn_runtime import load_prepared_runtime
from .v1_nest_input_replay import prepare_inputs
from .v1_lgn_spikes import generate_source_spikes, write_spike_trial, replace_lgn_inputs, input_plan_identity


@pytest.fixture(scope='module')
def source():
    functions, poisson = load_prepared_runtime(Path(os.environ['NEURONSIM_LGN_RUNTIME']))
    with h5py.File(os.environ['NEURONSIM_LGN_RATE_REFERENCE'], 'r') as data:
        clock = data['time'][:21]
    return functions.f_rate_to_spike_train, poisson.generate_inhomogenous_poisson, clock


def numpy_state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def tracked(function, states):
    def call(*args, **kwargs):
        result = function(*args, **kwargs)
        states.append((np.random.get_state(), random.getstate()))
        return result
    return call


def reference_dispatch(clock, rates, ids, trials, seed, primary, poisson):
    before = np.random.get_state(), random.getstate()
    outputs, seeds = [], []
    try:
        np.random.seed(seed)
        for trial in range(trials):
            rows = []
            for position, source_id in enumerate(ids):
                primary_seed = np.random.randint(10000)
                poisson_seed = None
                try:
                    events = np.asarray(primary(clock*1000.0, rates[position], primary_seed,
                        min(clock)*1000.0, max(clock)*1000.0, 0.1))
                    sampler = 'bernoulli'
                except Exception:
                    # Exact legacy fallback is used only as the source reference.
                    poisson_seed = np.random.randint(10000)
                    events = 1000.0*np.asarray(poisson(clock, rates[position], seed=poisson_seed))
                    sampler = 'poisson'
                rows.append(events)
                seeds.append(dict(trial=trial, source_id=int(source_id), sampler=sampler,
                    primary_seed=int(primary_seed), poisson_seed=None if poisson_seed is None else int(poisson_seed)))
            outputs.append(rows)
    finally:
        np.random.set_state(before[0]);random.setstate(before[1])
    return outputs, seeds


def test_mixed_source_dispatch_and_rng_chain_match(source):
    primary, poisson, clock = source
    ids = np.array([31, 7, 99])
    rates = np.array([np.full(len(clock), value) for value in [5.0, 200.0, 7.0]])
    expected_states, actual_states = [], []
    expected, expected_seeds = reference_dispatch(clock, rates, ids, 2, 12345,
        tracked(primary, expected_states), tracked(poisson, expected_states))
    numpy_before, python_before = np.random.get_state(), random.getstate()
    actual, seeds = generate_source_spikes(clock, rates, ids, 2, 12345,
        tracked(primary, actual_states), tracked(poisson, actual_states))
    assert seeds == expected_seeds
    assert [row['sampler'] for row in seeds] == ['bernoulli', 'poisson', 'bernoulli']*2
    for trial_actual, trial_expected in zip(actual, expected):
        for observed, reference in zip(trial_actual, trial_expected):
            assert np.array_equal(observed, reference)
    assert len(actual_states) == len(expected_states) == 6
    for a, b in zip(actual_states, expected_states):
        assert numpy_state_equal(a[0], b[0]) and a[1] == b[1]
    assert numpy_state_equal(np.random.get_state(), numpy_before)
    assert random.getstate() == python_before


@pytest.mark.parametrize('rates,expected_branch', [([10.0, 10.0], 'bernoulli'), ([0.0, 20.0], 'poisson')])
def test_source_guard_equality_and_unused_terminal_sample(source, rates, expected_branch):
    primary, poisson, _ = source
    clock = np.array([0.0, 0.01]);rates = np.array([rates]);ids = np.array([8])
    expected, expected_seeds = reference_dispatch(clock, rates, ids, 1, 54321, primary, poisson)
    actual, seeds = generate_source_spikes(clock, rates, ids, 1, 54321, primary, poisson)
    assert seeds == expected_seeds and seeds[0]['sampler'] == expected_branch
    assert np.array_equal(actual[0][0], expected[0][0])


def test_actual_clock_reaches_each_sampler_unchanged(source):
    _, _, clock = source
    seen = []
    def primary(t, rates, seed, start, end, limit):
        assert np.array_equal(t, clock*1000.0)
        assert start == min(clock)*1000.0 and end == max(clock)*1000.0 and limit == 0.1
        seen.append('primary');return []
    def poisson(t, rates, seed):
        assert np.array_equal(t, clock)
        seen.append('poisson');return []
    rates = np.array([np.full(len(clock), x) for x in [5.0, 200.0]])
    generate_source_spikes(clock, rates, np.array([1, 2]), 1, 123, primary, poisson)
    assert seen == ['primary', 'poisson']


def test_unexpected_primary_error_does_not_select_another_process(source):
    _, _, clock = source
    before = np.random.get_state(), random.getstate()
    primary = Mock(side_effect=RuntimeError('unexpected sampler defect'))
    poisson = Mock()
    with pytest.raises(RuntimeError, match='unexpected sampler defect'):
        generate_source_spikes(clock, np.full((1, len(clock)), 5.0), np.array([1]),
                               1, 123, primary, poisson)
    poisson.assert_not_called()
    assert numpy_state_equal(np.random.get_state(), before[0]) and random.getstate() == before[1]


def test_invalid_rates_are_rejected_before_sampling(source):
    _, _, clock = source
    primary, poisson = Mock(), Mock()
    rates = np.full((1, len(clock)), 5.0);rates[0, 1] = np.nan
    with pytest.raises(ValueError, match='CLOCK_RATES_OR_IDENTITIES'):
        generate_source_spikes(clock, rates, np.array([1]), 1, 123, primary, poisson)
    primary.assert_not_called();poisson.assert_not_called()


@pytest.mark.parametrize('trains', [
    [np.array([], dtype=np.float64), np.array([], dtype=np.float64)],
    [np.array([0.125, 0.125, 3000.0]), np.array([], dtype=np.float64)],
])
def test_export_preserves_silent_inventory_multiplicity_and_endpoint(tmp_path, trains):
    ids = np.array([31, 7], dtype=np.int64)
    seeds = [dict(source_id=int(i), primary_seed=4, poisson_seed=None, sampler='bernoulli') for i in ids]
    result = write_spike_trial(ids, trains, seeds, 3000.0, tmp_path/'trial')
    assert result['inventory_roundtrip_exact'] and result['event_order_exact']
    assert result['source_units'] == 2 and result['event_comparison']['all_event_pairs_exact']
    with np.load(tmp_path/'trial.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['source_id'], ids)
        assert np.array_equal(saved['source_counts'], [len(t) for t in trains])
        assert np.array_equal(saved['source_event_time_ms'], np.concatenate(trains))
    with h5py.File(tmp_path/'trial.h5') as saved:
        assert saved['spikes/lgn/timestamps'].attrs['units'] == 'ms'
        assert np.array_equal(saved['spikes/lgn/node_ids'][:], np.repeat(ids, [len(t) for t in trains]))


def test_zero_time_is_retained_but_rejected_for_native_input(tmp_path):
    ids = np.array([31])
    seeds = [dict(source_id=31, primary_seed=4, poisson_seed=None, sampler='bernoulli')]
    with pytest.raises(ValueError, match='V1_CONFIGURED_INPUT_PLAN_INVALID'):
        write_spike_trial(ids, [np.array([0.0])], seeds, 3000.0, tmp_path/'trial')
    with h5py.File(tmp_path/'trial.h5') as saved:
        assert np.array_equal(saved['spikes/lgn/timestamps'][:], [0.0])


@pytest.fixture
def replacement_case():
    raw = [(dict(ids=np.array([2, 0, 1]), spikes=[np.array([100.0]), np.array([]), np.array([200.0])]),
            [dict(source=np.array([0, 2]), target=np.array([5, 7]))]),
           (dict(ids=np.array([0]), spikes=[np.array([0.125, 3000.0])]),
            [dict(source=np.array([0]), target=np.array([8]))])]
    generated = prepare_inputs([(dict(ids=np.array([1, 0, 2]),
        spikes=[np.array([120.0]), np.array([]), np.array([200.0, 200.0])]), [])], 3000.0)
    origin = dict(status='SOURCE_INPUT_CANDIDATES_EXACT', candidate_event_and_inventory_identity_exact=True,
        source_clock=dict(dt=0.25, duration=3000.0), cases=[dict(target_node_set='lgn',
        all_event_pairs_exact=True, input_generator_inventory_exact=True, candidate_archive_population=0)])
    return raw, generated, prepare_inputs(raw, 3000.0), origin


def test_lgn_replacement_maps_full_ids_and_preserves_background_and_edges(replacement_case):
    raw, generated, baseline, origin = replacement_case
    result, report = replace_lgn_inputs(raw, generated, baseline, origin)
    assert result is not raw and result[1] is raw[1]
    assert result[0][1] is raw[0][1]
    assert np.array_equal(result[0][0]['ids'], [2, 0, 1])
    for observed, expected in zip(result[0][0]['spikes'], [[200.0, 200.0], [], [120.0]]):
        assert np.array_equal(observed, expected)
    current_original = prepare_inputs(raw, 3000.0)
    for key in ('population', 'source_id', 'source_counts', 'source_event_unit', 'source_event_time_ms'):
        assert np.array_equal(current_original[key], baseline[key])
    assert report['merged_events'] == 5 and report['background_events_unchanged']


def test_lgn_replacement_rejects_incomplete_generators(replacement_case):
    raw, _, baseline, origin = replacement_case
    generated = prepare_inputs([(dict(ids=np.array([0, 1]), spikes=[np.array([]), np.array([])]), [])], 3000.0)
    with pytest.raises(ValueError, match='COMPLETE_GENERATOR_INVENTORY'):
        replace_lgn_inputs(raw, generated, baseline, origin)


def test_lgn_replacement_rejects_another_original_input_block(replacement_case):
    raw, generated, baseline, origin = replacement_case
    baseline['source_event_time_ms'][0] += 0.01
    with pytest.raises(ValueError, match='ORIGINAL_INPUT_BINDING_DIFFERS'):
        replace_lgn_inputs(raw, generated, baseline, origin)


def test_simulator_input_identity_ignores_seed_bookkeeping_but_detects_event_change():
    plan = prepare_inputs([(dict(ids=np.array([0, 1]), spikes=[np.array([1.25]), np.array([])]), [])], 3000.0)
    identity = input_plan_identity(plan)
    plan['source_primary_seed'] = np.array([10, 20])
    assert input_plan_identity(plan) == identity
    plan['source_event_time_ms'][0] = 1.5
    assert input_plan_identity(plan) != identity


def test_input_identity_preserves_physical_ids_across_row_order_changes():
    first = prepare_inputs([(dict(ids=np.array([0, 1]), spikes=[np.array([2.0, 1.0, 1.0]), np.array([])]), [])], 3000.0)
    second = prepare_inputs([(dict(ids=np.array([1, 0]), spikes=[np.array([]), np.array([1.0, 1.0, 2.0])]), [])], 3000.0)
    assert input_plan_identity(first) == input_plan_identity(second)
    second['source_event_time_ms'][1] = 1.125
    assert input_plan_identity(first) != input_plan_identity(second)



def test_background_writer_preserves_the_shared_plan_and_sonata_contract(tmp_path):
    seeds = [dict(source_id=0, primary_seed=7, poisson_seed=None, sampler='bernoulli')]
    result = write_spike_trial(np.array([0]), [np.array([.125, .5])], seeds, 1., tmp_path/'bkg', population_name='bkg')
    assert result['sonata_metadata']['source_population_labels'] == ['bkg']
    assert result['event_comparison']['all_event_pairs_exact']
    with h5py.File(tmp_path/'bkg.h5') as handle:
        assert list(handle['spikes']) == ['bkg']
        assert np.array_equal(handle['spikes/bkg/timestamps'][:], [.125, .5])
    with pytest.raises(ValueError, match='POPULATION_NAME_INVALID'):
        write_spike_trial(np.array([0]), [np.array([.5])], seeds, 1., tmp_path/'bad', population_name='v1')
    assert not (tmp_path/'bad.h5').exists()


@pytest.mark.parametrize('constructed', [False, True])
def test_spike_export_preserves_rate_construction_binding(tmp_path, monkeypatch, constructed):
    archive = tmp_path/'rates'; (archive/'probe').mkdir(parents=True); (archive/'prepared').mkdir()
    (archive/'prepared/prepared.json').write_text('{}')
    config = dict(scope='development_source_runtime_qualification', unit_coverage='all_source_units',
                  stimulus=dict(duration_s=.002), spike_generation=dict(trials=1, starting_noise_seed=123))
    config_path = tmp_path/'config.yaml'; config_path.write_text(yaml.safe_dump(config))
    reference = dict(status='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT', config=config,
                     time_units='s', firing_rate_units='Hz', source_unit_ids=[3, 8], source_units=2, rate_shape=[2, 3])
    if constructed:
        reference['input_construction_binding'] = dict(delivery_sha256='a'*64, mode='canary',
            files={name: dict(bytes=1, sha256='b'*64) for name in CONSTRUCTION_FILES})
    path = archive/'probe/probe.json'; path.write_text(json.dumps(reference))
    rate = archive/'probe/firing_rates_f_tot.h5'
    with h5py.File(rate, 'w') as handle:
        handle['time'] = np.array([0., .001, .002]); handle['firing_rates_Hz'] = np.full((2, 3), 5.)
    (archive/'delivery.json').write_text(json.dumps(dict(exit_code=0, files=[dict(
        path=str(item.relative_to(archive)), bytes=item.stat().st_size, sha256=sha256_file(item)) for item in [path, rate]])))
    functions = SimpleNamespace(f_rate_to_spike_train=lambda *args: np.array([.5]))
    poisson = SimpleNamespace(generate_inhomogenous_poisson=Mock(side_effect=AssertionError('unused Poisson branch')))
    monkeypatch.setattr(spikes_module, 'load_prepared_runtime', lambda _: (functions, poisson))
    result = spikes_module.export_source_spikes(archive, config_path, tmp_path/'spikes')
    assert result['status'] == 'SOURCE_SPIKES_EXPORTED_EXACT'
    if constructed:
        assert result['input_construction_binding'] == reference['input_construction_binding']
    else:
        assert 'input_construction_binding' not in result
    assert result['source_fidelity_qualified'] is False and result['production_episodes'] == 0

"""Bound input bundles retain silence, exact timestamps, and construction identity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import h5py
import numpy as np
import pandas as pd
import yaml
import pytest

from .provenance import sha256_file
from . import v1_circuit_stimulus as stimulus_module
from . import v1_lgn_spikes as spikes_module
from .v1_circuit_stimulus import (COMPLETE, REPRESENTATIVES, background_rates, load_inputs)
from .v1_lgn_spikes import write_spike_trial
from .v1_nest_time_semantics import V1_DURATION_MS, V1_RESOLUTION_MS


def test_background_uses_declared_clock_and_rate():
    clock, rates = background_rates(dict(kind='source_constant_rate_sampler',
        rate_clock_step_ms=.25, rate_hz=1000., starting_noise_seed=12346), 3000.)
    assert clock.shape == (12001,) and clock[0] == 0. and clock[-1] == 3.
    assert rates.shape == (1, 12001) and np.all(rates == 1000.)


@pytest.mark.parametrize('change', [{'rate_hz': float('nan')}, {'rate_hz': -1},
    {'rate_hz': True}, {'rate_clock_step_ms': 1.}, {'starting_noise_seed': True}])
def test_background_rejects_invalid_source_specification(change):
    settings=dict(kind='source_constant_rate_sampler', rate_clock_step_ms=.25,
                  rate_hz=1000., starting_noise_seed=12346);settings.update(change)
    with pytest.raises(ValueError, match='SPECIFICATION_INVALID'):
        background_rates(settings, 3000.)


def bundle(tmp_path, full=False):
    root=tmp_path/'visual_inputs';root.mkdir()
    binding=dict(delivery_sha256='a'*64, mode='canary', files={'lgn_nodes': {'sha256':'b'*64}})
    populations={}
    for name,n in [('lgn',17400 if full else 14),('bkg',1)]:
        dest=root/name;dest.mkdir()
        ids=np.arange(n,dtype=np.uint64 if name=='bkg' else np.int64)
        trains=[np.array([.125,.5]) if i==0 else np.array([],dtype=np.float64) for i in range(n)]
        seeds=[dict(source_id=int(i),primary_seed=123,poisson_seed=None,sampler='bernoulli') for i in ids]
        row=write_spike_trial(ids,trains,seeds,3000.,dest/'trial_0000',population_name=name)
        populations[name]=dict(row,plan=name+'/'+row['plan'],sonata=name+'/'+row['sonata'])
    report=dict(status=COMPLETE if full else REPRESENTATIVES,exit_code=0,
        input_construction_binding=binding,complete_lgn_population=full,
        source_clock=dict(duration_ms=V1_DURATION_MS,resolution_ms=V1_RESOLUTION_MS),
        populations=populations,artifacts={p.relative_to(root).as_posix():sha256_file(p)
                                           for p in root.rglob('*') if p.is_file()})
    (root/'report.json').write_text(json.dumps(report))
    refresh_delivery(tmp_path)
    return binding,report


def refresh_delivery(root):
    paths=[p for p in (root/'visual_inputs').rglob('*') if p.is_file()]
    (root/'delivery.json').write_text(json.dumps(dict(exit_code=0,files=[
        dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in paths])))


def test_representatives_cannot_be_used_as_full_native_population(tmp_path):
    binding,_=bundle(tmp_path)
    with pytest.raises(ValueError,match='COMPLETE_VISUAL_POPULATION_REQUIRED'):
        load_inputs(tmp_path,binding)
    plan,_=load_inputs(tmp_path,binding,require_complete=False)
    assert len(plan['source_id'])==15 and plan['source_counts'].tolist()==[2]+[0]*13+[2]


def test_full_bundle_preserves_population_overlap_silence_and_times(tmp_path):
    binding,_=bundle(tmp_path,full=True)
    plan,report=load_inputs(tmp_path,binding)
    assert len(plan['source_id'])==17401 and len(plan['trains'])==17401
    assert plan['source_id'].dtype == np.dtype(np.int64)
    assert plan['source_id'][0]==plan['source_id'][-1]==0
    assert plan['population'][-1]==1 and (plan['source_counts']==0).sum()==17399
    assert plan['source_event_time_ms'].tolist()==[.125,.5,.125,.5]
    assert len(report['input_identity_sha256'])==64


def test_same_ids_with_changed_construction_are_rejected(tmp_path):
    binding,_=bundle(tmp_path)
    changed=deepcopy(binding);changed['files']['lgn_nodes']['sha256']='c'*64
    with pytest.raises(ValueError,match='BUNDLE_BINDING_DIFFERS'):
        load_inputs(tmp_path,changed,require_complete=False)


def test_sonata_units_cannot_be_silently_rescaled(tmp_path):
    binding,report=bundle(tmp_path)
    target=tmp_path/'visual_inputs/bkg/trial_0000.h5'
    with h5py.File(target,'r+') as h: h['spikes/bkg/timestamps'].attrs['units']='s'
    report['artifacts']['bkg/trial_0000.h5']=sha256_file(target)
    (tmp_path/'visual_inputs/report.json').write_text(json.dumps(report))
    refresh_delivery(tmp_path)
    with pytest.raises(ValueError,match='TIME_UNIT_DIFFERS'):
        load_inputs(tmp_path,binding,require_complete=False)


def test_changed_plan_receipt_is_rejected(tmp_path):
    binding,_=bundle(tmp_path)
    with (tmp_path/'visual_inputs/lgn/trial_0000.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='ARCHIVE_CHANGED'):
        load_inputs(tmp_path,binding,require_complete=False)


@pytest.fixture
def generated_case(tmp_path, monkeypatch):
    """Exercise export and serialization with small declared software fixtures."""
    config = yaml.safe_load((Path(__file__).parents[1]/'configs'/
        'neural_population_v1_constructed_visual_development.yaml').read_text())
    rate_root = tmp_path/'rates'
    (rate_root/'probe').mkdir(parents=True)
    (rate_root/'prepared').mkdir()
    (rate_root/'prepared/prepared.json').write_text('{"software_fixture": true}')
    binding = dict(delivery_sha256='a'*64, mode='software_fixture', files={})
    cells = pd.DataFrame({'model_id': ['a', 'b', 'c']}, index=[10, 20, 30])
    rates = np.array([[.01]*13, [1000.]*13, [.01]*13], dtype=np.float64)
    with h5py.File(rate_root/'probe/firing_rates_f_tot.h5', 'w') as dest:
        dest['time'] = np.linspace(0., 3., 13)
        dest['firing_rates_Hz'] = rates
    reference = dict(status='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT',
        config=deepcopy(config), input_construction_binding=deepcopy(binding),
        source_unit_ids=cells.index.tolist(), source_models=cells['model_id'].tolist(),
        source_units=3, rate_shape=[3, 13], time_units='s', firing_rate_units='Hz')

    def primary(time_ms, rates, seed, start, end, precision):
        return np.array([.125 + seed/10000.], dtype=np.float64)

    def poisson(clock, rates, seed):
        # Fixture retains the source fallback's global NumPy reseeding behavior.
        np.random.seed(seed)
        return np.array([.5 + np.random.random()*.25], dtype=np.float64)

    runtime = (SimpleNamespace(f_rate_to_spike_train=primary),
               SimpleNamespace(generate_inhomogenous_poisson=poisson))
    for module in (stimulus_module, spikes_module):
        monkeypatch.setattr(module, 'bound_json', lambda root, relative:
            (json.loads((root/relative).read_text()), {'exit_code': 0}))
        monkeypatch.setattr(module, 'delivered_file', lambda root, relative: root/relative)
        monkeypatch.setattr(module, 'load_prepared_runtime', lambda root: runtime)
    monkeypatch.setattr(stimulus_module, 'read_input_construction_cells',
                        lambda root: (cells, None, binding))
    monkeypatch.setattr(stimulus_module, 'read_input_inventory',
                        lambda *args: (np.array([0], dtype=np.uint64), None))

    def run(requested=None, *, name='inputs'):
        (rate_root/'probe/probe.json').write_text(json.dumps(reference))
        config_path = tmp_path/(name+'.yaml')
        config_path.write_text(yaml.safe_dump(config if requested is None else requested))
        return stimulus_module.generate_inputs(rate_root, tmp_path/'construction',
                                               config_path, tmp_path/name)

    return SimpleNamespace(config=config, reference=reference, binding=binding,
                           run=run, rate_file=rate_root/'probe/firing_rates_f_tot.h5')


@pytest.mark.parametrize('lgn_seed,bkg_seed', [(12345, 12346), (22345, 12346),
                                               (12345, 22346), (22345, 22346)])
def test_source_noise_changes_reuse_rates_and_keep_bound_identities(generated_case, lgn_seed, bkg_seed):
    case = generated_case
    rate_hash = sha256_file(case.rate_file)
    baseline = case.run(name='baseline')
    config = deepcopy(case.config)
    config['spike_generation']['starting_noise_seed'] = lgn_seed
    config['background']['starting_noise_seed'] = bkg_seed
    report = case.run(config, name='requested')
    assert report['config'] == config
    assert report['rate_config'] == case.config
    assert case.reference['config'] == case.config
    assert report['rate_report_sha256'] == baseline['rate_report_sha256']
    assert report['input_construction_binding'] == case.binding
    assert report['lgn_noise_seed'] == lgn_seed and report['background_noise_seed'] == bkg_seed
    assert sha256_file(case.rate_file) == rate_hash
    for name, changed in [('lgn', lgn_seed != 12345), ('bkg', bkg_seed != 12346)]:
        assert (report['populations'][name]['input_identity_sha256'] !=
                baseline['populations'][name]['input_identity_sha256']) is changed
    assert report['scientific_qualification'] is False
    assert report['production_episodes'] == 0


@pytest.mark.parametrize('path,value', [
    (('stimulus', 'direction_deg'), 270.), (('stimulus', 'duration_s'), 2.),
    (('background', 'rate_clock_step_ms'), .5), (('background', 'rate_hz'), 999.),
    (('unit_coverage',), 'model_representatives'), (('converter_python',), '/different'),
    (('spike_generation', 'trials'), 2), (('extra_field',), 'unbound')])
def test_noise_reuse_rejects_every_other_configuration_change(generated_case, path, value):
    config = deepcopy(generated_case.config)
    config['spike_generation']['starting_noise_seed'] = 22345
    config['background']['starting_noise_seed'] = 22346
    target = config
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match='RATE_BINDING_DIFFERS'):
        generated_case.run(config)


@pytest.mark.parametrize('changed', ['construction', 'unit_ids', 'unit_models'])
def test_noise_reuse_rejects_circuit_or_unit_inventory_drift(generated_case, changed):
    case = generated_case
    if changed == 'construction': case.binding['delivery_sha256'] = 'b'*64
    elif changed == 'unit_ids': case.reference['source_unit_ids'][0] = 11
    else: case.reference['source_models'][0] = 'other'
    with pytest.raises(ValueError, match='BINDING_DIFFERS'):
        case.run()


@pytest.mark.parametrize('location', ['requested', 'rate'])
@pytest.mark.parametrize('section', ['spike_generation', 'background'])
@pytest.mark.parametrize('value', [None, True, 12345., '12345', 'absent'])
def test_noise_reuse_requires_both_explicit_integer_seeds(generated_case, location, section, value):
    config = deepcopy(generated_case.config)
    target = config if location == 'requested' else generated_case.reference['config']
    if value == 'absent': del target[section]['starting_noise_seed']
    else: target[section]['starting_noise_seed'] = value
    with pytest.raises(ValueError, match='INTEGER_NOISE_SEEDS_REQUIRED'):
        generated_case.run(config)


@pytest.mark.parametrize('value', [True, 1.0])
def test_noise_reuse_does_not_coerce_single_trial(generated_case, value):
    config = deepcopy(generated_case.config)
    config['spike_generation']['trials'] = value
    with pytest.raises(ValueError, match='RATE_BINDING_DIFFERS'):
        generated_case.run(config)


def test_noise_development_configuration_changes_only_the_declared_seeds():
    configs = Path(__file__).parents[1]/'configs'
    original = yaml.safe_load((configs/'neural_population_v1_constructed_visual_development.yaml').read_text())
    changed = yaml.safe_load((configs/'neural_population_v1_noise_development.yaml').read_text())
    assert changed['spike_generation']['starting_noise_seed'] == 22345
    assert changed['background']['starting_noise_seed'] == 22346
    changed['spike_generation']['starting_noise_seed'] = original['spike_generation']['starting_noise_seed']
    changed['background']['starting_noise_seed'] = original['background']['starting_noise_seed']
    assert changed == original

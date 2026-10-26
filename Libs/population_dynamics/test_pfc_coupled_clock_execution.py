"""Physical controls, source fields, provenance labels and the real CLI."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from . import pfc_coupled_clock_execution as adapter
from .pfc_coupled_clock_execution import current_agreement, field_observations, retag_engine_metadata, select_settings, case_grid


def test_current_comparison_uses_physical_clock_and_all_delivered_values():
    coarse=dict(segment_start_ticks=np.array([0,20]),segment_stop_ticks=np.array([20,100]),
                current_A=np.array([[0.,0.],[1e-9,0.]]),cue_active=np.array([False,True]))
    fine={key:value.copy() for key,value in coarse.items()}
    for key in ('segment_start_ticks','segment_stop_ticks'):fine[key]*=2
    assert all(current_agreement(fine,coarse,.025,.05).values())
    fine['current_A'][1,0]=2e-9
    assert current_agreement(fine,coarse,.025,.05)['current_A'] is False


def test_complete_source_fields_preserve_empty_bins_units_and_spike_totals():
    config=dict(excitatory_neurons=8,inhibitory_neurons=2,excitatory_preference_bins=2,
                readout_bin_ms=5.,dt_ms=.05,duration_ms=15.)
    raw=dict(spike_ticks=np.array([0,99,100,100],dtype=np.int64),
             spike_unit_indices=np.array([0,7,8,9],dtype=np.int64))
    fields=field_observations(raw,config)
    np.testing.assert_array_equal(fields['field_counts'],[[1,1,0],[0,0,2],[0,0,0]])
    np.testing.assert_array_equal(fields['field_unit_counts'],[4,4,2])
    np.testing.assert_allclose(fields['rates_hz'],[[50,50,0],[0,0,200],[0,0,0]])
    raw['spike_unit_indices'][0]=10
    with pytest.raises(ValueError,match='VALID_SOURCE_FIELD_EVENTS_REQUIRED'):field_observations(raw,config)


def test_driver_provenance_is_not_mislabeled_as_native_fine_rng():
    result={'inputs_used':{'input_mode':'native_numpy_stream'},
            'cost':{'native_input_generation_seconds':1.25,'arithmetic_path':{'external_inputs':'native label'}}}
    retag_engine_metadata(result)
    assert result['inputs_used']['input_mode']=='coupled_binomial_clock_refinement_stream'
    assert result['inputs_used']['engine_input_mode_tag']=='native_numpy_stream'
    assert result['cost']['coupled_input_generation_seconds']==1.25
    assert 'native_input_generation_seconds' not in result['cost']


def test_real_cli_rejects_nonpilot_scope_without_outputs(tmp_path):
    config=tmp_path/'bad.yaml';config.write_text('schema: wrong\n')
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_coupled_clock_execution',
        '--config',str(config),'--project-root',str(tmp_path),'--output-root',str(tmp_path/'outputs'),
        '--validate-inputs-only'],capture_output=True,text=True)
    assert result.returncode!=0 and 'DEVELOPMENT_COUPLED_CLOCK_PILOT_REQUIRED' in result.stderr
    assert not (tmp_path/'outputs').exists()


def calibration_settings():
    return dict(schema='pfc_coupled_clock_calibration_v1',scope='development_paired_clock_calibration',
        source_rng_seeds=list(range(95002,95033)),reused_source_rng_seed=95001,
        confirmation_execution_released=False,conditions=['cue','omission','strong_cue'],
        coarse_dt_ms=.05,fine_dt_ms=[.025,.0125],
        blocks=[dict(source_rng_seed=seed,refinement_seeds=[95411+2*(seed-95001),95412+2*(seed-95001)],
            coarse_references={c:{'archive':f'{seed}/{c}'} for c in ['cue','omission','strong_cue']})
            for seed in range(95002,95033)])


def test_calibration_selection_preserves_complete_grid_distinct_streams_and_source_identity():
    settings=calibration_settings();original=copy.deepcopy(settings);aux=[]
    for seed in range(95002,95033):
        block=select_settings(settings,seed)
        assert block['source_rng_seed']==seed and block['coarse_references']['cue']['archive']==f'{seed}/cue'
        assert case_grid(block)==[(d,c) for d in [.025,.0125] for c in ['cue','omission','strong_cue']]
        aux.extend(block['refinement_seeds'])
    assert settings==original and len(set(aux))==62 and min(aux)==95413 and max(aux)==95474


@pytest.mark.parametrize('change,message',[
    ('missing','COMPLETE_UNIQUE_ORDERED_CALIBRATION_BLOCKS_REQUIRED'),
    ('auxiliary','REGISTERED_AUXILIARY_SEEDS_AND_ALL_CONDITIONS_REQUIRED'),
    ('condition','REGISTERED_AUXILIARY_SEEDS_AND_ALL_CONDITIONS_REQUIRED'),
    ('confirmation','FIXED_REMAINING_CALIBRATION_BLOCKS_REQUIRED')])
def test_calibration_selection_rejects_unregistered_design_changes(change,message):
    settings=calibration_settings()
    if change=='missing':settings['blocks'].pop()
    elif change=='auxiliary':settings['blocks'][0]['refinement_seeds']=[95411,95412]
    elif change=='condition':settings['blocks'][0]['coarse_references'].pop('omission')
    else:settings['confirmation_execution_released']=True
    with pytest.raises(ValueError,match=message):select_settings(settings,95002)


@pytest.mark.parametrize('seed',[None,95001,95033,96001])
def test_calibration_selection_cannot_repeat_pilot_or_consume_confirmation(seed):
    with pytest.raises(ValueError,match='REGISTERED_CALIBRATION_BLOCK_SELECTION_REQUIRED'):
        select_settings(calibration_settings(),seed)


def recurrent_gain_settings(schema=adapter.GAIN_SCHEMA):
    """Synthetic registration fixture; no source result is synthesized."""
    _, scope, cohorts = adapter.GAIN_REGISTRIES[schema]
    settings = dict(schema=schema, scope=scope,
        confirmation_execution_released=False, conditions=['cue', 'omission', 'strong_cue'],
        coarse_dt_ms=.05, fine_dt_ms=[.025, .0125], kernel_parameters=dict(adapter.GAIN_KERNEL),
        fine_clock_qualifications={str(dt): {'archive': 'synthetic_qualification/'+str(dt)}
                                   for dt in [.025, .0125]}, cohorts={})
    for cohort, (ne, ni, seeds, first) in cohorts.items():
        settings['cohorts'][cohort] = dict(excitatory_neurons=ne, inhibitory_neurons=ni,
            source_rng_seeds=seeds[:], blocks=[dict(source_rng_seed=seed,
                refinement_seeds=[first+2*i, first+2*i+1],
                coarse_references={condition: {'archive': f'synthetic/{cohort}/{seed}/{condition}'}
                                   for condition in settings['conditions']}) for i, seed in enumerate(seeds)])
    return settings


def test_recurrent_gain_selection_keeps_sizes_seeds_controls_and_auxiliaries_separate():
    settings = recurrent_gain_settings(); original = copy.deepcopy(settings); auxiliaries = []
    for cohort, (ne, ni, seeds, first) in adapter.GAIN_COHORTS.items():
        for index, seed in enumerate(seeds):
            selected = select_settings(settings, seed, cohort)
            assert selected['source_rng_seed'] == seed and selected['cohort'] == cohort
            assert selected['excitatory_neurons'] == ne and selected['inhibitory_neurons'] == ni
            assert selected['refinement_seeds'] == [first+2*index, first+2*index+1]
            assert case_grid(selected) == [(dt, c) for dt in [.025, .0125]
                                           for c in ['cue', 'omission', 'strong_cue']]
            auxiliaries.extend(selected['refinement_seeds'])
    assert settings == original and len(set(auxiliaries)) == 12
    assert not set(auxiliaries) & set(range(96001, 96129))


@pytest.mark.parametrize('change,message', [
    ('missing_cohort', 'COMPLETE_RECURRENT_GAIN_COHORTS_REQUIRED'),
    ('confirmation', 'COMPLETE_RECURRENT_GAIN_COHORTS_REQUIRED'),
    ('missing_block', 'EXACT_THREE_RECURRENT_GAIN_BLOCKS_PER_SIZE_REQUIRED'),
    ('seed', 'EXACT_THREE_RECURRENT_GAIN_BLOCKS_PER_SIZE_REQUIRED'),
    ('size', 'EXACT_THREE_RECURRENT_GAIN_BLOCKS_PER_SIZE_REQUIRED'),
    ('auxiliary', 'REGISTERED_RECURRENT_GAIN_AUXILIARY_SEEDS_REQUIRED'),
    ('kernel', 'REGISTERED_RECURRENT_GAIN_SIZE_SEED_AND_KERNEL_REQUIRED'),
    ('conditions', 'COMPLETE_RECURRENT_GAIN_CLOCK_GRID_REQUIRED'),
    ('fine_clock', 'COMPLETE_RECURRENT_GAIN_CLOCK_GRID_REQUIRED'),
    ('qualification', 'EXPLICIT_FINE_CLOCK_QUALIFICATION_ARCHIVES_REQUIRED')])
def test_recurrent_gain_registration_rejects_changed_scientific_design(change, message):
    settings = recurrent_gain_settings(); native = settings['cohorts']['reference_2560']
    if change == 'missing_cohort': settings['cohorts'].pop('expanded_100k')
    elif change == 'confirmation': settings['confirmation_execution_released'] = True
    elif change == 'missing_block': native['blocks'].pop()
    elif change == 'seed': native['source_rng_seeds'][0] = 96001
    elif change == 'size': native['excitatory_neurons'] = 80000
    elif change == 'auxiliary': native['blocks'][0]['refinement_seeds'] = [95411, 95412]
    elif change == 'kernel': settings['kernel_parameters']['Jpos_excit2excit'] = 1.6
    elif change == 'conditions': settings['conditions'].pop()
    elif change == 'fine_clock': settings['fine_dt_ms'] = [.025]
    else: settings['fine_clock_qualifications'].pop('0.0125')
    with pytest.raises(ValueError, match=message): select_settings(settings, 93001, 'reference_2560')


@pytest.mark.parametrize('cohort,seed', [(None, None), ('reference_2560', 95001),
    ('expanded_100k', 93001), ('reference_2560', 96001), ('reference_2560', 93004),
    ('expanded_100k', 95004), ('unknown', 93001)])
def test_recurrent_gain_selection_rejects_cross_size_or_reserved_seeds(cohort, seed):
    with pytest.raises(ValueError, match='REGISTERED_RECURRENT_GAIN_COHORT_AND_BLOCK_SELECTION_REQUIRED'):
        select_settings(recurrent_gain_settings(), seed, cohort)


def test_original_pilot_still_selects_six_cases_without_block_override():
    settings = dict(schema='pfc_coupled_clock_pilot_v1', scope='development_paired_clock_pilot',
        source_rng_seed=95001, confirmation_execution_released=False,
        conditions=['cue', 'omission', 'strong_cue'], coarse_dt_ms=.05, fine_dt_ms=[.025, .0125],
        coarse_references={c: {} for c in ['cue', 'omission', 'strong_cue']})
    assert select_settings(settings) is settings and len(case_grid(settings)) == 6
    with pytest.raises(ValueError, match='PILOT_BLOCK_OVERRIDE_FORBIDDEN'): select_settings(settings, 95001)
    with pytest.raises(ValueError, match='LEGACY_COHORT_OVERRIDE_FORBIDDEN'):
        select_settings(settings, cohort='expanded_100k')


def synthetic_qualification(root, settings, dt_ms, change=None):
    """Explicitly synthetic receipts exercise the real candidate qualification gate."""
    if not hasattr(adapter.native_execution, 'kernel_parameters'):
        pytest.skip('candidate source snapshot required for recurrent-gain qualification tests')
    native = adapter.native_execution
    receipt = dict(fixture_kind='synthetic_qualification_unit_fixture', scientific_evidence=False,
        status='PASS_NATIVE_STREAM_SOURCE_EXECUTION', source_entry_sha256=adapter.native_drive.SOURCE_ENTRY_SHA256,
        backend='numpy', numpy_version=np.__version__,
        executor_sha256=adapter.sha256_file(Path(adapter.executor.__file__)),
        drive_sha256=adapter.sha256_file(Path(adapter.native_drive.__file__)),
        input_replay_status='PASS_NATIVE_NUMPY_DRIVE_REPLAY', input_replay_passed_comparisons=24,
        global_cases_passed=2, global_arrays_passed=10, validated_dt_ms=[dt_ms],
        validated_kernel_parameters=[dict(adapter.GAIN_KERNEL)],
        kernel_parameterization_sha256=adapter.sha256_file(Path(native.kernel_parameters.__file__)))
    if change == 'failed': receipt['status'] = 'FAILED_SYNTHETIC_UNIT_FIXTURE'
    elif change == 'clock': receipt['validated_dt_ms'] = [.05]
    elif change == 'kernel': receipt['validated_kernel_parameters'][0]['Jpos_excit2excit'] = 1.6
    elif change == 'numpy': receipt['numpy_version'] = 'synthetic_incompatible_numpy'
    elif change == 'executor': receipt['executor_sha256'] = '0'*64
    directory = root/settings['fine_clock_qualifications'][str(dt_ms)]['archive']
    path = directory/'comparison/qualification.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt)+'\n')
    delivery = dict(fixture_kind='synthetic_qualification_unit_fixture', scientific_evidence=False,
        status='ARCHIVED_PFC_STREAM_VALIDATION', result_status='PASS_NATIVE_STREAM_SOURCE_EXECUTION',
        files=[dict(path='comparison/qualification.json', bytes=path.stat().st_size,
                    sha256=adapter.sha256_file(path))])
    if change == 'delivery': delivery['result_status'] = 'FAILED_SYNTHETIC_UNIT_FIXTURE'
    (directory/'delivery.json').write_text(json.dumps(delivery)+'\n')
    return receipt


@pytest.mark.parametrize('dt_ms', [.025, .0125])
def test_synthetic_qualification_exercises_exact_clock_and_kernel_gate(tmp_path, dt_ms):
    settings = select_settings(recurrent_gain_settings(), 93001, 'reference_2560')
    synthetic_qualification(tmp_path, settings, dt_ms)
    result = adapter.qualify_fine_clock(settings, tmp_path, dt_ms, adapter.GAIN_KERNEL)
    assert result['qualification']['fixture_kind'] == 'synthetic_qualification_unit_fixture'
    assert result['qualification']['scientific_evidence'] is False
    assert result['qualification']['validated_dt_ms'] == [dt_ms]


@pytest.mark.parametrize('change,message', [
    ('missing', 'SUCCESSFUL_FINE_CLOCK_QUALIFICATION_REQUIRED'),
    ('failed', 'PFC_SOURCE_EXECUTION_QUALIFICATION_MISMATCH:status'),
    ('delivery', 'PFC_SOURCE_EXECUTION_QUALIFICATION_DELIVERY_INVALID'),
    ('clock', 'PFC_SOURCE_EXECUTION_CLOCK_NOT_QUALIFIED'),
    ('kernel', 'PFC_SOURCE_EXECUTION_KERNEL_NOT_QUALIFIED'),
    ('numpy', 'PFC_SOURCE_EXECUTION_QUALIFICATION_MISMATCH:numpy_version'),
    ('executor', 'PFC_SOURCE_EXECUTION_QUALIFICATION_MISMATCH:executor_sha256')])
def test_fine_qualification_fails_closed_before_any_source_run(tmp_path, monkeypatch, change, message):
    settings = select_settings(recurrent_gain_settings(), 93001, 'reference_2560')
    if change != 'missing': synthetic_qualification(tmp_path, settings, .025, change)
    def forbidden(*args, **kwargs): raise AssertionError('source execution forbidden in gate test')
    monkeypatch.setattr(adapter.executor, 'simulate', forbidden)
    with pytest.raises(ValueError, match=message):
        adapter.qualify_fine_clock(settings, tmp_path, .025, adapter.GAIN_KERNEL)


@pytest.fixture
def real_gain_registration():
    config = os.environ.get('NEURONSIM_PFC_COUPLED_FIXTURE_CONFIG')
    root = os.environ.get('NEURONSIM_PFC_COUPLED_FIXTURE_ROOT')
    if not config or not root: pytest.skip('staged real coarse archives required')
    return adapter.yaml.safe_load(Path(config).read_text()), Path(root), Path(config)


def test_real_coarse_archive_rejects_numpy_drift(real_gain_registration, monkeypatch):
    registry, root, _ = real_gain_registration
    settings = select_settings(registry, registry['cohorts']['reference_2560']['source_rng_seeds'][0], 'reference_2560')
    monkeypatch.setattr(np, '__version__', 'synthetic_incompatible_numpy')
    with pytest.raises(ValueError, match='COARSE_SOURCE_NUMPY_VERSION_CHANGED'):
        adapter.load_coarse_inputs(settings, root, 'cue')


def test_real_coarse_archive_rejects_cross_size_pairing(real_gain_registration):
    registry, root, _ = real_gain_registration
    settings = select_settings(registry, registry['cohorts']['reference_2560']['source_rng_seeds'][0], 'reference_2560')
    other = select_settings(registry, registry['cohorts']['expanded_100k']['source_rng_seeds'][0], 'expanded_100k')
    settings['coarse_references'] = dict(settings['coarse_references'], cue=other['coarse_references']['cue'])
    with pytest.raises(ValueError, match='FROZEN_COMPLETE_REGISTERED_COARSE_CASE_REQUIRED'):
        adapter.load_coarse_inputs(settings, root, 'cue')


@pytest.mark.parametrize('cohort,seed', [('reference_2560', 93001), ('expanded_100k', 95001)])
def test_real_execution_cli_refuses_unstaged_fine_qualification(real_gain_registration, tmp_path, cohort, seed):
    registry, root, config = real_gain_registration
    seed = registry['cohorts'][cohort]['source_rng_seeds'][0]
    # Test mode deliberately stages coarse anchors only, even if future fine jobs finish.
    selected = select_settings(registry, seed, cohort)
    assert not (root/selected['fine_clock_qualifications']['0.025']['archive']/'delivery.json').exists()
    output = tmp_path/'forbidden_source_outputs'
    command = [sys.executable, '-B', '-m', 'Libs.population_dynamics.pfc_coupled_clock_execution',
        '--config', str(config), '--project-root', str(root), '--output-root', str(output),
        '--cohort', cohort, '--source-rng-seed', str(seed), '--case-index', '0']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'SUCCESSFUL_FINE_CLOCK_QUALIFICATION_REQUIRED' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('ne,ni', [(2048, 512), (80000, 20000)])
def test_full_unit_readouts_use_actual_cohort_dimensions(tmp_path, ne, ni):
    # Sparse synthetic events test real readout consumers; no simulator is called.
    config = dict(excitatory_neurons=ne, inhibitory_neurons=ni, excitatory_preference_bins=16,
                  dt_ms=.025, readout_bin_ms=5., duration_ms=5000.)
    raw = dict(spike_ticks=np.array([0, 4000, 4000, 8000], dtype=np.int64),
               spike_unit_indices=np.array([0, ne-1, ne, ne+ni-1], dtype=np.int64))
    result = adapter.write_source_readouts(tmp_path, raw, config, 'synthetic_readout', 93001, 'cue')
    assert result['temporal_bins'] == 1000 and result['scalar_phase_checks'] > 0
    with (tmp_path/'phase_responses.csv').open() as stream: phases = list(csv.DictReader(stream))
    assert phases and all(int(row['units']) == ne+ni and int(row['excitatory_neurons']) == ne
                          and int(row['inhibitory_neurons']) == ni for row in phases)
    with np.load(tmp_path/'phase_unit_counts.npz', allow_pickle=False) as stored:
        assert all(stored[key].shape == (ne+ni,) for key in stored.files)
    with (tmp_path/'temporal_readouts.csv').open() as stream: temporal = list(csv.DictReader(stream))
    assert len(temporal) == 1000
    assert sum(int(row['excitatory_spikes']) for row in temporal) == 2
    assert sum(int(row['inhibitory_spikes']) for row in temporal) == 2
    with np.load(tmp_path/'field_observations.npz', allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['field_unit_counts'], [ne//16]*16+[ni])
        assert stored['field_counts'].sum() == 4 and stored['rates_hz'][0, 0] == 1/(ne/16*.005)
        assert stored['rates_hz'][20, 16] == 1/(ni*.005)



def test_recurrent_gain_replication_fills_only_missing_clocks_for_all_existing_blocks():
    registry = recurrent_gain_settings(adapter.GAIN_REPLICATES_SCHEMA)
    original = copy.deepcopy(registry)
    selected = [select_settings(registry, seed, cohort)
                for cohort, bank in registry['cohorts'].items() for seed in bank['source_rng_seeds']]
    assert len(selected) == 58 and sum(len(case_grid(block)) for block in selected) == 348
    assert registry == original
    for cohort, first_seed, first_aux, final_aux in [
            ('reference_2560', 93004, 98717, 98774), ('expanded_100k', 95004, 95417, 95474)]:
        group = [block for block in selected if block['cohort'] == cohort]
        assert [block['source_rng_seed'] for block in group] == list(range(first_seed, first_seed+29))
        assert group[0]['refinement_seeds'] == [first_aux, first_aux+1]
        assert group[-1]['refinement_seeds'] == [final_aux-1, final_aux]
        assert all(block['schema'] == adapter.GAIN_REPLICATES_BLOCK_SCHEMA for block in group)
        assert all(block['kernel_parameters'] == adapter.GAIN_KERNEL for block in group)
        assert all(block['scope'] == adapter.GAIN_REPLICATES_SCOPE for block in group)


@pytest.mark.parametrize('cohort,seed', [('reference_2560', 93001), ('expanded_100k', 95003),
    ('reference_2560', 95004), ('expanded_100k', 93004), ('reference_2560', 96001)])
def test_recurrent_gain_replication_rejects_pilot_repetition_cross_size_and_confirmation(cohort, seed):
    with pytest.raises(ValueError, match='REGISTERED_RECURRENT_GAIN_COHORT_AND_BLOCK_SELECTION_REQUIRED'):
        select_settings(recurrent_gain_settings(adapter.GAIN_REPLICATES_SCHEMA), seed, cohort)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'auxiliary', 'scope', 'condition', 'confirmation'])
def test_recurrent_gain_replication_cannot_select_an_outcome_dependent_subset(change):
    registry = recurrent_gain_settings(adapter.GAIN_REPLICATES_SCHEMA)
    blocks = registry['cohorts']['reference_2560']['blocks']
    if change == 'missing': blocks.pop()
    elif change == 'duplicate': blocks[-1] = copy.deepcopy(blocks[0])
    elif change == 'auxiliary': blocks[0]['refinement_seeds'] = [98711, 98712]
    elif change == 'scope': registry['scope'] = adapter.GAIN_SCOPE
    elif change == 'condition': registry['conditions'].remove('omission')
    else: registry['confirmation_execution_released'] = True
    with pytest.raises(ValueError): select_settings(registry, 93004, 'reference_2560')


@pytest.mark.parametrize('dt_ms', [.025, .0125])
def test_recurrent_gain_replication_keeps_source_clock_qualification_gate(tmp_path, dt_ms):
    block = select_settings(recurrent_gain_settings(adapter.GAIN_REPLICATES_SCHEMA), 95004, 'expanded_100k')
    synthetic_qualification(tmp_path, block, dt_ms)
    result = adapter.qualify_fine_clock(block, tmp_path, dt_ms, adapter.GAIN_KERNEL)
    assert result['qualification']['validated_dt_ms'] == [dt_ms]


def refinement_repeat_settings():
    original=calibration_settings()
    first=dict(source_rng_seed=95001,coarse_references={c:{'archive':'pilot/'+c} for c in original['conditions']})
    blocks=[first]+copy.deepcopy(original['blocks'])
    for i,b in enumerate(blocks):
        b.update(original_refinement_seeds=[95411+2*i,95412+2*i],refinement_seeds=[99411+2*i,99412+2*i])
    return dict(schema=adapter.REFINEMENT_REPEAT_SCHEMA,scope=adapter.REFINEMENT_REPEAT_SCOPE,
        source_rng_seeds=list(range(95001,95033)),auxiliary_rng_seeds=list(range(99411,99475)),
        conditions=original['conditions'],fine_dt_ms=original['fine_dt_ms'],coarse_dt_ms=.05,
        confirmation_execution_released=False,auxiliary_repeat_index=1,new_original_independent_blocks=0,
        original_independent_blocks=32,new_development_episodes=192,maximum_reserved_confirmation_blocks=128,
        production_episodes=0,blocks=blocks)


def test_auxiliary_repeat_preserves_all32_original_blocks_without_new_independent_units():
    registry=refinement_repeat_settings();before=copy.deepcopy(registry)
    blocks=[select_settings(registry,seed) for seed in registry['source_rng_seeds']]
    assert len(blocks)==32 and sum(len(case_grid(b)) for b in blocks)==192
    assert registry==before and all(b['new_original_independent_blocks']==0 for b in blocks)
    assert blocks[0]['source_rng_seed']==95001 and blocks[-1]['source_rng_seed']==95032
    assert blocks[0]['refinement_seeds']==[99411,99412] and blocks[-1]['refinement_seeds']==[99473,99474]


@pytest.mark.parametrize('change',['missing','duplicate','seed_assignment','old_auxiliary','confirmation_seed','conditions','confirmation_release','independent_count','budget'])
def test_auxiliary_repeat_rejects_changed_bank_or_identity(change):
    r=refinement_repeat_settings()
    if change=='missing':r['blocks'].pop()
    elif change=='duplicate':r['blocks'][-1]=copy.deepcopy(r['blocks'][0])
    elif change=='seed_assignment':r['blocks'][0]['refinement_seeds']=[99413,99414]
    elif change=='old_auxiliary':r['auxiliary_rng_seeds'][:2]=[95411,95412];r['blocks'][0]['refinement_seeds']=[95411,95412]
    elif change=='confirmation_seed':r['auxiliary_rng_seeds'][:2]=[96001,96002];r['blocks'][0]['refinement_seeds']=[96001,96002]
    elif change=='conditions':r['conditions']=['cue','strong_cue']
    elif change=='confirmation_release':r['confirmation_execution_released']=True
    elif change=='independent_count':r['new_original_independent_blocks']=32
    else:r['maximum_reserved_confirmation_blocks']=256
    with pytest.raises(ValueError):select_settings(r,95001)


@pytest.mark.parametrize('seed',[None,95000,95033,96001])
def test_auxiliary_repeat_rejects_unregistered_block(seed):
    with pytest.raises(ValueError):select_settings(refinement_repeat_settings(),seed)


def test_array_route_resolves_full_repeat_and_legacy_gain_indices():
    r=refinement_repeat_settings()
    assert [adapter.select_array_seed(r,i) for i in range(32)]==list(range(95001,95033))
    g=recurrent_gain_settings(adapter.GAIN_REPLICATES_SCHEMA)
    assert adapter.select_array_seed(g,0,'expanded_100k')==95004
    assert adapter.select_array_seed(g,28,'expanded_100k')==95032
    for index in [-1,32]:
        with pytest.raises(ValueError):adapter.select_array_seed(r,index)
    with pytest.raises(ValueError):adapter.select_array_seed(r,0,'expanded_100k')
    r['blocks'].pop()
    with pytest.raises(ValueError):adapter.select_array_seed(r,0)


def test_submitted_launcher_array_branch_accepts_repeat_schema():
    import ast
    script=Path('launcher.sbatch')
    if not script.is_file():pytest.skip('staged submitted launcher required')
    text=script.read_text().split("<<'PY_PILOT'\n",1)[1].rsplit('\nPY_PILOT',1)[0]
    tree=ast.parse(text)
    main_try=next(n for n in tree.body if isinstance(n,ast.Try))
    branches=[n for n in main_try.body if isinstance(n,ast.If) and 'SLURM_ARRAY_TASK_ID' in ast.unparse(n.test)]
    assert len(branches)==1
    branch=compile(ast.Module(body=branches,type_ignores=[]),'<submitted-array-selection>','exec')
    context=dict(os=type('Environment',(),{'environ':{'SLURM_ARRAY_TASK_ID':'31'}})(),tests_only=False,
        selected_seed=None,selected_cohort=None,registry=refinement_repeat_settings(),select_array_seed=adapter.select_array_seed)
    exec(branch,context)
    assert context['selected_seed']=='95032'
    context['selected_seed']='95001'
    with pytest.raises(AssertionError):exec(branch,context)


def _temporal_settings():
    return dict(schema=adapter.TEMPORAL_CONTROL_SCHEMA, scope=adapter.TEMPORAL_CONTROL_SCOPE,
        source_rng_seed=95001, refinement_seeds=[95411,95412],
        temporal_transform=adapter.TEMPORAL_CONTROL_TRANSFORM,
        new_original_independent_blocks=0, new_auxiliary_streams=0, new_development_episodes=6,
        maximum_reserved_confirmation_blocks=128, confirmation_execution_released=False,
        production_episodes=0, conditions=['cue','omission','strong_cue'],
        fine_dt_ms=[.025,.0125], coarse_dt_ms=.05,
        coarse_references={name:{} for name in ['cue','omission','strong_cue']},
        original_fine_reference=dict(archive='original_A',delivery_sha256='a'*64))


def test_temporal_control_keeps_complete_single_block_and_original_auxiliary_streams():
    settings = _temporal_settings()
    assert adapter.select_settings(settings) is settings
    assert adapter.case_grid(settings) == [(dt,c) for dt in [.025,.0125]
        for c in ['cue','omission','strong_cue']]


@pytest.mark.parametrize('key,value', [
    ('source_rng_seed',96001), ('refinement_seeds',[99411,99412]),
    ('new_original_independent_blocks',1), ('new_auxiliary_streams',1),
    ('new_development_episodes',192), ('maximum_reserved_confirmation_blocks',129),
    ('confirmation_execution_released',True), ('production_episodes',6),
    ('conditions',['cue','strong_cue']), ('fine_dt_ms',[.0125]),
    ('temporal_transform','arbitrary_permutation'), ('original_fine_reference',{}),
])
def test_temporal_control_rejects_scope_or_lineage_changes(key,value):
    settings = _temporal_settings()
    settings[key] = value
    with pytest.raises(ValueError):
        adapter.case_grid(settings)

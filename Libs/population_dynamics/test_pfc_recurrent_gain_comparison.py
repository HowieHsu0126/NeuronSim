"""Pairing, selected-block estimands, full axes, and descriptive scope."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml
from . import pfc_recurrent_gain_comparison as gain


def paired_reports():
    config={key:1 for key in gain.PHYSICS}
    config.update(excitatory_neurons=32,inhibitory_neurons=8,dt_ms=.05,duration_ms=5000.,
                  stimulus_center_deg=120.,stimulus_width_deg=30.,sigma_weight_profile=20.,Jpos_excit2excit=1.6)
    before=dict(case_id='pfc_93001_cue',resolved_config=config,selected_case=dict(case_id='pfc_93001_cue',source_rng_seed=93001,stimulus_strength_nA=.06),
                dimensions=dict(n_e=32,n_i=8),driver=dict(numpy_version='pinned',seeds={'source_rng_seed':93001}),
                inputs_used=dict(ticks_consumed=100000,E={'events':17}),code=dict(drive_sha256='pinned-drive'))
    after=copy.deepcopy(before);after['resolved_config']['Jpos_excit2excit']=2.3
    return before,after


def test_only_supported_local_kernel_change_with_identical_source_inputs():
    before,after=paired_reports();result=gain.validate_input_pair(before,after)
    assert result['paired_initial_background_input_law'] is True and result['source_rng_seed']==93001
    first=gain.source.kernel_parameters.source_kernel(32,before['resolved_config'])
    second=gain.source.kernel_parameters.source_kernel(32,after['resolved_config'])
    assert first[0]==pytest.approx(1.6) and second[0]==pytest.approx(2.3)
    assert second[16]<first[16]


@pytest.mark.parametrize('change',['clock','size','condition','numpy','drive','events','parameter'])
def test_unmatched_input_or_parameter_cannot_be_paired(change):
    before,after=paired_reports()
    if change=='clock':after['resolved_config']['dt_ms']=.025
    elif change=='size':after['dimensions']['n_e']=80000
    elif change=='condition':after['selected_case']['stimulus_strength_nA']=.5
    elif change=='numpy':after['driver']['numpy_version']='different'
    elif change=='drive':after['code']['drive_sha256']='different'
    elif change=='events':after['inputs_used']['E']['events']=18
    else:after['resolved_config']['sigma_weight_profile']=21.
    with pytest.raises(ValueError):gain.validate_input_pair(before,after)


def test_block_selection_uses_seed_identity_and_recomputes_three_block_variance():
    seeds=[95003,95099,95001,95002]
    indices=gain.selected_blocks(seeds,[95001,95002,95003])
    np.testing.assert_array_equal(indices,[2,3,0])
    counts=np.asarray([[[3]],[[100]],[[0]],[[1]]],dtype=np.int64)
    _,variance=gain.unit_moments(counts[indices]);_,wrong=gain.unit_moments(counts)
    assert variance[0]==pytest.approx(7/3) and variance[0]!=wrong[0]
    with pytest.raises(ValueError):gain.selected_blocks([1,1,2],[1,2,3])


def test_histogram_subset_reindexes_the_original_block_axis():
    data=dict(condition_names=np.asarray(gain.CONDITIONS),phase_names=np.asarray(list(gain.PHASES)),window_ms=np.asarray(gain.WINDOWS),
        source_rng_seeds=np.asarray([95003,95099,95001,95002]),coordinates=np.asarray([[0,i,0,0,0,0]for i in range(4)]),observations=np.asarray([30,990,10,20]))
    selected=gain.select_histogram(data,[95001,95002,95003])
    assert selected=={(0,2,0,0,0,0):30,(0,0,0,0,0,0):10,(0,1,0,0,0,0):20}
    data['condition_names']=np.asarray(['cue','omission','strong_cue'])
    with pytest.raises(ValueError,match='AXES'):gain.select_histogram(data,[95001,95002,95003])


def test_independent_size_effects_do_not_pair_block_order():
    first=np.array([-2.,0.,2.]);second=np.array([-5.,1.,7.])
    result=gain.independent_effect(first,second,[1,2,3],[4,5,6])
    assert result==gain.independent_effect(first,second[::-1],[1,2,3],[6,5,4])
    assert result['expanded_minus_reference_mean_effect']==1 and result['cross_size_blocks_paired'] is False
    with pytest.raises(ValueError,match='INDEPENDENT'):gain.independent_effect(first,second,[1,2,3],[3,4,5])
    assert gain.describe([1,np.nan,3])==dict(blocks=3,defined_blocks=2,mean=None,block_sd=None)


def test_full_response_axes_preserve_silence_and_source_rounded_direction():
    config=paired_reports()[0]['resolved_config'];seeds=[93001,93002,93003]
    counts={f'pfc_{seed}_{condition}_{phase}':np.zeros(40,dtype=np.int64)for seed in seeds for condition in gain.CONDITIONS for phase in gain.PHASES}
    responses,rows=gain.response_arrays(counts,config,seeds)
    assert responses['values'].shape==(3,3,5,8) and len(rows)==45
    assert not responses['direction_defined'].any() and np.isnan(responses['decoded_angle_deg']).all()
    assert all(row['decoded_angle_deg'] is None and row['resultant_length'] is None for row in rows)
    unit=round(32*120/360);counts['pfc_93001_cue_late_delay'][unit]=10
    responses,rows=gain.response_arrays(counts,config,seeds)
    assert responses['decoded_angle_deg'][1,0,4]==pytest.approx(360*unit/32)
    row=next(row for row in rows if row['case_id']=='pfc_93001_cue'and row['phase']=='late_delay')
    assert row['cue_units']==3 and row['cue_selectivity_hz']==pytest.approx(10/3)
    assert responses['direction_defined'][1,0,4]


def test_complete_effect_outputs_preserve_every_control_phase_field_and_block(tmp_path):
    results={};seeds=dict(reference_2560=[1,2,3],expanded_100k=[4,5,6])
    for li,label in enumerate(gain.LABELS):
        results[label]={}
        for parameter in gain.PARAMETERS:
            changed=parameter=='candidate';value=(-2. if li==0 else 1.)if changed else 0.
            response=dict(values=np.full((3,3,5,8),value),direction_defined=np.zeros((3,3,5),dtype=bool),
                          decoded_angle_deg=np.full((3,3,5),np.nan),resultant_length=np.full((3,3,5),np.nan))
            histogram={(ci,bi,pi,wi,field,0):1 for ci in range(3)for bi in range(3)for pi in range(5)for wi in range(3)for field in range(17)}
            results[label][parameter]=dict(responses=response,field_rates=np.full((3,3,5,17),value),histogram=histogram,
                moments={'omission_mean_counts':np.full((5,20),value)},variance=[dict(condition='omission',phase='baseline',population='all',units=20,joint_rng_blocks=3,pooled_single_unit_ensemble_fano=None)])
    report=gain.compare_results(results,seeds,tmp_path)
    assert report['response_rows']==240 and report['field_rows']==510 and report['cdf_rows']==1530
    with np.load(tmp_path/'complete_empirical_cdf_effects.npz')as stored:
        assert stored['reference_2560_default'].shape==(3,3,5,3,17,1)
        assert np.all(stored['expanded_100k_paired_parameter_effect']==0)
    with np.load(tmp_path/'response_and_field_effects.npz')as stored:
        assert np.all(stored['reference_2560_paired_parameter_effects_hz']==-2)
        assert not stored['reference_2560_paired_angle_defined'].any()


@pytest.mark.parametrize('key',['confidence_intervals','p_values','scientific_parameter_domain_qualified','confirmation_execution_released'])
def test_real_cli_rejects_inference_or_qualification(tmp_path,key):
    settings=dict(schema='pfc_recurrent_gain_comparison_v1',scope='descriptive_parameter_development',confidence_intervals=False,p_values=False,scientific_parameter_domain_qualified=False,confirmation_execution_released=False)
    settings[key]=True;path=tmp_path/'wrong.yaml';path.write_text(yaml.safe_dump(settings))
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_recurrent_gain_comparison','--config',str(path),
        '--inputs',str(tmp_path/'missing.json'),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode!=0 and 'DESCRIPTIVE_UNQUALIFIED_PARAMETER_COMPARISON_ONLY' in result.stderr
    assert not (tmp_path/'out').exists()


def test_native_cached_moments_require_the_exact_source_parity_bridge():
    cases=[f'pfc_{seed}_{condition}'for seed in [93001,93002,93003]for condition in gain.CONDITIONS]
    headers={case:dict(binding=dict(report_sha256=case+'report',delivery_sha256=case+'delivery'))for case in cases}
    bridge=dict(status='PASS_STREAMED_SOURCE_EQUIVALENCE',passed_cases=9,registered_cases=9,
        cases=[dict(status='PASS',spike_arrays_exact=True)for case in cases],
        source_bindings=[dict(case_id=case,report_sha256=case+'report',delivery_sha256=case+'delivery',native_raw_sha256=case+'raw')for case in cases])
    fluctuation=dict(source_bindings=[dict(case_id=case,raw_sha256=case+'raw',native_report_sha256='native')for case in cases])
    phase=dict(native_report_sha256='native')
    gain.validate_native_cache_bridge(bridge,fluctuation,phase,headers)
    fluctuation['source_bindings'][0]['raw_sha256']='other-native-case'
    with pytest.raises(ValueError,match='SOURCE_PARITY_CHANGED'):gain.validate_native_cache_bridge(bridge,fluctuation,phase,headers)


def test_actual_archived_configs_and_all_source_headers_keep_their_purposes():
    root=Path(__file__).resolve().parents[2]
    protocol=yaml.safe_load((root/'Libs/configs/neural_population_pfc_recurrent_gain.yaml').read_text())
    index=json.loads((root/'registration_fixtures/index.json').read_text())
    for label in gain.LABELS:
        cohorts,checks=gain.prepare_headers(index,label,protocol)
        assert len(checks)==9 and all(row['paired_initial_background_input_law']for row in checks)
        for parameter in gain.PARAMETERS:
            config,registered,seeds,dimensions,_=cohorts[parameter]['registration']
            assert len(registered)==(96 if label=='expanded_100k'and parameter=='default'else 9)
            assert len(seeds)==(32 if label=='expanded_100k'and parameter=='default'else 3)
            assert dimensions['coupled_core_units']==protocol['cohorts'][label]['coupled_core_units']
            path=Path(index['sources'][label][parameter][0]['path'])/'source_run/config.yaml'
            if parameter=='candidate':
                assert config['source_replication']['purpose']=='source_parameter_domain_development'
                with pytest.raises(ValueError,match='explicit numerical calibration replication required'):
                    gain.source._source_registration(path)
            elif label=='reference_2560':assert 'source_replication'not in config
            else:assert config['source_replication']['purpose']=='numerical_calibration'


def test_actual_registration_rejects_changed_protocol_case_or_population():
    root=Path(__file__).resolve().parents[2]
    protocol=yaml.safe_load((root/'Libs/configs/neural_population_pfc_recurrent_gain.yaml').read_text())
    index=json.loads((root/'registration_fixtures/index.json').read_text())
    path=Path(index['sources']['reference_2560']['candidate'][0]['path'])/'source_run/config.yaml'
    cohort=copy.deepcopy(protocol['cohorts']['reference_2560']);cohort['source_rng_seeds']=[93001,93002,93999]
    with pytest.raises(ValueError,match='EXACT_MAIN_PROTOCOL'):gain.register_source_config(path,cohort,'candidate')
    cohort=copy.deepcopy(protocol['cohorts']['reference_2560']);cohort['coupled_core_units']=100000
    with pytest.raises(ValueError,match='SIZE_NORMALIZATION'):gain.register_source_config(path,cohort,'candidate')



def test_complete_real_native_readout_accepts_json_paths_and_rejects_changed_cache(tmp_path):
    root=Path(__file__).resolve().parents[2]
    protocol=yaml.safe_load((root/'Libs/configs/neural_population_pfc_recurrent_gain.yaml').read_text())
    index=json.loads((root/'registration_fixtures/index.json').read_text())
    cohorts,_=gain.prepare_headers(index,'reference_2560',protocol)
    cache=index['cache']['reference_2560']
    assert all(isinstance(path,str)for path in cache.values())
    result=gain.analyze_cohort(cohorts['default'],'reference_2560','default',[93001,93002,93003],cache,tmp_path/'complete')
    assert result['report']['cases']==9 and result['report']['units']==2560
    assert result['report']['native_cached_moments_reused'] is True
    assert result['responses']['values'].shape==(3,3,5,8)
    assert len(result['report']['condition_pairing'])==9
    changed=dict(cache);altered=tmp_path/'altered_moments.npz'
    altered.write_bytes(Path(cache['moments']).read_bytes()+b'changed')
    changed['moments']=str(altered)
    with pytest.raises(ValueError,match='NATIVE_CACHED_OUTPUT_BINDING_CHANGED'):
        gain.analyze_cohort(cohorts['default'],'reference_2560','default',[93001,93002,93003],changed,tmp_path/'rejected')

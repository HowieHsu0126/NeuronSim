"""Actual registration and variable-denominator/count-storage regression tests."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml
from . import pfc_recurrent_gain_comparison as gain
from . import pfc_recurrent_gain_replicates as replication


def fixture_inputs():
    root=Path(__file__).resolve().parents[2]
    return root,yaml.safe_load((root/'Libs/configs/neural_population_pfc_recurrent_gain_replicates.yaml').read_text()),json.loads((root/'fixtures/index.json').read_text())


@pytest.mark.parametrize('n',[29,32])
def test_descriptive_denominators_and_undefined_values(n):
    values=np.arange(n,dtype=float);result=gain.describe(values)
    assert result['blocks']==result['defined_blocks']==n
    assert result['mean']==values.mean() and result['block_sd']==pytest.approx(values.std(ddof=1))
    values[-1]=np.nan;result=gain.describe(values)
    assert result['blocks']==n and result['defined_blocks']==n-1 and result['mean'] is None and result['block_sd'] is None


def test_unequal_size_denominators_and_no_cross_size_pairing():
    left=np.arange(29.);right=np.arange(32.)
    result=gain.independent_effect(left,right,list(range(29)),list(range(100,132)))
    assert result['reference_blocks']==29 and result['expanded_blocks']==32 and not result['cross_size_blocks_paired']
    assert result['expanded_minus_reference_mean_effect']==1.5
    with pytest.raises(ValueError,match='DENOMINATORS'):gain.independent_effect(left,right,list(range(28)),list(range(100,132)))


def test_uint16_storage_checked_then_widened_before_arithmetic():
    raw=dict(spike_ticks=np.zeros(65536,dtype=np.int64),spike_unit_indices=np.zeros(65536,dtype=np.int64))
    with pytest.raises(ValueError,match='STORAGE_OVERFLOW'):replication.bin_counts(raw,1,total_bins=1)
    matrices=[np.full((20,4),65535-i,dtype=np.uint16)for i in range(32)]
    chunks=list(replication.matrix_chunks(matrices,3));assert len(chunks)==2
    assert chunks[0][2].dtype==np.int64 and int(chunks[0][2].sum())>65535
    result=replication.partition_moments(chunks[0][2]);oracle=np.stack(matrices).astype(np.float64)
    np.testing.assert_allclose(result['full32'][1],oracle[:,:,:3].var(axis=0,ddof=1).mean(0))
    np.testing.assert_allclose(result['new29'][1],oracle[3:,:,:3].var(axis=0,ddof=1).mean(0))
    assert set(replication.partition_moments(chunks[0][2],full_cached=True))=={'new29'}


def test_new29_variance_is_not_a_slice_of_the_pooled_32_estimate():
    counts=np.arange(32,dtype=np.int64)[:,None,None]*np.ones((32,20,2),dtype=np.int64);counts[:3]+=1000
    result=replication.partition_moments(counts)
    assert result['full32'][1][0]>1000*result['new29'][1][0]
    table={(0,i,0,0,0,0):i+1 for i in range(32)}
    selected=replication.subset_histogram(table,np.arange(3,32))
    assert len(selected)==29 and selected[0,0,0,0,0,0]==4 and selected[0,28,0,0,0,0]==32


def test_every_actual_config_shape_preserves_original_purpose():
    _,protocol,fixtures=fixture_inputs();shapes=[]
    for row in fixtures['registrations']:
        config,registered,seeds,dimensions,_=replication.source_registration(row['config_path'],row,protocol)
        shapes.append(len(seeds));assert len(registered)==3*len(seeds) and dimensions['coupled_core_units']==row['size']
        if row['origin']=='new':
            assert config['source_replication']['purpose']=='source_parameter_domain_development' and len(seeds)==29
            with pytest.raises(ValueError,match='explicit numerical calibration replication required'):gain.source._source_registration(row['config_path'])
    assert sorted(shapes)==[3,3,3,29,29,29,32]
    row=copy.deepcopy(fixtures['registrations'][-1]);row['source_rng_seed']=96128
    with pytest.raises(ValueError,match='REGISTERED_REPLICATION_CASE'):replication.source_registration(row['config_path'],row,protocol)


def test_real_native_source_binning_matches_independent_existing_phase_engine():
    _,protocol,fixtures=fixture_inputs();entry=fixtures['real_source']
    config,registered,_,dimensions,sha=replication.source_registration(Path(entry['path'])/'source_run/config.yaml',entry,protocol)
    status,header=gain.source._source_header(entry['path'],config,registered,sha);assert status['status']=='COMPLETE'
    raw,_,_=gain.source._source_values(header,config,registered,dimensions,sha)
    matrix=replication.bin_counts(raw,dimensions['coupled_core_units'])
    _,_,vectors,_=gain.source._source_phase_rows(raw,entry['case_id'],registered,config,dimensions)
    np.testing.assert_array_equal(matrix.sum(0,dtype=np.int64),raw['native_spike_counts'])
    for phase,(start,stop)in gain.PHASES.items():np.testing.assert_array_equal(matrix[round(start/5):round(stop/5)].sum(0,dtype=np.int64),vectors[entry['case_id']+'_'+phase])


def test_fixed_384_case_bank_rejects_missing_or_duplicate_cases():
    root,protocol,_=fixture_inputs();existing=json.loads((root/'preparation/existing_sources.json').read_text());entries=[dict(row,origin='existing')for row in existing]
    for family,cohort in protocol['cohorts'].items():
        for seed in cohort['new_source_rng_seeds']:
            for condition in gain.CONDITIONS:entries.append(dict(size=cohort['size'],parameter=cohort['parameter'],case_id=f'pfc_{seed}_{condition}',source_rng_seed=seed,origin='new'))
    assert len(replication.validate_bank(entries,existing,protocol))==384
    with pytest.raises(ValueError,match='COMPLETE_FIXED'):replication.validate_bank(entries[:-1],existing,protocol)
    with pytest.raises(ValueError,match='DUPLICATE'):replication.validate_bank(entries+[entries[0]],existing,protocol)


def test_comparison_keeps_all_axes_at_29_and_32_blocks(tmp_path):
    results={};seeds={}
    for label,n in zip(gain.LABELS,[29,32]):
        seeds[label]=list(range(100 if n==29 else 200,(100 if n==29 else 200)+n));results[label]={}
        for parameter in gain.PARAMETERS:
            x=-1. if parameter=='candidate'else 0.
            results[label][parameter]=dict(responses=dict(values=np.full((3,n,5,8),x),direction_defined=np.zeros((3,n,5),dtype=bool),decoded_angle_deg=np.full((3,n,5),np.nan),resultant_length=np.full((3,n,5),np.nan)),
                field_rates=np.full((3,n,5,17),x),moments={'omission_mean_counts':np.full((5,20),x)},
                variance=[dict(condition='omission',phase='baseline',population='all',units=20,joint_rng_blocks=n,ordered_offdiagonal_contribution_to_population_variance=x)],
                histogram={(ci,bi,pi,wi,field,0):1 for ci in range(3)for bi in range(n)for pi in range(5)for wi in range(3)for field in range(17)})
    report=gain.compare_results(results,seeds,tmp_path);assert report['response_rows']==240 and report['cdf_rows']==1530
    with np.load(tmp_path/'complete_empirical_cdf_effects.npz')as data:
        assert data['reference_2560_default'].shape==(3,29,5,3,17,1) and data['expanded_100k_default'].shape==(3,32,5,3,17,1)
    with (tmp_path/'cross_size_response_effects.csv').open()as stream:
        row=next(csv.DictReader(stream));assert row['reference_blocks']=='29'and row['expanded_blocks']=='32'


@pytest.mark.parametrize('key',['confidence_intervals','p_values','activation_threshold_introduced','scientific_parameter_domain_qualified'])
def test_cli_keeps_inference_and_activation_closed(tmp_path,key):
    settings=dict(schema='pfc_recurrent_gain_replicate_comparison_v1',scope='descriptive_fixed_replication',scientific_parameter_domain_qualified=False,confirmation_execution_released=False,confidence_intervals=False,p_values=False,activation_threshold_introduced=False)
    settings[key]=True;path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(settings))
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_recurrent_gain_replicates','--config',str(path),'--inputs',str(tmp_path/'none'),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode!=0 and 'DESCRIPTIVE_FIXED_REPLICATION_ONLY'in result.stderr and not(tmp_path/'out').exists()


def test_actual_native_mixed_3_plus_29_bank_runs_both_partition_readouts(tmp_path):
    _,protocol,fixtures=fixture_inputs();entries=fixtures['native_sources']
    assert len(entries)==192
    headers={};registrations={}
    for entry in entries:
        key=(2560,entry['parameter'],entry['case_id'])
        registration=replication.source_registration(Path(entry['path'])/'source_run/config.yaml',entry,protocol)
        config,registered,_,_,sha=registration
        status,header=gain.source._source_header(entry['path'],config,registered,sha)
        assert status['status']=='COMPLETE'and key not in headers
        headers[key]=header;registrations[key]=registration
    seeds=protocol['reference_full_rng_seeds'];scratch=tmp_path/'scratch';scratch.mkdir()
    for seed in seeds:
        for condition in gain.CONDITIONS:
            case=f'pfc_{seed}_{condition}'
            gain.validate_input_pair(headers[2560,'default',case]['report'],headers[2560,'candidate',case]['report'])
    for parameter in gain.PARAMETERS:
        pilot=replication.load_pilot(fixtures['pilot_root'],'reference_2560',parameter)
        result,pairs=replication.analyze_bank_cohort(headers,registrations,'reference_2560',parameter,seeds,pilot,{},tmp_path,scratch)
        assert len(pairs)==96
        for scope,n in [('new29',29),('full32',32)]:
            assert result[scope]['responses']['values'].shape==(3,n,5,8)
            assert result[scope]['report']['cases']==3*n and result[scope]['report']['unit_variance_denominator_blocks']==n
            assert len(result[scope]['variance'])==45
    assert not list(scratch.iterdir())

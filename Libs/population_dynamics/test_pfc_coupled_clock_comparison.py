"""One-block comparisons preserve all conditions, signs and zero denominators."""
import copy
import csv
import json
import os
from pathlib import Path
import io
import subprocess
import sys
import numpy as np
import pytest
from . import pfc_coupled_clock_comparison as comparison
from .pfc_delay_analysis import _check_readout
from .pfc_response_analysis import summarize_counts
from .pfc_coupled_clock_comparison import LABELS,CONDITIONS,PHASES,METRICS,discrepancy_rows,rms_comparison,response_rows,accumulate_case_histograms,cdf_on_support


def test_path_rows_keep_both_schemes_all_fields_conditions_and_zero_denominators():
    native={label:{condition:np.zeros((1000,17)) for condition in CONDITIONS} for label in LABELS}
    coupled={label:{condition:value.copy() for condition,value in level.items()} for label,level in native.items()}
    coupled[LABELS[1]]['cue'][:,0]=2.
    rows=discrepancy_rows({'native_same_seed':native,'coupled_inputs':coupled})
    assert len(rows)==2*2*3*5*17
    compared=rms_comparison(rows);assert len(compared)==2*3*5*17
    affected=[r for r in compared if r['condition']=='cue' and r['field_index']==0]
    assert len(affected)==10 and all(r['ratio_status']=='NATIVE_ZERO_COUPLED_NONZERO' for r in affected)
    assert all(r['coupled_over_native_rms'] is None for r in affected)
    assert all(r['coupled_minus_native_rms_hz']==2. for r in affected)


def test_missing_comparison_scheme_is_rejected():
    levels={label:{condition:np.zeros((1000,17)) for condition in CONDITIONS} for label in LABELS}
    with pytest.raises(ValueError,match='BOTH_CLOCK_COUPLING_SCHEMES_REQUIRED'):
        rms_comparison(discrepancy_rows({'native_same_seed':levels}))


def test_source_and_paired_response_differences_preserve_sign_and_vector_components():
    levels={label:np.zeros((3,5,8)) for label in LABELS}
    levels[LABELS[0]][CONDITIONS.index('cue'),:,METRICS.index('vector_x_hz')]=-3.
    levels[LABELS[1]][CONDITIONS.index('cue'),:,METRICS.index('vector_x_hz')]=-1.
    source,paired=response_rows({'coupled_inputs':levels})
    assert len(source)==len(paired)==2*3*5*8
    row=next(r for r in paired if r['left_cohort']==LABELS[0] and r['contrast']=='cue_minus_omission'
             and r['phase']=='late_delay' and r['metric']=='vector_x_hz')
    assert row['left_value']==-3. and row['right_value']==-1. and row['left_minus_right']==-2.


def test_real_cli_rejects_wrong_scope(tmp_path):
    path=tmp_path/'config.yaml';path.write_text('schema: wrong\n')
    run=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_coupled_clock_comparison',
       '--config',str(path),'--project-root',str(tmp_path),'--output',str(tmp_path/'output')],capture_output=True,text=True)
    assert run.returncode!=0 and 'DESCRIPTIVE_COUPLED_COMPARISON_SCOPE_REQUIRED' in run.stderr
    assert not (tmp_path/'output').exists()


def test_histograms_preserve_short_intervals_across_fixed_bins_and_longer_windows():
    config=dict(excitatory_neurons=8,inhibitory_neurons=2,excitatory_preference_bins=2,dt_ms=.05,duration_ms=5000.)
    raw={'spike_ticks':np.array([2020,2109],dtype=np.int64),'spike_unit_indices':np.array([0,0],dtype=np.int64)}
    hist={};fields=accumulate_case_histograms(raw,config,0,hist)
    assert fields.sum()==2 and fields[20,0]==fields[21,0]==1
    cue=list(PHASES).index('cue')
    assert hist[0,0,cue,0,0,1]==2 and hist.get((0,0,cue,0,0,2),0)==0
    assert hist[0,0,cue,1,0,2]==1 and hist[0,0,cue,2,0,2]==1
    raw={'spike_ticks':np.array([2020,2080,2109],dtype=np.int64),'spike_unit_indices':np.array([0,0,0],dtype=np.int64)}
    hist={};fields=accumulate_case_histograms(raw,config,0,hist)
    assert fields.sum()==3 and hist[0,0,cue,0,0,2]==1


def test_old_empirical_support_is_extended_right_continuously_without_clipping_new_support():
    result=cdf_on_support(np.array([[.2,.9,1.]]),np.array([0,2,4]),np.arange(7))
    np.testing.assert_array_equal(result,[[.2,.2,.9,.9,1.,1.,1.]])
    with pytest.raises(ValueError,match='COMPLETE_OBSERVED_CDF_SUPPORT_REQUIRED'):
        cdf_on_support(np.array([[.2,.9,.99]]),np.array([0,2,4]),np.arange(7))


@pytest.mark.parametrize("active", [True, False], ids=["active", "empty"])
def test_source_readout_csv_round_trip_preserves_validity_missing_and_numeric_checks(active):
    counts=np.zeros(10,dtype=np.int64)
    if active:counts[2]=3
    summary=summarize_counts(counts,8,90.,45.,100.)
    assert summary["direction_defined"]==active
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(summary))
    writer.writeheader();writer.writerow(summary);stream.seek(0)
    row=next(csv.DictReader(stream))
    assert _check_readout(row,summary)==len(summary)
    with pytest.raises(ValueError,match="circular direction validity changed"):
        _check_readout(dict(row,direction_defined=str(not active)),summary)
    for observed in ("nan",str(summary["excitatory_rate_hz"]+1.)):
        with pytest.raises(ValueError,match="independent raw-event readout mismatch"):
            _check_readout(dict(row,excitatory_rate_hz=observed),summary)
    if not active:
        assert row["decoded_angle_deg"]==""
        with pytest.raises(ValueError,match="undefined circular quantity must remain missing"):
            _check_readout(dict(row,decoded_angle_deg="0"),summary)


def test_explicit_three_clock_pairs_keep_left_minus_right_for_fields_responses_and_cdfs():
    levels={label:{condition:np.full((1000,17),value,dtype=float) for condition in CONDITIONS}
            for label,value in zip(LABELS,[1.,3.,7.])}
    rows=discrepancy_rows({'coupled_inputs':levels},comparison.ALL_CLOCK_PAIRS)
    assert len(rows)==3*3*5*17
    assert {r['signed_mean_hz'] for r in rows if r['left_cohort']==LABELS[0] and r['right_cohort']==LABELS[2]}=={-6.}
    values={label:np.full((3,5,8),value) for label,value in zip(LABELS,[1.,3.,7.])}
    source,contrasts=response_rows({'coupled_inputs':values},comparison.ALL_CLOCK_PAIRS)
    assert len(source)==len(contrasts)==3*3*5*8
    assert {r['left_minus_right'] for r in source if r['left_cohort']==LABELS[0] and r['right_cohort']==LABELS[2]}=={-6.}
    cdfs={label:np.broadcast_to(np.array([value,1.]),(3,5,3,17,2)).copy()
          for label,value in zip(LABELS,[.1,.2,.4])}
    rows=comparison.count_cdf_rows({'coupled_inputs':cdfs},np.array([0,1]),comparison.ALL_CLOCK_PAIRS)
    assert len(rows)==3*3*5*3*17*2
    far=[r for r in rows if r['left_cohort']==LABELS[0] and r['right_cohort']==LABELS[2] and r['count_threshold']==0]
    assert all(r['left_minus_right']==pytest.approx(-.3) for r in far)


def synthetic_case(directory,ne,ni,seed,condition,clock_index):
    """Explicitly synthetic sparse events; never invokes a source simulator."""
    directory.mkdir(parents=True,exist_ok=True);dt=comparison.DT[clock_index]
    config=dict(excitatory_neurons=ne,inhibitory_neurons=ni,excitatory_preference_bins=16,
                dt_ms=dt,readout_bin_ms=5.,duration_ms=5000.,stimulus_start_ms=100.,
                stimulus_center_deg=120.,stimulus_width_deg=30.,sigma_weight_profile=20.,Jpos_excit2excit=2.3)
    case=dict(case_id=f'pfc_{seed}_{condition}',source_rng_seed=seed,
              stimulus_strength_nA=comparison.coupled_execution.CONDITIONS[condition],distractor_strength_nA=0.)
    times=[50.];units=[1]
    if condition=='cue':times.append([105.,110.,115.][clock_index]);units.append(0)
    elif condition=='strong_cue':
        extra=([105.,115.],[110.],[])[clock_index];times.extend(extra);units.extend([0]*len(extra))
    raw=dict(spike_ticks=np.asarray([round(t/dt) for t in times],dtype=np.uint32),
             spike_unit_indices=np.asarray(units,dtype=np.uint32))
    np.savez_compressed(directory/'raw.npz',**raw)
    np.save(directory/'initial_voltage.npy',np.zeros(ne+ni),allow_pickle=False)
    np.savez(directory/'final_rng.npz',fixture_state=np.asarray([seed]))
    np.savez(directory/'terminal_state.npz',fixture_state=np.asarray([0.]))
    np.savez(directory/'units.npz',source_unit_ids=np.arange(ne+ni))
    (directory/'unit_metadata.json').write_text(json.dumps(dict(evidence_kind='synthetic_unit_fixture')))
    np.savez(directory/'applied_current_observation.npz',segment_start_ticks=np.asarray([0,round(100/dt)]),
             segment_stop_ticks=np.asarray([round(100/dt),round(5000/dt)]),current_A=np.asarray([[0.],[case['stimulus_strength_nA']*1e-9]]))
    (directory/'applied_current_metadata.json').write_text(json.dumps(dict(evidence_kind='synthetic_unit_fixture')))
    (directory/'report.json').write_text(json.dumps(dict(evidence_kind='synthetic_unit_fixture',source_simulations=0)))
    paths={path.name:path for path in directory.iterdir() if path.is_file()}
    result=comparison.inspect_case_readouts(paths,config,case,delivered_readouts=False)
    result.update(config=config,case=case,paths=paths,report={'evidence_kind':'synthetic_unit_fixture'})
    result['paired']['inputs_used']={'fixture':'identical source input consumption within one clock'}
    return result


@pytest.mark.parametrize('cohort,ne,ni,seed',[('reference_2560',2048,512,93001),('expanded_100k',80000,20000,95001)])
def test_complete_unit_statistics_distinguish_bin_shift_from_phase_counts_and_keep_all_axes(tmp_path,cohort,ne,ni,seed):
    block=dict(cohort=cohort,source_rng_seed=seed,excitatory_neurons=ne,inhibitory_neurons=ni)
    def cases(condition):
        return {label:synthetic_case(tmp_path/'synthetic_inputs'/condition/label,ne,ni,seed,condition,i)
                for i,label in enumerate(LABELS)}
    artifact_root=os.environ.get('NEURONSIM_PFC_GAIN_NUMERICAL_FIXTURE_OUTPUT')
    out=Path(artifact_root)/(cohort+'_'+str(seed)) if artifact_root else tmp_path/'synthetic_readouts'
    result=comparison.write_gain_block(out,block,cases,evidence_kind='synthetic_unit_fixture')
    assert len(result['cases'])==9 and result['count_support']==[0,1,2]
    (out/'fixture_report.json').write_text(json.dumps(dict(evidence_kind='synthetic_unit_fixture',
        source_simulations=0,scientific_result=False,**result),indent=2)+'\n')
    cue=list(CONDITIONS).index('cue');phase=list(PHASES).index('cue');nbins=round((PHASES['cue'][1]-PHASES['cue'][0])/5)
    with np.load(out/'unit_count_clock_differences.npz',allow_pickle=False) as stored:
        assert stored['phase_count_left_minus_right'].shape==(1,1,3,3,5,ne+ni)
        assert stored['bin_count_differences'].shape==(1,1,3,3,5,3,ne+ni)
        assert stored['phase_count_left_minus_right'].dtype==np.int64
        assert stored['cohort_names'].tolist()==[cohort] and stored['source_rng_seeds'].tolist()==[seed]
        assert stored['phase_count_left_minus_right'][0,0,0,cue,phase,0]==0
        difference=stored['bin_count_differences'][0,0,0,cue,phase,:,0]
        np.testing.assert_allclose(difference,[0.,2/nbins,np.sqrt(2/nbins)])
        strong=list(CONDITIONS).index('strong_cue')
        assert stored['phase_count_left_minus_right'][0,0,0,strong,phase,0]==1
        assert stored['bin_count_differences'][0,0,0,strong,phase,0,0]==pytest.approx(1/nbins)
        np.testing.assert_array_equal(stored['source_unit_ids'],np.arange(ne+ni))
    with np.load(out/'source_count_cdfs.npz',allow_pickle=False) as stored:
        assert stored['cdf'].shape==(1,1,3,3,5,3,17,3)
        assert stored['cdf_left_minus_right'].shape==stored['cdf'].shape
        assert np.all(stored['cdf'][...,-1]==1.)
        np.testing.assert_array_equal(stored['field_unit_counts'],[ne//16]*16+[ni])
        assert stored['expected_observations'][0,0,0,cue,phase,0,0]==nbins*(ne//16)
        np.testing.assert_allclose(stored['cdf_left_minus_right'][0,0,2],stored['cdf'][0,0,0]-stored['cdf'][0,0,2])
    with (out/'semantic_phase_readouts.csv').open() as stream: rows=list(csv.DictReader(stream))
    empty=[row for row in rows if row['phase']=='late_delay']
    assert empty and all(row['direction_defined']=='False' and row['decoded_angle_deg']=='' for row in empty)
    with (out/'field_clock_discrepancies.csv').open() as stream: row=next(csv.DictReader(stream))
    assert row['cohort']==cohort and int(row['source_rng_seed'])==seed and 'left_clock' in row and 'left_cohort' not in row
    with np.load(out/(LABELS[0]+'_count_histograms.npz'),allow_pickle=False) as stored:
        assert stored['cohort_names'].tolist()==[cohort] and stored['source_rng_seeds'].tolist()==[seed]


def synthetic_fine_binding(tmp_path,monkeypatch):
    """Hash-bound synthetic documents test this reader, not source qualification."""
    if not hasattr(comparison.coupled_execution,'GAIN_SCOPE'):
        pytest.skip('candidate executor snapshot required for recurrent-gain binding fixtures')
    seed=93001;condition='cue';coarse=synthetic_case(tmp_path/'coarse',2048,512,seed,condition,0)
    root=tmp_path/'synthetic_archive';relative=Path('source_cases')/comparison.case_key(.025,condition,seed)
    result=synthetic_case(root/relative,2048,512,seed,condition,1);paths=result['paths']
    np.savez(paths['final_rng.npz'],coarse__fixture_state=np.asarray([seed]),auxiliary_fixture=np.asarray([1]))
    comparison.coupled_execution.write_source_readouts(root/relative,result['raw'],result['config'],result['case']['case_id'],seed,condition)
    qualifier=dict(receipt_sha256='synthetic_receipt',delivery_sha256='synthetic_delivery',
                   qualification={'evidence_kind':'synthetic_unit_fixture','source_simulations':0})
    monkeypatch.setattr(comparison.coupled_execution,'qualify_fine_clock',lambda *args:qualifier)
    block=dict(scope=comparison.coupled_execution.GAIN_SCOPE,cohort='reference_2560',source_rng_seed=seed,
               refinement_seeds=[98711,98712],kernel_parameters=dict(comparison.coupled_execution.GAIN_KERNEL),
               coarse_references={condition:{'archive':'synthetic_coarse','delivery_sha256':'synthetic'}})
    code={module.__name__.split('.')[-1]:comparison.sha256_file(Path(module.__file__)) for module in
          (comparison.coupled_execution.executor,comparison.coupled_execution.native_drive,
           comparison.coupled_execution.native_execution,comparison.coupled_execution.pfc_clock_coupling,
           comparison.coupled_execution.pfc_coupled_drive)}
    report=dict(evidence_kind='synthetic_unit_fixture',source_simulations=0,
        status='COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE',source_execution_complete=True,scope=block['scope'],
        cohort=block['cohort'],source_rng_seed=seed,refinement_seeds=block['refinement_seeds'],kernel_parameters=block['kernel_parameters'],
        case_id=result['case']['case_id'],condition=condition,dt_ms=.025,coarse_dt_ms=.05,resolved_config=result['config'],
        selected_case=result['case'],coarse_reference=block['coarse_references'][condition],coupled_core_units=2560,
        duration_ms=5000.,settings_sha256='synthetic_protocol_sha256',numpy_version=np.__version__,
        confirmation_execution_released=False,coarse_final_rng_matches_archived_episode=True,coarse_current_agreement={'fixture':True},
        validations={'fixture':{'passed':True}},inputs_used={'input_mode':'coupled_binomial_clock_refinement_stream'},
        fine_clock_qualification=copy.deepcopy(qualifier),code=code,
        runner_sha256=comparison.sha256_file(Path(comparison.coupled_execution.__file__)))
    return root,relative,block,coarse,report


def bind_synthetic_report(root,relative,report):
    report['outputs']={path.name:dict(bytes=path.stat().st_size,sha256=comparison.sha256_file(path))
                       for path in (root/relative).iterdir() if path.is_file() and path.name!='report.json'}
    (root/relative/'report.json').write_text(json.dumps(report)+'\n')
    return {str(path.relative_to(root)):dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,
            sha256=comparison.sha256_file(path)) for path in (root/relative).iterdir() if path.is_file()}


def test_synthetic_bound_fine_reader_checks_delivered_temporal_phase_and_unit_outputs(tmp_path,monkeypatch):
    root,relative,block,coarse,report=synthetic_fine_binding(tmp_path,monkeypatch)
    manifest=bind_synthetic_report(root,relative,report)
    result=comparison.read_fine_case(block,tmp_path,root,manifest,.025,'cue',coarse,'synthetic_protocol_sha256')
    assert result['semantic_values_verified']>1000 and result['phase_counts'].shape==(5,2560)


@pytest.mark.parametrize('change,message',[
    ('cohort','REGISTERED_COUPLED_GAIN_CASE_BINDING_REQUIRED'),('seed','REGISTERED_COUPLED_GAIN_CASE_BINDING_REQUIRED'),
    ('auxiliary','REGISTERED_COUPLED_GAIN_CASE_BINDING_REQUIRED'),('kernel','REGISTERED_COUPLED_GAIN_CASE_BINDING_REQUIRED'),
    ('coarse_reference','REGISTERED_COUPLED_GAIN_CASE_BINDING_REQUIRED'),
    ('qualification','FINE_CLOCK_QUALIFICATION_BINDING_CHANGED'),('temporal','independent raw-event readout mismatch'),
    ('undefined','undefined circular quantity must remain missing'),('initial','COUPLED_COARSE_INITIAL_STATE_CHANGED'),
    ('rng','COUPLED_COARSE_FINAL_RNG_CHANGED'),('current','COUPLED_PHYSICAL_CONTROL_CHANGED')])
def test_bound_fine_reader_rejects_scientific_identity_and_semantic_drift(tmp_path,monkeypatch,change,message):
    root,relative,block,coarse,report=synthetic_fine_binding(tmp_path,monkeypatch);directory=root/relative
    if change=='cohort':report['cohort']='expanded_100k'
    elif change=='seed':report['source_rng_seed']=93002
    elif change=='auxiliary':report['refinement_seeds']=[95411,95412]
    elif change=='kernel':report['kernel_parameters']=dict(sigma_weight_profile=20.,Jpos_excit2excit=1.6)
    elif change=='coarse_reference':report['coarse_reference']={'archive':'different'}
    elif change=='qualification':report['fine_clock_qualification']['receipt_sha256']='other_receipt'
    elif change in ('temporal','undefined'):
        path=directory/'temporal_readouts.csv'
        with path.open() as stream:rows=list(csv.DictReader(stream))
        rows[0]['excitatory_rate_hz' if change=='temporal' else 'decoded_angle_deg']='1'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    elif change=='initial':
        values=np.load(directory/'initial_voltage.npy');values[0]=1.;np.save(directory/'initial_voltage.npy',values)
    elif change=='rng':np.savez(directory/'final_rng.npz',coarse__fixture_state=np.asarray([99]),auxiliary_fixture=np.asarray([1]))
    else:
        with np.load(directory/'applied_current_observation.npz') as stored:values={key:stored[key] for key in stored.files}
        values['current_A'][1,0]+=1e-9;np.savez(directory/'applied_current_observation.npz',**values)
    manifest=bind_synthetic_report(root,relative,report)
    with pytest.raises(ValueError,match=message):
        comparison.read_fine_case(block,tmp_path,root,manifest,.025,'cue',coarse,'synthetic_protocol_sha256')


def test_missing_fine_case_cannot_be_replaced_by_ordinary_same_seed_data(tmp_path,monkeypatch):
    root,relative,block,coarse,report=synthetic_fine_binding(tmp_path,monkeypatch)
    manifest=bind_synthetic_report(root,relative,report)
    with pytest.raises(ValueError):
        comparison.read_fine_case(block,tmp_path,root,manifest,.0125,'cue',coarse,'synthetic_protocol_sha256')


@pytest.fixture
def real_reader_registration():
    root=os.environ.get('NEURONSIM_PFC_GAIN_FIXTURE_ROOT');protocol=os.environ.get('NEURONSIM_PFC_GAIN_FIXTURE_PROTOCOL')
    if not root or not protocol:pytest.skip('CPU04 staged native and expanded source anchors required')
    return Path(root),comparison.yaml.safe_load(Path(protocol).read_text())


@pytest.mark.parametrize('cohort,seed,ne,ni',[('reference_2560',93001,2048,512),('expanded_100k',95001,80000,20000)])
def test_real_coarse_reader_preserves_both_size_observation_denominators(real_reader_registration,cohort,seed,ne,ni):
    root,protocol=real_reader_registration
    seed=protocol['cohorts'][cohort]['source_rng_seeds'][0]
    block=comparison.coupled_execution.select_settings(protocol,seed,cohort)
    result=comparison.read_coarse_case(block,root,'cue')
    np.testing.assert_array_equal(result['fields']['field_unit_counts'],[ne//16]*16+[ni])
    assert result['phase_counts'].shape==(5,ne+ni)
    assert result['fields']['field_counts'].sum()==len(result['raw']['spike_ticks'])
    np.testing.assert_allclose(result['fields']['rates_hz'],result['fields']['field_counts']/np.asarray([ne//16]*16+[ni])[None,:]/.005)


@pytest.mark.parametrize('replication', [False, True])
@pytest.mark.parametrize('change',['missing_block','cross_cohort','wrong_seed','missing_archive'])
def test_source_index_requires_complete_registered_blocks(tmp_path,change,replication):
    if not hasattr(comparison.coupled_execution,'GAIN_COHORTS'):
        pytest.skip('candidate executor snapshot required for recurrent-gain registration')
    cohorts=(comparison.coupled_execution.GAIN_REPLICATES_COHORTS if replication else comparison.coupled_execution.GAIN_COHORTS)
    blocks=[dict(cohort=cohort,source_rng_seed=seed) for cohort,(_,_,seeds,_) in cohorts.items() for seed in seeds]
    rows=[dict(block,archive=None,delivery_sha256=None) for block in blocks]
    index=dict(schema=comparison.GAIN_INDEX_SCHEMA,protocol_sha256='synthetic_protocol_sha256',blocks=rows)
    if change=='missing_block':rows.pop()
    elif change=='cross_cohort':rows[0]['cohort']='expanded_100k'
    elif change=='wrong_seed':rows[0]['source_rng_seed']=96001
    path=tmp_path/'synthetic_index.json';path.write_text(json.dumps(index))
    message='COMPLETED_SOURCE_ARCHIVE_AND_DELIVERY_HASH_REQUIRED' if change=='missing_archive' else ('COMPLETE_58_BLOCK_SOURCE_INDEX_REQUIRED' if replication else 'COMPLETE_SIX_BLOCK_SOURCE_INDEX_REQUIRED')
    with pytest.raises(ValueError,match=message):comparison.indexed_source_archives(path,{'protocol_sha256':'synthetic_protocol_sha256'},blocks,tmp_path)


def test_real_cli_rejects_pending_source_index_without_analysis_outputs(real_reader_registration,tmp_path):
    root,protocol=real_reader_registration
    config=os.environ['NEURONSIM_PFC_GAIN_FIXTURE_ANALYSIS_CONFIG'];index=os.environ['NEURONSIM_PFC_GAIN_FIXTURE_SOURCE_INDEX']
    output=tmp_path/'no_scientific_result'
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_coupled_clock_comparison',
        '--config',config,'--project-root',str(root),'--source-index',index,'--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and 'COMPLETED_SOURCE_ARCHIVE_AND_DELIVERY_HASH_REQUIRED' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('replication', [False, True])
def test_dispatch_consumes_all_registered_cases_without_cross_size_pairing(tmp_path,monkeypatch,replication):
    if not hasattr(comparison.coupled_execution,'GAIN_COHORTS'):
        pytest.skip('candidate executor snapshot required for recurrent-gain registration')
    cohorts=(comparison.coupled_execution.GAIN_REPLICATES_COHORTS if replication else comparison.coupled_execution.GAIN_COHORTS)
    blocks=[dict(cohort=cohort,source_rng_seed=seed) for cohort,(_,_,seeds,_) in cohorts.items() for seed in seeds]
    settings={'scope':'development_recurrent_gain_coupled_clock_comparison','protocol_sha256':'synthetic_protocol_sha256'}
    config=tmp_path/'synthetic_analysis.yaml';config.write_text(comparison.yaml.safe_dump(settings))
    index_path=tmp_path/'synthetic_index.json';index_path.write_text('{}')
    index={'blocks':blocks,'evidence_kind':'synthetic_unit_fixture'};visited=[]
    monkeypatch.setattr(comparison,'analysis_registration',lambda *args:({},blocks))
    monkeypatch.setattr(comparison,'indexed_source_archives',lambda *args:(index,[(tmp_path,{},block) for block in blocks]))
    def coarse(block,root,condition):
        visited.append((block['cohort'],block['source_rng_seed'],.05,condition));return dict(block,condition=condition)
    def fine(block,project,root,manifest,dt,condition,reference,sha):
        assert reference['cohort']==block['cohort'] and reference['source_rng_seed']==block['source_rng_seed']
        visited.append((block['cohort'],block['source_rng_seed'],dt,condition));return dict(block,condition=condition)
    monkeypatch.setattr(comparison,'read_coarse_case',coarse);monkeypatch.setattr(comparison,'read_fine_case',fine)
    def write(output,block,provider,**kwargs):
        for condition in CONDITIONS:assert set(provider(condition))==set(LABELS)
        return dict(block,row_counts={'synthetic_dispatch':9})
    monkeypatch.setattr(comparison,'write_gain_block',write)
    report=comparison.run_recurrent_gain(config,tmp_path,tmp_path/'output',index_path)
    assert len(set(visited))==len(visited)==(522 if replication else 54)
    assert report['status']=='COMPLETE_SYNTHETIC_RECURRENT_GAIN_READOUT_VALIDATION'
    assert report['source_simulations']==report['complete_source_episodes_consumed']==0
    assert report['synthetic_cases_consumed']==(522 if replication else 54) and report['scientific_result'] is False

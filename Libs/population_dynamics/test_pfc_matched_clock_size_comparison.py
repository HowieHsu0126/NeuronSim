"""Explicit streamed-size admission and independent cohort observation axes."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from . import pfc_matched_clock_size_comparison as size
from . import pfc_fluctuation_comparison as clocks
from .test_pfc_fluctuation_comparison import make_cohort,bind


def test_streamed_native_size_requires_explicit_admission_and_keeps_own_blocks(tmp_path):
    root=tmp_path/'reference';seeds=[95201,95202,95203,95204]
    make_cohort(root,.0125,native=True,seeds=seeds)
    delivery=clocks.read(root/'delivery.json');delivery['readout_mode']='fluctuations';clocks.write_json(root/'delivery.json',delivery);bind(root)
    with pytest.raises(ValueError,match='scale, phase or block'):clocks.load_cohort(root)
    cohort=clocks.load_cohort(root,registered_size=2560)
    assert cohort['report']['joint_rng_blocks']==seeds and cohort['config']['dt_ms']==.0125
    np.testing.assert_array_equal(cohort['sizes'],[128]*16+[512])
    with pytest.raises(ValueError,match='scale, phase or block'):clocks.load_cohort(root,registered_size=100000)
    with pytest.raises(ValueError,match='explicit streamed size'):clocks.load_cohort(root,native=True,registered_size=2560)


def metadata_cohorts():
    config={key:'shared'for key in clocks.PHYSICAL_KEYS}
    config.update(dt_ms=.0125,readout_bin_ms=5.,duration_ms=5000.,excitatory_preference_bins=16,synapse_execution='factorized_uniform_projections')
    driver=dict(numpy_version='2.4.2',source_entry_sha256='source',rng='MT19937',random_stream_policy='source_global',
                binomial_n=1000,binomial_probability=1.75e-5,dt_seconds=1.25e-5,external_rate_Hz=1.4,initial_draw_order='I_then_E',packed_draw_order='tick_major_I_then_E',total_ticks=400000)
    cohorts={};settings=dict(dt_ms=.0125,bin_ms=5.,cohorts={})
    for label,ne,ni,seeds in zip(size.LABELS,[2048,80000],[512,20000],[[95201,95202,95203],[95001,95002,95003]]):
        own=dict(config,excitatory_neurons=ne,inhibitory_neurons=ni,weight_scaling_factor=2048/ne)
        binding=dict(backend='numpy',code=dict(executor='same'),qualification_sha256='qualified',qualification_delivery_sha256='delivery',driver=dict(driver,n_e=ne,n_i=ni))
        cohorts[label]=dict(config=own,report=dict(joint_rng_blocks=seeds,coupled_core_units=ne+ni,common_execution_binding=binding),phase_responses={},count_histograms={})
        settings['cohorts'][label]=dict(source_rng_seeds=seeds,coupled_units=ne+ni)
    return cohorts,settings


def test_matched_source_contract_accepts_separate_size_specific_rng_axes():
    cohorts,settings=metadata_cohorts();size.validate_cohorts(cohorts,settings)


@pytest.mark.parametrize('changed',['clock','source','numpy','qualification','seed_overlap'])
def test_clock_source_or_independence_mismatch_is_rejected(changed):
    cohorts,settings=metadata_cohorts();second=cohorts[size.LABELS[1]]
    if changed=='clock':second['config']['dt_ms']=.05
    elif changed=='source':second['report']['common_execution_binding']['driver']['source_entry_sha256']='other'
    elif changed=='numpy':second['report']['common_execution_binding']['driver']['numpy_version']='other'
    elif changed=='qualification':second['report']['common_execution_binding']['qualification_sha256']='other'
    else:
        second['report']['joint_rng_blocks']=cohorts[size.LABELS[0]]['report']['joint_rng_blocks']
        settings['cohorts'][size.LABELS[1]]['source_rng_seeds']=second['report']['joint_rng_blocks']
    with pytest.raises(ValueError):size.validate_cohorts(cohorts,settings)


def test_cdf_export_preserves_both_seed_axes_field_sizes_and_denominators():
    cohorts={}
    for label,blocks,field_size,start in [(size.LABELS[0],2,128,95201),(size.LABELS[1],3,5000,95001)]:
        coordinates=[];mass=[];sizes=np.array([field_size]*16+[field_size*4])
        for ci in range(3):
            for bi in range(blocks):
                for pi,(begin,end)in enumerate(size.PHASES.values()):
                    for wi,width in enumerate([5,50,100]):
                        for fi,n in enumerate(sizes):
                            total=int(n*round((end-begin)/width));spiking=bi+1
                            coordinates.extend([(ci,bi,pi,wi,fi,0),(ci,bi,pi,wi,fi,2)]);mass.extend([total-spiking,spiking])
        cohorts[label]=dict(sizes=sizes,report=dict(joint_rng_blocks=list(range(start,start+blocks))),count_histograms=dict(
            coordinates=np.asarray(coordinates),observations=np.asarray(mass),shape=(3,blocks,5,3,17),windows_ms=np.asarray([5,50,100])))
    rows,distances,payload=size.compare_cdfs(cohorts,[5,50,100])
    assert 'source_rng_seeds'not in payload and 'field_unit_counts'not in payload
    np.testing.assert_array_equal(payload[size.LABELS[0]+'_source_rng_seeds'],[95201,95202])
    np.testing.assert_array_equal(payload[size.LABELS[1]+'_source_rng_seeds'],[95001,95002,95003])
    np.testing.assert_array_equal(payload['count_support'],[0,1,2])
    assert payload[size.LABELS[0]+'_cdf'].shape[1]==2 and payload[size.LABELS[1]+'_cdf'].shape[1]==3
    assert rows[0]['left_unit_window_denominator']==128*20 and rows[0]['right_unit_window_denominator']==5000*20
    assert rows[0]['left_blocks']==2 and rows[0]['right_blocks']==3 and rows[0]['cross_size_pairing']is False
    assert len(distances)==3*5*3*17


def test_source_conditions_paired_only_within_each_independent_size():
    cohorts={}
    for label,blocks,start in [(size.LABELS[0],2,95201),(size.LABELS[1],3,95001)]:
        values=np.zeros((3,blocks,5,8));values[1]=np.arange(blocks)[:,None,None]+1;values[2]=values[1]+2
        cohorts[label]=dict(report=dict(joint_rng_blocks=list(range(start,start+blocks))),phase_responses=dict(values=values,
            direction_defined=np.zeros((3,blocks,5),bool),decoded_angle_deg=np.full((3,blocks,5),np.nan),resultant_length=np.zeros((3,blocks,5))))
    rows,paired,payload=size.compare_responses(cohorts)
    assert len(rows)==len(paired)==120 and paired[0]['left_mean']==1.5 and paired[0]['right_mean']==2.
    assert paired[0]['left_minus_right']==-.5 and paired[0]['cross_size_pairing']is False
    assert paired[0]['within_size_conditions_paired']is True
    assert payload[size.LABELS[0]+'_paired_condition_values'].shape==(3,2,5,8)
    assert np.isnan(payload[size.LABELS[1]+'_decoded_angle_deg']).all()


def test_real_cli_rejects_paired_size_scope_with_preserved_failure(tmp_path):
    config=tmp_path/'wrong.yaml';config.write_text(yaml.safe_dump(dict(schema='pfc_independent_matched_clock_size_comparison_v1',scope='development_calibration',
        inference='descriptive_only',lag_ms=[0,5,10,100],count_windows_ms=[5,50,100],cross_size_pairing=True)))
    result=subprocess.run([sys.executable,'-B','-m',size.__name__,'--config',str(config),'--reference-archive',str(tmp_path/'ref'),
                           '--expanded-archive',str(tmp_path/'large'),'--output',str(tmp_path/'output')],capture_output=True,text=True)
    assert result.returncode!=0 and 'EXPLICIT_INDEPENDENT_SIZE_DESCRIPTION_REQUIRED'in result.stderr
    assert json.loads((tmp_path/'output/report.json').read_text())['scientific_fidelity_qualified']is False

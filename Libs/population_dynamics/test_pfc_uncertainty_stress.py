"""Known-truth stress protocol, inherited intervals, and explicit failure paths."""
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from . import pfc_uncertainty_stress as stress


@pytest.fixture(scope='module')
def multiplicities():
    return stress.paired_bootstrap_multiplicities(128,5000,7821)


def test_finite_laws_preserve_exact_mass_and_block_mean():
    expected=[.0097,.010103,.0095,.0105]
    for law,mean in zip(stress.LAWS,expected):
        d=np.asarray(law['d_numerators'],dtype=np.int64)
        reference=np.asarray(stress.REFERENCE_COUNT_MASSES)
        masses=reference+np.column_stack((d,-2*d,d))
        assert np.all(masses>=0)
        np.testing.assert_array_equal(masses.sum(1),np.full(len(d),100000))
        np.testing.assert_array_equal(masses@np.array([0,1,2]),np.full(len(d),20000))
        truth=stress.law_truth(law)
        assert truth['mean_d']==pytest.approx(mean)
        assert truth['expected_count']==.2 and truth['exact_expected_count']=='1/5'
    assert stress.law_truth(stress.LAWS[1])['exact_mean_d']=='10103/1000000'
    assert stress.law_truth(stress.LAWS[1])['rare_probability']==.01


def test_complete_cdf_k_and_frozen_reference_routes():
    d=np.array([960,980,5000],dtype=np.int64)
    values,difference,k,lookup,lo,hi,departures,roundtrip=stress.cdf_primitives(d)
    np.testing.assert_allclose(difference[:2],np.vstack((d,-d))/100000,atol=1e-16)
    np.testing.assert_array_equal(difference[2:],np.zeros((2,3)))
    np.testing.assert_allclose(values[4:6],np.array([[.82]*3,[.98]*3]),atol=0)
    np.testing.assert_array_equal(k,d/100000)
    assert values.shape==(8,3) and len(lookup)==8 and departures==0
    np.testing.assert_array_equal(lo[[2,3,6,7]],np.ones(4))
    assert roundtrip['maximum_probability_roundtrip_error']<1e-15
    with pytest.raises(ValueError,match='EXACT_VALID_INTEGER'):
        stress.cdf_primitives(np.array([8001,5000]))


def test_constant_realization_keeps_frozen_variable_law_unresolved(multiplicities):
    rows=stress.analyze_sample(np.full(128,960),multiplicities,stress.law_truth(stress.LAWS[1]))
    assert len(rows)==2
    assert all(row['status']=='UNRESOLVED_EMPIRICAL_CONSTANT_VARIABLE_LAW' for row in rows)
    assert all(row['simultaneous_covered'] is None and row['within_margin'] is None for row in rows)


def test_seeded_data_and_bootstrap_streams_are_separate():
    law=stress.LAWS[1]
    first=stress.sample_law(law,1,2,128)
    again=stress.sample_law(law,1,2,128)
    np.testing.assert_array_equal(first[0],again[0]);assert first[1:]==again[1:]
    new_data=stress.sample_law(law,1,2,128,data_seed=97103)
    new_boot=stress.sample_law(law,1,2,128,bootstrap_seed=97104)
    assert new_data[1]==first[1] and not np.array_equal(new_data[0],first[0])
    np.testing.assert_array_equal(new_boot[0],first[0]);assert new_boot[1]!=first[1]
    assert stress.sample_law(law,1,3,128)[1]!=first[1]
    assert sum(first[3])==128 and first[2]==first[3][2]


def test_shared_draw_maximum_and_floor_reconstruct_the_full_group(multiplicities):
    d=np.tile(np.array([850,1050]),64)
    rows=stress.analyze_sample(d,multiplicities,stress.law_truth(stress.LAWS[2]))
    draws=(multiplicities/128)@((d-d.mean())/100000)
    se=draws.std(ddof=1)
    fresh=float(np.quantile(np.abs(draws)/se,.975,method='higher'))
    assert rows[0]['fresh_critical']==pytest.approx(fresh,abs=1e-9)
    for row,critical in zip(rows,[fresh,max(fresh,stress.FLOOR)]):
        assert row['status']=='DEFINED'
        assert row['applied_critical']==pytest.approx(critical,abs=1e-9)
        assert row['k_mean_standard_error']==pytest.approx(se,abs=1e-15)
        assert row['direct_mean_standard_error']==pytest.approx(se,abs=1e-15)
        assert row['k_upper']==pytest.approx(d.mean()/100000+critical*se,abs=1e-13)
        lo=json.loads(row['lower_by_threshold']);hi=json.loads(row['upper_by_threshold'])
        np.testing.assert_allclose(lo,[.0095-critical*se,-.0095-critical*se,0,0],atol=1e-13)
        np.testing.assert_allclose(hi,[.0095+critical*se,-.0095+critical*se,0,0],atol=1e-13)
        assert row['simultaneous_covered'] is True
        assert row['frozen_tail_departures']==0
        assert row['frozen_tail_upper']==pytest.approx(1-.025**(1/128))
    assert rows[1]['applied_critical']==stress.FLOOR


def test_no_rare_nonconstant_fixture_falsifies_even_floor_sensitivity(multiplicities):
    d=np.tile(np.array([960,980]),64)
    rows=stress.analyze_sample(d,multiplicities,stress.law_truth(stress.LAWS[1]))
    for row in rows:
        assert row['status']=='DEFINED' and row['k_mean_standard_error']>0
        assert row['simultaneous_covered'] is False
        assert row['within_margin'] is True and row['erroneous_within_margin'] is True
        assert row['k_upper']<.01<stress.law_truth(stress.LAWS[1])['mean_d']


def test_exact_binomial_rate_intervals_and_unresolved_denominator():
    assert stress.exact_binomial_interval(0,100)==pytest.approx((0,1-.025**.01))
    assert stress.exact_binomial_interval(100,100)==pytest.approx((.025**.01,1))
    lower,upper=stress.exact_binomial_interval(3,10)
    cdf=sum(math.comb(10,k)*upper**k*(1-upper)**(10-k) for k in range(4))
    survival=sum(math.comb(10,k)*lower**k*(1-lower)**(10-k) for k in range(3,11))
    assert cdf==pytest.approx(.025) and survival==pytest.approx(.025)
    law=stress.LAWS[0]
    records=[dict(law=law['name'],mode=mode,status=status,simultaneous_covered=covered)
             for mode in stress.MODES for status,covered in [('DEFINED',True),('UNRESOLVED_EMPIRICAL_CONSTANT_VARIABLE_LAW',None)]]
    rows=stress.summarize(records,[law])
    coverage=[row for row in rows if row['metric']=='simultaneous_covered']
    assert all(row['repetitions']==2 and row['defined_repetitions']==1 and row['rate']==.5 for row in coverage)


@pytest.mark.parametrize('key,value',[('method_adopted',True),('coverage_qualified',True),('scope','confirmation')])
def test_real_cli_rejects_qualification_or_adoption(tmp_path,key,value):
    config=dict(schema='pfc_nonconstant_uncertainty_stress_v1',scope='known_truth_falsification',method_adopted=False,coverage_qualified=False)
    config[key]=value
    path=tmp_path/'wrong.yaml';path.write_text(yaml.safe_dump(config))
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_uncertainty_stress',
        '--config',str(path),'--project-root',str(tmp_path),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode!=0 and 'BOUNDED_SYNTHETIC_FALSIFICATION_ONLY_REQUIRED' in result.stderr
    assert not (tmp_path/'out').exists()


def test_canonical_protocol_rejects_unapproved_repetition_budget(tmp_path):
    source=Path(__file__).resolve().parents[1]/'configs/neural_population_pfc_uncertainty_stress.yaml'
    config=yaml.safe_load(source.read_text())
    assert config['laws']==list(stress.LAWS) and config['reference_count_masses']==[82000,16000,2000]
    config['repetitions_per_law']=501
    changed=tmp_path/'changed.yaml';changed.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='FIXED_STRESS_PROTOCOL_REQUIRED'):
        stress.run(changed,tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()

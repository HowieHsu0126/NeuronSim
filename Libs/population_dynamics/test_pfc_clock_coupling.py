"""Exact finite-law checks and preservation of the source coarse RNG stream."""
from collections import defaultdict
from math import comb
import subprocess
import sys

import numpy as np
import pytest

from .pfc_clock_coupling import split_counts, CoupledClockRows
from .pfc_native_drive import create_drive


class MissingDraw(Exception):
    def __init__(self,n,p):self.n,self.p=n,p


class EnumeratedBinomial:
    def __init__(self,draws):self.draws,self.index=draws,0
    def binomial(self,n,p):
        n=np.asarray(n);assert n.shape==(1,)
        if self.index==len(self.draws):raise MissingDraw(int(n[0]),p)
        value=self.draws[self.index];self.index+=1
        return np.asarray([value],dtype=np.int64)


def conditional_distribution(parent,p):
    """Enumerate actual sampler calls, including data-dependent binomial n."""
    outcomes=defaultdict(float);pending=[([],1.)]
    while pending:
        draws,weight=pending.pop()
        try:
            result=split_counts(np.asarray([parent],dtype=np.uint16),p,EnumeratedBinomial(draws))
        except MissingDraw as missing:
            for k in range(missing.n+1):
                probability=comb(missing.n,k)*missing.p**k*(1-missing.p)**(missing.n-k)
                if probability:pending.append((draws+[k],weight*probability))
        else:outcomes[tuple(map(int,result[:,0]))]+=weight
    return outcomes


@pytest.mark.parametrize('n,p',[(1,.00007),(1,1.),(3,.2),(4,.8)])
def test_actual_sampler_has_exact_independent_binomial_children(n,p):
    joint=defaultdict(float)
    for parent in range(n+1):
        parent_probability=comb(n,parent)*p**parent*(1-p)**(n-parent)
        conditional=conditional_distribution(parent,p)
        assert sum(conditional.values())==pytest.approx(1.,abs=1e-14)
        assert sum(sum(pair)*value for pair,value in conditional.items())==pytest.approx(parent,abs=1e-13)
        for pair,value in conditional.items():joint[pair]+=parent_probability*value
    for a in range(n+1):
        for b in range(n+1):
            expected=comb(n,a)*comb(n,b)*(p/2)**(a+b)*(1-p/2)**(2*n-a-b)
            assert joint[a,b]==pytest.approx(expected,rel=1e-12,abs=1e-16)


def test_recursive_refinement_has_independent_quarter_tick_marginals():
    n,p=2,.35;joint=defaultdict(float);difference_variance=0.
    for parent in range(n+1):
        parent_p=comb(n,parent)*p**parent*(1-p)**(n-parent)
        for (a,b),pair_p in conditional_distribution(parent,p).items():
            for left,left_p in conditional_distribution(a,p/2).items():
                for right,right_p in conditional_distribution(b,p/2).items():
                    values=left+right;weight=parent_p*pair_p*left_p*right_p
                    joint[values]+=weight;difference_variance+=(sum(values)-parent)**2*weight
    for values,actual in joint.items():
        expected=np.prod([comb(n,k)*(p/4)**k*(1-p/4)**(n-k) for k in values])
        assert actual==pytest.approx(expected,rel=1e-12,abs=1e-16)
    assert len(joint)==(n+1)**4
    assert difference_variance==pytest.approx(3*n*p*p/4,rel=1e-13)


def fixture():
    config=dict(excitatory_neurons=8,inhibitory_neurons=4,dt_ms=.05,duration_ms=1.,source_rng_seed=95201)
    return config,dict(source_rng_seed=95201)


def test_children_do_not_change_coarse_inputs_initialization_or_rng_state():
    config,case=fixture();initial,coarse=create_drive(config,case)
    reference_initial,reference=create_drive(config,case)
    np.testing.assert_array_equal(initial,reference_initial)
    coupled=CoupledClockRows(coarse,[95211,95212])
    expected=reference.take(20);actual=[]
    for _ in range(20):actual.append(coupled.take_row()[0][0])
    np.testing.assert_array_equal(actual,np.concatenate([expected['I'],expected['E']],axis=1))
    for left,right in zip(coarse.get_state(),reference.get_state()):np.testing.assert_array_equal(left,right)
    assert coarse.event_totals==reference.event_totals
    assert coupled.metadata()['fine_source_seed_sequence_preserved'] is False


def test_row_stream_is_reproducible_has_full_support_and_retains_previous_output():
    config,case=fixture();streams=[CoupledClockRows(create_drive(config,case)[1],[95211,95212]) for _ in range(2)]
    first=streams[0].take_row();saved=[x.copy() for x in first];reference=streams[1].take_row()
    for a,b in zip(first,reference):np.testing.assert_array_equal(a,b)
    streams[0].take_row()
    for level,(before,after) in enumerate(zip(saved,first)):
        np.testing.assert_array_equal(before,after)
        assert after.shape==(2**level,12) and after.dtype==np.uint16
        assert np.all(after<=1000)
    source=np.array([0,1,999,1000],dtype=np.uint16)
    refined=split_counts(source,1.,np.random.RandomState(25))
    assert np.all(refined<=source) and np.all(refined<=1000)


def test_invalid_input_and_reused_source_seed_are_rejected_without_draws():
    rng=np.random.RandomState(27);state=rng.get_state()
    for values,p in [(np.array([1.5]),.1),(np.array([-1]),.1),(np.array([1001]),.1),(np.array([1]),0.)]:
        with pytest.raises(ValueError,match='VALID_PARENT'):split_counts(values,p,rng)
    for a,b in zip(state,rng.get_state()):np.testing.assert_array_equal(a,b)
    config,case=fixture()
    with pytest.raises(ValueError,match='REUSE_SOURCE_SEEDS'):CoupledClockRows(create_drive(config,case)[1],[95201,95212])


def test_real_cli_rejects_wrong_scope_without_output(tmp_path):
    config=tmp_path/'invalid.yaml';config.write_text('schema: invalid\n')
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_clock_coupling',
        '--config',str(config),'--project-root',str(tmp_path),'--output',str(tmp_path/'output')],capture_output=True,text=True)
    assert result.returncode!=0 and 'DEVELOPMENT_INPUT_COUPLING_SCOPE_REQUIRED' in result.stderr
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('p', [7e-5 / 4, (7e-5 / 4) / (1 - 7e-5 / 4),
                             7e-5 / 8, (7e-5 / 8) / (1 - 7e-5 / 8), .5])
def test_small_count_kernel_matches_native_values_rng_and_following_calls(p):
    from .pfc_clock_coupling import _refinement_binomial
    parent = np.tile(np.arange(9, dtype=np.int64), 4096)[::-1]
    actual, native = np.random.RandomState(95427), np.random.RandomState(95427)
    actual.normal(); native.normal()  # Retain the cached Gaussian too.
    np.testing.assert_array_equal(_refinement_binomial(parent, p, actual), native.binomial(parent, p))
    for a, b in zip(actual.get_state(), native.get_state()):
        np.testing.assert_array_equal(a, b)
    # The native parameter cache is not in get_state. Changing its contents
    # must not change any subsequent native result, including the last n,p.
    for n, probability in [(int(parent[-1]), p), (1000, .03), (3, .5), (7, .8)]:
        np.testing.assert_array_equal(actual.binomial(n, probability, 100),
                                      native.binomial(n, probability, 100))
    np.testing.assert_array_equal(actual.normal(size=32), native.normal(size=32))


def test_small_count_residual_boundaries_match_scalar_native_inversion():
    import math
    from .pfc_clock_coupling import _small_binomial_masses, _small_binomial_from_uniforms
    for p in (7e-5 / 4, 7e-5 / 8, .5):
        masses = _small_binomial_masses(p)
        for n in range(9):
            for boundary in np.cumsum(masses[n, :n + 1]):
                for u in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
                    if not 0 <= u < 1:
                        continue
                    residual = float(u); px = math.exp(n * math.log(1 - p)); k = 0
                    while residual > px:
                        k += 1
                        if k > n:
                            break
                        residual -= px
                        px = ((n - k + 1) * p * px) / (k * (1 - p))
                    result = _small_binomial_from_uniforms(np.array([n]), np.array([u]), masses)
                    if k > n:
                        assert result is None
                    else:
                        assert int(result[0]) == k


def test_small_count_restart_restores_rng_and_outside_domain_uses_native(monkeypatch):
    from . import pfc_clock_coupling as coupling
    actual, native = np.random.RandomState(95428), np.random.RandomState(95428)
    parent = np.array([0, 1, 4, 0], dtype=np.int64)
    monkeypatch.setattr(coupling, '_small_binomial_from_uniforms', lambda *args: None)
    np.testing.assert_array_equal(coupling._refinement_binomial(parent, 7e-5 / 4, actual),
                                  native.binomial(parent, 7e-5 / 4))
    for n, p in [(np.array([0, 9, 1000]), 7e-5 / 4), (parent, .2)]:
        np.testing.assert_array_equal(coupling._refinement_binomial(n, p, actual), native.binomial(n, p))
    for a, b in zip(actual.get_state(), native.get_state()):
        np.testing.assert_array_equal(a, b)


def test_recursive_small_count_kernel_matches_dense_rows_and_rng(monkeypatch):
    from . import pfc_clock_coupling as coupling
    config = dict(excitatory_neurons=800, inhibitory_neurons=200, dt_ms=.05,
                  duration_ms=5., source_rng_seed=95009)
    case = dict(source_rng_seed=95009)
    actual = CoupledClockRows(create_drive(config, case)[1], [95427, 95428])
    native = CoupledClockRows(create_drive(config, case)[1], [95427, 95428])
    optimized = coupling._refinement_binomial
    for _ in range(100):
        monkeypatch.setattr(coupling, '_refinement_binomial', optimized)
        rows = actual.take_row()
        monkeypatch.setattr(coupling, '_refinement_binomial', lambda n, p, r: r.binomial(n, p))
        expected = native.take_row()
        for a, b in zip(rows, expected):
            np.testing.assert_array_equal(a, b)
        for key, value in actual.state_arrays().items():
            np.testing.assert_array_equal(value, native.state_arrays()[key])

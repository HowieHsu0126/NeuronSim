"""Exact paired design, signed contrasts, and clock-specific zero categories."""
from fractions import Fraction as F
from itertools import combinations,product
import pytest
from .pfc_first_passage_clock_comparison import paired_block
from .pfc_first_passage_history_summary import variance_components


def row(index,value,variance=0,category="conditional_integration_required"):
    return dict(sample_index=index,frame_index=index,unit_index=index,window_start_ms=100.,window_stop_ms=105.,
                conditional_estimate=float(value),inner_monte_carlo_variance=float(variance),category=category,
                first_spike_tick=None if category=="no_first_spike" else 2004)


def test_exact1536_case_paired_covariance_and_inner_noise():
    coarse=[F(0),F(1,4),F(1,2),F(3,4)]
    fine=[F(1,4),F(0),F(3,4),F(1,2)]
    total=total_square=total_e=F(0);count=0
    for subset in combinations(range(4),2):
        for bits in product((0,1),repeat=8):
            cr=[];fr=[];d=[];v=[]
            for j,u in enumerate(subset):
                qc=coarse[u]*(bits[4*j]+bits[4*j+1]);qf=fine[u]*(bits[4*j+2]+bits[4*j+3])
                vc=coarse[u]**2*(bits[4*j]-bits[4*j+1])**2
                vf=fine[u]**2*(bits[4*j+2]-bits[4*j+3])**2
                cr.append(row(u,qc,vc,"no_first_spike" if coarse[u]==0 else "conditional_integration_required"))
                fr.append(row(u,qf,vf,"no_first_spike" if fine[u]==0 else "conditional_integration_required"))
                d.append(qf-qc);v.append(vf+vc)
            expected_mean=sum(d)/2;sample_var=(d[0]-d[1])**2/2
            expected_e=(sample_var+sum(v)/2)/4
            stats,_,_=paired_block(cr,fr,4)
            assert stats["sample_mean"]==float(expected_mean)
            assert stats["within_archive_mean_variance_diagnostic"]==float(expected_e)
            total+=expected_mean;total_square+=expected_mean**2;total_e+=expected_e;count+=1
    assert count==1536 and total/count==0 and total_square/count==total_e/count==F(25,192)


def test_clock_specific_zeros_stay_paired_and_cdf_sign_is_opposite():
    coarse=[row(0,.25),row(1,0,category="no_first_spike")]
    fine=[row(0,0,category="refractory_covers_remaining_window"),row(1,.75)]
    stats,rows,categories=paired_block(coarse,fine,2)
    assert stats["sample_denominator"]==2 and stats["sample_mean"]==.25
    assert stats["fine_minus_coarse_cdf_le1"]==-.25
    assert [r["fine_minus_coarse_count_ge2"] for r in rows]==[-.25,.75]
    assert sum(categories.values())==2 and stats["within_archive_mean_variance_diagnostic"]==0
    with pytest.raises(ValueError,match="BLOCK_DENOMINATOR"):
        variance_components([-.25,.75],[0,0],2)


def test_mismatched_or_duplicated_physical_windows_are_rejected():
    coarse=[row(0,.1),row(1,.2)];fine=[row(0,.1),row(1,.2)]
    fine[1]["unit_index"]=9
    with pytest.raises(ValueError,match="PHYSICAL_WINDOW_PAIRING"):paired_block(coarse,fine,4)
    with pytest.raises(ValueError,match="PAIR_DENOMINATOR"):paired_block([coarse[0]]*2,[coarse[0]]*2,4)

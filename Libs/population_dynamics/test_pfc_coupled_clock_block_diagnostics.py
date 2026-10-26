import importlib.util
import itertools
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("clock_diagnostics",Path(__file__).with_name("pfc_coupled_clock_block_diagnostics.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    rows=[]
    for pair,contrast,phase,metric in itertools.product(m.PAIRS,m.CONTRASTS,m.PHASES,m.METRICS):
        vals={m.CLOCKS[0]:1.0,m.CLOCKS[1]:2.0,m.CLOCKS[2]:3.0}
        rows.append(dict(cohort="expanded_100k",source_rng_seed=95004,contrast=contrast,phase=phase,metric=metric,
            units="Hz",evidence_kind="archived_source_execution",left_clock=pair[0],right_clock=pair[1],
            left_value=vals[pair[0]],right_value=vals[pair[1]],left_minus_right=vals[pair[0]]-vals[pair[1]]))
    return rows


def test_full_pivot_and_pair_orientation():
    result=m.pivot_block(fixture(),"expanded_100k",95004)
    assert len(result)==120 and result[0][m.CLOCKS[2]]==3.0 and result[0]["partition"]=="remaining29"


@pytest.mark.parametrize("change",["missing","duplicate","identity","nonfinite","difference","inconsistent_clock"])
def test_invalid_response_rejected(change):
    rows=fixture()
    if change=="missing": rows.pop()
    elif change=="duplicate": rows.append(dict(rows[0]))
    elif change=="identity": rows[0]["source_rng_seed"]=95005
    elif change=="nonfinite": rows[0]["left_value"]=float("nan")
    elif change=="difference": rows[0]["left_minus_right"]=2.0
    else: rows[0].update(left_value=4.0,left_minus_right=2.0)
    with pytest.raises(ValueError):m.pivot_block(rows,"expanded_100k",95004)


def test_zero_membership_and_partition_union():
    rows=[]
    for i in range(32):
        rows.append(dict(cohort="expanded_100k",source_rng_seed=95001+i,partition="pilot3" if i<3 else "remaining29",
            contrast="cue_minus_omission",phase="late_delay",metric="cue_selectivity_hz",
            **dict(zip(m.CLOCKS,(0.0 if i==0 else 10.0,1.0 if i==0 else 10.0,2.0 if i==0 else 10.0)))))
    result=m.response_diagnostics(rows)
    full=[r for r in result if r["partition"]=="full32_descriptive" and r["right_clock"]==m.CLOCKS[1]][0]
    assert full["blocks"]==32 and full["exact_zero_membership_changed"]==1
    assert full["maximum_change_source_rng_seeds"]=="95001" and full["mean_signed_change"]==-1/32
    assert len(result)==9


def test_cdf_block_maxima_preserve_all_dimensions():
    rows=[]
    for cohort,start in m.STARTS.items():
        for offset,pair,condition,phase,window,field in itertools.product(range(32),m.PAIRS,("omission","cue","strong_cue"),m.PHASES,(5,20,50),range(17)):
            rows.append(dict(cohort=cohort,source_rng_seed=str(start+offset),left_clock=pair[0],right_clock=pair[1],
                condition=condition,phase=phase,window_ms=str(window),field_index=str(field),partition="pilot3" if offset<3 else "remaining29",
                maximum_absolute_cdf_difference=field/20,first_maximizing_count_threshold="1"))
    result=m.cdf_block_maxima(rows)
    assert len(result)==2880 and all(r["maximum_absolute_cdf_difference"]==.8 for r in result)
    assert result[0]["maximizing_window_field"]=="5:16;20:16;50:16"
    rows.pop()
    with pytest.raises(ValueError):m.cdf_block_maxima(rows)

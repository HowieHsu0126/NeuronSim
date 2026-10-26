"""Timing boundaries, held-out independence, loss correction and route choice."""
from copy import deepcopy

import pytest

from .pfc_history_timing_prediction import time_bin, heldout_predictions, decision


def record(seed,index,value,variance=0.):
    return dict(source_rng_seed=seed,time_bin=index,estimate=value,inner_variance=variance)


def test_source_refractory_boundaries_keep_eligible_zeros_distinct():
    row=dict(category="conditional_integration_required",window_stop_tick=100,
             first_spike_tick=40,refractory_ticks=40,conditional_estimate=0.,inner_monte_carlo_variance=0.)
    assert time_bin(row,.05,[0.,1.,2.,3.])==(0,1.)
    assert time_bin(dict(row,first_spike_tick=20),.05,[0.,1.,2.,3.])==(1,2.)
    assert time_bin(dict(row,first_spike_tick=0),.05,[0.,1.,2.,3.])==(2,3.)
    assert time_bin(dict(row,first_spike_tick=39),.05,[0.,1.,2.,3.])[0]==1
    assert time_bin(dict(row,category="no_first_spike",first_spike_tick=None),.05,[0.,1.,2.,3.])==(-1,0.)
    with pytest.raises(ValueError,match="POSITIVE_POST_REFRACTORY_TIME"):
        time_bin(dict(row,first_spike_tick=60),.05,[0.,1.,2.,3.])


def test_heldout_outcomes_do_not_enter_either_fit():
    records=[record(seed,index,value) for seed in (1,2,3) for index,value in ((-1,0.),(0,.125),(1,.25),(2,.5))]
    original,fold=heldout_predictions(records,1,3)
    assert fold["training_bin_counts"]=={0:2,1:2,2:2}
    assert [r["timing_prediction"] for r in original]==[0.,.125,.25,.5]
    changed=deepcopy(records)
    for row in changed:
        if row["source_rng_seed"]==1 and row["time_bin"]>=0:row["estimate"]=.75
    other,_=heldout_predictions(changed,1,3)
    assert [(r["baseline_prediction"],r["timing_prediction"]) for r in other]==[(r["baseline_prediction"],r["timing_prediction"]) for r in original]


def test_noise_corrected_losses_can_be_negative_and_cancel_in_gain():
    rows=[record(seed,index,0. if index<0 else .5,0. if index<0 else .25)
          for seed in (1,2,3) for index in (-1,0,1,2)]
    predictions,fold=heldout_predictions(rows,1,3)
    assert fold["baseline_noise_corrected_loss"]==fold["timing_noise_corrected_loss"]==-.1875
    assert all(r["paired_loss_reduction"]==0. for r in predictions)
    rows=[dict(row,inner_variance=0.) for row in rows]
    clean,_=heldout_predictions(rows,1,3)
    assert [r["paired_loss_reduction"] for r in predictions]==[r["paired_loss_reduction"] for r in clean]


def test_missing_training_bin_and_influence_do_not_promote_subset_success():
    rows=[record(seed,index,.25) for seed in (1,2,3) for index in (0,1)]
    rows.append(record(1,2,.75))
    _,fold=heldout_predictions(rows,1,3)
    assert fold["missing_training_bins"]==[2] and fold["paired_loss_reduction"] is None
    assert decision([fold])["status"]=="INSUFFICIENT_TRAINING_BIN_COVERAGE"
    folds=[dict(source_rng_seed=seed,complete=True,paired_loss_reduction=gain) for seed,gain in ((1,10.),(2,-1.),(3,-1.))]
    result=decision(folds)
    assert result["aggregate_gain"]>0 and result["gain_after_largest_positive_block_removed"]<0
    assert not result["prepare_timing_stratum_census"]
    for row in folds:row["paired_loss_reduction"]=1.
    assert decision(folds)["prepare_timing_stratum_census"]

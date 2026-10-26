import numpy as np
import pytest
from pfc_response_gap_diagnostics import decompose,SEEDS


def test_single_block_contribution_is_not_deleted():
    x=np.zeros(32);x[7]=32
    summary,rows=decompose(x,.5,SEEDS)
    assert summary["estimate"]==1 and summary["point_outside_margin"]
    assert summary["maximum_share_source_rng_seeds"]=="95008" and summary["omitted_point_decision_changes"]==1
    assert rows[7]["mean_if_block_omitted"]==0 and sum(r["contribution_to_full_mean"] for r in rows)==1


def test_broad_bias_keeps_all_omitted_means_outside():
    summary,rows=decompose(np.full(32,2.),1.,SEEDS)
    assert summary["estimate"]==2 and summary["block_sd"]==0 and summary["omitted_point_decision_changes"]==0
    assert all(r["omitted_mean_outside_margin"] for r in rows)


@pytest.mark.parametrize("values,margin",[(np.zeros(31),1),(np.full(32,np.nan),1),(np.zeros(32),0)])
def test_invalid_samples_fail(values,margin):
    with pytest.raises(ValueError):decompose(values,margin,SEEDS)

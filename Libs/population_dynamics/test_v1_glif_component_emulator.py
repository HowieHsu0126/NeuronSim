import math

import numpy as np
import pytest

from .v1_glif_component_emulator import alpha_propagator, arrival_counts


def test_equal_time_constants_use_closed_form():
    p31, p32 = alpha_propagator(5.0, 5.0, 100.0, 0.25)
    exp = math.exp(-0.25 / 5.0)
    assert p31 == 0.5 * 0.25 * 0.25 / 100.0 * exp
    assert p32 == 0.25 / 100.0 * exp


def test_invalid_propagator_constants_reject():
    with pytest.raises(ValueError, match="PROPAGATOR_DOMAIN"):
        alpha_propagator(0.0, 5.0, 100.0, 0.25)


def test_precise_events_keep_stamp_delay_and_multiplicity():
    counts = arrival_counts(np.array([0.125, 0.249, 0.25, 0.251]), 0.75, 0.25, 8)
    assert counts.tolist() == [0, 0, 0, 3, 1, 0, 0, 0]


def test_arrivals_after_component_stop_do_not_enter_earlier_steps():
    counts = arrival_counts(np.array([1.875]), 0.75, 0.25, 8)
    assert counts.sum() == 0

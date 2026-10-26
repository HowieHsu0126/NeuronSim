"""Native RNG preservation, arbitrary chunk boundaries and causal event checks."""
import numpy as np
import pytest

from . import pfc_native_drive as native
from . import pfc_deterministic_replay as source
from .pfc_first_passage_witness import FocalOverrideDrive, check_event_boundary, bound_prefix_state
from . import pfc_coupled_drive as coupled


def fixture():
    config = dict(dt_ms=.05, duration_ms=5., excitatory_neurons=8, inhibitory_neurons=2,
        excitatory_preference_bins=4, weight_scaling_factor=256., stimulus_start_ms=0.,
        stimulus_duration_ms=0., stimulus_center_deg=120., stimulus_width_deg=30.,
        stimulus_strength_nA=0., distractor_start_ms=0., distractor_duration_ms=0.,
        distractor_center_deg=90., distractor_width_deg=40.)
    case = dict(case_id="override_test", source_rng_seed=95001,
                stimulus_strength_nA=0., distractor_strength_nA=0.)
    return config, case


@pytest.mark.parametrize("chunk", [1, 7, 37, 100])
def test_only_focal_interval_changes_with_native_rng_and_accounting_preserved(chunk):
    config, case = fixture()
    _, a = native.create_drive(config, case); _, b = native.create_drive(config, case)
    witness = np.arange(25, dtype=np.uint16) % 5
    override = FocalOverrideDrive(a, unit_index=3, start_tick=12, stop_tick=37, counts=witness)
    actual = {"E": [], "I": []}; original = {"E": [], "I": []}
    while b.remaining:
        size = min(chunk, b.remaining)
        for target, blocks in ((actual, override.take(size)), (original, b.take(size))):
            for key in target: target[key].append(blocks[key].copy())
    actual = {k: np.concatenate(v) for k, v in actual.items()}
    original = {k: np.concatenate(v) for k, v in original.items()}
    np.testing.assert_array_equal(override.original_counts, original["E"][12:37, 3])
    expected = original["E"].copy(); expected[12:37, 3] = witness
    np.testing.assert_array_equal(actual["E"], expected)
    np.testing.assert_array_equal(actual["I"], original["I"])
    assert a.event_totals == b.event_totals and override.ticks == b.ticks == 100
    assert all(np.array_equal(v, native._state_arrays(b.get_state())[k])
               for k, v in native._state_arrays(override.get_state()).items())
    proof = override.evidence()
    assert int(actual["E"].sum()) == b.event_totals["E"] + proof["event_delta_E"]
    assert int(actual["I"].sum()) == b.event_totals["I"]


def test_identity_override_through_complete_source_preserves_all_arrays():
    config, case = fixture()
    initial, original = native.create_drive(config, case)
    counts = original.take(100)
    baseline = source.simulate(config, case, initial, counts["E"], counts["I"])
    _, stream = native.create_drive(config, case)
    override = FocalOverrideDrive(stream, unit_index=3, start_tick=12, stop_tick=37,
                                 counts=counts["E"][12:37, 3])
    actual = source.simulate(config, case, initial, drive=override)
    for family in ("raw", "terminal_state"):
        for key, value in baseline[family].items():
            np.testing.assert_array_equal(actual[family][key], value)
    assert override.evidence()["event_delta_E"] == 0
    assert actual["inputs_used"]["E"] == baseline["inputs_used"]["E"]


def test_invalid_witness_rejected_before_drawing():
    config, case = fixture(); _, stream = native.create_drive(config, case)
    for counts in ([1], [1., 2.], [-1, 0], [0, 1001], [True, False]):
        with pytest.raises(ValueError, match="COMPLETE_INTEGER_INPUT_COUNTS"):
            FocalOverrideDrive(stream, unit_index=3, start_tick=12, stop_tick=14, counts=counts)
        assert stream.ticks == 0
    with pytest.raises(ValueError, match="FOCAL_E_INDEX"):
        FocalOverrideDrive(stream, unit_index=8, start_tick=12, stop_tick=14, counts=[1, 2])


def test_event_boundary_uses_window_ordinals_and_preserves_later_feedback():
    baseline = np.array([[2, 3], [10, 3], [11, 8], [15, 8], [19, 8]])
    changed = np.array([[2, 3], [10, 3], [11, 8], [15, 3], [15, 8], [18, 1], [19, 3]])
    kwargs = dict(unit_index=3, window_start_tick=10, first_spike_tick=10,
                  expected_hit_tick=15, window_stop_tick=20)
    result = check_event_boundary(baseline, changed, **kwargs)
    assert result["modified_focal_window_ticks"] == [10, 15, 19]
    altered = changed.copy(); altered[2, 1] = 7
    with pytest.raises(ValueError, match="PREHIT_NETWORK_EVENTS_CHANGED"):
        check_event_boundary(baseline, altered, **kwargs)
    altered = changed.copy(); altered[4, 1] = 7
    with pytest.raises(ValueError, match="HIT_TICK_NONFOCAL_EVENTS_CHANGED"):
        check_event_boundary(baseline, altered, **kwargs)
    altered = changed.copy(); altered[3, 0] = 14
    with pytest.raises(ValueError, match="FIRST_ADDITIONAL_SPIKE_MISMATCH"):
        check_event_boundary(baseline, altered, **kwargs)


@pytest.mark.parametrize("dt", [.025, .0125])
@pytest.mark.parametrize("chunk", [1, 7, 37])
def test_fine_override_preserves_all_rngs_and_partial_child_buffers(dt, chunk):
    config, case = fixture(); config["dt_ms"] = dt
    _, a = coupled.create_coupled_drive(config, case, .05, [95411, 95412])
    _, b = coupled.create_coupled_drive(config, case, .05, [95411, 95412])
    counts = np.arange(25, dtype=np.uint16) % 5
    override = FocalOverrideDrive(a, unit_index=3, start_tick=12, stop_tick=37, counts=counts)
    kind = "coupled_binomial_clock_refinement_stream"
    while b.ticks < 51:
        size = min(chunk, 51-b.ticks); start = b.ticks
        actual, original = override.take(size), b.take(size)
        expected = original["E"].copy()
        lo, hi = max(start, 12), min(start+size, 37)
        if lo < hi: expected[lo-start:hi-start, 3] = counts[lo-12:hi-12]
        np.testing.assert_array_equal(actual["E"], expected)
        np.testing.assert_array_equal(actual["I"], original["I"])
        state_a, state_b = bound_prefix_state(a, kind), bound_prefix_state(b, kind)
        assert state_a.keys() == state_b.keys()
        for key in state_a: np.testing.assert_array_equal(state_a[key], state_b[key])
    assert override.evidence()["input_slots"] == 25

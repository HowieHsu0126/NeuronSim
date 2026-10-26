"""Pulse control boundaries are discrete emission times, not arrival times."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from .mesocircuit_continuation import (prepare_pulse_protocol, replace_future_pulses,
                                      configure_update_interval, check_update_segments)
from .mesocircuit_source import load_config

NET = {"thalamic_input": True, "thalamic_input_type": "pulses",
       "th_pulse_start": 3., "th_interval": 7., "th_delay_pulse_generator": 1.}
SIM = {"sim_resolution": .1, "t_presim": 5., "t_sim": 15.}
ORIGINAL = [2., 9., 9.5, 10., 10.1, 11., 14.]


def plan(**kwargs):
    return prepare_pulse_protocol(NET, SIM, dict(initial_emission_ms=ORIGINAL, **kwargs))


def test_emission_at_boundary_and_inflight_arrivals_survive_replacement():
    p = plan(decision_time_ms=10., future_emission_ms=[10.2, 12., 14.])
    r = p.record()
    assert r["committed_emission_ms"] == [2., 9., 9.5, 10.]
    assert r["inflight_tc_arrival_ms"] == [10.5, 11.]
    np.testing.assert_allclose(r["effective_tc_arrival_ms"], [3, 10, 10.5, 11, 11.2, 13, 15], rtol=0, atol=1e-14)
    assert r["full_microscopic_state_qualified"] is False
    assert r["paired_stochastic_future_noise_verified"] is False


def test_omission_empty_and_override_have_distinct_semantics():
    assert prepare_pulse_protocol(NET, SIM, None) is None
    unchanged = plan(decision_time_ms=10.)
    cleared = plan(decision_time_ms=10., future_emission_ms=[])
    assert unchanged.effective_ticks == unchanged.initial_ticks
    assert cleared.effective_ticks == (20, 90, 95, 100)
    empty = prepare_pulse_protocol(NET, SIM, {"initial_emission_ms": []})
    assert empty.initial_ticks == empty.effective_ticks == ()
    native = prepare_pulse_protocol(NET, SIM, {"decision_time_ms": 10.})
    assert native.initial_ticks == (20, 90, 160) and not native.initial_override


@pytest.mark.parametrize("boundary", [5., 19.9])
def test_decision_at_presimulation_and_late_cancellation(boundary):
    p = plan(decision_time_ms=boundary, future_emission_ms=[])
    assert all(t <= p.decision_tick for t in p.effective_ticks)


def test_time_zero_decision_is_supported_without_requiring_history():
    p = prepare_pulse_protocol(NET, dict(SIM, t_presim=0., t_sim=20.),
                               {"decision_time_ms": 0., "future_emission_ms": []})
    assert p.decision_tick == 0 and p.effective_ticks == ()


@pytest.mark.parametrize("settings,error", [
    ({}, "FIELDS_INVALID"), ({"unknown": 1}, "FIELDS_INVALID"),
    ({"future_emission_ms": []}, "REQUIRES_DECISION"),
    ({"decision_time_ms": None}, "FINITE_PULSE_TIME"),
    ({"decision_time_ms": True}, "FINITE_PULSE_TIME"),
    ({"decision_time_ms": float("inf")}, "FINITE_PULSE_TIME"),
    ({"decision_time_ms": 9.95}, "ALIGN_WITH_GRID"),
    ({"decision_time_ms": 4.9}, "AFTER_PRESIMULATION"),
    ({"decision_time_ms": 20.}, "BEFORE_STOP"),
    ({"decision_time_ms": 10., "future_emission_ms": [10.]}, "STRICTLY_FUTURE"),
    ({"decision_time_ms": 10., "future_emission_ms": [np.nextafter(10., 11.)]}, "STRICTLY_FUTURE"),
    ({"initial_emission_ms": [10., np.nextafter(10., 11.)]}, "STRICTLY_INCREASE"),
    ({"initial_emission_ms": [2., 1.]}, "STRICTLY_INCREASE"),
    ({"initial_emission_ms": [0.]}, "OUTSIDE_RECORDED_WINDOW"),
    ({"initial_emission_ms": [19.]}, "OUTSIDE_RECORDED_WINDOW"),
    ({"initial_emission_ms": [float("nan")]}, "FINITE_PULSE_TIME"),
    ({"initial_emission_ms": [[2.]]}, "FINITE_PULSE_TIME"),
    ({"initial_emission_ms": [True]}, "FINITE_PULSE_TIME"),
    ({"initial_emission_ms": None}, "LIST_REQUIRED"),
    ({"decision_time_ms": 10., "future_emission_ms": None}, "LIST_REQUIRED"),
])
def test_invalid_protocol_fails_before_runtime(settings, error):
    with pytest.raises(ValueError, match=error):
        prepare_pulse_protocol(NET, SIM, settings)


@pytest.mark.parametrize("override", [{"thalamic_input": False}, {"thalamic_input_type": "poisson"}])
def test_unsupported_native_control_is_rejected(override):
    with pytest.raises(ValueError, match="REQUIRES_NATIVE_THALAMIC_PULSES"):
        prepare_pulse_protocol(dict(NET, **override), SIM, {"decision_time_ms": 10.})


class Generator:
    def __init__(self, times, *, reject=False, ignore=False):
        self.spike_times = times
        self.calls = 0
        self.reject = reject
        self.ignore = ignore

    def set(self, *, spike_times):
        self.calls += 1
        if self.reject:
            raise RuntimeError("native rejection")
        if not self.ignore:
            self.spike_times = spike_times


def test_runtime_preserves_absent_schedule_and_applies_empty_schedule():
    generator = Generator(ORIGINAL)
    runtime = SimpleNamespace(biological_time=10.)
    result = replace_future_pulses(generator, runtime, plan(decision_time_ms=10.))
    assert not result["future_schedule_replaced"] and generator.calls == 0
    result = replace_future_pulses(generator, runtime, plan(decision_time_ms=10., future_emission_ms=[]))
    assert result["future_schedule_replaced"] and generator.spike_times == [] and generator.calls == 1


def test_wrong_time_or_modified_prefix_never_calls_setter():
    generator = Generator(ORIGINAL)
    p = plan(decision_time_ms=10., future_emission_ms=[12.])
    with pytest.raises(ValueError, match="WRONG_NATIVE_TIME"):
        replace_future_pulses(generator, SimpleNamespace(biological_time=9.9), p)
    generator.spike_times = [2.]
    with pytest.raises(ValueError, match="CHANGED_BEFORE_DECISION"):
        replace_future_pulses(generator, SimpleNamespace(biological_time=10.), p)
    assert generator.calls == 0


@pytest.mark.parametrize("reject", [False, True])
def test_native_setter_rejection_or_wrong_readback_propagates(reject):
    generator = Generator(ORIGINAL, reject=reject, ignore=not reject)
    with pytest.raises((ValueError, RuntimeError), match="native rejection|READBACK_MISMATCH"):
        replace_future_pulses(generator, SimpleNamespace(biological_time=10.),
                              plan(decision_time_ms=10., future_emission_ms=[]))


def test_real_config_loader_checks_continuation_before_native_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/neural_population_mesocircuit.yaml").read_text())
    cfg["source_dir"] = str(root / "external/mesocircuit_model")
    cfg["pulse_control"] = {"decision_time_ms": 60., "future_emission_ms": [70., 90.]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)["pulse_control"] == cfg["pulse_control"]
    cfg["pulse_control"]["future_emission_ms"] = [60.]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="STRICTLY_FUTURE"):
        load_config(path)
    cfg["pulse_control"] = None
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="FIELDS_INVALID"):
        load_config(path)


class Kernel:
    biological_time = 0.
    min_delay = 1.
    max_delay = 2.2

    def __init__(self):
        self.settings = []

    def GetDefaults(self, model):
        assert model == "static_synapse"
        return {"delay": 1.}

    def SetKernelStatus(self, values):
        self.settings.append(values)
        self.min_delay, self.max_delay = values["min_delay"], values["max_delay"]


def bounded_net():
    return dict(NET, delay_type="linear", extent=1., mask_radius=np.full((2,2),.5),
                delay_offset_matrix=np.full((2,2),.5), prop_speed_matrix=np.full((2,2),.3), delay_poisson=1.5)


def test_resolution_update_sets_both_extrema_before_simulation_and_preserves_parameters():
    backend = Kernel();net = bounded_net();before = deepcopy(net)
    record = configure_update_interval(backend, net, SIM, "resolution")
    assert backend.settings == [{"min_delay": .1, "max_delay": 2.2}]
    assert not record["source_native_random_sequence_preserved"]
    for key in net:np.testing.assert_array_equal(net[key], before[key])
    p = plan(decision_time_ms=10.3, future_emission_ms=[10.4])
    check_update_segments(backend, SIM, p, record)
    assert record["segmentation_aligned"]
    assert record["effective_min_delay_ms"] == .1


def test_native_profile_is_noop_and_rejects_unsafe_segment_durations():
    backend = Kernel();record = configure_update_interval(backend, bounded_net(), SIM, "native")
    assert backend.settings == []
    with pytest.raises(ValueError, match="SEGMENTATION_CHANGES_RNG"):
        check_update_segments(backend, SIM, plan(decision_time_ms=10.3), record)
    assert not record["segmentation_aligned"]
    check_update_segments(backend, SIM, plan(decision_time_ms=10.), record)
    assert record["segmentation_aligned"]


def test_each_segment_is_checked_including_presimulation():
    backend = Kernel();record = configure_update_interval(backend, bounded_net(), SIM, "native")
    sim = dict(SIM, t_presim=5.3, t_sim=14.7)
    p = prepare_pulse_protocol(NET, sim, {"decision_time_ms":10.})
    with pytest.raises(ValueError, match="SEGMENTATION_CHANGES_RNG"):
        check_update_segments(backend, sim, p, record)
    assert [x["aligned_with_kernel_min_delay"] for x in record["segments"]] == [False, False, True]


def test_resolution_profile_does_not_truncate_unbounded_normal_delays():
    backend = Kernel()
    with pytest.raises(ValueError, match="BOUNDED_LINEAR"):
        configure_update_interval(backend, dict(bounded_net(),delay_type="normal"), SIM, "resolution")
    assert backend.settings == []


def test_update_profile_cannot_change_after_time_advances():
    backend = Kernel();backend.biological_time = .1
    with pytest.raises(ValueError, match="TIME_ZERO"):
        configure_update_interval(backend, bounded_net(), SIM, "resolution")
    assert backend.settings == []


def test_update_profile_readback_failure_is_not_silently_accepted():
    backend = Kernel();record = configure_update_interval(backend, bounded_net(), SIM, "resolution")
    backend.min_delay = 1.
    with pytest.raises(ValueError, match="READBACK_MISMATCH"):
        check_update_segments(backend, SIM, plan(decision_time_ms=10.), record)

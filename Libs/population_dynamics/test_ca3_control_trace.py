"""Synthetic current fixtures test commands and holds, not scientific retrieval."""
from dataclasses import replace

import numpy as np
import pytest

from . import ca3_control_trace as control
from .ca3_native_observation import Group, Interval


def fixture():
    groups, offset = [], 0
    for i, name in enumerate(control.BACKGROUND_SIGMA):
        n = 825 if name == control.PYRAMIDAL else 2
        groups.append(Group(i, offset, offset+n, 0, i, offset, offset+n, 0, name))
        offset += n
    pattern = np.tile(np.arange(275, dtype=np.int64), 4)
    commands = control.source_commands(pattern, 0)
    return groups, commands


def intervals(groups, commands, mutate=None):
    units = groups[-1].stop
    for t in range(control.DURATION_MS):
        current = np.ones(units, dtype='<f4')
        current[commands['unit_ids'][commands['time_ms'] == t]] = control.SOURCE_SET_CURRENT
        if mutate is not None:
            mutate(t, current)
        yield Interval(t, t+1, np.zeros(units, dtype=np.int64), current.copy(), current)


def test_ordered_assignments_keep_source_duplicates_and_offsets():
    commands = control.source_commands(np.zeros(1100, dtype=np.int64), 9000)
    assert len(commands['unit_ids']) == 3300
    assert commands['unit_ids'][:1100].tolist() == [9000]*1100
    assert commands['unit_ids'][1100:2200].tolist() == [9275]*1100
    assert commands['time_ms'][:55].tolist() == [520]*55
    assert np.all(commands['set_current'] == 10000)


@pytest.mark.parametrize('pattern', [np.zeros(1099, dtype=int), np.full(1100, 275), np.full(1100, -1), np.zeros(1100)])
def test_invalid_source_pattern_rejected(pattern):
    with pytest.raises(ValueError, match='PATTERN'):
        control.source_commands(pattern, 0)


def test_complete_schedule_and_inhibitory_holds():
    groups, commands = fixture()
    rows, checks = control.check_currents(groups, intervals(groups, commands), commands)
    assert checks['ordered_assignments_verified'] == 3*4*138
    assert checks['inhibitory_held_unit_ms_verified'] == 60*14
    assert checks['actual_current_values_checked'] == 1001*839
    assert len(rows) == 7*8
    schedule = control.background_schedule(groups)
    for row in schedule:
        assert row['realized_future_current_model_visible'] is False
        if row['phase'] == 'cue_1' and row['group'] != control.PYRAMIDAL:
            assert row['held_from_ms'] == 519
            assert row['cpp_lognormal_s'] is None


def test_cue_is_assignment_not_addition():
    groups, commands = fixture()
    def mutate(t, current):
        if t == 520:
            current[0] += 1
    with pytest.raises(ValueError, match='CUE_CURRENT_NOT_DELIVERED'):
        control.check_currents(groups, intervals(groups, commands, mutate), commands)


def test_inhibitory_hold_compares_float_bits():
    groups, commands = fixture()
    def mutate(t, current):
        if t == 519:
            current[825] = 0.
        if 520 <= t < 540:
            current[825] = -0.
    with pytest.raises(ValueError, match='HOLD_MISMATCH'):
        control.check_currents(groups, intervals(groups, commands, mutate), commands)


def test_source_background_cap_is_not_applied_to_uncapped_cue_draws():
    groups, commands = fixture()
    def mutate(t, current):
        if t == 520:
            current[824] = 20000
    rows, _ = control.check_currents(groups, intervals(groups, commands, mutate), commands)
    assert next(r for r in rows if r['phase'] == 'cue_1' and r['group'] == control.PYRAMIDAL)['maximum_raw'] == 20000
    def outside(t, current):
        if t == 519:
            current[824] = 20000
    with pytest.raises(ValueError, match='CAP_VIOLATION'):
        control.check_currents(groups, intervals(groups, commands, outside), commands)


def test_complete_time_support_required():
    groups, commands = fixture()
    def short():
        for value in intervals(groups, commands):
            if value.start_ms == 1000:
                return
            yield value
    with pytest.raises(ValueError, match='COMPLETE_SOURCE_SCHEDULE'):
        control.check_currents(groups, short(), commands)
    def shifted():
        for value in intervals(groups, commands):
            yield replace(value, start_ms=value.start_ms+1)
    with pytest.raises(ValueError, match='ORDERED_NATIVE_CLOCK'):
        control.check_currents(groups, shifted(), commands)

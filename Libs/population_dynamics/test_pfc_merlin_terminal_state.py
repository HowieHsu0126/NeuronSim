"""Complete persistent state, source units and field comparison contracts."""
import json

import numpy as np
import pytest

from . import pfc_merlin_terminal_state as terminal
from .pfc_merlin_terminal_state import compare_states


def states():
    names = {"E__" + key for key in ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "s_NMDA", "x",
                "I_stim", "lastspike", "not_refractory", "source_indices", "archive_indices", "spike_counts")}
    names |= {"I__" + key for key in ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "lastspike",
                "not_refractory", "source_indices", "archive_indices", "spike_counts")}
    return {name: np.array([.1], dtype=np.float64) for name in names}


def test_all_source_fields_required():
    x = states(); assert len(compare_states(x, x, "episode")) == 21
    y = dict(x); y.pop("E__x")
    with pytest.raises(ValueError, match="FIELDS_DIFFER"):
        compare_states(x, y, "episode")


def test_synaptic_state_tolerance_and_units_are_separate_from_voltage():
    x = states(); y = {k: a.copy() for k, a in x.items()}; y["E__s_NMDA"] += 1e-6
    rows = compare_states(x, y, "episode")
    row = next(r for r in rows if r["array"] == "E__s_NMDA")
    assert row["units"] == "dimensionless" and row["atol"] == 1e-12 and row["rtol"] == 1e-10
    assert not row["passed"]


def test_refractory_state_uses_exact_comparison():
    x = states(); y = {k: a.copy() for k, a in x.items()}; y["E__lastspike"] += 1e-14
    row = next(r for r in compare_states(x, y, "episode") if r["array"] == "E__lastspike")
    assert row["exact"] if "exact" in row else row["comparison"] == "exact"
    assert not row["passed"] and row["units"] == "s"


CHECKPOINT_ROOT = "hdfs://haruna/checkpoints/20261024/fixture-run"


def owner(circuit=63, block=6, condition=0):
    return dict(circuit_index=circuit, block_index=block, condition_index=condition,
                episode_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{condition:02d}")


@pytest.mark.parametrize("condition", range(10))
@pytest.mark.parametrize("circuit,first_index", [(23, 690), (63, 1890)])
def test_c23_and_c63_complete_condition_families_bind_exact_checkpoint(circuit, first_index, condition):
    prefix = f"terminal_tasks/task_{first_index + condition:04d}"
    assert terminal.checkpoint_path(owner(circuit, 6, condition), CHECKPOINT_ROOT, prefix) == (
        CHECKPOINT_ROOT + "/" + prefix + "/pfc_terminal_state")


@pytest.mark.parametrize("block,index", [(6, 1899), (7, 1909), (8, 1919)])
def test_three_history_blocks_have_distinct_task_indices(block, index):
    prefix = f"terminal_tasks/task_{index:04d}"
    assert terminal.checkpoint_path(owner(63, block, 9), CHECKPOINT_ROOT, prefix) == (
        CHECKPOINT_ROOT + "/" + prefix + "/pfc_terminal_state")


@pytest.mark.parametrize("identity,prefix", [
    (owner(0, 6, 0), "terminal_tasks/task_0000"),
    (owner(99, 8, 9), "terminal_tasks/task_2999"),
])
def test_active_domain_edge_owners_bind_without_shortcuts(identity, prefix):
    assert terminal.checkpoint_path(identity, CHECKPOINT_ROOT, prefix).endswith("/" + prefix + "/pfc_terminal_state")


def test_none_prefix_preserves_old_default_checkpoint():
    assert terminal.checkpoint_path(owner(23, 6, 0), CHECKPOINT_ROOT + "/", None) == (
        CHECKPOINT_ROOT + "/pfc_terminal_state")


@pytest.mark.parametrize("prefix", [
    "terminal_tasks/task_0690", "terminal_tasks/task_1891", "terminal_tasks/task_01890",
    "terminal_tasks/task_1890/", "terminal_tasks//task_1890",
    "../terminal_tasks/task_1890", "/terminal_tasks/task_1890", "terminal_tasks/../task_1890",
    True, 1890, ["terminal_tasks/task_1890"],
])
def test_wrong_owner_index_or_path_prefix_is_rejected(prefix):
    with pytest.raises(ValueError, match="PFC_STATE_TASK_CHECKPOINT_INVALID"):
        terminal.checkpoint_path(owner(), CHECKPOINT_ROOT, prefix)


@pytest.mark.parametrize("field,value", [
    ("circuit_index", -1), ("circuit_index", 100), ("circuit_index", True),
    ("circuit_index", 63.0), ("circuit_index", "63"), ("circuit_index", None),
    ("block_index", 5), ("block_index", 9), ("block_index", 6.0),
    ("block_index", "6"), ("block_index", False),
    ("condition_index", -1), ("condition_index", 10), ("condition_index", 0.0),
    ("condition_index", "0"), ("condition_index", False),
    ("episode_id", "pfc_working_memory_circuit_023_block_006_condition_00"),
])
def test_checkpoint_owner_requires_strict_types_ranges_and_episode_identity(field, value):
    identity = owner()
    identity[field] = value
    with pytest.raises(ValueError, match="PFC_STATE_TASK_CHECKPOINT_INVALID"):
        terminal.checkpoint_path(identity, CHECKPOINT_ROOT, "terminal_tasks/task_1890")


@pytest.mark.parametrize("change", ["prefix", "owner_type", "episode_owner"])
def test_invalid_checkpoint_fails_before_hdfs_archive_or_array_comparison(tmp_path, monkeypatch, change):
    for name, value in (("RH2_JOB_RUN_ID", "fixture-run"), ("ARNOLD_TRIAL_ID", "fixture-trial"),
                        ("ARNOLD_APPLIED_GPU_NUM", "0"), ("RH2_CHECKPOINT_HDFS", CHECKPOINT_ROOT)):
        monkeypatch.setenv(name, value)
    identity = owner()
    prefix = "terminal_tasks/task_1890"
    if change == "prefix":
        prefix = "terminal_tasks/task_0690"
    elif change == "owner_type":
        identity["circuit_index"] = "63"
    else:
        identity["episode_id"] = "pfc_working_memory_circuit_023_block_006_condition_00"
    request = tmp_path / "request.json"
    request.write_text(json.dumps(dict(schema="pfc_persisted_terminal_state_comparison_v1", identity=identity)))
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid checkpoint reached HDFS/archive/array operations")
    monkeypatch.setattr(terminal, "hdfs", forbidden)
    monkeypatch.setattr(terminal, "extract_selected", forbidden)
    monkeypatch.setattr(terminal, "compare_states", forbidden)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="PFC_STATE_TASK_CHECKPOINT_INVALID"):
        terminal.run(request, work, prefix)
    assert not work.exists()

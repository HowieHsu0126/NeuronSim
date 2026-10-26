"""Trace/evaluation contracts; the real AppWorld replay runs on CPU04."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from . import appworld_reference as reference


@pytest.fixture
def config():
    return reference.load_config(Path(__file__).parents[1] / "configs/neural_population_appworld_reference.yaml")


@pytest.mark.parametrize("steps,message", [
    ([{"label": "x", "load_state": "absent"}, {"label": "end", "code": "pass"}], "MUST_EXIST"),
    ([{"label": "x", "save_state": "p"}, {"label": "y", "save_state": "p"}, {"label": "end", "code": "pass"}], "OVERWRITE"),
    ([{"label": "x", "save_state": "p"}], "MUST_END_WITH_CODE"),
    ([{"label": "x", "save_state": "p"}, {"label": "y", "load_state": "p"}], "MUST_END_WITH_CODE"),
    ([{"label": "x", "code": "pass", "save_state": "p"}], "EXACTLY_ONE"),
    ([{"label": "x", "code": "pass"}, {"label": "x", "code": "pass"}], "UNIQUE"),
])
def test_invalid_trace_cannot_overwrite_or_score_stale_state(config, steps, message):
    config["steps"] = steps
    with pytest.raises(ValueError, match=message):
        reference.validate_config(config)


def test_formal_and_test_partitions_are_not_debug_inputs(config):
    for key, value in [("dataset", "test_normal"), ("run_class", "formal"), ("random_seed", True)]:
        altered = deepcopy(config)
        altered[key] = value
        with pytest.raises(ValueError):
            reference.validate_config(altered)


def native_stub(monkeypatch, tmp_path, *, execution_error=False, checkpoint_error=False):
    module = ModuleType("appworld")
    root = [None]

    class World:
        id_to_time_freezer = {}
        closed = 0

        @classmethod
        def close_all(cls):
            cls.closed += 1

        def __init__(self, **kwargs):
            assert kwargs["load_ground_truth"] is False
            self.task = SimpleNamespace(ground_truth=None, instruction="fixture", generator_id="07b42fd",
                                        db_version="0.1.0", datetime=SimpleNamespace(isoformat=lambda: "fixture"))
            self.num_interactions = 0
            self.environment_io = []
            self.terminal = root[0] / "experiments/outputs/reference/tasks/07b42fd_1/dbs"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise AssertionError("Per-instance exit reuses a stale clock after native load_state")

        def execute(self, code):
            self.num_interactions += 1
            self.environment_io.append(code)
            self.terminal.mkdir(parents=True, exist_ok=True)
            (self.terminal / "fixture.json").write_text('{}')
            return "Execution failed. Traceback: fixture" if execution_error else "ok"

        def save_state(self, name):
            if checkpoint_error:
                raise OSError("checkpoint fixture error")
            return name

        def load_state(self, name):
            pass

    module.AppWorld = World
    module.update_root = lambda path: root.__setitem__(0, Path(path))
    tasks = ModuleType("appworld.task")
    tasks.load_task_ids = lambda dataset: ["07b42fd_1"]
    monkeypatch.setitem(sys.modules, "appworld", module)
    monkeypatch.setitem(sys.modules, "appworld.task", tasks)
    monkeypatch.setattr(reference.importlib.metadata, "version", lambda name: reference.APPWORLD_VERSION)
    # Restore these environment keys after each test; caches are disposable local fixtures.
    for name in ("APPWORLD_ROOT", "APPWORLD_CACHE", "TMPDIR"):
        monkeypatch.setenv(name, os.environ.get(name, str(tmp_path)))
    data = tmp_path / "data/datasets"
    data.mkdir(parents=True)
    (data / "train.txt").write_text("07b42fd_1\n")
    return World


def test_active_world_is_not_closed_or_reconfigured(monkeypatch, tmp_path, config):
    world = native_stub(monkeypatch, tmp_path)
    world.id_to_time_freezer = {"existing": object()}
    before = os.environ["APPWORLD_ROOT"]
    with pytest.raises(ValueError, match="NO_ACTIVE_WORLD"):
        reference.run_reference(config, tmp_path, tmp_path / "run", ["fixture"])
    assert world.closed == 0
    assert os.environ["APPWORLD_ROOT"] == before
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("checkpoint_error", [False, True])
def test_failed_operations_keep_trace_and_nonzero_manifest(monkeypatch, tmp_path, config, checkpoint_error):
    native_stub(monkeypatch, tmp_path, execution_error=not checkpoint_error, checkpoint_error=checkpoint_error)
    report, code = reference.run_reference(config, tmp_path, tmp_path / "run", ["fixture"])
    assert code == 1 and report["status"] == "FAILED"
    trace = [json.loads(row) for row in (tmp_path / "run/trace.jsonl").read_text().splitlines()]
    assert trace[-1]["operation_succeeded"] is False
    assert trace[-1]["operation"] == ("save_state" if checkpoint_error else "code")
    assert json.loads((tmp_path / "run/manifest.json").read_text())["exit_code"] == 1


def test_negative_task_result_is_retained_not_an_evaluator_crash(monkeypatch, tmp_path, config):
    native_stub(monkeypatch, tmp_path)
    report, code = reference.run_reference(config, tmp_path, tmp_path / "run", ["fixture"])
    assert code == 2 and report["terminal_databases_persisted"]
    native = ModuleType("appworld.evaluator")

    def evaluate_task(**kwargs):
        assert kwargs["suppress_errors"] is True and kwargs["save_report"] is False
        return SimpleNamespace(to_dict=lambda **unused: {"success": False, "num_tests": 2,
                               "passes": [{"requirement": "unchanged"}], "failures": [{"requirement": "missing artist"}]})

    native.evaluate_task = evaluate_task
    monkeypatch.setitem(sys.modules, "appworld.evaluator", native)
    report, code = reference.evaluate_reference(tmp_path / "run", tmp_path / "evaluation", ["fixture"])
    assert code == 2 and report["native_evaluation"]["success"] is False
    assert len(report["native_evaluation"]["failures"]) == 1
    assert not report["agent_execution_performed"]
    with pytest.raises(FileExistsError):
        reference.run_reference(config, tmp_path, tmp_path / "run", ["fixture"])


@pytest.mark.parametrize("changed_input", ["terminal_database", "source_partition"])
def test_evaluator_rejects_changed_inputs(monkeypatch, tmp_path, config, changed_input):
    native_stub(monkeypatch, tmp_path)
    reference.run_reference(config, tmp_path, tmp_path / "run", ["fixture"])
    if changed_input == "terminal_database":
        changed_path = tmp_path / "run/native/experiments/outputs/reference/tasks/07b42fd_1/dbs/fixture.json"
        message = "ARTIFACT_MISMATCH"
    else:
        changed_path = tmp_path / "data/datasets/train.txt"
        message = "SOURCE_PARTITION_CHANGED"
    changed_path.write_text('changed input')
    with pytest.raises(ValueError, match=message):
        reference.evaluate_reference(tmp_path / "run", tmp_path / "evaluation", ["fixture"])
    assert not (tmp_path / "evaluation").exists()


def test_cleanup_error_retains_completed_trace_but_rejects_run(monkeypatch, tmp_path, config):
    world = native_stub(monkeypatch, tmp_path)

    def fail_cleanup():
        raise RuntimeError("native cleanup failure")

    monkeypatch.setattr(world, "close_all", fail_cleanup)
    report, code = reference.run_reference(config, tmp_path, tmp_path / "run", ["fixture"])
    assert code == 1 and report["status"] == "FAILED"
    assert report["steps_completed"] == len(config["steps"])
    assert report["cleanup_error"] == "native cleanup failure"
    assert json.loads((tmp_path / "run/manifest.json").read_text())["exit_code"] == 1

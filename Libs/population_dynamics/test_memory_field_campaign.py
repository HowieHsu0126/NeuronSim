"""MemNO deployment source staging, delivery and resource-stop semantics."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest


def launcher():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("memory_field_campaign", scripts / "run_memory_field_campaign.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_resource_failure_signals_child_group_and_propagates(tmp_path, monkeypatch):
    module = launcher()
    class Child:
        pid, returncode = 9123, None
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            if getattr(self, "signalled", False):
                self.returncode = -signal.SIGTERM
                return self.returncode
            raise subprocess.TimeoutExpired("fixture", timeout)
    child, signals = Child(), []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kw: child)
    def stop(pid, sig):
        signals.append((pid, sig))
        child.signalled = True
    monkeypatch.setattr(module.os, "killpg", stop)
    def failed(*args):
        raise RuntimeError("ROOT_STORAGE_CRITICAL")
    monkeypatch.setattr(module, "node_probe", failed)
    with pytest.raises(RuntimeError, match="ROOT_STORAGE_CRITICAL"):
        module.supervised_command(["fixture"], tmp_path, {}, tmp_path, tmp_path / "log")
    assert signals == [(9123, signal.SIGTERM)]
    assert child.returncode == -signal.SIGTERM


def test_source_staging_and_declaration_bound_delivery(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    source = tmp_path / "source"
    source.write_text("declared source input")
    prep = tmp_path / "preparation"
    prep.mkdir()
    proof = dict(status="PREPARED_SOURCE_BOUND_MEMNO", files=[], inputs=[dict(source=str(source),
        path="train/source", bytes=source.stat().st_size, sha256=module.digest(source))])
    module.write_json(prep / "preparation.json", proof)
    monkeypatch.setattr(module, "node_probe", lambda *args: None)
    def command(*args):
        work = args[1]
        training = work / "training"
        training.mkdir()
        (training / "run.json").write_text(json.dumps(dict(status="RUNNING")))
        return 0
    monkeypatch.setattr(module, "supervised_command", command)
    work, destination = tmp_path / "work", tmp_path / "delivery"
    report = module.run(prep, work, destination, 41, stop_after_epochs=1)
    assert report["status"] == "PASS"
    receipt = json.loads((destination / "delivery.json").read_text())
    assert receipt["complete_training"] is False
    assert not (destination / "inputs").exists()
    assert (work / "archive_complete").exists()
    for row in receipt["files"]:
        assert module.digest(destination / row["path"]) == row["sha256"]


def test_changed_source_rejects_before_training(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    prep = tmp_path / "preparation"
    prep.mkdir()
    source = tmp_path / "source"
    source.write_text("original")
    module.write_json(prep / "preparation.json", dict(status="PREPARED_SOURCE_BOUND_MEMNO", files=[],
        inputs=[dict(source=str(source), path="train/source", bytes=source.stat().st_size,
                     sha256=module.digest(source))]))
    source.write_text("modified")
    monkeypatch.setattr(module, "node_probe", lambda *args: None)
    monkeypatch.setattr(module, "supervised_command", lambda *args: pytest.fail("training must not start"))
    with pytest.raises(ValueError, match="source manifest identity changed"):
        module.run(prep, tmp_path / "work", tmp_path / "delivery", 42, stop_after_epochs=1)

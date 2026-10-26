"""Synthetic host-binding tests; no native task or world is constructed."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import yaml

from population_dynamics import appworld_benchmark_native_binding as binding


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    config = binding.load_benchmark_config(
        Path(__file__).resolve().parents[1] / "configs/neural_population_appworld_benchmark.yaml"
    )
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "native_fixture.py"
    source_file.write_bytes(b"# synthetic frozen source\n")
    archive = {
        "schema": "appworld_source_archive_manifest_v1", "repository": "StonyBrookNLP/appworld",
        "revision": config["source"]["inspected_revision"],
        "files": {"native_fixture.py": {
            "bytes": source_file.stat().st_size,
            "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        }},
    }
    archive_path = tmp_path / "archive.json"
    archive_path.write_text(json.dumps(archive))
    config["source"]["archive_manifest"] = {
        "path": str(archive_path), "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    interpreter = tmp_path / "native-env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 99\n")
    interpreter.chmod(0o700)
    data = tmp_path / "benchmark"
    (data / "data" / "datasets").mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    packages = {"appworld": config["source"]["candidate_package_version"],
                "appworld-agents": "0.1.0.dev0", "pydantic": "2.12.4", "fastapi": "0.120.0"}
    frozen = "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))
    frozen += "synthetic-extra==0.1\n"
    native = {
        "status": "PREPARED_RUNTIME_AND_SPLIT_METADATA", "native_python": str(interpreter),
        "source_checkout": str(source), "source_revision": config["source"]["inspected_revision"],
        "requested_data_version": config["source"]["candidate_data_version"],
        "packages": packages, "package_freeze_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
        "data_root": str(data), "appworld_cache": str(cache), "hostname": "previous-preparation-host",
        "execution_backend": "merlin", "run_id": "previous-run", "trial_id": "previous-trial",
    }
    manifest = {"schema": "appworld_split_membership_v1", "appworld_version": packages["appworld"],
                "data_root": str(data), "splits": {}}
    inspected = {"executable": str(interpreter), "prefix": str(interpreter.parent.parent),
                 "base_prefix": "/synthetic-base-python", "packages": deepcopy(packages),
                 "package_freeze_sha256": native["package_freeze_sha256"], "splits": {}}
    for split in ("train", "dev", "test_normal", "test_challenge"):
        membership = data / "data" / "datasets" / f"{split}.txt"
        # Raw native membership need not be a line-for-line list of expanded IDs.
        membership.write_text(split + "case\n")
        ids = [split + "case_" + str(variant) for variant in (1, 2, 3)]
        manifest["splits"][split] = {
            "membership_file": str(membership), "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
            "task_ids": ids, "task_count": 3, "scenario_count": 1,
        }
        inspected["splits"][split] = {"task_ids": list(ids)}
    original = {
        "status": "PASS_DATA_VERSION_AND_NATIVE_SPLIT_ADMISSION", "exit_code": 0,
        "environment": native, "hostname": native["hostname"], "run_id": native["run_id"],
        "trial_id": native["trial_id"], "splits": {
            split: {key: entry[key] for key in ("membership_sha256", "task_count", "scenario_count")}
            for split, entry in manifest["splits"].items()
        },
    }
    paths = {"config_path": tmp_path / "config.yaml", "native_environment": tmp_path / "preparation.json",
             "split_manifest": tmp_path / "splits.json", "output": tmp_path / "bound.json"}
    paths["config_path"].write_text(yaml.safe_dump(config))
    paths["native_environment"].write_text(json.dumps(original))
    paths["split_manifest"].write_text(json.dumps(manifest))
    state = SimpleNamespace(paths=paths, config=config, original=original, native=native,
                            manifest=manifest, inspected=inspected, source_file=source_file,
                            calls=[], locations=[], inspection_failure=False)

    def identity(require_allocation=False):
        assert require_allocation is True
        return {"execution_backend": "merlin", "hostname": socket.gethostname(),
                "run_id": "current-run", "trial_id": "current-trial",
                "checkpoint_path": "hdfs://fixture/checkpoints/current-run"}

    def inspect(command, **kwargs):
        state.calls.append((command, kwargs))
        if state.inspection_failure:
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic native import failure")
        result = deepcopy(state.inspected)
        for split, entry in result["splits"].items():
            membership = data / "data" / "datasets" / f"{split}.txt"
            payload = membership.read_bytes()
            entry.update(membership_file=str(membership), membership_bytes=len(payload),
                         membership_sha256=hashlib.sha256(payload).hexdigest())
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(binding, "execution_identity", identity)
    monkeypatch.setattr(binding, "_check_execution_location", lambda path: state.locations.append(path))
    monkeypatch.setattr(binding.subprocess, "run", inspect)
    return state


def test_binding_on_another_host_preserves_original_and_paths(native_fixture, monkeypatch):
    state = native_fixture
    monkeypatch.setenv("PYTHONPATH", "/synthetic-caller-only")
    before_environment = dict(os.environ)
    original_bytes = state.paths["native_environment"].read_bytes()
    report = binding.bind_native_runtime(**state.paths)
    assert report["status"] == "PREPARED_RUNTIME_AND_SPLIT_METADATA"
    assert report["hostname"] == socket.gethostname() != state.native["hostname"]
    assert report["run_id"] == "current-run" and report["trial_id"] == "current-trial"
    assert report["original_preparation"]["receipt"] == state.original
    assert state.paths["native_environment"].read_bytes() == original_bytes
    for key in ("native_python", "source_checkout", "data_root", "appworld_cache"):
        assert report[key] == state.native[key]
    assert report["formal_eligible"] is False and report["task_worlds_created"] == 0
    assert report["task_contents_read"] is False
    assert len(state.calls) == 1
    command, kwargs = state.calls[0]
    assert command[:4] == [state.native["native_python"], "-I", "-B", "-c"]
    assert kwargs["env"]["APPWORLD_CACHE"] == state.native["appworld_cache"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "PYTHONPATH" not in kwargs["env"]
    assert dict(os.environ) == before_environment
    assert json.loads(kwargs["input"])["splits"] == ["train", "dev", "test_normal", "test_challenge"]
    assert all(entry["native_id_order_matches"] for entry in report["verification"]["memberships"].values())
    assert state.locations[0] == state.paths["output"]
    assert json.loads(state.paths["output"].read_text()) == report


@pytest.mark.parametrize("defect", ["package", "freeze", "executable", "prefix", "membership_order", "membership_bytes"])
def test_binding_rejects_runtime_or_membership_mismatch_without_receipt(native_fixture, defect):
    state = native_fixture
    if defect == "package":
        state.inspected["packages"]["pydantic"] = "different"
    elif defect == "freeze":
        state.inspected["package_freeze_sha256"] = "0" * 64
    elif defect in {"executable", "prefix"}:
        state.inspected[defect] = "/different/" + defect
    elif defect == "membership_order":
        state.inspected["splits"]["train"]["task_ids"].reverse()
    else:
        path = Path(state.manifest["splits"]["train"]["membership_file"])
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        binding.bind_native_runtime(**state.paths)
    assert not state.paths["output"].exists()


def test_binding_rejects_source_change_before_native_inspection(native_fixture):
    state = native_fixture
    state.source_file.write_text("# changed source\n")
    with pytest.raises(ValueError, match="bound bytes"):
        binding.bind_native_runtime(**state.paths)
    assert not state.calls and not state.paths["output"].exists()


def test_binding_rejects_manifest_divergence_from_original_admission(native_fixture):
    state = native_fixture
    state.manifest["splits"]["train"]["membership_sha256"] = "0" * 64
    state.paths["split_manifest"].write_text(json.dumps(state.manifest))
    with pytest.raises(ValueError, match="original data admission"):
        binding.bind_native_runtime(**state.paths)
    assert not state.calls and not state.paths["output"].exists()


def test_binding_rejects_existing_output_before_inspection(native_fixture):
    state = native_fixture
    state.paths["output"].write_text("preserved receipt\n")
    with pytest.raises(FileExistsError):
        binding.bind_native_runtime(**state.paths)
    assert not state.calls
    assert state.paths["output"].read_text() == "preserved receipt\n"


def test_binding_cli_preserves_native_inspection_failure(native_fixture, capsys):
    state = native_fixture
    state.inspection_failure = True
    arguments = []
    for flag, key in (("config", "config_path"), ("native-environment", "native_environment"),
                      ("split-manifest", "split_manifest"), ("output", "output")):
        arguments += ["--" + flag, str(state.paths[key])]
    assert binding.main(arguments) == 2
    assert "synthetic native import failure" in capsys.readouterr().err
    assert not state.paths["output"].exists()


def test_binding_rejects_non_merlin_allocation(native_fixture, monkeypatch):
    state = native_fixture
    monkeypatch.setattr(binding, "execution_identity", lambda **kwargs: {"execution_backend": "slurm"})
    with pytest.raises(ValueError, match="genuine Merlin allocation"):
        binding.bind_native_runtime(**state.paths)
    assert not state.calls and not state.paths["output"].exists()

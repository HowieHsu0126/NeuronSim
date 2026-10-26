"""Bind a restored native AppWorld runtime to its current allocated host."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .appworld_benchmark_development import _admit_source, load_benchmark_config
from .appworld_benchmark_execution import execution_identity
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, validate_split_manifest
from .appworld_benchmark_runner import _check_execution_location


_PACKAGES = ("appworld", "appworld-agents", "pydantic", "fastapi")
_SPLITS = ("train", "dev", "test_normal", "test_challenge")
_NATIVE_INSPECTION = r"""
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

request = json.load(sys.stdin)
result = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "packages": {name: importlib.metadata.version(name) for name in request["packages"]},
    "splits": {},
}
frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
result["package_freeze_sha256"] = hashlib.sha256(frozen.encode()).hexdigest()
with contextlib.redirect_stdout(sys.stderr):
    from appworld import update_root
    from appworld.task import load_task_ids
    root = Path(request["data_root"])
    update_root(str(root.resolve()))
    for split in request["splits"]:
        membership_file = root / "data" / "datasets" / (split + ".txt")
        payload = membership_file.read_bytes()
        result["splits"][split] = {
            "membership_file": str(membership_file),
            "membership_bytes": len(payload),
            "membership_sha256": hashlib.sha256(payload).hexdigest(),
            "task_ids": list(load_task_ids(split)),
        }
print(json.dumps(result))
"""


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: a JSON object is required")
    return value, hashlib.sha256(payload).hexdigest()


def _original_environment(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("status") == "PREPARED_RUNTIME_AND_SPLIT_METADATA":
        return document
    if (document.get("status") == "PASS_DATA_VERSION_AND_NATIVE_SPLIT_ADMISSION"
            and type(document.get("exit_code")) is int and document["exit_code"] == 0
            and isinstance(document.get("environment"), dict)
            and document["environment"].get("status") == "PREPARED_RUNTIME_AND_SPLIT_METADATA"):
        return document["environment"]
    raise ValueError("a completed native runtime and split preparation receipt is required")


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must retain its original absolute path")
    return Path(value)


def _inspect_native(native: dict[str, Any]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        APPWORLD_CACHE=native["appworld_cache"], TMPDIR=str(Path(native["data_root"]).parent),
        CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    request = {"data_root": native["data_root"], "packages": list(_PACKAGES), "splits": list(_SPLITS)}
    completed = subprocess.run(
        [native["native_python"], "-I", "-B", "-c", _NATIVE_INSPECTION],
        input=json.dumps(request), text=True, capture_output=True, check=False,
        cwd=native["data_root"], env=environment, timeout=120,
    )
    if completed.returncode:
        raise ValueError(f"native runtime inspection failed ({completed.returncode}): {completed.stderr[-4000:]}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise ValueError("native runtime inspection must return a JSON object")
    return result


def bind_native_runtime(
    *, config_path: Path, native_environment: Path, split_manifest: Path, output: Path,
) -> dict[str, Any]:
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("an existing native binding receipt cannot be overwritten")
    identity = execution_identity(require_allocation=True)
    if identity["execution_backend"] != "merlin":
        raise ValueError("restored native binding requires a genuine Merlin allocation")
    _check_execution_location(output)
    config = load_benchmark_config(config_path)
    original, original_sha256 = _read_object(native_environment)
    native = _original_environment(original)
    if "splits" in original and not isinstance(original["splits"], dict):
        raise ValueError("original split admission summary must be a mapping")
    manifest, manifest_sha256 = _read_object(split_manifest)
    if config.get("seeds") != list(EXPERIMENT_SEEDS) or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("canonical seeds must remain exactly 41, 42, 43")
    if (native.get("source_revision") != config["source"]["inspected_revision"]
            or native.get("requested_data_version") != config["source"]["candidate_data_version"]):
        raise ValueError("original native source or data version differs from the canonical config")
    packages = native.get("packages", {})
    if not isinstance(packages, dict) or any(not isinstance(packages.get(name), str) or not packages[name] for name in _PACKAGES):
        raise ValueError("all four original native package versions are required")
    if (manifest.get("schema") != "appworld_split_membership_v1"
            or manifest.get("appworld_version") != packages["appworld"]
            or packages["appworld"] != config["source"]["candidate_package_version"]
            or manifest.get("data_root") != native.get("data_root")):
        raise ValueError("native preparation and admitted split manifest identities differ")
    validate_split_manifest(manifest)
    if set(manifest["splits"]) != set(_SPLITS):
        raise ValueError("all four original AppWorld splits are required")
    freeze_hash = native.get("package_freeze_sha256")
    if not isinstance(freeze_hash, str) or re.fullmatch(r"[0-9a-f]{64}", freeze_hash) is None:
        raise ValueError("the original full pip-freeze SHA-256 is required")
    paths = {name: _absolute_path(native.get(name), name)
             for name in ("native_python", "source_checkout", "data_root", "appworld_cache")}
    interpreter = paths["native_python"]
    if interpreter.parent.name != "bin" or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("the original native environment interpreter is unavailable")
    prefix = interpreter.parent.parent
    for directory in (prefix, paths["source_checkout"], paths["data_root"], paths["appworld_cache"]):
        if not directory.is_dir():
            raise ValueError(f"restored native directory is unavailable: {directory}")
        _check_execution_location(directory / "native_binding_location_check")
    if "archive_manifest" not in config["source"]:
        raise ValueError("restored native binding requires explicit source.archive_manifest")
    source = _admit_source(config, paths["source_checkout"])
    for split, entry in manifest["splits"].items():
        membership_file = paths["data_root"] / "data" / "datasets" / f"{split}.txt"
        if membership_file.is_symlink() or not membership_file.is_file():
            raise ValueError(f"{split}: native membership file is unavailable or a symlink")
        if not membership_file.resolve().is_relative_to(paths["data_root"].resolve()):
            raise ValueError(f"{split}: membership file escapes the original data root")
        if "membership_file" in entry and entry["membership_file"] != str(membership_file):
            raise ValueError(f"{split}: admitted membership path differs from the native path")
        if "splits" in original:
            admitted = original["splits"].get(split, {})
            if any(admitted.get(key) != entry.get(key)
                   for key in ("membership_sha256", "task_count", "scenario_count")):
                raise ValueError(f"{split}: manifest differs from original data admission")
    inspection = _inspect_native(native)
    if (inspection.get("executable") != str(interpreter)
            or inspection.get("prefix") != str(prefix)):
        raise ValueError("actual native interpreter or environment prefix differs from preparation")
    if inspection.get("packages") != {name: packages[name] for name in _PACKAGES}:
        raise ValueError("actual native package versions differ from preparation")
    if inspection.get("package_freeze_sha256") != freeze_hash:
        raise ValueError("actual native full pip-freeze hash differs from preparation")
    observed_splits = inspection.get("splits")
    if not isinstance(observed_splits, dict) or set(observed_splits) != set(_SPLITS):
        raise ValueError("native inspection did not return all four splits")
    memberships = {}
    for split, entry in manifest["splits"].items():
        observed = observed_splits[split]
        expected_path = paths["data_root"] / "data" / "datasets" / f"{split}.txt"
        if (not isinstance(observed, dict) or observed.get("membership_file") != str(expected_path)
                or type(observed.get("membership_bytes")) is not int or observed["membership_bytes"] < 0
                or observed.get("membership_sha256") != entry["membership_sha256"]
                or observed.get("task_ids") != entry["task_ids"]):
            raise ValueError(f"{split}: native membership bytes or exact loader order differ from admission")
        memberships[split] = {
            "membership_file": str(expected_path), "membership_bytes": observed["membership_bytes"],
            "membership_sha256": observed["membership_sha256"], "native_id_order_matches": True,
            "task_count": entry["task_count"], "scenario_count": entry["scenario_count"],
        }
    receipt = deepcopy(native)
    for key in ("execution_backend", "recovery_backend", "hostname", "run_id", "trial_id",
                "checkpoint_path", "slurm_job_id", "job_id"):
        receipt.pop(key, None)
    receipt.update(
        schema="appworld_native_runtime_binding_v1", status="PREPARED_RUNTIME_AND_SPLIT_METADATA",
        scope="restored_runtime_and_split_membership_only", formal_eligible=False,
        formal_evaluation_released=False, task_contents_read=False, task_worlds_created=0,
        model_generations=0, recorded_utc=datetime.now(timezone.utc).isoformat(), **identity,
        original_preparation={"path": str(native_environment.absolute()), "sha256": original_sha256,
                              "receipt": deepcopy(original)},
        split_manifest={"path": str(split_manifest.absolute()), "sha256": manifest_sha256},
        verification={"source": source,
                      "runtime": {key: value for key, value in inspection.items() if key != "splits"},
                      "memberships": memberships},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "native-environment", "split-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = bind_native_runtime(
            config_path=arguments.config, native_environment=arguments.native_environment,
            split_manifest=arguments.split_manifest, output=arguments.output,
        )
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"FAILED_NATIVE_BINDING: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "output": str(arguments.output),
                      "execution_backend": report["execution_backend"], "hostname": report["hostname"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

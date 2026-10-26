#!/usr/bin/env python3
"""Prepare and persist CUGA on Merlin CPU without running benchmark tasks.

The inputs JSON uses the existing bootstrap files mapping of relative Libs paths
to content_base64/bytes/sha256 records. Supply the cached prior launcher separately.
Large runtime archives stay on HDFS. The emitted environment remains bound to the
actual CPU host; restoration elsewhere needs a separate verified runtime binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import tarfile
import time
import traceback
from typing import Any

from scripts.run_appworld_benchmark_merlin import (
    HdfsStore, declared_content, file_identity, restore_archive, run_command, write_json,
)

PRIOR_LAUNCHER_BYTES = 811636
PRIOR_LAUNCHER_SHA256 = "489ea4b9a5e99738741d3839611efdb0a6f9a05d663f7da8001f8e1678446a26"
PREPARATION_BOUNDARY = '\nif [ "$code" = 0 ]; then '
PREPARATION_EXIT = '\nexit "$code"\n'
CUGA_REVISION = "cbbdf40f8cb99a56a417b3957139a103e7096890"
PREPARATION_STATUS = "CUGA_PACKAGES_PREPARED_NATIVE_INTEGRATION_REQUIRED"
LEGACY_REPOSITORY = Path("/workspace/NeuronSim")
LEGACY_WORK = Path("/tmp/neuronsim-appworld-cuga")
LEGACY_SCRATCH = Path("/data/neuronsim/neuronsim_appworld_benchmark")
IMAGE_PYTHON_BASE = Path("/opt/python/cpython-3.13.12-linux-x86_64-gnu")
IMAGE_PYTHON_VERSION = "3.13.12"
IMAGE_PYTHON_ALIAS = Path("/opt/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13")
BENCHMARK_OUTPUT = Path("Output/neural_population_dynamics/agentic_extension/benchmark")
CHECK_SCRIPTS = ("validate_appworld_cuga_tools.sbatch", "validate_appworld_cuga_runtime.sbatch")
CHECK_MODULES = (
    "__init__.py", "provenance.py", "resource_probe.py", "appworld_reference.py",
    "appworld_collaboration.py", "appworld_action_decoder.py", "appworld_local_actor.py",
    "appworld_benchmark_protocol.py", "appworld_benchmark_execution.py",
    "appworld_benchmark_runner.py", "appworld_benchmark_transport.py",
    "appworld_benchmark_development.py", "appworld_benchmark_collaboration.py",
    "appworld_benchmark_native_tools.py", "appworld_benchmark_cuga_channel.py",
    "appworld_benchmark_cuga_process.py", "appworld_benchmark_cuga_runtime.py",
    "appworld_benchmark_cuga_tools.py", "appworld_benchmark_cuga_runtime_check.py",
    "test_appworld_benchmark_cuga_tools.py",
)
REQUIRED_FILES = frozenset(
    [f"Libs/scripts/{name}" for name in CHECK_SCRIPTS]
    + [f"Libs/population_dynamics/{name}" for name in CHECK_MODULES]
)
MERLIN_CPU_CONDITION = (
    "not os.environ.get('RH2_JOB_RUN_ID') or "
    "not os.environ.get('ARNOLD_TRIAL_ID') or "
    "os.environ.get('ARNOLD_APPLIED_GPU_NUM') != '0'"
)


def require_merlin_cpu(environment: dict[str, str]) -> dict[str, Any]:
    run_id, trial_id = environment.get("RH2_JOB_RUN_ID", ""), environment.get("ARNOLD_TRIAL_ID", "")
    checkpoint = environment.get("RH2_CHECKPOINT_HDFS", "").rstrip("/")
    hostname = socket.gethostname()
    if (
        not run_id or not trial_id or "/" in run_id or "/" in trial_id
        or environment.get("ARNOLD_APPLIED_GPU_NUM") != "0"
        or "SLURM_JOB_ID" in environment
        or hostname.split(".")[0].lower() == "mgmt2"
        or not checkpoint.startswith("hdfs://") or checkpoint.rsplit("/", 1)[-1] != run_id
    ):
        raise ValueError("CUGA_PREPARATION_REQUIRES_REAL_MERLIN_CPU_ALLOCATION")
    return {
        "execution_backend": "merlin", "run_id": run_id, "trial_id": trial_id,
        "hostname": hostname, "allocated_gpus": 0,
        "allocated_cpus": environment.get("ARNOLD_HEAD_CPU"),
        "checkpoint_path": checkpoint + "/appworld/cuga_runtime_preparation",
        "compatibility_execution_id": f"merlin_{trial_id}_cuga",
    }


def prepare_bootstrap_environment(
    environment: dict[str, str], *, work: Path, output: Path,
    store: HdfsStore, report: dict[str, Any],
) -> dict[str, str]:
    """Give the unchanged prior prefix a private, writable pip and uv location."""
    prefix = work / "bootstrap-env"
    run_command(
        "create_bootstrap_environment",
        [sys.executable, "-m", "venv", str(prefix)],
        environment, work, output, store, report, [],
    )
    cache = work / "uv-cache"
    cache.mkdir(exist_ok=True)
    report["bootstrap_environment"] = {
        "prefix": str(prefix), "python": str(prefix / "bin/python3"),
        "uv_cache_directory": str(cache),
        "created_by": sys.executable, "purpose": "private_pip_and_uv_installation",
    }
    return dict(
        environment, VIRTUAL_ENV=str(prefix), UV_CACHE_DIR=str(cache),
        PATH=str(prefix / "bin") + os.pathsep + environment.get("PATH", ""),
    )


def preparation_prefix(path: Path) -> bytes:
    """Keep the verified install/adaptation/prepare section byte-for-byte."""
    payload = path.read_bytes()
    if len(payload) != PRIOR_LAUNCHER_BYTES or hashlib.sha256(payload).hexdigest() != PRIOR_LAUNCHER_SHA256:
        raise ValueError("PRIOR_CUGA_LAUNCHER_IDENTITY_DIFFERS")
    text = payload.decode("utf-8")
    if text.count(PREPARATION_BOUNDARY) != 1:
        raise ValueError("PRIOR_CUGA_PREPARATION_BOUNDARY_DIFFERS")
    prefix = text.split(PREPARATION_BOUNDARY, 1)[0]
    if not prefix.endswith("code=$?"):
        raise ValueError("PRIOR_CUGA_PREPARATION_EXIT_CAPTURE_DIFFERS")
    return (prefix + PREPARATION_EXIT).encode("utf-8")



def staging_prefix(path: Path) -> bytes:
    """Reuse only the verified source/fixture bootstrap, without installation."""
    prefix = preparation_prefix(path).decode("utf-8")
    start, end = "python3 - <<'CUGA_BOOTSTRAP'\n", "\nCUGA_BOOTSTRAP\n"
    if prefix.count(start) != 1 or prefix.count(end) != 1:
        raise ValueError("PRIOR_CUGA_STAGING_BOUNDARY_DIFFERS")
    offset = prefix.index(start)
    stop = prefix.index(end, offset) + len(end)
    install_boundary = "\nset +e\n"
    if prefix.count(install_boundary) != 1:
        raise ValueError("PRIOR_CUGA_DIRECTORY_SETUP_BOUNDARY_DIFFERS")
    setup_stop = prefix.index(install_boundary, stop)
    # Retain the pinned launcher's directory ownership setup before extraction.
    # Package installation starts after this boundary and remains excluded.
    return ("#!/bin/bash\nset -euo pipefail\nmkdir -p "
            + str(LEGACY_WORK) + "\n" + prefix[offset:setup_stop] + "\n").encode("utf-8")

def stage_current_sources(inputs: dict[str, Any], destination: Path) -> dict[str, dict[str, Any]]:
    files = inputs.get("files")
    if not isinstance(files, dict) or not REQUIRED_FILES.issubset(files):
        raise ValueError("CURRENT_CUGA_CHECK_SOURCE_SET_REQUIRED")
    identities = {}
    for name, record in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "Libs":
            raise ValueError("UNSAFE_CURRENT_SOURCE_PATH: " + name)
        payload = declared_content(record)
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        identities[name] = {"bytes": len(payload), "sha256": record["sha256"]}
    return identities


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("CUGA_PLATFORM_ADAPTATION_ANCHOR_DIFFERS: " + old)
    return text.replace(old, new, 1)


def adapt_current_checks(repository: Path) -> dict[str, dict[str, str]]:
    """Apply the established Merlin CPU/storage adaptation to current checks."""
    paths = [repository / "Libs/scripts" / name for name in CHECK_SCRIPTS]
    paths.append(repository / "Libs/population_dynamics/resource_probe.py")
    records = {}
    for path in paths:
        original = path.read_bytes()
        text = original.decode("utf-8")
        if path.name == "resource_probe.py":
            text = replace_once(
                text, 'if not os.environ.get("SLURM_JOB_ID"):',
                'if not os.environ.get("RH2_JOB_RUN_ID") or not os.environ.get("ARNOLD_TRIAL_ID") '
                'or os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0":',
            )
            text = replace_once(text, "NODE_RESOURCE_PROBE_REQUIRES_SLURM", "NODE_RESOURCE_PROBE_REQUIRES_MERLIN_CPU")
            text = replace_once(
                text, 'cpus_requested=os.environ.get("SLURM_CPUS_PER_TASK"), samples=samples,',
                'cpus_requested=os.environ.get("ARNOLD_HEAD_CPU"), execution_backend="merlin", '
                'merlin_run_id=os.environ["RH2_JOB_RUN_ID"], '
                'merlin_trial_id=os.environ["ARNOLD_TRIAL_ID"], samples=samples,',
            )
        else:
            text = replace_once(text, "/opt/conda/bin/python - ", "python3 - ")
            if path.name == "validate_appworld_cuga_runtime.sbatch":
                text = replace_once(
                    text, "if (hostname not in {'cpu03', 'cpu04'} or",
                    "if (" + MERLIN_CPU_CONDITION + " or",
                )
            text = replace_once(
                text,
                "if filesystem not in {'ext4', 'xfs', 'btrfs', 'ext3', 'ext2'} or shutil.disk_usage(base).free < 512*1024**2:",
                "if " + MERLIN_CPU_CONDITION
                + " or filesystem not in {'ext4', 'xfs', 'btrfs', 'ext3', 'ext2', 'overlay'}"
                + " or shutil.disk_usage(base).free < 512*1024**2:",
            )
        payload = text.encode("utf-8")
        path.write_bytes(payload)
        records[path.relative_to(repository).as_posix()] = {
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "adapted_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return records


RUNTIME_IDENTITY = """
import importlib.metadata, json, pathlib, platform, socket, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
source = pathlib.Path(receipt['source_checkout']).resolve(strict=True)
actual = {
    'executable': str(pathlib.Path(sys.executable).absolute()),
    'prefix': str(pathlib.Path(sys.prefix).resolve(strict=True)),
    'base_prefix': str(pathlib.Path(sys.base_prefix).resolve(strict=True)),
    'python_version': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in receipt['packages']},
    'hostname': socket.gethostname(),
    'source_checkout': str(source),
}
if receipt['hostname'].split('.')[0] != actual['hostname'].split('.')[0]:
    raise ValueError('CUGA preparation belongs to another host')
if actual['executable'] != receipt['python'] or actual['prefix'] != str(source / '.venv'):
    raise ValueError('CUGA interpreter differs from prepared source/.venv')
if actual['packages'] != receipt['packages'] or actual['python_version'] != receipt['python_version']:
    raise ValueError('CUGA prepared package or Python identity changed')
if not (source / 'uv.lock').is_file() or not (source / 'pyproject.toml').is_file():
    raise ValueError('CUGA pinned source/lock is missing')
pathlib.Path(sys.argv[2]).write_text(json.dumps(actual, indent=2) + '\\n')
"""


def declared_runtime_roots(
    environment: dict[str, Any], identity: dict[str, Any],
) -> tuple[Path, Path]:
    """Admit the original managed base or the verified fixed-image Python."""
    if environment.get("status") != PREPARATION_STATUS or environment.get("source_revision") != CUGA_REVISION:
        raise ValueError("COMPLETED_PINNED_CUGA_PREPARATION_REQUIRED")
    source = Path(environment["source_checkout"])
    base = Path(identity["base_prefix"])
    managed_base = base != LEGACY_WORK / "python" and base.is_relative_to(LEGACY_WORK / "python")
    image_base = (
        base == IMAGE_PYTHON_BASE
        and identity.get("python_version") == IMAGE_PYTHON_VERSION
        and environment.get("python_version") == IMAGE_PYTHON_VERSION
    )
    if (
        not source.is_absolute() or not base.is_absolute()
        or ".." in source.parts or ".." in base.parts
        or source == LEGACY_SCRATCH or not source.is_relative_to(LEGACY_SCRATCH)
        or not (managed_base or image_base)
    ):
        raise ValueError("CUGA_RUNTIME_ROOT_DIFFERS_FROM_ORIGINAL_RECIPE")
    if (
        identity.get("source_checkout") != str(source)
        or identity.get("prefix") != str(source / ".venv")
        or identity.get("executable") != environment.get("python")
        or Path(environment["python"]).parent != source / ".venv/bin"
        or identity.get("python_version") != environment.get("python_version")
        or identity.get("packages") != environment.get("packages")
    ):
        raise ValueError("CUGA_RUNTIME_PREFIX_IDENTITY_DIFFERS")
    return source, base


def runtime_roots(environment: dict[str, Any], identity: dict[str, Any]) -> tuple[Path, Path]:
    roots = declared_runtime_roots(environment, identity)
    if any(root.resolve(strict=True) != root or not root.is_dir() for root in roots):
        raise ValueError("CUGA_RUNTIME_ROOT_MUST_BE_ORIGINAL_REAL_DIRECTORY")
    return roots


def restore_prepared_runtime(
    origin: dict[str, Any], identity: dict[str, Any], *,
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any],
    environment_path: Path,
) -> None:
    """Restore verified packages and bind them only after a real host probe."""
    if (origin.get("schema") != "appworld_cuga_merlin_preparation_v1"
            or origin.get("persisted") is not True
            or origin.get("runtime_prepared") is not True
            or origin.get("runtime_persisted") is not True
            or not origin.get("run_id") or not origin.get("trial_id")):
        raise ValueError("PERSISTED_CUGA_RUNTIME_RECEIPT_REQUIRED")
    environment = origin["environment"]
    original_identity = origin["runtime_identity"]
    roots = declared_runtime_roots(environment, original_identity)
    descriptor = origin["runtime_bundle"]
    if (origin["runtime_archive"]["roots"] != [str(root) for root in roots]
            or not descriptor["uri"].startswith("hdfs://")
            or type(descriptor["bytes"]) is not int or descriptor["bytes"] <= 0
            or len(descriptor["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in descriptor["sha256"])
            or any(origin["runtime_archive"][key] != descriptor[key]
                   for key in ("bytes", "sha256"))):
        raise ValueError("CUGA_RUNTIME_ARCHIVE_BINDING_DIFFERS")
    write_json(output / "origin_runtime_receipt.json", origin)
    write_json(output / "origin_environment.json", environment)
    report["stage"] = "restore_runtime"
    retained = tuple(root for root in roots if root == IMAGE_PYTHON_BASE and root.exists())
    aliases = {}
    if origin["runtime_archive"]["symlink_targets"].get(environment["python"]) == str(IMAGE_PYTHON_ALIAS):
        if roots[1] != IMAGE_PYTHON_BASE:
            raise ValueError("CUGA_IMAGE_ALIAS_REQUIRES_DECLARED_IMAGE_BASE")
        aliases[IMAGE_PYTHON_ALIAS] = IMAGE_PYTHON_BASE / "bin/python3.13"
    restore_archive(
        store, work, "cuga_restored_runtime", descriptor["uri"], descriptor,
        roots, roots, existing_identical_roots=retained, existing_link_aliases=aliases,
    )
    candidate = dict(
        environment, status="RESTORATION_VERIFICATION_PENDING",
        hostname=identity["hostname"], job_id=identity["compatibility_execution_id"],
        execution_backend="merlin", run_id=identity["run_id"], trial_id=identity["trial_id"],
        runtime_origin={"run_id": origin["run_id"], "trial_id": origin["trial_id"],
                        "hostname": environment["hostname"], "bundle": descriptor},
    )
    candidate_path = work / "candidate_environment.json"
    write_json(candidate_path, candidate)
    runtime_path = output / "runtime_identity.json"
    run_command(
        "verify_restored_runtime_identity",
        [candidate["python"], "-c", RUNTIME_IDENTITY, str(candidate_path), str(runtime_path)],
        dict(os.environ, CUDA_VISIBLE_DEVICES=""), work, output, store, report,
        [runtime_path],
    )
    actual = json.loads(runtime_path.read_text())
    admitted = dict(candidate, status=PREPARATION_STATUS)
    runtime_roots(admitted, actual)
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(environment_path, admitted)
    write_json(output / "environment.json", admitted)
    report.update(
        runtime_prepared=True, runtime_persisted=True, runtime_reused=True,
        environment=admitted, runtime_identity=actual, runtime_bundle=descriptor,
        runtime_bundle_uri=descriptor["uri"], runtime_bundle_bytes=descriptor["bytes"],
        runtime_bundle_sha256=descriptor["sha256"], runtime_archive=origin["runtime_archive"],
        runtime_origin={"run_id": origin["run_id"], "trial_id": origin["trial_id"],
                        "checkpoint_path": origin["checkpoint_path"]},
        existing_image_python_verified_without_overwrite=[str(root) for root in retained],
        existing_python_link_aliases_verified={str(alias): str(target) for alias, target in aliases.items()},
        packages_reinstalled=False,
    )

def archive_runtime(roots: tuple[Path, Path], destination: Path) -> dict[str, Any]:
    """Preserve exact prefixes and links for the existing bounded archive restorer."""
    links = {}
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            raise ValueError("CUGA runtime archive roots must be real directories")
        for directory, directories, files in os.walk(root, followlinks=False):
            for name in [*directories, *files]:
                path = Path(directory) / name
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    if not any(target.is_relative_to(allowed) for allowed in roots):
                        raise ValueError("CUGA_RUNTIME_LINK_ESCAPES_DECLARED_ROOTS: " + str(path))
                    links[str(path)] = os.readlink(path)
    with tarfile.open(destination, "w:gz", dereference=False) as archive:
        for root in roots:
            archive.add(root, arcname=root.as_posix().lstrip("/"), recursive=True)
    return {"roots": [str(root) for root in roots], "symlink_targets": links, **file_identity(destination)}


def publish_evidence(store: HdfsStore, work: Path, output: Path, report: dict[str, Any]) -> None:
    excluded = tuple(Path(name) for name in report.get("unclosed_output_paths", []))
    files = {}
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if any(relative.is_relative_to(parent) for parent in excluded):
            continue
        if path.is_symlink():
            raise ValueError("CUGA_PREPARATION_EVIDENCE_LINK_UNSUPPORTED: " + str(relative))
        if path.is_file():
            files[relative.as_posix()] = file_identity(path)
    manifest = {
        "schema": "appworld_cuga_preparation_artifacts_v1",
        "scope": "cpu_runtime_preparation_and_structural_checks",
        "run_id": report["run_id"], "trial_id": report["trial_id"],
        "runtime_archives_included": False,
        "unclosed_outputs_excluded": [str(path) for path in excluded], "files": files,
    }
    manifest_path, bundle = work / "output_manifest.json", work / "result.tar.gz"
    write_json(manifest_path, manifest)
    with tarfile.open(bundle, "w:gz") as archive:
        for name in files:
            archive.add(output / name, arcname="artifacts/" + name, recursive=False)
        archive.add(manifest_path, arcname="output_manifest.json", recursive=False)
    report["output_bundle"] = store.publish(bundle, "result.tar.gz")
    report["artifact_manifest"] = store.publish(manifest_path, "output_manifest.json")
    report["persisted"] = True
    write_json(work / "receipt.json", report)
    store.publish(work / "receipt.json", "receipt.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--prior-launcher", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--runtime-receipt", type=Path, help="Reuse a persisted CUGA package runtime without reinstalling")
    args = parser.parse_args(argv)
    identity = require_merlin_cpu(dict(os.environ))
    work = Path("/tmp") / ("neuronsim-cuga-preparation-" + identity["run_id"])
    work.mkdir(exist_ok=False)
    output = work / "artifacts"
    output.mkdir()
    store = HdfsStore(identity["checkpoint_path"], work, output)
    report: dict[str, Any] = {
        "schema": "appworld_cuga_merlin_preparation_v1", **identity,
        "scope": "cpu_runtime_preparation_and_structural_checks",
        "status": "FAILED", "exit_code": 1, "persisted": False,
        "runtime_prepared": False, "runtime_persisted": False,
        "structural_checks_passed": False, "native_integration_qualified": False,
        "formal_eligible": False, "model_generations": 0, "appworld_tasks_executed": 0,
        "started_at": time.time(), "prior_launcher_sha256": PRIOR_LAUNCHER_SHA256,
        "prior_launcher_bytes": PRIOR_LAUNCHER_BYTES,
        "model_name_for_synthetic_check": args.model_name, "stages": {},
    }
    preparation_archive = LEGACY_REPOSITORY / BENCHMARK_OUTPUT / "cuga_runtime" / (
        "run_" + identity["compatibility_execution_id"]
    )
    check_archives: list[tuple[str, Path]] = []

    def interrupted(signum, frame):
        raise InterruptedError("MERLIN_JOB_SIGNAL: " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if LEGACY_REPOSITORY.exists() or LEGACY_WORK.exists():
            raise FileExistsError("ORIGINAL_CUGA_RECIPE_PATH_ALREADY_EXISTS")
        prefix = (staging_prefix(args.prior_launcher) if args.runtime_receipt
                  else preparation_prefix(args.prior_launcher))
        prepared_script = work / "prepare.sh"
        prepared_script.write_bytes(prefix)
        write_json(output / "source_manifest.json", stage_current_sources(
            json.loads(args.inputs.read_text()), work / "current_source",
        ))
        report["preparation_prefix_sha256"] = hashlib.sha256(prefix).hexdigest()
        report["preparation_segment"] = {
            "source_start_byte": 0,
            "source_end_byte": len(prefix) - len(PREPARATION_EXIT.encode("utf-8")),
            "ends_before": PREPARATION_BOUNDARY, "appended_exit": PREPARATION_EXIT,
        }
        if args.runtime_receipt:
            report["preparation_segment"] = {
                "kind": "verified_CUGA_BOOTSTRAP_AND_DIRECTORY_SETUP",
                "package_installation_executed": False,
            }
        environment_path = preparation_archive / "environment.json"
        if args.runtime_receipt:
            run_command("stage_runtime_checks", ["bash", str(prepared_script)],
                        dict(os.environ), work, output, store, report, [])
            restore_prepared_runtime(
                json.loads(args.runtime_receipt.read_text()), identity, work=work,
                output=output, store=store, report=report, environment_path=environment_path,
            )
            environment = report["environment"]
        else:
            bootstrap_environment = prepare_bootstrap_environment(
                dict(os.environ), work=work, output=output, store=store, report=report,
            )
            run_command("prepare_packages", ["bash", str(prepared_script)], bootstrap_environment,
                        work, output, store, report, [])
            environment_path = preparation_archive / "environment.json"
            environment = json.loads(environment_path.read_text())
            if environment.get("job_id") != identity["compatibility_execution_id"]:
                raise ValueError("CUGA_PREPARATION_EXECUTION_ID_DIFFERS")
            shutil.copy2(environment_path, output / "environment.json")
            run_command(
                "verify_runtime_identity",
                [environment["python"], "-c", RUNTIME_IDENTITY,
                 str(environment_path), str(output / "runtime_identity.json")],
                dict(os.environ, CUDA_VISIBLE_DEVICES=""), work, output, store, report,
                [output / "runtime_identity.json", output / "environment.json"],
            )
            runtime_identity = json.loads((output / "runtime_identity.json").read_text())
            roots = runtime_roots(environment, runtime_identity)
            report.update(runtime_prepared=True, environment=environment, runtime_identity=runtime_identity)
            runtime_bundle = work / "runtime.tar.gz"
            archive = archive_runtime(roots, runtime_bundle)
            descriptor = store.publish(runtime_bundle, "runtime.tar.gz")
            report.update(
                runtime_persisted=True, runtime_bundle=descriptor,
                runtime_bundle_uri=descriptor["uri"], runtime_bundle_bytes=descriptor["bytes"],
                runtime_bundle_sha256=descriptor["sha256"], runtime_archive=archive,
            )
        for path in sorted((work / "current_source").rglob("*")):
            if path.is_file():
                target = LEGACY_REPOSITORY / path.relative_to(work / "current_source")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        write_json(output / "check_platform_adaptation.json", adapt_current_checks(LEGACY_REPOSITORY))
        check_environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES="",
            PATH=str(Path(environment["python"]).parent) + ":" + os.environ.get("PATH", ""),
            SLURM_CPUS_PER_TASK=os.environ.get("ARNOLD_HEAD_CPU", ""),
        )
        for kind, script_name in zip(("tools", "graph"), CHECK_SCRIPTS):
            compatibility_id = f"merlin_{identity['trial_id']}_cuga_{kind}"
            directory = LEGACY_REPOSITORY / BENCHMARK_OUTPUT / "software_validation" / ("run_" + compatibility_id)
            directory.parent.mkdir(parents=True, exist_ok=True)
            check_archives.append((kind, directory))
            command = ["bash", str(LEGACY_REPOSITORY / "Libs/scripts" / script_name),
                       str(LEGACY_REPOSITORY), str(environment_path)]
            if kind == "graph":
                command.append(args.model_name)
            run_command(
                "check_" + kind, command,
                dict(check_environment, SLURM_JOB_ID=compatibility_id),
                work, output, store, report, [],
            )
            result = json.loads((directory / "validation.json").read_text())
            if result.get("status") != "PASS":
                raise ValueError("CUGA_STRUCTURAL_CHECK_NOT_PASS: " + kind)
        report.update(status="CUGA_RUNTIME_PREPARED_STRUCTURAL_CHECKS_PASSED",
                      structural_checks_passed=True, exit_code=0)
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="CUGA_RUNTIME_PREPARED_CHECKS_INCOMPLETE" if report["runtime_prepared"] else "FAILED",
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
        )
    finally:
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - report["started_at"]
        try:
            if not report.get("unclosed_output_paths"):
                if preparation_archive.is_dir():
                    shutil.copytree(preparation_archive, output / "package_preparation")
                for kind, directory in check_archives:
                    if directory.is_dir():
                        shutil.copytree(directory, output / ("check_" + kind))
        except Exception as error:
            report.update(status="FAILED_EVIDENCE_COLLECTION", exit_code=1,
                          evidence_error_type=type(error).__name__, evidence_error=str(error))
        write_json(output / "execution_summary.json", report)
        try:
            publish_evidence(store, work, output, report)
        except Exception as error:
            report.update(persisted=False, exit_code=1, publication_error_type=type(error).__name__,
                          publication_error=str(error))
            write_json(work / "receipt.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

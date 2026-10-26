"""Verify CUGA restoration and the actual public import boundary on Merlin CPU."""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import traceback


PUBLIC_IMPORT_PROBE = """import importlib, json, pathlib, sys
public = pathlib.Path("/app")
sys.path.insert(0, str(public))
names = [
    "population_dynamics.appworld_benchmark_cuga_process",
    "population_dynamics.appworld_benchmark_cuga_runtime",
    "population_dynamics.appworld_benchmark_cuga_tools",
    "population_dynamics.appworld_benchmark_cuga_receipt",
]
for name in names:
    importlib.import_module(name)
paths = {name: str(pathlib.Path(module.__file__).resolve())
         for name, module in sys.modules.items()
         if name.startswith("population_dynamics") and getattr(module, "__file__", None)}
checks = {
    "private_repository_hidden": not pathlib.Path(sys.argv[1]).exists(),
    "procfs_absent": pathlib.Path("/proc").is_dir() and not any(pathlib.Path("/proc").iterdir()),
    "work_directory": pathlib.Path.cwd() == pathlib.Path("/work"),
    "public_modules_only": bool(paths) and all(
        pathlib.Path(path).is_relative_to(public) for path in paths.values()),
}
result = {"status": "PASS" if all(checks.values()) else "FAIL",
          "isolation_checks": checks, "public_modules": paths,
          "private_repository_path": sys.argv[1], "proc_path": "/proc",
          "working_directory": str(pathlib.Path.cwd()),
          "native_tasks_executed": 0, "model_generations": 0, "graph_loop_executed": False}
pathlib.Path("/work/public_import.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result))
if not all(checks.values()):
    raise RuntimeError("Public CUGA sandbox isolation check failed")
"""

BUILD_SANDBOX_COMMAND = """import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
sys.path.insert(0, record["application_lib"])
from population_dynamics.appworld_benchmark_cuga_process import _bubblewrap_command, _python_runtime
executable, prefix, base_prefix = _python_runtime(
    record["python"], record["python_base_prefix"])
command = _bubblewrap_command(
    executable=executable, prefix=prefix, base_prefix=base_prefix,
    application_lib=pathlib.Path(record["application_lib"]),
    sandbox=pathlib.Path(record["sandbox"]),
    owner_root=pathlib.Path(record["owner_root"]),
    protected_roots=(pathlib.Path(record["private_repository"]),),
    worker_arguments=("-c", record["probe"], record["private_repository"]),
)
pathlib.Path(record["command_path"]).write_text(json.dumps(list(command), indent=2) + "\\n")
"""


def qualify_public_import(bootstrap, binding, environment, work, output, store, report):
    """Execute the public import probe with the production sandbox command builder."""
    sandbox = output / "public_import_sandbox"
    sandbox.mkdir()
    command_path = output / "sandbox_command.json"
    specification = work / "public_import_probe.json"
    bootstrap.write_json(specification, {
        "python": environment["python"], "python_base_prefix": str(binding["python_base_prefix"]),
        "application_lib": str(binding["public_application_lib"]), "sandbox": str(sandbox),
        "owner_root": str(output), "private_repository": str(work),
        "command_path": str(command_path), "probe": PUBLIC_IMPORT_PROBE,
    })
    probe_environment = dict(
        os.environ, PYTHONPATH="/app", PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
    )
    bootstrap.run_command(
        "cuga_public_import_command",
        [environment["python"], "-c", BUILD_SANDBOX_COMMAND, str(specification)],
        probe_environment, work, output, store, report, [command_path],
    )
    command = json.loads(command_path.read_text())
    binary = report["bubblewrap"]
    if not command or command[0] != binary["executable"]:
        raise ValueError("CUGA_PUBLIC_IMPORT_SANDBOX_BINARY_DIFFERS")
    bootstrap.verify_bytes(Path(command[0]), binary)
    observation = {
        "command": command, "exit_code": None, "process_started": False, "sandbox_executed": False,
        "bubblewrap_sha256": binary["sha256"], "isolation_checks": {},
    }
    report["sandbox_probe"] = observation
    stage = "cuga_public_import_sandbox"
    probe_path = sandbox / "public_import.json"
    try:
        bootstrap.run_command(
            stage, command, probe_environment, work, output, store, report, [probe_path],
        )
        public_import = json.loads(probe_path.read_text())
        observation.update(sandbox_executed=True,
                           isolation_checks=public_import["isolation_checks"])
        report["public_import"] = public_import
        if public_import.get("status") != "PASS":
            raise ValueError("CUGA_PUBLIC_IMPORT_SANDBOX_CHECK_FAILED")
        return public_import
    finally:
        stage_record = report.get("stages", {}).get(stage, {})
        observation["exit_code"] = stage_record.get("exit_code")
        # A started process is distinct from successful confinement checks.
        observation["process_started"] = stage_record.get("pid") is not None
        observation["sandbox_executed"] = probe_path.is_file()


def main():
    inputs = json.loads(Path(sys.argv[1]).read_text())
    if os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0":
        raise ValueError("Merlin CPU allocation required")
    identity = {"run_id": os.environ["RH2_JOB_RUN_ID"],
                "trial_id": os.environ["ARNOLD_TRIAL_ID"], "hostname": socket.gethostname(),
                "execution_backend": "merlin", "allocated_gpus": 0,
                "resource_source": {"provider": "merlin", "gpu_count_environment": "ARNOLD_APPLIED_GPU_NUM"}}
    work = Path("/tmp") / ("appworld-cuga-binding-" + identity["run_id"])
    work.mkdir()
    output = work.with_name(work.name + "-artifacts")
    output.mkdir()
    for name, row in inputs["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "Libs":
            raise ValueError("Invalid declared source path")
        content = base64.b64decode(row["content_base64"], validate=True)
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Declared source differs: " + name)
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    spec = importlib.util.spec_from_file_location(
        "appworld_cuga_binding_bootstrap", work / "Libs/scripts/run_appworld_benchmark_merlin.py")
    bootstrap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bootstrap
    spec.loader.exec_module(bootstrap)
    prefix = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/appworld_cuga_binding"
    store = bootstrap.HdfsStore(prefix, work, output)
    report = {"schema": "appworld_cuga_merlin_binding_check_v1", **identity,
              "status": "RUNNING", "native_tasks_executed": 0, "model_generations": 0,
              "graph_loop_executed": False, "formal_eligible": False,
              "source_sha256": {name: row["sha256"] for name, row in inputs["files"].items()}}
    try:
        bootstrap.validate_cuga_inputs(inputs)
        binding = bootstrap.prepare_cuga_context(
            inputs, identity, dict(os.environ), work, output, store, report)
        environment = json.loads(binding["cuga_environment"].read_text())
        public_import = qualify_public_import(
            bootstrap, binding, environment, work, output, store, report)
        report.update(status="PASS", exit_code=0, public_import=public_import,
                      binding={name: str(path) for name, path in binding.items()})
    except Exception as error:
        report.update(status="FAIL", exit_code=1, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    bootstrap.write_json(output / "binding_check.json", report)
    published = bootstrap.publish_closed(store, output, [
        output / "binding_check.json", output / "bubblewrap_identity.json",
        output / "sandbox_command.json", output / "public_import_sandbox",
    ])
    report.update(persisted=True, published=published)
    bootstrap.write_json(output / "receipt.json", report)
    store.publish(output / "receipt.json", "receipt.json")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "exit_code", "run_id", "trial_id", "error")}), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

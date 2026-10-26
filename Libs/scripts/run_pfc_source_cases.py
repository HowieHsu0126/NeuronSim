"""Stage and execute registered PFC source cases in one Slurm allocation.

Single-case and array callers retain run_<job> archives. Explicit case-list
callers use run_<job>_<case>. Each rank stops on its first failed case; other
ranks finish. The coordinator serializes durable copy-out after all ranks exit.

Case-list callers may bind frozen preparation/validation receipts using all three
NEURONSIM_PFC_SOURCE_{CONFIG,CASES,CODE}_SHA256 environment values. CODE hashes
UTF-8 ``json.dumps(named_file_hashes, sort_keys=True, indent=2, allow_nan=False)
+ "\\n"``. Keys are the six ``Libs/population_dynamics/<module>.py`` paths in
_MODULES and ``Libs/scripts/run_pfc_source_cases.py``; values are lowercase SHA256.
The batch delivery records these expected and actual bindings. Source success
and archive success are separate: any incomplete successful-case archive fails
the batch while preserving all other cases and the original source status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import yaml

_MODULES = ("pfc_source_execution", "pfc_deterministic_replay", "pfc_native_drive",
            "pfc_input_pairing", "provenance", "resource_probe", "pfc_kernel_parameters")
_BINDING_ENV = {name + "_sha256": "NEURONSIM_PFC_SOURCE_" + name.upper() + "_SHA256"
                for name in ("config", "cases", "code")}
_CODE_PATHS = {"Libs/population_dynamics/" + name + ".py":
               "Libs/population_dynamics/" + name + ".py" for name in _MODULES}
_CODE_PATHS["Libs/scripts/run_pfc_source_cases.py"] = "run_pfc_source_cases.py"
_SOURCE_OUTPUTS = {"config.yaml", "request.json", "raw.npz", "terminal_state.npz", "units.npz",
                   "unit_metadata.json", "initial_voltage.npy", "final_rng.npz"}


def code_fingerprint(named_file_hashes):
    """Hash the named mapping using the existing provenance canonical JSON format."""
    payload = json.dumps(named_file_hashes, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def staged_input_bindings(work, grouped, expected):
    """Read each staged input once; selection consumes these exact payloads."""
    config_bytes = (work / "source_config.yaml").read_bytes()
    cases_bytes = (work / "cases.json").read_bytes() if grouped else None
    files = {name: hashlib.sha256((work / relative).read_bytes()).hexdigest()
             for name, relative in _CODE_PATHS.items()}
    binding = dict(status="UNBOUND", expected=expected, code_files=files,
                   code_fingerprint_format="SHA256 of UTF-8 sorted-key indent=2 JSON mapping plus newline",
                   actual=dict(config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                               cases_sha256=hashlib.sha256(cases_bytes).hexdigest() if grouped else None,
                               code_sha256=code_fingerprint(files)))
    return binding, config_bytes, cases_bytes


def verify_input_bindings(binding, grouped, requested):
    if not requested:
        return
    binding["status"] = "FAILED"
    if not grouped:
        raise ValueError("Frozen input bindings require an explicit case-list file")
    expected = binding["expected"]
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in "0123456789abcdef" for c in value) for value in expected.values()):
        raise ValueError("All three frozen input bindings require lowercase SHA256 values")
    changed = [key for key, value in expected.items() if value != binding["actual"][key]]
    if changed:
        raise ValueError("Frozen PFC source input binding mismatch: " + ",".join(changed))
    binding["status"] = "VERIFIED"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_cases(config, explicit=None, array_index=None, case_file=None, *, case_bytes=None):
    explicit, array_index, case_file = (None if value == "" else value
                                        for value in (explicit, array_index, case_file))
    modes = sum(value is not None for value in (explicit, array_index, case_file))
    if modes != 1:
        raise ValueError("Choose one explicit case, array index, or case-list file")
    registered = [row["case_id"] for row in config["cases"]]
    if len(registered) != len(set(registered)):
        raise ValueError("Duplicate registered case IDs")
    if case_file is not None:
        selected = json.loads(Path(case_file).read_bytes() if case_bytes is None else case_bytes)
        if not isinstance(selected, list) or not selected:
            raise ValueError("Case-list file requires a nonempty JSON list")
    elif array_index is not None:
        index = int(array_index)
        if not 0 <= index < len(registered):
            raise ValueError("Source array index outside registered cases")
        selected = [registered[index]]
    else:
        selected = [explicit]
    if any(not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
           for name in selected):
        raise ValueError("Invalid case ID")
    if len(selected) != len(set(selected)) or not set(selected) <= set(registered):
        raise ValueError("Cases must be unique registered IDs")
    return selected


def health(work, index, memory_gib=2):
    from population_dynamics.resource_probe import probe
    result = probe(work / ("resource_%04d.json" % index), required_memory_gib=memory_gib)
    if result["status"] != "PASS":
        raise RuntimeError("PFC_SOURCE_RESOURCE_BLOCKED:" + ",".join(result["reasons"]))


def execute_case(work, case_id, rank, plan):
    from population_dynamics.provenance import sha256_file
    case_work = work / "cases" / case_id
    case_work.mkdir(parents=True, exist_ok=False)
    record = dict(status="PREPARING", job_id=plan["job_id"], case_id=case_id, rank=rank,
                  backend=plan["backend"], runtime=str(case_work), node=os.environ.get("SLURMD_NODENAME"),
                  interpreter=sys.executable, python_version=sys.version,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source_execution_complete=False, scientific_fidelity_qualified=False, production_episodes=0)
    code = 1
    child = None
    try:
        health(case_work, 1)
        # All concurrent reads/copies below are node-local.
        shutil.copytree(work / "Libs", case_work / "Libs")
        for name in ("source_config.yaml", "launcher.sbatch", "run_pfc_source_cases.py"):
            shutil.copy2(work / name, case_work / name)
        shutil.copytree(work / "qualification", case_work / "qualification")
        record.update(config_source=plan["config_source"],
                      config_sha256=sha256_file(case_work / "source_config.yaml"),
                      qualification_source=plan["qualification_source"],
                      qualification_sha256=sha256_file(case_work / "qualification/comparison/qualification.json"))
        env = dict(os.environ, PYTHONPATH=str(case_work / "Libs"), TMPDIR=str(case_work),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        command = [sys.executable, "-B", str(case_work / "run_pfc_source_cases.py"), "--engine",
                   "--config", str(case_work / "source_config.yaml"), "--case-id", case_id,
                   "--output", str(case_work / "source_run"),
                   "--qualification", str(case_work / "qualification"), "--backend", plan["backend"]]
        record.update(status="RUNNING", command=command)
        write_json(case_work / "execution.json", record)
        with (case_work / "source_run.log").open("x") as log:
            child = subprocess.Popen(command, cwd=case_work, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["engine_pid"] = child.pid
            write_json(case_work / "execution.json", record)
            health_index = 1
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    health_index += 1
                    health(case_work, health_index)
        code = child.returncode
        report = json.loads((case_work / "source_run/report.json").read_text())
        record.update(status=report["status"], source_execution_complete=report["source_execution_complete"],
                      execution_observation=json.loads((case_work / "source_run/engine_observation.json").read_text()))
        if code == 0 and report["source_execution_complete"] is not True:
            code = 1
    except BaseException as error:
        code = code or 1
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        record.update(status="FAILED", error=str(error), traceback=traceback.format_exc())
    record.update(exit_code=code, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(case_work / "execution.json", record)
    return record


def run_rank(work):
    plan = json.loads((work / "plan.json").read_text())
    rank = int(os.environ["SLURM_PROCID"])
    count = int(os.environ.get("SLURM_STEP_NUM_TASKS", os.environ.get("SLURM_NTASKS", "1")))
    if count != plan["tasks"] or not 0 <= rank < count:
        raise ValueError("Slurm rank geometry differs from staged plan")
    path = work / "ranks" / ("%d.json" % rank)
    rows = [dict(case_id=name, rank=rank, status="NOT_RUN", exit_code=None)
            for name in plan["case_ids"][rank::count]]
    ledger = dict(job_id=plan["job_id"], rank=rank, pid=os.getpid(), cases=rows)
    write_json(path, ledger)
    code = 0
    for index, row in enumerate(rows):
        row["status"] = "RUNNING"
        write_json(path, ledger)
        rows[index] = execute_case(work, row["case_id"], rank, plan)
        write_json(path, ledger)
        if rows[index]["exit_code"]:
            code = rows[index]["exit_code"]
            break
    ledger.update(exit_code=code, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(path, ledger)
    return code


def archive_tree(source, target):
    from population_dynamics.provenance import sha256_file
    target.mkdir(parents=True, exist_ok=False)
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(path)
        shutil.copy2(path, destination)
        if sha256_file(destination) != digest:
            raise ValueError("Source execution archive mismatch: " + str(relative))
        files.append(dict(path=str(relative), sha256=digest, bytes=destination.stat().st_size))
    return files


def validate_successful_archive(archive, files, row, bindings):
    """Check archive completeness against already copy-verified manifest entries.

    Numerical checks remain the source producer's responsibility. This checks
    the metadata and output bindings required to consume its successful result,
    without rereading or rehashing large arrays after archive_tree verified them.
    """
    manifest = {item["path"]: item for item in files}
    required = {"execution.json", "source_config.yaml", "source_run/report.json",
                "source_run/engine_observation.json", "qualification/delivery.json",
                "qualification/comparison/qualification.json", *_CODE_PATHS.values()}
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError("Successful case archive missing required metadata: " + ",".join(missing))
    execution = json.loads((archive / "execution.json").read_text())
    report = json.loads((archive / "source_run/report.json").read_text())
    observation = json.loads((archive / "source_run/engine_observation.json").read_text())
    if (row.get("source_execution_complete") is not True or row.get("exit_code") != 0
            or any(item.get("case_id") != row["case_id"]
                   or item.get("status") != "PFC_SOURCE_EXECUTION_COMPLETE"
                   or item.get("source_execution_complete") is not True for item in (execution, report))
            or execution.get("exit_code") != 0 or not isinstance(observation, dict) or not observation):
        raise ValueError("Successful case archive execution/report/observation metadata disagree")
    validations = report.get("validations")
    if (not isinstance(validations, dict) or not validations
            or any(not isinstance(check, dict) or check.get("passed") is not True
                   for check in validations.values())):
        raise ValueError("Successful case archive requires every declared source validation to pass")
    config_sha = bindings["actual"]["config_sha256"]
    if (manifest["source_config.yaml"]["sha256"] != config_sha
            or execution.get("config_sha256") != config_sha or report.get("config_sha256") != config_sha):
        raise ValueError("Successful case archive config differs from staged input")
    if any(manifest[_CODE_PATHS[name]]["sha256"] != digest
           for name, digest in bindings["code_files"].items()):
        raise ValueError("Successful case archive code differs from staged input")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not _SOURCE_OUTPUTS <= outputs.keys():
        raise ValueError("Successful case archive source output inventory is incomplete")
    for name, expected in outputs.items():
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Invalid source output identity in successful archive")
        actual = manifest.get("source_run/" + name)
        if (not isinstance(expected, dict) or actual is None or expected.get("path") != name
                or any(expected.get(key) != actual[key] for key in ("sha256", "bytes"))):
            raise ValueError("Successful case archive output binding missing or changed: " + name)
    if outputs["config.yaml"]["sha256"] != config_sha:
        raise ValueError("Successful case archive source config output differs from staged input")


def run_engine(arguments):
    from population_dynamics import pfc_source_execution as source
    output = Path(arguments[arguments.index("--output") + 1])
    observation = dict(engine_pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                       numerical_threads={key: os.environ.get(key) for key in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    original = source.executor.simulate

    def observed_simulate(*args, **kwargs):
        observation["simulator_start_monotonic_ns"] = time.monotonic_ns()
        try:
            return original(*args, **kwargs)
        finally:
            observation["simulator_stop_monotonic_ns"] = time.monotonic_ns()

    source.executor.simulate = observed_simulate
    sys.argv = [sys.argv[0], *arguments]
    try:
        return source.main()
    finally:
        observation["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write_json(output / "engine_observation.json", observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-worker", type=Path)
    parser.add_argument("--engine", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.engine:
        return run_engine(remaining)
    if remaining:
        parser.error("Unexpected arguments: " + " ".join(remaining))
    if args.rank_worker is not None:
        sys.path.insert(0, str(args.rank_worker / "Libs"))
        return run_rank(args.rank_worker)
    repo = Path(os.environ.get("NEURONSIM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
    backend = os.environ["NEURONSIM_PFC_SOURCE_BACKEND"]
    job = os.environ["SLURM_JOB_ID"]
    grouped = bool(os.environ.get("NEURONSIM_PFC_SOURCE_CASES_JSON"))
    tasks = int(os.environ.get("SLURM_NTASKS", "1")) if grouped else 1
    if backend == "numpy":
        if (os.environ.get("SLURMD_NODENAME") not in {"cpu03", "cpu04", "cpu05", "cpu06"}
                or os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
                or not 1 <= tasks <= 8 or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) != 1):
            raise ValueError("Authorized CPU-only allocation with at most eight one-CPU tasks required")
    elif backend != "cuda" or grouped:
        raise ValueError("Packed source execution currently requires the NumPy backend")
    config_path = Path(os.environ["NEURONSIM_PFC_SOURCE_CONFIG"]).resolve()
    qualification = Path(os.environ["NEURONSIM_PFC_SOURCE_QUALIFICATION"]).resolve()
    cases_path = Path(os.environ["NEURONSIM_PFC_SOURCE_CASES_JSON"]).resolve() if grouped else None
    expected = {key: os.environ.get(name) for key, name in _BINDING_ENV.items()}
    requested = any(name in os.environ for name in _BINDING_ENV.values())
    base = Path(os.environ.get("NEURONSIM_PFC_OUTPUT_BASE",
                str(repo / "Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/streamed_source_execution")))
    scratch = Path(os.environ.get("NEURONSIM_PFC_SCRATCH", "/tmp" if backend == "numpy" else "/dev/shm"))
    work = scratch / ("neuronsim-pfc-source-cases-" + job)
    work.mkdir(parents=True, exist_ok=False)
    plan = dict(job_id=job, backend=backend, tasks=tasks, case_ids=[], grouped=grouped,
                config_source=str(config_path), qualification_source=str(qualification), output_base=str(base),
                cases_source=str(cases_path) if grouped else None, runtime=str(work),
                input_bindings=dict(status="REQUESTED" if requested else "UNBOUND", expected=expected, actual={}),
                started_utc=datetime.now(timezone.utc).isoformat(), status="PREPARING",
                scientific_fidelity_qualified=False, production_episodes=0)
    rows = []
    plan["cases"] = rows
    write_json(work / "plan.json", plan)
    code = 1
    try:
        package = work / "Libs/population_dynamics"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"")
        for name in _MODULES:
            shutil.copy2(repo / "Libs/population_dynamics" / (name + ".py"), package / (name + ".py"))
        source_launcher = repo / "Libs/scripts/run_pfc_source_cases.sbatch"
        if not source_launcher.exists():
            # Supported staged source trees can contain only the historical launcher.
            source_launcher = repo / "Output/neural_population_dynamics/operations/pfc_source_execution.sbatch"
        shutil.copy2(source_launcher, work / "launcher.sbatch")
        shutil.copy2(Path(__file__), work / "run_pfc_source_cases.py")
        shutil.copy2(config_path, work / "source_config.yaml")
        if grouped:
            shutil.copy2(cases_path, work / "cases.json")
        binding, config_bytes, cases_bytes = staged_input_bindings(work, grouped, expected)
        plan["input_bindings"] = binding
        verify_input_bindings(binding, grouped, requested)
        config = yaml.safe_load(config_bytes)
        selected = select_cases(config, os.environ.get("NEURONSIM_PFC_SOURCE_CASE_ID"),
                                os.environ.get("SLURM_ARRAY_TASK_ID"),
                                work / "cases.json" if grouped else None, case_bytes=cases_bytes)
        rows.extend(dict(case_id=name, rank=i % tasks, status="NOT_RUN", exit_code=None)
                    for i, name in enumerate(selected))
        plan.update(case_ids=selected, registered_case_ids=[row["case_id"] for row in config["cases"]])
        sys.path.insert(0, str(work / "Libs"))
        health(work, 1, tasks * 2)
        if shutil.disk_usage(work).free < tasks * 2 * 1024**3:
            raise RuntimeError("Source scratch requires 2 GiB per simultaneous case")
        (work / "qualification/comparison").mkdir(parents=True)
        for name in ("delivery.json", "comparison/qualification.json"):
            shutil.copy2(qualification / name, work / "qualification" / name)
        command = ["srun", "--ntasks=" + str(tasks), "--cpus-per-task=1", "--kill-on-bad-exit=0",
                   sys.executable, "-B", str(work / "run_pfc_source_cases.py"), "--rank-worker", str(work)]
        plan.update(status="RUNNING", command=command)
        write_json(work / "plan.json", plan)
        env = dict(os.environ, PYTHONPATH=str(work / "Libs"), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        step = subprocess.run(command, cwd=work, env=env)
        code = step.returncode
        plan["srun_exit_code"] = code
        for rank in range(tasks):
            path = work / "ranks" / ("%d.json" % rank)
            expected = [row for row in rows if row["rank"] == rank]
            if not path.is_file():
                for row in expected:
                    row["status"] = "NOT_REPORTED"
                code = code or 1
                continue
            ledger = json.loads(path.read_text())
            if [row["case_id"] for row in ledger["cases"]] != [row["case_id"] for row in expected]:
                raise ValueError("Rank case accounting differs from plan")
            for row in ledger["cases"]:
                rows[selected.index(row["case_id"])] = row
            code = code or ledger.get("exit_code") or 0
        if any(row["status"] != "PFC_SOURCE_EXECUTION_COMPLETE" or row["exit_code"] != 0 for row in rows):
            code = code or 1
        plan["rank_peak_rss_bytes"] = {
            str(rank): json.loads((work / "ranks" / ("%d.json" % rank)).read_text()).get("max_rss_bytes")
            for rank in range(tasks) if (work / "ranks" / ("%d.json" % rank)).is_file()}
    except BaseException as error:
        plan.update(error=str(error), traceback=traceback.format_exc())
        code = code or 1
    # Retain every attempted case and every unexecuted assignment, even after a failure.
    for row in rows:
        source = work / "cases" / row["case_id"]
        success = row["status"] == "PFC_SOURCE_EXECUTION_COMPLETE" or row.get("source_execution_complete") is True
        row["archive_status"] = "NOT_AVAILABLE"
        try:
            if not source.is_dir():
                if success:
                    raise FileNotFoundError("Successful case directory missing: " + str(source))
                continue
            label = job + ("_" + row["case_id"] if grouped else "")
            archive = base / ("run_" + label)
            row["archive"] = str(archive)
            files = archive_tree(source, archive)
            if success:
                validate_successful_archive(archive, files, row, plan["input_bindings"])
            write_json(archive / "delivery.json", dict(status="ARCHIVED_PFC_SOURCE_EXECUTION",
                       result_status=row["status"], source_execution_complete=row.get("source_execution_complete", False),
                       job_id=job, case_id=row["case_id"], rank=row["rank"], exit_code=row.get("exit_code"),
                       files=files, scientific_fidelity_qualified=False, production_episodes=0))
            row["archive_status"] = "COMPLETE"
        except Exception as error:
            row.update(archive_status="FAILED", archive_error=str(error), archive_traceback=traceback.format_exc())
            code = code or 1
    plan.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code, cases=rows,
                coordinator_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(work / "plan.json", plan)
    batch = base / "batches" / ("run_" + job)
    batch.mkdir(parents=True, exist_ok=False)
    for name in ("plan.json", "source_config.yaml", "cases.json", "launcher.sbatch", "run_pfc_source_cases.py"):
        if (work / name).is_file():
            shutil.copy2(work / name, batch / name)
    if (work / "ranks").is_dir():
        archive_tree(work / "ranks", batch / "ranks")
    write_json(batch / "delivery.json", plan)
    # Case manifests have already compared every copied output before this removal.
    if code == 0:
        shutil.rmtree(work)
    print(json.dumps(dict(status=plan["status"], archive=str(batch), exit_code=code, cases=len(rows))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

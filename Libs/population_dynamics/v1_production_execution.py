"""Submit one qualified V1 block through the existing source launchers.

The circuit must already be constructed with its frozen production declaration.
Every condition uses a matching passed rate-code reference and spike canary.
Only deterministic rates may be reused across noise blocks. Submission receipts
do not count as completed episodes; the existing block exporter owns that count.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import warnings

import yaml

from .benchmark_production import ProductionCohort, require, glif_backend_qualification
from .benchmark_production_block import BLOCK_SCHEMA, GLIF_INPUT_SCHEMA
from .v1_glif_production import verify_circuit_resources
from .circuit_campaign import campaign_config
from .provenance import freeze_json, sha256_file
from .v1_circuit_stimulus import rate_config_without_noise
from .v1_execution_recipe import require_same_recipe
from .v1_nest_input_replay import bound_json
from .v1_source_node_qualification import load_source_node_qualification, qualified_source_case, source_node_export_launcher

SCHEMA = "neural_v1_production_execution_v1"
PHASES = ("rates", "spikes", "native", "export")
SOURCE_PATHS = ("circuit_config", "recurrent_source", "input_source", "prepared_runtime",
                "native_canary", "parameters", "point_config", "authority_source")
FAMILY = Path("Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source")


def _path(base: Path, value) -> Path:
    require(isinstance(value, str) and bool(value.strip()), "V1_EXECUTION_PATH_REQUIRED")
    return (base / value).resolve()


def _report(archive: Path, relative: str, bindings: dict) -> dict:
    report, delivery = bound_json(archive, relative)
    require(delivery.get("exit_code") == 0, "V1_EXECUTION_SOURCE_NOT_COMPLETE:" + str(archive))
    bindings[str(archive / "delivery.json")] = sha256_file(archive / "delivery.json")
    bindings[str(archive / relative)] = sha256_file(archive / relative)
    return report



GPU_PARTITIONS = {"cpu03": "gpu", "cpu04": "cpu", "cpu05": "cpu", "cpu06": "cpu", "gpu03": "gpu1"}


def gpu_resources(cohort, identity, document: dict, base: Path, bindings: dict) -> dict:
    require(isinstance(document, dict) and set(document) == {
        "node", "partition", "library", "detector", "software", "circuit_reference", "snapshot", "circuit_validation"},
        "V1_EXECUTION_GPU_RESOURCES_INVALID")
    require(document["node"] in GPU_PARTITIONS and document["partition"] == GPU_PARTITIONS[document["node"]],
            "V1_EXECUTION_GPU_PLACEMENT_INVALID")
    value = {key: str(_path(base, path)) if key not in ("node", "partition") else path
             for key,path in document.items()}
    require(all(Path(path).exists() for key,path in value.items() if key not in ("node", "partition")),
            "V1_EXECUTION_GPU_RESOURCE_MISSING")
    qualification = glif_backend_qualification(cohort)
    require(sha256_file(Path(value["library"])) == qualification["library_sha256"],
            "V1_EXECUTION_GPU_LIBRARY_NOT_QUALIFIED")
    expected_software = _path(cohort.qualification_path.parent, qualification["software_directory"])
    require(Path(value["software"]) == expected_software, "V1_EXECUTION_GPU_SOFTWARE_SNAPSHOT_DIFFERS")
    for name,digest in qualification["runtime_code"].items():
        path = expected_software/name
        require(sha256_file(path) == digest, "V1_EXECUTION_GPU_RUNTIME_NOT_QUALIFIED:"+name)
        bindings[str(path)] = digest
    bindings.update(verify_circuit_resources(cohort, identity, Path(value["circuit_reference"]),
                                            Path(value["snapshot"]), Path(value["circuit_validation"])))
    for key in ("library", "detector"):
        bindings[value[key]] = sha256_file(Path(value[key]))
    return value


def prepare(cohort: ProductionCohort, repo: Path, config: Path, resources: Path,
            output: Path, *, resume: bool = False) -> dict:
    """Validate the complete block and write all recipes before any submission."""
    cohort.require_current()
    require(cohort.scenario == "v1_visual", "V1_EXECUTION_SCENARIO_REQUIRED")
    require(campaign_config(config) == cohort.campaign, "V1_EXECUTION_CAMPAIGN_DIFFERS")
    document = yaml.safe_load(resources.read_text())
    required = {"schema", "circuit_index", "block_index", "conditions", "memory_gib",
                "native_python", "export_python", *SOURCE_PATHS}
    require(isinstance(document, dict), "V1_EXECUTION_RESOURCES_SCHEMA_INVALID")
    backend = document.get("backend", "native")
    extra = {"backend", "gpu"} if backend == "glif_cuda" else ({"backend"} if "backend" in document else set())
    if "walltime_minutes" in document:
        extra.add("walltime_minutes")
    if "source_node_qualification" in document:
        extra.add("source_node_qualification")
    require(backend in ("native", "glif_cuda") and set(document) == required | extra and document["schema"] == SCHEMA,
            "V1_EXECUTION_RESOURCES_SCHEMA_INVALID")
    circuit_index, block_index = document["circuit_index"], document["block_index"]
    identity = cohort.identity(circuit_index, block_index, 0)
    paths = {key: _path(resources.parent, document[key]) for key in SOURCE_PATHS
             if key != "prepared_runtime" or document[key] is not None}
    require(all(path.exists() for path in paths.values()), "V1_EXECUTION_RESOURCE_MISSING")
    interpreters = {key: document[key] for key in ("native_python", "export_python")}
    # Interpreters can reside on compute-node-local storage.
    require(all(isinstance(value, str) and Path(value).is_absolute() for value in interpreters.values()),
            "V1_EXECUTION_ABSOLUTE_INTERPRETERS_REQUIRED")
    memory = document["memory_gib"]
    memory_phases = {"rates", "spikes", "gpu", "export"} if backend == "glif_cuda" else set(PHASES)
    require(isinstance(memory, dict) and set(memory) == memory_phases
            and all(type(value) is int and value > 0 for value in memory.values()),
            "V1_EXECUTION_MEMORY_RESERVATIONS_REQUIRED")
    walltime = document.get("walltime_minutes")
    if "walltime_minutes" in document:
        require(isinstance(walltime, dict) and set(walltime) == memory_phases
                and all(type(value) is int and value >= 0 for value in walltime.values()),
                "V1_EXECUTION_WALLTIME_MINUTES_INVALID")
    actual_config = yaml.safe_load(paths["circuit_config"].read_text())
    cohort.verify_declaration(actual_config, identity)
    require(actual_config == cohort.v1_construction_config(identity),
            "V1_EXECUTION_DECLARED_CONSTRUCTION_RECIPE_DIFFERS")
    bindings = {str(paths["circuit_config"]): sha256_file(paths["circuit_config"])}
    source_node = None
    if "source_node_qualification" in document:
        source_node_path = _path(resources.parent, document["source_node_qualification"])
        source_node = load_source_node_qualification(source_node_path, cohort.qualification_sha256, bindings)
    for role in ("recurrent_source", "input_source"):
        report = _report(paths[role], "primary/report.json", bindings)
        require(report.get("mode") == "full" and report.get("original_config") == actual_config
                and report.get("original_config_sha256") == bindings[str(paths["circuit_config"])],
                "V1_EXECUTION_CONSTRUCTION_BINDING_DIFFERS:" + role)
        rng = report["rng"]
        if role == "recurrent_source":
            require(rng.get("builder_seed") == identity["connectivity_seed"]
                    and rng.get("glif_assignment_seed") == identity["connectivity_seed"] + 1000000,
                    "V1_EXECUTION_RECURRENT_SEEDS_DIFFER")
        else:
            require(rng.get("lgn_selection_python_seed") == identity["connectivity_seed"],
                    "V1_EXECUTION_INPUT_SEED_DIFFERS")
    gpu = None
    if backend == "glif_cuda":
        gpu = gpu_resources(cohort, identity, document["gpu"], resources.parent, bindings)
        reference = _report(Path(gpu["circuit_reference"]), "primary/report.json", bindings)
        for role, source in (("recurrent", "recurrent_source"), ("inputs", "input_source")):
            require(reference["bindings"][role]["files"]["primary/report.json"]["sha256"]
                    == bindings[str(paths[source]/"primary/report.json")],
                    "V1_EXECUTION_GPU_CONSTRUCTION_REFERENCE_DIFFERS")
    if "prepared_runtime" in paths:
        prepared = _report(paths["prepared_runtime"], "prepared/prepared.json", bindings)
        require(prepared.get("status") == "SOURCE_RUNTIME_PREPARED", "V1_EXECUTION_PREPARED_RUNTIME_REQUIRED")
    _report(paths["native_canary"], "primary/report.json", bindings)
    native_delivery = json.loads((paths["native_canary"] / "delivery.json").read_text())
    require(native_delivery.get("status") in ("PASS_POINTNET_NATIVE_DIAGNOSTIC", "PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT"),
            "V1_EXECUTION_REFERENCE_CLOCK_CANARY_REQUIRED")
    canary_config = paths["native_canary"] / "original_config.yaml"
    rows = [row for row in native_delivery["files"] if row["path"] == "original_config.yaml"]
    require(len(rows) == 1 and canary_config.stat().st_size == rows[0]["bytes"]
            and sha256_file(canary_config) == rows[0]["sha256"], "V1_EXECUTION_NATIVE_CANARY_CONFIG_CHANGED")
    bindings[str(canary_config)] = rows[0]["sha256"]
    require_same_recipe(actual_config, yaml.safe_load(canary_config.read_text()), family="circuit")
    records = document["conditions"]
    require(isinstance(records, list) and len(records) == 10, "V1_EXECUTION_TEN_CONDITIONS_REQUIRED")
    conditions, recipes = [], []
    for index, record in enumerate(records):
        require(isinstance(record, dict) and set(record) == {"condition_index", "rate_code_reference", "spike_canary", "rate_source"}
                and type(record["condition_index"]) is int and record["condition_index"] == index,
                "V1_EXECUTION_CONDITION_ORDER_OR_FIELDS_DIFFER")
        identity = cohort.identity(circuit_index, block_index, index)
        recipe = cohort.v1_visual_config(identity)
        if source_node is not None:
            reference = qualified_source_case(source_node, recipe)
            record = dict(record, spike_canary=reference["spike_canary"])
            if record["rate_source"] is None:
                record["rate_code_reference"] = reference["rate_code_reference"]
        spike_canary = _path(resources.parent, record["spike_canary"])
        rate_code = None
        if record["rate_source"] is None:
            require("prepared_runtime" in paths, "V1_EXECUTION_PREPARED_RUNTIME_REQUIRED")
            rate_code = _path(resources.parent, record["rate_code_reference"])
            rate_reference = _report(rate_code, "probe/probe.json", bindings)
            require(rate_reference.get("status") == "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT",
                    "V1_EXECUTION_RATE_CODE_REFERENCE_REQUIRED")
            require_same_recipe(recipe, rate_reference["config"], family="visual")
        spike_reference = _report(spike_canary, "visual_inputs/report.json", bindings)
        require(json.loads((spike_canary / "delivery.json").read_text()).get("status") == "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY",
                "V1_EXECUTION_SPIKE_CANARY_REQUIRED")
        require_same_recipe(recipe, spike_reference["config"], family="visual")
        rate_source = None
        if record["rate_source"] is not None:
            rate_source = _path(resources.parent, record["rate_source"])
            rates = _report(rate_source, "probe/probe.json", bindings)
            require(rates.get("status") == "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT"
                    and rates.get("source_units") == 17400
                    and rate_config_without_noise(rates["config"]) == rate_config_without_noise(recipe)
                    and rates.get("input_construction_binding", {}).get("delivery_sha256")
                        == bindings[str(paths["input_source"] / "delivery.json")],
                    "V1_EXECUTION_REUSED_RATE_CIRCUIT_OR_CONDITION_DIFFERS")
        recipes.append(recipe)
        conditions.append(dict(identity=identity, rate_code_reference=str(rate_code) if rate_code else None, spike_canary=str(spike_canary),
                               rate_source=str(rate_source) if rate_source else None))
    launchers = {phase: repo / "Libs/scripts" / name for phase, name in (
        ("rates", "run_v1_source_rates.sbatch"), ("spikes", "run_v1_source_spikes.sbatch"),
        ("native", "run_v1_native_episode.sbatch"), ("export", "run_v1_production_block_export.sbatch"))}
    if backend == "glif_cuda":
        launchers.pop("native")
        launchers.update(gpu_primary=repo/"Libs/scripts/run_v1_glif_network.sbatch",
                         gpu_replay=repo/"Libs/scripts/run_v1_glif_network.sbatch",
                         source_plan=repo/"Libs/scripts/run_v1_production_block_export.sbatch")
    if source_node is not None:
        launchers.update({role: Path(source_node["launchers"][role]["path"]) for role in ("rates", "spikes")})
        export_launcher = source_node_export_launcher(repo, source_node_path.parent, source_node["node"])
        launchers["export"] = export_launcher
        if backend == "glif_cuda":
            launchers["source_plan"] = export_launcher
    require(all(path.is_file() for path in launchers.values()), "V1_EXECUTION_LAUNCHER_MISSING")
    bindings.update({str(path): sha256_file(path) for path in launchers.values()})
    plan = dict(schema=SCHEMA, scenario_id="v1_visual", circuit_index=circuit_index, block_index=block_index,
                config=str(output.resolve() / "campaign.yaml"), qualification=str(cohort.qualification_path.resolve()),
                identity_table=str(cohort.identity_table.resolve()), repo=str(repo.resolve()),
                paths={key: str(value) for key, value in paths.items()}, interpreters=interpreters,
                memory_gib=memory, conditions=conditions, source_bindings=bindings,
                launchers={key: str(value) for key, value in launchers.items()},
                source_qualification_sha256=cohort.qualification_sha256,
                identity_table_sha256=cohort.identity_table_sha256, production_episodes=0)
    if source_node is not None:
        plan["source_cpu"] = dict(node=source_node["node"], partition=source_node["partition"],
                                  qualification=str(source_node_path))
    if gpu is not None:
        plan.update(backend="glif_cuda", gpu=gpu)
    if walltime is not None:
        plan["walltime_minutes"] = walltime
    if output.exists():
        require(resume, "V1_EXECUTION_EXPLICIT_RESUME_REQUIRED")
        require((output / "plan.json").exists(), "V1_EXECUTION_PLAN_MISSING")
    else:
        output.mkdir(parents=True)
    freeze_json(output / "plan.json", plan)
    snapshot = output / "campaign.yaml"
    if snapshot.exists():
        require(campaign_config(snapshot) == cohort.campaign, "V1_EXECUTION_CAMPAIGN_SNAPSHOT_CHANGED")
    else:
        with snapshot.open("x") as stream:
            stream.write(config.read_text())
    recipes_root = output / "recipes"
    recipes_root.mkdir(exist_ok=True)
    for index, recipe in enumerate(recipes):
        path = recipes_root / (f"condition_{index:02d}.yaml")
        text = yaml.safe_dump(recipe, sort_keys=False)
        if path.exists():
            require(path.read_text() == text, "V1_EXECUTION_RECIPE_CHANGED")
        else:
            with path.open("x") as stream:
                stream.write(text)
    return plan


SOURCE_PHASES = ("rates", "spikes", "source_plan")
ACTIVE_JOB_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}
TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                       "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def job_states(jobs: list[int]) -> dict[int, tuple[str, str]]:
    """Read explicit accounting states before reusing or replacing source work."""
    require(bool(jobs) and all(type(job) is int and job > 0 for job in jobs),
            "V1_EXECUTION_JOB_IDS_INVALID")
    result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(map(str, sorted(set(jobs)))),
                             "--format=JobIDRaw,State,ExitCode"], text=True, capture_output=True)
    if result.returncode:
        warnings.warn("V1_EXECUTION_ACCOUNTING_UNAVAILABLE; querying the controller for each exact job: "
                      + result.stderr.strip(), RuntimeWarning, stacklevel=2)
    rows = [line.split("|") for line in result.stdout.splitlines()] if result.returncode == 0 else []
    states = {}
    for job in set(jobs):
        matches = [row for row in rows if row[0] == str(job)]
        require(len(matches) <= 1, "V1_EXECUTION_JOB_STATE_UNKNOWN:" + str(job))
        if not matches:
            # Accounting can lag or be unavailable on a compute node.
            # The controller must resolve the exact job; unknown work is not resubmitted.
            live = subprocess.run(["scontrol", "show", "job", "-o", str(job)],
                                  text=True, capture_output=True)
            values = dict(item.split("=", 1) for item in live.stdout.split() if "=" in item)
            require(live.returncode == 0 and values.get("JobId") == str(job)
                    and "JobState" in values and "ExitCode" in values,
                    "V1_EXECUTION_JOB_STATE_UNKNOWN:" + str(job)
                    + "; accounting=" + result.stderr.strip() + "; controller=" + live.stderr.strip())
            state, code = values["JobState"], values["ExitCode"]
        else:
            require(len(matches[0]) >= 3, "V1_EXECUTION_JOB_STATE_UNKNOWN:" + str(job))
            state, code = matches[0][1:3]
        # Slurm accounting can append the cancelling user to CANCELLED.
        state = state.split()[0]
        require(state in ACTIVE_JOB_STATES | TERMINAL_JOB_STATES,
                "V1_EXECUTION_JOB_STATE_UNKNOWN:" + str(job) + ":" + state)
        states[job] = (state, code)
    return states


def source_barrier_jobs(submission: dict) -> tuple[int, ...]:
    """Wait for all outstanding CPU source work, regardless of its result.

    A cancelled source-plan job does not prove earlier spike work has ended.
    Carry the inherited barrier too, so early cancellation cannot release a lane.
    GPU and export jobs do not occupy a source lane.
    """
    jobs = [row["job_id"] for row in submission["jobs"] if row["phase"] in SOURCE_PHASES]
    require(bool(jobs), "V1_EXECUTION_SOURCE_JOBS_REQUIRED")
    jobs.extend(submission.get("start_after_any_job_ids", []))
    if submission.get("start_after_job_id") is not None:
        jobs.append(submission["start_after_job_id"])
    states = job_states(jobs)
    return tuple(sorted(job for job, (state, _) in states.items() if state in ACTIVE_JOB_STATES))


def recover_source_block(cohort: ProductionCohort, original: Path, output: Path, *,
                         start_after_any_jobs: tuple[int, ...] = (),
                         gpu_node: str | None = None, gpu_memory_gib: int | None = None,
                         source_node_qualification: Path | None = None,
                         walltime_minutes: dict[str, int] | None = None,
                         resume: bool = False) -> dict:
    """Recover terminated source work in a new root and reuse completed spikes."""
    require(original.resolve() != output.resolve(), "V1_EXECUTION_RECOVERY_NEW_ROOT_REQUIRED")
    prior = json.loads((original / "submission.json").read_text())
    plan = deepcopy(json.loads((original / "plan.json").read_text()))
    require(plan.get("backend") == "glif_cuda" and len(prior["jobs"]) == 14
            and [row["phase"] for row in prior["jobs"]] == ["spikes"] * 10
                + ["source_plan", "gpu_primary", "gpu_replay", "export"]
            and all(row["rate_source"] is not None for row in plan["conditions"]),
            "V1_EXECUTION_SOURCE_RECOVERY_REQUIRES_SPIKE_CHAIN")
    states = job_states([row["job_id"] for row in prior["jobs"]])
    require(all(state in TERMINAL_JOB_STATES for state, _ in states.values()),
            "V1_EXECUTION_RECOVERY_ORIGINAL_WORK_NOT_TERMINAL")
    require(all(states[row["job_id"]][0] == "CANCELLED" for row in prior["jobs"][-3:]),
            "V1_EXECUTION_RECOVERY_REQUIRES_UNEXECUTED_GPU_AND_EXPORT")
    retained = {row["condition_index"]: row["job_id"] for row in prior["jobs"][:10]
                if states[row["job_id"]] == ("COMPLETED", "0:0")}
    require(all(row["condition_index"] == index for index, row in enumerate(prior["jobs"][:10])),
            "V1_EXECUTION_RECOVERY_CONDITION_ORDER_DIFFERS")
    if gpu_node is not None:
        require(gpu_node in GPU_PARTITIONS, "V1_EXECUTION_GPU_PLACEMENT_INVALID")
        plan["gpu"].update(node=gpu_node, partition=GPU_PARTITIONS[gpu_node])
    if gpu_memory_gib is not None:
        require(type(gpu_memory_gib) is int and gpu_memory_gib > 0, "V1_EXECUTION_MEMORY_RESERVATIONS_REQUIRED")
        plan["memory_gib"]["gpu"] = gpu_memory_gib
    if walltime_minutes is not None:
        require(set(walltime_minutes) == {"rates", "spikes", "gpu", "export"}
                and all(type(value) is int and value >= 0 for value in walltime_minutes.values()),
                "V1_EXECUTION_WALLTIME_MINUTES_INVALID")
        plan["walltime_minutes"] = dict(walltime_minutes)
    if source_node_qualification is not None:
        selected = load_source_node_qualification(source_node_qualification, cohort.qualification_sha256,
                                                  plan["source_bindings"])
        for index, condition in enumerate(plan["conditions"]):
            identity = cohort.identity(plan["circuit_index"], plan["block_index"], index)
            require(condition["identity"] == identity, "V1_EXECUTION_RECOVERY_CONDITION_IDENTITY_DIFFERS")
            reference = qualified_source_case(selected, cohort.v1_visual_config(identity))
            condition["spike_canary"] = reference["spike_canary"]
        plan["source_cpu"] = dict(node=selected["node"], partition=selected["partition"],
                                  qualification=str(source_node_qualification))
        for role in ("rates", "spikes"):
            plan["launchers"][role] = selected["launchers"][role]["path"]
        export_launcher = source_node_export_launcher(Path(plan["repo"]), source_node_qualification.parent,
                                                       selected["node"])
        for role in ("source_plan", "export"):
            plan["launchers"][role] = str(export_launcher)
        plan["source_bindings"].update({str(path): sha256_file(Path(path))
                                        for path in plan["launchers"].values()})
    require(not output.exists() or resume, "V1_EXECUTION_EXPLICIT_RESUME_REQUIRED")
    output.mkdir(parents=True, exist_ok=True)
    plan["config"] = str(output.resolve() / "campaign.yaml")
    freeze_json(output / "plan.json", plan)
    for name in ["campaign.yaml", *[f"recipes/condition_{index:02d}.yaml" for index in range(10)]]:
        target = output / name
        payload = (original / name).read_bytes()
        if target.exists():
            require(target.read_bytes() == payload, "V1_EXECUTION_RECOVERY_RECIPE_CHANGED")
        else:
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(payload)
    freeze_json(output / "recovery.json", dict(original_submission=str(original / "submission.json"),
        original_submission_sha256=sha256_file(original / "submission.json"), original_states=states,
        retained_spike_jobs=retained, start_after_any_job_ids=list(start_after_any_jobs)))
    return submit(cohort, plan, output, resume=resume, start_after_any_jobs=start_after_any_jobs,
                  retained_spike_jobs=retained)


def _retained_job(job_id: int) -> None:
    result = subprocess.run(["sacct", "-n", "-P", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode"],
                            text=True, capture_output=True, check=True)
    rows = [line.split("|") for line in result.stdout.splitlines() if line.split("|")[0] == str(job_id)]
    require(len(rows) == 1, "V1_EXECUTION_RETAINED_JOB_STATE_UNKNOWN:" + str(job_id))
    state, exit_code = rows[0][1:3]
    require(state in ("PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED")
            or (state == "COMPLETED" and exit_code == "0:0"),
            "V1_EXECUTION_RETAINED_JOB_NOT_SUCCESSFUL:" + str(job_id) + ":" + state)


def _submit(output: Path, name: str, command: list[str], overrides: dict, *, resume: bool) -> int:
    directory = output / "jobs" / name
    request = dict(command=command, environment_overrides=overrides)
    if directory.exists():
        require(resume and json.loads((directory / "request.json").read_text()) == request,
                "V1_EXECUTION_SUBMISSION_REQUEST_CHANGED")
        require((directory / "response.json").exists(), "V1_EXECUTION_SUBMISSION_OUTCOME_UNKNOWN:" + name)
        result = json.loads((directory / "response.json").read_text())
        require(result.get("returncode") == 0 and type(result.get("job_id")) is int,
                "V1_EXECUTION_SUBMISSION_REQUIRES_RECONCILIATION:" + name)
        _retained_job(result["job_id"])
        return result["job_id"]
    directory.mkdir(parents=True)
    freeze_json(directory / "request.json", request)
    environment = {key: value for key, value in os.environ.items()
                   if key != "CUDA_VISIBLE_DEVICES" and not key.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "NRNSIM_V1_", "NRNSIM_BENCHMARK_"))}
    environment.update(PROJECT_PREFIX="NrnSim", TMPDIR="/tmp", **overrides)
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    response = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    job = result.stdout.strip().split(";")[0]
    if result.returncode == 0 and job.isdecimal() and int(job) > 0:
        response["job_id"] = int(job)
    freeze_json(directory / "response.json", response)
    require("job_id" in response, "V1_EXECUTION_SUBMISSION_REQUIRES_RECONCILIATION:" + name)
    return response["job_id"]



def check_gpu_availability(gpu: dict, *, source_cpu: bool = False) -> None:
    result = subprocess.run(["sinfo", "--noheader", "--Node", "--nodes="+gpu["node"],
        "--partition="+gpu["partition"], "--format=%N|%P|%t"], text=True, capture_output=True, check=True)
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    require(any(len(row) == 3 and row[0].strip() == gpu["node"]
                and row[1].strip().rstrip("*") == gpu["partition"]
                and row[2].strip() in ("idle", "mix", "alloc") for row in rows),
            "V1_EXECUTION_SOURCE_CPU_NOT_AVAILABLE" if source_cpu else "V1_EXECUTION_GPU_NODE_NOT_AVAILABLE")


def submit(cohort: ProductionCohort, plan: dict, output: Path, *, resume: bool = False,
           hold_gpu: bool = False, start_after_job: int | None = None,
           start_after_any_jobs: tuple[int, ...] = (),
           retained_spike_jobs: dict[int, int] | None = None) -> dict:
    """Submit a source chain, optionally holding GPU work for resource recovery."""
    require(start_after_job is None or (type(start_after_job) is int and start_after_job > 0),
            "V1_EXECUTION_START_DEPENDENCY_INVALID")
    require(isinstance(start_after_any_jobs, tuple)
            and all(type(job) is int and job > 0 for job in start_after_any_jobs)
            and len(set(start_after_any_jobs)) == len(start_after_any_jobs)
            and not (start_after_job is not None and start_after_any_jobs),
            "V1_EXECUTION_SOURCE_BARRIER_INVALID")
    retained_spike_jobs = {} if retained_spike_jobs is None else retained_spike_jobs
    require(isinstance(retained_spike_jobs, dict) and all(
        type(index) is int and 0 <= index < 10 and type(job) is int and job > 0
        for index, job in retained_spike_jobs.items()), "V1_EXECUTION_RETAINED_SPIKES_INVALID")
    cohort.require_current()
    require(plan == json.loads((output / "plan.json").read_text())
            and plan["source_qualification_sha256"] == cohort.qualification_sha256
            and plan["identity_table_sha256"] == cohort.identity_table_sha256
            and campaign_config(Path(plan["config"])) == cohort.campaign,
            "V1_EXECUTION_PREPARED_PLAN_CHANGED")
    require(all(sha256_file(Path(path)) == digest for path, digest in plan["source_bindings"].items()),
            "V1_EXECUTION_PREPARED_SOURCE_CHANGED")
    require(len(plan["conditions"]) == 10, "V1_EXECUTION_TEN_CONDITIONS_REQUIRED")
    for index, condition in enumerate(plan["conditions"]):
        identity = cohort.identity(plan["circuit_index"], plan["block_index"], index)
        require(condition["identity"] == identity
                and yaml.safe_load((output / "recipes" / f"condition_{index:02d}.yaml").read_text())
                    == cohort.v1_visual_config(identity), "V1_EXECUTION_PREPARED_RECIPE_CHANGED")
    repo, paths = Path(plan["repo"]), plan["paths"]
    if plan.get("backend") == "glif_cuda" and not all(
            (output/"jobs"/("block_"+phase)/"response.json").is_file() for phase in ("gpu_primary", "gpu_replay")):
        check_gpu_availability(plan["gpu"])
    if plan.get("source_cpu") and (
            not (output / "jobs" / "block_export" / "response.json").is_file()
            or not all((output / "jobs" / f"condition_{index:02d}_spikes" / "response.json").is_file()
                       for index in range(10))):
        check_gpu_availability(plan["source_cpu"], source_cpu=True)
    # Check completed archives before the first new submission. The source-plan
    # job still validates all ten archives through the existing public consumer.
    if retained_spike_jobs:
        require(plan.get("backend") == "glif_cuda", "V1_EXECUTION_RETAINED_SPIKES_REQUIRE_GLIF")
        states = job_states(list(retained_spike_jobs.values()))
        for index, job in retained_spike_jobs.items():
            require(states[job] == ("COMPLETED", "0:0"), "V1_EXECUTION_RETAINED_SPIKES_NOT_COMPLETE")
            condition = plan["conditions"][index]
            archive = repo / FAMILY / "visual_inputs" / ("spikes_" + str(job))
            report, delivery = bound_json(archive, "visual_inputs/report.json")
            require(delivery.get("exit_code") == 0
                    and delivery.get("status") == "PASS_CONSTRUCTED_VISUAL_INPUT_FULL"
                    and delivery.get("rate_archive") == condition["rate_source"]
                    and delivery.get("input_construction_archive") == paths["input_source"]
                    and report.get("exit_code") == 0 and report.get("complete_lgn_population") is True
                    and report.get("config") == cohort.v1_visual_config(condition["identity"])
                    and report.get("input_construction_binding", {}).get("delivery_sha256")
                        == sha256_file(Path(paths["input_source"]) / "delivery.json"),
                    "V1_EXECUTION_RETAINED_SPIKES_BINDING_DIFFERS")
    jobs, sources, previous = [], [], start_after_job

    def enqueue(phase, index, arguments, overrides):
        nonlocal previous
        name = f"condition_{index:02d}_{phase}" if index is not None else "block_"+phase
        suffix = f"C{plan['circuit_index']:03d}B{plan['block_index']:03d}" + (f"K{index:02d}" if index is not None else "")
        on_gpu = phase in ("gpu_primary", "gpu_replay")
        partition = plan["gpu"]["partition"] if on_gpu else "gpu"
        node = plan["gpu"]["node"] if on_gpu else "cpu03"
        if phase in ("rates", "spikes", "source_plan", "export") and plan.get("source_cpu"):
            node, partition = plan["source_cpu"]["node"], plan["source_cpu"]["partition"]
        memory_key = "gpu" if on_gpu else ("export" if phase == "source_plan" else phase)
        # Existing plans retain their unlimited reservation. Explicit budgets
        # let bounded source jobs fit scheduler backfill windows.
        walltime = plan.get("walltime_minutes", {}).get(memory_key, 0)
        command = ["sbatch", "--parsable", "--job-name=NrnSim_V1" + suffix + "_" + phase,
                   "--partition="+partition, "--nodelist="+node, "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
                   "--mem=" + str(plan["memory_gib"][memory_key]) + "G",
                   "--gres=gpu:1" if on_gpu else "--gres=none", "--time=" + str(walltime),
                   "--no-requeue", "--kill-on-invalid-dep=yes", "--chdir=/tmp",
                   "--output=/tmp/NrnSim_V1Production-%j.log"]
        if on_gpu and hold_gpu:
            command.append("--hold")
        if previous is not None:
            command.append("--dependency=afterok:" + str(previous))
        elif start_after_any_jobs:
            command.append("--dependency=afterany:" + ":".join(map(str, start_after_any_jobs)))
        command += [plan["launchers"][phase], *map(str, arguments)]
        previous = _submit(output, name, command, overrides, resume=resume)
        jobs.append(dict(phase=phase, condition_index=index, job_id=previous))
        return previous

    for index, condition in enumerate(plan["conditions"]):
        recipe = output.resolve() / "recipes" / f"condition_{index:02d}.yaml"
        rates = condition["rate_source"]
        if rates is None:
            job = enqueue("rates", index, [repo], dict(
                NRNSIM_V1_LGN_CONFIG=str(recipe), NRNSIM_V1_LGN_INPUT_CONSTRUCTION=paths["input_source"],
                NRNSIM_V1_LGN_CODE_REFERENCE=condition["rate_code_reference"],
                NRNSIM_V1_LGN_PREPARED_ARCHIVE=paths["prepared_runtime"]))
            rates = str(repo / FAMILY / "visual_inputs" / ("rates_" + str(job)))
        if index in retained_spike_jobs:
            job = retained_spike_jobs[index]
            jobs.append(dict(phase="spikes", condition_index=index, job_id=job, reused=True))
        else:
            job = enqueue("spikes", index, [repo, "full", rates, paths["input_source"], condition["spike_canary"], recipe], {})
        spikes = str(repo / FAMILY / "visual_inputs" / ("spikes_" + str(job)))
        if plan.get("backend") == "glif_cuda":
            sources.append(dict(condition_index=index, sources=dict(rate_source=rates, spike_source=spikes,
                circuit_reference=plan["gpu"]["circuit_reference"], circuit_validation=plan["gpu"]["circuit_validation"],
                authority_source=paths["authority_source"])))
            continue
        job = enqueue("native", index, [repo, paths["circuit_config"], "full", paths["recurrent_source"],
                      paths["input_source"], paths["parameters"], paths["point_config"], paths["native_canary"]],
                      dict(NRNSIM_V1_NEST_PYTHON=plan["interpreters"]["native_python"],
                           NRNSIM_V1_SPIKE_INPUTS=spikes, NRNSIM_V1_NATIVE_CODE_REFERENCE=paths["native_canary"]))
        native = str(repo / FAMILY / "native_pointnet" / ("run_" + str(job)))
        sources.append(dict(condition_index=index, sources=dict(rate_source=rates, spike_source=spikes,
                                                               native_source=native, authority_source=paths["authority_source"])))
    if plan.get("backend") == "glif_cuda":
        input_plan = output.resolve()/"source_inputs.json"
        cases = output.resolve()/"glif_cases.json"
        freeze_json(input_plan, dict(schema=GLIF_INPUT_SCHEMA, circuit_index=plan["circuit_index"],
            block_index=plan["block_index"], conditions=[dict(condition_index=row["condition_index"],
                spike_source=row["sources"]["spike_source"]) for row in sources]))
        enqueue("source_plan", None, [repo, plan["config"], plan["qualification"], plan["identity_table"],
            plan["circuit_index"], plan["block_index"], input_plan, cases, "prepare_glif"],
            dict(NRNSIM_BENCHMARK_PYTHON=plan["interpreters"]["export_python"]))
        executions = {}
        for phase in ("gpu_primary", "gpu_replay"):
            package = output.resolve()/phase
            job = enqueue(phase, None, [repo, plan["gpu"]["circuit_reference"], plan["gpu"]["snapshot"],
                package, plan["interpreters"]["export_python"], plan["gpu"]["library"], plan["gpu"]["detector"],
                "source_batch", cases, plan["gpu"]["software"]], {})
            executions[phase] = str(package/("run_"+str(job)))
        for row in sources:
            row["sources"].update(executions)
    manifest = output.resolve() / "sources.json"
    source_manifest = dict(schema=BLOCK_SCHEMA, scenario_id="v1_visual", circuit_index=plan["circuit_index"],
                           block_index=plan["block_index"], conditions=sources)
    if plan.get("backend") == "glif_cuda":
        source_manifest["execution_backend"] = "glif_cuda"
    freeze_json(manifest, source_manifest)
    enqueue("export", None, [repo, plan["config"], plan["qualification"], plan["identity_table"],
                             plan["circuit_index"], plan["block_index"], manifest, output.resolve() / "block"],
            dict(NRNSIM_BENCHMARK_PYTHON=plan["interpreters"]["export_python"]))
    report = dict(status="SUBMITTED_V1_PRODUCTION_BLOCK", jobs=jobs, inputs=str(manifest),
                  production_episodes=0, completion_report=str(output.resolve() / "block/report.json"))
    if hold_gpu and plan.get("backend") == "glif_cuda":
        report["gpu_jobs_submitted_held"] = True
    if start_after_job is not None:
        report["start_after_job_id"] = start_after_job
    if start_after_any_jobs:
        report["start_after_any_job_ids"] = list(start_after_any_jobs)
    freeze_json(output / "submission.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "qualification", "identity-table", "resources", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hold-gpu", action="store_true",
                        help="Submit GPU phases held while CPU phases remain eligible.")
    parser.add_argument("--start-after-job", type=int,
                        help="Start the first source phase after this Slurm job succeeds.")
    parser.add_argument("--start-after-any-jobs", type=int, nargs="+", default=[],
                        help="Wait for independent CPU source jobs to terminate; their failure does not cancel this block.")
    args = parser.parse_args(argv)
    cohort = ProductionCohort.load(args.config, "v1_visual", args.qualification, args.identity_table)
    plan = prepare(cohort, args.repo, args.config, args.resources, args.output, resume=args.resume)
    result = submit(cohort, plan, args.output, resume=args.resume, hold_gpu=args.hold_gpu,
                    start_after_job=args.start_after_job,
                     start_after_any_jobs=tuple(args.start_after_any_jobs)) if args.submit else dict(
        status="PREPARED_V1_PRODUCTION_BLOCK_EXECUTION", planned_conditions=10, production_episodes=0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

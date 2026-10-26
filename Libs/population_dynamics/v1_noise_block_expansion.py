"""Expand a verified V1 circuit to the remaining frozen noise blocks.

Reuse qualified rate-code and spike canaries. Generate each missing low-contrast
rate direction once on the circuit's own input construction. Source production
is submitted separately after the existing plan validator succeeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import yaml

from .benchmark_production import ProductionCohort, require
from .provenance import freeze_json, sha256_file
from .v1_execution_recipe import require_same_recipe
from .v1_nest_input_replay import bound_json
from .v1_source_node_qualification import load_source_node_qualification, qualified_source_case
from .v1_production_execution import source_barrier_jobs, FAMILY, GPU_PARTITIONS, _submit, prepare, submit


def _cohort(q: Path) -> ProductionCohort:
    return ProductionCohort.load(q / "frozen_campaign.yaml", "v1_visual",
                                q / "qualification.json", q / "identities.csv")


def prepare_directions(repo: Path, q: Path, circuit: int, references: Path,
                       readback: Path, gpu_node: str, source_node_qualification: Path | None = None,
                       rate_concurrency: int | None = None) -> dict:
    cohort = _cohort(q)
    base = q / f"circuit_{circuit:03d}"
    output = base / "noise_block_expansion"
    require(not output.exists(), "NOISE_EXPANSION_ALREADY_PREPARED")
    proof = json.loads(readback.read_text())
    require(proof["status"] == "PASS_COMPLETE_PRODUCTION_BLOCK_PUBLIC_REUSE"
            and proof["circuit_index"] == circuit and proof["block_index"] == 0
            and proof["complete_episodes"] == proof["reused_episodes"] == 10
            and proof["source_qualification_sha256"] == cohort.qualification_sha256
            and proof["identity_table_sha256"] == cohort.identity_table_sha256
            and sha256_file(Path(proof["dataset"]) / "dataset.json") == proof["catalog_sha256"],
            "NOISE_EXPANSION_ACCEPTED_FIRST_BLOCK_REQUIRED")
    template = yaml.safe_load((base / "reference_execution/block_000_resources.yaml").read_text())
    accepted = json.loads((base / "input_acceptance.json").read_text())
    require(accepted["status"] == "PASS_SOURCE_INPUT_DELIVERY_AND_REPLAY"
            and template["input_source"] == accepted["source"]
            and template["circuit_index"] == circuit and template["block_index"] == 0,
            "NOISE_EXPANSION_CIRCUIT_INPUT_BINDING_DIFFERS")
    require(gpu_node in GPU_PARTITIONS, "NOISE_EXPANSION_GPU_PLACEMENT_INVALID")
    template["gpu"].update(node=gpu_node, partition=GPU_PARTITIONS[gpu_node])
    source_node = None
    if source_node_qualification is not None:
        source_node_qualification = source_node_qualification.resolve()
        source_node = load_source_node_qualification(source_node_qualification, cohort.qualification_sha256, {})
        template["source_node_qualification"] = str(source_node_qualification)
    if rate_concurrency is None:
        rate_concurrency = 7 if source_node is not None else 2
    require(type(rate_concurrency) is int and 1 <= rate_concurrency <= 7,
            "NOISE_EXPANSION_RATE_CONCURRENCY_INVALID")
    qualified = json.loads(references.read_text())["records"]
    require([r["block_index"] for r in qualified] == list(range(1, 8)),
            "NOISE_EXPANSION_SEVEN_DIRECTION_REFERENCES_REQUIRED")
    records, recipes = [], []
    for previous in qualified:
        index = previous["block_index"]
        identity = cohort.identity(circuit, index, 9)
        recipe = cohort.v1_visual_config(identity)
        require(recipe["stimulus"]["direction_deg"] == 45 * index
                and recipe["stimulus"]["contrast_percent"] == 20,
                "NOISE_EXPANSION_FROZEN_LOW_CONTRAST_DIRECTION_DIFFERS")
        if source_node is not None:
            reference = qualified_source_case(source_node, recipe)
            previous = dict(previous, **{key: reference[key] for key in ("rate_code_reference", "spike_canary")})
        rate, rate_delivery = bound_json(Path(previous["rate_code_reference"]), "probe/probe.json")
        spike, spike_delivery = bound_json(Path(previous["spike_canary"]), "visual_inputs/report.json")
        require(rate_delivery["exit_code"] == spike_delivery["exit_code"] == 0
                and rate["status"] == "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT"
                and spike_delivery["status"] == "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY",
                "NOISE_EXPANSION_QUALIFIED_CODE_REFERENCE_REQUIRED")
        require_same_recipe(recipe, rate["config"], family="visual")
        require_same_recipe(recipe, spike["config"], family="visual")
        records.append(dict(block_index=index, direction_deg=45 * index, identity=identity,
            recipe=str(output / f"block_{index:03d}.yaml"),
            rate_code_reference=previous["rate_code_reference"], spike_canary=previous["spike_canary"]))
        recipes.append(recipe)
    cohort.require_current()
    output.mkdir()
    for row, recipe in zip(records, recipes):
        Path(row["recipe"]).write_text(yaml.safe_dump(recipe, sort_keys=False))
    (output / "execution_template.yaml").write_text(yaml.safe_dump(template, sort_keys=False))
    report = dict(status="PREPARED_SEVEN_LOW_CONTRAST_DIRECTIONS", repo=str(repo),
        cohort_root=str(q), circuit_index=circuit, input_source=template["input_source"],
        input_source_job=accepted["source_job_id"], records=records, accepted_readback=str(readback),
        rate_concurrency=rate_concurrency,
        source_simulations=0, production_episodes=0, recorded_utc=datetime.now(timezone.utc).isoformat())
    freeze_json(output / "preparation.json", report)
    return report


def _command(name: str, minutes: int, node: str = "cpu03") -> list[str]:
    require(node in ("cpu03", "cpu05", "cpu06"), "NOISE_EXPANSION_CPU_PLACEMENT_INVALID")
    return ["sbatch", "--parsable", "--job-name=NrnSim_V1" + name,
        "--partition=" + GPU_PARTITIONS[node], "--nodelist=" + node, "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        "--mem=4G", "--gres=none", "--time=" + str(minutes), "--no-requeue",
        "--kill-on-invalid-dep=yes", "--chdir=/tmp", "--output=/tmp/NrnSim_V1Expansion-%j.log"]


def _environment(repo: Path, python: str) -> list[str]:
    return ["env", "PYTHONPATH=" + str(repo / "Libs"), "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", python, "-B"]


def submit_directions(q: Path, circuit: int) -> dict:
    root = q / f"circuit_{circuit:03d}/noise_block_expansion"
    prepared = json.loads((root / "preparation.json").read_text())
    require(prepared["status"] == "PREPARED_SEVEN_LOW_CONTRAST_DIRECTIONS"
            and prepared["circuit_index"] == circuit, "NOISE_EXPANSION_PREPARATION_DIFFERS")
    require(not (root / "jobs").exists() and not (root / "submission.json").exists(),
            "NOISE_EXPANSION_RECONCILE_EXISTING_SUBMISSIONS")
    repo = Path(prepared["repo"])
    template = yaml.safe_load((root / "execution_template.yaml").read_text())
    require(template["input_source"] == prepared["input_source"]
            and template["circuit_index"] == circuit, "NOISE_EXPANSION_TEMPLATE_CHANGED")
    source_node = None
    if template.get("source_node_qualification"):
        source_node = load_source_node_qualification(Path(template["source_node_qualification"]),
                                                     _cohort(q).qualification_sha256, {})
    node = source_node["node"] if source_node is not None else "cpu03"
    state = subprocess.check_output(["sinfo", "-N", "-n", node, "-p", GPU_PARTITIONS[node], "-h", "-o", "%t"],
                                    text=True).strip()
    require(state in ("mix", "idle", "alloc", "drng", "drain"), "NOISE_EXPANSION_CPU_NODE_UNAVAILABLE")
    concurrency = prepared.get("rate_concurrency", 2)
    require(type(concurrency) is int and 1 <= concurrency <= 7,
            "NOISE_EXPANSION_RATE_CONCURRENCY_INVALID")
    jobs, lanes = [], [None] * concurrency
    report = dict(status="SUBMITTING_LOW_CONTRAST_DIRECTIONS", jobs=jobs, circuit_index=circuit,
        rate_concurrency=concurrency, source_simulations=0, production_episodes=0,
        recorded_utc=datetime.now(timezone.utc).isoformat())
    report_path = root / "submission.json"
    try:
        for offset, row in enumerate(prepared["records"]):
            block, lane = row["block_index"], offset % concurrency
            name = f"C{circuit:03d}B{block:03d}"
            command = _command(name + "_LowRate", 240, node)
            parent = lanes[lane]
            if parent:
                # Directions are independent. A failed resource predecessor
                # must not cancel later conditions; their preparation stays afterok.
                command.append("--dependency=afterany:" + str(parent))
            launcher = (source_node["launchers"]["rates"]["path"] if source_node is not None
                        else str(repo / "Libs/scripts/run_v1_source_rates.sbatch"))
            command += [launcher, str(repo)]
            overrides = dict(NRNSIM_V1_LGN_CODE_REFERENCE=row["rate_code_reference"],
                NRNSIM_V1_LGN_CONFIG=row["recipe"], NRNSIM_V1_LGN_INPUT_CONSTRUCTION=template["input_source"],
                NRNSIM_V1_LGN_PREPARED_ARCHIVE=row["rate_code_reference"])
            rate_job = _submit(root, name + "_rates", command, overrides, resume=False)
            lanes[lane] = rate_job
            resources = dict(template, block_index=block, conditions=[dict(r) for r in template["conditions"]])
            resources["conditions"][9] = dict(condition_index=9, rate_code_reference=None,
                rate_source=str(repo / FAMILY / "visual_inputs" / f"rates_{rate_job}"), spike_canary=row["spike_canary"])
            path = root / f"block_{block:03d}_resources.yaml"
            with path.open("x") as stream:
                stream.write(yaml.safe_dump(resources, sort_keys=False))
            output = q / f"circuit_{circuit:03d}/block_{block:03d}"
            arguments = _environment(repo, template["export_python"]) + [
                "-m", "population_dynamics.v1_production_execution", "--repo", str(repo),
                "--config", str(q / "frozen_campaign.yaml"), "--qualification", str(q / "qualification.json"),
                "--identity-table", str(q / "identities.csv"), "--resources", str(path), "--output", str(output)]
            command = _command(name + "_Prepare", 15, node) + [
                "--dependency=afterok:" + str(rate_job), "--wrap", shlex.join(arguments)]
            prep_job = _submit(root, name + "_prepare", command, {}, resume=False)
            jobs.append(dict(block_index=block, direction_deg=row["direction_deg"], rate_job=rate_job,
                rate_dependency=parent, preparation_job=prep_job, expected_plan=str(output / "plan.json")))
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        parents = [r["preparation_job"] for r in jobs]
        arguments = _environment(repo, template["export_python"]) + [
            "-m", "population_dynamics.v1_noise_block_expansion", "remaining",
            "--cohort-root", str(q), "--circuit-index", str(circuit)]
        command = _command(f"C{circuit:03d}RemainingBlocks", 30, node) + [
            "--dependency=afterok:" + ":".join(map(str, parents)), "--wrap", shlex.join(arguments)]
        remaining = _submit(root, "remaining_block_preparation", command, {}, resume=False)
        report.update(status="SUBMITTED_SEVEN_DIRECTIONS_AND_99_BLOCK_PREPARATIONS",
            remaining_preparation_job=remaining, remaining_preparation_dependencies=parents,
            expected_complete_plans=99,
            expected_complete_preparation=str(root / f"preparation_{remaining}.json"))
        return report
    except BaseException as error:
        report.update(status="FAILED_OR_PARTIAL_NOISE_EXPANSION_SUBMISSION",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def prepare_remaining(q: Path, circuit: int) -> dict:
    base = q / f"circuit_{circuit:03d}"
    root = base / "noise_block_expansion"
    metadata = json.loads((root / "preparation.json").read_text())
    repo = Path(metadata["repo"])
    cohort = _cohort(q)
    template = yaml.safe_load((root / "execution_template.yaml").read_text())
    require(template["circuit_index"] == circuit, "NOISE_EXPANSION_CIRCUIT_CHANGED")
    angles = {0: template["conditions"][9]}
    for angle in range(1, 8):
        # Completed plans bind the actual successful rate execution, including
        # recovery attempts. Historical submission inputs remain unchanged.
        plan = json.loads((base / f"block_{angle:03d}/plan.json").read_text())
        require(plan["circuit_index"] == circuit and plan["block_index"] == angle,
                "NOISE_EXPANSION_RATE_PLAN_IDENTITY_CHANGED")
        condition = plan["conditions"][9]
        require(condition["identity"] == cohort.identity(circuit, angle, 9)
                and condition["rate_source"] is not None,
                "NOISE_EXPANSION_COMPLETED_RATE_PLAN_REQUIRED")
        angles[angle] = dict(condition_index=9, **{key: condition[key] for key in
            ("rate_source", "rate_code_reference", "spike_canary")})
    report = dict(status="PREPARING_REMAINING_NOISE_BLOCKS", circuit_index=circuit,
        job_id=os.environ["SLURM_JOB_ID"], blocks=[], source_simulations=0, production_episodes=0)
    try:
        for block in range(1, 100):
            output = base / f"block_{block:03d}"
            if block < 8:
                plan = json.loads((output / "plan.json").read_text())
            else:
                resources = dict(template, block_index=block, conditions=[dict(r) for r in template["conditions"]])
                resources["conditions"][9] = dict(angles[block % 8])
                path = root / f"block_{block:03d}_resources.yaml"
                if path.exists():
                    require(yaml.safe_load(path.read_text()) == resources, "NOISE_EXPANSION_RETAINED_RESOURCES_CHANGED")
                else:
                    with path.open("x") as stream:
                        stream.write(yaml.safe_dump(resources, sort_keys=False))
                plan = prepare(cohort, repo, q / "frozen_campaign.yaml", path, output, resume=output.exists())
            require(plan["circuit_index"] == circuit and plan["block_index"] == block
                    and all(r["rate_source"] is not None for r in plan["conditions"])
                    and [r["identity"] for r in plan["conditions"]] ==
                        [cohort.identity(circuit, block, k) for k in range(10)],
                    "NOISE_EXPANSION_COMPLETE_PREPARED_BLOCK_REQUIRED")
            report["blocks"].append(dict(block_index=block, plan=str(output / "plan.json"), condition_count=10))
        report["status"] = "PREPARED_99_ADDITIONAL_COMPLETE_NOISE_BLOCKS"
        return report
    except BaseException as error:
        report.update(status="FAILED_REMAINING_NOISE_BLOCK_PREPARATION",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["recorded_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(root / f"preparation_{os.environ['SLURM_JOB_ID']}.json", report)



def _source_prerequisite(job: int) -> int | None:
    observed = subprocess.run(["scontrol", "show", "job", "-o", str(job)],
                              text=True, capture_output=True)
    if observed.returncode == 0:
        values = dict(item.split("=", 1) for item in observed.stdout.split() if "=" in item)
        state, code = values["JobState"], values["ExitCode"]
    else:
        require("Invalid job id" in observed.stderr + observed.stdout, "NOISE_SOURCE_PARENT_QUERY_FAILED")
        output = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", str(job),
                                          "-o", "JobIDRaw,State,ExitCode"], text=True)
        rows = [line.split("|") for line in output.splitlines() if line.split("|")[0] == str(job)]
        require(len(rows) == 1, "NOISE_SOURCE_PARENT_ACCOUNTING_MISSING")
        state, code = rows[0][1:3]
    if state == "COMPLETED":
        require(code == "0:0", "NOISE_SOURCE_PARENT_FAILED")
        return None
    require(state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"),
            "NOISE_SOURCE_PARENT_NOT_EXECUTABLE")
    return job


def submit_source_blocks(q: Path, circuit: int, preparation: Path, first: int,
                         last: int, gpu_node: str | None = None) -> dict:
    """Dispatch at most six validated blocks while preserving three source lanes."""
    require(1 <= first <= last < 100 and last - first < 6, "NOISE_SOURCE_BATCH_ONE_TO_SIX_BLOCKS_REQUIRED")
    prepared = json.loads(preparation.read_text())
    require(prepared["status"] == "PREPARED_99_ADDITIONAL_COMPLETE_NOISE_BLOCKS"
            and prepared["circuit_index"] == circuit, "NOISE_SOURCE_COMPLETE_PREPARATION_REQUIRED")
    records = {row["block_index"]: row for row in prepared["blocks"]}
    require(len(prepared["blocks"]) == 99 and set(records) == set(range(1, 100)),
            "NOISE_SOURCE_COMPLETE_PLAN_INDEX_REQUIRED")
    plans = [(Path(records[i]["plan"]).parent, json.loads(Path(records[i]["plan"]).read_text()))
             for i in range(first, last + 1)]
    estimated = ((2 + len(plans)) * (q / "identities.csv").stat().st_size
                 + sum(Path(name).stat().st_size for _, plan in plans for name in plan["source_bindings"])
                 + 16 * 1024**2)
    require(estimated <= 256 * 1024**2, "NOISE_SOURCE_MANAGEMENT_IO_LIMIT")
    destination = q / f"circuit_{circuit:03d}/source_batches"
    receipt = destination / f"batch_{first:03d}_{last:03d}.json"
    require(not receipt.exists(), "NOISE_SOURCE_RECONCILE_EXISTING_BATCH")
    destination.mkdir(exist_ok=True)
    report = dict(status="SUBMITTING_BOUNDED_SOURCE_BATCH", circuit_index=circuit, first_block=first,
        last_block=last, source_lanes=3, preparation=str(preparation), blocks=[],
        estimated_control_plane_read_bytes=estimated, production_episodes=0,
        recorded_utc=datetime.now(timezone.utc).isoformat())

    def save():
        temporary = receipt.with_suffix(".writing")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(receipt)

    save()
    cohort = _cohort(q)
    started = time.monotonic()
    try:
        for original, original_plan in plans:
            require(time.monotonic() - started < 240, "NOISE_SOURCE_MANAGEMENT_TIME_LIMIT")
            block = original_plan["block_index"]
            require(original_plan["circuit_index"] == circuit and original_plan.get("backend") == "glif_cuda",
                    "NOISE_SOURCE_PREPARED_CIRCUIT_OR_BACKEND_DIFFERS")
            plan = deepcopy(original_plan)
            output = original
            if gpu_node is not None and gpu_node != plan["gpu"]["node"]:
                require(gpu_node in GPU_PARTITIONS, "NOISE_SOURCE_GPU_NODE_INVALID")
                output = original / "gpu_admission"
                plan["gpu"].update(node=gpu_node, partition=GPU_PARTITIONS[gpu_node])
                plan["config"] = str(output / "campaign.yaml")
                output.mkdir(exist_ok=True)
                freeze_json(output / "plan.json", plan)
                for relative in ["campaign.yaml", *[f"recipes/condition_{i:02d}.yaml" for i in range(10)]]:
                    target = output / relative
                    contents = (original / relative).read_bytes()
                    if target.exists():
                        require(target.read_bytes() == contents, "NOISE_SOURCE_RETAINED_CONFIGURATION_CHANGED")
                    else:
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(contents)
            require(not (output / "jobs").exists() and not (output / "submission.json").exists(),
                    "NOISE_SOURCE_RECONCILE_EXISTING_BLOCK")
            barrier, parent = (), None
            if block > 3:
                previous = Path(records[block - 3]["plan"]).parent
                available = [p for p in (previous / "submission.json",
                                         previous / "gpu_admission/submission.json") if p.exists()]
                require(len(available) == 1, "NOISE_SOURCE_UNIQUE_PREVIOUS_SUBMISSION_REQUIRED")
                earlier = json.loads(available[0].read_text())
                parents = [r["job_id"] for r in earlier["jobs"] if r["phase"] == "source_plan"]
                require(len(parents) == 1, "NOISE_SOURCE_PREVIOUS_PLAN_JOB_REQUIRED")
                parent = parents[0]
                barrier = source_barrier_jobs(earlier)
            row = dict(circuit_index=circuit, block_index=block, lane=(block - 1) % 3,
                previous_block=block - 3 if block > 3 else None, previous_source_plan_job=parent,
                start_after_any_job_ids=list(barrier), original_plan=str(original / "plan.json"),
                execution_root=str(output), gpu_node=plan["gpu"]["node"], gpu_memory_gib=plan["memory_gib"]["gpu"])
            report["blocks"].append(row)
            save()
            result = submit(cohort, plan, output, start_after_any_jobs=barrier)
            row.update(submission=str(output / "submission.json"), jobs=result["jobs"])
            save()
        report["status"] = "SUBMITTED_BOUNDED_THREE_LANE_SOURCE_BATCH"
        return report
    except BaseException as error:
        report.update(status="FAILED_OR_PARTIAL_SOURCE_BATCH_DISPATCH",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "remaining", "source-blocks"))
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--circuit-index", type=int, required=True, choices=range(1, 100))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--readback", type=Path)
    parser.add_argument("--gpu-node", choices=tuple(GPU_PARTITIONS))
    parser.add_argument("--source-node-qualification", type=Path)
    parser.add_argument("--rate-concurrency", type=int, choices=range(1, 8))
    parser.add_argument("--preparation-report", type=Path)
    parser.add_argument("--first-block", type=int)
    parser.add_argument("--last-block", type=int)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if any(getattr(args, name) is None for name in ("repo", "references", "readback")):
            parser.error("prepare requires --repo, --references and --readback")
        result = prepare_directions(args.repo.resolve(), args.cohort_root.resolve(), args.circuit_index,
                                    args.references.resolve(), args.readback.resolve(), args.gpu_node or "cpu04",
                                    args.source_node_qualification, args.rate_concurrency)
    elif args.action == "submit":
        result = submit_directions(args.cohort_root.resolve(), args.circuit_index)
    elif args.action == "remaining":
        result = prepare_remaining(args.cohort_root.resolve(), args.circuit_index)
    else:
        if any(getattr(args, name) is None for name in ("preparation_report", "first_block", "last_block")):
            parser.error("source-blocks requires --preparation-report, --first-block and --last-block")
        result = submit_source_blocks(args.cohort_root.resolve(), args.circuit_index,
            args.preparation_report.resolve(), args.first_block, args.last_block, args.gpu_node)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

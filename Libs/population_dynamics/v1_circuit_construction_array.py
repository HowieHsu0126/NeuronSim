"""Route Slurm array elements to the existing full V1 source constructors.

The array index selects an already prepared circuit. Per-circuit receipts bind
Slurm's raw job IDs to archives, so input construction uses its own recurrent
graph even when array elements start out of order. These are construction
receipts; complete episodes are counted by the production block consumer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import yaml

from .provenance import freeze_json, sha256_file


FAMILY = Path("Output/neural_population_dynamics/multicircuit/v1_asset_intake/"
              "original_graph_route/circuit_construction_source")
STAGES = {
    "recurrent": ("run_v1_recurrent_construction.sbatch", "recurrent_construction",
                  "PASS_SOURCE_RECURRENT_BUILD_REPEATABILITY"),
    "inputs": ("run_v1_input_construction.sbatch", "input_construction",
               "PASS_SOURCE_INPUT_BUILD_REPEATABILITY"),
}



ROUTING_SCHEMA = "v1_construction_cpu_routes_v1"
ROUTING_CONFIG = Path("Libs/configs/neural_population_v1_construction_execution.yaml")


def node_launcher_text(repo: Path, stage: str, node: str) -> str:
    """Change only the source wrapper's CPU placement checks."""
    if node not in ("cpu05", "cpu06"):
        raise ValueError("V1_CONSTRUCTION_BULK_CPU_REQUIRED")
    launcher = STAGES[stage][0]
    original = (repo / "Libs/scripts" / launcher).read_text()
    replacements = [
        ("node not in ('cpu03', 'cpu04')", f"node not in ('cpu03', 'cpu04', '{node}')"),
        ("mode == 'full' and node != 'cpu03'", f"mode == 'full' and node not in ('cpu03', '{node}')"),
    ]
    for before, after in replacements:
        if original.count(before) != 1:
            raise ValueError("V1_CONSTRUCTION_SOURCE_NODE_GUARD_CHANGED")
        original = original.replace(before, after, 1)
    return original


def _input_qualification_path(repo: Path, settings: dict, node: str) -> str | None:
    qualifications = settings.get("input_qualifications", {})
    if not isinstance(qualifications, dict):
        raise ValueError("V1_CONSTRUCTION_INPUT_QUALIFICATION_INVALID")
    if node not in qualifications:
        return None
    configured = qualifications[node]
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("V1_CONSTRUCTION_INPUT_QUALIFICATION_INVALID")
    return str((repo / configured).resolve())


def select_launcher(repo: Path, stage: str, preparation: Path, output: Path,
                    index: int, prepared: dict) -> tuple[Path, dict | None]:
    node = os.environ["SLURMD_NODENAME"]
    canonical = repo / "Libs/scripts" / STAGES[stage][0]
    config = repo / ROUTING_CONFIG
    if node == "cpu03":
        if stage != "inputs" or not config.is_file():
            return canonical, None
    elif node not in ("cpu05", "cpu06") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("V1_CONSTRUCTION_CPU_PLACEMENT_INVALID")
    settings = yaml.safe_load(config.read_text())
    if not isinstance(settings, dict):
        raise ValueError("V1_CONSTRUCTION_NODE_ROUTE_NOT_CONFIGURED")
    input_qualification = (_input_qualification_path(repo, settings, node)
                           if stage == "inputs" else None)
    if node == "cpu03" and input_qualification is None:
        return canonical, None
    if (settings.get("schema") != ROUTING_SCHEMA
            or settings.get("preparation_sha256") != sha256_file(preparation)
            or (node != "cpu03" and node not in settings.get("nodes", {}))):
        raise ValueError("V1_CONSTRUCTION_NODE_ROUTE_NOT_CONFIGURED")
    if node == "cpu03":
        # The canonical launcher retains its source checks. Only the explicit
        # implementation snapshot and its canary qualification are routed.
        return canonical, dict(node=node, configuration=str(config),
                               launcher_sha256=sha256_file(canonical),
                               input_qualification=input_qualification)
    selected = settings["nodes"][node]
    pilot = selected["pilot_circuit_index"]
    if type(pilot) is not int or not 0 <= pilot < 100:
        raise ValueError("V1_CONSTRUCTION_NODE_PILOT_INVALID")
    pilots = [row for row in prepared["circuits"] if row["circuit_index"] == pilot]
    if len(pilots) != 1:
        raise ValueError("V1_CONSTRUCTION_NODE_PILOT_NOT_IN_FROZEN_COHORT")
    launcher = (repo / selected["launchers"][stage]).resolve()
    if launcher.read_text() != node_launcher_text(repo, stage, node):
        raise ValueError("V1_CONSTRUCTION_NODE_LAUNCHER_CHANGED")
    proof_path = output / f"circuit_{pilot:03d}" / stage / "report.json"
    configured_report = settings.get("qualification_reports", {}).get(node, {}).get(stage)
    if configured_report is not None:
        proof_path = repo / configured_report
    if index != pilot:
        if not proof_path.is_file():
            raise ValueError("V1_CONSTRUCTION_COMPLETE_NODE_PILOT_REQUIRED")
        proof = json.loads(proof_path.read_text())
        if (proof.get("status") != "COMPLETE_SOURCE_CONSTRUCTION" or proof.get("exit_code") != 0
                or proof.get("node") != node or proof.get("stage") != stage
                or proof.get("circuit") != pilots[0]
                or proof.get("node_route", {}).get("profile") != selected
                or proof.get("node_route", {}).get("launcher_sha256") != sha256_file(launcher)):
            raise ValueError("V1_CONSTRUCTION_COMPLETE_NODE_PILOT_REQUIRED")
        if input_qualification is not None and (
                proof.get("node_route", {}).get("input_qualification") != input_qualification
                or proof.get("command", [])[-1:] != [input_qualification]
                or proof.get("environment_overrides", {}).get("NRNSIM_V1_INPUT_CODE_REFERENCE")
                != input_qualification):
            raise ValueError("V1_CONSTRUCTION_INPUT_QUALIFICATION_DIFFERS_FROM_PILOT")
        archive = Path(proof["archive"])
        delivery = json.loads((archive / "delivery.json").read_text())
        if (sha256_file(archive / "delivery.json") != proof["delivery_sha256"]
                or delivery.get("status") != STAGES[stage][2] or delivery.get("exit_code") != 0
                or delivery.get("mode") != "full" or delivery.get("job_id") != proof["job_id"]
                or delivery.get("node") != node):
            raise ValueError("V1_CONSTRUCTION_NODE_PILOT_DELIVERY_CHANGED")
    route = dict(node=node, pilot_circuit_index=pilot, is_pilot=index == pilot,
                 configuration=str(config), profile=selected,
                 launcher_sha256=sha256_file(launcher),
                 qualification_report=str(proof_path) if index != pilot else None)
    if input_qualification is not None:
        route["input_qualification"] = input_qualification
    return launcher, route


def run(repo: Path, stage: str, preparation: Path, output: Path,
        qualification: Path) -> dict:
    """Execute one prepared circuit and retain its exact launcher exit status."""
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    array_id = os.environ["SLURM_ARRAY_JOB_ID"]
    job_id = os.environ["SLURM_JOB_ID"]
    if index < 0 or not array_id.isdecimal() or not job_id.isdecimal():
        raise ValueError("V1_CONSTRUCTION_ARRAY_ID_INVALID")
    prepared = json.loads(preparation.read_text())
    rows = [row for row in prepared["circuits"] if row["circuit_index"] == index]
    if len(rows) != 1:
        raise ValueError("V1_CONSTRUCTION_PREPARED_CIRCUIT_REQUIRED")
    circuit = rows[0]
    config = Path(circuit["config"])
    if sha256_file(config) != circuit["config_sha256"]:
        raise ValueError("V1_CONSTRUCTION_PREPARED_CONFIG_CHANGED")
    launcher, family, expected_status = STAGES[stage]
    archive = repo / FAMILY / family / ("run_" + job_id)
    circuit_output = output / f"circuit_{index:03d}"
    launcher_path, node_route = select_launcher(repo, stage, preparation, output, index, prepared)
    command = ["bash", str(launcher_path), str(repo), str(config)]
    environment_overrides = {}
    if node_route is not None and "input_qualification" in node_route:
        # The launcher checks the snapshot against the same canary qualification.
        qualification = Path(node_route["input_qualification"])
        environment_overrides["NRNSIM_V1_INPUT_CODE_REFERENCE"] = str(qualification)
    parent = None
    if stage == "recurrent":
        command += [str(qualification), "full"]
    else:
        parent = json.loads((circuit_output / "recurrent/report.json").read_text())
        if (parent.get("status") != "COMPLETE_SOURCE_CONSTRUCTION"
                or parent.get("exit_code") != 0 or parent.get("circuit") != circuit
                or parent.get("stage") != "recurrent"):
            raise ValueError("V1_CONSTRUCTION_MATCHING_RECURRENT_COMPLETION_REQUIRED")
        command += ["full", parent["archive"], str(qualification)]
    destination = circuit_output / stage
    destination.mkdir(parents=True, exist_ok=False)
    request = dict(stage=stage, circuit=circuit, array_job_id=array_id,
                   array_task_id=index, job_id=job_id, command=command,
                   preparation=str(preparation), archive=str(archive),
                   parent_job_id=parent["job_id"] if parent else None,
                   production_episodes=0)
    request["node"] = os.environ["SLURMD_NODENAME"]
    if node_route is not None:
        request["node_route"] = node_route
    if environment_overrides:
        request["environment_overrides"] = environment_overrides
    freeze_json(destination / "request.json", request)
    report = dict(request, status="FAILED_SOURCE_CONSTRUCTION", exit_code=1)
    try:
        result = subprocess.run(command, check=False, env=dict(os.environ, **environment_overrides))
        report["exit_code"] = result.returncode
        if result.returncode == 0:
            delivery = json.loads((archive / "delivery.json").read_text())
            if (delivery.get("exit_code") != 0 or delivery.get("status") != expected_status
                    or delivery.get("job_id") != job_id or delivery.get("mode") != "full"
                    or (node_route is not None and delivery.get("node") != node_route["node"])):
                raise ValueError("V1_CONSTRUCTION_SOURCE_DELIVERY_NOT_COMPLETE")
            report.update(status="COMPLETE_SOURCE_CONSTRUCTION",
                          delivery_sha256=sha256_file(archive / "delivery.json"))
    except Exception as error:
        report.update(exit_code=report["exit_code"] or 1,
                      error_type=type(error).__name__, error=str(error))
    finally:
        report["recorded_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(destination / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(**vars(args))
    print(json.dumps(report), flush=True)
    return report["exit_code"] if report["exit_code"] >= 0 else 128 - report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

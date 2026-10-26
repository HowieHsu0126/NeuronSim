"""Prepare the two missing V1 gray/low-contrast conditions using existing runners."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import yaml

from .benchmark_tasks import v1_visual_config
from .circuit_campaign import campaign_config, episode_identities
from .provenance import freeze_json


def prepare(repo: Path, config: Path, reference: Path, output: Path, *, submit: bool = False,
            condition_indices: tuple[int, ...] = (8, 9), reused_rates: dict[int, int] | None = None) -> dict:
    if output.exists():
        raise FileExistsError("V1_BENCHMARK_PREPARATION_EXISTS")
    if not condition_indices or len(set(condition_indices)) != len(condition_indices) or not set(condition_indices) <= {8, 9}:
        raise ValueError("V1_MISSING_CONDITION_INDICES_REQUIRED")
    campaign = campaign_config(config)
    protocol = campaign["scenarios"]["v1_visual"]["task_protocol"]
    base = yaml.safe_load((repo / "Libs/configs/neural_population_v1_constructed_visual_development.yaml").read_text())
    original = json.loads(reference.read_text())
    reused_rates = reused_rates or {}
    if not set(reused_rates) <= set(condition_indices) or any(type(job) is not int or job <= 0 for job in reused_rates.values()):
        raise ValueError("V1_REUSED_RATE_JOB_INVALID")
    reference_jobs = {row["phase"]: row for row in original["jobs"] if row.get("direction_deg") == 0}
    phases = ("representative_rates", "representative_spikes", "full_rates", "full_spikes", "full_native_and_replay")
    if set(reference_jobs) != set(phases):
        raise ValueError("V1_COMPLETE_EXISTING_DIRECTION_CHAIN_REQUIRED")
    output.mkdir(parents=True)
    record = dict(status="PREPARED_V1_BENCHMARK_CONDITIONS", submitted=False, jobs=[],
                  native_complete_conditions=0, production_episodes=0,
                  reused_direction_conditions=8, exposed_circuit_retains_development_identity=True)
    for index in condition_indices:
        identity = next(row for row in episode_identities(campaign, "v1_visual")
                        if row["circuit_index"] == 0 and row["block_index"] == 0 and row["condition_index"] == index)
        visual = v1_visual_config(base, protocol, identity)
        # The first condition envelope uses the same existing source noise seed
        # as the eight direction references. This remains an exposed circuit.
        visual["spike_generation"]["starting_noise_seed"] = base["spike_generation"]["starting_noise_seed"]
        visual["background"]["starting_noise_seed"] = base["background"]["starting_noise_seed"]
        label = "gray" if index == 8 else "low_contrast"
        path = output / (label + ".yaml")
        path.write_text(yaml.safe_dump(visual, sort_keys=False))
        replaced = {}; previous = None
        selected_phases = phases
        if index in reused_rates:
            reused = reused_rates[index]
            rates = repo / "Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/visual_inputs" / ("rates_" + str(reused))
            delivery = json.loads((rates / "delivery.json").read_text())
            rate_report = json.loads((rates / "probe/probe.json").read_text())
            representative_config = dict(visual, unit_coverage="model_representatives")
            if (delivery.get("exit_code") != 0 or rate_report.get("status") != "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT"
                    or rate_report.get("config") != representative_config):
                raise ValueError("V1_REUSED_COMPLETE_REPRESENTATIVE_RATES_REQUIRED")
            # Existing generators authenticate these declared archives at use.
            record["jobs"].append(dict(condition=label, phase="representative_rates", job_id=reused,
                                       reused=True, archive=str(rates)))
            replaced[reference_jobs["representative_rates"]["job_id"]] = reused
            selected_phases = phases[1:]
        for phase in selected_phases:
            command = copy.deepcopy(reference_jobs[phase]["command"])
            command = [arg for arg in command if not arg.startswith("--dependency=")]
            for position, argument in enumerate(command):
                if argument.startswith("--job-name="):
                    command[position] = "--job-name=NrnSim_V1" + ("Gray" if index == 8 else "LowContrast") + phase.title().replace("_", "")
                elif argument.startswith("--export="):
                    values = argument.split(",")
                    values = [item for item in values if not item.startswith("TMPDIR=")]
                    values.append("TMPDIR=/tmp")
                    values.append("SLURM_CPU_BIND=none")
                    if phase == "full_native_and_replay":
                        values.append("NRNSIM_V1_NATIVE_CODE_REFERENCE=" + str(repo / "Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/native_pointnet/run_598291"))
                    for value_index, value in enumerate(values):
                        if value.startswith("NRNSIM_V1_LGN_CONFIG="):
                            values[value_index] = "NRNSIM_V1_LGN_CONFIG=" + str(path.resolve())
                        for old, new in replaced.items():
                            values[value_index] = values[value_index].replace(str(old), str(new))
                    command[position] = ",".join(values)
                else:
                    for old, new in replaced.items():
                        command[position] = command[position].replace(str(old), str(new))
            if previous is not None:
                command.insert(2, "--dependency=afterok:" + str(previous))
            row = dict(condition=label, phase=phase, command=command)
            if submit:
                environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
                result = subprocess.run(command, text=True, capture_output=True, env=environment)
                row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                record["jobs"].append(row)
                freeze_json(output / (label + "_" + phase + "_submission.json"), row)
                if result.returncode:
                    raise RuntimeError("V1_BENCHMARK_SUBMISSION_FAILED:" + result.stderr)
                previous = int(result.stdout.strip().split(";")[0])
                row["job_id"] = previous
                replaced[reference_jobs[phase]["job_id"]] = previous
            else:
                record["jobs"].append(row)
    record.update(status="SUBMITTED_V1_BENCHMARK_CONDITIONS" if submit else record["status"], submitted=submit)
    freeze_json(output / "submission.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--condition", choices=("gray", "low_contrast", "both"), default="both")
    parser.add_argument("--reuse-gray-rates-job", type=int)
    parser.add_argument("--reuse-low-contrast-rates-job", type=int)
    args = parser.parse_args()
    indices = {"gray": (8,), "low_contrast": (9,), "both": (8, 9)}[args.condition]
    reused = {index: job for index, job in ((8, args.reuse_gray_rates_job), (9, args.reuse_low_contrast_rates_job)) if job is not None}
    prepare(args.repo, args.config, args.reference, args.output, submit=args.submit, condition_indices=indices, reused_rates=reused)


if __name__ == "__main__":
    main()

"""Prospective identities and production prerequisites for three neural circuits.

An identity is a complete source-simulator episode, never a neuron, bin, crop,
replay, or model-generated augmentation. Circuit and stochastic replication
levels remain distinct in every consumer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator

from .config import load_config
from .provenance import freeze_json, sha256_file

SCENARIOS = ("v1_visual", "pfc_working_memory", "ca3_pattern_completion")
SPLITS = ("train", "validation", "calibration", "test")
# Four source streams use offsets 0/10M/20M/30M. The circuit/block
# encoding below occupies at most 1M values under the supported 1000 limits.
SEED_NAMESPACE_SPAN = 31000000
MAX_CAMPAIGN_SEED = 2**31 - 2  # Fits the retained CA3 host-engine seed contract.


def validate_seed_namespaces(config: dict) -> None:
    occupied = [(100000000 * (index + 1), name + ":development")
                for index, name in enumerate(SCENARIOS)]
    for name in SCENARIOS:
        specification = config["scenarios"][name]
        if "production_seed_namespace" not in specification:
            continue
        base = specification["production_seed_namespace"]
        if (type(base) is not int or base < 1
                or base + SEED_NAMESPACE_SPAN - 1 > MAX_CAMPAIGN_SEED):
            raise ValueError("PRODUCTION_SEED_NAMESPACE_OUT_OF_SOURCE_RANGE:" + name)
        for previous, owner in occupied:
            if max(base, previous) < min(base, previous) + SEED_NAMESPACE_SPAN:
                raise ValueError("PRODUCTION_SEED_NAMESPACE_OVERLAP:" + name + ":" + owner)
        occupied.append((base, name + ":production"))


def seed_namespace(config: dict, scenario: str, *, production: bool = False) -> int:
    if scenario not in SCENARIOS or type(production) is not bool:
        raise ValueError("CIRCUIT_SEED_NAMESPACE_SCOPE_INVALID")
    validate_seed_namespaces(config)
    if production:
        base = config["scenarios"][scenario].get("production_seed_namespace")
        if base is not None:
            return base
        # PFC is owned by its existing campaign; preserve its supported identity
        # allocation. V1/CA3 production must separate previously exposed seeds.
        if scenario != "pfc_working_memory":
            raise ValueError("FRESH_PRODUCTION_SEED_NAMESPACE_REQUIRED:" + scenario)
    return 100000000 * (SCENARIOS.index(scenario) + 1)


def campaign_config(path: Path) -> dict:
    campaign = load_config(path, validate_training_seeds=False)["simulation_campaign"]
    if campaign.get("schema") != "neural_circuit_campaign_v1":
        raise ValueError("CIRCUIT_CAMPAIGN_SCHEMA_REQUIRED")
    if campaign.get("sample_unit") != "complete_mechanistic_episode":
        raise ValueError("COMPLETE_MECHANISTIC_EPISODES_REQUIRED")
    if set(campaign.get("scenarios", {})) != set(SCENARIOS):
        raise ValueError("THREE_DISTINCT_CIRCUITS_REQUIRED")
    counts = campaign["sampling"]
    for name in ("circuit_instances", "initial_noise_blocks_per_circuit", "conditions_per_block"):
        if type(counts.get(name)) is not int or counts[name] < 1:
            raise ValueError("POSITIVE_SAMPLING_COUNT_REQUIRED:" + name)
    total = counts["circuit_instances"] * counts["initial_noise_blocks_per_circuit"] * counts["conditions_per_block"]
    if total != campaign["episodes_per_scenario"] or total != 100000:
        raise ValueError("ONE_HUNDRED_THOUSAND_COMPLETE_EPISODES_REQUIRED")
    split_counts = campaign["circuit_split_counts"]
    if (set(split_counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in split_counts.values())
            or sum(split_counts.values()) != counts["circuit_instances"]):
        raise ValueError("WHOLE_CIRCUIT_SPLIT_COUNTS_INVALID")
    if counts["circuit_instances"] > 1000 or counts["initial_noise_blocks_per_circuit"] > 1000:
        raise ValueError("SEED_NAMESPACE_CAPACITY_EXCEEDED")
    validate_seed_namespaces(campaign)
    for name, specification in campaign["scenarios"].items():
        if specification["minimum_core_neurons"] < 100000:
            raise ValueError("COUPLED_CORE_SCALE_BELOW_AUTHORITY:" + name)
        protocol = specification.get("task_protocol")
        if isinstance(protocol, dict) and protocol.get("schema") == "neural_circuit_benchmark_tasks_v1":
            from .benchmark_tasks import validate_protocol

            validate_protocol(protocol, name)
    return campaign


def episode_identities(config: dict, scenario: str, *, production: bool = False) -> Iterator[dict]:
    """Preserve development seeds; production explicitly selects its reserved pool."""
    if scenario not in SCENARIOS:
        raise ValueError("UNKNOWN_CIRCUIT_SCENARIO")
    sampling = config["sampling"]
    namespace = seed_namespace(config, scenario, production=production)
    circuit = 0
    for split in SPLITS:
        for _ in range(config["circuit_split_counts"][split]):
            circuit_id = f"{scenario}_circuit_{circuit:03d}"
            for block in range(sampling["initial_noise_blocks_per_circuit"]):
                seed = circuit * 1000 + block
                for condition in range(sampling["conditions_per_block"]):
                    yield dict(scenario_id=scenario, circuit_id=circuit_id, circuit_index=circuit,
                               episode_id=f"{circuit_id}_block_{block:03d}_condition_{condition:02d}",
                               block_id=f"{circuit_id}_block_{block:03d}", block_index=block,
                               condition_index=condition, split=split,
                               connectivity_seed=namespace + circuit,
                               initial_state_seed=namespace + 10000000 + seed,
                               background_seed=namespace + 20000000 + seed,
                               stimulus_seed=namespace + 30000000 + seed,
                               sample_unit="complete_mechanistic_episode")
            circuit += 1


def pfc_source_admission_prerequisites(specification: dict, qualification: dict | None) -> list[str]:
    """Apply the approved reference-source scope without promoting source checks."""
    policy = specification.get("source_admission")
    if policy is None:
        return []  # Existing frozen campaign snapshots retain their original contract.
    if (not isinstance(policy, dict) or policy.get("schema") != "pfc_source_admission_v1"
            or policy.get("status") != "AUTHORIZED"
            or policy.get("reference_cohort") != "numpy_dt_0p0125"
            or policy.get("reference_dt_ms") != 0.0125
            or policy.get("reference_backend") != "numpy"
            or policy.get("reference_source_definition") != "discrete_adapted_EPFL"
            or policy.get("reference_qualification_independent_of_coarse_engine") is not True
            or policy.get("coarse_engine_cross_clock_acceptance_required") is not True):
        return ["pfc_source_admission_policy"]
    if qualification is None:
        return []
    route = qualification.get("source_admission_route")
    clock = qualification.get("source_clock", {})
    dt = specification.get("task_protocol", {}).get("dt_ms")
    missing = []
    if not isinstance(clock, dict) or clock.get("dt_ms") != dt:
        missing.append("pfc_source_clock_binding")
        clock = {} if not isinstance(clock, dict) else clock
    if route == "independent_reference_source":
        if (dt != policy["reference_dt_ms"] or clock.get("backend") != policy["reference_backend"]
                or qualification.get("source_definition") != policy["reference_source_definition"]):
            missing.append("pfc_reference_source_binding")
    elif route == "coarse_engine":
        acceptance = qualification.get("cross_clock_acceptance", {})
        if (not isinstance(acceptance, dict) or acceptance.get("status") != "PASS"
                or acceptance.get("complete_frozen_domain") is not True
                or acceptance.get("combined_error_budget_accepted") is not True
                or acceptance.get("reference_cohort") != policy["reference_cohort"]
                or not acceptance.get("evidence")):
            missing.append("pfc_coarse_engine_cross_clock_acceptance")
    else:
        missing.append("pfc_source_admission_route")
    return missing


def production_prerequisites(config: dict, scenario: str, qualification: dict | None) -> list[str]:
    specification = config["scenarios"][scenario]
    admission_missing = (pfc_source_admission_prerequisites(specification, qualification)
                         if scenario == "pfc_working_memory" else [])
    missing = [name for name in ("parameter_domain", "task_protocol", "observation_map")
               if not isinstance(specification.get(name), dict) or not specification[name]]
    try:
        seed_namespace(config, scenario, production=True)
    except ValueError as error:
        missing.append(str(error))
    for name in ("parameter_domain", "task_protocol", "observation_map"):
        value = specification.get(name)
        if isinstance(value, dict) and value.get("frozen") is False:
            missing.append(name + "_not_frozen")
    missing.extend(admission_missing)
    if qualification is None:
        return [*missing, "source_qualification"]
    if qualification.get("scenario_id") != scenario or qualification.get("status") != "PASS":
        missing.append("source_qualification")
    if qualification.get("coupled_core_neurons", 0) < specification["minimum_core_neurons"]:
        missing.append("coupled_core_scale")
    for field in ("source_reproduction", "numerical_integrity", "driver_statistics", "control_delivery",
                  "full_unit_recording", "coupling", "scaling_fidelity"):
        if qualification.get("checks", {}).get(field) is not True:
            missing.append(field)
    if not qualification.get("source_revision") or not qualification.get("evidence"):
        missing.append("source_evidence")
    return missing


def describe(config: dict) -> dict:
    block_count = config["sampling"]["initial_noise_blocks_per_circuit"]
    conditions = config["sampling"]["conditions_per_block"]
    return dict(schema=config["schema"], sample_unit=config["sample_unit"],
                planned_episodes_total=len(SCENARIOS) * config["episodes_per_scenario"],
                planned_episodes_per_scenario=config["episodes_per_scenario"],
                planned_episodes_by_split={k: n * block_count * conditions for k, n in config["circuit_split_counts"].items()},
                sampling=config["sampling"], training_seeds=[41, 42, 43],
                completed_episodes_not_inferred_from_plan=True,
                scenarios={name: dict(production_ready=False,
                                     prerequisites=production_prerequisites(config, name, None),
                                     source_family=config["scenarios"][name]["source_family"])
                           for name in SCENARIOS})


def freeze_identities(config: dict, scenario: str, qualification_path: Path, output: Path) -> None:
    qualification = json.loads(qualification_path.read_text())
    missing = production_prerequisites(config, scenario, qualification)
    if missing:
        raise ValueError("CIRCUIT_PRODUCTION_NOT_QUALIFIED:" + ",".join(missing))
    if output.exists():
        raise FileExistsError("PROSPECTIVE_IDENTITY_TABLE_ALREADY_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".writing")
    try:
        rows = episode_identities(config, scenario, production=True)
        first = next(rows)
        with temporary.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(rows)
        temporary.replace(output)
        freeze_json(output.with_suffix(".json"), dict(
            scenario_id=scenario, resolved_campaign=config, status="FROZEN_IDENTITIES_ONLY",
            source_qualification_sha256=sha256_file(qualification_path),
            identity_table_sha256=sha256_file(output), complete_episodes=0))
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    config = campaign_config(args.config)
    if args.scenario:
        if args.qualification is None:
            parser.error("--scenario requires --qualification before identity freeze")
        freeze_identities(config, args.scenario, args.qualification, args.output)
    else:
        freeze_json(args.output, describe(config))
    print(json.dumps(describe(config)))


if __name__ == "__main__":
    main()

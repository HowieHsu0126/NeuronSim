"""Resolve the 100 circuit x 100 noise block x 10 condition benchmark envelope."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from itertools import islice
from pathlib import Path

import yaml

from .benchmark_tasks import condition_for_episode, pfc_working_memory_config, v1_visual_config
from .ca3_realization import SourceRealization
from .circuit_campaign import SCENARIOS, campaign_config, episode_identities
from .provenance import freeze_json, sha256_file


PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET = 100000000


def resolve_episode(campaign: dict, scenario: str, identity: dict) -> dict:
    protocol = campaign["scenarios"][scenario]["task_protocol"]
    condition = condition_for_episode(protocol, scenario, identity)
    result = dict(identity=identity, condition=condition, dt_ms=protocol["dt_ms"],
                  duration_ticks=protocol["condition_duration_ticks"][identity["condition_index"]],
                  production_eligible=False, complete_episodes=0)
    if scenario == "v1_visual":
        result["builder_seeds"] = dict(builder_seed=identity["connectivity_seed"],
                                       glif_assignment_seed=identity["connectivity_seed"] + 1000000)
        result["initial_state"] = "source_reset_not_initial_state_seed_draw"
        result["visual_noise_seeds"] = dict(lgn=identity["stimulus_seed"], background=identity["background_seed"])
    elif scenario == "pfc_working_memory":
        result["source_seeds"] = dict(initial_rng_seed=identity["initial_state_seed"],
                                       external_noise_seed=identity["background_seed"])
        result["initial_state"] = protocol["initial_state_policy"]
        result["random_stream_policy"] = protocol["random_stream_policy"]
        result["circuit_parameter_domain_bound"] = False
        result["unused_identity_seeds"] = ["connectivity_seed", "stimulus_seed"]
    elif scenario == "ca3_pattern_completion":
        result["formation_seed"] = identity["connectivity_seed"]
        result["retrieval_seeds"] = dict(uniform=identity["background_seed"],
                                         engines=identity["initial_state_seed"], cue_membership=identity["stimulus_seed"])
        result["initial_state"] = "source_reset_not_initial_state_seed_draw"
        result["engine_seed_semantics"] = "common_seed_for_nine_persistent_source_distribution_engines"
    return result


def prepare(config_path: Path, output: Path, scenarios: tuple[str, ...] = ("v1_visual", "ca3_pattern_completion")) -> dict:
    """Write executable candidate identities; these cannot release production."""
    if output.exists():
        raise FileExistsError("BENCHMARK_SAMPLING_OUTPUT_EXISTS")
    campaign = campaign_config(config_path)
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(name not in SCENARIOS for name in scenarios):
        raise ValueError("BENCHMARK_SAMPLING_SCENARIOS_INVALID")
    output.mkdir(parents=True)
    counts = {}
    for scenario in scenarios:
        path = output / (scenario + ".csv")
        count = 0
        with path.open("x", newline="") as stream:
            writer = None
            for identity in episode_identities(campaign, scenario):
                resolved = resolve_episode(campaign, scenario, identity)
                row = dict(identity, condition_id=resolved["condition"]["condition_id"],
                           dt_ms=resolved["dt_ms"], duration_ticks=resolved["duration_ticks"])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                count += 1
        counts[scenario] = count
        freeze_json(output / (scenario + "_example.json"),
                    resolve_episode(campaign, scenario, next(episode_identities(campaign, scenario))))
    report = dict(status="CANDIDATE_SAMPLING_PREPARED", candidate_rows=counts,
                  sampling=campaign["sampling"], split_counts=campaign["circuit_split_counts"],
                  task_protocols={name: campaign["scenarios"][name]["task_protocol"] for name in counts},
                  complete_episodes=0, production_eligible=False,
                  initial_state_seed_semantics={name: resolve_episode(
                      campaign, name, next(episode_identities(campaign, name)))["initial_state"] for name in counts})
    freeze_json(output / "report.json", report)
    return report


def prepare_circuit(config_path: Path, base_path: Path, scenario: str,
                    circuit_index: int, output: Path) -> dict:
    """Bind one prospective circuit to the existing source construction path.

    Before production qualification, this construction retains development
    scope. Its intended split is recorded separately from its executed scope.
    """
    campaign = campaign_config(config_path)
    if scenario not in ("v1_visual", "ca3_pattern_completion"):
        raise ValueError("BENCHMARK_V1_OR_CA3_CIRCUIT_REQUIRED")
    if type(circuit_index) is not int or not 0 <= circuit_index < campaign["sampling"]["circuit_instances"]:
        raise ValueError("BENCHMARK_CIRCUIT_INDEX_INVALID")
    if output.exists():
        raise FileExistsError("BENCHMARK_CIRCUIT_PREPARATION_EXISTS")
    identity = next(row for row in episode_identities(campaign, scenario)
                    if row["circuit_index"] == circuit_index)
    config = yaml.safe_load(base_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("BENCHMARK_CIRCUIT_BASE_CONFIG_MAPPING_REQUIRED")
    config = copy.deepcopy(config)
    if scenario == "v1_visual":
        if (config.get("expected_full_nodes") != 230924 or config.get("full_fraction") != 1.0
                or config.get("canary_fraction") != 0.01):
            raise ValueError("BENCHMARK_V1_FULL_SOURCE_SCALE_REQUIRED")
        config.update(builder_seed=identity["connectivity_seed"],
                      glif_assignment_seed=identity["connectivity_seed"] + 1000000)
    else:
        if campaign_config(base_path) != campaign:
            raise ValueError("BENCHMARK_CA3_SAME_CAMPAIGN_CONFIG_REQUIRED")
        config["ca3_source_realization"] = SourceRealization(
            identity["circuit_id"], identity["circuit_id"] + "_formation", "formation", "development",
            identity["connectivity_seed"], identity["connectivity_seed"], 1).record()
    record = dict(status="PREPARED_BENCHMARK_CIRCUIT_CONSTRUCTION_INPUT",
                  scenario_id=scenario, circuit_id=identity["circuit_id"], circuit_index=circuit_index,
                  intended_split=identity["split"], construction_scope="development",
                  connectivity_seed=identity["connectivity_seed"],
                  complete_episodes=0, production_eligible=False)
    config["benchmark_circuit_candidate"] = record
    output.mkdir(parents=True)
    (output / "circuit.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    freeze_json(output / "report.json", record)
    return record



def prepare_visual_block(config_path: Path, base_path: Path, circuit_index: int,
                         block_index: int, output: Path) -> dict:
    """Prepare ten native visual recipes from one canonical circuit/noise identity."""
    campaign = campaign_config(config_path)
    sampling = campaign["sampling"]
    if (type(circuit_index) is not int or not 0 <= circuit_index < sampling["circuit_instances"]
            or type(block_index) is not int or not 0 <= block_index < sampling["initial_noise_blocks_per_circuit"]):
        raise ValueError("BENCHMARK_VISUAL_CIRCUIT_OR_BLOCK_INDEX_INVALID")
    if output.exists():
        raise FileExistsError("BENCHMARK_VISUAL_BLOCK_PREPARATION_EXISTS")
    base = yaml.safe_load(base_path.read_text())
    if (not isinstance(base, dict) or base.get("scope") != "development_source_runtime_qualification"
            or base.get("unit_coverage") != "all_source_units"):
        raise ValueError("BENCHMARK_FULL_SOURCE_VISUAL_BASE_REQUIRED")
    protocol = campaign["scenarios"]["v1_visual"]["task_protocol"]
    identities = [row for row in episode_identities(campaign, "v1_visual")
                  if row["circuit_index"] == circuit_index and row["block_index"] == block_index]
    if len(identities) != 10:
        raise ValueError("BENCHMARK_TEN_VISUAL_CONDITIONS_REQUIRED")
    output.mkdir(parents=True)
    conditions = []
    for identity in identities:
        resolved = resolve_episode(campaign, "v1_visual", identity)
        config = v1_visual_config(base, protocol, identity)
        name = resolved["condition"]["condition_id"] + ".yaml"
        (output / name).write_text(yaml.safe_dump(config, sort_keys=False))
        conditions.append(dict(identity=identity, condition=resolved["condition"], config_path=name))
    report = dict(status="PREPARED_V1_COMPLETE_DEVELOPMENT_BLOCK_RECIPES", circuit_index=circuit_index,
                  block_index=block_index, circuit_id=identities[0]["circuit_id"], block_id=identities[0]["block_id"],
                  intended_split=identities[0]["split"], actual_execution_scope="development",
                  conditions=conditions, complete_episodes=0, production_eligible=False,
                  noise_pairing=protocol["noise_pairing"], initial_state_policy=protocol["initial_state_policy"])
    freeze_json(output / "report.json", report)
    return report


def prepare_episode(config_path: Path, base_path: Path, scenario: str,
                    episode_index: int, output: Path) -> dict:
    """Prepare PFC controls on an existing source circuit, with development scope."""
    if scenario != "pfc_working_memory":
        raise ValueError("BENCHMARK_PFC_EPISODE_REQUIRED")
    campaign = campaign_config(config_path)
    if type(episode_index) is not int or not 0 <= episode_index < campaign["episodes_per_scenario"]:
        raise ValueError("BENCHMARK_EPISODE_INDEX_INVALID")
    if output.exists():
        raise FileExistsError("BENCHMARK_EPISODE_PREPARATION_EXISTS")
    identity = next(islice(episode_identities(campaign, scenario), episode_index, episode_index + 1))
    base = yaml.safe_load(base_path.read_text())
    if not isinstance(base, dict):
        raise ValueError("BENCHMARK_EPISODE_BASE_CONFIG_MAPPING_REQUIRED")
    config = pfc_working_memory_config(base, campaign["scenarios"][scenario]["task_protocol"], identity)
    # The PFC source dependencies are optional for the V1/CA3 preparation paths.
    from .pfc_source_execution import _case_config

    _case_config(config, identity["episode_id"])
    report = dict(status="PREPARED_PFC_DEVELOPMENT_EPISODE_INPUT",
                  **config["benchmark_episode_candidate"],
                  resolved_episode=resolve_episode(campaign, scenario, identity))
    output.mkdir(parents=True)
    (output / "episode.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    freeze_json(output / "report.json", report)
    return report


def prepare_pfc_block(config_path: Path, base_path: Path, circuit_index: int,
                      block_index: int, output: Path) -> dict:
    """Prepare all PFC conditions on one actual source coupling operator."""
    campaign = campaign_config(config_path)
    sampling = campaign["sampling"]
    if (type(circuit_index) is not int or not 0 <= circuit_index < sampling["circuit_instances"]
            or type(block_index) is not int or not 0 <= block_index < sampling["initial_noise_blocks_per_circuit"]):
        raise ValueError("BENCHMARK_PFC_CIRCUIT_OR_BLOCK_INDEX_INVALID")
    if output.exists():
        raise FileExistsError("BENCHMARK_PFC_BLOCK_PREPARATION_EXISTS")
    base = yaml.safe_load(base_path.read_text())
    if not isinstance(base, dict):
        raise ValueError("BENCHMARK_EPISODE_BASE_CONFIG_MAPPING_REQUIRED")
    from .pfc_source_execution import _case_config
    from .pfc_kernel_parameters import circuit_provenance, resolve_kernel_parameters
    from .pfc_native_drive import SOURCE_ENTRY_SHA256

    start = (circuit_index * sampling["initial_noise_blocks_per_circuit"] + block_index) * 10
    identities = list(islice(episode_identities(campaign, "pfc_working_memory"), start, start + 10))
    protocol = campaign["scenarios"]["pfc_working_memory"]["task_protocol"]
    configs, conditions = [], []
    for identity in identities:
        config = pfc_working_memory_config(base, protocol, identity)
        _, _, dimensions = _case_config(config, identity["episode_id"])
        if dimensions["coupled_core_units"] < campaign["scenarios"]["pfc_working_memory"]["minimum_core_neurons"]:
            raise ValueError("BENCHMARK_PFC_FULL_COUPLED_SOURCE_REQUIRED")
        resolved = resolve_episode(campaign, "pfc_working_memory", identity)
        name = resolved["condition"]["condition_id"] + ".yaml"
        configs.append((name, config))
        conditions.append(dict(identity=identity, condition=resolved["condition"], config_path=name))
    report = dict(status="PREPARED_PFC_COMPLETE_DEVELOPMENT_BLOCK_RECIPES",
                  circuit_index=circuit_index, block_index=block_index,
                  circuit_id=identities[0]["circuit_id"], block_id=identities[0]["block_id"],
                  intended_split=identities[0]["split"], actual_execution_scope="development",
                  conditions=conditions, resolved_kernel_parameters=resolve_kernel_parameters(base),
                  circuit_provenance=circuit_provenance(base, SOURCE_ENTRY_SHA256),
                  coupled_core_units=dimensions["coupled_core_units"],
                  duration_ticks=dimensions["duration_ticks"],
                  complete_episodes=0, production_eligible=False,
                  noise_pairing=protocol["noise_pairing"], initial_state_policy=protocol["initial_state_policy"])
    output.mkdir(parents=True)
    for name, config in configs:
        (output / name).write_text(yaml.safe_dump(config, sort_keys=False))
    freeze_json(output / "report.json", report)
    return report


def prepare_pfc_reference_size_block(config_path: Path, base_path: Path, circuit_index: int,
                                     block_index: int, output: Path) -> dict:
    """Prepare an original-size comparator without changing the100k contract."""
    from .pfc_source_execution import _case_config
    from .pfc_source_dataset import _benchmark_registration
    from .pfc_kernel_parameters import circuit_provenance, resolve_kernel_parameters, source_kernel
    from .pfc_native_drive import SOURCE_ENTRY_SHA256

    campaign = campaign_config(config_path)
    sampling = campaign["sampling"]
    if (type(circuit_index) is not int or not 0 <= circuit_index < sampling["circuit_instances"]
            or type(block_index) is not int or not 0 <= block_index < sampling["initial_noise_blocks_per_circuit"]):
        raise ValueError("PFC_REFERENCE_SIZE_CIRCUIT_OR_BLOCK_INDEX_INVALID")
    protocol = campaign["scenarios"]["pfc_working_memory"]["task_protocol"]
    if protocol.get("dt_ms") != .0125 or protocol.get("duration_ms") != 5000.:
        raise ValueError("PFC_REFERENCE_SIZE_MATCHED_REFERENCE_TASK_REQUIRED")
    base = yaml.safe_load(base_path.read_text())
    candidate = base.get("benchmark_circuit_candidate", {}) if isinstance(base, dict) else {}
    if (not isinstance(base, dict) or base.get("scope") != "source_reproduction"
            or base.get("excitatory_neurons") != 80000 or base.get("inhibitory_neurons") != 20000
            or base.get("weight_scaling_factor") != .0256 or base.get("dt_ms") != .0125
            or base.get("duration_ms") != 5000. or not isinstance(candidate, dict)
            or candidate.get("status") != "PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE"
            or candidate.get("construction_scope") != "development"
            or candidate.get("production_eligible") is not False
            or "benchmark_task_protocol" in base or "benchmark_episode_candidate" in base):
        raise ValueError("PFC_REFERENCE_SIZE_EXISTING100K_DEVELOPMENT_CIRCUIT_REQUIRED")
    if output.exists():
        raise FileExistsError("PFC_REFERENCE_SIZE_BLOCK_PREPARATION_EXISTS")
    start = (circuit_index * sampling["initial_noise_blocks_per_circuit"] + block_index) * 10
    canonical = list(islice(episode_identities(campaign, "pfc_working_memory"), start, start + 10))
    # Reuse source compiler validation of the original candidate's parameter,
    # alias, split and provenance. No100k kernel is generated or simulated.
    original = pfc_working_memory_config(base, protocol, canonical[0])
    _, _, original_dimensions = _case_config(original, canonical[0]["episode_id"])
    comparator = copy.deepcopy(base)
    comparator.update(excitatory_neurons=2048, inhibitory_neurons=512, weight_scaling_factor=1.0,
        purpose="original-size2560 matched-clock source comparator; not a100k campaign episode or production",
        normalization_semantics="N_E * factor =2048 and N_I * factor =512; original inhibitory self exclusion retained")
    provenance = circuit_provenance(comparator, SOURCE_ENTRY_SHA256)
    parameters = resolve_kernel_parameters(comparator)
    comparator["benchmark_circuit_candidate"] = dict(
        status="PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE", scenario_id="pfc_working_memory",
        circuit_id=canonical[0]["circuit_id"], circuit_index=circuit_index,
        intended_split=canonical[0]["split"], parameters=parameters,
        construction_scope="development", production_eligible=False, complete_episodes=0,
        effective_kernel_sha256=hashlib.sha256(source_kernel(2048, comparator).tobytes()).hexdigest(),
        **provenance)
    declaration = dict(schema="pfc_source_size_comparison_input_v1", scope="development_size_reference",
        comparator_role="original_size2560", coupled_core_units=2560, target_coupled_core_units=100000,
        original_circuit_config=str(base_path.resolve()), original_circuit_config_sha256=sha256_file(base_path),
        original_circuit_candidate=copy.deepcopy(candidate), original_dimensions=original_dimensions,
        comparator_namespace=dict(offset=PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET,
            rule="canonical circuit/block stream identity plus comparator offset; paired conditions reuse one history",
            independence="distinct stream labels; statistical independence requires the frozen source-law design"),
        episode_alias_reason="same canonical task/circuit/block/condition alias at a separately declared source size and output root; exclude from100k episode totals",
        cross_size_noise_pairing=False, source_quality_qualified=False, scientific_claim_qualified=False,
        campaign_episodes=0, production_episodes=0, production_eligible=False)
    configs, conditions = [], []
    for canonical_identity in canonical:
        identity = copy.deepcopy(canonical_identity)
        for name in ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed"):
            identity[name] += PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET
        config = pfc_working_memory_config(comparator, protocol, identity)
        config["pfc_source_size_comparison"] = dict(declaration, canonical_campaign_identity=canonical_identity)
        _, _, dimensions = _case_config(config, identity["episode_id"])
        _benchmark_registration(config)
        if dimensions["coupled_core_units"] != 2560 or dimensions["duration_ticks"] != 400000:
            raise ValueError("PFC_REFERENCE_SIZE_COMPLETE_COMPARATOR_REQUIRED")
        condition = condition_for_episode(protocol, "pfc_working_memory", identity)
        name = condition["condition_id"] + ".yaml"
        configs.append((name, config))
        conditions.append(dict(identity=identity, canonical_campaign_identity=canonical_identity,
                               condition=condition, config_path=name))
    report = dict(declaration, status="PREPARED_PFC_REFERENCE_SIZE_DEVELOPMENT_BLOCK_RECIPES",
        circuit_index=circuit_index, block_index=block_index, circuit_id=canonical[0]["circuit_id"],
        block_id=canonical[0]["block_id"], intended_split=canonical[0]["split"], conditions=conditions,
        resolved_kernel_parameters=parameters, circuit_provenance=provenance,
        duration_ticks=400000, dt_ms=.0125, complete_episodes=0, new_source_simulations=0)
    output.mkdir(parents=True)
    for name, config in configs:
        (output / name).write_text(yaml.safe_dump(config, sort_keys=False))
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--circuit-scenario", choices=("v1_visual", "ca3_pattern_completion"))
    parser.add_argument("--scenario", choices=SCENARIOS, help="Prepare candidate identities for one scenario")
    parser.add_argument("--episode-scenario", choices=("pfc_working_memory",))
    parser.add_argument("--block-scenario", choices=("v1_visual", "pfc_working_memory"))
    parser.add_argument("--reference-size-block", action="store_true",
                        help="Prepare a2560-unit PFC development comparator, not campaign episodes")
    parser.add_argument("--block-index", type=int)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--circuit-index", type=int)
    parser.add_argument("--base-config", type=Path)
    args = parser.parse_args()
    if sum(value is not None for value in (args.circuit_scenario, args.episode_scenario, args.block_scenario, args.scenario)) + int(args.reference_size_block) > 1:
        parser.error("select only one scenario mode")
    if args.reference_size_block:
        if args.circuit_index is None or args.block_index is None or args.base_config is None or args.episode_index is not None:
            parser.error("--reference-size-block requires --circuit-index, --block-index and --base-config")
        prepare_pfc_reference_size_block(args.config, args.base_config, args.circuit_index, args.block_index, args.output)
    elif args.block_scenario:
        if args.circuit_index is None or args.block_index is None or args.base_config is None or args.episode_index is not None:
            parser.error("--block-scenario requires --circuit-index, --block-index and --base-config")
        prepare_block = prepare_pfc_block if args.block_scenario == "pfc_working_memory" else prepare_visual_block
        prepare_block(args.config, args.base_config, args.circuit_index, args.block_index, args.output)
    elif args.block_index is not None:
        parser.error("--block-index requires --block-scenario")
    elif args.episode_scenario:
        if args.episode_index is None or args.base_config is None or args.circuit_index is not None:
            parser.error("--episode-scenario requires --episode-index and --base-config")
        prepare_episode(args.config, args.base_config, args.episode_scenario, args.episode_index, args.output)
    elif args.episode_index is not None:
        parser.error("--episode-index requires --episode-scenario")
    elif args.circuit_scenario:
        if args.circuit_index is None or args.base_config is None:
            parser.error("--circuit-scenario requires --circuit-index and --base-config")
        prepare_circuit(args.config, args.base_config, args.circuit_scenario, args.circuit_index, args.output)
    elif args.circuit_index is not None or args.base_config is not None:
        parser.error("--circuit-index and --base-config require --circuit-scenario")
    else:
        prepare(args.config, args.output, (args.scenario,) if args.scenario else ("v1_visual", "ca3_pattern_completion"))


if __name__ == "__main__":
    main()

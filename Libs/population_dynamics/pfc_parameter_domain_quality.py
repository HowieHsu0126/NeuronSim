"""Prepare full-domain PFC source task coverage with no qualification decision.

Three joint initial/external histories describe task regimes. These histories do
not supply a confidence interval or replace the100000-episode benchmark.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

import yaml

from .benchmark_sampling import prepare_pfc_block
from .circuit_campaign import campaign_config
from .pfc_kernel_parameters import resolve_kernel_parameters, circuit_provenance
from .pfc_native_drive import SOURCE_ENTRY_SHA256
from .pfc_observation_resolution import event_summary, control_boundaries, CHUNK_EVENTS
from .pfc_reference_quality import readout_episode
from .pfc_source_execution import _case_config, _qualification
from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_parameter_domain_quality_v1"
SCENARIO = "pfc_working_memory"
BLOCKS = (6, 7, 8)
EPISODES = 3000
SPLITS = dict(train=70, validation=10, calibration=10, test=10)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def intended_split(index: int) -> str:
    return "train" if index < 70 else "validation" if index < 80 else "calibration" if index < 90 else "test"


def validate_source_config(config: dict, identity: dict) -> dict:
    require(config.get("scope") == "source_reproduction" and config.get("dt_ms") == .0125
            and config.get("duration_ms") == 5000. and config.get("excitatory_neurons") == 80000
            and config.get("inhibitory_neurons") == 20000 and config.get("readout_bin_ms") == 5.
            and config.get("random_stream_policy") == "independent_initial_external",
            "PFC_DOMAIN_COMPLETE100K_REFERENCE_SOURCE_REQUIRED")
    candidate = config.get("benchmark_episode_candidate", {})
    require(candidate.get("identity") == identity and candidate.get("production_eligible") is False
            and candidate.get("execution_scope") == "development"
            and candidate.get("circuit_parameter_domain_bound") is False,
            "PFC_DOMAIN_DEVELOPMENT_IDENTITY_REQUIRED")
    _, _, dimensions = _case_config(config, identity["episode_id"])
    require(dimensions["coupled_core_units"] == 100000 and dimensions["duration_ticks"] == 400000,
            "PFC_DOMAIN_COMPLETE_COUPLED_EPISODE_REQUIRED")
    return dimensions


def validate_plan(plan: dict) -> None:
    require(plan.get("schema") == SCHEMA and plan.get("scope") == "development"
            and plan.get("block_indices") == list(BLOCKS) and plan.get("planned_complete_episodes") == EPISODES
            and plan.get("complete_episodes") == 0 and plan.get("production_episodes") == 0
            and plan.get("source_quality_qualified") is False and plan.get("task_sampling_qualified") is False
            and plan.get("scientific_claim_qualified") is False and plan.get("production_eligible") is False
            and plan.get("circuit_split_counts") == SPLITS,
            "PFC_DOMAIN_PLAN_SCOPE_OR_COUNTS_CHANGED")
    rows = plan.get("rows", [])
    require(len(rows) == EPISODES, "PFC_DOMAIN_COMPLETE3000_RECIPES_REQUIRED")
    names = set()
    for index, (row, expected) in enumerate(zip(rows, product(range(100), BLOCKS, range(10)))):
        identity = row["identity"]
        c, b, k = expected
        circuit = f"pfc_working_memory_circuit_{c:03d}"
        block = circuit + f"_block_{b:03d}"
        name = block + f"_condition_{k:02d}"
        require(row.get("task_index") == index and tuple(identity.get(key) for key in
                ("circuit_index", "block_index", "condition_index")) == expected
                and identity.get("circuit_id") == circuit and identity.get("block_id") == block
                and identity.get("episode_id") == name and identity.get("split") == intended_split(c)
                and name not in names, "PFC_DOMAIN_DUPLICATE_OR_CHANGED_EPISODE_IDENTITY")
        require(identity.get("initial_state_seed") == 210000000 + c*1000+b
                and identity.get("background_seed") == 220000000 + c*1000+b
                and identity.get("connectivity_seed") == 200000000+c
                and identity.get("stimulus_seed") == 230000000+c*1000+b,
                "PFC_DOMAIN_DATASET_STREAM_IDENTITY_CHANGED")
        names.add(name)


def prepare(campaign_path: Path, parameter_root: Path, blocks: tuple[int, ...], output: Path) -> dict:
    require(blocks == BLOCKS, "PFC_DOMAIN_DECLARED_BLOCKS6_7_8_REQUIRED")
    campaign = campaign_config(campaign_path)
    require(campaign["sampling"] == dict(circuit_instances=100, initial_noise_blocks_per_circuit=100,
                conditions_per_block=10) and campaign["circuit_split_counts"] == SPLITS,
            "PFC_DOMAIN_CANONICAL100_CIRCUIT_ENVELOPE_REQUIRED")
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    require(protocol.get("frozen") is True and protocol.get("dt_ms") == .0125
            and protocol.get("duration_ms") == 5000., "PFC_DOMAIN_FROZEN_REFERENCE_TASK_REQUIRED")
    require(not output.exists(), "PFC_DOMAIN_PREPARATION_EXISTS")
    parameters, seen = [], set()
    for circuit in range(100):
        path = parameter_root / f"pfc_working_memory_circuit_{circuit:03d}/circuit.yaml"
        source = yaml.safe_load(path.read_text())
        candidate = source.get("benchmark_circuit_candidate", {})
        values = resolve_kernel_parameters(source)
        key = tuple(sorted(values.items()))
        require(key not in seen and candidate.get("circuit_index") == circuit
                and candidate.get("intended_split") == intended_split(circuit)
                and candidate.get("parameters") == values
                and candidate.get("production_eligible") is False,
                "PFC_DOMAIN_EXISTING_DISTINCT_PARAMETER_CANDIDATES_REQUIRED")
        require(source.get("excitatory_neurons") == 80000 and source.get("inhibitory_neurons") == 20000
                and source.get("dt_ms") == .0125 and source.get("duration_ms") == 5000.
                and source.get("readout_bin_ms") == 5., "PFC_DOMAIN_COMPLETE100K_REFERENCE_SOURCE_REQUIRED")
        provenance = circuit_provenance(source, SOURCE_ENTRY_SHA256)
        require(all(candidate.get(k) == v for k,v in provenance.items()),
                "PFC_DOMAIN_EXISTING_PARAMETER_PROVENANCE_CHANGED")
        seen.add(key)
        parameters.append(dict(circuit_index=circuit, path=str(path.resolve()), sha256=sha256_file(path),
                               parameters=values, **provenance))
    output.mkdir(parents=True)
    rows, block_reports = [], []
    for parameter in parameters:
        circuit = parameter["circuit_index"]
        for block in blocks:
            directory = output / "recipes" / f"circuit_{circuit:03d}" / f"block_{block:03d}"
            report = prepare_pfc_block(campaign_path, Path(parameter["path"]), circuit, block, directory)
            for row in report["conditions"]:
                path = directory / row["config_path"]
                validate_source_config(yaml.safe_load(path.read_text()), row["identity"])
                rows.append(dict(task_index=len(rows), identity=row["identity"], condition=row["condition"],
                    config_path=str(path.relative_to(output)), config_sha256=sha256_file(path)))
            block_reports.append(str((directory / "report.json").relative_to(output)))
    plan = dict(schema=SCHEMA, scope="development", rows=rows, parameters=parameters,
        block_reports=block_reports, block_indices=list(blocks), planned_complete_episodes=EPISODES,
        campaign_path=str(campaign_path.resolve()), campaign_sha256=sha256_file(campaign_path),
        parameter_root=str(parameter_root.resolve()), circuit_split_counts=SPLITS, task_protocol=protocol,
        raw_dt_ms=.0125, duration_ms=5000., coupled_core_units=100000,
        descriptive_readout_bin_ms=5., formal_observation_resolution_replaced=False,
        history_semantics="three joint initial/external histories nested in each circuit; paired ten conditions",
        uncertainty_status="UNQUALIFIED_TASK_SAMPLING_PENDING_FORMAL_UNCERTAINTY_PROTOCOL",
        scientific_scope="pipeline and full-parameter task-regime coverage; no CI or source/task/domain PASS",
        complete_episodes=0, production_episodes=0, production_eligible=False,
        source_quality_qualified=False, task_sampling_qualified=False, scientific_claim_qualified=False)
    validate_plan(plan)
    freeze_json(output / "execution.json", plan)
    return plan


def load_task(preparation: Path, index: int) -> tuple[dict, dict, Path]:
    plan = json.loads((preparation / "execution.json").read_text())
    validate_plan(plan)
    require(type(index) is int and 0 <= index < EPISODES, "PFC_DOMAIN_TASK_INDEX_INVALID")
    row = plan["rows"][index]
    relative = Path(row["config_path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "PFC_DOMAIN_RECIPE_PATH_INVALID")
    path = preparation / relative
    require(sha256_file(path) == row["config_sha256"], "PFC_DOMAIN_RECIPE_CHANGED")
    validate_source_config(yaml.safe_load(path.read_text()), row["identity"])
    return plan, row, path


def qualified_mapping(path: Path) -> tuple[dict, dict]:
    # Resolve the missing prerequisite before reading recipes or starting outputs.
    require(path.is_file(), "PFC_DOMAIN_COMPLETE100_PROOF_MAPPING_REQUIRED")
    mapping = json.loads(path.read_text())
    require(isinstance(mapping, dict) and set(mapping) == {str(c) for c in range(100)}
            and all(isinstance(v, str) and Path(v).is_absolute() for v in mapping.values()),
            "PFC_DOMAIN_COMPLETE100_PROOF_MAPPING_REQUIRED")
    report = json.loads((path.parent / "report.json").read_text())
    require(report.get("status") == "COMPLETE100_EXACT_TUPLE_REFERENCE_KERNEL_BINDINGS"
            and report.get("complete_circuits") == 100 and report.get("qualifications") == mapping
            and len(report.get("bindings", [])) == 100 and report.get("source_quality_qualified") is False
            and report.get("scientific_parameter_domain_qualified") is False
            and report.get("production_episodes") == 0, "PFC_DOMAIN_COMPLETE100_PROOF_REPORT_REQUIRED")
    require({r.get("circuit_index") for r in report["bindings"]} == set(range(100)),
            "PFC_DOMAIN_COMPLETE100_PROOF_REPORT_REQUIRED")
    return mapping, report


def source_task(project: Path, preparation: Path, index: int, proofs: Path, output: Path) -> int:
    mapping, report = qualified_mapping(proofs)
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURMD_NODENAME") in
            ("cpu03", "cpu04", "cpu05", "cpu06") and not os.environ.get("SLURM_JOB_GPUS")
            and not os.environ.get("SLURM_STEP_GPUS"), "PFC_DOMAIN_ALLOCATED_CPU_REQUIRED")
    _, row, config_path = load_task(preparation, index)
    config = yaml.safe_load(config_path.read_text())
    circuit = row["identity"]["circuit_index"]
    directory = Path(mapping[str(circuit)])
    binding = next(r for r in report["bindings"] if r["circuit_index"] == circuit)
    require(binding.get("qualification") == str(directory) and binding.get("parameters") ==
            resolve_kernel_parameters(config), "PFC_DOMAIN_EXACT_TUPLE_PROOF_CHANGED")
    verified = _qualification(directory, "numpy", config["dt_ms"], config)
    require(verified["receipt_sha256"] == binding.get("receipt_sha256")
            and verified["delivery_sha256"] == binding.get("delivery_sha256"),
            "PFC_DOMAIN_EXACT_TUPLE_PROOF_CHANGED")
    environment = dict(os.environ, NEURONSIM_PROJECT_ROOT=str(project.resolve()),
        NEURONSIM_PFC_SOURCE_CONFIG=str(config_path.resolve()),
        NEURONSIM_PFC_SOURCE_QUALIFICATION=str(directory.resolve()),
        NEURONSIM_PFC_SOURCE_CASE_ID=row["identity"]["episode_id"], NEURONSIM_PFC_SOURCE_BACKEND="numpy",
        NEURONSIM_PFC_OUTPUT_BASE=str(output.resolve()), NEURONSIM_PFC_SCRATCH="/tmp",
        CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.pop("SLURM_ARRAY_TASK_ID", None)
    environment.pop("NEURONSIM_PFC_SOURCE_CASES_JSON", None)
    command = [os.environ.get("NEURONSIM_PFC_PYTHON", os.sys.executable), "-B",
               str(project / "Libs/scripts/run_pfc_source_cases.py")]
    return subprocess.run(command, cwd="/tmp", env=environment).returncode


def frozen_observation_map(campaign_path: Path) -> dict:
    campaign = campaign_config(campaign_path)
    observation = campaign["scenarios"][SCENARIO]["observation_map"]
    temporal = observation.get("temporal_resolution", {})
    require(observation.get("frozen") is True
            and observation.get("unit_order") == "all_E_then_all_I_ascending_source_indices"
            and observation.get("populations") == ["E", "I"]
            and observation.get("excitatory_preference_angle_deg") == "360 * source_index / N_E"
            and observation.get("inhibitory_preference_angle_deg") is None
            and observation.get("population_fields") == "16_equal_E_preference_bins_plus_complete_I_population"
            and temporal.get("scientific_bin_ticks") == 80 and temporal.get("scientific_bin_ms") == 1.
            and temporal.get("event_tick_ms") == .0125,
            "PFC_DOMAIN_FROZEN1MS_OBSERVATION_MAP_REQUIRED")
    return observation


def observation_readout(directory: Path, config_path: Path, row: dict, campaign_path: Path,
                        output: Path) -> dict:
    """Read authenticated development events at the frozen map and1ms bin.

    The existing5ms export supplies source authentication. The event kernel
    reuses integer ticks and avoids a5000-by100000 dense count matrix.
    """
    observation = frozen_observation_map(campaign_path)
    require(not output.exists(), "PFC_DOMAIN_OBSERVATION_OUTPUT_EXISTS")
    report = json.loads((directory / "report.json").read_text())
    require(report.get("status") == "PASS_REFERENCE_SOURCE_QUALITY_READOUT"
            and report.get("identity") == row["identity"] and report.get("config_sha256") == sha256_file(config_path)
            and report.get("core_units") == 100000 and report.get("duration_ticks") == 400000
            and report.get("dt_ms") == .0125 and report.get("production_episodes") == 0
            and report.get("source_qualification_granted") is False,
            "PFC_DOMAIN_OBSERVATION_COMPLETE_READOUT_REQUIRED")
    source_archive = Path(report["source_archive"])
    source_delivery_path = source_archive / "delivery.json"
    require(sha256_file(source_delivery_path) == report["source_delivery_sha256"],
            "PFC_DOMAIN_OBSERVATION_SOURCE_DELIVERY_CHANGED")
    source_delivery = json.loads(source_delivery_path.read_text())
    native_entries = [item for item in source_delivery["files"] if item["path"] == "source_run/report.json"]
    require(len(native_entries) == 1, "PFC_DOMAIN_OBSERVATION_SOURCE_REPORT_BINDING_REQUIRED")
    source_report_path = source_archive / "source_run/report.json"
    source_report_sha256 = sha256_file(source_report_path)
    require(source_report_sha256 == native_entries[0]["sha256"]
            and report.get("source_report_sha256", source_report_sha256) == source_report_sha256,
            "PFC_DOMAIN_OBSERVATION_SOURCE_REPORT_CHANGED")
    native = json.loads(source_report_path.read_text())
    require(native.get("status") == "PFC_SOURCE_EXECUTION_COMPLETE"
            and native.get("source_execution_complete") is True
            and native.get("case_id") == row["identity"]["episode_id"]
            and native.get("config_sha256") == report["config_sha256"]
            and native.get("production_episodes") == 0,
            "PFC_DOMAIN_OBSERVATION_COMPLETE_SOURCE_REQUIRED")
    root = directory / "export/dataset"
    catalog_path = root / "dataset.json"
    require(sha256_file(catalog_path) == report["output_dataset_catalog_sha256"],
            "PFC_DOMAIN_OBSERVATION_CATALOG_CHANGED")
    catalog = json.loads(catalog_path.read_text())
    require(catalog.get("complete_episodes") == 1 and len(catalog.get("shards", [])) == 1
            and catalog["binding"]["config_sha256"] == report["config_sha256"],
            "PFC_DOMAIN_OBSERVATION_EPISODE_DENOMINATOR_CHANGED")
    shard = catalog["shards"][0]
    require(len(shard["episodes"]) == 1 and shard["episodes"][0]["episode_id"] == row["identity"]["episode_id"],
            "PFC_DOMAIN_OBSERVATION_EPISODE_IDENTITY_CHANGED")
    relative = Path(shard["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "PFC_DOMAIN_OBSERVATION_PATH_INVALID")
    path = root / relative
    require(sha256_file(path) == shard["sha256"], "PFC_DOMAIN_OBSERVATION_EVENT_SHARD_CHANGED")
    circuit = catalog["circuits"][shard["episodes"][0]["circuit_id"]]
    unit_path = root / circuit["path"]
    require(sha256_file(unit_path) == circuit["sha256"] and circuit["units"] == 100000,
            "PFC_DOMAIN_OBSERVATION_UNIT_MAP_CHANGED")
    with h5py.File(unit_path, "r") as units:
        expected_fields = np.r_[np.arange(80000) * 16 // 80000, np.full(20000,16)]
        require(np.array_equal(units["unit_ids"][:], np.arange(100000))
                and np.array_equal(units["field_index"][:], expected_fields)
                and np.array_equal(units["populations"].asstr()[:], np.r_[np.full(80000,"E"),np.full(20000,"I")]),
                "PFC_DOMAIN_OBSERVATION_SOURCE_UNIT_MAP_INVALID")
    with h5py.File(path, "r") as stored:
        episode = stored[row["identity"]["episode_id"]]
        metadata = json.loads(episode.attrs["metadata_json"])
        require(metadata["dt_ms"] == .0125 and metadata["duration_ticks"] == 400000
                and metadata["episode_id"] == row["identity"]["episode_id"]
                and metadata["seeds"] == dict(initial_rng_seed=row["identity"]["initial_state_seed"],
                    external_noise_seed=row["identity"]["background_seed"])
                and episode["spike_ticks"].shape == episode["spike_unit_indices"].shape == (report["events"],),
                "PFC_DOMAIN_OBSERVATION_SOURCE_EPISODE_CHANGED")
        def chunks():
            for start in range(0, report["events"], CHUNK_EVENTS):
                stop = min(start + CHUNK_EVENTS, report["events"])
                yield episode["spike_ticks"][start:stop], episode["spike_unit_indices"][start:stop]
        values = event_summary(chunks(), 100000)
        transitions = control_boundaries(episode["control_ticks"][:], episode["control_values"][:],80,400000)
    require(values["retained_events"] + values["terminal_events"] == report["events"]
            and values["terminal_events"] == report["terminal_tick_events"]
            and all(row["boundary_aligned"] for row in transitions),
            "PFC_DOMAIN_OBSERVATION_EVENT_OR_CONTROL_CONSERVATION_FAILED")
    old_path = directory / "unit_population_statistics.npz"
    require(sha256_file(old_path) == report["outputs"][old_path.name]["sha256"],
            "PFC_DOMAIN_OBSERVATION5MS_OUTPUT_CHANGED")
    with np.load(old_path, allow_pickle=False) as old:
        require(np.array_equal(old["population_field_counts"], values["field_counts"][400])
                and np.array_equal(old["phase_unit_counts"].sum(axis=0), values["unit_counts"]),
                "PFC_DOMAIN_OBSERVATION5MS_COUNTS_DIFFER")
    output.mkdir(parents=True)
    arrays = dict(population_field_counts=values["field_counts"][80], vector_x=values["vector_x"][80],
        vector_y=values["vector_y"][80], unit_counts=values["unit_counts"],
        late_unit_counts=values["late_unit_counts"], terminal_unit_counts=values["terminal_unit_counts"])
    np.savez(output / "observation.npz", **arrays)
    result = dict(status="DESCRIBED_FROZEN1MS_SOURCE_OBSERVATION", identity=row["identity"],
        condition=row["condition"], source_archive=report["source_archive"],
        source_delivery_sha256=report["source_delivery_sha256"], source_report_sha256=source_report_sha256,
        config_sha256=report["config_sha256"], development_readout_report_sha256=sha256_file(directory / "report.json"),
        source_dataset_catalog_sha256=report["output_dataset_catalog_sha256"], observation_map=observation,
        observation_campaign_sha256=sha256_file(campaign_path), bin_ticks=80, bin_ms=1.,
        native_event_tick_ms=.0125, native_event_ticks_preserved=True, core_units=100000,
        bins=5000, population_fields=17, retained_events=values["retained_events"],
        terminal_events=values["terminal_events"], all_unit_integer_reaggregation_exact=True,
        existing5ms_counts_exact=True, source_control_transitions=transitions,
        covariance_sample_axis="within_episode_time; between_history_uncertainty_not_estimated",
        direction_scientific_eligibility_qualified=False, scientific_accuracy_qualified=False,
        source_qualification_granted=False, scientific_claim_qualified=False, production_episodes=0,
        output_sha256=sha256_file(output / "observation.npz"))
    freeze_json(output / "report.json", result)
    return result


def readout_task(preparation: Path, index: int, source: Path, output: Path, *,
                 observation_bin_ms: float = 5., campaign_path: Path | None = None) -> dict:
    _, row, path = load_task(preparation,index)
    require(observation_bin_ms in (1.,5.), "PFC_DOMAIN_SUPPORTED_OBSERVATION_BIN_REQUIRED")
    if observation_bin_ms == 5.:
        return readout_episode(path,source,output)
    require(campaign_path is not None, "PFC_DOMAIN_OBSERVATION_CAMPAIGN_REQUIRED")
    frozen_observation_map(campaign_path)
    # Both views retain independent output identities. Source recipes stay5ms.
    descriptive = output / "development5ms"
    if not descriptive.exists():
        readout_episode(path,source,descriptive)
    # A completed5ms export survives a later1ms failure. The observation
    # consumer verifies its recipe, source, catalogue, shard and old counts.
    return observation_readout(descriptive,path,row,campaign_path,output / "observation1ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign", type=Path, required=True)
    prep.add_argument("--parameter-root", type=Path, required=True)
    prep.add_argument("--blocks", default="6,7,8")
    prep.add_argument("--output", type=Path, required=True)
    source = sub.add_parser("source")
    source.add_argument("--project", type=Path, required=True)
    source.add_argument("--prepared", type=Path, required=True)
    source.add_argument("--index", type=int, required=True)
    source.add_argument("--qualifications", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    read = sub.add_parser("readout")
    read.add_argument("--prepared", type=Path, required=True)
    read.add_argument("--index", type=int, required=True)
    read.add_argument("--source", type=Path, required=True)
    read.add_argument("--output", type=Path, required=True)
    read.add_argument("--observation-bin-ms", type=float, choices=(1.,5.), default=5.)
    read.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.campaign, args.parameter_root, tuple(map(int,args.blocks.split(","))), args.output)
        print(json.dumps(dict(status="PREPARED3000_FULL_DOMAIN_SOURCE_RECIPES", rows=len(plan["rows"]),
                             complete_episodes=0, production_episodes=0, source_quality_qualified=False)))
    elif args.command == "source":
        return source_task(args.project,args.prepared,args.index,args.qualifications,args.output)
    else:
        print(json.dumps(readout_task(args.prepared,args.index,args.source,args.output,
            observation_bin_ms=args.observation_bin_ms,campaign_path=args.campaign),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

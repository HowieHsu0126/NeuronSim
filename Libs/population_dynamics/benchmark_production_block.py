"""Process one complete V1/CA3 production block through existing episode APIs.

V1 consumes completed native sources. CA3 uses the established native executor.
This module neither qualifies a source nor submits jobs or relabels development
circuits. All ten canonical conditions are required before any processing.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

from . import benchmark_production as production
from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import bound_json
from .v1_glif_production import prepared_source_block

BLOCK_SCHEMA = "neural_benchmark_production_block_v1"
V1_SOURCE_FIELDS = ("rate_source", "spike_source", "native_source", "authority_source")
GLIF_SOURCE_FIELDS = ("rate_source", "spike_source", "circuit_reference", "circuit_validation",
                      "gpu_primary", "gpu_replay", "authority_source")
GLIF_INPUT_SCHEMA = "neural_v1_glif_source_inputs_v1"
CA3_SOURCE_FIELDS = ("prepared_config", "packaged", "formation", "analysis", "current_replay_engine")


@dataclass(frozen=True)
class BlockEntry:
    identity: dict
    sources: dict[str, Path]
    backend: str = "native"


def load_block(cohort: production.ProductionCohort, path: Path,
               circuit_index: int, block_index: int) -> list[BlockEntry]:
    """Resolve a complete ordered block without creating output or executing."""
    cohort.require_current()
    production.require(cohort.scenario in production.LEGACY_SCENARIOS,
                       "BENCHMARK_BLOCK_V1_OR_CA3_REQUIRED")
    document = json.loads(path.read_text())
    production.require(document.get("schema") == BLOCK_SCHEMA
                       and document.get("scenario_id") == cohort.scenario
                       and type(document.get("circuit_index")) is int
                       and type(document.get("block_index")) is int
                       and document.get("circuit_index") == circuit_index
                       and document.get("block_index") == block_index,
                       "BENCHMARK_BLOCK_IDENTITY_DIFFERS")
    records = document.get("conditions")
    production.require(isinstance(records, list) and len(records) == 10,
                       "BENCHMARK_BLOCK_TEN_CONDITIONS_REQUIRED")
    backend = document.get("execution_backend", "native")
    production.require(backend in ("native", "glif_cuda") and
                       (cohort.scenario == "v1_visual" or backend == "native"),
                       "BENCHMARK_BLOCK_BACKEND_UNSUPPORTED")
    fields = (GLIF_SOURCE_FIELDS if backend == "glif_cuda" else V1_SOURCE_FIELDS) if cohort.scenario == "v1_visual" else CA3_SOURCE_FIELDS
    entries = []
    for index, record in enumerate(records):
        production.require(isinstance(record, dict)
                           and type(record.get("condition_index")) is int
                           and record["condition_index"] == index,
                           "BENCHMARK_BLOCK_CONDITION_ORDER_DIFFERS")
        sources = record.get("sources")
        production.require(isinstance(sources, dict) and set(sources) == set(fields)
                           and all(isinstance(value, str) and value.strip()
                                   for value in sources.values()),
                           "BENCHMARK_BLOCK_SOURCE_PATHS_REQUIRED")
        resolved = {key: (path.parent / value).resolve() for key, value in sources.items()}
        entries.append(BlockEntry(cohort.identity(circuit_index, block_index, index), resolved, backend))
    if cohort.scenario == "v1_visual":
        # Reused rates/coordinate authority are valid. Native executions and
        # sampled input archives belong to individual complete conditions.
        for field in (("spike_source",) if backend == "glif_cuda" else ("spike_source", "native_source")):
            production.require(len({entry.sources[field] for entry in entries}) == 10,
                               "BENCHMARK_BLOCK_SOURCE_REUSED:" + field)
        if backend == "glif_cuda":
            for field in ("circuit_reference", "circuit_validation", "gpu_primary", "gpu_replay", "authority_source"):
                production.require(len({entry.sources[field] for entry in entries}) == 1,
                                   "BENCHMARK_BLOCK_GLIF_SHARED_SOURCE_DIFFERS:"+field)
            production.require(entries[0].sources["gpu_primary"] != entries[0].sources["gpu_replay"],
                               "BENCHMARK_BLOCK_GLIF_DISTINCT_REPLAY_REQUIRED")
    else:
        # One learned circuit and replay engine serve the whole retrieval block.
        for field in fields:
            production.require(len({entry.sources[field] for entry in entries}) == 1,
                               "BENCHMARK_BLOCK_CA3_SHARED_CIRCUIT_DIFFERS:" + field)
    return entries


def process_block(cohort: production.ProductionCohort, inputs: Path,
                  circuit_index: int, block_index: int, output: Path,
                  *, resume: bool = False) -> dict:
    entries = load_block(cohort, inputs, circuit_index, block_index)
    if output.exists():
        production.require(resume, "BENCHMARK_BLOCK_EXPLICIT_RESUME_REQUIRED")
        allowed = {"report.json", "attempts", "dataset"}
        production.require(all(path.name in allowed for path in output.iterdir()),
                           "BENCHMARK_BLOCK_UNRESOLVED_OUTPUT")
    # One sequential writer catalog enforces the shared circuit and stores static
    # metadata once for the complete block.
    dataset_root = output / "dataset"
    # Validate every retained episode before processing later conditions.
    reused = {}
    for entry in entries:
        identity = entry.identity
        result = production.completed_episode(cohort, identity, dataset_root,
                                               resume=resume)
        if result is not None:
            reused[identity["episode_id"]] = result
    if (dataset_root / "dataset.json").exists():
        identity = entries[0].identity
        dataset = production.CircuitDataset(dataset_root, splits=(identity["split"],),
                                            allow_test=identity["split"] == "test")
        production.require(set(dataset.episodes) <= {entry.identity["episode_id"] for entry in entries}
                           and set(dataset.catalog["circuits"]) <= {identity["circuit_id"]},
                           "BENCHMARK_BLOCK_CONTAINS_OTHER_IDENTITIES")
    completed_path = output / "report.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        production.require(len(reused) == 10
                           and completed.get("status") == "COMPLETE_QUALIFIED_PRODUCTION_BLOCK"
                           and completed.get("complete_episodes") == 10
                           and completed.get("expected_episodes") == 10
                           and [result.get("identity") for result in completed.get("results", [])]
                               == [entry.identity for entry in entries]
                           and completed.get("scenario_id") == cohort.scenario
                           and completed.get("circuit_index") == circuit_index
                           and completed.get("block_index") == block_index
                           and completed.get("source_qualification_sha256") == cohort.qualification_sha256
                           and completed.get("identity_table_sha256") == cohort.identity_table_sha256,
                           "BENCHMARK_BLOCK_COMMITTED_REPORT_DIFFERS")
        return dict(completed, status="REUSED_COMMITTED_PRODUCTION_BLOCK", reused_episodes=10)
    for entry in entries:
        if entry.identity["episode_id"] in reused:
            continue
        retained = dataset_root / "source_execution" / entry.identity["episode_id"] / "result/report.json"
        for name, path in entry.sources.items():
            if cohort.scenario == "ca3_pattern_completion" and name == "packaged" and resume and retained.exists():
                continue
            production.require(path.exists(), "BENCHMARK_BLOCK_SOURCE_MISSING:" + name + ":" + str(path))
    output.mkdir(parents=True, exist_ok=True)
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=attempts))
    report = dict(schema=BLOCK_SCHEMA, status="PROCESSING_QUALIFIED_PRODUCTION_BLOCK",
                  scenario_id=cohort.scenario, circuit_index=circuit_index,
                  block_index=block_index, expected_episodes=10, complete_episodes=len(reused),
                  reused_episodes=len(reused), results=[], inputs=str(inputs.resolve()),
                  source_qualification_sha256=cohort.qualification_sha256,
                  identity_table_sha256=cohort.identity_table_sha256)
    committed = dict(reused)
    try:
        use_glif = entries[0].backend == "glif_cuda" and len(committed) < 10
        context = prepared_source_block(cohort, entries, attempt) if use_glif else nullcontext(None)
        with context as glif:
            if glif is not None:
                report["execution_backend"] = "glif_cuda"
                report["replay_comparison"] = glif.comparison
            for entry in entries:
                identity, sources = entry.identity, entry.sources
                if identity["episode_id"] in committed:
                    continue
                report["active_condition_index"] = identity["condition_index"]
                destination = dataset_root
                episode_resume = resume or bool(committed)
                if entry.backend == "glif_cuda":
                    index = identity["condition_index"]
                    result = production.write_episode(cohort, identity, glif.dataset, destination,
                        admission=glif.admissions[index], resume=episode_resume,
                        source_episode_id=glif.episode_ids[index])
                elif cohort.scenario == "v1_visual":
                    result = production.export_v1(cohort, identity, sources["rate_source"],
                        sources["spike_source"], sources["native_source"], sources["authority_source"],
                        destination, resume=episode_resume)
                else:
                    result = production.run_ca3(cohort, identity, sources["prepared_config"],
                        sources["packaged"], sources["formation"], sources["analysis"], destination,
                        sources["current_replay_engine"], resume=episode_resume)
                production.require(result.get("identity") == identity and result.get("complete_episodes") == 1
                                   and result.get("status") in ("EXPORTED_QUALIFIED_PRODUCTION_EPISODE",
                                                               "REUSED_COMMITTED_PRODUCTION_EPISODE"),
                                   "BENCHMARK_BLOCK_EPISODE_NOT_COMMITTED")
                committed[identity["episode_id"]] = result
        report.pop("active_condition_index", None)
        report["status"] = "COMPLETE_QUALIFIED_PRODUCTION_BLOCK"
    except BaseException as error:
        report.update(status="FAILED_QUALIFIED_PRODUCTION_BLOCK",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["results"] = sorted(committed.values(), key=lambda result: result["identity"]["condition_index"])
        report["complete_episodes"] = len(committed)
        freeze_json(attempt / "report.json", report)
    freeze_json(output / "report.json", report)
    return report



def prepare_glif_inputs(cohort, path: Path, circuit_index: int, block_index: int, output: Path) -> dict:
    """Bind newly completed input producers before the scheduled GPU process."""
    cohort.require_current()
    production.glif_backend_qualification(cohort)
    document = json.loads(path.read_text())
    production.require(cohort.scenario == "v1_visual" and document["schema"] == GLIF_INPUT_SCHEMA
        and document["circuit_index"] == circuit_index and document["block_index"] == block_index
        and len(document["conditions"]) == 10, "GLIF_PRODUCTION_INPUT_PLAN_DIFFERS")
    cases = []
    for index, row in enumerate(document["conditions"]):
        production.require(row["condition_index"] == index, "GLIF_PRODUCTION_INPUT_ORDER_DIFFERS")
        identity = cohort.identity(circuit_index, block_index, index)
        archive = (path.parent/row["spike_source"]).resolve()
        report, delivery = bound_json(archive, "visual_inputs/report.json")
        production.require(delivery["exit_code"] == report["exit_code"] == 0
            and report["complete_lgn_population"] and report["config"] == cohort.v1_visual_config(identity),
            "GLIF_PRODUCTION_INPUT_RECIPE_DIFFERS")
        cases.append(dict(identity=identity, input_archive=str(archive),
            input_report_sha256=sha256_file(archive/"visual_inputs/report.json"), input_config=report["config"]))
    production.require(len({row["input_archive"] for row in cases}) == 10, "GLIF_PRODUCTION_INPUT_REUSED")
    result = dict(schema="v1_source_input_block_v1", scenario_id="v1_visual",
                  circuit_index=circuit_index, block_index=block_index, cases=cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    freeze_json(output, result)
    return dict(status="PREPARED_QUALIFIED_GLIF_SOURCE_INPUTS", cases=10, production_episodes=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "qualification", "identity-table", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scenario", choices=production.LEGACY_SCENARIOS, required=True)
    parser.add_argument("--circuit-index", type=int, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--runtime-qualification", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-glif-inputs", action="store_true")
    args = parser.parse_args(argv)
    cohort = production.ProductionCohort.load(args.config, args.scenario,
                                             args.qualification, args.identity_table,
                                             runtime_qualification=args.runtime_qualification)
    if args.prepare_glif_inputs:
        result = prepare_glif_inputs(cohort, args.inputs, args.circuit_index, args.block_index, args.output)
    else:
        result = process_block(cohort, args.inputs, args.circuit_index, args.block_index,
                               args.output, resume=args.resume)
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

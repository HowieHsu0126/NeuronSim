"""Build reusable native CA3 cue controls against the existing corrected backend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .ca3_benchmark_cues import runtime_program
from .ca3_coupled_extension import extend_source
from .ca3_reference import prepare_retrieval_program, RETRIEVAL_PROJECT, RETRIEVAL_MAIN, RETRIEVAL_PATTERNS
from .benchmark_tasks import ca3_commands, ca3_cue_targets, condition_for_episode
from .ca3_control_trace import BACKGROUND_SIGMA, PYRAMIDAL, check_currents, source_phase
from .ca3_native_observation import ENVIRONMENT_VARIABLE, iter_intervals, read_header
from .ca3_event_counters import compare_event_counters
from .ca3_host_current_replay import compare_currents
from .ca3_benchmark_readout import compare_native_prefix
from .ca3_realization import SourceRealization, contract, from_record
from .ca3_retrieval_execution import qualify_input
from .ca3_formation_execution import native_launch_command
from .carlsim_records import network_header, spike_blocks
from .circuit_campaign import campaign_config, episode_identities
from .circuit_data import CircuitDataset, CircuitEpisode, CircuitUnits, CircuitArchiveWriter
from .resource_probe import probe
from .provenance import freeze_json, sha256_file


def build(author_root: Path, framework: Path, backend: Path, cuda: Path, output: Path,
          *, duration_ms: int = 1001) -> dict:
    """Compile the host program once; reuse authenticated observed GPU libraries."""
    if not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("CA3_BENCHMARK_FRESH_COMPUTE_BUILD_REQUIRED")
    output.mkdir(parents=True)
    source = output / "source"
    (source / "src").mkdir(parents=True)
    project = author_root / RETRIEVAL_PROJECT
    paths = (project / "generateCONFIGStateSTP.h", project / RETRIEVAL_MAIN,
             author_root / RETRIEVAL_PATTERNS)
    raw = tuple(path.read_bytes().decode() for path in paths)
    header, main, pattern, extension = extend_source(*raw, multiplier=2, role="retrieval")
    config = source / "generateCONFIGStateSTP.h"; config.write_bytes(header.encode())
    cpp = source / RETRIEVAL_MAIN; cpp.write_bytes(main.encode())
    setup = project / "generateSETUPStateSTP.h"
    shutil.copy2(setup, source / setup.name)
    prepare_retrieval_program(cpp, config, local_input=True, population_multiplier=2)
    effective, control = runtime_program(cpp.read_bytes(), duration_ms=duration_ms)
    cpp.write_bytes(effective)
    (output / "trainPat.csv").write_bytes(pattern.encode())
    manifest = json.loads((backend / "manifest.json").read_text())
    listed = {row["path"]: row for row in manifest["files"]}
    (output / "lib").mkdir()
    libraries = {}
    for name in ("libcarlsim.so.6.0.0", "libcarlsim-stopwatch.so", "libcurand.so.10"):
        key = "lib/" + name
        row = listed[key]; path = backend / key
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("CA3_BENCHMARK_BACKEND_CONTENT_CHANGED:" + key)
        shutil.copy2(path, output / key)
        libraries[name] = row["sha256"]
    observed = json.loads((backend / "build_report.json").read_text())
    if not observed.get("native_observation_overlay") or not observed.get("conductance_memory_overlay"):
        raise ValueError("CA3_BENCHMARK_CORRECTED_OBSERVED_BACKEND_REQUIRED")
    (output / "bin").mkdir()
    binary = output / "bin/ca3_benchmark"
    definitions = ("CSTP_DOUBLES", "JK_CA3_SNN", "LN_FIX_ALL_DECAY_CPU", "LN_FIX_ALL_DECAY_GPU",
                   "LN_GET_FIRING", "LN_GPU_INFO", "LN_GPU_STDP", "LN_I_CALC_TYPES",
                   "LN_UPDATE_CURSPIKES", "__CUDA11__", "__LN_EXT__", "__NO_PTHREADS__")
    includes = [cuda / "include", framework / "tools/stopwatch", framework / "carlsim/interface/inc",
                framework / "carlsim/kernel/inc", framework / "carlsim/monitor"]
    command = ["/usr/bin/g++", "-O3", "-DNDEBUG", "-std=gnu++14"]
    command += ["-D" + name for name in definitions]
    command += ["-I" + str(path) for path in includes]
    # Boost is only used for its counting iterator; use the same existing include
    # directory as the verified source build instead of adding a dependency.
    command += ["-idirafter", observed["boost_include"]]
    command += [str(cpp), "-o", str(binary), "-L" + str(output / "lib"),
                "-Wl,-rpath,$ORIGIN/../lib", "-l:libcarlsim.so.6.0.0", "-lcarlsim-stopwatch",
                "-l:libcurand.so.10", str(cuda / "lib/libcudart_static.a"), "-ldl", "-lrt", "-lpthread"]
    with (output / "build.log").open("x") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    report = dict(status="BUILT_CA3_BENCHMARK_NATIVE_CONTROLS", command=command,
                  binary_sha256=sha256_file(binary), control_program=control,
                  pattern_sha256=sha256_file(output / "trainPat.csv"),
                  parameter_identity=sha256_file(config),
                  coupled_extension=extension, backend_libraries=libraries,
                  backend_build_report=observed, backend_manifest_sha256=sha256_file(backend / "manifest.json"),
                  source_headers={path.name: hashlib.sha256(body.encode()).hexdigest()
                                  for path, body in zip(paths, raw)},
                  scientific_fidelity_qualified=False, production_episodes=0)
    freeze_json(output / "build_report.json", report)
    return report


def retrieval_realization(formation_report: dict, identity: dict) -> SourceRealization | None:
    """Use the formed circuit's actual seed, with legacy development preserved."""
    declared = formation_report.get("source_realization")
    if declared is None:
        return None
    formed = from_record(declared["realization"])
    if (declared != contract(formed) or formed.phase != "formation"
            or formed.circuit_id != identity["circuit_id"]
            or formed.circuit_seed != identity["connectivity_seed"]):
        raise ValueError("CA3_BENCHMARK_CANDIDATE_REQUIRES_ITS_OWN_FORMED_CIRCUIT")
    return SourceRealization(formed.circuit_id, identity["block_id"], "retrieval", "development",
                             formed.circuit_seed, identity["background_seed"], identity["initial_state_seed"])


def validate_runtime_composition(
        packaged: Path, built: dict, runtime_composition: Path | None = None
) -> tuple[dict, dict | None]:
    """Validate one explicit backend replacement without changing parent evidence."""
    libraries = dict(built["backend_libraries"])
    validated = None
    if runtime_composition is not None:
        descriptor_path = Path(runtime_composition)
        raw = descriptor_path.read_bytes()
        composition = json.loads(raw)
        required = {"runtime_root", "parent_runtime", "candidate_library",
                    "candidate_build_report", "candidate_receipt"}
        if not isinstance(composition, dict) or set(composition) != required:
            raise ValueError("CA3_RUNTIME_COMPOSITION_SCHEMA_INVALID")
        parent = composition["parent_runtime"]
        library = composition["candidate_library"]
        if (not isinstance(parent, dict) or set(parent) != {"root", "build_report_sha256"}
                or not isinstance(library, dict) or set(library) != {"name", "bytes", "sha256"}
                or not isinstance(composition["runtime_root"], str)
                or not isinstance(parent["root"], str)):
            raise ValueError("CA3_RUNTIME_COMPOSITION_BINDING_INVALID")
        runtime_root = Path(composition["runtime_root"])
        parent_root = Path(parent["root"])
        if (not runtime_root.is_absolute() or not parent_root.is_absolute()
                or runtime_root.resolve() != packaged.resolve()
                or runtime_root.resolve() == parent_root.resolve()):
            raise ValueError("CA3_RUNTIME_COMPOSITION_DISTINCT_ROOTS_REQUIRED")
        parent_raw = (parent_root / "build_report.json").read_bytes()
        if (hashlib.sha256(parent_raw).hexdigest() != parent["build_report_sha256"]
                or (packaged / "build_report.json").read_bytes() != parent_raw
                or json.loads(parent_raw) != built):
            raise ValueError("CA3_RUNTIME_COMPOSITION_PARENT_REPORT_CHANGED")
        # The parent keeps its original backend. Only the separate runtime may differ.
        for name, expected in libraries.items():
            if sha256_file(parent_root / "lib" / name) != expected:
                raise ValueError("CA3_RUNTIME_COMPOSITION_PARENT_LIBRARY_CHANGED:" + name)

        library_name = "libcarlsim.so.6.0.0"
        if (library["name"] != library_name or type(library["bytes"]) is not int
                or library["bytes"] <= 0 or not isinstance(library["sha256"], str)
                or len(library["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in library["sha256"])):
            raise ValueError("CA3_RUNTIME_COMPOSITION_CANDIDATE_LIBRARY_INVALID")
        records = {}
        for role in ("candidate_build_report", "candidate_receipt"):
            binding = composition[role]
            if (not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}
                    or not isinstance(binding["path"], str)
                    or type(binding["bytes"]) is not int or binding["bytes"] <= 0):
                raise ValueError("CA3_RUNTIME_COMPOSITION_ARTIFACT_BINDING_INVALID:" + role)
            path = Path(binding["path"])
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise ValueError("CA3_RUNTIME_COMPOSITION_ARTIFACT_PATH_INVALID:" + role)
            payload = path.read_bytes()
            if (len(payload) != binding["bytes"]
                    or hashlib.sha256(payload).hexdigest() != binding["sha256"]):
                raise ValueError("CA3_RUNTIME_COMPOSITION_ARTIFACT_CHANGED:" + role)
            records[role] = json.loads(payload)
            if not isinstance(records[role], dict):
                raise ValueError("CA3_RUNTIME_COMPOSITION_ARTIFACT_INVALID:" + role)
        candidate = records["candidate_build_report"]
        candidate_file = candidate.get("library")
        expected_identity = {key: library[key] for key in ("bytes", "sha256")}
        if (candidate.get("status") != "BUILT_UNQUALIFIED_CA3_SM80_BACKEND"
                or candidate.get("candidate_qualified") is not False
                or candidate.get("scientific_fidelity_qualified") is not False
                or candidate.get("framework_revision") != built["backend_build_report"]["framework_revision"]
                or candidate.get("original_library_sha256") != libraries[library_name]
                or candidate.get("original_curand_sha256") != libraries["libcurand.so.10"]
                or candidate.get("cuda_architectures") != ["sm_80"]
                or not isinstance(candidate_file, dict)
                or candidate_file.get("path") != "lib/" + library_name
                or {key: candidate_file.get(key) for key in expected_identity} != expected_identity):
            raise ValueError("CA3_RUNTIME_COMPOSITION_CANDIDATE_REPORT_DIFFERS")
        receipt = records["candidate_receipt"]
        receipt_file = receipt.get("candidate_library")
        # A transport recovery may have builds_run=0 and preserve an earlier failed
        # job. Its own successful publication and the bound library are the evidence.
        if (type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0
                or receipt.get("persisted") is not True or not isinstance(receipt_file, dict)
                or {key: receipt_file.get(key) for key in expected_identity} != expected_identity):
            raise ValueError("CA3_RUNTIME_COMPOSITION_SUCCESSFUL_RECEIPT_REQUIRED")
        candidate_path = packaged / "lib" / library_name
        if (candidate_path.is_symlink() or not candidate_path.is_file()
                or candidate_path.stat().st_size != library["bytes"]):
            raise ValueError("CA3_RUNTIME_COMPOSITION_CANDIDATE_LIBRARY_DIFFERS")
        libraries[library_name] = library["sha256"]
        validated = dict(
            **composition,
            descriptor=dict(path=str(descriptor_path.resolve()), bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest()),
            scope="ca3_sm80_candidate_prefix_replay",
            candidate_qualified=False, scientific_fidelity_qualified=False,
            candidate_receipt_execution=dict(
                status=receipt.get("status"), merlin_job_run_id=receipt.get("merlin_job_run_id"),
                merlin_trial_id=receipt.get("merlin_trial_id"),
                exit_code=receipt["exit_code"], persisted=receipt["persisted"]))
    for name, expected in libraries.items():
        if sha256_file(packaged / "lib" / name) != expected:
            raise ValueError("CA3_BENCHMARK_RUNTIME_LIBRARY_CHANGED:" + name)
    return libraries, validated


def run(packaged: Path, formation: Path, analysis: Path, config: Path, output: Path,
        *, block_index: int, condition_index: int, circuit_index: int = 0,
        current_replay_engine: Path | None = None, prefix_reference: Path | None = None,
        production_identity: dict | None = None,
        runtime_composition: Path | None = None) -> dict:
    """Execute source-format retrieval on a verified formed circuit.

    Standalone calls retain development seeds. The production bridge explicitly
    supplies its admitted frozen identity. Source-format recordings remain in
    the existing development archive until the production writer admits them.
    """
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES")
            or output.exists() or not 0 <= block_index < 100 or not 0 <= condition_index < 10):
        raise ValueError("CA3_BENCHMARK_ALLOCATED_FRESH_RETRIEVAL_REQUIRED")
    campaign = campaign_config(config)
    if type(circuit_index) is not int or not 0 <= circuit_index < campaign["sampling"]["circuit_instances"]:
        raise ValueError("CA3_BENCHMARK_CIRCUIT_INDEX_INVALID")
    identity = next(row for row in episode_identities(campaign, "ca3_pattern_completion",
                    production=production_identity is not None)
                    if row["circuit_index"] == circuit_index and row["block_index"] == block_index
                    and row["condition_index"] == condition_index)
    if production_identity is not None and production_identity != identity:
        raise ValueError("CA3_BENCHMARK_PRODUCTION_IDENTITY_DIFFERS")
    protocol = campaign["scenarios"]["ca3_pattern_completion"]["task_protocol"]
    condition = condition_for_episode(protocol, "ca3_pattern_completion", identity)
    duration_ms = protocol["retrieval_duration_ms"]
    built = json.loads((packaged / "build_report.json").read_text())
    if current_replay_engine is not None:
        engine_build = json.loads(current_replay_engine.with_name("host_current_replay_build.json").read_text())
        if (engine_build.get("status") != "BUILT_AND_VALIDATED"
                or sha256_file(current_replay_engine) != engine_build["binary_sha256"]):
            raise ValueError("CA3_BENCHMARK_CURRENT_REPLAY_ENGINE_CHANGED")
    if built["control_program"]["duration_ms"] != duration_ms:
        raise ValueError("CA3_BENCHMARK_BINARY_PROTOCOL_DURATION_DIFFERS")
    binary = packaged / "bin/ca3_benchmark"
    if built["status"] != "BUILT_CA3_BENCHMARK_NATIVE_CONTROLS" or sha256_file(binary) != built["binary_sha256"]:
        raise ValueError("CA3_BENCHMARK_NATIVE_BINARY_CHANGED")
    expected_pattern = built.get("pattern_sha256", built["backend_build_report"]["pattern_sha256"])
    if sha256_file(packaged / "trainPat.csv") != expected_pattern:
        raise ValueError("CA3_BENCHMARK_SOURCE_PATTERN_CHANGED")
    effective_libraries, composition = validate_runtime_composition(packaged, built, runtime_composition)
    normalized = analysis / "network65p_assem550_divnorm.dat"
    native_formation = json.loads((formation / "native_report.json").read_text())
    realization = retrieval_realization(native_formation, identity)
    if realization is None and circuit_index != 0:
        raise ValueError("CA3_BENCHMARK_NEW_CIRCUIT_REQUIRES_EXPLICIT_FORMATION_IDENTITY")
    binding = qualify_input(formation, analysis / "formation_replay.json", normalized,
                            analysis / "normalization.json", analysis / "normalization_replay.json",
                            coupled_extension=True, source_realization=realization)
    circuit_seed = realization.circuit_seed if realization is not None else 10
    circuit_id = realization.circuit_id if realization is not None else "ca3-development-" + binding["normalized_input_sha256"][:20]
    output.mkdir(parents=True)
    health = probe(output / "resource_preflight.json", required_memory_gib=128)
    if health["status"] != "PASS":
        raise RuntimeError("CA3_BENCHMARK_RESOURCE_PREFLIGHT_FAILED")
    run_dir = output / "native_run"; run_dir.mkdir()
    (run_dir / normalized.name).symlink_to(normalized.resolve())
    shutil.copy2(packaged / "trainPat.csv", run_dir / "trainPat.csv")
    targets = ca3_cue_targets(condition, identity["stimulus_seed"])
    target_path = output / "cue_targets.txt"
    target_path.write_text("\n".join(" ".join(map(str, row)) for row in targets) + "\n")
    trace = output / "native_observation.bin"
    command = native_launch_command(packaged, binary)
    command += [str(target_path.resolve()), str(identity["background_seed"]),
               str(identity["initial_state_seed"]), str(circuit_seed)]
    environment = dict(os.environ)
    environment[ENVIRONMENT_VARIABLE] = str(trace.resolve())
    environment["LD_LIBRARY_PATH"] = str((packaged / "lib").resolve())
    report = dict(status="RUNNING_CA3_BENCHMARK_DEVELOPMENT", command=command,
                  condition=condition, candidate_identity=identity, actual_split="development",
                  formed_circuit=binding, dt_ms=1.0, duration_ms=duration_ms,
                  task_protocol=protocol,
                  complete_episodes=0, production_episodes=0, scientific_fidelity_qualified=False,
                  started_utc=datetime.now(timezone.utc).isoformat())
    if composition is not None:
        report.update(runtime_composition=composition, effective_runtime_libraries=effective_libraries)
    started = time.perf_counter()
    try:
        child = None
        with (output / "native.log").open("x") as log:
            child = subprocess.Popen(command, cwd=run_dir, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        health = probe(output / "resource_latest.json", required_memory_gib=8)
                        if child.poll() is None and health["status"] != "PASS":
                            child.terminate(); child.wait()
                            raise RuntimeError("CA3_BENCHMARK_RESOURCE_STOP:" + ",".join(health["reasons"]))
                if child.returncode != 0:
                    raise subprocess.CalledProcessError(child.returncode, command)
            finally:
                if child.poll() is None:
                    child.terminate(); child.wait()
        if sha256_file(normalized) != binding["normalized_input_sha256"]:
            raise ValueError("CA3_BENCHMARK_NORMALIZED_INPUT_CHANGED")
        units, groups = read_header(trace)
        header = network_header(normalized)
        if units != 168106 or [(g.group_id, g.name, g.start, g.stop) for g in groups] != [
                (g.group_id, g.name, g.start, g.stop) for g in header["groups"]]:
            raise ValueError("CA3_BENCHMARK_FULL_UNIT_MAP_DIFFERS")
        pattern = np.asarray(list(map(int, (run_dir / "trainPat.csv").read_text().split())), dtype=np.int32)
        pyramidal = next(group for group in groups if group.name == "CA3_Pyramidal")
        controls = ca3_commands(pattern, pyramidal.start, condition, identity["stimulus_seed"])
        _, delivery = check_currents(groups, iter_intervals(trace, expected_stop_ms=duration_ms), controls,
                                     expected_stop_ms=duration_ms)
        if current_replay_engine is not None:
            current_realization = realization or SourceRealization(circuit_id,
                f"block_{block_index:03d}", "retrieval", "development", circuit_seed,
                identity["background_seed"], identity["initial_state_seed"])
            replay = compare_currents(current_replay_engine, groups, controls,
                                      iter_intervals(trace, expected_stop_ms=duration_ms), realization=current_realization,
                                      expected_stop_ms=duration_ms)
            report["independent_current_replay"] = replay
            if not replay["exact_float32_bits_equal"]:
                raise ValueError("CA3_BENCHMARK_INDEPENDENT_CURRENT_REPLAY_FAILED")
        counts = np.zeros(units, dtype=np.int64)
        fields = np.zeros((duration_ms, len(groups)), dtype=np.int64)
        event_ticks, event_units = [], []
        for group in groups:
            path = run_dir / "results" / ("spk_" + group.name + ".dat")
            for rows in spike_blocks(path, units=group.units, stop_ms=duration_ms):
                np.add.at(counts, rows[:, 1] + group.start, 1)
                np.add.at(fields[:, group.group_id], rows[:, 0], 1)
                event_ticks.append(rows[:, 0]); event_units.append(rows[:, 1] + group.start)
        ticks = np.concatenate(event_ticks) if event_ticks else np.empty(0, dtype=np.int64)
        event_ids = np.concatenate(event_units) if event_units else np.empty(0, dtype=np.int64)
        compare_event_counters(trace, np.column_stack((ticks, event_ids)), units, report, expected_stop_ms=duration_ms)
        if prefix_reference is not None:
            old = json.loads((prefix_reference / "report.json").read_text())
            if (duration_ms != 1210 or old.get("duration_ms") != 1001
                    or old.get("status") != "COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE"
                    or old.get("native_terminal_footer_verified") is not True
                    or old.get("unit_millisecond_counts_compared") != units * 1001
                    or old.get("condition") != condition or old.get("candidate_identity") != identity
                    or old.get("formed_circuit") != binding):
                raise ValueError("CA3_BENCHMARK_EXTENSION_MATCHED_COMPLETE_REFERENCE_REQUIRED")
            report["original_protocol_prefix"] = dict(
                reference=str(prefix_reference),
                **compare_native_prefix(prefix_reference / "native_observation.bin", trace))
        order = np.argsort(ticks, kind="stable")
        field_index = np.empty(units, dtype=np.int64); populations = np.empty(units, dtype="U32")
        for group in groups:
            field_index[group.start:group.stop] = group.group_id
            populations[group.start:group.stop] = group.name
        field_names = tuple(group.name for group in groups) + tuple("learned_memory_" + str(i) for i in range(3))
        for memory in range(3):
            begin = pyramidal.start + memory * 550
            field_index[begin:begin + 550] = len(groups) + memory
        provenance = dict(connectivity_identity=binding["normalized_input_sha256"],
                          parameter_identity=built.get("parameter_identity", built["backend_build_report"]["source_file_byte_checks"][1]["staged_sha256"]))
        revision = built["backend_build_report"]["framework_revision"]
        circuit = CircuitUnits(circuit_id=circuit_id, unit_ids=np.arange(units), populations=populations,
            field_index=field_index, field_names=field_names,
            coordinate_semantics="source_cell_types_no_anatomical_geometry", source_revision=revision,
            circuit_provenance=provenance)
        control_targets = np.unique(controls["unit_ids"])
        control_values = np.zeros((duration_ms, len(control_targets) + len(groups) + 2), dtype=np.float32)
        if len(control_targets):
            columns = np.searchsorted(control_targets, controls["unit_ids"])
            control_values[controls["time_ms"], columns] = 1
        features = [dict(name="set_current_unit_" + str(int(unit)), unit="dimensionless",
                         assignment_value="10000", assignment_unit="pA") for unit in control_targets]
        features += [dict(name=g.name + "_refresh_background", unit="dimensionless") for g in groups]
        features += [dict(name="CA3_Pyramidal_cpp_lognormal_s", unit="dimensionless"),
                     dict(name="CA3_Pyramidal_background_cap_enabled", unit="dimensionless")]
        for tick in range(duration_ms):
            _, cue = source_phase(tick)
            for index, group in enumerate(groups):
                control_values[tick, len(control_targets) + index] = int(not cue or group.name == PYRAMIDAL)
            control_values[tick, -2:] = (1.0 if cue else BACKGROUND_SIGMA[PYRAMIDAL], int(not cue))
        episode = CircuitEpisode(episode_id=f"{circuit_id}-block-{block_index:03d}-{condition['condition_id']}",
            circuit_id=circuit_id, block_id=f"{circuit_id}-block-{block_index:03d}", split="development",
            duration_ticks=duration_ms, dt_ms=1.0, spike_ticks=ticks[order], spike_unit_indices=event_ids[order],
            control_ticks=np.arange(duration_ms), control_values=control_values, control_features=tuple(features),
            seeds=dict(circuit_seed=circuit_seed, uniform_seed=identity["background_seed"],
                       engine_seed=identity["initial_state_seed"], cue_seed=identity["stimulus_seed"]),
            source_record=dict(source_revision=revision, circuit_provenance=provenance, condition=condition,
                               complete_unit_recording=True, simulation_completed=True,
                               recorded_until_tick=duration_ms, expected_duration_ticks=duration_ms, core_unit_count=units,
                               background_schedule="source_retrieval_lognormal_and_uniform_current_law",
                               formation_binding=binding))
        archive = CircuitArchiveWriter(output / "dataset", dict(schema="neural_circuit_episodes_v1",
                    scenario_id="ca3_pattern_completion", source_revision=revision, scope="source_reproduction",
                    sample_unit="complete_mechanistic_episode"))
        archive.register_circuit(circuit); archive.add(episode); archive.close()
        dataset = CircuitDataset(output / "dataset", splits=("development",))
        window = dataset.window(episode.episode_id, boundary_tick=520, history_ticks=1,
                                horizon_ticks=1, bin_ticks=1)
        expected = np.zeros((2, units), dtype=np.int64)
        selected = (ticks >= 519) & (ticks < 521)
        np.add.at(expected, (ticks[selected] - 519, event_ids[selected]), 1)
        if (not np.array_equal(np.concatenate((window["history"], window["targets"])), expected)
                or not np.array_equal(window["unit_ids"], circuit.unit_ids)
                or not np.array_equal(window["known_controls"], control_values[519:521])):
            raise ValueError("CA3_BENCHMARK_ARCHIVE_CONSUMER_READBACK_DIFFERS")
        np.savez_compressed(output / "population_counts.npz", unit_counts=counts, population_counts_per_ms=fields)
        np.savez_compressed(output / "ordered_cue_commands.npz", **controls)
        report.update(status="COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE", complete_episodes=1,
                      units=units, events=len(ticks), control_delivery=delivery,
                      archive_consumer_readback=dict(units=units, ticks=[519, 520], counts_exact=True,
                                                     unit_ids_exact=True, controls_exact=True),
                      all_unit_event_counter_equality=True, native_terminal_footer_verified=True)
    except BaseException as error:
        report.update(status="FAILED_CA3_BENCHMARK_DEVELOPMENT", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    builder = sub.add_parser("build")
    for name in ("author-root", "framework", "backend", "cuda", "output"):
        builder.add_argument("--" + name, type=Path, required=True)
    builder.add_argument("--retrieval-duration-ms", type=int, choices=(1001, 1210), default=1001)
    runner = sub.add_parser("run")
    for name in ("packaged", "formation", "analysis", "config", "output"):
        runner.add_argument("--" + name, type=Path, required=True)
    runner.add_argument("--circuit-index", type=int, default=0)
    runner.add_argument("--block-index", type=int, required=True)
    runner.add_argument("--condition-index", type=int, required=True)
    runner.add_argument("--current-replay-engine", type=Path)
    runner.add_argument("--prefix-reference", type=Path)
    runner.add_argument("--runtime-composition", type=Path,
                        help="Explicit unqualified backend composition; parent runtime evidence stays unchanged")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.author_root, args.framework, args.backend, args.cuda, args.output,
              duration_ms=args.retrieval_duration_ms)
    else:
        run(args.packaged, args.formation, args.analysis, args.config, args.output,
            block_index=args.block_index, condition_index=args.condition_index, circuit_index=args.circuit_index,
            current_replay_engine=args.current_replay_engine, prefix_reference=args.prefix_reference,
            runtime_composition=args.runtime_composition)


if __name__ == "__main__":
    main()

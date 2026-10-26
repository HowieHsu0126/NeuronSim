"""Execute the pinned EPFL working-memory source and archive all E/I spikes.

This is a source-reproduction comparator, not an exact reproduction of Compte
2000 or a qualified 100k circuit. No neuronal or synaptic equation is replaced.
"""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import pstats
import resource
import socket
import sys
import time

import brian2 as b2
import h5py
import numpy as np
import yaml

from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits
from .provenance import canonical_json, freeze_json, sha256_file
from .pfc_factorized_source import EXECUTION_NAME, install_uniform_projections
from .pfc_initial_rng import install_initial_voltage_rng
from . import pfc_kernel_parameters as kernel_parameters

SOURCE_ENTRY = "neurodynex3/working_memory_network/wm_model.py"


def load_source(source: Path):
    manifest = json.loads((source / "source.json").read_text())
    entry = next(row for row in manifest["files"] if row["path"] == SOURCE_ENTRY)
    if (manifest["repository"] != "https://github.com/EPFL-LCN/neuronaldynamics-exercises"
            or sha256_file(source / SOURCE_ENTRY) != entry["sha256"]):
        raise ValueError("PFC_SOURCE_IDENTITY_MISMATCH")
    spec = importlib.util.spec_from_file_location("neuronsim_pinned_epfl_wm", source / SOURCE_ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, manifest, entry


def case_settings(config):
    """Resolve source randomization blocks and legal cue/distractor currents."""
    policy = config.get("random_stream_policy", "source_global")
    if policy not in ("source_global", "independent_initial_external"):
        raise ValueError("PFC_RANDOM_STREAM_POLICY_UNSUPPORTED")
    cases = []
    for row in config['cases']:
        if policy == "source_global":
            if any(key in entry for entry in (config, row)
                   for key in ("initial_rng_seed", "external_noise_seed")):
                raise ValueError("PFC_INDEPENDENT_SEEDS_REQUIRE_EXPLICIT_POLICY")
            seed = row.get('source_rng_seed', config['source_rng_seed'])
            seeds = {"source_rng_seed": seed}
            block = f'pfc_source_block_{seed}' if 'source_rng_seed' in row else 'pfc_source_block_0'
        else:
            if any("source_rng_seed" in entry for entry in (config, row)):
                raise ValueError("PFC_INDEPENDENT_STREAMS_REQUIRE_NAMED_SEEDS")
            initial = row.get("initial_rng_seed", config.get("initial_rng_seed"))
            seed = row.get("external_noise_seed", config.get("external_noise_seed"))
            seeds = {"initial_rng_seed": initial, "external_noise_seed": seed}
            block = f"pfc_initial_{initial}_external_{seed}"
        if any(type(value) is not int or not 0 <= value < 2**32 for value in seeds.values()):
            raise ValueError('PFC_SOURCE_RANDOM_SEED_INVALID')
        cue = row.get('stimulus_strength_nA', config['stimulus_strength_nA'])
        distractor = row['distractor_strength_nA']
        if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in (cue, distractor)):
            raise ValueError('PFC_CONTROL_CURRENT_INVALID')
        case_id = row['case_id']
        if not isinstance(case_id, str) or Path(case_id).name != case_id or case_id in ('', '.', '..'):
            raise ValueError('PFC_CASE_ID_INVALID')
        # source_rng_seed remains an internal execution key. Independent-policy
        # archives use the explicit initial_rng_seed/external_noise_seed names.
        resolved = dict(row, **{key:value for key,value in seeds.items() if key not in row})
        cases.append(dict(resolved, source_rng_seed=seed, random_stream_policy=policy,
                          stimulus_strength_nA=float(cue), distractor_strength_nA=float(distractor),
                          block_id=block))
    if not cases or len({r['case_id'] for r in cases}) != len(cases):
        raise ValueError('PFC_CASE_IDENTITIES_MUST_BE_NONEMPTY_AND_UNIQUE')
    return cases


def episode_seeds(case):
    if case['random_stream_policy'] == 'source_global':
        return {"source_rng_seed": case["source_rng_seed"]}
    return {key:case[key] for key in ("initial_rng_seed", "external_noise_seed")}


def stream_description(case):
    return ("source_global_rng_not_separated" if case["random_stream_policy"] == "source_global"
            else "independent_initial_voltage_and_external_poisson")


def verify_matched_prefixes(records, duration_ticks):
    """Compare same-RNG event prefixes up to the first actual control difference."""
    checks = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if (left['seed'], left.get('initial_rng_seed')) != (right['seed'], right.get('initial_rng_seed')):
                continue
            if not np.array_equal(left['control_ticks'], right['control_ticks']):
                raise ValueError('PFC_MATCHED_CONTROL_CLOCK_CHANGED')
            different = np.flatnonzero(np.any(left['controls'] != right['controls'], axis=1))
            boundary = int(left['control_ticks'][different[0]]) if len(different) else duration_ticks
            a, b = left['ticks'] < boundary, right['ticks'] < boundary
            if (not np.array_equal(left['ticks'][a], right['ticks'][b])
                    or not np.array_equal(left['ids'][a], right['ids'][b])):
                raise ValueError('PFC_MATCHED_CONTROL_PREFIX_CHANGED')
            checks.append(dict(source_rng_seed=left['seed'], left_case=left['case_id'],
                               right_case=right['case_id'], prefix_stop_tick=boundary,
                               event_count=int(a.sum()), prefix_equal=True,
                               initial_rng_seed=left.get('initial_rng_seed')))
    return checks


def control_program(config, case):
    changes = sorted({0., config["stimulus_start_ms"], config["stimulus_start_ms"]+config["stimulus_duration_ms"],
                      config["distractor_start_ms"], config["distractor_start_ms"]+config["distractor_duration_ms"]})
    values = np.array([[case["stimulus_strength_nA"] if config["stimulus_start_ms"] <= t < config["stimulus_start_ms"]+config["stimulus_duration_ms"] else 0.,
                        case["distractor_strength_nA"] if config["distractor_start_ms"] <= t < config["distractor_start_ms"]+config["distractor_duration_ms"] else 0.] for t in changes])
    ticks = np.rint(np.array(changes)/config["dt_ms"]).astype(np.int64)
    features = ({"name":"cue_current", "unit":"nA", "population":"E",
                 "center_degree":config["stimulus_center_deg"], "width_degree":config["stimulus_width_deg"]},
                {"name":"distractor_current", "unit":"nA", "population":"E",
                 "center_degree":config["distractor_center_deg"], "width_degree":config["distractor_width_deg"]})
    return ticks, values, features


def resume_cases(output, binding, units, resolved_cases, duration_ticks):
    """Validate committed episodes before archive recovery mutates state.

    Closed shards interrupted before catalog publication use the existing
    writer recovery contract. Raw files without complete indexed HDF5 records
    are rejected, including orphan raw files from an interrupted writing shard.
    """
    dataset = CircuitDataset(output / "dataset", splits=("development",))
    catalog = dataset.catalog
    if catalog["binding"] != binding:
        raise ValueError("PFC_RESUME_BINDING_CHANGED")
    if set(catalog["circuits"]) != {units.circuit_id}:
        raise ValueError("PFC_RESUME_CIRCUIT_IDENTITY_CHANGED")
    circuit = catalog["circuits"][units.circuit_id]
    if circuit["units"] != len(units.unit_ids) or circuit["provenance"] != units.circuit_provenance:
        raise ValueError("PFC_RESUME_CIRCUIT_IDENTITY_CHANGED")
    with h5py.File(dataset._verified(circuit["path"], circuit["sha256"]), "r") as stored:
        expected = dict(unit_ids=units.unit_ids, populations=units.populations, field_index=units.field_index,
                        field_names=np.asarray(units.field_names))
        for name, value in expected.items():
            actual = stored[name].asstr()[:] if name in ("populations", "field_names") else stored[name][:]
            if not np.array_equal(actual, value):
                raise ValueError("PFC_RESUME_OBSERVATION_MAP_CHANGED:" + name)
        if ("coordinates" in stored or stored.attrs["coordinate_units"] != "not_available"
                or stored.attrs["coordinate_semantics"] != units.coordinate_semantics
                or stored.attrs["source_revision"] != units.source_revision
                or json.loads(stored.attrs["circuit_provenance_json"]) != units.circuit_provenance):
            raise ValueError("PFC_RESUME_OBSERVATION_MAP_CHANGED")
    indexed = {case["case_id"]: case for case in resolved_cases}
    shards = list(catalog["shards"])
    if catalog["complete_episodes"] != sum(len(row["episodes"]) for row in shards):
        raise ValueError("PFC_RESUME_EPISODE_CATALOG_CHANGED")
    closed = dataset.root / f"episodes_{len(shards):06d}.h5"
    if closed.exists():
        with h5py.File(closed, "r") as stored:
            rows = json.loads(stored.attrs["completed_episode_index_json"])
        shards.append(dict(path=closed.name, sha256=sha256_file(closed), episodes=rows))
    recovered = {}
    for shard in shards:
        with h5py.File(dataset._verified(shard["path"], shard["sha256"]), "r") as stored:
            rows = shard["episodes"]
            if set(stored.keys()) != {row["episode_id"] for row in rows}:
                raise ValueError("PFC_RESUME_SHARD_INDEX_CHANGED")
            for row in rows:
                name = row["episode_id"]
                if name not in indexed or name in recovered:
                    raise ValueError("PFC_RESUME_UNKNOWN_OR_DUPLICATE_EPISODE")
                group, case = stored[name], indexed[name]
                metadata = json.loads(group.attrs["metadata_json"])
                episode = CircuitEpisode(**metadata,
                    **{key:group[key][:] for key in ("spike_ticks", "spike_unit_indices", "control_ticks", "control_values")})
                episode.validate(len(units.unit_ids))
                expected = dict(episode_id=name, circuit_id=units.circuit_id, block_id=case["block_id"],
                                split="development", duration_ticks=duration_ticks, dt_ms=binding["resolved_source_config"]["dt_ms"])
                if (any(getattr(episode, key) != value or row[key] != value for key, value in expected.items())
                        or row["event_count"] != len(episode.spike_ticks)
                        or episode.seeds != episode_seeds(case)
                        or episode.control_kind != "piecewise_constant"
                        or episode.task_outcome != "not_scientifically_assessed"):
                    raise ValueError("PFC_RESUME_EPISODE_IDENTITY_CHANGED")
                ticks, values, features = control_program(binding["resolved_source_config"], case)
                if (not np.array_equal(episode.control_ticks, ticks)
                        or not np.array_equal(episode.control_values, values)
                        or canonical_json(episode.control_features) != canonical_json(features)):
                    raise ValueError("PFC_RESUME_CONTROL_PROGRAM_CHANGED")
                record = episode.source_record
                expected_record = dict(source_revision=binding["source_revision"],
                    native_source_sha256=binding["source_entry_sha256"],
                    circuit_provenance=units.circuit_provenance,
                    initial_and_background_streams=stream_description(case),
                    synapse_execution=binding["synapse_execution"],
                    connection_identity_kind="fixed_source_rule_and_parameters_not_native_edge_replay")
                if any(record.get(key) != value for key, value in expected_record.items()):
                    raise ValueError("PFC_RESUME_SOURCE_RECORD_CHANGED")
                raw = output / (name + "_native.npz")
                if sha256_file(raw) != record["raw_sha256"]:
                    raise ValueError("PFC_RESUME_RAW_DIGEST_CHANGED:" + name)
                with np.load(raw, allow_pickle=False) as archive:
                    if set(archive.files) != {"spike_unit_indices", "spike_seconds", "spike_ticks",
                                              "native_spike_counts", "terminal_voltage_mV"}:
                        raise ValueError("PFC_RESUME_RAW_SCHEMA_CHANGED")
                    arrays = {key:archive[key] for key in archive.files}
                if (not all(np.isfinite(value).all() for value in arrays.values())
                        or not np.array_equal(arrays["spike_ticks"], episode.spike_ticks)
                        or not np.array_equal(arrays["spike_unit_indices"], episode.spike_unit_indices)
                        or arrays["spike_ticks"].dtype.kind not in "iu"
                        or arrays["spike_unit_indices"].dtype.kind not in "iu"
                        or arrays["native_spike_counts"].dtype.kind not in "iu"
                        or arrays["native_spike_counts"].shape != (len(units.unit_ids),)
                        or arrays["terminal_voltage_mV"].shape != (len(units.unit_ids),)
                        or arrays["spike_seconds"].shape != episode.spike_ticks.shape
                        or not np.array_equal(np.bincount(episode.spike_unit_indices, minlength=len(units.unit_ids)),
                                              arrays["native_spike_counts"])
                        or not np.allclose(arrays["spike_seconds"]/(episode.dt_ms/1000), episode.spike_ticks, rtol=0, atol=1e-8)):
                    raise ValueError("PFC_RESUME_RAW_EPISODE_MISMATCH:" + name)
                if case["random_stream_policy"] == "independent_initial_external":
                    initial_name = name + "_initial_voltage.npy"
                    initial_path = output / initial_name
                    if (record.get("initial_voltage_file") != initial_name
                            or record.get("initial_voltage_units") != "mV"
                            or record.get("initial_voltage_role") != "source_provenance_not_model_input"
                            or record.get("initial_voltage_sha256") != sha256_file(initial_path)):
                        raise ValueError("PFC_RESUME_INITIAL_VOLTAGE_BINDING_CHANGED")
                    initial = np.load(initial_path, allow_pickle=False)
                    if (initial.dtype != np.dtype(np.float64) or initial.shape != (len(units.unit_ids),)
                            or not np.isfinite(initial).all() or not ((initial >= -60.) & (initial < -50.)).all()):
                        raise ValueError("PFC_RESUME_INITIAL_VOLTAGE_INVALID")
                elapsed = record.get("elapsed_seconds")
                if elapsed is not None and (not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0):
                    raise ValueError("PFC_RESUME_ELAPSED_TIME_INVALID")
                result = dict(case_id=name, source_rng_seed=case["source_rng_seed"],
                    stimulus_strength_nA=case["stimulus_strength_nA"], distractor_strength_nA=case["distractor_strength_nA"],
                    native_events=len(episode.spike_ticks), core_units=len(units.unit_ids),
                    elapsed_seconds=elapsed, raw_file=raw.name, raw_sha256=record["raw_sha256"])
                if case["random_stream_policy"] == "independent_initial_external":
                    result.update(episode_seeds(case), source_rng_seed_role="external_noise",
                                  initial_voltage_file=record["initial_voltage_file"],
                                  initial_voltage_sha256=record["initial_voltage_sha256"])
                if elapsed is None:
                    result["elapsed_time_status"] = "unavailable_in_legacy_episode_record"
                recovered[name] = dict(case=result, saved=dict(case_id=name, seed=case["source_rng_seed"],
                    ticks=episode.spike_ticks, ids=episode.spike_unit_indices, control_ticks=ticks, controls=values,
                    initial_rng_seed=case.get("initial_rng_seed")))
    raw_names = {path.name for path in output.iterdir() if path.name.endswith("_native.npz")}
    if raw_names != {name + "_native.npz" for name in recovered}:
        raise ValueError("PFC_RESUME_UNTRACKED_OR_MISSING_RAW_FILES")
    initial_names = {path.name for path in output.iterdir() if path.name.endswith("_initial_voltage.npy")}
    expected_initial = {case["case_id"] + "_initial_voltage.npy" for case in resolved_cases
                        if case["case_id"] in recovered and case["random_stream_policy"] == "independent_initial_external"}
    if initial_names != expected_initial:
        raise ValueError("PFC_RESUME_UNTRACKED_OR_MISSING_INITIAL_VOLTAGE")
    return recovered


def execute(source: Path, config_path: Path, output: Path, *, resume=False, native_input_observer=None):
    if resume and native_input_observer is not None:
        raise ValueError("PFC_INPUT_OBSERVATION_REQUIRES_FRESH_SOURCE_EXECUTION")
    config = yaml.safe_load(config_path.read_text())
    if config["scope"] != "source_reproduction" or config["expected_source_revision"] != json.loads((source / "source.json").read_text())["revision"]:
        raise ValueError("PFC_REPRODUCTION_SOURCE_BINDING_REQUIRED")
    if output.exists() and not resume:
        raise FileExistsError("PFC_REPRODUCTION_OUTPUT_EXISTS")
    if resume and not output.is_dir():
        raise FileNotFoundError("PFC_RESUME_OUTPUT_REQUIRED")
    if resume and (output / "report.json").exists():
        raise FileExistsError("PFC_REPRODUCTION_ALREADY_COMPLETE")
    resolved_cases = case_settings(config)
    kernel = kernel_parameters.resolve_kernel_parameters(config)
    synapse_execution = config.get("synapse_execution", "native_source")
    if synapse_execution not in ("native_source", EXECUTION_NAME):
        raise ValueError("PFC_SYNAPSE_EXECUTION_UNSUPPORTED")
    module, manifest, entry = load_source(source)
    if native_input_observer is not None:
        native_input_observer.install(module)
    output.mkdir(parents=True, exist_ok=resume)
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    dt = config["dt_ms"]
    duration_ticks = int(round(config["duration_ms"] / dt))
    if not np.isclose(duration_ticks * dt, config["duration_ms"], rtol=0, atol=1e-9):
        raise ValueError("PFC_DURATION_NOT_ON_NATIVE_CLOCK")
    provenance = kernel_parameters.circuit_provenance(config, entry["sha256"])
    fields = config["excitatory_preference_bins"]
    field_index = np.concatenate([np.arange(n_e) * fields // n_e, np.full(n_i, fields)])
    units = CircuitUnits("pfc_source_reference", np.arange(n_e + n_i),
                         np.array(["E"] * n_e + ["I"] * n_i), field_index,
                         tuple([f"E_preference_{i*360/fields:g}_{(i+1)*360/fields:g}_deg" for i in range(fields)] + ["I_population"]),
                         "E angle=360*source_index/N_E degrees; I has no preference coordinate",
                         manifest["revision"], circuit_provenance=provenance)
    binding = dict(schema=SCHEMA, scope="source_reproduction", scenario_id="pfc_working_memory",
                   source_revision=manifest["revision"], sample_unit="complete_mechanistic_episode",
                   config_sha256=sha256_file(config_path), source_entry_sha256=entry["sha256"],
                   resolved_source_config=config, synapse_execution=synapse_execution,
                   resolved_kernel_parameters=kernel,
                   kernel_parameterization_sha256=sha256_file(Path(kernel_parameters.__file__)),
                   execution_implementation_sha256=sha256_file(Path(__file__).with_name("pfc_factorized_source.py"))
                       if synapse_execution == EXECUTION_NAME else entry["sha256"])
    if resolved_cases[0]["random_stream_policy"] == "independent_initial_external":
        binding["initial_voltage_implementation_sha256"] = sha256_file(Path(__file__).with_name("pfc_initial_rng.py"))
    recovered = resume_cases(output, binding, units, resolved_cases, duration_ticks) if resume else {}
    cases, saved = [], []
    # Publish each completed source episode before starting the next simulation.
    with CircuitArchiveWriter(output / "dataset", binding, resume=resume, target_shard_bytes=1) as writer:
        writer.register_circuit(units)
        for case in resolved_cases:
            if case["case_id"] in recovered:
                cases.append(recovered[case["case_id"]]["case"])
                saved.append(recovered[case["case_id"]]["saved"])
                continue
            if native_input_observer is not None:
                native_input_observer.begin_case(case, config)
            b2.start_scope()
            b2.defaultclock.dt = dt * b2.ms
            b2.prefs.codegen.target = "numpy"
            seed = case["source_rng_seed"]
            random.seed(seed)
            np.random.seed(seed)
            b2.seed(seed)
            initial_trace = (install_initial_voltage_rng(module, case["initial_rng_seed"])
                             if case["random_stream_policy"] == "independent_initial_external" else None)
            factory = install_uniform_projections(module) if synapse_execution == EXECUTION_NAME else None
            started = time.perf_counter()
            result = module.simulate_wm(
                N_excitatory=n_e, N_inhibitory=n_i,
                weight_scaling_factor=config["weight_scaling_factor"],
                sigma_weight_profile=kernel["sigma_weight_profile"],
                Jpos_excit2excit=kernel["Jpos_excit2excit"],
                stimulus_center_deg=config["stimulus_center_deg"],
                stimulus_width_deg=config["stimulus_width_deg"],
                stimulus_strength=case["stimulus_strength_nA"] * b2.namp,
                t_stimulus_start=config["stimulus_start_ms"] * b2.ms,
                t_stimulus_duration=config["stimulus_duration_ms"] * b2.ms,
                distractor_center_deg=config["distractor_center_deg"],
                distractor_width_deg=config["distractor_width_deg"],
                distractor_strength=case["distractor_strength_nA"] * b2.namp,
                t_distractor_start=config["distractor_start_ms"] * b2.ms,
                t_distractor_duration=config["distractor_duration_ms"] * b2.ms,
                monitored_subset_size=config["state_monitored_per_population"],
                sim_time=config["duration_ms"] * b2.ms)
            elapsed = time.perf_counter() - started
            if native_input_observer is not None:
                native_input_observer.finish_case(case, result)
            if factory is not None and (len(factory.projections) != 3
                    or sum(p.summed_nmda for p in factory.projections) != 1
                    or sum(p.exclude_self for p in factory.projections) != 1):
                raise ValueError("PFC_FACTORIZED_PROJECTION_COVERAGE_CHANGED")
            spike_e, spike_i = result[1], result[5]
            native_ids = np.concatenate([np.asarray(spike_e.i, dtype=np.int64), np.asarray(spike_i.i, dtype=np.int64) + n_e])
            native_seconds = np.concatenate([np.asarray(spike_e.t / b2.second), np.asarray(spike_i.t / b2.second)])
            native_ticks = native_seconds / (dt / 1000)
            ticks = np.rint(native_ticks).astype(np.int64)
            if not np.allclose(native_ticks, ticks, rtol=0, atol=1e-8):
                raise ValueError("PFC_SOURCE_EVENTS_OFF_NATIVE_CLOCK")
            counts = np.concatenate([np.asarray(spike_e.count, dtype=np.int64), np.asarray(spike_i.count, dtype=np.int64)])
            if len(counts) != n_e + n_i or not np.array_equal(np.bincount(native_ids, minlength=n_e+n_i), counts):
                raise ValueError("PFC_SPIKE_MONITOR_NOT_COMPLETE")
            recorded_until = int(round(float(b2.defaultclock.t / b2.ms) / dt))
            if recorded_until != duration_ticks:
                raise ValueError("PFC_SOURCE_EARLY_TERMINATION")
            order = np.lexsort((native_ids, ticks))
            ticks, native_ids, native_seconds = ticks[order], native_ids[order], native_seconds[order]
            control_ticks, controls, control_features = control_program(config, case)
            raw = output / (case["case_id"] + "_native.npz")
            with raw.open("xb") as stream:
                np.savez_compressed(stream, spike_unit_indices=native_ids, spike_seconds=native_seconds,
                                    spike_ticks=ticks, native_spike_counts=counts,
                                    terminal_voltage_mV=np.concatenate([np.asarray(spike_e.source.v / b2.mV), np.asarray(spike_i.source.v / b2.mV)]))
            record = dict(complete_unit_recording=True, simulation_completed=True,
                          recorded_until_tick=recorded_until, expected_duration_ticks=duration_ticks,
                          core_unit_count=n_e+n_i, source_revision=manifest["revision"], elapsed_seconds=elapsed,
                          circuit_provenance=provenance, raw_sha256=sha256_file(raw),
                          native_source_sha256=entry["sha256"],
                          initial_and_background_streams=stream_description(case),
                          synapse_execution=synapse_execution,
                          connection_identity_kind="fixed_source_rule_and_parameters_not_native_edge_replay")
            if initial_trace is not None:
                initial = initial_trace.initial_voltage_mV
                if (initial.shape != (n_e+n_i,) or [len(draw.values_mV) for draw in initial_trace.draws] != [n_i,n_e]):
                    raise ValueError("PFC_INITIAL_VOLTAGE_UNIT_MAPPING_CHANGED")
                initial_path = output / (case["case_id"] + "_initial_voltage.npy")
                with initial_path.open("xb") as stream:
                    np.save(stream, initial, allow_pickle=False)
                record.update(initial_voltage_file=initial_path.name,
                              initial_voltage_sha256=sha256_file(initial_path),
                              initial_voltage_units="mV", initial_voltage_rng=initial_trace.rng_name,
                              initial_voltage_role="source_provenance_not_model_input")
            writer.add(CircuitEpisode(case["case_id"], units.circuit_id, case["block_id"], "development",
                                     duration_ticks, dt, ticks, native_ids,
                                     control_ticks, controls, control_features,
                                     seeds=episode_seeds(case), source_record=record,
                                     task_outcome="not_scientifically_assessed"))
            saved.append(dict(case_id=case['case_id'], seed=seed, ticks=ticks, ids=native_ids,
                              control_ticks=control_ticks, controls=controls, initial_rng_seed=case.get("initial_rng_seed")))
            cases.append(dict(case_id=case["case_id"], source_rng_seed=seed, stimulus_strength_nA=case["stimulus_strength_nA"], distractor_strength_nA=case["distractor_strength_nA"], native_events=len(ticks), core_units=n_e+n_i,
                              elapsed_seconds=elapsed, raw_file=raw.name, raw_sha256=sha256_file(raw)))
            if initial_trace is not None:
                cases[-1].update(episode_seeds(case), source_rng_seed_role="external_noise",
                                 initial_voltage_file=record["initial_voltage_file"],
                                 initial_voltage_sha256=record["initial_voltage_sha256"])
            print(json.dumps(cases[-1]), flush=True)
    prefix_checks = verify_matched_prefixes(saved, duration_ticks)
    shared_prefix = all(row['prefix_equal'] for row in prefix_checks) if prefix_checks else None
    dataset = CircuitDataset(output / "dataset", splits=("development",))
    observed_total = 0
    bin_ticks = int(round(config["readout_bin_ms"] / dt))
    for case in cases:
        readout = dataset.window(case["case_id"], boundary_tick=duration_ticks//2,
                                 history_ticks=duration_ticks//2, horizon_ticks=duration_ticks//2, bin_ticks=bin_ticks)
        with np.load(output / case["raw_file"], allow_pickle=False) as raw:
            expected = np.zeros((duration_ticks//bin_ticks, n_e+n_i), dtype=np.int64)
            for t, unit in zip(raw["spike_ticks"], raw["spike_unit_indices"]):
                if t < duration_ticks:
                    expected[t//bin_ticks, unit] += 1
            actual = np.concatenate([readout["history"], readout["targets"]])
            if not np.array_equal(expected, actual):
                raise ValueError("PFC_ARCHIVE_INDEPENDENT_COUNT_REPLAY_FAILED")
            observed_total += int(actual.sum())
    report = dict(status="NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS", cases=cases,
                  source_revision=manifest["revision"], source_entry_sha256=entry["sha256"],
                  brian2_version=b2.__version__, numpy_version=np.__version__,
                  matched_pre_intervention_prefix_identical=shared_prefix,
                  matched_prefix_checks=prefix_checks,
                  independent_source_rng_blocks=(len({r["source_rng_seed"] for r in resolved_cases})
                      if resolved_cases[0]["random_stream_policy"] == "source_global" else None),
                  random_stream_policy=resolved_cases[0]["random_stream_policy"],
                  source_randomization_blocks=len({(r["source_rng_seed"],r.get("initial_rng_seed")) for r in resolved_cases}),
                  independently_replayed_events=observed_total, scope="source_reproduction",
                  full_scale_qualified=False, scientific_fidelity_qualified=False,
                  parameters_differ_from_original_compte=True, resolved_config=config,
                  resolved_kernel_parameters=kernel, circuit_provenance=provenance,
                  kernel_parameterization_sha256=binding["kernel_parameterization_sha256"],
                  python_profiler_active=sys.getprofile() is not None,
                  synapse_execution=synapse_execution,
                  represented_uniform_connections=2*n_e*n_i+n_i*(n_i-1),
                  stored_uniform_connections=0 if synapse_execution == EXECUTION_NAME else 2*n_e*n_i+n_i*(n_i-1),
                  execution_implementation_sha256=binding["execution_implementation_sha256"])
    if "initial_voltage_implementation_sha256" in binding:
        report["initial_voltage_implementation_sha256"] = binding["initial_voltage_implementation_sha256"]
    report.update(resumed=bool(resume), reused_complete_case_ids=[case["case_id"] for case in resolved_cases if case["case_id"] in recovered])
    freeze_json(output / "report.json", report)
    return report


def profiled_execute(source: Path, config: Path, output: Path, profile_output: Path,
                     *, resume=False):
    """Profile the canonical entry point without replacing source operations.

    Timings include source setup, all-unit export and archive count replay, but
    exclude Python imports, staging and copyout. cProfile overhead is included;
    these measurements locate costs and cannot establish a matched-cost speedup.
    Peak RSS is the Linux process high-water mark, not per-case allocated memory.
    """
    if profile_output.exists():
        raise FileExistsError("PFC_EXECUTION_PROFILE_EXISTS")
    if profile_output.resolve().is_relative_to(output.resolve()):
        raise ValueError("PFC_EXECUTION_PROFILE_MUST_BE_OUTSIDE_SOURCE_OUTPUT")
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    started_utc = datetime.now(timezone.utc).isoformat()
    started, cpu_started = time.perf_counter(), time.process_time()
    report = None
    failure = None
    try:
        report = profiler.runcall(execute, source, config, output, resume=resume)
        return report
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        elapsed, cpu_elapsed = time.perf_counter()-started, time.process_time()-cpu_started
        stats = pstats.Stats(profiler)
        functions = [dict(filename=filename, line=line, function=name,
                          primitive_calls=values[0], total_calls=values[1],
                          self_seconds=values[2], cumulative_seconds=values[3])
                     for (filename, line, name), values in stats.stats.items()]
        functions.sort(key=lambda row: (-row["self_seconds"], row["filename"], row["line"], row["function"]))
        freeze_json(profile_output, dict(
            status="PROFILED_SOURCE_EXECUTION_COMPLETE" if report is not None else "PROFILED_SOURCE_EXECUTION_FAILED",
            started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(), python_version=sys.version,
            source=str(source.resolve()), config=str(config.resolve()),
            output=str(output.resolve()), resumed=bool(resume),
            report_sha256=sha256_file(output / "report.json") if report is not None else None,
            wall_seconds=elapsed, process_cpu_seconds=cpu_elapsed,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            measurement_scope="execute: source setup, simulation, all-unit export, archive count replay; excludes imports, staging and copyout",
            profiler="stdlib.cProfile", profiler_overhead_included=True,
            cumulative_function_times_are_nonadditive=True,
            independent_cost_comparison_qualified=False,
            production_episodes=0, functions=functions, failure=failure))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse only fully validated committed source episodes.")
    parser.add_argument("--profile-output", type=Path,
                        help="Optional cProfile/cost JSON outside source output; diagnostic timings include profiler overhead.")
    args = parser.parse_args()
    if args.profile_output is None:
        result = execute(args.source, args.config, args.output, resume=args.resume)
    else:
        result = profiled_execute(args.source, args.config, args.output, args.profile_output,
                                  resume=args.resume)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

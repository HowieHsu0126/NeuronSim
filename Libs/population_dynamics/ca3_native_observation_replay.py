"""Bind observed CA3 retrieval to native counters and unchanged event multisets.

This is a separate observation-equivalence check. It never relaxes the
exact-binary fresh-process replay contract. Native counters are compared with
independently binned AER records at every (millisecond, global neuron), including
duplicates and silent entries. Current tensors are streamed one interval at a
time. Source measurement does not qualify pattern completion or production.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import BadZipFile

import numpy as np

from . import ca3_native_observation as observer
from .ca3_retrieval_readouts import _bound_inputs, DURATION_MS, source_identity
from .carlsim_event_order import compare_events
from .carlsim_records import spike_blocks
from .carlsim_retrieval_replay import INPUT_FIELDS, SOURCE_FIELDS, _clock, _digest
from .provenance import sha256_file

PINNED_SOURCE_ROOT = Path(__file__).resolve().parents[1]/"external/carlsim_ca3"
REBUILT_ARTIFACTS = ("bin/ca3_retrieval", "lib/libcarlsim.so.6.0.0",
                    "lib/libcarlsim-stopwatch.so")
SHARED_RUNTIME = "lib/libcurand.so.10"
OBSERVER_ADAPTER_PATH = "Libs/population_dynamics/ca3_native_observation.py"
MEMORY_ADAPTER_PATH = "Libs/population_dynamics/ca3_conductance_memory.py"
GPU_SOURCE_FILE = "carlsim/kernel/src/gpu_module/snn_gpu_module.cu"


@dataclass
class BoundRun:
    root: Path
    build: dict
    native: dict
    recording: dict
    groups: tuple
    saved: dict
    hashes: dict[str, str]
    manifest: dict[str, str]
    overlay_source_hashes: dict[str, str]
    source_verification: dict


def same_coupled_configuration(left: dict, right: dict) -> bool:
    """Compare bound configuration content; build-local paths remain provenance."""
    required = {"sha256", "resolved"}
    return (required <= left.keys() and required <= right.keys()
            and {k: v for k, v in left.items() if k != "path"}
            == {k: v for k, v in right.items() if k != "path"})


def _json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("CA3_OBSERVATION_RECEIPT_MUST_BE_OBJECT")
    return result


def _overlay_sources() -> tuple[dict[str, bytes], dict[str, bytes]]:
    # The generated hashes are recomputed using the current, identity-bound
    # adapter and verified original bytes, not accepted from the run's claims.
    inputs = {name: (PINNED_SOURCE_ROOT/name).read_bytes() for name in observer.PINNED_FILES}
    return inputs, observer.overlay_bytes(inputs)


def _expected_overlay() -> dict[str, str]:
    _, patched = _overlay_sources()
    return {name: hashlib.sha256(value).hexdigest()
            for name, value in patched.items()}


def _verify_memory_overlay(build: dict, native: dict, manifest: dict[str, str],
                           originals: dict[str, bytes], observer_sources: dict[str, bytes],
                           *, observed: bool) -> tuple[dict[str, str], dict]:
    """Bind the memory transform after observation without rewriting either receipt.

    Historical unobserved runtimes contain the corrector and its receipt but no
    GPU source snapshot. Their proof remains receipt/implementation based; the
    new observed runtime must also supply its final compiled-source snapshot.
    """
    declared = build.get("conductance_memory_overlay")
    if declared != native.get("conductance_memory_overlay"):
        raise ValueError("CA3_OBSERVATION_MEMORY_BUILD_NATIVE_RECEIPTS_DIFFER")
    original = originals[GPU_SOURCE_FILE]
    intermediate = observer_sources.get(GPU_SOURCE_FILE, original) if observed else original
    original_hash = hashlib.sha256(original).hexdigest()
    input_hash = hashlib.sha256(intermediate).hexdigest()
    final_hash = input_hash
    artifacts = ({name: hashlib.sha256(value).hexdigest() for name, value in observer_sources.items()}
                 if observed else {})
    evidence = dict(conductance_memory_enabled=declared is not None,
                    memory_receipt_and_implementation_verified=False,
                    gpu_source_snapshot_verified=False, original_gpu_sha256=original_hash,
                    overlay_input_gpu_sha256=input_hash, final_gpu_sha256=final_hash)
    if declared is not None:
        # Legacy observer-only bundles do not require the optional corrector.
        try:
            from . import ca3_conductance_memory as memory
        except ImportError as error:
            raise ValueError("CA3_OBSERVATION_MEMORY_IMPLEMENTATION_UNAVAILABLE") from error
        if (memory.GPU_FILE != GPU_SOURCE_FILE or memory.PINNED_SHA256 != original_hash
                or not isinstance(declared, dict)):
            raise ValueError("CA3_OBSERVATION_MEMORY_PINNED_SOURCE_REQUIRED")
        final_hash = hashlib.sha256(memory.transform_source(intermediate)).hexdigest()
        expected = dict(schema_version=1, overlay_id=memory.OVERLAY_ID,
            original_sha256={GPU_SOURCE_FILE: original_hash},
            input_sha256={GPU_SOURCE_FILE: input_hash},
            overlay_sha256={GPU_SOURCE_FILE: final_hash},
            rise_buffer_allocation=memory._RISE_FLAGS.copy(),
            initialization_correction=dict(previous_extent="syn_gPitch * syn_gLength",
                corrected_extent="syn_gPitch * numNReg", allocation_rows="numNReg",
                applies_to=[channel+"_syn_g" for channel in memory._CHANNELS],
                previously_uninitialized_rows_are_now_zeroed=True),
            kernel_arithmetic_unchanged=True, precision_unchanged=True,
            native_build_verified=False, native_execution_verified=False, native_qualification=False)
        # These memory-receipt flags are not promoted by the builder or runner;
        # their overall build/native execution states are checked separately.
        if declared != expected:
            raise ValueError("CA3_OBSERVATION_EXACT_MEMORY_OVERLAY_RECEIPT_REQUIRED")
        implementation_hash = sha256_file(Path(memory.__file__))
        if any(record.get("conductance_memory_implementation_sha256") != implementation_hash
               for record in (build, native)):
            raise ValueError("CA3_OBSERVATION_MEMORY_IMPLEMENTATION_IDENTITY_CHANGED")
        if manifest.get(MEMORY_ADAPTER_PATH) != implementation_hash:
            raise ValueError("CA3_OBSERVATION_MEMORY_MANIFEST_ADAPTER_IDENTITY_CHANGED")
        evidence.update(memory_receipt_and_implementation_verified=True,
                        implementation_sha256=implementation_hash, final_gpu_sha256=final_hash)
        if observed:
            artifacts[GPU_SOURCE_FILE] = final_hash
    snapshot = "observer_source/"+GPU_SOURCE_FILE
    if (observed and declared is not None) or snapshot in manifest:
        if manifest.get(snapshot) != final_hash:
            raise ValueError("CA3_OBSERVATION_MEMORY_GPU_SOURCE_IDENTITY_CHANGED")
        evidence["gpu_source_snapshot_verified"] = True
        if observed:
            artifacts[GPU_SOURCE_FILE] = final_hash
    return artifacts, evidence


def _manifest(root: Path) -> dict[str, str]:
    result = {}
    for row in _json(root/"manifest.json")["files"]:
        relative = Path(row["path"])
        name = str(relative)
        if (relative.is_absolute() or ".." in relative.parts or name in ("", ".")
                or name in result or not _digest(row["sha256"])):
            raise ValueError("CA3_OBSERVATION_MANIFEST_PATH_OR_HASH_INVALID")
        if sha256_file(root/relative) != row["sha256"]:
            raise ValueError("CA3_OBSERVATION_MANIFEST_FILE_CHANGED: "+name)
        result[name] = row["sha256"]
    required = {"build_report.json", "trainPat.csv", SHARED_RUNTIME, *REBUILT_ARTIFACTS}
    if not required <= result.keys():
        raise ValueError("CA3_OBSERVATION_RUNTIME_MANIFEST_COVERAGE_REQUIRED")
    return result


def _bind(root: Path, *, observed: bool, expected_overlay: dict[str, str],
          original_sources: dict[str, bytes], observer_sources: dict[str, bytes],
          coupled_extension: bool = False, source_realization_config: Path | None = None) -> BoundRun:
    build, native, recording, groups, saved, _, hashes = _bound_inputs(root, coupled_extension=coupled_extension, source_realization_config=source_realization_config)
    manifest = _manifest(root)
    if (manifest["build_report.json"] != hashes["build_report.json"]
            or manifest["bin/ca3_retrieval"] != native["binary_sha256"]
            or manifest["trainPat.csv"] != native["pattern_sha256"]):
        raise ValueError("CA3_OBSERVATION_RUNTIME_BUILD_BINDING_INVALID")
    if any(build.get(k) != native.get(k) for k in SOURCE_FIELDS):
        raise ValueError("CA3_OBSERVATION_NATIVE_SOURCE_DIFFERS_FROM_BUILD")
    if (type(native.get("native_pid")) is not int or native["native_pid"] <= 0
            or not isinstance(native.get("slurm_job_id"), str) or not native["slurm_job_id"]
            or _clock(native["finished_utc"]) <= _clock(native["started_utc"])):
        raise ValueError("CA3_OBSERVATION_COMPLETED_PROCESS_IDENTITY_REQUIRED")
    for name in INPUT_FIELDS:
        if not _digest(native.get(name)):
            raise ValueError("CA3_OBSERVATION_QUALIFIED_INPUT_DIGEST_REQUIRED: "+name)
    normalized = "native_run/"+source_identity(coupled_extension=coupled_extension)["input_network"]
    hashes[normalized] = sha256_file(root/normalized)
    if hashes[normalized] != native["normalized_input_sha256"]:
        raise ValueError("CA3_OBSERVATION_NORMALIZED_INPUT_CHANGED")
    final_sources, source_verification = _verify_memory_overlay(
        build, native, manifest, original_sources, observer_sources, observed=observed)

    if not observed:
        if build.get("native_observation_overlay") or native.get("native_observation_overlay"):
            raise ValueError("CA3_OBSERVATION_REFERENCE_MUST_BE_UNINSTRUMENTED")
    else:
        adapter_sha = sha256_file(Path(observer.__file__))
        for record, executed in ((build, False), (native, True)):
            overlay = record.get("native_observation_overlay", {})
            expected = dict(framework_revision=observer.REVISION,
                enabled_by=observer.ENVIRONMENT_VARIABLE, original_sha256=observer.PINNED_FILES,
                overlay_sha256=expected_overlay, current_units=observer.CURRENT_UNIT,
                counter_source="native nSpikeCnt", count_interval_ms=1,
                accumulated_count_dtype="int64",
                current_source="runtime extCurrent; GPU device-to-host or CPU direct",
                native_build_verified=True, native_execution_verified=executed)
            if any(overlay.get(key) != value for key, value in expected.items()):
                raise ValueError("CA3_OBSERVATION_EXACT_OVERLAY_RECEIPT_REQUIRED")
            if record.get("observer_implementation_sha256") != adapter_sha:
                raise ValueError("CA3_OBSERVATION_ADAPTER_IDENTITY_CHANGED")
        if manifest.get(OBSERVER_ADAPTER_PATH) != adapter_sha:
            raise ValueError("CA3_OBSERVATION_MANIFEST_ADAPTER_IDENTITY_CHANGED")
        for name, digest in final_sources.items():
            if manifest.get("observer_source/"+name) != digest:
                raise ValueError("CA3_OBSERVATION_PATCHED_SOURCE_IDENTITY_CHANGED: "+name)
        if (native.get("native_observation_stream_validated") is not True
                or native.get("observed_native_duration_ms") != DURATION_MS
                or native.get("host_request_runtime_current_equal") is not True):
            raise ValueError("CA3_OBSERVATION_COMPLETED_TRACE_RECEIPT_REQUIRED")
        trace = root/"native_observation.bin"
        hashes["native_observation.bin"] = sha256_file(trace)
        if (hashes["native_observation.bin"] != native.get("native_observation_sha256")
                or trace.stat().st_size != native.get("native_observation_bytes")):
            raise ValueError("CA3_OBSERVATION_TRACE_DIGEST_OR_SIZE_CHANGED")
        units, trace_groups = observer.read_header(trace)
        if (units != native["core_neurons"] or len(trace_groups) != len(groups)
                or any((a.group_id,a.name,a.start,a.stop) !=
                       (b.group_id,b.name,b.start,b.stop) for a,b in zip(trace_groups,groups))):
            raise ValueError("CA3_OBSERVATION_TRACE_GROUP_MAP_CHANGED")
        # The pinned author configuration assigns every regular group to GPU0.
        if any(g.gpu != 1 or g.net_id != 0 for g in trace_groups):
            raise ValueError("CA3_OBSERVATION_SOURCE_GPU_GROUP_MAP_CHANGED")
        local = sorted(trace_groups, key=lambda g: g.local_start)
        if (local[0].local_start != 0 or local[-1].local_stop != units
                or any(a.local_stop != b.local_start for a,b in zip(local,local[1:]))):
            raise ValueError("CA3_OBSERVATION_LOCAL_GROUP_COVERAGE_CHANGED")
    return BoundRun(root,build,native,recording,tuple(groups),saved,hashes,manifest,
                    final_sources,source_verification)


def _events(run: BoundRun, *, retain: bool) -> tuple[np.ndarray | None, dict]:
    rows = run.recording["files"]
    files = {row["group"]:row for row in rows}
    if len(files) != len(rows) or set(files) != {g.name for g in run.groups}:
        raise ValueError("CA3_OBSERVATION_EVENT_FILE_COVERAGE_CHANGED")
    retained = []
    total = 0
    for group in run.groups:
        row = files[group.name]
        relative = "native_run/results/spk_"+group.name+".dat"
        path = run.root/relative
        if row["relative_path"] != relative or sha256_file(path) != row["sha256"]:
            raise ValueError("CA3_OBSERVATION_BOUND_EVENT_FILE_CHANGED: "+group.name)
        run.hashes[relative] = row["sha256"]
        unit_counts = np.zeros(group.units,dtype=np.int64)
        per_ms = np.zeros(DURATION_MS,dtype=np.int64)
        count, first, last = 0, None, None
        for block in spike_blocks(path,units=group.units,stop_ms=DURATION_MS):
            if not len(block):
                continue
            np.add.at(unit_counts,block[:,1],1)
            np.add.at(per_ms,block[:,0],1)
            count += len(block)
            first = int(block[:,0].min()) if first is None else min(first,int(block[:,0].min()))
            last = int(block[:,0].max()) if last is None else max(last,int(block[:,0].max()))
            if retain:
                global_rows = block.astype(np.int64)
                global_rows[:,1] += group.start
                retained.append(global_rows)
        if ((count,first,last) != (row["records"],row["first_event_ms"],row["last_event_ms"])
                or not np.array_equal(unit_counts,run.saved["unit_counts"][group.start:group.stop])
                or not np.array_equal(per_ms,run.saved["population_counts_per_ms"][:,group.group_id])):
            raise ValueError("CA3_OBSERVATION_RAW_DECODER_COUNTS_DIFFER: "+group.name)
        total += count
    if (total != run.recording["records"] or total != int(run.saved["unit_counts"].sum())
            or total != int(run.saved["population_counts_per_ms"].sum())):
        raise ValueError("CA3_OBSERVATION_EVENT_DENOMINATOR_CHANGED")
    events = (np.concatenate(retained) if retained else np.empty((0,2),dtype=np.int64)) if retain else None
    return events, files


def _compare_counters(trace: Path, events: np.ndarray, units: int, result: dict) -> None:
    ordered = events[np.argsort(events[:,0],kind="stable")]
    per_ms = np.bincount(ordered[:,0],minlength=DURATION_MS)
    offsets = np.concatenate(([0],np.cumsum(per_ms,dtype=np.int64)))
    tested = 0
    for interval in observer.iter_intervals(trace,expected_stop_ms=DURATION_MS):
        t = interval.start_ms
        expected = np.bincount(ordered[offsets[t]:offsets[t+1],1],minlength=units)
        if interval.native_counts.shape != (units,) or not np.array_equal(expected,interval.native_counts):
            if interval.native_counts.shape == (units,):
                differing = np.flatnonzero(expected != interval.native_counts)
                result["first_counter_mismatch"] = dict(start_ms=t,stop_ms=t+1,
                    differing_units=int(len(differing)),
                    examples=[dict(global_unit_id=int(i),aer_count=int(expected[i]),
                                   native_count=int(interval.native_counts[i])) for i in differing[:8]])
            raise ValueError("CA3_OBSERVATION_PER_UNIT_MILLISECOND_COUNTER_MISMATCH")
        tested += units
    result["unit_millisecond_counts_compared"] = tested
    result["native_intervals_compared"] = DURATION_MS


def _unchanged(run: BoundRun) -> None:
    for name, digest in {**run.hashes, **run.manifest}.items():
        if sha256_file(run.root/name) != digest:
            raise ValueError("CA3_OBSERVATION_INPUT_CHANGED_DURING_REPLAY: "+name)


def compare(original: Path, observed: Path, output: Path, *, coupled_extension: bool = False,
            source_realization_config: Path | None = None) -> dict:
    """Write a failure receipt before propagating any failed verification."""
    original,observed,output = map(Path,(original,observed,output))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    family = "COUPLED_EXTENSION" if coupled_extension else "NATIVE"
    result = dict(status="FAILED_"+family+"_CA3_OBSERVATION_REPLAY",
        original=str(original),observed=str(observed),duration_ms=DURATION_MS,groups=[],
        implementation_sha256=sha256_file(Path(__file__)),
        helper_implementation_sha256={name:sha256_file(Path(__file__).with_name(name+".py")) for name in
            ("ca3_native_observation","ca3_retrieval_readouts","carlsim_records","carlsim_event_order","carlsim_retrieval_replay")},
        native_internal_spike_counter_verified=False,actual_current_trace_verified=False,
        terminal_native_clock_independently_verified=False,
        observation_source_execution_equivalence_qualified=False,
        time_or_id_tolerance=0,multiplicity_preserved=True,
        exact_binary_fresh_replay_contract_unchanged=True,
        current_units=observer.CURRENT_UNIT.copy(),full_state_restart_available=False,
        complete_formation_retrieval=False,pattern_completion_qualified=False,
        scientific_fidelity_qualified=False,full_scale_qualified=False,production_episodes=0,
        upstream_receipt_validation="execution-bound hashes; upstream scientific results are not requalified")
    try:
        if (original.resolve()==observed.resolve()
                or (original/"native_run").resolve()==(observed/"native_run").resolve()):
            raise ValueError("CA3_OBSERVATION_DISTINCT_NATIVE_RUNTIMES_REQUIRED")
        originals, observer_sources = _overlay_sources()
        expected_overlay = {name: hashlib.sha256(value).hexdigest() for name, value in observer_sources.items()}
        left = _bind(original,observed=False,expected_overlay=expected_overlay,
                     original_sources=originals,observer_sources=observer_sources,coupled_extension=coupled_extension,
                     source_realization_config=source_realization_config)
        right = _bind(observed,observed=True,expected_overlay=expected_overlay,
                      original_sources=originals,observer_sources=observer_sources,coupled_extension=coupled_extension,
                      source_realization_config=source_realization_config)
        if (left.source_verification["conductance_memory_enabled"] !=
                right.source_verification["conductance_memory_enabled"]):
            raise ValueError("CA3_OBSERVATION_MEMORY_OVERLAY_PRESENCE_DIFFERS")
        result["source_overlay_verification"] = dict(original=left.source_verification,
                                                    observed=right.source_verification)
        if right.source_verification["conductance_memory_enabled"]:
            result["helper_implementation_sha256"]["ca3_conductance_memory"] = right.source_verification["implementation_sha256"]
        for key in (*[k for k in SOURCE_FIELDS if k!="binary_sha256"],*INPUT_FIELDS):
            if left.native.get(key) != right.native.get(key):
                raise ValueError("CA3_OBSERVATION_REFERENCE_SOURCE_OR_INPUT_CHANGED: "+key)
        if coupled_extension:
            if (left.native["coupled_extension"] != right.native["coupled_extension"]
                    or not same_coupled_configuration(left.native["coupled_extension_config"],
                                                       right.native["coupled_extension_config"])):
                raise ValueError("CA3_OBSERVATION_COUPLED_IDENTITY_DIFFERS")
            result.update(coupled_extension=right.native["coupled_extension"],
                coupled_extension_config=right.native["coupled_extension_config"],
                coupled_configuration_provenance={role: run.native["coupled_extension_config"]
                                                  for role, run in (("original", left), ("observed", right))})
        if source_realization_config is not None:
            result['source_realization'] = right.native['source_realization']
            result['realization_circuit_binding'] = right.native['realization_circuit_binding']
        if left.groups != right.groups:
            raise ValueError("CA3_OBSERVATION_REFERENCE_GROUP_MAP_CHANGED")
        if left.manifest[SHARED_RUNTIME] != right.manifest[SHARED_RUNTIME]:
            raise ValueError("CA3_OBSERVATION_CURAND_RUNTIME_CHANGED")
        if _clock(right.native["started_utc"]) < _clock(left.native["finished_utc"]):
            raise ValueError("CA3_OBSERVATION_SUBSEQUENT_NATIVE_PROCESS_REQUIRED")
        result.update(original_report_sha256=left.hashes["native_report.json"],
            original_recording_sha256=left.hashes["observations/recording.json"],
            observed_report_sha256=right.hashes["native_report.json"],
            normalized_input_sha256=right.native["normalized_input_sha256"],
            native_observation_sha256=right.hashes["native_observation.bin"],
            original_framework_sha256=observer.PINNED_FILES.copy(),
            observer_intermediate_sha256=expected_overlay,
            bound_overlay_sha256=right.overlay_source_hashes,units=right.native["core_neurons"],
            rebuilt_artifacts={name:dict(original=left.manifest[name],observed=right.manifest[name])
                               for name in REBUILT_ARTIFACTS},
            native_processes=[{k:run.native[k] for k in ("slurm_job_id","native_pid","started_utc","finished_utc")}
                              for run in (left,right)],
            runtime_manifests_compared_by_scientific_role=True)
        _, left_files = _events(left,retain=False)
        events, right_files = _events(right,retain=True)
        for group in left.groups:
            relative = left_files[group.name]["relative_path"]
            entry = compare_events(original/relative,observed/relative,
                                   units=group.units,duration_ms=DURATION_MS)
            if (entry["original_rows"] != left_files[group.name]["records"]
                    or entry["repeated_rows"] != right_files[group.name]["records"]):
                raise ValueError("CA3_OBSERVATION_EVENT_COMPARISON_DENOMINATOR_CHANGED")
            result["groups"].append(dict(group=group.name,**entry))
        result["event_multisets_equal"] = all(g["header_bytes_equal"] and g["event_multiset_equal"] for g in result["groups"])
        result["event_order_equal"] = all(g["event_order_equal"] for g in result["groups"])
        result["event_file_bytes_equal"] = all(left_files[g.name]["sha256"]==right_files[g.name]["sha256"] for g in left.groups)
        result["strict_file_order_status"] = "EQUAL" if result["event_file_bytes_equal"] else "DIFFERENT"
        result["events_compared"] = int(len(events))
        if not result["event_multisets_equal"]:
            raise ValueError("CA3_OBSERVATION_REFERENCE_EVENT_MULTISET_MISMATCH")
        _compare_counters(observed/"native_observation.bin",events,right.native["core_neurons"],result)
        _unchanged(left)
        _unchanged(right)
        result.update(status="PASS_"+family+"_CA3_OBSERVATION_REPLAY",
            native_internal_spike_counter_verified=True,actual_current_trace_verified=True,
            terminal_native_clock_independently_verified=True,
            observation_source_execution_equivalence_qualified=True)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,EOFError,BadZipFile) as error:
        result.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        with output.open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
        raise
    result["elapsed_seconds"] = time.perf_counter()-started
    with output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original","observed","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--coupled-extension",action="store_true")
    parser.add_argument("--source-realization-config", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.original,args.observed,args.output,coupled_extension=args.coupled_extension,
                         source_realization_config=args.source_realization_config)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,EOFError,BadZipFile) as error:
        print(str(error),file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

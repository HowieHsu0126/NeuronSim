"""Validate staged original per-group covariance outputs and join the full domain.

Input schema pfc_history_covariance_intake_v1 has groups entries containing
 group_key, report:{path,bytes,sha256}, tables:{filename:{path,bytes,sha256}}.
All files are local, manifest-driven stages of the original immutable results.
Network staging belongs to the existing Merlin HDFS receive path. This module
never measures covariance, simulates, infers uncertainty, or qualifies sources.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re

SCHEMA = "pfc_history_covariance_intake_v1"
SOURCE_SCHEMA = "pfc_matched1ms_history_coincidences_v1"
SOURCE_STATUS = "COMPLETE_REAL3_HISTORY_MATCHED1MS_UNIT_COVARIANCE"
VARIANCE_NAME = "matched_history_variance1ms.csv"
PAIR_NAME = "history_pair_unit_bin_coincidences.csv"
PHASE_EDGES = (0, 8000, 24080, 32000, 36000, 192000, 196000, 320000, 400000)
POPULATIONS = {"excitatory": 80000, "inhibitory": 20000, "all_units": 100000}
VARIANCE_FIELDS = (
    "phase_index", "start_tick", "stop_tick", "population", "units", "independent_histories",
    "time_bins", "bin_ticks", "bin_ms", "complete_history_events",
    "unordered_history_pair_unit_bin_coincidences", "population_mean_variance_rate_Hz2",
    "diagonal_contribution_rate_Hz2", "ordered_offdiagonal_contribution_rate_Hz2", "sample_axis",
)
PAIR_FIELDS = ("left_history_index", "right_history_index", "phase_index",
               "excitatory_unit_bin_coincidences", "inhibitory_unit_bin_coincidences")
OWNERS = ("circuit_index", "condition_index")
REQUIRED_GROUPS = {30 * circuit + condition for circuit in range(100) for condition in range(10)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value, field):
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None,
            "TABLE_INTEGER_INVALID:" + field)
    return int(value)


def number(value, field):
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError("TABLE_NUMBER_INVALID:" + field) from None
    require(result.is_finite(), "TABLE_NONFINITE_VALUE:" + field)
    return result


def bound_file(binding, base):
    require(isinstance(binding, dict) and isinstance(binding.get("path"), str)
            and type(binding.get("bytes")) is int and binding["bytes"] >= 0
            and valid_hash(binding.get("sha256")), "ORIGINAL_FILE_BINDING_REQUIRED")
    require("://" not in binding["path"], "LOCAL_VERIFIED_STAGE_REQUIRED")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    data = path.read_bytes()
    require(len(data) == binding["bytes"] and sha256(data) == binding["sha256"],
            "ORIGINAL_FILE_BINDING_CHANGED:" + str(path))
    return data, dict(path=str(path), bytes=len(data), sha256=binding["sha256"])


def validate_report(report, key):
    require(isinstance(report, dict), "SINGLE_GROUP_SOURCE_REPORT_REQUIRED")
    circuit, condition = divmod(key, 30)
    require(report.get("schema") == SOURCE_SCHEMA and report.get("status") == SOURCE_STATUS,
            "SINGLE_GROUP_SOURCE_REPORT_REQUIRED")
    # Older single reports omit both owner fields. Every source identity below
    # must still declare the exact circuit and condition; partial owners fail.
    if "circuit_index" in report or "condition_index" in report:
        require(type(report.get("circuit_index")) is int and report["circuit_index"] == circuit
                and type(report.get("condition_index")) is int and report["condition_index"] == condition,
                "GROUP_OWNER_IDENTITY_CHANGED")
    for field, expected in (("core_units", 100000), ("duration_ticks", 400000),
                            ("observation_bin_ticks", 80), ("phase_count", 8),
                            ("complete_source_episodes_reused", 3), ("independent_histories", 3),
                            ("history_pairs", 3), ("source_simulations", 0), ("production_episodes", 0)):
        require(type(report.get(field)) is int and report[field] == expected,
                "GROUP_SOURCE_CONTRACT_CHANGED:" + field)
    require(report.get("history_pairs_are_independent_replicates") is False,
            "HISTORY_PAIRS_NOT_INDEPENDENT_UNITS")
    require(all(report.get(field) is False for field in
                ("source_qualification_granted", "scaling_qualified", "uncertainty_coverage_qualified")),
            "DEVELOPMENT_QUALIFICATION_SCOPE_CHANGED")
    execution = report.get("execution")
    require(isinstance(execution, dict) and execution.get("backend") == "merlin"
            and isinstance(execution.get("run_id"), str) and bool(execution["run_id"])
            and isinstance(execution.get("trial_id"), str) and bool(execution["trial_id"]),
            "RESULT_EXECUTION_IDENTITY_REQUIRED")
    require(valid_hash(report.get("implementation_sha256")) and valid_hash(report.get("request_sha256")),
            "RESULT_IMPLEMENTATION_BINDING_REQUIRED")
    bindings = report.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == 3, "EXACT_THREE_SOURCE_HISTORY_BINDINGS_REQUIRED")
    initial, background = set(), set()
    for position, binding in enumerate(bindings):
        require(isinstance(binding, dict) and isinstance(binding.get("identity"), dict),
                "GROUP_HISTORY_IDENTITY_CHANGED")
        identity = binding["identity"]
        block, index = 6 + position, key + 10 * position
        prefix = f"pfc_working_memory_circuit_{circuit:03d}"
        require(type(binding.get("task_index")) is int and binding["task_index"] == index
                and identity.get("circuit_index") == circuit and type(identity.get("circuit_index")) is int
                and identity.get("block_index") == block and type(identity.get("block_index")) is int
                and identity.get("condition_index") == condition and type(identity.get("condition_index")) is int
                and identity.get("circuit_id") == prefix
                and identity.get("block_id") == prefix + f"_block_{block:03d}"
                and identity.get("episode_id") == prefix + f"_block_{block:03d}_condition_{condition:02d}"
                and identity.get("scenario_id") == "pfc_working_memory"
                and identity.get("sample_unit") == "complete_mechanistic_episode"
                and identity.get("split") == ("train" if circuit < 70 else "validation" if circuit < 80
                                             else "calibration" if circuit < 90 else "test"),
                "GROUP_HISTORY_IDENTITY_CHANGED")
        for field, values in (("initial_state_seed", initial), ("background_seed", background)):
            value = identity.get(field)
            require(type(value) is int and value >= 0, "SOURCE_STREAM_IDENTITY_INVALID:" + field)
            values.add(value)
        require(valid_hash(binding.get("raw_sha256")) and valid_hash(binding.get("source_report_sha256")),
                "ORIGINAL_SOURCE_HASH_REQUIRED")
        for field in ("retained_binary_unit_bins", "terminal_events_excluded"):
            require(type(binding.get(field)) is int and binding[field] >= 0, "SOURCE_EVENT_COUNT_INVALID:" + field)
        backend = binding.get("source_execution_backend")
        if "source_execution_backend" not in binding:
            # Legacy Merlin reports bind the run and task through their original
            # HDFS checkpoint. Validate those fields without changing evidence.
            source_run, source_trial = binding.get("source_run"), binding.get("source_trial")
            checkpoint = binding.get("checkpoint_path")
            require(isinstance(source_run, str) and re.fullmatch(r"[0-9a-f]{16}", source_run) is not None
                    and isinstance(source_trial, str) and re.fullmatch(r"[0-9]+", source_trial) is not None
                    and int(source_trial) > 0 and valid_hash(binding.get("source_bundle_sha256"))
                    and valid_hash(binding.get("manifest_sha256")) and isinstance(checkpoint, str)
                    and re.fullmatch(r"hdfs://[^/\s]+/(?:[^/\s]+/)*checkpoints/[0-9]{8}/"
                                     + re.escape(source_run) + f"/pfc/task_{index:04d}", checkpoint) is not None,
                    "ORIGINAL_LEGACY_MERLIN_SOURCE_BINDING_REQUIRED")
            backend = "merlin"
        require(backend in ("merlin", "slurm"), "ORIGINAL_SOURCE_BACKEND_INVALID")
        require(isinstance(binding.get("source_run"), str) and bool(binding["source_run"]),
                "ORIGINAL_SOURCE_EXECUTION_REQUIRED")
        if backend == "merlin":
            require(isinstance(binding.get("source_trial"), str) and bool(binding["source_trial"])
                    and valid_hash(binding.get("source_bundle_sha256")) and valid_hash(binding.get("manifest_sha256")),
                    "ORIGINAL_MERLIN_SOURCE_BINDING_REQUIRED")
        else:
            require(binding["source_run"].isdecimal() and binding.get("source_trial") is None
                    and binding.get("source_bundle_sha256") is None and binding.get("manifest_sha256") is None
                    and valid_hash(binding.get("source_delivery_sha256"))
                    and valid_hash(binding.get("transfer_receipt_sha256"))
                    and isinstance(binding.get("transfer_run"), str) and bool(binding["transfer_run"])
                    and isinstance(binding.get("transfer_trial"), str) and bool(binding["transfer_trial"]),
                    "ORIGINAL_SLURM_TRANSFER_BINDING_REQUIRED")
    require(len(initial) == len(background) == 3, "SOURCE_HISTORIES_NOT_DISTINCT")


def csv_rows(data, fields):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    require(reader.fieldnames == list(fields), "ORIGINAL_SINGLE_GROUP_TABLE_HEADER_REQUIRED")
    rows = list(reader)
    require(len(rows) == 24 and all(None not in row and all(value is not None for value in row.values())
                                  for row in rows), "EXACT24_GROUP_TABLE_ROWS_REQUIRED")
    return rows


def validate_tables(variance, pairs):
    seen = set()
    for row in variance:
        phase = integer(row["phase_index"], "phase_index")
        population = row["population"]
        require(phase in range(8) and population in POPULATIONS and (phase, population) not in seen,
                "VARIANCE_CELL_DUPLICATE_OR_INVALID")
        seen.add((phase, population))
        expected = {"start_tick": PHASE_EDGES[phase], "stop_tick": PHASE_EDGES[phase + 1],
                    "units": POPULATIONS[population], "independent_histories": 3,
                    "time_bins": (PHASE_EDGES[phase + 1] - PHASE_EDGES[phase]) // 80, "bin_ticks": 80}
        require(all(integer(row[field], field) == value for field, value in expected.items())
                and number(row["bin_ms"], "bin_ms") == 1
                and row["sample_axis"] == "matched_time_across_independent_histories_within_circuit_and_condition",
                "VARIANCE_PHASE_POPULATION_CLOCK_CHANGED")
        for field in ("complete_history_events", "unordered_history_pair_unit_bin_coincidences"):
            integer(row[field], field)
        for field in ("population_mean_variance_rate_Hz2", "diagonal_contribution_rate_Hz2"):
            require(number(row[field], field) >= 0, "NEGATIVE_VARIANCE_COMPONENT:" + field)
        number(row["ordered_offdiagonal_contribution_rate_Hz2"], "ordered_offdiagonal_contribution_rate_Hz2")
    seen = set()
    for row in pairs:
        phase = integer(row["phase_index"], "phase_index")
        left, right = integer(row["left_history_index"], "left_history_index"), integer(row["right_history_index"], "right_history_index")
        require(phase in range(8) and (left, right) in ((0, 1), (0, 2), (1, 2))
                and (phase, left, right) not in seen, "HISTORY_PAIR_CELL_DUPLICATE_OR_INVALID")
        seen.add((phase, left, right))
        integer(row["excitatory_unit_bin_coincidences"], "excitatory_unit_bin_coincidences")
        integer(row["inhibitory_unit_bin_coincidences"], "inhibitory_unit_bin_coincidences")


def validate_single(source, base=None):
    """Development read API: decode one real group without writing any outputs."""
    base = Path(base or ".").resolve()
    require(isinstance(source, dict), "INTAKE_GROUP_SOURCE_INVALID")
    key = source.get("group_key")
    require(type(key) is int and key in REQUIRED_GROUPS, "COVARIANCE_GROUP_KEY_INVALID")
    data, report_binding = bound_file(source.get("report"), base)
    report = json.loads(data)
    validate_report(report, key)
    tables = source.get("tables")
    require(isinstance(tables, dict) and set(tables) == {VARIANCE_NAME, PAIR_NAME}, "EXACT_TWO_ORIGINAL_TABLES_REQUIRED")
    outputs = report.get("outputs")
    require(isinstance(outputs, dict), "REPORT_TABLE_BINDING_CHANGED")
    decoded, table_bindings = {}, {}
    for name, fields in ((VARIANCE_NAME, VARIANCE_FIELDS), (PAIR_NAME, PAIR_FIELDS)):
        expected = outputs.get(name)
        require(isinstance(expected, dict) and isinstance(tables[name], dict)
                and expected.get("bytes") == tables[name].get("bytes")
                and expected.get("sha256") == tables[name].get("sha256"), "REPORT_TABLE_BINDING_CHANGED:" + name)
        data, binding = bound_file(tables[name], base)
        decoded[name], table_bindings[name] = csv_rows(data, fields), binding
    validate_tables(decoded[VARIANCE_NAME], decoded[PAIR_NAME])
    circuit, condition = divmod(key, 30)
    owned = lambda row: dict(circuit_index=str(circuit), condition_index=str(condition), **row)
    return dict(group_key=key, report=report,
                variance_rows=[owned(row) for row in decoded[VARIANCE_NAME]],
                pair_rows=[owned(row) for row in decoded[PAIR_NAME]],
                source=dict(group_key=key, report=report_binding, tables=table_bindings,
                            execution=report["execution"], implementation_sha256=report["implementation_sha256"],
                            bindings=report["bindings"]))


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    require(not output.exists(), "IMMUTABLE_INTAKE_OUTPUT_EXISTS")
    data = manifest_path.read_bytes()
    manifest = json.loads(data)
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA
            and isinstance(manifest.get("groups"), list), "INTAKE_MANIFEST_SCHEMA_INVALID")
    entries = manifest["groups"]
    require(all(isinstance(entry, dict) for entry in entries), "INTAKE_GROUP_SOURCE_INVALID")
    keys = [entry.get("group_key") for entry in entries]
    require(all(type(key) is int and key in REQUIRED_GROUPS for key in keys), "COVARIANCE_GROUP_KEY_INVALID")
    require(len(keys) == len(set(keys)), "DUPLICATE_COVARIANCE_GROUP_SOURCE")
    require(set(keys) == REQUIRED_GROUPS, "MISSING_COVARIANCE_GROUPS:required1000:actual" + str(len(keys)))
    variance, pairs, sources, histories = [], [], [], {}
    for entry in sorted(entries, key=lambda value: value["group_key"]):
        group = validate_single(entry, manifest_path.parent)
        for binding in group["report"]["bindings"]:
            identity = binding["identity"]
            key = identity["circuit_index"], identity["block_index"]
            streams = identity["initial_state_seed"], identity["background_seed"]
            require(key not in histories or histories[key] == streams, "PAIRED_CONDITION_SOURCE_STREAMS_CHANGED")
            histories[key] = streams
        variance.extend(group["variance_rows"])
        pairs.extend(group["pair_rows"])
        sources.append(group["source"])
    require(len(variance) == len(pairs) == 24000 and len(histories) == 300,
            "COMPLETE_DOMAIN_TABLE_DENOMINATOR_CHANGED")
    output.mkdir(parents=True)
    for name, fields, rows in ((VARIANCE_NAME, OWNERS + VARIANCE_FIELDS, variance),
                              (PAIR_NAME, OWNERS + PAIR_FIELDS, pairs)):
        with (output / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    report = dict(schema=SCHEMA, status="COMPLETE1000_DESCRIPTIVE_MATCHED1MS_COVARIANCE_INTAKE",
        completed_groups=1000, required_groups=1000, completed_group_keys=sorted(REQUIRED_GROUPS),
        circuit_instances=100, conditions_per_circuit=10, block_indices=[6, 7, 8], independent_histories_per_circuit=3,
        history_pairs_per_group=3, history_pairs_are_independent_replicates=False, total_history_units=300,
        complete_source_episodes_reused=3000, core_units=100000, duration_ticks=400000, observation_bin_ticks=80,
        phase_count=8, manifest=dict(path=str(manifest_path), bytes=len(data), sha256=sha256(data)),
        sources=sources, outputs={name: dict(bytes=(output / name).stat().st_size,
                    sha256=sha256((output / name).read_bytes()), rows=24000) for name in (VARIANCE_NAME, PAIR_NAME)},
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scientific_parameter_domain_qualified=False, scaling_qualified=False, uncertainty_coverage_qualified=False,
        measurement="intake of original matched-time1ms results; signed values preserved; no covariance recomputation or inference")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.manifest, args.output)
    print(json.dumps({key: report[key] for key in ("status", "completed_groups", "required_groups")}))


if __name__ == "__main__":
    main()

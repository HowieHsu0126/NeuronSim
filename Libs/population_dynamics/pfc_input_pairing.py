"""Independently verify paired PFC initial states and recorded native inputs.

This reads the archived native observation validation. It performs no simulation,
imports no Brian code, and makes no statistical independence or neural-fidelity
claim. Run on a compute node: captured count arrays are read in bounded chunks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

from .provenance import freeze_json, sha256_file

_CHUNK_ROWS = 256


def _relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or str(path) in ("", "."):
        raise ValueError("PFC_PAIRING_ARTIFACT_PATH_INVALID:" + str(name))
    return path.as_posix()


def _under(prefix, name):
    return prefix + "/" + _relative(name)


def _resolve_cases(config):
    if (config.get("random_stream_policy") != "independent_initial_external"
            or config.get("scope") != "source_reproduction" or "source_rng_seed" in config):
        raise ValueError("PFC_PAIRING_INDEPENDENT_SOURCE_DESIGN_REQUIRED")
    resolved = []
    for row in config["cases"]:
        initial = row.get("initial_rng_seed", config.get("initial_rng_seed"))
        external = row.get("external_noise_seed", config.get("external_noise_seed"))
        if ("source_rng_seed" in row or any(type(seed) is not int or not 0 <= seed < 2**32
                                           for seed in (initial, external))):
            raise ValueError("PFC_PAIRING_SEED_DECLARATION_INVALID")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or _relative(case_id) != case_id or len(Path(case_id).parts) != 1:
            raise ValueError("PFC_PAIRING_CASE_ID_INVALID")
        cue = row.get("stimulus_strength_nA", config["stimulus_strength_nA"])
        distractor = row["distractor_strength_nA"]
        if any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
               for value in (cue, distractor)):
            raise ValueError("PFC_PAIRING_CONTROL_INVALID")
        resolved.append(dict(row, initial_rng_seed=initial, external_noise_seed=external,
                             source_rng_seed=external, random_stream_policy="independent_initial_external",
                             stimulus_strength_nA=float(cue), distractor_strength_nA=float(distractor),
                             block_id=f"pfc_initial_{initial}_external_{external}"))
    return resolved


@contextmanager
def _array(path, shape, dtype):
    values = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        if not isinstance(values, np.ndarray) or values.shape != shape or values.dtype != np.dtype(dtype):
            raise ValueError("PFC_PAIRING_ARRAY_SCHEMA_CHANGED:" + str(path))
        yield values
    finally:
        if isinstance(values, np.memmap):
            values._mmap.close()
        elif hasattr(values, "close"):
            values.close()


def _summarize_array(path, shape, *, initial=False):
    total = 0
    with _array(path, shape, np.float64 if initial else np.uint16) as values:
        for start in range(0, shape[0], _CHUNK_ROWS):
            block = values[start:start + _CHUNK_ROWS]
            if initial:
                if not np.isfinite(block).all() or np.any(block < -60.0) or np.any(block >= -50.0):
                    raise ValueError("PFC_PAIRING_INITIAL_VOLTAGE_SUPPORT_INVALID")
            else:
                if np.any(block > 1000):
                    raise ValueError("PFC_PAIRING_BINOMIAL_SUPPORT_INVALID")
                total += int(block.sum(dtype=np.uint64))
    return total if not initial else None


def _difference(left, right, shape, dtype):
    mismatches, all_equal = 0, True
    with _array(left, shape, dtype) as a, _array(right, shape, dtype) as b:
        for start in range(0, shape[0], _CHUNK_ROWS):
            x, y = a[start:start + _CHUNK_ROWS], b[start:start + _CHUNK_ROWS]
            equal = np.array_equal(x, y)
            all_equal = all_equal and equal
            if not equal:
                mismatches += int(np.count_nonzero(x != y))
    return dict(array_equal=bool(all_equal), mismatched_values=mismatches,
                values_compared=int(np.prod(shape, dtype=np.int64)))


def _run(validation, report):
    delivery_path = validation / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    report["delivery_sha256"] = sha256_file(delivery_path)
    if delivery["status"] != "ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY":
        raise ValueError("PFC_PAIRING_ARCHIVED_NATIVE_VALIDATION_REQUIRED")
    declared = {}
    for row in delivery["files"]:
        name = _relative(row["path"])
        if name in declared:
            raise ValueError("PFC_PAIRING_DUPLICATE_DELIVERY_PATH:" + name)
        declared[name] = row
    consumed = {}

    def consume(name, expected=None):
        name = _relative(name)
        row = declared[name]
        if expected is not None and row["sha256"] != expected:
            raise ValueError("PFC_PAIRING_DECLARED_DIGEST_MISMATCH:" + name)
        if name not in consumed:
            path = validation / name
            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError("PFC_PAIRING_ARTIFACT_CHANGED:" + name)
            consumed[name] = dict(path=name, sha256=row["sha256"], bytes=row["bytes"])
            report["consumed_files"].append(consumed[name])
        return validation / name

    def load(name, expected=None):
        return json.loads(consume(name, expected).read_text())

    comparison = load("comparison/comparison.json")
    if comparison["status"] != "PASS_NATIVE_INPUT_OBSERVATION_PARITY" or comparison != delivery["comparison"]:
        raise ValueError("PFC_PAIRING_SOURCE_OBSERVATION_PARITY_REQUIRED")
    execution = load("capture/execution.json", comparison["capture_execution_sha256"])
    inputs = load("capture/inputs/report.json", execution["observation_report_sha256"])
    native = load("capture/native/report.json", execution["native_report_sha256"])
    if (execution["status"] != "RECORDED_NATIVE_PFC_INPUT_OBSERVATION" or execution["mode"] != "capture"
            or inputs["status"] != "NATIVE_SOURCE_INPUT_OBSERVATION_COMPLETE" or inputs["mode"] != "capture"
            or inputs.get("draw_capture_performed") is not True
            or native["status"] != "NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
            or native["random_stream_policy"] != "independent_initial_external"):
        raise ValueError("PFC_PAIRING_COMPLETE_CAPTURE_REQUIRED")
    for key in ("source_entry_sha256", "brian2_version", "numpy_version"):
        if inputs[key] != native[key]:
            raise ValueError("PFC_PAIRING_SOURCE_IDENTITY_MISMATCH:" + key)
    config_candidates = sorted(name for name, row in declared.items()
                               if name.startswith("Libs/configs/") and name.endswith((".yaml", ".yml"))
                               and row["sha256"] == execution["config_sha256"])
    if not config_candidates:
        raise ValueError("PFC_PAIRING_ARCHIVED_CONFIG_MISSING")
    consume(config_candidates[0], execution["config_sha256"])
    config = native["resolved_config"]
    if native["source_revision"] != config["expected_source_revision"]:
        raise ValueError("PFC_PAIRING_SOURCE_REVISION_MISMATCH")
    expected = _resolve_cases(config)
    report["cases"] = [dict(case=row, validation_status="pending") for row in expected]
    identities = [row["case_id"] for row in expected]
    observed_ids = [[row["case_id"] for row in document["cases"]] for document in (inputs, native, comparison)]
    report["denominator"]["resolved_cases"] = len(expected)
    if (len(expected) != 8 or len(set(identities)) != 8 or execution["cases"] != 8
            or any(ids != identities for ids in observed_ids)):
        raise ValueError("PFC_PAIRING_COMPLETE_EIGHT_CASE_SET_REQUIRED")
    initial_seeds = sorted({row["initial_rng_seed"] for row in expected})
    external_seeds = sorted({row["external_noise_seed"] for row in expected})
    controls = sorted({(row["stimulus_strength_nA"], row["distractor_strength_nA"]) for row in expected})
    combinations = {(row["initial_rng_seed"], row["external_noise_seed"],
                     (row["stimulus_strength_nA"], row["distractor_strength_nA"])) for row in expected}
    if (len(initial_seeds) != 2 or len(external_seeds) != 2 or len(controls) != 2
            or controls[0] != (0.0, 0.0) or controls[1][0] <= 0 or controls[1][1] != 0.0
            or combinations != set(product(initial_seeds, external_seeds, controls))):
        raise ValueError("PFC_PAIRING_COMPLETE_TWO_BY_TWO_BY_TWO_DESIGN_REQUIRED")
    sizes = {"E": config["excitatory_neurons"], "I": config["inhibitory_neurons"]}
    dt_ms, duration_ms = config["dt_ms"], config["duration_ms"]
    if (any(type(value) is not int or value <= 0 for value in sizes.values())
            or any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
                   for value in (dt_ms, duration_ms))):
        raise ValueError("PFC_PAIRING_NATIVE_GEOMETRY_INVALID")
    ticks = int(round(duration_ms / dt_ms))
    if ticks <= 0 or not np.isclose(ticks * dt_ms, duration_ms, rtol=0, atol=1e-9):
        raise ValueError("PFC_PAIRING_NATIVE_CLOCK_INVALID")
    units = sum(sizes.values())
    if (comparison["captured_draw_values_recounted"] != 8 * ticks * units
            or comparison["initial_voltage_values_compared"] != 8 * units):
        raise ValueError("PFC_PAIRING_PRIOR_COMPARISON_DENOMINATOR_MISMATCH")
    report.update(source_entry_sha256=native["source_entry_sha256"], config_sha256=execution["config_sha256"],
                  source_revision=native["source_revision"], config_file=config_candidates[0],
                  design=dict(initial_rng_seeds=initial_seeds, external_noise_seeds=external_seeds,
                              controls=[dict(stimulus_strength_nA=a, distractor_strength_nA=b) for a, b in controls],
                              population_units=sizes, duration_ms=duration_ms, dt_ms=dt_ms,
                              captured_ticks=ticks, tick_interval="[0, duration_ticks)",
                              initial_voltage_units="mV", initial_voltage_order="E_then_I",
                              input_population_unit_order="ascending_zero_based_population_index"))
    report["denominator"].update(expected_count_values=8 * ticks * units, expected_initial_voltage_values=8 * units)
    report["gates"]["complete_design_coverage"] = True
    records = []
    for position, case in enumerate(expected):
        mapping = report["cases"][position]
        observed, native_case, prior = inputs["cases"][position], native["cases"][position], comparison["cases"][position]
        receipt_name = _under("capture/inputs", observed["receipt_file"])
        receipt = load(receipt_name, observed["receipt_sha256"])
        if (observed["mode"] != "capture" or receipt["mode"] != "capture"
                or receipt["status"] != "NATIVE_INPUT_OBSERVATION_CASE_COMPLETE"
                or receipt["case_id"] != case["case_id"] or receipt["case"] != case
                or receipt["source_entry_sha256"] != native["source_entry_sha256"]
                or receipt["source_population_archive_order"] != ["E", "I"]
                or receipt["source_provider_construction_order"] != ["I", "E"]
                or any(prior[key] is not True for key in ("terminal_states_equal", "numpy_rng_equal", "python_rng_equal"))):
            raise ValueError("PFC_PAIRING_NATIVE_CASE_BINDING_CHANGED:" + case["case_id"])
        for key in ("initial_rng_seed", "external_noise_seed", "source_rng_seed", "stimulus_strength_nA", "distractor_strength_nA"):
            if native_case[key] != case[key]:
                raise ValueError("PFC_PAIRING_NATIVE_PARAMETER_CHANGED:" + case["case_id"] + ":" + key)
        if native_case["core_units"] != units or native_case["source_rng_seed_role"] != "external_noise":
            raise ValueError("PFC_PAIRING_NATIVE_UNIT_OR_STREAM_ROLE_CHANGED")
        initial_name = _under("capture/native", native_case["initial_voltage_file"])
        initial_path = consume(initial_name, native_case["initial_voltage_sha256"])
        _summarize_array(initial_path, (units,), initial=True)
        mapping.update(receipt_file=receipt_name, receipt_sha256=observed["receipt_sha256"],
                       initial_voltage_file=initial_name, initial_voltage_sha256=native_case["initial_voltage_sha256"],
                       populations={})
        if [row["population"] for row in receipt["providers"]] != ["I", "E"]:
            raise ValueError("PFC_PAIRING_NATIVE_PROVIDER_SET_CHANGED")
        record = dict(case=case, initial_path=initial_path, counts={})
        for provider in receipt["providers"]:
            population = provider["population"]
            shape = (ticks, sizes[population])
            if (provider["units"] != sizes[population] or provider["counts_shape"] != list(shape)
                    or provider["expected_sample_rows"] != ticks or provider["observed_sampler_calls"] != ticks
                    or provider["dt_seconds"] != dt_ms / 1000
                    or not np.isclose(provider["terminal_clock_seconds"], duration_ms / 1000, rtol=0, atol=1e-12)
                    or provider["target_variable"] != "s_AMPA" or provider["N"] != 1000
                    or provider["rate_Hz"] != 1.4 or provider["probability"] != 1.4 * (dt_ms / 1000)
                    or provider["weight_dimensionless"] != 1.0 or provider["when"] != "synapses"
                    or provider["order"] != 0
                    or provider["source_unit_order"] != "ascending_zero_based_population_index"):
                raise ValueError("PFC_PAIRING_NATIVE_INPUT_GEOMETRY_CHANGED")
            name = _under("capture/inputs", provider["counts_file"])
            path = consume(name, provider["counts_sha256"])
            events = _summarize_array(path, shape)
            if events != provider["captured_external_events"]:
                raise ValueError("PFC_PAIRING_CAPTURE_EVENT_RECOUNT_FAILED")
            info = dict(path=name, sha256=provider["counts_sha256"], shape=list(shape), external_events=events)
            mapping["populations"][population] = info
            record["counts"][population] = dict(info, verified_path=path)
            report["denominator"]["validated_count_arrays"] += 1
            report["denominator"]["validated_count_values"] += ticks * sizes[population]
        if sum(row["external_events"] for row in mapping["populations"].values()) != prior["external_events"]:
            raise ValueError("PFC_PAIRING_PRIOR_EVENT_TOTAL_CHANGED")
        mapping["validation_status"] = "verified"
        records.append(record)
        report["denominator"]["validated_cases"] += 1
        report["denominator"]["validated_initial_voltage_values"] += units
    report["input_identity_mapping_complete"] = True

    def compare_records(kind, left, right, population=None):
        if population is None:
            a, b, shape, dtype = left["initial_path"], right["initial_path"], (units,), np.float64
            a_events = b_events = None
        else:
            a_info, b_info = left["counts"][population], right["counts"][population]
            a, b, shape, dtype = a_info["verified_path"], b_info["verified_path"], (ticks, sizes[population]), np.uint16
            a_events, b_events = a_info["external_events"], b_info["external_events"]
        row = dict(kind=kind, population=population or "E_and_I_initial_voltage",
                   left_case=left["case"]["case_id"], right_case=right["case"]["case_id"],
                   left_external_events=a_events, right_external_events=b_events,
                   **_difference(a, b, shape, dtype))
        report["comparisons"].append(row)
        return row

    for external in external_seeds:
        group = [record for record in records if record["case"]["external_noise_seed"] == external]
        for other in group[1:]:
            for population in ("E", "I"):
                compare_records("within_external_seed", group[0], other, population)
    for initial in initial_seeds:
        group = [record for record in records if record["case"]["initial_rng_seed"] == initial]
        for other in group[1:]:
            compare_records("within_initial_seed", group[0], other)
    indexed = {(r["case"]["initial_rng_seed"], r["case"]["external_noise_seed"],
                (r["case"]["stimulus_strength_nA"], r["case"]["distractor_strength_nA"])): r for r in records}
    external_distinctions = []
    for initial, control in product(initial_seeds, controls):
        left, right = [indexed[(initial, external, control)] for external in external_seeds]
        rows = [compare_records("between_external_seeds", left, right, population) for population in ("E", "I")]
        external_distinctions.append(sum(row["mismatched_values"] for row in rows) > 0)
    for external, control in product(external_seeds, controls):
        left, right = [indexed[(initial, external, control)] for initial in initial_seeds]
        compare_records("between_initial_seeds", left, right)
    for kind, expected_count in report["denominator"]["expected_comparisons"].items():
        count = sum(row["kind"] == kind for row in report["comparisons"])
        report["denominator"]["completed_comparisons"][kind] = count
        if count != expected_count:
            raise ValueError("PFC_PAIRING_COMPARISON_DENOMINATOR_INCOMPLETE:" + kind)
    report["gates"].update(
        fixed_external_seed_inputs_equal=all(r["array_equal"] for r in report["comparisons"] if r["kind"] == "within_external_seed"),
        fixed_initial_seed_voltages_equal=all(r["array_equal"] for r in report["comparisons"] if r["kind"] == "within_initial_seed"),
        distinct_external_seed_joint_realizations=all(external_distinctions),
        distinct_initial_seed_realizations=all(not r["array_equal"] for r in report["comparisons"] if r["kind"] == "between_initial_seeds"))
    report["status"] = "PASS_NATIVE_INPUT_PAIRING" if all(report["gates"].values()) else "FAIL_NATIVE_INPUT_PAIRING"


def analyze(validation: Path, output: Path):
    validation, output = Path(validation), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_NATIVE_INPUT_PAIRING_INTEGRITY", created_utc=datetime.now(timezone.utc).isoformat(),
                  validation_root=str(validation.resolve()), implementation_sha256=sha256_file(Path(__file__)),
                  scope="source randomization engineering: exact pairing and realization distinction",
                  statistical_independence_established=False, biological_independence_established=False,
                  neural_response_effect_assessed=False, source_inputs_replayed=False, new_simulations=0,
                  scientific_fidelity_qualified=False, gpu_parity_qualified=False, production_episodes=0,
                  input_identity_mapping_complete=False, consumed_files=[], cases=[], comparisons=[],
                  gates=dict(complete_design_coverage=False, fixed_external_seed_inputs_equal=False,
                             fixed_initial_seed_voltages_equal=False, distinct_external_seed_joint_realizations=False,
                             distinct_initial_seed_realizations=False),
                  denominator=dict(expected_cases=8, resolved_cases=None, validated_cases=0,
                                   expected_count_arrays=16, validated_count_arrays=0,
                                   expected_count_values=None, validated_count_values=0,
                                   expected_initial_voltage_values=None, validated_initial_voltage_values=0,
                                   expected_comparisons=dict(within_external_seed=12, within_initial_seed=6,
                                                            between_external_seeds=8, between_initial_seeds=4),
                                   completed_comparisons={}))
    try:
        _run(validation, report)
    except Exception as error:
        # Publish the failed qualification and every comparison completed before
        # the error; the CLI propagates failure through its exit status.
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        report["status"] = "FAILED_NATIVE_INPUT_PAIRING_INTEGRITY"
    for kind in report["denominator"]["expected_comparisons"]:
        report["denominator"]["completed_comparisons"][kind] = sum(
            row["kind"] == kind for row in report["comparisons"])
    with (output / "pair_comparisons.csv").open("x", encoding="utf-8", newline="") as stream:
        columns = ("kind", "population", "left_case", "right_case", "array_equal", "mismatched_values",
                   "values_compared", "left_external_events", "right_external_events")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["comparisons"])
    report["comparison_table"] = "pair_comparisons.csv"
    report["comparison_table_sha256"] = sha256_file(output / "pair_comparisons.csv")
    freeze_json(output / "pairing.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.validation, args.output)
    print(json.dumps(dict(status=report["status"], gates=report["gates"], denominator=report["denominator"],
                          failure=report.get("failure")), indent=2))
    return 0 if report["status"] == "PASS_NATIVE_INPUT_PAIRING" else 1


if __name__ == "__main__":
    sys.exit(main())

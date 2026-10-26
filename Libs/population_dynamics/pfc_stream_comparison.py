"""Compare recorded arrays from two completed PFC streamed source archives.

This is an archive-only diagnostic: differences are completed comparison
outcomes. Recorded initial voltage, final RNG state and aggregate input counts
do not establish equality of the unarchived external input streams.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .pfc_scale_analysis import (
    _source_registration, _source_header, _source_values, _source_bound_file,
)
from .provenance import sha256_file


ARTIFACTS = (
    "raw.npz", "initial_voltage.npy", "terminal_state.npz",
    "final_rng.npz", "units.npz",
)
CURRENT_ARTIFACT = "applied_current_observation.npz"
CHUNK_VALUES = 65536
COMPLETE_STATUSES = (
    "COMPLETE_RECORDED_ARRAYS_IDENTICAL",
    "COMPLETE_RECORDED_ARRAYS_DIFFERENT",
)


def _archive_root(path: Path) -> Path:
    path = Path(path)
    return (path.parent if path.name == "source_run" else path).resolve()


def _origin_table(path: Path | None) -> dict:
    if path is None:
        return {}
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("PFC_STREAM_ORIGIN_TABLE_INVALID")
    mapped = {}
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key]
            for key in ("staged", "archive", "case_id", "delivery_sha256")
        ):
            raise ValueError("PFC_STREAM_ORIGIN_ROW_INVALID")
        key = str(_archive_root(Path(row["staged"])))
        if key in mapped:
            raise ValueError("PFC_STREAM_DUPLICATE_STAGED_ORIGIN")
        mapped[key] = row
    return mapped


def _source_origin(header: dict, origins: dict | None) -> dict:
    staged = _archive_root(header["archive"])
    if origins is None:
        return dict(archive=str(staged), origin_status="DIRECT_ARCHIVE",
                    delivery_sha256=header["binding"]["delivery_sha256"])
    row = origins.get(str(staged))
    if (row is None or row["case_id"] != header["report"]["case_id"]
            or row["delivery_sha256"] != header["binding"]["delivery_sha256"]):
        raise ValueError("PFC_STREAM_STAGED_ORIGIN_BINDING_MISMATCH")
    durable = _archive_root(Path(row["archive"]))
    receipt = durable / "delivery.json"
    if not receipt.is_file():
        raise ValueError("PFC_STREAM_DURABLE_DELIVERY_MISSING")
    if sha256_file(receipt) != row["delivery_sha256"]:
        raise ValueError("PFC_STREAM_DURABLE_DELIVERY_MISMATCH")
    return dict(archive=str(durable), staged_archive=str(staged),
                origin_status="DURABLE_DELIVERY_VERIFIED",
                delivery_sha256=row["delivery_sha256"])


def _bound(header: dict, name: str) -> Path:
    return _source_bound_file(
        header["archive"], header["manifest"], "source_run/" + name,
        header["report"]["outputs"][name],
    )


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _versions(mapping: dict) -> dict:
    return {key: value for key, value in mapping.items() if "version" in key.lower()}


def _validated_run(path, config, registered, dimensions, config_sha, origins):
    status, header = _source_header(path, config, registered, config_sha)
    if header is None or status["status"] != "COMPLETE":
        raise ValueError("PFC_STREAM_COMPLETE_SOURCE_REQUIRED")
    origin = _source_origin(header, origins)
    raw, execution, _ = _source_values(header, config, registered, dimensions, config_sha)
    report = header["report"]
    observation = header["binding"]["applied_current_observation"]
    current_available = observation["status"] == "VERIFIED_ACTUAL_CURRENT"
    if not current_available and observation["status"] != "UNAVAILABLE_LEGACY_ARCHIVE":
        raise ValueError("PFC_STREAM_CURRENT_AVAILABILITY_INVALID")
    arrays = {"raw.npz": raw}
    for name in ARTIFACTS[1:]:
        source = _bound(header, name)
        arrays[name] = (
            {"initial_voltage_mV": np.load(source, allow_pickle=False)}
            if name.endswith(".npy") else _npz(source)
        )
    if current_available:
        arrays[CURRENT_ARTIFACT] = _npz(_bound(header, CURRENT_ARTIFACT))
    qualification = report["backend_qualification"]
    binding = dict(
        **origin, case_id=report["case_id"], selected_case=report["selected_case"],
        report_sha256=header["binding"]["report_sha256"],
        config_sha256=config_sha, source_entry_sha256=report["source_entry_sha256"],
        source_revision=config["expected_source_revision"],
        backend=report["backend"], producer_code=report["code"],
        backend_qualification=qualification, execution_binding=execution,
        kernel_parameters=header["binding"]["kernel_parameters"],
        versions=dict(report=_versions(report), driver=_versions(report["driver"]),
                      qualification=_versions(qualification["qualification"])),
        outputs=report["outputs"],
        source_external_event_totals=report["source_external_event_totals"],
        inputs_used=report["inputs_used"], costs=report.get("costs"),
    )
    current = dict(availability="AVAILABLE" if current_available else "UNAVAILABLE",
                   source_evidence=observation)
    return arrays, binding, current


def _validate_array(values: np.ndarray, label: str) -> None:
    if values.dtype.kind not in "biufUS":
        raise ValueError("PFC_STREAM_UNSUPPORTED_ARRAY_DTYPE:" + label)
    if values.dtype.kind in "biuf":
        flat = values.reshape(-1)
        for start in range(0, values.size, CHUNK_VALUES):
            if not np.isfinite(flat[start:start + CHUNK_VALUES]).all():
                raise ValueError("PFC_STREAM_NONFINITE_ARRAY:" + label)


def _compare_array(artifact: str, name: str, candidate, reference) -> dict:
    def description(values):
        return None if values is None else dict(
            shape=list(values.shape), dtype=values.dtype.str, values=int(values.size))
    row = dict(artifact=artifact, array=name, candidate=description(candidate),
               reference=description(reference), exact_equal=False,
               value_mismatch_count=None, max_abs_numeric_delta=None)
    for side, values in (("candidate", candidate), ("reference", reference)):
        if values is not None:
            _validate_array(values, artifact + "/" + name + "/" + side)
    if candidate is None or reference is None:
        row["status"] = "MISSING_CANDIDATE_ARRAY" if candidate is None else "MISSING_REFERENCE_ARRAY"
        return row
    row.update(shape_equal=candidate.shape == reference.shape,
               dtype_equal=candidate.dtype == reference.dtype)
    if not row["shape_equal"]:
        row["status"] = "SHAPE_MISMATCH"
        return row
    numeric = candidate.dtype.kind in "biuf" and reference.dtype.kind in "biuf"
    integers = candidate.dtype.kind in "biu" and reference.dtype.kind in "biu"
    left, right = candidate.reshape(-1), reference.reshape(-1)
    mismatches, maximum = 0, 0
    for start in range(0, candidate.size, CHUNK_VALUES):
        a, b = left[start:start + CHUNK_VALUES], right[start:start + CHUNK_VALUES]
        # Object comparison avoids lossy numeric promotion when dtypes differ.
        different = np.not_equal(a, b) if row["dtype_equal"] else np.not_equal(
            a.astype(object), b.astype(object))
        mismatches += int(np.count_nonzero(different))
        if numeric and np.any(different):
            if integers:
                delta = max(abs(int(x) - int(y)) for x, y in zip(a[different], b[different]))
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = np.max(np.abs(a[different].astype(np.longdouble)
                                          - b[different].astype(np.longdouble)))
            maximum = max(maximum, delta)
    row["value_mismatch_count"] = mismatches
    if numeric:
        delta = int(maximum) if integers else float(maximum)
        if integers or np.isfinite(delta):
            row["max_abs_numeric_delta"] = delta
        else:
            row["numeric_delta_note"] = "OUTSIDE_FINITE_JSON_FLOAT_RANGE"
    row["exact_equal"] = row["dtype_equal"] and mismatches == 0
    row["status"] = "IDENTICAL" if row["exact_equal"] else (
        "DTYPE_MISMATCH" if not row["dtype_equal"] else "VALUE_MISMATCH")
    return row


def compare(candidate: Path, reference: Path, config_path: Path, output: Path,
            *, input_origins: Path | None = None) -> dict:
    """Write output/report.json; completed differences return normally, invalid input raises."""
    output, config_path = Path(output), Path(config_path)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(
        status="FAILED", comparison_completed=False,
        scope="one_registered_source_case_recorded_array_diagnostic",
        started_utc=datetime.now(timezone.utc).isoformat(),
        requested_inputs=dict(candidate=str(candidate), reference=str(reference)),
        source_bindings={}, array_results=[],
        full_external_streams_compared=False,
        external_input_equivalence_established=False,
        initial_rng_state_separately_compared=False,
        input_check_scope="recorded initial voltages, final RNG arrays and aggregate external event totals only",
        current_action_semantics="recorded delivered currents are observations, not independent per-unit actions",
        source_fidelity_qualified=False, full_scale_qualified=False,
        scientific_fidelity_qualified=False, working_memory_qualified=False,
        production_episodes=0, new_episodes=0, new_source_simulations=0,
        source_episodes_added_to_campaign=0,
        comparison_implementation_sha256=sha256_file(Path(__file__)),
    )
    try:
        config, registered, _, dimensions = _source_registration(config_path)
        config_sha = sha256_file(config_path)
        origins = _origin_table(input_origins) if input_origins is not None else None
        report.update(config_sha256=config_sha, resolved_config=config,
                      registered_source_cases=len(registered))
        if input_origins is not None:
            report["input_origins_sha256"] = sha256_file(Path(input_origins))
        data, current = {}, {}
        for role, path in (("candidate", candidate), ("reference", reference)):
            data[role], report["source_bindings"][role], current[role] = _validated_run(
                path, config, registered, dimensions, config_sha, origins)
        a, b = report["source_bindings"]["candidate"], report["source_bindings"]["reference"]
        if a["case_id"] != b["case_id"] or a["selected_case"] != b["selected_case"]:
            raise ValueError("PFC_STREAM_CASE_MISMATCH")
        report.update(case_id=a["case_id"], source_cases_compared=1)
        artifacts = list(ARTIFACTS)
        both_current = all(row["availability"] == "AVAILABLE" for row in current.values())
        report["current_observations"] = dict(
            **current, comparison_status="COMPARED" if both_current else "UNAVAILABLE")
        if both_current:
            artifacts.append(CURRENT_ARTIFACT)
        report["compared_artifacts"] = artifacts
        report["uncompared_artifacts"] = [] if both_current else [dict(
            artifact=CURRENT_ARTIFACT, reason="not recorded in both source archives")]
        for artifact in artifacts:
            left, right = data["candidate"][artifact], data["reference"][artifact]
            for name in sorted(set(left) | set(right)):
                report["array_results"].append(
                    _compare_array(artifact, name, left.get(name), right.get(name)))
        exact = all(row["exact_equal"] for row in report["array_results"])
        report.update(
            status=COMPLETE_STATUSES[0] if exact else COMPLETE_STATUSES[1],
            comparison_completed=True, recorded_arrays_exact_equal=exact,
            arrays_compared=len(report["array_results"]),
            arrays_not_exact=sum(not row["exact_equal"] for row in report["array_results"]),
            recorded_external_event_totals=dict(
                candidate=a["source_external_event_totals"],
                reference=b["source_external_event_totals"],
                exact_equal=a["source_external_event_totals"] == b["source_external_event_totals"]),
        )
    except Exception as error:
        report.update(status="FAILED", comparison_completed=False,
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Completed candidate archive root")
    parser.add_argument("--reference", type=Path, required=True, help="Completed reference archive root")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; writes report.json")
    parser.add_argument("--input-origins", type=Path, help="Existing staged-to-durable source origin table")
    args = parser.parse_args(argv)
    report = compare(args.candidate, args.reference, args.config, args.output,
                     input_origins=args.input_origins)
    print(json.dumps(dict(status=report["status"], case_id=report["case_id"],
                          arrays_compared=report["arrays_compared"],
                          arrays_not_exact=report["arrays_not_exact"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

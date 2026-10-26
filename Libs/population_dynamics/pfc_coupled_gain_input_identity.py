"""Compare stored inputs of one six-case coupled PFC gain block.

This checks persisted input arrays and their declared metadata. It does not
reconstruct external input streams, execute a simulator, or release a scientific
gate. Unit-provenance differences are retained separately from input equality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

import numpy as np

CLOCKS = (0.025, 0.0125)
CONDITIONS = ("cue", "omission", "strong_cue")
STRENGTHS = {"cue": 0.06, "omission": 0.0, "strong_cue": 0.5}
ARRAY_FILES = (
    "initial_voltage.npy", "final_rng.npz",
    "applied_current_observation.npz", "units.npz",
)
JSON_FILES = ("applied_current_metadata.json", "unit_metadata.json")
INPUT_FILES = ARRAY_FILES + JSON_FILES
INPUT_CODE = ("pfc_native_drive", "pfc_clock_coupling", "pfc_coupled_drive")
EXCLUDED_CONFIG_FIELDS = frozenset((
    "purpose", "cases", "source_replication", "source_rng_seed",
    "Jpos_excit2excit", "sigma_weight_profile",
))
RNG_MEMBERS = frozenset(
    prefix + "__" + member
    for prefix in ("coarse", "refinement_1", "refinement_2")
    for member in ("algorithm", "key", "position", "has_gauss", "cached_gaussian")
)
DELIVERY_STATUSES = {
    "baseline": frozenset((
        "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT",
        "COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK",
    )),
    "candidate": frozenset(("COMPLETE_COUPLED_CLOCK_RECURRENT_GAIN_BLOCK",)),
}
REPORT_SCOPES = {
    "baseline": frozenset((
        "development_paired_clock_pilot", "development_paired_clock_calibration",
    )),
    "candidate": frozenset((
        "development_coupled_clock_recurrent_gain_pilot",
        "development_coupled_clock_recurrent_gain_replicates",
    )),
}
PASS_STATUS = "PASS_STORED_COUPLED_GAIN_INPUT_IDENTITY"
FAIL_STATUS = "FAILED_STORED_COUPLED_GAIN_INPUT_IDENTITY"


class IdentityError(ValueError):
    """A required archive, schema, or identity binding did not hold."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _relative_path(value):
    if not isinstance(value, str) or not value:
        raise IdentityError("NONEMPTY_RELATIVE_MANIFEST_PATH_REQUIRED")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise IdentityError("UNSAFE_MANIFEST_PATH:" + value)
    return path


def _json_object(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise IdentityError("JSON_OBJECT_REQUIRED:" + str(path))
    return result


def _manifest(delivery):
    rows = delivery.get("files")
    if not isinstance(rows, list):
        raise IdentityError("DELIVERY_FILE_LIST_REQUIRED")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise IdentityError("MANIFEST_ENTRY_OBJECT_REQUIRED")
        name = str(_relative_path(row.get("path")))
        if name in result:
            raise IdentityError("DUPLICATE_MANIFEST_PATH:" + name)
        if (type(row.get("bytes")) is not int or row["bytes"] < 0
                or not _valid_sha(row.get("sha256"))):
            raise IdentityError("MANIFEST_SIZE_AND_SHA256_REQUIRED:" + name)
        result[name] = dict(path=name, bytes=row["bytes"], sha256=row["sha256"])
    return result


def _copy_verified(root, relative, destination, binding):
    source = (root / str(_relative_path(relative))).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise IdentityError("ARCHIVE_FILE_ESCAPES_ROOT:" + relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    observed_bytes = destination.stat().st_size
    observed_sha = sha256_file(destination)
    if observed_bytes != binding["bytes"] or observed_sha != binding["sha256"]:
        raise IdentityError(
            "DELIVERY_BOUND_FILE_CHANGED:" + relative
            + ":expected=" + str(binding)
            + ":observed=" + str(dict(bytes=observed_bytes, sha256=observed_sha))
        )
    return dict(binding, verified_before_load=True)


def _case_specs(seed):
    return [
        (dt, condition, "source_cases/dt_" + str(dt).replace(".", "p")
         + "/pfc_" + str(seed) + "_" + condition + "/report.json")
        for dt in CLOCKS for condition in CONDITIONS
    ]


def _failure(context, error):
    return dict(context=context, error_type=type(error).__name__, error=str(error))


def _stage_archive(root, expected_sha, stage, role, seed):
    if not _valid_sha(expected_sha):
        raise IdentityError("EXPECTED_DELIVERY_SHA256_REQUIRED:" + role)
    delivery_source = (root / "delivery.json").resolve(strict=True)
    if not delivery_source.is_relative_to(root) or not delivery_source.is_file():
        raise IdentityError("DELIVERY_FILE_ESCAPES_ARCHIVE:" + role)
    delivery_binding = dict(
        path="delivery.json", bytes=delivery_source.stat().st_size, sha256=expected_sha,
    )
    _copy_verified(root, "delivery.json", stage / "delivery.json", delivery_binding)
    delivery = _json_object(stage / "delivery.json")
    if delivery.get("status") not in DELIVERY_STATUSES[role] or delivery.get("verified") is not True:
        raise IdentityError("COMPLETE_VERIFIED_COUPLED_BLOCK_REQUIRED:" + role)
    manifest = _manifest(delivery)
    expected_reports = {relative for _, _, relative in _case_specs(seed)}
    actual_reports = {
        name for name in manifest
        if name.startswith("source_cases/") and name.endswith("/report.json")
    }
    if actual_reports != expected_reports:
        raise IdentityError("EXACT_SIX_CASE_SET_REQUIRED:" + json.dumps(dict(
            missing=sorted(expected_reports - actual_reports),
            extra=sorted(actual_reports - expected_reports),
        )))
    info = dict(
        archive=str(root), expected_delivery_sha256=expected_sha,
        delivery_binding=delivery_binding, delivery_status=delivery["status"],
        job_id=delivery.get("job_id"), cases={},
    )
    cases = {}
    for dt, condition, relative in _case_specs(seed):
        entry = dict(dt_ms=dt, condition=condition, report=None, artifacts={}, failures=[])
        binding_record = dict(report=None, artifacts={})
        info["cases"][relative] = binding_record
        cases[relative] = entry
        try:
            binding_record["report"] = _copy_verified(
                root, relative, stage / relative, manifest[relative])
            report = _json_object(stage / relative)
            entry["report"] = report
            outputs = report.get("outputs")
            if not isinstance(outputs, dict):
                raise IdentityError("CASE_OUTPUT_BINDINGS_REQUIRED")
            parent = str(PurePosixPath(relative).parent)
            for name in INPUT_FILES:
                try:
                    declared = outputs.get(name)
                    full_name = parent + "/" + name
                    delivered = manifest.get(full_name)
                    if (not isinstance(declared, dict) or delivered is None
                            or declared.get("bytes") != delivered["bytes"]
                            or declared.get("sha256") != delivered["sha256"]):
                        raise IdentityError("CASE_AND_DELIVERY_INPUT_BINDINGS_DIFFER:" + name)
                    target = stage / full_name
                    binding_record["artifacts"][name] = _copy_verified(
                        root, full_name, target, delivered)
                    entry["artifacts"][name] = target
                except Exception as error:
                    entry["failures"].append(_failure(role + ":" + full_name, error))
        except Exception as error:
            entry["failures"].append(_failure(role + ":" + relative, error))
    return info, cases


def json_differences(baseline, candidate, path=""):
    """Keep every differing JSON field, including missing versus explicit null."""
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        rows = []
        for key in sorted(set(baseline) | set(candidate)):
            here = path + "." + key if path else key
            if key not in baseline or key not in candidate:
                rows.append(dict(
                    path=here, baseline_present=key in baseline,
                    candidate_present=key in candidate,
                    baseline=baseline.get(key), candidate=candidate.get(key),
                ))
            else:
                rows.extend(json_differences(baseline[key], candidate[key], here))
        return rows
    if baseline == candidate:
        return []
    return [dict(path=path, baseline=baseline, candidate=candidate)]


def _add_check(checks, name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))


def _metadata_checks(before, after, seed, dt, condition):
    checks = []
    ticks = int(round(5000.0 / dt))
    refinement = [95411 + 2 * (seed - 95001), 95412 + 2 * (seed - 95001)]
    case_id = "pfc_" + str(seed) + "_" + condition
    expected = dict(
        status="COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE",
        source_execution_complete=True, case_id=case_id, condition=condition,
        source_rng_seed=seed, refinement_seeds=refinement, dt_ms=dt,
        coarse_dt_ms=0.05, duration_ms=5000.0, coupled_core_units=100000,
        completed_duration_ticks=ticks, native_fine_seed_sequence_preserved=False,
    )
    for role, report in (("baseline", before), ("candidate", after)):
        for name, value in expected.items():
            _add_check(checks, role + "." + name, report.get(name) == value,
                       expected=value, observed=report.get(name))
        _add_check(checks, role + ".scope", report.get("scope") in REPORT_SCOPES[role],
                   observed=report.get("scope"))
        config = report.get("resolved_config")
        config = config if isinstance(config, dict) else {}
        fixed = dict(scope="source_reproduction", excitatory_neurons=80000,
                     inhibitory_neurons=20000, dt_ms=dt, duration_ms=5000.0)
        for name, value in fixed.items():
            _add_check(checks, role + ".resolved_config." + name,
                       config.get(name) == value, expected=value, observed=config.get(name))
        expected_kernel = dict(sigma_weight_profile=20.0,
                               Jpos_excit2excit=1.6 if role == "baseline" else 2.3)
        resolved_kernel = dict(
            sigma_weight_profile=config.get("sigma_weight_profile", 20.0),
            Jpos_excit2excit=config.get("Jpos_excit2excit", 1.6),
        )
        _add_check(checks, role + ".registered_kernel", resolved_kernel == expected_kernel,
                   expected=expected_kernel, observed=resolved_kernel)
        if "kernel_parameters" in report:
            _add_check(checks, role + ".declared_kernel",
                       report["kernel_parameters"] == expected_kernel,
                       observed=report["kernel_parameters"])
        selected = report.get("selected_case")
        selected = selected if isinstance(selected, dict) else {}
        for name, value in dict(case_id=case_id, source_rng_seed=seed,
                                stimulus_strength_nA=STRENGTHS[condition],
                                distractor_strength_nA=0.0).items():
            _add_check(checks, role + ".selected_case." + name,
                       selected.get(name) == value, expected=value, observed=selected.get(name))
        drive = report.get("drive")
        drive = drive if isinstance(drive, dict) else {}
        drive_expected = dict(
            input_mode="coupled_binomial_clock_refinement_stream",
            n_e=80000, n_i=20000, dt_seconds=dt * 1e-3,
            coarse_dt_seconds=0.05 * 1e-3, total_ticks=ticks, ticks_generated=ticks,
            trajectory_retained=False, refinement_factor=int(round(0.05 / dt)),
            all_refinement_levels_generated=True, buffered_fine_rows=0,
        )
        for name, value in drive_expected.items():
            _add_check(checks, role + ".drive." + name, drive.get(name) == value,
                       expected=value, observed=drive.get(name))
        coupling = drive.get("coupling")
        coupling = coupling if isinstance(coupling, dict) else {}
        _add_check(checks, role + ".coupling",
                   coupling.get("coupling") == "binomial_parent_refinement"
                   and coupling.get("refinement_seeds") == refinement)
        coarse = coupling.get("coarse_source")
        coarse = coarse if isinstance(coarse, dict) else {}
        _add_check(checks, role + ".coarse_source_identity",
                   coarse.get("random_stream_policy") == "source_global"
                   and coarse.get("seeds") == {"source_rng_seed": seed}
                   and coarse.get("trajectory_retained") is False
                   and coarse.get("n_e") == 80000 and coarse.get("n_i") == 20000
                   and coarse.get("dt_seconds") == 0.05 * 1e-3
                   and coarse.get("ticks_generated") == coarse.get("total_ticks") == 100000)
        used = report.get("inputs_used")
        used = used if isinstance(used, dict) else {}
        _add_check(checks, role + ".input_consumption",
                   used.get("input_mode") == "coupled_binomial_clock_refinement_stream"
                   and used.get("ticks_consumed") == ticks)
        totals = drive.get("external_event_totals", {})
        for population, size in (("E", 80000), ("I", 20000)):
            consumption = used.get(population, {})
            _add_check(checks, role + ".input_consumption." + population,
                       isinstance(consumption, dict) and isinstance(totals, dict)
                       and consumption.get("values") == ticks * size
                       and type(consumption.get("events")) is int
                       and consumption["events"] >= 0
                       and consumption["events"] == totals.get(population))
        agreement = report.get("coarse_current_agreement")
        _add_check(checks, role + ".existing_coarse_anchor_checks",
                   report.get("coarse_final_rng_matches_archived_episode") is True
                   and isinstance(agreement, dict) and bool(agreement)
                   and all(value is True for value in agreement.values()))
        validations = report.get("validations")
        _add_check(checks, role + ".existing_case_validation",
                   isinstance(validations, dict) and bool(validations)
                   and all(isinstance(row, dict) and row.get("passed") is True
                           for row in validations.values()))
    for name in ("selected_case", "drive", "inputs_used"):
        differences = json_differences(before.get(name), after.get(name), name)
        _add_check(checks, name + "_equal", not differences, differences=differences)
    physical = [
        {key: value for key, value in report.get("resolved_config", {}).items()
         if key not in EXCLUDED_CONFIG_FIELDS}
        for report in (before, after)
    ]
    differences = json_differences(*physical)
    _add_check(checks, "resolved_physical_config_equal", not differences,
               excluded_fields=sorted(EXCLUDED_CONFIG_FIELDS), differences=differences)
    for name in INPUT_CODE:
        left = before.get("code", {}).get(name)
        right = after.get("code", {}).get(name)
        _add_check(checks, "input_generator_code." + name,
                   _valid_sha(left) and _valid_sha(right) and left == right,
                   baseline=left, candidate=right)
    return checks


def _array_map(path):
    path = Path(path)
    if path.suffix == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    with np.load(path, allow_pickle=False) as stored:
        if len(stored.files) != len(set(stored.files)):
            raise IdentityError("DUPLICATE_NPZ_MEMBER:" + str(path))
        return {key: stored[key] for key in stored.files}


def compare_stored_arrays(baseline, candidate, artifact, expected_units=None):
    """Compare complete keys, dtype, shape and values; never select RNG subsets."""
    left, right = _array_map(baseline), _array_map(candidate)
    result = dict(
        artifact=artifact, baseline_keys=sorted(left), candidate_keys=sorted(right),
        complete_key_set_equal=set(left) == set(right), members=[],
        equality="equal key set, dtype, shape, and numpy.array_equal",
    )
    required_rng = artifact != "final_rng.npz" or set(left) == set(right) == RNG_MEMBERS
    result["complete_coarse_and_refinement_rng_members"] = required_rng
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        row = dict(member=name, baseline_present=a is not None, candidate_present=b is not None)
        for role, values in (("baseline", a), ("candidate", b)):
            if values is not None:
                row[role] = dict(shape=list(values.shape), dtype=str(values.dtype),
                                 elements=int(values.size))
        compatible = (a is not None and b is not None
                      and a.shape == b.shape and a.dtype == b.dtype)
        equal = compatible and bool(np.array_equal(a, b))
        row.update(equal=bool(equal), differing_elements=(
            int(np.count_nonzero(a != b)) if compatible else None))
        result["members"].append(row)
    initial_shape = True
    if artifact == "initial_voltage.npy" and expected_units is not None:
        initial_shape = all(
            set(arrays) == {"array"} and arrays["array"].shape == (expected_units,)
            and arrays["array"].dtype == np.dtype("float64")
            for arrays in (left, right)
        )
    result["registered_initial_shape_and_dtype"] = initial_shape
    result["passed"] = bool(result["complete_key_set_equal"] and required_rng
                            and initial_shape and all(row["equal"] for row in result["members"]))
    return result


def _report_metadata(report):
    fields = (
        "scope", "case_id", "condition", "source_rng_seed", "refinement_seeds",
        "dt_ms", "coarse_dt_ms", "duration_ms", "coupled_core_units",
        "completed_duration_ticks", "selected_case", "resolved_config", "drive",
        "inputs_used", "code", "runner_sha256", "kernel_parameters", "numpy_version",
        "coarse_reference", "coarse_final_rng_matches_archived_episode", "coarse_current_agreement",
    )
    result = {key: report[key] for key in fields if key in report}
    if isinstance(report.get("fine_clock_qualification"), dict):
        qualification = report["fine_clock_qualification"]
        result["fine_clock_qualification_binding"] = {
            key: qualification[key] for key in
            ("directory", "receipt_file", "receipt_sha256", "delivery_sha256", "scope")
            if key in qualification
        }
    return result


def _compare_pair(before, after, seed, dt, condition):
    result = dict(
        case_id="pfc_" + str(seed) + "_" + condition, source_rng_seed=seed,
        condition=condition, dt_ms=dt, metadata_checks=[], array_checks=[],
        failures=before["failures"] + after["failures"], passed=False,
    )
    a, b = before["report"], after["report"]
    if a is None or b is None:
        return result
    result["metadata"] = dict(baseline=_report_metadata(a), candidate=_report_metadata(b))
    result["preserved_provenance_differences"] = json_differences(
        {key: a.get(key) for key in ("scope", "code", "runner_sha256", "kernel_parameters",
                                     "coarse_reference", "fine_clock_qualification")},
        {key: b.get(key) for key in ("scope", "code", "runner_sha256", "kernel_parameters",
                                     "coarse_reference", "fine_clock_qualification")},
    )
    result["excluded_config_differences"] = json_differences(
        {key: a["resolved_config"][key] for key in EXCLUDED_CONFIG_FIELDS
         if key in a.get("resolved_config", {})},
        {key: b["resolved_config"][key] for key in EXCLUDED_CONFIG_FIELDS
         if key in b.get("resolved_config", {})},
    )
    try:
        result["metadata_checks"] = _metadata_checks(a, b, seed, dt, condition)
    except Exception as error:
        result["failures"].append(_failure("metadata_admission", error))
    for name in ARRAY_FILES:
        if name not in before["artifacts"] or name not in after["artifacts"]:
            continue
        try:
            result["array_checks"].append(compare_stored_arrays(
                before["artifacts"][name], after["artifacts"][name], name,
                expected_units=100000,
            ))
        except Exception as error:
            result["failures"].append(_failure(name, error))
    for name in JSON_FILES:
        if name not in before["artifacts"] or name not in after["artifacts"]:
            continue
        try:
            left_path, right_path = before["artifacts"][name], after["artifacts"][name]
            left, right = _json_object(left_path), _json_object(right_path)
            same_bytes = left_path.read_bytes() == right_path.read_bytes()
            comparison = dict(
                artifact=name, exact_file_content_equal=same_bytes,
                structured_content_equal=left == right,
                baseline=left, candidate=right, differences=json_differences(left, right),
            )
            if name == "applied_current_metadata.json":
                comparison["passed"] = same_bytes and left == right
                result["applied_current_metadata"] = comparison
            else:
                comparison["required_equal"] = False
                provenance_keys = ("circuit_provenance", "resolved_kernel_parameters")
                common_units = []
                for metadata in (left, right):
                    view = dict(metadata)
                    if isinstance(view.get("coupling"), dict):
                        view["coupling"] = {
                            key: value for key, value in view["coupling"].items()
                            if key not in provenance_keys
                        }
                    common_units.append(view)
                comparison["common_unit_semantics_equal"] = common_units[0] == common_units[1]
                comparison["preserved_provenance_paths"] = [
                    "coupling." + key for key in provenance_keys
                ]
                comparison["scope"] = (
                    "Kernel and circuit provenance differences are retained; all other unit "
                    "metadata and the complete unit arrays must agree."
                )
                result["unit_metadata"] = comparison
        except Exception as error:
            result["failures"].append(_failure(name, error))
    result["passed"] = bool(
        not result["failures"] and result["metadata_checks"]
        and all(row["passed"] for row in result["metadata_checks"])
        and len(result["array_checks"]) == len(ARRAY_FILES)
        and all(row["passed"] for row in result["array_checks"])
        and result.get("applied_current_metadata", {}).get("passed") is True
        and result.get("unit_metadata", {}).get("common_unit_semantics_equal") is True
    )
    return result


def check_archives(baseline_archive, candidate_archive, baseline_delivery_sha256,
                   candidate_delivery_sha256, source_rng_seed, output):
    baseline, candidate, output = (
        Path(path).resolve() for path in (baseline_archive, candidate_archive, output)
    )
    if any(output == root or output.is_relative_to(root) for root in (baseline, candidate)):
        raise IdentityError("OUTPUT_MUST_NOT_BE_INSIDE_IMMUTABLE_INPUT_ARCHIVE")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="pfc_coupled_gain_stored_input_identity_v1", status=FAIL_STATUS,
        source_rng_seed=source_rng_seed, expected_pairs=6, pairs=[], archives={}, failures=[],
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_simulations=0, new_rng_draws=0, full_external_streams_compared=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        production_episodes=0, full_bank_qualified=False,
        scope=("One existing six-case block: persisted initial/RNG/control/unit arrays and metadata. "
               "External input trajectories were not retained; this is not a retrospective "
               "elementwise comparison of the full external streams or a scientific release."),
    )
    try:
        if type(source_rng_seed) is not int or not 95001 <= source_rng_seed <= 95032:
            raise IdentityError("REGISTERED_100K_DEVELOPMENT_SOURCE_SEED_REQUIRED")
        with tempfile.TemporaryDirectory(prefix="pfc-coupled-gain-input-identity-", dir="/tmp") as temporary:
            staged = {}
            for role, root, expected_sha in (
                ("baseline", baseline, baseline_delivery_sha256),
                ("candidate", candidate, candidate_delivery_sha256),
            ):
                try:
                    info, cases = _stage_archive(
                        root, expected_sha, Path(temporary) / role, role, source_rng_seed)
                    report["archives"][role] = info
                    staged[role] = cases
                except Exception as error:
                    report["failures"].append(_failure(role + "_archive_admission", error))
            if len(staged) == 2:
                for dt, condition, relative in _case_specs(source_rng_seed):
                    try:
                        pair = _compare_pair(staged["baseline"][relative], staged["candidate"][relative],
                                             source_rng_seed, dt, condition)
                    except Exception as error:
                        pair = dict(case_id="pfc_" + str(source_rng_seed) + "_" + condition,
                                    dt_ms=dt, passed=False, failures=[_failure(relative, error)])
                    report["pairs"].append(pair)
        report["passed_pairs"] = sum(pair["passed"] for pair in report["pairs"])
        report["unit_metadata_difference_pairs"] = sum(
            pair.get("unit_metadata", {}).get("exact_file_content_equal") is False
            for pair in report["pairs"]
        )
        if (not report["failures"] and len(report["pairs"]) == 6
                and report["passed_pairs"] == 6):
            report["status"] = PASS_STATUS
    except Exception as error:
        report["failures"].append(_failure("identity_check", error))
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--baseline-delivery-sha256", required=True)
    parser.add_argument("--candidate-delivery-sha256", required=True)
    parser.add_argument("--source-rng-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = check_archives(
            args.baseline_archive, args.candidate_archive, args.baseline_delivery_sha256,
            args.candidate_delivery_sha256, args.source_rng_seed, args.output)
    except Exception as error:
        print(json.dumps(_failure("cli", error)), file=sys.stderr)
        return 2
    print(json.dumps(dict(status=report["status"], pairs=len(report["pairs"]),
                          passed_pairs=report.get("passed_pairs", 0),
                          report=str(args.output / "report.json"))))
    return 0 if report["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())

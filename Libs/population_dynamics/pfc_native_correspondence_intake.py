"""Receive completed PFC correspondence evidence without repeating array work.

The completed execution wrapper authenticates scientific payloads during staging.
This consumer checks its terminal metadata and retains the complete case family.
Correspondence does not grant source qualification or production admission.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from .provenance import canonical_json, freeze_json, sha256_file

FALSE_FLAGS = ("full_scale_qualified", "scientific_fidelity_qualified",
               "working_memory_qualified", "fidelity_at_requested_scale_qualified")
ARRAY_NAMES = ("native_spike_counts", "spike_seconds", "spike_ticks",
               "spike_unit_indices", "terminal_voltage_mV")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("PFC_CORRESPONDENCE_INTAKE:" + reason)


def scheduler(job_id: str) -> dict:
    require(job_id.isdecimal(), "invalid job ID")
    result = subprocess.run(["sacct", "-j", job_id, "-X", "-n", "-P", "-o",
                             "JobID,State,ExitCode,ElapsedRaw,NodeList"],
                            check=True, capture_output=True, text=True, timeout=30)
    rows = [line.split("|") for line in result.stdout.splitlines()
            if line.split("|")[0] == job_id]
    require(len(rows) == 1, "missing or ambiguous scheduler receipt")
    _, state, code, seconds, node, *_ = rows[0]
    require(state in ("COMPLETED", "FAILED") and node == "cpu03",
            "job is not terminal on CPU03")
    require(seconds.isdecimal(), "invalid scheduler cost")
    return dict(job_id=job_id, state=state, exit_code=code,
                elapsed_seconds=int(seconds), node=node)


def intake(root: Path, request_path: Path, job_id: str, reused_path: Path) -> dict:
    root = root.resolve()
    observed = scheduler(job_id)
    verified: dict[str, dict] = {}

    def path(value) -> Path:
        p = Path(value)
        p = (p if p.is_absolute() else root / p).resolve()
        require(p.is_relative_to(root), "metadata outside project")
        return p

    def read(value, expected=None, *, structured=True):
        p = path(value)
        size = p.stat().st_size
        require(size <= 16 * 1024**2, "metadata too large")
        digest = sha256_file(p)
        if expected is not None:
            require(digest == expected["sha256"] and size == expected["bytes"],
                    "metadata binding changed: " + str(p))
        data = p.read_bytes()
        require(sha256_file(p) == digest, "metadata changed during read")
        verified[str(p.relative_to(root))] = dict(bytes=size, sha256=digest)
        return json.loads(data) if structured else data

    def diagnostic(value, *, native_report=False):
        # The source-native report predates the wrapper's production counter.
        # Only that report may omit it; delivery and execution must declare zero.
        zero_production = (value.get("production_episodes") == 0
                           or native_report and "production_episodes" not in value)
        require(zero_production and all(
            value.get(key) is False for key in FALSE_FLAGS if key in value),
            "scientific eligibility changed")

    def argument(command, name):
        require(command.count(name) == 1, "missing or duplicate argument: " + name)
        return command[command.index(name) + 1]

    request_path = path(request_path)
    request = read(request_path)
    require(request.get("scientific_execution_authorized") is True
            and request.get("production_admission_amendment_authorized") is False
            and request.get("scientific_protocol_changed") is False
            and request.get("source_cases_regenerated") is False,
            "request authority or scope changed")
    diagnostic(request)
    execution_path = request_path.parent / ("run_" + job_id) / "execution.json"
    execution = read(execution_path)
    require(execution.get("job_id") == job_id and execution.get("request") == request
            and execution.get("node") == observed["node"]
            and execution.get("scientific_protocol_changed") is False,
            "execution request or job mismatch")
    diagnostic(execution)
    require(execution.get("status") in ("COMPLETE_AUTHORIZED_NATIVE_AND_COMPARISON",
            "COMPLETE_COMPARISON_WITH_NUMERICAL_FAILURE")
            and execution.get("native_exit_code") == 0
            and execution.get("comparison_exit_code") in (0, 1)
            and execution.get("exit_code") == execution["comparison_exit_code"]
            and observed["exit_code"] == str(execution["exit_code"]) + ":0",
            "execution or scheduler is incomplete")
    phases = execution.get("phases", [])
    require([p.get("name") for p in phases] == ["native_preflight", "native_validation",
            "source_comparison"] and [p.get("exit_code") for p in phases] ==
            [0, 0, execution["comparison_exit_code"]], "incomplete execution phases")
    native_command = request["native_command"]
    native_root = path(argument(native_command, "--output"))
    preparation = path(argument(native_command, "--preparation"))
    design_path = preparation / "design.json"
    design = read(design_path)
    diagnostic(design)
    require(design.get("status") == "REGISTERED_BEFORE_NATIVE_REFERENCE_EXECUTION"
            and design.get("scope") == "native_reference_clock_correspondence"
            and (design["excitatory_neurons"], design["inhibitory_neurons"],
                 design["duration_ms"], design["dt_ms"]) == (80000, 20000, 5000, 0.0125),
            "registered source scale or clock changed")
    config_path = preparation / design["native_config_file"]
    read(config_path, dict(bytes=config_path.stat().st_size,
                          sha256=design["native_config_sha256"]), structured=False)
    canonical = path(argument(native_command, "--canonical-config"))
    read(canonical, dict(bytes=canonical.stat().st_size,
                        sha256=design["source_config_sha256"]), structured=False)
    delivery_path = native_root / "delivery.json"
    delivery = read(delivery_path, dict(bytes=delivery_path.stat().st_size,
                                      sha256=execution["native_delivery_sha256"]))
    diagnostic(delivery)
    require(delivery.get("job_id") == job_id and delivery.get("status") ==
            "NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE" and delivery.get("exit_code") == 0
            and delivery.get("archive_status") == "COMPLETE"
            and delivery.get("native_exit_code") == 0
            and delivery.get("design_sha256") == verified[str(design_path.relative_to(root))]["sha256"]
            and delivery.get("native_config_sha256") == design["native_config_sha256"]
            and delivery.get("source_config_sha256") == design["source_config_sha256"],
            "native delivery binding changed")
    files = delivery["files"]
    require(len(files) <= 1000, "archive inventory too large")
    entries = {}
    for row in files:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and row["path"] not in entries, "invalid archive path")
        p = native_root / relative
        require(not p.is_symlink() and p.is_file() and p.stat().st_size == row["bytes"],
                "delivered file missing or resized")
        entries[row["path"]] = row
    native = read(native_root / "native/report.json", entries["native/report.json"])
    native_execution = read(native_root / "execution.json", entries["execution.json"])
    read(native_root / "native_config.yaml", entries["native_config.yaml"], structured=False)
    read(native_root / "Libs/population_dynamics/pfc_factorized_source.py",
         entries["Libs/population_dynamics/pfc_factorized_source.py"], structured=False)
    require(entries["Libs/population_dynamics/pfc_factorized_source.py"]["sha256"] ==
            design["reference_execution_implementation_sha256"]
            and entries["native_config.yaml"]["sha256"] == design["native_config_sha256"]
            and native_execution.get("native_report_sha256") == entries["native/report.json"]["sha256"],
            "native execution/config/helper binding changed")
    diagnostic(native, native_report=True)
    diagnostic(native_execution)
    require(native_execution.get("job_id") == job_id
            and native_execution.get("status") == "NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE"
            and native_execution.get("native_exit_code") == 0
            and native_execution.get("design_sha256") == delivery["design_sha256"]
            and native_execution.get("native_config_sha256") == design["native_config_sha256"]
            and native_execution.get("source_config_sha256") == design["source_config_sha256"],
            "native execution receipt changed")
    require(native.get("status") == "NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
            and native.get("scope") == "source_reproduction"
            and native.get("full_scale_qualified") is False
            and native.get("scientific_fidelity_qualified") is False
            and native.get("source_revision") == design["source_revision"]
            and native.get("source_entry_sha256") == design["source_entry_sha256"]
            and native.get("execution_implementation_sha256") == design["reference_execution_implementation_sha256"]
            and native.get("synapse_execution") == design["reference_synapse_execution"],
            "native source identity changed")
    for field in ("excitatory_neurons", "inhibitory_neurons", "dt_ms", "duration_ms"):
        require(native["resolved_config"][field] == design[field], "native dimensions changed")
    runtime = delivery["runtime_versions"]
    require(runtime == native_execution["runtime_versions"]
            and runtime["python"].startswith("3.10.14") and all(runtime[key] ==
            design["reference_runtime"][key + "_version"] for key in ("brian2", "numpy", "scipy")),
            "native runtime changed")
    comparison_path = path(request["comparison_output"])
    require(path(execution["comparison_report"]) == comparison_path, "comparison output changed")
    comparison = read(comparison_path)
    diagnostic(comparison)
    require(comparison.get("status") == execution["comparison_status"] and
            comparison.get("status") == ("PASS_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE"
            if execution["exit_code"] == 0 else "FAILED_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE"),
            "comparison terminal status changed")
    require(comparison["design_sha256"] == delivery["design_sha256"]
            and comparison["native_config_sha256"] == design["native_config_sha256"]
            and comparison["config_sha256"] == design["source_config_sha256"]
            and comparison["source_entry_sha256"] == design["source_entry_sha256"]
            and comparison["source_revision"] == design["source_revision"],
            "comparison source/design binding changed")
    bindings = comparison["reference_bindings"]
    require(len(bindings) == 1 and bindings[0]["delivery_sha256"] == execution["native_delivery_sha256"]
            and bindings[0]["report_sha256"] == entries["native/report.json"]["sha256"]
            and bindings[0]["execution_receipt_sha256"] == entries["execution.json"]["sha256"]
            and bindings[0]["helper_sha256"] == design["reference_execution_implementation_sha256"],
            "comparison native bindings changed")
    require([c["case_id"] for c in native["cases"]] == design["case_ids"] == request["cases"]
            == [c["case_id"] for c in comparison["cases"]]
            and delivery["completed_cases"] == design["cases"] == len(request["cases"]),
            "selected case denominator changed")
    for case, compared in zip(native["cases"], comparison["cases"]):
        require(case["core_units"] == 100000 and case["source_rng_seed"] == compared["source_rng_seed"]
                and case["native_events"] == compared["native_events"]
                and case["elapsed_seconds"] == compared["native_seconds"], "native case metadata changed")
        require(entries["native/" + case["raw_file"]]["sha256"] == case["raw_sha256"],
                "native raw binding changed")
    source_bindings = comparison["source_bindings"]
    require([{k: c[k] for k in ("case_id", "delivery_sha256", "report_sha256")}
             for c in source_bindings] == design["preexisting_source_bindings"],
            "candidate source binding changed")
    reused = read(reused_path)
    require(reused.get("status") == "PASS_TERMINAL_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE_INTAKE"
            and reused["job_id"] == request["reused_completed_job_id"]
            and reused["original_case"] == request["reused_completed_case"]
            and reused.get("source_qualification_granted") is False
            and reused.get("source_history_population_uncertainty_established") is False,
            "reused acceptance changed")
    diagnostic(reused)
    for name, row in reused["metadata_verified"].items():
        read(name, row, structured=False)
    old = read(reused["comparison_report"])
    require((reused["core_units"], reused["duration_ms"], reused["dt_ms"],
             reused["parameter_circuits"]) == (100000, 5000, 0.0125, 1)
            and [c["case_id"] for c in old["cases"]] == [request["reused_completed_case"]],
            "reused source scope changed")
    require(old["tolerances"] == comparison["tolerances"] == dict(other_arrays="exact",
            relative=design["relative_tolerance"],
            terminal_voltage_mV_absolute=design["terminal_voltage_mV_max_absolute_error"])
            and old["canonical_case_ids"] == comparison["canonical_case_ids"] == design["canonical_case_ids"],
            "canonical family or frozen tolerance changed")
    cases = old["cases"] + comparison["cases"]
    by_id = {case["case_id"]: case for case in cases}
    require(len(cases) == len(by_id) == design["canonical_registered_cases"] == 9
            and set(by_id) == set(design["canonical_case_ids"]), "omitted or duplicated canonical case")
    require(all(case["source_rng_seed"] in (94001, 94002, 94003) for case in cases)
            and len({case["source_rng_seed"] for case in cases}) == 3
            and comparison["common_execution_binding"] == old["common_execution_binding"],
            "source block or execution scope changed")
    failures = []
    for case in cases:
        arrays = case["arrays"]
        require([a["array"] for a in arrays] == list(ARRAY_NAMES), "array denominator changed")
        failed = [a["array"] for a in arrays if a["passed"] is False]
        require(all(type(a["passed"]) is bool for a in arrays)
                and failed == case["failed_arrays"]
                and case["status"] == ("FAILED_NUMERICAL_EQUIVALENCE" if failed else "PASS"),
                "array outcome changed")
        if failed:
            failures.append(case["case_id"])
        for array in arrays:
            require(array["rtol"] == 0 and array["atol"] ==
                    (design["terminal_voltage_mV_max_absolute_error"] if array["array"] ==
                     "terminal_voltage_mV" else 0), "array tolerance changed")
        require(case["values_compared"] == sum(a["values_compared"] for a in arrays)
                and all(math.isfinite(case[k]) and case[k] >= 0 for k in ("native_seconds", "candidate_seconds")),
                "case comparison or cost incomplete")
    require(comparison["failed_case_ids"] == [c["case_id"] for c in comparison["cases"]
            if c["failed_arrays"]] and comparison["array_values_compared"] == sum(
            c["values_compared"] for c in comparison["cases"]), "comparison totals changed")
    return dict(status="FAILED_COMPLETE_NATIVE_CORRESPONDENCE_INTAKE" if failures else
                "PASS_COMPLETE_NATIVE_CORRESPONDENCE_INTAKE", exit_code=int(bool(failures)),
                scheduler=observed, reused_completed_job_id=reused["job_id"],
                canonical_case_ids=design["canonical_case_ids"], compared_cases=9,
                passed_cases=9-len(failures), failed_case_ids=failures,
                cases=[by_id[name] for name in design["canonical_case_ids"]],
                array_values_compared=sum(c["values_compared"] for c in cases),
                native_events=sum(c["native_events"] for c in cases),
                native_execute_seconds=sum(c["native_seconds"] for c in cases),
                candidate_execute_seconds=sum(c["candidate_seconds"] for c in cases),
                cost_interpretation="Separate executions; no matched-cost speedup claim.",
                core_units=100000, duration_ms=5000, dt_ms=0.0125,
                parameter_circuits=1, independent_source_blocks=3,
                tolerances=comparison["tolerances"], metadata_verified=verified,
                native_archive_files=len(files), native_archive_bytes=sum(f["bytes"] for f in files),
                bulk_authentication_reused_from_completed_wrapper=True,
                scientific_arrays_rehashed=False, scientific_comparison_repeated=False,
                new_source_simulations=0, production_episodes=0,
                source_qualification_granted=False, source_history_population_uncertainty_established=False,
                protected_confirmation_accessed=False, production_admission_amendment_approved=False,
                scientific_protocol_changed=False, **dict.fromkeys(FALSE_FLAGS, False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reused-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = intake(args.project_root, args.request, args.job_id, args.reused_acceptance)
        freeze_json(args.output, report)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(canonical_json(dict(status="REJECTED_NATIVE_CORRESPONDENCE_INTAKE", error=str(error))), end="")
        return 2
    print(canonical_json({k: v for k, v in report.items() if k not in ("cases", "metadata_verified")}), end="")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

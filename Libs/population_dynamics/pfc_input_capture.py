"""Record native PFC drive and compare observation against untouched sampling."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .pfc_input_observation import NativeInputObserver
from .pfc_reference import case_settings, execute
from .pfc_reference_replay import compare as compare_native
from .provenance import sha256_file, freeze_json


def record(source: Path, config: Path, output: Path, mode: str):
    output.mkdir(parents=True, exist_ok=False)
    observer = NativeInputObserver(output / "inputs", mode=mode)
    try:
        execute(source, config, output / "native", native_input_observer=observer)
        inputs = observer.finalize()
        report = dict(status="RECORDED_NATIVE_PFC_INPUT_OBSERVATION", mode=mode,
                      recorded_utc=datetime.now(timezone.utc).isoformat(),
                      config_sha256=sha256_file(config), native_report_sha256=sha256_file(output / "native/report.json"),
                      observation_report_sha256=sha256_file(output / "inputs/report.json"),
                      cases=len(inputs["cases"]), implementation_sha256=sha256_file(Path(__file__)),
                      observer_implementation_sha256=sha256_file(Path(__file__).with_name("pfc_input_observation.py")),
                      reference_implementation_sha256=sha256_file(Path(__file__).with_name("pfc_reference.py")),
                      source_parity_verified=False, gpu_parity_qualified=False,
                      scientific_fidelity_qualified=False, production_episodes=0)
        freeze_json(output / "execution.json", report)
        return report
    finally:
        observer.close()


def _bound(root: Path, name: str, digest: str):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or sha256_file(root / relative) != digest:
        raise ValueError("PFC_INPUT_COMPARISON_ARTIFACT_CHANGED:" + name)
    return root / relative


def _load(root: Path, mode: str):
    execution = json.loads((root / "execution.json").read_text())
    if execution["status"] != "RECORDED_NATIVE_PFC_INPUT_OBSERVATION" or execution["mode"] != mode:
        raise ValueError("PFC_INPUT_COMPARISON_COMPLETE_EXECUTION_REQUIRED")
    _bound(root, "native/report.json", execution["native_report_sha256"])
    report = json.loads(_bound(root, "inputs/report.json", execution["observation_report_sha256"]).read_text())
    if (report["status"] != "NATIVE_SOURCE_INPUT_OBSERVATION_COMPLETE" or report["mode"] != mode
            or len(report["cases"]) != execution["cases"]):
        raise ValueError("PFC_INPUT_COMPARISON_COMPLETE_OBSERVATION_REQUIRED")
    return execution, report



def _bind_native_case(root, row, receipt, native_report, native_case, expected_case):
    """Tie observed native state and input geometry to this exact source case."""
    config = native_report["resolved_config"]
    if (receipt["case"] != expected_case or row["case_id"] != expected_case["case_id"]
            or native_case["case_id"] != expected_case["case_id"]
            or receipt["source_entry_sha256"] != native_report["source_entry_sha256"]):
        raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_CASE_MISMATCH")
    for name in ("source_rng_seed", "stimulus_strength_nA", "distractor_strength_nA"):
        if native_case[name] != expected_case[name]:
            raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_PARAMETER_MISMATCH:" + name)
    for name in ("initial_rng_seed", "external_noise_seed"):
        if name in expected_case and native_case.get(name) != expected_case[name]:
            raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_PARAMETER_MISMATCH:" + name)
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    dt = config["dt_ms"] / 1000
    ticks = int(round(config["duration_ms"] / config["dt_ms"]))
    if ticks <= 0 or not np.isclose(ticks * dt, config["duration_ms"] / 1000, rtol=0, atol=1e-12):
        raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_CLOCK_INVALID")
    providers = receipt["providers"]
    if [p["population"] for p in providers] != ["I", "E"]:
        raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_PROVIDER_MISMATCH")
    for provider, units in zip(providers, (n_i, n_e)):
        if (provider["units"] != units or provider["expected_sample_rows"] != ticks
                or provider["dt_seconds"] != dt
                or not np.isclose(provider["terminal_clock_seconds"], ticks * dt, rtol=0, atol=1e-12)
                or provider["target_variable"] != "s_AMPA" or provider["N"] != 1000
                or provider["rate_Hz"] != 1.4 or provider["probability"] != 1.4 * dt
                or provider["weight_dimensionless"] != 1.0
                or provider["when"] != "synapses" or provider["order"] != 0
                or provider["source_unit_order"] != "ascending_zero_based_population_index"):
            raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_INPUT_GEOMETRY_MISMATCH")
        if receipt["mode"] == "capture" and (provider["counts_shape"] != [ticks, units]
                                               or provider["observed_sampler_calls"] != ticks):
            raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_CAPTURE_LENGTH_MISMATCH")
    for kind in ("terminal_state", "numpy_rng", "python_rng", "generated_code"):
        for suffix in ("_file", "_sha256"):
            if receipt[kind + suffix] != row[kind + suffix]:
                raise ValueError("PFC_INPUT_OBSERVATION_RECEIPT_SNAPSHOT_MISMATCH")
    state = _bound(root / "inputs", row["terminal_state_file"], row["terminal_state_sha256"])
    native = _bound(root / "native", native_case["raw_file"], native_case["raw_sha256"])
    with np.load(state, allow_pickle=False) as snapshot, np.load(native, allow_pickle=False) as raw:
        for pop, units, offset in (("E", n_e, 0), ("I", n_i, n_e)):
            if (not np.array_equal(snapshot[pop + "__source_indices"], np.arange(units))
                    or not np.array_equal(snapshot[pop + "__archive_indices"], np.arange(units) + offset)):
                raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_UNIT_MAPPING_MISMATCH")
        counts = np.concatenate([snapshot[pop + "__spike_counts"] for pop in ("E", "I")])
        voltage = np.concatenate([snapshot[pop + "__v"] / 0.001 for pop in ("E", "I")])
        if (native_case["core_units"] != n_e + n_i or counts.shape != (n_e + n_i,)
                or voltage.shape != (n_e + n_i,)
                or not np.array_equal(counts, raw["native_spike_counts"])
                or not np.array_equal(counts, np.bincount(raw["spike_unit_indices"], minlength=n_e + n_i))
                or not np.array_equal(voltage, raw["terminal_voltage_mV"])
                or int(counts.sum()) != native_case["native_events"]):
            raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_STATE_MISMATCH")
    return 2 * (n_e + n_i)


def compare(audit: Path, capture: Path, output: Path, *, plain: Path | None = None):
    output.mkdir(parents=True, exist_ok=False)
    a_execution, a = _load(audit, "audit")
    b_execution, b = _load(capture, "capture")
    for key in ("config_sha256", "observer_implementation_sha256", "reference_implementation_sha256"):
        if a_execution[key] != b_execution[key]:
            raise ValueError("PFC_INPUT_COMPARISON_EXECUTION_BINDING_CHANGED:" + key)
    for key in ("source_entry_sha256", "brian2_version", "numpy_version"):
        if a[key] != b[key]:
            raise ValueError("PFC_INPUT_COMPARISON_SOURCE_BINDING_CHANGED:" + key)
    identities = [row["case_id"] for row in a["cases"]]
    if not identities or len(identities) != len(set(identities)) or identities != [row["case_id"] for row in b["cases"]]:
        raise ValueError("PFC_INPUT_COMPARISON_CASE_SET_CHANGED")
    native = compare_native(audit / "native", capture / "native", output / "native_replay.json")
    plain_report = None
    if plain is not None:
        plain_report = compare_native(plain, audit / "native", output / "plain_native_replay.json")
    native_reports = [json.loads((root / "native/report.json").read_text()) for root in (audit, capture)]
    resolved = case_settings(native_reports[0]["resolved_config"])
    if any([row["case_id"] for row in report["cases"]] != identities for report in native_reports):
        raise ValueError("PFC_INPUT_OBSERVATION_NATIVE_CASE_SET_MISMATCH")
    if [row["case_id"] for row in resolved] != identities:
        raise ValueError("PFC_INPUT_OBSERVATION_CONFIG_CASE_SET_MISMATCH")
    state_values = rng_values = draw_values = native_state_bindings = 0
    cases = []
    for index, (left, right) in enumerate(zip(a["cases"], b["cases"])):
        receipts = [json.loads(_bound(root / "inputs", row["receipt_file"], row["receipt_sha256"]).read_text())
                    for root, row in ((audit, left), (capture, right))]
        if any(receipt["status"] != "NATIVE_INPUT_OBSERVATION_CASE_COMPLETE" for receipt in receipts):
            raise ValueError("PFC_INPUT_COMPARISON_CASE_INCOMPLETE")
        for key in ("case", "source_entry_sha256", "source_population_archive_order", "source_provider_construction_order"):
            if receipts[0][key] != receipts[1][key]:
                raise ValueError("PFC_INPUT_COMPARISON_CASE_BINDING_CHANGED:" + key)
        for root, row, receipt, native_report in zip(
                (audit, capture), (left, right), receipts, native_reports):
            native_state_bindings += _bind_native_case(
                root, row, receipt, native_report, native_report["cases"][index], resolved[index])
        for kind in ("terminal_state", "numpy_rng"):
            paths = [_bound(root / "inputs", row[kind + "_file"], row[kind + "_sha256"])
                     for root, row in ((audit, left), (capture, right))]
            with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    raise ValueError("PFC_INPUT_COMPARISON_ARRAY_SET_CHANGED")
                for name in x.files:
                    first, second = x[name], y[name]
                    if first.dtype != second.dtype or first.shape != second.shape or first.tobytes() != second.tobytes():
                        raise ValueError("PFC_INPUT_OBSERVATION_ALTERED_NATIVE_STATE:" + kind + ":" + name)
                    if kind == "terminal_state": state_values += first.size
                    else: rng_values += first.size
        python_states = [json.loads(_bound(root / "inputs", row["python_rng_file"], row["python_rng_sha256"]).read_text())
                         for root, row in ((audit, left), (capture, right))]
        if python_states[0] != python_states[1]:
            raise ValueError("PFC_INPUT_OBSERVATION_ALTERED_PYTHON_RNG")
        for root, row in ((audit, left), (capture, right)):
            _bound(root / "inputs", row["generated_code_file"], row["generated_code_sha256"])
        providers = receipts[1]["providers"]
        if [row["population"] for row in providers] != ["I", "E"]:
            raise ValueError("PFC_INPUT_COMPARISON_PROVIDER_SET_CHANGED")
        total = 0
        for provider in providers:
            path = _bound(capture / "inputs", provider["counts_file"], provider["counts_sha256"])
            counts = np.load(path, allow_pickle=False, mmap_mode="r")
            if (counts.dtype != np.dtype("uint16") or list(counts.shape) != provider["counts_shape"]
                    or counts.shape != (provider["expected_sample_rows"], provider["units"])
                    or provider["observed_sampler_calls"] != counts.shape[0]):
                raise ValueError("PFC_INPUT_COMPARISON_FULL_DRAW_ARRAY_REQUIRED")
            events = 0
            for start in range(0, counts.shape[0], 256):
                block = counts[start:start + 256]
                if np.any(block > 1000): raise ValueError("PFC_INPUT_DRAW_SUPPORT_INVALID")
                events += int(block.sum(dtype=np.uint64))
            if events != provider["captured_external_events"]:
                raise ValueError("PFC_INPUT_DRAW_RECOUNT_MISMATCH")
            total += events
            draw_values += counts.size
            del counts
        cases.append(dict(case_id=left["case_id"], external_events=total,
                          terminal_states_equal=True, numpy_rng_equal=True, python_rng_equal=True))
    report = dict(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY", cases=cases,
                  audit_execution_sha256=sha256_file(audit / "execution.json"),
                  capture_execution_sha256=sha256_file(capture / "execution.json"),
                  terminal_array_values_compared=state_values, numpy_rng_values_compared=rng_values,
                  captured_draw_values_recounted=draw_values,
                  observation_values_bound_to_native_case=native_state_bindings,
                  native_array_values_compared=native["fresh_process_array_values_checked"],
                  initial_voltage_values_compared=native["fresh_process_initial_voltage_values_checked"],
                  plain_native_array_values_compared=None if plain_report is None else plain_report["fresh_process_array_values_checked"],
                  plain_source_execution_compared=plain is not None,
                  scope="native observation parity and drive recount; not a GPU comparison or independent source replicate",
                  source_inputs_replayed=False, gpu_parity_qualified=False,
                  scientific_fidelity_qualified=False, production_episodes=0,
                  implementation_sha256=sha256_file(Path(__file__)))
    freeze_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recording = sub.add_parser("record")
    for name in ("source", "config", "output"): recording.add_argument("--" + name, type=Path, required=True)
    recording.add_argument("--mode", choices=("audit", "capture"), required=True)
    comparison = sub.add_parser("compare")
    for name in ("audit", "capture", "output"): comparison.add_argument("--" + name, type=Path, required=True)
    comparison.add_argument("--plain", type=Path)
    args = parser.parse_args()
    if args.command == "record": result = record(args.source, args.config, args.output, args.mode)
    else: result = compare(args.audit, args.capture, args.output, plain=args.plain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

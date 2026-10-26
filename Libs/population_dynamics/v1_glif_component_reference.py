"""Capture isolated native GLIF/PSC components for later backend comparison.

These deterministic test cases are not coupled-network benchmark episodes.
Parameters come from the existing 111-model Allen GLIF3 reference bank.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_cell_parameters import BMTK_CONVERTER_SHA256
from .v1_nest_input_replay import NEST_VERSION, bound_json
from .v1_nest_time_semantics import V1_RESOLUTION_MS, encode_precise_times, source_refractory_steps

DURATION_MS = 500.0
REQUESTED_DELAY_MS = 0.625
DEVICE_CONNECTION_DELAY_MS = 1.0
SIGNALS = ("V_m", "I_syn", "I", "ASCurrents_sum", "threshold", "threshold_spike", "threshold_voltage")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def input_protocols() -> dict[str, np.ndarray]:
    return {
        "burst": 0.125 + V1_RESOLUTION_MS * np.arange(200, dtype=np.float64),
        "sparse": np.array([0.125, 0.5, 1.001, 1.249, 1.251, 10.01, 10.125,
                            20.0, 50.001, 100.125, 250.5, 499.875], dtype=np.float64),
    }


def build_cases(references: list[dict]) -> list[dict]:
    require(bool(references), "GLIF_REFERENCE_EMPTY_BANK")
    identities = [row["neuronal_model_id"] for row in references]
    require(len(set(identities)) == len(identities), "GLIF_REFERENCE_DUPLICATE_MODEL")
    cases = []
    for index, reference in enumerate(references):
        p = reference["converted_parameters"]
        require(reference["converted_template"] == "nest:glif_psc"
                and p["after_spike_currents"] is True
                and p["adapting_threshold"] is False
                and p["spike_dependent_threshold"] is False,
                "GLIF_REFERENCE_SOURCE_GLIF3_REQUIRED")
        require(all(len(p[key]) == 2 for key in ("asc_init", "asc_amps", "asc_decay")),
                "GLIF_REFERENCE_TWO_ASC_COMPONENTS_REQUIRED")
        tau = np.asarray(p["tau_syn"], dtype=np.float64)
        require(tau.ndim == 1 and tau.size > 0 and np.isfinite(tau).all() and np.all(tau > 0),
                "GLIF_REFERENCE_RECEPTOR_CONSTANTS_INVALID")
        rheobase = float(p["g"] * (p["V_th"] - p["E_L"]))
        require(np.isfinite(rheobase) and rheobase > 0 and p["C_m"] > 0,
                "GLIF_REFERENCE_INTRINSIC_DOMAIN_INVALID")
        prefix = "model_" + str(reference["neuronal_model_id"])
        def case(kind: str, source: str | None, receptor: int | None, weight: float) -> dict:
            return dict(case_id=prefix + "_" + kind, source_index=index, kind=kind,
                        input_kind=source, receptor_type=receptor, weight_pA=float(weight))
        cases.append(case("passive", None, None, 0.0))
        # A declared numerical challenge, not a benchmark or biological stimulus.
        burst_weight = 2 * rheobase + 2 * float(np.abs(p["asc_amps"]).sum())
        cases.append(case("burst_r1", "burst", 1, burst_weight))
        for receptor in range(1, len(tau) + 1):
            for sign, name in ((1, "positive"), (-1, "negative")):
                cases.append(case(f"sparse_{name}_r{receptor}", "sparse", receptor,
                                  sign * 0.25 * rheobase))
    return cases


def recording_matrices(events: dict, native_ids: np.ndarray,
                       expected_times: np.ndarray) -> dict[str, np.ndarray]:
    ids = np.asarray(native_ids)
    times = np.asarray(expected_times)
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(np.unique(ids)) == len(ids),
            "GLIF_REFERENCE_CASE_IDENTITIES_INVALID")
    require(times.ndim == 1 and times.dtype == np.float64 and len(times) > 0
            and np.isfinite(times).all() and np.all(np.diff(times) > 0),
            "GLIF_REFERENCE_TIME_COORDINATES_INVALID")
    senders = np.asarray(events["senders"])
    recorded = np.asarray(events["times"])
    count = len(ids) * len(times)
    require(senders.shape == recorded.shape == (count,) and senders.dtype.kind in "iu"
            and recorded.dtype == np.float64, "GLIF_REFERENCE_RECORDING_SHAPE_INVALID")
    id_order = np.argsort(ids)
    event_order = np.lexsort((recorded, senders))
    require(np.array_equal(senders[event_order].reshape(len(ids), -1),
                           np.broadcast_to(ids[id_order, None], (len(ids), len(times))))
            and np.array_equal(recorded[event_order].reshape(len(ids), -1),
                               np.broadcast_to(times, (len(ids), len(times)))),
            "GLIF_REFERENCE_RECORDING_COORDINATES_DIFFER")
    restore = np.argsort(id_order)
    result = {}
    for name in SIGNALS:
        values = np.asarray(events[name])
        require(values.shape == (count,) and values.dtype == np.float64 and np.isfinite(values).all(),
                "GLIF_REFERENCE_SIGNAL_PRECISION_OR_SHAPE_INVALID:" + name)
        result[name] = values[event_order].reshape(len(ids), -1)[restore]
    return result


def run_reference(parameters: Path, output: Path) -> dict:
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_REFERENCE_FRESH_COMPUTE_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_NATIVE_GLIF_COMPONENT_REFERENCE", job_id=os.environ["SLURM_JOB_ID"],
                  benchmark_episodes=0, source_network_episodes=0, production_episodes=0,
                  scientific_qualification=False, gpu_backend_qualified=False,
                  duration_ms=DURATION_MS, dt_ms=V1_RESOLUTION_MS, native_simulation_calls=0,
                  cells_created=0, source_bank=str(parameters),
                  code_sha256=sha256_file(Path(__file__)))
    started = time.monotonic()
    try:
        import nest  # Optional native dependency; pure helpers and tests need no NEST.
        require(nest.__version__ == NEST_VERSION, "GLIF_REFERENCE_NATIVE_VERSION_DIFFERS")
        bank, delivery = bound_json(parameters, "reference/intake.json")
        require(delivery["status"] == bank["status"] == "REFERENCE_PARAMETERS_ACQUIRED"
                and delivery["exit_code"] == 0 and len(bank["references"]) == 111
                and bank["bmtk_converter_sha256"] == BMTK_CONVERTER_SHA256,
                "GLIF_REFERENCE_COMPLETE_BOUND_SOURCE_BANK_REQUIRED")
        references = bank["references"]
        cases = build_cases(references)
        protocols = input_protocols()
        report.update(source_bank_intake_sha256=sha256_file(parameters / "reference/intake.json"),
                      source_models_loaded=111, planned_component_cases=len(cases),
                      references=[{key: row[key] for key in ("specimen_id", "neuronal_model_id",
                                  "converted_template", "converted_parameters")} for row in references],
                      input_protocols_ms={key: value.tolist() for key, value in protocols.items()})
        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=V1_RESOLUTION_MS, local_num_threads=1))
        # A one-tick communication interval exposes every interior multimeter sample.
        nest.SetKernelStatus(dict(min_delay=V1_RESOLUTION_MS, max_delay=DEVICE_CONNECTION_DELAY_MS))
        defaults = nest.GetDefaults("glif_psc")
        require(set(SIGNALS).issubset(set(defaults["recordables"])),
                "GLIF_REFERENCE_NATIVE_RECORDABLES_MISSING")
        tics_per_ms = float(nest.GetKernelStatus("tics_per_ms"))
        report.update(nest_version=nest.__version__, kernel_rng_seed=int(nest.GetKernelStatus("rng_seed")),
                      stochastic_devices_used=False, native_recordables=[str(x) for x in defaults["recordables"]],
                      source_refractory_steps=source_refractory_steps(
                          np.array([row["converted_parameters"]["t_ref"] for row in references]), tics_per_ms).tolist())
        ids = []
        for case in cases:
            p = deepcopy(references[case["source_index"]]["converted_parameters"])
            cell = nest.Create("glif_psc", params=p)
            native_id = int(cell.tolist()[0])
            ids.append(native_id)
            case["native_id"] = native_id
            report["cells_created"] = len(ids)
            actual = nest.GetStatus(cell)[0]
            require(all(np.array_equal(np.asarray(actual[key]), np.asarray(value)) for key, value in p.items()),
                    "GLIF_REFERENCE_NATIVE_PARAMETER_ROUNDTRIP_DIFFERS:" + case["case_id"])
        nodes = nest.NodeCollection(ids)
        initial_vm = np.asarray(nest.GetStatus(nodes, "V_m"), dtype=np.float64)
        initial_asc = np.stack([np.asarray(x, dtype=np.float64) for x in nest.GetStatus(nodes, "ASCurrents")])
        generators = {key: nest.Create("spike_generator", params=dict(spike_times=value.tolist(), precise_times=True))
                      for key, value in protocols.items()}
        delay = float(np.floor(REQUESTED_DELAY_MS / V1_RESOLUTION_MS + 0.5) * V1_RESOLUTION_MS)
        for case in cases:
            if case["input_kind"] is None:
                continue
            generator = generators[case["input_kind"]]
            target = nest.NodeCollection([case["native_id"]])
            nest.Connect(generator, target, syn_spec=dict(weight=case["weight_pA"], delay=REQUESTED_DELAY_MS,
                                                          receptor_type=case["receptor_type"]))
            connection = nest.GetConnections(source=generator, target=target)
            native = connection.get()
            expected = dict(source=int(generator.tolist()[0]), target=case["native_id"],
                            receptor=case["receptor_type"], weight=case["weight_pA"], delay=delay)
            for key, value in expected.items():
                actual = np.asarray(native[key]).reshape(-1)
                require(actual.size == 1 and actual[0] == value,
                        "GLIF_REFERENCE_NATIVE_INPUT_CONNECTION_DIFFERS:" + case["case_id"] + ":" + key)
            case["requested_delay_ms"] = REQUESTED_DELAY_MS
            case["native_delay_ms"] = delay
        meter = nest.Create("multimeter", params=dict(interval=V1_RESOLUTION_MS, record_from=list(SIGNALS), record_to="memory"))
        spikes = nest.Create("spike_recorder", params=dict(record_to="memory"))
        input_spikes = nest.Create("spike_recorder", params=dict(record_to="memory"))
        nest.Connect(meter, nodes, syn_spec=dict(delay=DEVICE_CONNECTION_DELAY_MS))
        nest.Connect(nodes, spikes, syn_spec=dict(delay=DEVICE_CONNECTION_DELAY_MS))
        for generator in generators.values():
            nest.Connect(generator, input_spikes, syn_spec=dict(delay=DEVICE_CONNECTION_DELAY_MS))
        report["native_simulation_calls"] = 1
        nest.Simulate(DURATION_MS)
        require(float(nest.GetKernelStatus("biological_time")) == DURATION_MS,
                "GLIF_REFERENCE_NATIVE_DURATION_DIFFERS")
        native_ids = np.asarray(ids, dtype=np.int64)
        times = np.arange(1, int(DURATION_MS / V1_RESOLUTION_MS), dtype=np.float64) * V1_RESOLUTION_MS
        recorded_events = nest.GetStatus(meter, "events")[0]
        report.update(kernel_min_delay_ms=float(nest.GetKernelStatus("min_delay")),
                      kernel_max_delay_ms=float(nest.GetKernelStatus("max_delay")),
                      expected_recording_rows=len(native_ids) * len(times),
                      observed_recording_rows=len(recorded_events["times"]))
        matrices = recording_matrices(recorded_events, native_ids, times)
        spike_events = nest.GetStatus(spikes, "events")[0]
        spike_ids = np.asarray(spike_events["senders"], dtype=np.int64)
        spike_times = np.asarray(spike_events["times"], dtype=np.float64)
        require(np.isin(spike_ids, native_ids).all() and np.isfinite(spike_times).all()
                and np.all((spike_times > 0) & (spike_times <= DURATION_MS)),
                "GLIF_REFERENCE_NATIVE_SPIKE_COORDINATES_INVALID")
        input_events = nest.GetStatus(input_spikes, "events")[0]
        input_ids = np.asarray(input_events["senders"], dtype=np.int64)
        input_times = np.asarray(input_events["times"], dtype=np.float64)
        for key, generator in generators.items():
            actual = np.sort(input_times[input_ids == int(generator.tolist()[0])])
            expected = encode_precise_times(protocols[key], tics_per_ms)["decoded_times_ms"]
            require(np.array_equal(actual, expected), "GLIF_REFERENCE_PRECISE_INPUT_REPLAY_DIFFERS:" + key)
        require(len(input_times) == sum(len(value) for value in protocols.values()),
                "GLIF_REFERENCE_EXTRA_INPUT_EVENTS")
        counts = {int(gid): int(count) for gid, count in zip(*np.unique(spike_ids, return_counts=True))}
        burst_models = []
        for index, case in enumerate(cases):
            case["spike_count"] = counts.get(case["native_id"], 0)
            p = references[case["source_index"]]["converted_parameters"]
            if case["kind"] == "passive" and p["V_m"] == p["E_L"] and not np.asarray(p["asc_init"]).any():
                require(case["spike_count"] == 0 and np.array_equal(matrices["V_m"][index], np.full(len(times), initial_vm[index])),
                        "GLIF_REFERENCE_PASSIVE_CONTROL_DIFFERS:" + case["case_id"])
            if case["kind"].startswith("sparse_"):
                current = matrices["I_syn"][index]
                sign = 1 if case["weight_pA"] > 0 else -1
                require(np.all(sign * current >= 0) and np.any(sign * current > 0),
                        "GLIF_REFERENCE_RECEPTOR_RESPONSE_MISSING_OR_WRONG_SIGN:" + case["case_id"])
            if case["kind"] == "burst_r1" and case["spike_count"] >= 2:
                burst_models.append(case["source_index"])
        arrays = dict(native_ids=native_ids, time_ms=times, initial_voltage_mV=initial_vm,
                      terminal_voltage_mV=np.asarray(nest.GetStatus(nodes, "V_m"), dtype=np.float64),
                      initial_ASC_pA=initial_asc,
                      terminal_ASC_pA=np.stack([np.asarray(x, dtype=np.float64) for x in nest.GetStatus(nodes, "ASCurrents")]),
                      spike_native_ids=spike_ids, spike_times_ms=spike_times,
                      input_spike_native_ids=input_ids, input_spike_times_ms=input_times,
                      **{"signal_" + key: value for key, value in matrices.items()})
        artifact = output / "reference_arrays.npz"
        np.savez_compressed(artifact, **arrays)
        with np.load(artifact, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(np.array_equal(saved[key], value) for key, value in arrays.items()),
                    "GLIF_REFERENCE_SERIALIZED_ARRAYS_DIFFER")
        report.update(status="COMPLETE_NATIVE_GLIF_COMPONENT_REFERENCE", source_models=111, component_cases=len(cases), cases=cases,
                      parameter_roundtrip_exact=True, input_connection_readback_exact=True,
                      precise_input_replay_exact=True, recorded_signal_names=list(SIGNALS),
                      signal_samples_per_case=len(times), recorded_signal_values=len(cases) * len(times) * len(SIGNALS),
                      source_models_with_repeated_spikes=len(set(burst_models)),
                      repeated_spike_coverage_complete=len(set(burst_models)) == len(references),
                      native_delay_ms=delay, requested_delay_ms=REQUESTED_DELAY_MS,
                      artifact=artifact.name, artifact_sha256=sha256_file(artifact),
                      initial_and_terminal_states_separate_from_interior_recording=True,
                      observation_scope="Native signal recordings and boundary ASC components; per-receptor hidden PSC states are not directly observed")
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output / "report.json", report)
        shutil.copy2(__file__, output / "v1_glif_component_reference.py")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_reference(args.parameters, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ("cases", "references")}))


if __name__ == "__main__":
    main()

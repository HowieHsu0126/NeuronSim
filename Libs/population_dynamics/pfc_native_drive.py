"""Bounded-memory NumPy drive preserving the pinned PFC source RNG sequence.

A packed block is tick-major, with inhibitory columns before excitatory columns.
Its scalar binomial draws therefore follow the native I-then-E calls at every
clock tick. No neural simulation or generative-model qualification occurs here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from .pfc_input_pairing import _array, _relative, _resolve_cases, _under
from .provenance import freeze_json, sha256_file

SOURCE_ENTRY_SHA256 = "416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720"
_BINOMIAL_N = 1000
_RATE_HZ = 1.4
_MAX_WORKING_BLOCK_BYTES = 256 * 1024 * 1024
_BLOCK_SIZES = (1, 37, 256)


class NativeDrive:
    """One RNG stream, counters and no retained trajectory or previous block."""

    def __init__(self, n_e, n_i, dt_seconds, total_ticks, rng, policy, seeds):
        self.n_e, self.n_i = n_e, n_i
        self.dt_seconds, self.total_ticks = dt_seconds, total_ticks
        self.random_stream_policy, self.seeds = policy, dict(seeds)
        self.probability = _RATE_HZ * dt_seconds
        self.max_rows = _MAX_WORKING_BLOCK_BYTES // (10 * (n_e + n_i))
        if self.max_rows < 1:
            raise ValueError("PFC_DRIVE_SINGLE_ROW_EXCEEDS_MEMORY_BOUND")
        self._rng = rng
        self.ticks = 0
        self._events = {"E": 0, "I": 0}

    @property
    def event_totals(self):
        return dict(self._events)

    @property
    def remaining(self):
        return self.total_ticks - self.ticks

    def take(self, rows):
        if isinstance(rows, (bool, np.bool_)) or not isinstance(rows, (int, np.integer)) or rows <= 0:
            raise ValueError("PFC_DRIVE_ROWS_MUST_BE_POSITIVE_INTEGER")
        rows = int(rows)
        if rows > self.remaining:
            raise ValueError("PFC_DRIVE_REQUEST_EXCEEDS_REMAINING_TICKS")
        if rows > self.max_rows:
            raise ValueError("PFC_DRIVE_REQUEST_EXCEEDS_MEMORY_BOUND")
        # NumPy fills the C-order shape in scalar draw order. With equal n,p for
        # both populations this is exactly I[t], E[t], I[t+1], E[t+1], ... .
        native = self._rng.binomial(_BINOMIAL_N, self.probability,
                                    size=(rows, self.n_i + self.n_e))
        if (native.shape != (rows, self.n_i + self.n_e) or native.dtype.kind not in "iu"
                or int(native.min()) < 0 or int(native.max()) > _BINOMIAL_N):
            raise ValueError("PFC_DRIVE_NATIVE_BINOMIAL_SUPPORT_INVALID")
        packed = native.astype(np.uint16)
        del native
        inhibitory, excitatory = packed[:, :self.n_i], packed[:, self.n_i:]
        self._events["I"] += int(inhibitory.sum(dtype=np.uint64))
        self._events["E"] += int(excitatory.sum(dtype=np.uint64))
        self.ticks += rows
        return {"E": excitatory, "I": inhibitory}

    def get_state(self):
        """Return the standard five-part MT19937 state without advancing it."""
        return self._rng.get_state()

    def metadata(self):
        return dict(source_entry_sha256=SOURCE_ENTRY_SHA256,
                    random_stream_policy=self.random_stream_policy, seeds=dict(self.seeds),
                    rng="numpy.random.RandomState(MT19937)", numpy_version=np.__version__,
                    binomial_n=_BINOMIAL_N, external_rate_Hz=_RATE_HZ,
                    binomial_probability=self.probability, n_e=self.n_e, n_i=self.n_i,
                    dt_seconds=self.dt_seconds, total_ticks=self.total_ticks,
                    ticks_generated=self.ticks, external_event_totals=self.event_totals,
                    initial_draw_order="I_then_E", initial_voltage_archive_order="E_then_I",
                    packed_draw_order="tick_major_I_then_E", returned_dtype="uint16",
                    max_rows=self.max_rows, working_block_bound_bytes=_MAX_WORKING_BLOCK_BYTES,
                    trajectory_retained=False)


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("PFC_DRIVE_SEED_INVALID")
    return value


def create_drive(config, case):
    """Return initial mV values in E/I archive order and a fresh native driver.

    source_global uses one private RandomState for both initial uniform draws
    and subsequent external inputs. The independent policy uses separate streams.
    A resolved case may carry the reference runner's internal source_rng_seed
    alias, provided its independent-policy label and external seed agree.
    """
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    dt_ms, duration_ms = config["dt_ms"], config["duration_ms"]
    if (any(type(n) is not int or n <= 0 for n in (n_e, n_i))
            or any(type(x) not in (int, float) or not np.isfinite(x) or x <= 0
                   for x in (dt_ms, duration_ms))):
        raise ValueError("PFC_DRIVE_GEOMETRY_OR_CLOCK_INVALID")
    dt_seconds = dt_ms * 1e-3
    total_ticks = int(round(duration_ms / dt_ms))
    if total_ticks < 1 or not np.isclose(total_ticks * dt_ms, duration_ms, rtol=0, atol=1e-9):
        raise ValueError("PFC_DRIVE_DURATION_OFF_NATIVE_CLOCK")
    probability = _RATE_HZ * dt_seconds
    if not 0 <= probability <= 1 or (_BINOMIAL_N * probability > 5 and _BINOMIAL_N * (1 - probability) > 5):
        raise ValueError("PFC_DRIVE_NATIVE_NORMAL_APPROXIMATION_UNSUPPORTED")
    policy = config.get("random_stream_policy", "source_global")
    if case.get("random_stream_policy", policy) != policy:
        raise ValueError("PFC_DRIVE_CASE_POLICY_MISMATCH")
    if policy == "source_global":
        if any(key in source for source in (config, case) for key in ("initial_rng_seed", "external_noise_seed")):
            raise ValueError("PFC_DRIVE_INDEPENDENT_SEEDS_REQUIRE_EXPLICIT_POLICY")
        source_seed = _seed(case.get("source_rng_seed", config.get("source_rng_seed")))
        seeds = {"source_rng_seed": source_seed}
        initial_rng = external_rng = np.random.RandomState(source_seed)
    elif policy == "independent_initial_external":
        if "source_rng_seed" in config:
            raise ValueError("PFC_DRIVE_INDEPENDENT_POLICY_REQUIRES_NAMED_SEEDS")
        initial_seed = _seed(case.get("initial_rng_seed", config.get("initial_rng_seed")))
        external_seed = _seed(case.get("external_noise_seed", config.get("external_noise_seed")))
        if ("source_rng_seed" in case and (case.get("random_stream_policy") != policy
                                           or case["source_rng_seed"] != external_seed)):
            raise ValueError("PFC_DRIVE_INTERNAL_EXTERNAL_SEED_ALIAS_MISMATCH")
        seeds = dict(initial_rng_seed=initial_seed, external_noise_seed=external_seed)
        initial_rng, external_rng = np.random.RandomState(initial_seed), np.random.RandomState(external_seed)
    else:
        raise ValueError("PFC_DRIVE_RANDOM_STREAM_POLICY_UNSUPPORTED")
    inhibitory = initial_rng.uniform(-60.0, high=-50.0, size=n_i)
    excitatory = initial_rng.uniform(-60.0, high=-50.0, size=n_e)
    initial = np.concatenate((excitatory, inhibitory))
    driver = NativeDrive(n_e, n_i, dt_seconds, total_ticks, external_rng, policy, seeds)
    return initial, driver


def _state_arrays(state):
    return dict(algorithm=np.array(state[0]), key=np.asarray(state[1], dtype=np.uint32),
                position=np.array(state[2], dtype=np.int64), has_gauss=np.array(state[3], dtype=np.int8),
                cached_gaussian=np.array(state[4], dtype=np.float64))


def _compare_state(state, reference):
    expected_keys = {"algorithm", "key", "position", "has_gauss", "cached_gaussian"}
    if set(reference.files) != expected_keys:
        raise ValueError("PFC_DRIVE_NATIVE_RNG_SNAPSHOT_SCHEMA_CHANGED")
    actual = _state_arrays(state)
    rows = {}
    for key in sorted(expected_keys):
        expected, observed = reference[key], actual[key]
        if expected.dtype != observed.dtype or expected.shape != observed.shape:
            raise ValueError("PFC_DRIVE_NATIVE_RNG_SNAPSHOT_SHAPE_CHANGED:" + key)
        rows[key] = dict(equal=expected.tobytes() == observed.tobytes(),
                         mismatched_values=int(np.count_nonzero(expected != observed)))
        if key in ("position", "has_gauss", "cached_gaussian", "algorithm"):
            rows[key].update(expected=expected.item(), observed=observed.item())
    return dict(equal=all(row["equal"] for row in rows.values()), components=rows)


def verify(validation: Path, pairing_path: Path, output: Path):
    validation, pairing_path, output = Path(validation), Path(pairing_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_NATIVE_NUMPY_DRIVE_REPLAY", created_utc=datetime.now(timezone.utc).isoformat(),
                  implementation_sha256=sha256_file(Path(__file__)), numpy_version=np.__version__,
                  source_entry_sha256=SOURCE_ENTRY_SHA256, block_sizes=list(_BLOCK_SIZES),
                  expected_cases=8, expected_comparisons=24, completed_comparisons=0, passed_comparisons=0,
                  cases=[], consumed_files=[], count_values_compared=0, initial_values_compared=0,
                  scope="native NumPy input-stream reproduction and block-boundary invariance",
                  neural_simulation_performed=False, scientific_fidelity_qualified=False, gpu_parity_qualified=False,
                  mathematical_independence_established=False, biological_independence_established=False,
                  production_episodes=0)
    try:
        pairing = json.loads(pairing_path.read_text())
        delivery_path = validation / "delivery.json"
        delivery = json.loads(delivery_path.read_text())
        report.update(pairing_sha256=sha256_file(pairing_path), delivery_sha256=sha256_file(delivery_path))
        if (pairing["status"] != "PASS_NATIVE_INPUT_PAIRING" or pairing["input_identity_mapping_complete"] is not True
                or not all(value is True for value in pairing["gates"].values())
                or pairing["delivery_sha256"] != report["delivery_sha256"]
                or delivery["status"] != "ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY"):
            raise ValueError("PFC_DRIVE_PAIRED_SOURCE_CAPTURE_REQUIRED")
        declared = {_relative(row["path"]): row for row in delivery["files"]}
        if len(declared) != len(delivery["files"]):
            raise ValueError("PFC_DRIVE_DUPLICATE_ARCHIVE_PATH")
        checked = {}

        def consume(name, digest=None):
            name = _relative(name)
            row = declared[name]
            if digest is not None and row["sha256"] != digest:
                raise ValueError("PFC_DRIVE_SOURCE_LINEAGE_CHANGED:" + name)
            if name not in checked:
                path = validation / name
                if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                    raise ValueError("PFC_DRIVE_SOURCE_CONTENT_CHANGED:" + name)
                checked[name] = dict(path=name, sha256=row["sha256"], bytes=row["bytes"])
                report["consumed_files"].append(checked[name])
            return validation / name

        def load(name, digest=None):
            return json.loads(consume(name, digest).read_text())

        prior = load("comparison/comparison.json")
        execution = load("capture/execution.json", prior["capture_execution_sha256"])
        native = load("capture/native/report.json", execution["native_report_sha256"])
        inputs = load("capture/inputs/report.json", execution["observation_report_sha256"])
        if (prior != delivery["comparison"] or prior["status"] != "PASS_NATIVE_INPUT_OBSERVATION_PARITY"
                or native["status"] != "NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
                or execution["status"] != "RECORDED_NATIVE_PFC_INPUT_OBSERVATION" or execution["mode"] != "capture"
                or inputs["status"] != "NATIVE_SOURCE_INPUT_OBSERVATION_COMPLETE" or inputs["mode"] != "capture"
                or any(value["source_entry_sha256"] != SOURCE_ENTRY_SHA256 for value in (native, inputs, pairing))
                or execution["config_sha256"] != pairing["config_sha256"]
                or native["brian2_version"] != "2.5.4" or inputs["brian2_version"] != "2.5.4"):
            raise ValueError("PFC_DRIVE_PINNED_NATIVE_OBSERVATION_REQUIRED")
        consume(pairing["config_file"], pairing["config_sha256"])
        config = native["resolved_config"]
        cases = _resolve_cases(config)
        ids = [case["case_id"] for case in cases]
        if (len(cases) != 8 or len(set(ids)) != 8 or execution["cases"] != 8
                or any([row["case_id"] for row in obj["cases"]] != ids for obj in (native, inputs, prior))
                or [row["case"] for row in pairing["cases"]] != cases):
            raise ValueError("PFC_DRIVE_COMPLETE_SOURCE_CASE_SET_REQUIRED")
        report.update(config_sha256=execution["config_sha256"], source_numpy_version=native["numpy_version"],
                      observed_random_stream_policy=config["random_stream_policy"],
                      source_global_native_qualification=False,
                      case_ids=ids, expected_count_values=3 * prior["captured_draw_values_recounted"])
        freeze_json(output / "run.json", {key: value for key, value in report.items() if key not in ("cases", "consumed_files")})
        records = []
        for index, case in enumerate(cases):
            row, truth, paired = inputs["cases"][index], native["cases"][index], pairing["cases"][index]
            receipt = load(_under("capture/inputs", row["receipt_file"]), row["receipt_sha256"])
            if (receipt["case"] != case or receipt["mode"] != "capture"
                    or receipt["status"] != "NATIVE_INPUT_OBSERVATION_CASE_COMPLETE"
                    or paired["validation_status"] != "verified"):
                raise ValueError("PFC_DRIVE_NATIVE_CASE_BINDING_CHANGED")
            for key in ("initial_rng_seed", "external_noise_seed", "source_rng_seed", "stimulus_strength_nA", "distractor_strength_nA"):
                if truth[key] != case[key]:
                    raise ValueError("PFC_DRIVE_NATIVE_CASE_PARAMETER_CHANGED:" + key)
            initial_name = _under("capture/native", truth["initial_voltage_file"])
            if paired["initial_voltage_file"] != initial_name or paired["initial_voltage_sha256"] != truth["initial_voltage_sha256"]:
                raise ValueError("PFC_DRIVE_NATIVE_INITIAL_BINDING_CHANGED")
            initial_path = consume(initial_name, truth["initial_voltage_sha256"])
            rng_name = _under("capture/inputs", row["numpy_rng_file"])
            if receipt["numpy_rng_file"] != row["numpy_rng_file"] or receipt["numpy_rng_sha256"] != row["numpy_rng_sha256"]:
                raise ValueError("PFC_DRIVE_NATIVE_RNG_BINDING_CHANGED")
            rng_path = consume(rng_name, row["numpy_rng_sha256"])
            providers = {p["population"]: p for p in receipt["providers"]}
            if len(receipt["providers"]) != 2 or set(providers) != {"E", "I"}:
                raise ValueError("PFC_DRIVE_NATIVE_PROVIDER_SET_CHANGED")
            paths = {}
            for population in ("E", "I"):
                provider, mapping = providers[population], paired["populations"][population]
                name = _under("capture/inputs", provider["counts_file"])
                if mapping["path"] != name or mapping["sha256"] != provider["counts_sha256"]:
                    raise ValueError("PFC_DRIVE_NATIVE_COUNTS_BINDING_CHANGED")
                paths[population] = consume(name, provider["counts_sha256"])
            records.append(dict(case=case, raw_case=config["cases"][index], initial_path=initial_path,
                                rng_path=rng_path, rng_sha256=row["numpy_rng_sha256"], providers=providers,
                                paths=paths))
        for block_size in _BLOCK_SIZES:
            for record in records:
                case = record["case"]
                initial, drive = create_drive(config, record["raw_case"])
                if block_size > drive.max_rows:
                    raise ValueError("PFC_DRIVE_VERIFICATION_BLOCK_EXCEEDS_BOUND")
                for population, units in (("E", drive.n_e), ("I", drive.n_i)):
                    provider = record["providers"][population]
                    if (provider["counts_shape"] != [drive.total_ticks, units]
                            or provider["expected_sample_rows"] != drive.total_ticks
                            or provider["observed_sampler_calls"] != drive.total_ticks
                            or provider["units"] != units or provider["dt_seconds"] != drive.dt_seconds
                            or provider["N"] != _BINOMIAL_N or provider["rate_Hz"] != _RATE_HZ
                            or provider["probability"] != drive.probability
                            or provider["target_variable"] != "s_AMPA" or provider["weight_dimensionless"] != 1.0
                            or provider["source_unit_order"] != "ascending_zero_based_population_index"):
                        raise ValueError("PFC_DRIVE_SOURCE_INPUT_GEOMETRY_CHANGED")
                with _array(record["initial_path"], (drive.n_e + drive.n_i,), np.float64) as original:
                    initial_equal = np.array_equal(initial, original)
                    initial_mismatches = int(np.count_nonzero(initial != original))
                values_compared = {"E": 0, "I": 0}
                mismatches, expected_events = {"E": 0, "I": 0}, {"E": 0, "I": 0}
                equal, examples = {"E": True, "I": True}, []
                with _array(record["paths"]["E"], (drive.total_ticks, drive.n_e), np.uint16) as e:
                    with _array(record["paths"]["I"], (drive.total_ticks, drive.n_i), np.uint16) as i:
                        truth = {"E": e, "I": i}
                        while drive.remaining:
                            start = drive.ticks
                            count = min(block_size, drive.remaining)
                            generated = drive.take(count)
                            for population in ("E", "I"):
                                observed, original = generated[population], truth[population][start:start + count]
                                if int(original.max()) > _BINOMIAL_N:
                                    raise ValueError("PFC_DRIVE_NATIVE_CAPTURE_SUPPORT_INVALID")
                                same = np.array_equal(observed, original)
                                equal[population] = equal[population] and same
                                values_compared[population] += int(original.size)
                                expected_events[population] += int(original.sum(dtype=np.uint64))
                                if not same:
                                    different = observed != original
                                    mismatches[population] += int(np.count_nonzero(different))
                                    if len(examples) < 8:
                                        for position in np.argwhere(different)[:8 - len(examples)]:
                                            t, unit = (int(x) for x in position)
                                            examples.append(dict(population=population, tick=start + t, unit_index=unit,
                                                                 expected=int(original[t, unit]), observed=int(observed[t, unit])))
                state = drive.get_state()
                with np.load(record["rng_path"], allow_pickle=False) as original_state:
                    rng_comparison = _compare_state(state, original_state)
                totals_valid = drive.event_totals == expected_events
                if any(expected_events[p] != record["providers"][p]["captured_external_events"] for p in ("E", "I")):
                    raise ValueError("PFC_DRIVE_NATIVE_CAPTURE_EVENT_TOTAL_CHANGED")
                complete = drive.ticks == drive.total_ticks and all(
                    values_compared[p] == drive.total_ticks * (drive.n_e if p == "E" else drive.n_i) for p in ("E", "I"))
                passed = bool(initial_equal and all(equal.values()) and rng_comparison["equal"] and totals_valid and complete)
                stem = case["case_id"] + "_block_" + str(block_size)
                state_file = output / (stem + "_rng.npz")
                with state_file.open("xb") as stream:
                    np.savez(stream, **_state_arrays(state))
                result = dict(case_id=case["case_id"], block_rows=block_size, passed=passed,
                              initial_equal=bool(initial_equal), initial_mismatched_values=initial_mismatches,
                              initial_values_compared=int(initial.size), populations={p: dict(
                                  equal=bool(equal[p]), mismatched_values=mismatches[p], values_compared=values_compared[p],
                                  expected_events=expected_events[p], generated_events=drive.event_totals[p]) for p in ("E", "I")},
                              final_rng=rng_comparison, complete_tick_consumption=complete,
                              expected_ticks=drive.total_ticks, generated_ticks=drive.ticks,
                              first_mismatch_examples=examples, driver=drive.metadata(),
                              generated_rng_file=state_file.name, generated_rng_sha256=sha256_file(state_file),
                              reference_rng_sha256=record["rng_sha256"], pairing_sha256=report["pairing_sha256"],
                              delivery_sha256=report["delivery_sha256"], implementation_sha256=report["implementation_sha256"],
                              scientific_fidelity_qualified=False, production_episodes=0)
                receipt_path = output / (stem + ".json")
                freeze_json(receipt_path, result)
                report["cases"].append(dict(result, receipt_file=receipt_path.name, receipt_sha256=sha256_file(receipt_path)))
                report["completed_comparisons"] += 1
                report["passed_comparisons"] += int(passed)
                report["count_values_compared"] += sum(values_compared.values())
                report["initial_values_compared"] += int(initial.size)
                print(json.dumps(dict(case_id=case["case_id"], block_rows=block_size, passed=passed,
                                      mismatches=mismatches, final_rng_equal=rng_comparison["equal"])), flush=True)
        report["status"] = ("PASS_NATIVE_NUMPY_DRIVE_REPLAY" if report["passed_comparisons"] == 24
                            and report["count_values_compared"] == report["expected_count_values"]
                            else "FAIL_NATIVE_NUMPY_DRIVE_REPLAY")
    except Exception as error:
        report["status"] = "FAILED_NATIVE_NUMPY_DRIVE_REPLAY"
        report["failure"] = dict(type=type(error).__name__, message=str(error))
    with (output / "drive_comparisons.csv").open("x", encoding="utf-8", newline="") as stream:
        columns = ("case_id", "block_rows", "passed", "population", "equal", "mismatched_values", "values_compared",
                   "expected_events", "generated_events", "initial_equal", "final_rng_equal", "complete_tick_consumption")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in report["cases"]:
            for population, values in case["populations"].items():
                writer.writerow(dict(case_id=case["case_id"], block_rows=case["block_rows"], passed=case["passed"],
                                     population=population, **values, initial_equal=case["initial_equal"],
                                     final_rng_equal=case["final_rng"]["equal"],
                                     complete_tick_consumption=case["complete_tick_consumption"]))
    report["not_completed_comparisons"] = report["expected_comparisons"] - report["completed_comparisons"]
    report["comparison_table"] = "drive_comparisons.csv"
    report["comparison_table_sha256"] = sha256_file(output / "drive_comparisons.csv")
    freeze_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.validation, args.pairing, args.output)
    print(json.dumps({key: report.get(key) for key in
                     ("status", "expected_comparisons", "completed_comparisons", "passed_comparisons", "failure")}, indent=2))
    return 0 if report["status"] == "PASS_NATIVE_NUMPY_DRIVE_REPLAY" else 1


if __name__ == "__main__":
    sys.exit(main())

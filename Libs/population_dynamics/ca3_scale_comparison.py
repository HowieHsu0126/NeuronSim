"""Compare already replayed CA3 source readouts across two unpaired sizes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .ca3_retrieval_readouts import AUTHOR_MAIN_SHA256, SETUP_SHA256, source_identity
from .provenance import sha256_file


MEMORY_IMPLEMENTATION = "5f4b0ae7d61a9453338fbd72aedb2bf138d34cf9e3d95880a54249933c724153"
READOUT_REPLAY_IMPLEMENTATION = "a694fc32e782fe354ecffe6c71579f1b5abe96e83854972533dde5dcee0bfc7a"
ANALYSIS = "analysis.json"
READOUT = "primary_readouts/readouts/report.json"
CSV = "primary_readouts/readouts/descriptive_readouts.csv"
REPLAY = "primary_readouts/readout_replay.json"
RECONCILIATION = "reconciliation.json"
CONTROL = "control/report.json"
CONSUMED = (ANALYSIS, READOUT, CSV, REPLAY, RECONCILIATION, CONTROL)
WINDOWS = ((500, 520, 520, 540), (730, 750, 750, 770), (960, 980, 980, 1000))
FOLLOWING_MS = (210, 210, 1)
IDENTITY_COLUMNS = ["cue_id", "assembly_local_start", "assembly_global_start",
                    "baseline_start_ms", "baseline_stop_ms", "cue_start_ms", "cue_stop_ms",
                    "following_source_background_ms"]
SUBSET_FIELDS = ["units", "baseline_event_count", "cue_event_count", "baseline_rate_hz", "cue_rate_hz",
                 "baseline_active_units", "cue_active_units", "baseline_spiking_fraction", "cue_spiking_fraction",
                 "delta_rate_hz", "delta_spiking_fraction"]
SUBSETS = ("target", "other", "eligible")
CONTRASTS = [f"target_minus_other_{phase}_{metric}" for metric in ("rate_hz", "spiking_fraction")
             for phase in ("baseline", "cue", "delta")]
COLUMNS = IDENTITY_COLUMNS + [f"{subset}_{field}" for subset in SUBSETS for field in SUBSET_FIELDS] + CONTRASTS
METRICS = [f"{subset}_{phase}_{metric}" for subset in SUBSETS for phase in ("baseline", "cue", "delta")
           for metric in ("rate_hz", "spiking_fraction")] + CONTRASTS
FLAGS = ("actual_current_trace_verified", "native_internal_spike_counter_verified",
         "terminal_native_clock_independently_verified", "observation_source_execution_equivalence_qualified")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Object receipt required: {path}")
    return value


def _entry(delivery: dict, name: str) -> dict:
    entries = [row for row in delivery["files"] if row["path"] == name]
    _require(len(entries) == 1, f"Exactly one delivery binding required: {name}")
    return entries[0]


def _bound(root: Path, delivery: dict, name: str) -> tuple[Path, str]:
    path, row = root / name, _entry(delivery, name)
    _require(not path.is_symlink() and path.is_file(), f"Regular consumed file required: {name}")
    digest = sha256_file(path)
    _require(path.stat().st_size == row["bytes"] and digest == row["sha256"], f"Consumed file changed: {name}")
    return path, digest


def _delivery_valid(delivery: dict, status: str) -> None:
    _require(delivery["status"] == status and delivery["exit_code"] == 0
             and delivery["copied_files_verified"] is True, "Completed verified delivery required")


def _native_inputs(analysis: dict, proof: dict, identity: dict, coupled: bool) -> tuple[dict, dict]:
    """Only small native metadata is read; declared large payload hashes are reused."""
    family = "COUPLED_EXTENSION" if coupled else "NATIVE"
    deliveries, bindings = {}, {}
    for role, proof_key in (("primary", "original_report_sha256"), ("observed", "observed_report_sha256")):
        entry = analysis["input_archives"][role]
        root = Path(entry["path"])
        _require(sha256_file(root / "delivery.json") == entry["delivery_sha256"], "Native delivery binding changed")
        delivery = _json(root / "delivery.json")
        _delivery_valid(delivery, ("COUPLED_" if coupled else "NATIVE_REFERENCE_")
                        + "RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED")
        path, digest = _bound(root, delivery, "native_report.json")
        native = _json(path)
        _require(digest == entry["native_report_sha256"] == proof[proof_key], "Native/reconciliation identity mismatch")
        _require(native["status"] == family + "_RETRIEVAL_EXIT_ZERO" and native["exit_code"] == 0
                 and native["native_run_executed"] is True and native["retrieval_executed"] is True
                 and native["normalized_input_qualified"] is True, "Completed source retrieval required")
        _require(native["scope"] == ("coupled_extension_retrieval" if coupled else "native_source_retrieval")
                 and native["core_neurons"] == identity["core_units"]
                 and native["populations"] == identity["populations"]
                 and native["expected_duration_ms"] == 1001, "Source population/scope mismatch")
        _require(native.get("retrieval_condition", "source") == "source", "Source condition required")
        _require(native["integration"] == "RUNGE_KUTTA4_5" and native["delay_ms"] == 1,
                 "Matched RK4/5 and delay required")
        contract = native["retrieval_input_contract"]
        _require(contract["source_carlsim_seed"] == 10 and contract["source_host_srand_seed"] == 3
                 and contract["source_estdp_enabled"] is True
                 and contract["effective_input_binding"] == identity["input_network"], "Source seed/input contract mismatch")
        _require(native["conductance_memory_implementation_sha256"] == MEMORY_IMPLEMENTATION,
                 "Native conductance correction mismatch")
        _require(native["normalized_input_sha256"] == proof["normalized_input_sha256"]
                 and native["pattern_sha256"] == identity["pattern_sha256"], "Source input identity mismatch")
        if role == "observed":
            _require(native["native_observation_sha256"] == proof["native_observation_sha256"], "Observed trace binding mismatch")
        deliveries[role] = delivery
        bindings[role] = dict(archive=str(root), delivery_sha256=entry["delivery_sha256"], native_report_sha256=digest,
                              integration=native["integration"], source_rng=native["source_rng"],
                              source_carlsim_seed=10, source_host_srand_seed=3)
    return deliveries, bindings


def _csv_rows(path: Path, identity: dict, pyramidal_start: int) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == 47
                 and set(reader.fieldnames) == set(COLUMNS), "47-column readout schema required")
        rows = list(reader)
    _require(len(rows) == 3, "All three cue rows required")
    seen = set()
    denominators = dict(target=identity["assembly_units"] - identity["eligible_units"],
                        other=2 * (identity["assembly_units"] - identity["eligible_units"]),
                        eligible=identity["eligible_units"])
    for row in rows:
        _require(set(row) == set(COLUMNS) and None not in row.values(), "Malformed cue row")
        values = {key: float(value) for key, value in row.items()}
        _require(all(math.isfinite(value) for value in values.values()), "Nonfinite cue value")
        cue = values["cue_id"]
        _require(cue in (1, 2, 3) and cue not in seen, "Missing or duplicate cue identity")
        seen.add(cue)
        index = int(cue) - 1
        expected = dict(zip(("baseline_start_ms", "baseline_stop_ms", "cue_start_ms", "cue_stop_ms"), WINDOWS[index]))
        expected.update(assembly_local_start=identity["assembly_starts"][index],
                        assembly_global_start=pyramidal_start + identity["assembly_starts"][index],
                        following_source_background_ms=FOLLOWING_MS[index])
        _require(all(values[key] == value for key, value in expected.items()), "Cue schedule/identity mismatch")
        for subset, denominator in denominators.items():
            _require(values[subset + "_units"] == denominator, "Subset denominator mismatch")
            for phase in ("baseline", "cue"):
                count, active = (values[f"{subset}_{phase}_{field}"] for field in ("event_count", "active_units"))
                _require(count >= 0 and count.is_integer() and active.is_integer()
                         and 0 <= active <= min(denominator, count), "Invalid count/active-unit denominator")
        # Finite signed deltas and zero values are retained verbatim. Arithmetic
        # agreement was already checked by the bound 141-scalar independent replay.
    return sorted(rows, key=lambda row: int(row["cue_id"]))


def _load(root: Path, coupled: bool) -> tuple[dict, list[dict], list[dict]]:
    root = Path(root)
    cohort = "coupled_extension" if coupled else "native_reference"
    identity = source_identity(coupled_extension=coupled)
    units = identity["core_units"]
    delivery = _json(root / "delivery.json")
    status = ("PASS_COUPLED_" if coupled else "PASS_NATIVE_REFERENCE_") + "OBSERVATION_RECONCILIATION_AND_CONTROL"
    _delivery_valid(delivery, status)
    paths, digests = {}, {}
    for name in CONSUMED:
        paths[name], digests[name] = _bound(root, delivery, name)
    analysis, readouts, replay, proof, control = (_json(paths[name]) for name in (ANALYSIS, READOUT, REPLAY, RECONCILIATION, CONTROL))
    _require(analysis["status"] == status and analysis["exit_code"] == 0 and analysis["units"] == units
             and analysis["duration_ms"] == 1001 and analysis["descriptive_readouts_independently_replayed"] is True,
             "Completed readout/observation analysis required")
    _require(all(analysis[key] is True and proof[key] is True for key in FLAGS), "Clock/current verification missing")
    _require(proof["status"] == ("PASS_COUPLED_EXTENSION_" if coupled else "PASS_NATIVE_") + "CA3_OBSERVATION_REPLAY"
             and proof["units"] == units and proof["duration_ms"] == 1001 and proof["event_multisets_equal"] is True
             and proof["native_intervals_compared"] == 1001
             and proof["unit_millisecond_counts_compared"] == units * 1001, "Complete reconciled clock/counters required")
    _require(proof["helper_implementation_sha256"]["ca3_conductance_memory"] == MEMORY_IMPLEMENTATION,
             "Reconciled conductance correction mismatch")
    for role in ("original", "observed"):
        correction = proof["source_overlay_verification"][role]
        _require(correction["conductance_memory_enabled"] is True
                 and correction["memory_receipt_and_implementation_verified"] is True
                 and correction["implementation_sha256"] == MEMORY_IMPLEMENTATION, "Verified conductance correction required")
    native_deliveries, native_bindings = _native_inputs(analysis, proof, identity, coupled)
    expected_readout = "DESCRIBED_COUPLED_EXTENSION_CA3_RETRIEVAL_EVENTS" if coupled else "DESCRIBED_SOURCE_SCHEDULED_CA3_RETRIEVAL_EVENTS"
    _require(readouts["status"] == expected_readout and readouts["duration_ms"] == 1001
             and readouts["time_resolution_ms"] == 1 and readouts["cues_retained"] == 3
             and readouts["full_record_recounted_units"] == units
             and readouts["window_counted_pyramidal_units"] == identity["populations"]["CA3_Pyramidal"]
             and readouts["full_record_unit_counts_match"] is True
             and readouts["full_record_population_counts_per_ms_match"] is True, "Complete source readouts required")
    _require(readouts["source_scheduled_window_start_ms"] == [[row[0], row[2]] for row in WINDOWS]
             and readouts["source_scheduled_window_stop_ms"] == [[row[1], row[3]] for row in WINDOWS], "Readout window mismatch")
    _require(readouts["target_never_boosted_units"] == identity["assembly_units"] - identity["eligible_units"]
             and readouts["other_never_boosted_units"] == 2 * (identity["assembly_units"] - identity["eligible_units"])
             and readouts["eligible_for_boost_units"] == identity["eligible_units"], "Readout subset denominator mismatch")
    _require(readouts["normalized_input_sha256"] == proof["normalized_input_sha256"]
             and readouts["native_report_sha256"] == proof["original_report_sha256"]
             and readouts["recording_sha256"] == proof["original_recording_sha256"], "Readout source/recording mismatch")
    for name, digest in readouts["input_sha256"].items():
        _require(_entry(native_deliveries["primary"], name)["sha256"] == digest, "Readout native input binding mismatch: " + name)
    _require(readouts["pattern_sha256"] == identity["pattern_sha256"]
             and readouts["overlay_source_sha256"] == identity["overlay_sha256"]
             and readouts["source_config_sha256"] == identity["config_sha256"]
             and readouts["original_author_main_sha256"] == AUTHOR_MAIN_SHA256
             and readouts["source_setup_sha256"] == SETUP_SHA256, "Source measurement program mismatch")
    _require(proof["helper_implementation_sha256"]["ca3_retrieval_readouts"] == readouts["implementation_sha256"],
             "Within-scale readout producer binding mismatch")
    _require(readouts["artifacts"]["descriptive_readouts.csv"] == digests[CSV], "Readout CSV binding mismatch")
    _require(replay["status"] == ("PASS_COUPLED_EXTENSION_RAW_EVENT_READOUT_REPLAY" if coupled else "PASS_INDEPENDENT_RAW_EVENT_READOUT_REPLAY")
             and replay["source_report_sha256"] == digests[READOUT]
             and replay["implementation_sha256"] == READOUT_REPLAY_IMPLEMENTATION
             and replay["all_three_cues_retained"] is True and replay["scalar_values_compared"] == 141
             and replay["window_count_values_compared"] == 6 * identity["populations"]["CA3_Pyramidal"],
             "Complete independent 141-scalar readout replay required")
    reuse = analysis["readout_reuse"]
    _require(reuse["primary_report_sha256"] == proof["original_report_sha256"]
             and reuse["observed_report_sha256"] == proof["observed_report_sha256"]
             and reuse["reconciliation_sha256"] == digests[RECONCILIATION]
             and reuse["original_reports_preserved"] is True, "Readout reuse identity mismatch")
    for name in (READOUT, CSV, REPLAY):
        _require(reuse["artifacts"][name.removeprefix("primary_readouts/")] == digests[name], "Reused artifact binding mismatch")
    _require(control["status"] == ("VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY" if coupled else "VERIFIED_SOURCE_CA3_CONTROL_DELIVERY")
             and control["units"] == units and control["intervals_verified"] == 1001
             and control["actual_current_values_checked"] == units * 1001
             and control["source_executions"] == 1 and control["source_cues"] == 3, "Complete source control required")
    control_inputs = control["input_sha256"]
    _require(control_inputs["reconciliation"] == digests[RECONCILIATION]
             and control_inputs["native_report.json"] == proof["observed_report_sha256"]
             and control_inputs["native_observation.bin"] == proof["native_observation_sha256"]
             and control_inputs["trainPat.csv"] == identity["pattern_sha256"]
             and control_inputs["main_source"] == identity["overlay_sha256"]
             and control_inputs["config_source"] == identity["config_sha256"], "Control source binding mismatch")
    _require(control["current_units"]["model_specific_physical_unit"] == "pA"
             and control["numeric_conversion_applied"] is False
             and control["observer_current_units"] == proof["current_units"], "Physical current/observer annotation mismatch")
    observation = control["observation_map"]
    _require(observation["time_unit"] == "ms" and observation["interval_ms"] == 1, "Control observation clock mismatch")
    groups = {row["name"]: row for row in observation["groups"]}
    _require(len(groups) == len(observation["groups"]) == 8
             and {name: row["stop_exclusive"] - row["start"] for name, row in groups.items()} == identity["populations"],
             "Control population identity mismatch")
    checked = {row["group"]: row for row in readouts["checked_event_files"]}
    _require(len(checked) == len(readouts["checked_event_files"]) == 8
             and set(checked) == set(identity["populations"]), "All eight population readouts required")
    populations = []
    for name in sorted(checked):
        row = checked[name]
        _require(row["units"] == identity["populations"][name] and type(row["records"]) is int and row["records"] >= 0
                 and row["full_record_counts_exact"] is True
                 and row["relative_path"] == "native_run/results/spk_" + name + ".dat"
                 and row["sha256"] == _entry(native_deliveries["primary"], row["relative_path"])["sha256"],
                 "Population readout denominator/binding mismatch")
        populations.append(dict(cohort=cohort, network_units=units, population=name, population_units=row["units"],
                                duration_ms=1001, event_count=row["records"], mean_rate_hz=row["records"] / (row["units"] * 1.001)))
    pyramidal = checked["CA3_Pyramidal"]
    _require(sum(row["records"] for row in checked.values()) == readouts["raw_records"] == proof["events_compared"]
             and pyramidal["records"] == readouts["pyramidal_records"] == replay["pyramidal_events"]
             and pyramidal["sha256"] == replay["raw_events_sha256"], "Population event denominator mismatch")
    rows = _csv_rows(paths[CSV], identity, groups["CA3_Pyramidal"]["start"])
    binding = dict(cohort=cohort, archive=str(root.resolve()), delivery_sha256=sha256_file(root / "delivery.json"),
                   consumed_sha256=digests, native_inputs=native_bindings, network_units=units,
                   producer_implementation_sha256=readouts["implementation_sha256"],
                   independent_readout_replay_implementation_sha256=replay["implementation_sha256"],
                   current_units=control["current_units"], observer_current_units=proof["current_units"],
                   synthetic_fixture=delivery.get("synthetic_fixture", False))
    return binding, rows, populations


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compare(reference: Path, coupled: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    loaded = [_load(reference, False), _load(coupled, True)]
    combined = [dict(row, cohort=binding["cohort"], network_units=binding["network_units"])
                for binding, rows, _ in loaded for row in rows]
    differences = [dict(cue_id=cue + 1, metric=metric, native_value=float(loaded[0][1][cue][metric]),
                        coupled_value=float(loaded[1][1][cue][metric]),
                        coupled_minus_native=float(loaded[1][1][cue][metric]) - float(loaded[0][1][cue][metric]))
                   for cue in range(3) for metric in METRICS]
    populations = [row for _, _, rows in loaded for row in rows]
    population_differences = [dict(population=left["population"], native_mean_rate_hz=left["mean_rate_hz"],
                                   coupled_mean_rate_hz=right["mean_rate_hz"],
                                   coupled_minus_native=right["mean_rate_hz"] - left["mean_rate_hz"])
                              for left, right in zip(loaded[0][2], loaded[1][2])]
    _require(all(math.isfinite(row["coupled_minus_native"]) for row in differences + population_differences), "Nonfinite scale difference")
    output.mkdir(parents=True, exist_ok=False)
    tables = {"cue_readouts.csv": (COLUMNS + ["cohort", "network_units"], combined),
              "cue_differences.csv": (list(differences[0]), differences),
              "population_readouts.csv": (list(populations[0]), populations),
              "population_differences.csv": (list(population_differences[0]), population_differences)}
    for name, (columns, rows) in tables.items():
        _write_csv(output / name, columns, rows)
    report = dict(status="DESCRIPTIVE_REPLAYED_CA3_SCALE_COMPARISON", inputs=[row[0] for row in loaded],
        cue_rows=6, cue_difference_rows=len(differences), population_rows=16, population_difference_rows=8,
        source_executions_per_scale=1, source_executions_reused=2, new_source_executions=0, production_episodes=0,
        paired_across_sizes=False, scientific_fidelity_qualified=False, source_scale_qualified=False,
        source_scaling_qualified=False, pattern_completion_qualified=False, neuroscientific_claim_qualified=False,
        causal_claim_qualified=False, independent_noise_qualified=False, full_scale_qualified=False,
        inferential_statistics_performed=False, p_values=False, confidence_intervals=False,
        synthetic_inputs=any(row[0]["synthetic_fixture"] for row in loaded),
        measurement_contract=dict(duration_ms=1001, cue_windows_ms=[list(row) for row in WINDOWS],
            subset_window_seconds=.020, population_rate_denominator_seconds=1.001,
            following_source_background_ms=list(FOLLOWING_MS), correction_implementation_sha256=MEMORY_IMPLEMENTATION,
            matching_keys=["cue_id", "population"], delta="coupled minus native", cue_averaging_performed=False),
        limitations=["One configured source execution per scale; cues and neurons are not independent replicates.",
            "Size, connectivity, assembly size and random realization differ; differences are not pure size causality.",
            "Source seeds10/3 do not establish matched random draws or independent noise realizations.",
            "The third cue has only1ms of following source background; all three cues remain visible.",
            "Previously verified readouts and physical current annotations are reused; raw events and traces are not reread."],
        outputs={name: dict(sha256=sha256_file(output / name), bytes=(output / name).stat().st_size) for name in tables})
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "coupled", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.coupled, args.output)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        parser.exit(2, f"CA3 scale comparison failed: {error}\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

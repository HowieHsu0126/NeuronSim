"""Self-contained synthetic receipts test comparison contracts, not CA3 biology."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from . import ca3_scale_comparison as comparison
from .ca3_retrieval_readouts import AUTHOR_MAIN_SHA256, SETUP_SHA256, source_identity
from .provenance import sha256_file


def test_cli_executes_and_rejects_changed_inputs(tmp_path):
    reference = archive(tmp_path / "reference_cli", 1)
    coupled = archive(tmp_path / "coupled_cli", 2)
    output = tmp_path / "cli_output"
    command = [sys.executable, "-B", "-m", "population_dynamics.ca3_scale_comparison",
               "--reference", str(reference), "--coupled", str(coupled), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "report.json").read_text())["cue_difference_rows"] == 72
    (reference / comparison.CSV).write_text("changed input\n")
    command[-1] = str(tmp_path / "rejected_output")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Consumed file changed" in result.stderr
    assert not Path(command[-1]).exists()


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def file_row(root, relative):
    path = root / relative
    return dict(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path))


def cue_rows(scale):
    identity = source_identity(coupled_extension=scale == 2)
    rows = []
    for cue, window in enumerate(comparison.WINDOWS, 1):
        row = dict(cue_id=cue, assembly_local_start=(cue - 1) * 275 * scale,
                   assembly_global_start=(cue - 1) * 275 * scale,
                   baseline_start_ms=window[0], baseline_stop_ms=window[1],
                   cue_start_ms=window[2], cue_stop_ms=window[3],
                   following_source_background_ms=comparison.FOLLOWING_MS[cue - 1])
        for subset, units in (("target", 137 * scale), ("other", 274 * scale), ("eligible", 138 * scale)):
            baseline, during = 2 * scale, (3 + cue) * scale
            if scale == 2 and subset == "target" and cue == 1:
                during = scale  # Negative within-run and between-size contrasts remain visible.
            if scale == 2 and subset == "eligible" and cue == 3:
                during += scale
            row[subset + "_units"] = units
            for phase, count in (("baseline", baseline), ("cue", during)):
                row[f"{subset}_{phase}_event_count"] = count
                row[f"{subset}_{phase}_active_units"] = count
                row[f"{subset}_{phase}_rate_hz"] = count / (units * .020)
                row[f"{subset}_{phase}_spiking_fraction"] = count / units
            for metric in ("rate_hz", "spiking_fraction"):
                row[f"{subset}_delta_{metric}"] = row[f"{subset}_cue_{metric}"] - row[f"{subset}_baseline_{metric}"]
        for metric in ("rate_hz", "spiking_fraction"):
            for phase in ("baseline", "cue", "delta"):
                row[f"target_minus_other_{phase}_{metric}"] = row[f"target_{phase}_{metric}"] - row[f"other_{phase}_{metric}"]
        assert row["assembly_local_start"] == identity["assembly_starts"][cue - 1]
        rows.append(row)
    return rows


def archive(root: Path, scale: int, mutate=None):
    root.mkdir()
    coupled = scale == 2
    identity = source_identity(coupled_extension=coupled)
    units = identity["core_units"]
    family = "COUPLED_EXTENSION" if coupled else "NATIVE"
    status = ("PASS_COUPLED_" if coupled else "PASS_NATIVE_REFERENCE_") + "OBSERVATION_RECONCILIATION_AND_CONTROL"
    normalized, trace = digest(f"normalized-{scale}"), digest(f"trace-{scale}")
    producer = ("e430e67dd3dd6ca04500aea9b62bf73b1cd356f3e78ae49aac8a9cd843cc6e62" if coupled
                else "2636708503bc52c64f36d2e27f41b2483c320ca3bddc3320b4865dd26a17c4b1")
    native = {}
    for role in ("primary", "observed"):
        native[role] = dict(status=family + "_RETRIEVAL_EXIT_ZERO", exit_code=0, synthetic_fixture=True,
            native_run_executed=True, retrieval_executed=True, normalized_input_qualified=True,
            scope="coupled_extension_retrieval" if coupled else "native_source_retrieval",
            core_neurons=units, populations=identity["populations"], expected_duration_ms=1001,
            retrieval_condition="source", integration="RUNGE_KUTTA4_5", delay_ms=1,
            source_rng="synthetic configured source seeds10/3", normalized_input_sha256=normalized,
            pattern_sha256=identity["pattern_sha256"], native_observation_sha256=trace,
            conductance_memory_implementation_sha256=comparison.MEMORY_IMPLEMENTATION,
            retrieval_input_contract=dict(source_carlsim_seed=10, source_host_srand_seed=3,
                source_estdp_enabled=True, effective_input_binding=identity["input_network"]))
    checked, groups, cursor = [], [], 0
    # Source IDs are retained independently within each size, never paired.
    population_order = ["CA3_Pyramidal"] + sorted(set(identity["populations"]) - {"CA3_Pyramidal"})
    for index, name in enumerate(population_order):
        n = identity["populations"][name]
        records = (index + 1) * 100 * scale
        if coupled and name == "CA3_Pyramidal":
            records += 20
        elif coupled and name == "CA3_Axo_Axonic":
            records -= 10
        checked.append(dict(group=name, units=n, records=records, full_record_counts_exact=True,
            relative_path="native_run/results/spk_" + name + ".dat", sha256=digest(f"{scale}-{name}")))
        groups.append(dict(group_id=index, name=name, start=cursor, stop_exclusive=cursor + n))
        cursor += n
    counts = {row["group"]: row for row in checked}
    annotation = dict(api_documented_unit="mA", model_specific_physical_unit="unresolved", numeric_conversion_applied=False)
    readouts = dict(status="DESCRIBED_COUPLED_EXTENSION_CA3_RETRIEVAL_EVENTS" if coupled else "DESCRIBED_SOURCE_SCHEDULED_CA3_RETRIEVAL_EVENTS",
        duration_ms=1001, time_resolution_ms=1, cues_retained=3, full_record_recounted_units=units,
        window_counted_pyramidal_units=75377 * scale, full_record_unit_counts_match=True,
        full_record_population_counts_per_ms_match=True, target_never_boosted_units=137 * scale,
        other_never_boosted_units=274 * scale, eligible_for_boost_units=138 * scale,
        source_scheduled_window_start_ms=[[w[0], w[2]] for w in comparison.WINDOWS],
        source_scheduled_window_stop_ms=[[w[1], w[3]] for w in comparison.WINDOWS],
        normalized_input_sha256=normalized, pattern_sha256=identity["pattern_sha256"],
        overlay_source_sha256=identity["overlay_sha256"], source_config_sha256=identity["config_sha256"],
        original_author_main_sha256=AUTHOR_MAIN_SHA256, source_setup_sha256=SETUP_SHA256,
        checked_event_files=checked, raw_records=sum(row["records"] for row in checked),
        pyramidal_records=counts["CA3_Pyramidal"]["records"], implementation_sha256=producer)
    proof = dict(status="PASS_" + family + "_CA3_OBSERVATION_REPLAY", units=units, duration_ms=1001,
        event_multisets_equal=True, native_intervals_compared=1001, unit_millisecond_counts_compared=units * 1001,
        normalized_input_sha256=normalized, native_observation_sha256=trace, current_units=annotation,
        events_compared=readouts["raw_records"],
        helper_implementation_sha256=dict(ca3_conductance_memory=comparison.MEMORY_IMPLEMENTATION,
                                          ca3_retrieval_readouts=producer),
        source_overlay_verification={role: dict(conductance_memory_enabled=True,
            memory_receipt_and_implementation_verified=True, implementation_sha256=comparison.MEMORY_IMPLEMENTATION)
                                     for role in ("original", "observed")}, **dict.fromkeys(comparison.FLAGS, True))
    replay = dict(status="PASS_COUPLED_EXTENSION_RAW_EVENT_READOUT_REPLAY" if coupled else "PASS_INDEPENDENT_RAW_EVENT_READOUT_REPLAY",
        implementation_sha256=comparison.READOUT_REPLAY_IMPLEMENTATION, all_three_cues_retained=True,
        scalar_values_compared=141, window_count_values_compared=6 * 75377 * scale,
        pyramidal_events=counts["CA3_Pyramidal"]["records"], raw_events_sha256=counts["CA3_Pyramidal"]["sha256"])
    control = dict(status="VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY" if coupled else "VERIFIED_SOURCE_CA3_CONTROL_DELIVERY",
        units=units, intervals_verified=1001, actual_current_values_checked=units * 1001, source_executions=1, source_cues=3,
        current_units=dict(annotation, model_specific_physical_unit="pA"), numeric_conversion_applied=False,
        observer_current_units=annotation.copy(), observation_map=dict(time_unit="ms", interval_ms=1, groups=groups))
    analysis = dict(status=status, exit_code=0, units=units, duration_ms=1001,
                    descriptive_readouts_independently_replayed=True, **dict.fromkeys(comparison.FLAGS, True))
    state = dict(rows=cue_rows(scale), native=native, readouts=readouts, proof=proof, replay=replay,
                 control=control, analysis=analysis)
    if mutate:
        mutate(state)
    analysis["input_archives"] = {}
    native_hashes = {}
    for role in ("primary", "observed"):
        location = root / ("synthetic_" + role)
        write_json(location / "native_report.json", native[role])
        native_hashes[role] = sha256_file(location / "native_report.json")
        declared = [file_row(location, "native_report.json")]
        declared.append(dict(path="observations/recording.json", bytes=0, sha256=digest(f"{scale}-{role}-recording")))
        declared += [dict(path=row["relative_path"], bytes=0, sha256=row["sha256"]) for row in checked]
        write_json(location / "delivery.json", dict(status=("COUPLED_" if coupled else "NATIVE_REFERENCE_")
            + "RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED", exit_code=0, copied_files_verified=True,
            synthetic_fixture=True, files=declared))
        analysis["input_archives"][role] = dict(path=str(location), native_report_sha256=native_hashes[role],
            delivery_sha256=sha256_file(location / "delivery.json"))
    proof.update(original_report_sha256=native_hashes["primary"], observed_report_sha256=native_hashes["observed"],
                 original_recording_sha256=digest(f"{scale}-primary-recording"))
    readouts.update(native_report_sha256=native_hashes["primary"], recording_sha256=proof["original_recording_sha256"],
        input_sha256={"native_report.json": native_hashes["primary"], "observations/recording.json": proof["original_recording_sha256"]})
    csv_path = root / comparison.CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison.COLUMNS)
        writer.writeheader(); writer.writerows(state["rows"])
    readouts["artifacts"] = {"descriptive_readouts.csv": sha256_file(csv_path)}
    write_json(root / comparison.READOUT, readouts)
    replay["source_report_sha256"] = sha256_file(root / comparison.READOUT)
    write_json(root / comparison.REPLAY, replay)
    write_json(root / comparison.RECONCILIATION, proof)
    control["input_sha256"] = dict(reconciliation=sha256_file(root / comparison.RECONCILIATION),
        **{"native_report.json": native_hashes["observed"], "native_observation.bin": trace,
           "trainPat.csv": identity["pattern_sha256"], "main_source": identity["overlay_sha256"],
           "config_source": identity["config_sha256"]})
    write_json(root / comparison.CONTROL, control)
    analysis["readout_reuse"] = dict(primary_report_sha256=native_hashes["primary"], observed_report_sha256=native_hashes["observed"],
        reconciliation_sha256=sha256_file(root / comparison.RECONCILIATION), original_reports_preserved=True,
        artifacts={name.removeprefix("primary_readouts/"): sha256_file(root / name)
                   for name in (comparison.READOUT, comparison.CSV, comparison.REPLAY)})
    write_json(root / comparison.ANALYSIS, analysis)
    write_json(root / "delivery.json", dict(status=status, exit_code=0, copied_files_verified=True, synthetic_fixture=True,
        files=[file_row(root, name) for name in comparison.CONSUMED]))
    return root


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_complete_descriptive_comparison_preserves_all_signed_values_and_denominators(tmp_path):
    reference, coupled = archive(tmp_path / "reference", 1), archive(tmp_path / "coupled", 2)
    output = tmp_path / "comparison"
    result = comparison.compare(reference, coupled, output)
    assert result["status"] == "DESCRIPTIVE_REPLAYED_CA3_SCALE_COMPARISON"
    assert result["synthetic_inputs"] and not result["paired_across_sizes"] and not result["source_scaling_qualified"]
    assert result["inputs"][0]["producer_implementation_sha256"] != result["inputs"][1]["producer_implementation_sha256"]
    combined = read_csv(output / "cue_readouts.csv")
    assert len(combined) == 6
    assert [int(row["following_source_background_ms"]) for row in combined] == [210, 210, 1] * 2
    assert [int(row["target_units"]) for row in combined] == [137] * 3 + [274] * 3
    assert float(combined[3]["target_delta_rate_hz"]) < 0
    differences = read_csv(output / "cue_differences.csv")
    assert len(differences) == 72
    indexed = {(int(row["cue_id"]), row["metric"]): row for row in differences}
    assert float(indexed[1, "target_cue_rate_hz"]["coupled_minus_native"]) == pytest.approx(1 / (137 * .020) - 4 / (137 * .020))
    assert float(indexed[1, "other_cue_rate_hz"]["coupled_minus_native"]) == 0
    populations = read_csv(output / "population_readouts.csv")
    assert len(populations) == 16 and {int(row["network_units"]) for row in populations} == {84053, 168106}
    population_deltas = {row["population"]: float(row["coupled_minus_native"])
                         for row in read_csv(output / "population_differences.csv")}
    assert len(population_deltas) == 8 and population_deltas["CA3_Axo_Axonic"] < 0
    assert population_deltas["CA3_Pyramidal"] > 0 and population_deltas["CA3_Ivy"] == 0
    assert result["inputs"][0]["current_units"]["model_specific_physical_unit"] == "pA"
    assert result["inputs"][0]["observer_current_units"]["model_specific_physical_unit"] == "unresolved"
    with pytest.raises(FileExistsError):
        comparison.compare(reference, coupled, output)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["rows"].pop(), "three cue"),
    (lambda s: s["rows"][1].update(cue_id=1), "duplicate cue"),
    (lambda s: s["rows"][0].update(target_units=136), "Subset denominator"),
    (lambda s: s["proof"].update(terminal_native_clock_independently_verified=False), "Clock/current"),
    (lambda s: s["control"].update(intervals_verified=1000), "Complete source control"),
    (lambda s: s["proof"]["source_overlay_verification"]["original"].update(implementation_sha256="0" * 64), "conductance correction"),
    (lambda s: s["replay"].update(scalar_values_compared=140), "141-scalar"),
    (lambda s: s["rows"][0].update(target_cue_rate_hz=float("inf")), "Nonfinite cue"),
    (lambda s: s["native"]["primary"].update(retrieval_condition="cue_omission"), "Source condition"),
    (lambda s: s["native"]["primary"].update(integration="RUNGE_KUTTA4_10"), "RK4/5"),
])
def test_rejects_changed_scientific_or_measurement_contract(tmp_path, mutate, match):
    reference = archive(tmp_path / "reference", 1, mutate)
    coupled = archive(tmp_path / "coupled", 2)
    with pytest.raises(ValueError, match=match):
        comparison.compare(reference, coupled, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_rejects_consumed_receipt_or_native_delivery_tampering(tmp_path):
    reference, coupled = archive(tmp_path / "reference", 1), archive(tmp_path / "coupled", 2)
    (reference / comparison.RECONCILIATION).write_text("{}")
    with pytest.raises(ValueError, match="Consumed file changed"):
        comparison.compare(reference, coupled, tmp_path / "comparison")
    reference = archive(tmp_path / "other_reference", 1)
    (reference / "synthetic_primary/delivery.json").write_text("{}")
    with pytest.raises(ValueError, match="Native delivery binding changed"):
        comparison.compare(reference, coupled, tmp_path / "comparison")

"""Descriptive comparison of complete, replayed PFC source archives."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .provenance import sha256_file

SOURCE_REVISION = "d7c4f07fa83a052829e40790fd1594955fe1c067"
SOURCE_ENTRY_SHA256 = "416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720"
CONSUMED = ("analysis/analysis.json", "analysis/readout_replay.json", "analysis/responses.csv")
PHASES = {"baseline": [0, 100], "cue": [100, 300], "early_delay": [300, 500],
          "middle_delay": [1000, 2000], "late_delay": [4000, 5000]}
PHYSICAL = {
    "dt_ms": 0.05, "duration_ms": 5000,
    "stimulus_center_deg": 120, "stimulus_width_deg": 30,
    "stimulus_start_ms": 100, "stimulus_duration_ms": 200,
    "distractor_center_deg": 90, "distractor_width_deg": 40,
    "distractor_start_ms": 400, "distractor_duration_ms": 50,
}
CONDITIONS = {0.06: "cue", 0.0: "omission", 0.5: "strong_cue"}
METRICS = ("excitatory_rate_hz", "inhibitory_rate_hz", "cue_selectivity_hz", "resultant_length")
COLUMNS = (
    "case_id,source_rng_seed,condition,phase,start_ms,stop_ms,units,"
    "excitatory_neurons,inhibitory_neurons,stimulus_strength_nA,"
    "cue_fraction_e,effective_cue_center_deg,stimulated_e_units,"
    "stimulated_e_fraction,excitatory_spikes,inhibitory_spikes,"
    "excitatory_rate_hz,inhibitory_rate_hz,cue_units,cue_region_rate_hz,"
    "noncue_region_rate_hz,cue_selectivity_hz,vector_x_spikes,"
    "vector_y_spikes,resultant_length,decoded_angle_deg,"
    "circular_cue_error_deg,direction_defined,all_unit_rate_hz"
).split(",")
NUMERIC = [key for key in COLUMNS if key not in (
    "case_id", "condition", "phase", "direction_defined",
    "decoded_angle_deg", "circular_cue_error_deg")]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, label: str) -> float:
    value = float(value)
    _require(math.isfinite(value), f"Nonfinite value: {label}")
    return value


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load(root: Path, expanded: bool) -> tuple[dict, list[dict]]:
    root = Path(root)
    cohort, backend = ("expanded", "source_equation_cuda") if expanded else ("reference", "native_brian2")
    units, ne, ni, factor = (100000, 80000, 20000, 0.0256) if expanded else (2560, 2048, 512, 1.0)
    seeds = [94001, 94002, 94003] if expanded else [93001, 93002, 93003]
    status = ("DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC" if expanded
              else "DESCRIBED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC")
    delivery = json.loads((root / "delivery.json").read_text(encoding="utf-8"))
    _require(isinstance(delivery.get("files"), list), f"{cohort}: missing delivery files")
    digests = {}
    for name in CONSUMED:
        matches = [entry for entry in delivery["files"] if entry["path"] == name]
        _require(len(matches) == 1, f"{cohort}: delivery must bind {name} exactly once")
        path, record = root / name, matches[0]
        digest = sha256_file(path)
        _require(path.stat().st_size == record["bytes"] and digest == record["sha256"],
                 f"{cohort}: delivered file changed: {name}")
        digests[name] = digest
    analysis = json.loads((root / CONSUMED[0]).read_text(encoding="utf-8"))
    replay = json.loads((root / CONSUMED[1]).read_text(encoding="utf-8"))
    _require(analysis["status"] == status, f"{cohort}: unexpected analysis status")
    _require(analysis["units"] == units, f"{cohort}: unexpected unit count")
    _require(analysis["complete_source_episodes"] == 9, f"{cohort}: incomplete source episodes")
    _require(analysis["source_revision"] == SOURCE_REVISION, f"{cohort}: source revision mismatch")
    _require(analysis["source_entry_sha256"] == SOURCE_ENTRY_SHA256, f"{cohort}: source entry mismatch")
    _require(analysis["source_rng_seeds"] == seeds, f"{cohort}: source cohort mismatch")
    _require(analysis["phases_ms"] == PHASES, f"{cohort}: phase schedule mismatch")
    artifacts = analysis["artifacts"]
    _require(isinstance(artifacts, dict), f"{cohort}: invalid artifact mapping")
    _require(artifacts["responses.csv"] == digests[CONSUMED[2]], f"{cohort}: response digest mismatch")
    _require(analysis["readout_replay_sha256"] == digests[CONSUMED[1]], f"{cohort}: replay digest mismatch")
    _require(replay["status"] == "PASS" and replay["complete_episodes"] == 9
             and replay["units"] == units, f"{cohort}: complete matching readout replay required")
    _require(replay["artifacts"] == artifacts, f"{cohort}: replay artifact binding mismatch")
    if expanded:
        _require(analysis["common_execution_binding"]["backend"] == "cuda", "Expanded backend mismatch")
    config = analysis["configuration"]
    for key, expected in PHYSICAL.items():
        _require(config[key] == expected, f"{cohort}: physical schedule mismatch: {key}")
    for key, expected in (("excitatory_neurons", ne), ("inhibitory_neurons", ni), ("weight_scaling_factor", factor)):
        _require(config[key] == expected, f"{cohort}: size/normalization mismatch: {key}")
    _require(math.isclose(ne * factor, 2048, rel_tol=0, abs_tol=1e-12)
             and math.isclose(ni * factor, 512, rel_tol=0, abs_tol=1e-12),
             f"{cohort}: normalized population weights differ")
    cases = config["cases"]
    _require(len(cases) == 9, f"{cohort}: expected nine configured cases")
    by_id, condition_seeds = {}, set()
    for case in cases:
        case_id = case["case_id"]
        _require(isinstance(case_id, str) and case_id and case_id not in by_id, f"{cohort}: invalid case identity")
        _require(case["source_rng_seed"] in seeds and case["stimulus_strength_nA"] in CONDITIONS,
                 f"{cohort}: invalid case seed/strength")
        _require(case["distractor_strength_nA"] == 0, f"{cohort}: unexpected distractor strength")
        by_id[case_id] = case
        condition_seeds.add((case["source_rng_seed"], case["stimulus_strength_nA"]))
    _require(len(condition_seeds) == 9, f"{cohort}: missing or duplicated condition/seed")
    with (root / CONSUMED[2]).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(COLUMNS)
                 and set(reader.fieldnames) == set(COLUMNS), f"{cohort}: response schema mismatch")
        rows = list(reader)
    _require(len(rows) == 45, f"{cohort}: expected all 45 case/phase rows")
    seen = set()
    for row in rows:
        _require(set(row) == set(COLUMNS) and None not in row.values(), f"{cohort}: malformed response row")
        case_id, phase = row["case_id"], row["phase"]
        _require(case_id in by_id and phase in PHASES, f"{cohort}: unknown case/phase")
        identity = (case_id, phase)
        _require(identity not in seen, f"{cohort}: duplicate case/phase: {identity}")
        seen.add(identity)
        case = by_id[case_id]
        values = {key: _number(row[key], f"{cohort}:{case_id}:{phase}:{key}") for key in NUMERIC}
        expected = {"source_rng_seed": case["source_rng_seed"], "units": units,
                    "excitatory_neurons": ne, "inhibitory_neurons": ni,
                    "stimulus_strength_nA": case["stimulus_strength_nA"],
                    "start_ms": PHASES[phase][0], "stop_ms": PHASES[phase][1]}
        for key, value in expected.items():
            _require(values[key] == value, f"{cohort}: row identity/configuration mismatch: {identity}:{key}")
        _require(row["condition"] == CONDITIONS[case["stimulus_strength_nA"]], f"{cohort}: condition mismatch")
        row.update(cohort=cohort, backend=backend)
    _require(seen == {(case_id, phase) for case_id in by_id for phase in PHASES}, f"{cohort}: incomplete coverage")
    binding = {"cohort": cohort, "backend": backend, "units": units, "archive": str(root.resolve()),
               "delivery_sha256": sha256_file(root / "delivery.json"), "consumed_sha256": digests,
               "configuration": config, "source_rng_seeds": seeds}
    return binding, rows


def compare(reference: Path, expanded: Path, output: Path) -> dict:
    archives = [_load(reference, False), _load(expanded, True)]
    descriptive = []
    for binding, rows in archives:
        for condition in CONDITIONS.values():
            for phase in PHASES:
                for metric in METRICS:
                    values = [float(row[metric]) for row in rows
                              if row["condition"] == condition and row["phase"] == phase]
                    _require(len(values) == 3, "A descriptive cell must retain all three source cases")
                    stats = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
                    _require(all(math.isfinite(value) for value in stats.values()), "Nonfinite descriptive summary")
                    descriptive.append({key: binding[key] for key in ("cohort", "backend", "units")}
                                       | {"condition": condition, "phase": phase, "metric": metric, "n": 3} | stats)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "combined.csv", COLUMNS + ["cohort", "backend"],
               [row for _, rows in archives for row in rows])
    _write_csv(output / "descriptive.csv",
               ["cohort", "backend", "units", "condition", "phase", "metric", "n", "mean", "min", "max"],
               descriptive)
    report = {
        "status": "DESCRIPTIVE_REPLAYED_SIZE_COMPARISON",
        "inputs": [binding for binding, _ in archives],
        "source_episodes_reused": 18, "new_source_episodes": 0, "production_episodes": 0,
        "paired_across_sizes": False, "source_scaling_qualified": False,
        "hypothesis": ("Distinguish whether absence of delay response is specific to weak cues at both sizes "
                       "or appears only after scaling. These summaries inform subsequent source qualification."),
        "limitations": [
            "Independent size cohorts are not paired; all conditions and source cases are retained.",
            "Size, normalized weights, cue discretization and backend differ together; this is not a pure N causal effect.",
            "N_E times scaling factor is 2048 and N_I times scaling factor is 512; self-exclusion retains a finite-size correction.",
            "Phase gaps do not establish continuous memory. No angular averages, inferential statistics or formal claims are produced.",
            "Only consumed readout files are checked here; full arrays are bound through the previously passing replay.",
        ],
        "outputs": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                    for name in ("combined.csv", "descriptive.csv")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report



CLOCKS_MS = (0.05, 0.025, 0.0125)
CLOCK_METRICS = METRICS + ("all_unit_rate_hz", "circular_cue_error_deg")
DRIVER_INVARIANTS = ("binomial_n", "external_rate_Hz", "initial_draw_order",
    "initial_voltage_archive_order", "packed_draw_order", "random_stream_policy",
    "returned_dtype", "rng", "numpy_version", "source_entry_sha256", "n_e", "n_i")


def _clock_archive(root: Path) -> tuple[dict, list[dict]]:
    """Read the existing replay receipt; never infer new samples from unit counts."""
    root = Path(root)
    delivery = json.loads((root / "delivery.json").read_text(encoding="utf-8"))
    digests = {}
    for name in CONSUMED:
        matches = [entry for entry in delivery["files"] if entry["path"] == name]
        _require(len(matches) == 1, f"Clock archive must bind {name} once")
        path, entry = root / name, matches[0]
        digest = sha256_file(path)
        _require(path.stat().st_size == entry["bytes"] and digest == entry["sha256"],
                 f"Clock readout changed: {name}")
        digests[name] = digest
    analysis = json.loads((root / CONSUMED[0]).read_text(encoding="utf-8"))
    replay = json.loads((root / CONSUMED[1]).read_text(encoding="utf-8"))
    config, common = analysis["configuration"], analysis["common_execution_binding"]
    dt, units = config["dt_ms"], analysis["units"]
    _require(dt in CLOCKS_MS and units >= 100000, "Registered full-scale clocks required")
    _require(analysis["status"] in ("PARTIAL_SOURCE_EXECUTION_DIAGNOSTIC",
                                  "DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC"), "Replayed source readouts required")
    _require(analysis["source_revision"] == SOURCE_REVISION
             and analysis["source_entry_sha256"] == SOURCE_ENTRY_SHA256
             and analysis["phases_ms"] == PHASES, "Clock source or phase identity changed")
    _require(not analysis["failed_source_attempts"] and not analysis["unresolved_input_paths"],
             "Unresolved or failed supplied sources prevent a complete clock comparison")
    for key, value in PHYSICAL.items():
        if key != "dt_ms":
            _require(config[key] == value, f"Clock physical protocol changed: {key}")
    ne, ni, factor = (config[key] for key in ("excitatory_neurons", "inhibitory_neurons", "weight_scaling_factor"))
    _require(ne + ni == units and math.isclose(ne * factor, 2048, rel_tol=0, abs_tol=1e-10)
             and math.isclose(ni * factor, 512, rel_tol=0, abs_tol=1e-10), "Clock size or normalization changed")
    driver = common["driver"]
    _require(common["backend"] == "numpy" and driver["random_stream_policy"] == "source_global"
             and driver["binomial_n"] == 1000 and driver["external_rate_Hz"] == 1.4
             and driver["dt_seconds"] == dt / 1000
             and driver["binomial_probability"] == 1.4 * (dt / 1000)
             and driver["n_e"] == ne and driver["n_i"] == ni
             and driver["source_entry_sha256"] == SOURCE_ENTRY_SHA256
             and driver["total_ticks"] == driver["ticks_generated"] == round(config["duration_ms"] / dt),
             "Native clock-dependent input law or execution length changed")
    complete = analysis["complete_source_episodes"]
    _require(type(complete) is int and 1 <= complete <= 9
             and analysis["registered_source_cases"] == 9, "Clock case denominator changed")
    _require(replay["status"] == "PASS" and replay["complete_episodes"] == complete
             and replay["registered_source_cases"] == 9 and replay["units"] == units
             and replay["source_bindings"] == analysis["source_bindings"]
             and replay["config_sha256"] == analysis["config_sha256"]
             and replay["phase_unit_count_values_checked"] == analysis["phase_unit_count_values_verified"]
             == complete * units * len(PHASES), "Complete matching full-unit readout replay required")
    _require(replay["artifacts"] == analysis["artifacts"]
             and analysis["artifacts"]["responses.csv"] == digests[CONSUMED[2]]
             and analysis["readout_replay_sha256"] == digests[CONSUMED[1]], "Clock replay artifact binding changed")
    cases = config["cases"]
    by_id = {case["case_id"]: case for case in cases}
    _require(len(cases) == len(by_id) == 9, "All nine source registrations must remain")
    with (root / CONSUMED[2]).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(COLUMNS)
                 and set(reader.fieldnames) == set(COLUMNS), "Clock readout columns changed")
        rows = list(reader)
    seen = set()
    for row in rows:
        _require(set(row) == set(COLUMNS) and None not in row.values(), "Malformed clock row")
        case_id, phase = row["case_id"], row["phase"]
        _require(case_id in by_id and phase in PHASES and (case_id, phase) not in seen,
                 "Unknown or duplicate clock case/phase")
        seen.add((case_id, phase)); case = by_id[case_id]
        values = {key: _number(row[key], key) for key in NUMERIC}
        expected = dict(source_rng_seed=case["source_rng_seed"], stimulus_strength_nA=case["stimulus_strength_nA"],
                        units=units, excitatory_neurons=ne, inhibitory_neurons=ni,
                        start_ms=PHASES[phase][0], stop_ms=PHASES[phase][1])
        _require(all(values[key] == value for key, value in expected.items())
                 and row["condition"] == CONDITIONS[case["stimulus_strength_nA"]]
                 and case["distractor_strength_nA"] == 0, "Clock row differs from registered case")
        _require(row["direction_defined"] in ("True", "False"), "Direction validity missing")
        if row["direction_defined"] == "True":
            for key in ("decoded_angle_deg", "circular_cue_error_deg"):
                _number(row[key], key)
        else:
            _require(row["decoded_angle_deg"] == row["circular_cue_error_deg"] == "",
                     "Undefined direction cannot carry a fabricated angle")
        row["dt_ms"] = dt
    present = {case_id for case_id, _ in seen}
    _require(len(present) == complete and len(rows) == complete * len(PHASES)
             and seen == {(case_id, phase) for case_id in present for phase in PHASES},
             "Every completed clock case must retain all phases")
    _require(set(analysis["missing_source_cases"]) == set(by_id) - present,
             "Clock missing-case denominator changed")
    binding = dict(archive=str(root.resolve()), delivery_sha256=sha256_file(root / "delivery.json"),
        consumed_sha256=digests, dt_ms=dt, configuration=config, common_execution_binding=common,
        complete_case_ids=sorted(present), missing_case_ids=analysis["missing_source_cases"],
        phase_unit_count_values_verified=analysis["phase_unit_count_values_verified"])
    return binding, rows


def compare_clocks(archives: list[Path], output: Path) -> dict:
    _require(len(archives) == len(CLOCKS_MS), "Exactly three registered clock archives required")
    loaded = [_clock_archive(path) for path in archives]
    by_clock = {binding["dt_ms"]: (binding, rows) for binding, rows in loaded}
    _require(len(by_clock) == len(CLOCKS_MS) and set(by_clock) == set(CLOCKS_MS),
             "Missing or duplicate registered clock")
    loaded = [by_clock[clock] for clock in CLOCKS_MS]
    first = loaded[0][0]
    configuration = {key: value for key, value in first["configuration"].items() if key != "dt_ms"}
    common = first["common_execution_binding"]
    for binding, _ in loaded:
        _require({key: value for key, value in binding["configuration"].items() if key != "dt_ms"}
                 == configuration, "Only dt may change in the source configuration")
        _require(binding["complete_case_ids"] == first["complete_case_ids"],
                 "Clock case coverage differs; do not select a supported intersection")
        actual = binding["common_execution_binding"]
        _require(all(actual["code"][key] == common["code"][key]
                     for key in ("executor_sha256", "drive_sha256"))
                 and all(actual["driver"][key] == common["driver"][key] for key in DRIVER_INVARIANTS),
                 "Clock dynamics implementation or native driver semantics changed")
    differences = []
    for (_, coarse), (_, fine) in zip(loaded, loaded[1:]):
        fine_rows = {(row["case_id"], row["phase"]): row for row in fine}
        for left in coarse:
            right = fine_rows[left["case_id"], left["phase"]]
            for metric in CLOCK_METRICS:
                defined = metric != "circular_cue_error_deg" or (
                    left["direction_defined"] == right["direction_defined"] == "True")
                a, b = (float(left[metric]), float(right[metric])) if defined else (None, None)
                difference = a - b if defined else None
                if defined and metric == "circular_cue_error_deg":
                    difference = (difference + 180) % 360 - 180
                differences.append(dict(case_id=left["case_id"], source_rng_seed=left["source_rng_seed"],
                    condition=left["condition"], phase=left["phase"], metric=metric,
                    coarse_dt_ms=left["dt_ms"], fine_dt_ms=right["dt_ms"], coarse_value=a, fine_value=b,
                    coarse_minus_fine=difference, comparison_status="DEFINED" if defined else "UNDEFINED_DIRECTION"))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "by_clock.csv", COLUMNS + ["dt_ms"], [row for _, rows in loaded for row in rows])
    _write_csv(output / "adjacent_clock_differences.csv", list(differences[0]), differences)
    cases = first["complete_case_ids"]
    report = dict(status="DESCRIPTIVE_REPLAYED_CLOCK_COMPARISON", inputs=[binding for binding, _ in loaded],
        dt_ms=list(CLOCKS_MS), registered_cases_per_clock=9, completed_cases_per_clock=len(cases),
        full_registered_case_coverage=len(cases) == 9, source_episodes_reused=len(cases) * len(CLOCKS_MS),
        source_rng_blocks_compared=sorted({int(row["source_rng_seed"]) for row in loaded[0][1]}),
        matched_physical_input_paths=False, distributional_convergence_qualified=False,
        full_scale_qualified=False, scientific_fidelity_qualified=False, production_episodes=0,
        new_source_episodes=0, confidence_intervals=None, p_values=None, q_values=None,
        scope="Realized phase readouts at each native clock, retaining the complete registered case denominator.",
        interpretation="Differences combine clock discretization and changed native stochastic realizations; no equivalence decision.",
        observation_scope="Existing replay verified all-unit phase counts. Phase gaps, joint temporal laws and full state trajectories are not inferred.",
        outputs={name: dict(sha256=sha256_file(output / name), bytes=(output / name).stat().st_size)
                 for name in ("by_clock.csv", "adjacent_clock_differences.csv")})
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--expanded", type=Path)
    parser.add_argument("--clock-readout", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.clock_readout:
        if args.reference is not None or args.expanded is not None:
            parser.error("Choose size comparison or clock comparison")
    elif args.reference is None or args.expanded is None:
        parser.error("Size comparison requires --reference and --expanded")
    try:
        if args.clock_readout:
            compare_clocks(args.clock_readout, args.output)
        else:
            compare(args.reference, args.expanded, args.output)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        parser.exit(2, f"PFC readout comparison failed: {error}\n")


if __name__ == "__main__":
    main()

"""Count-based source-history precision planning; no uncertainty qualification."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import re

SCHEMA = "pfc_source_history_precision_v1"
CIRCUITS = (23, 63, 52, 86)
BLOCKS = (3, 4, 5, 6, 7, 8)
CONDITIONS = tuple(range(10))
COHORTS = {"pilot_fine": (3, 4, 5), "domain_fine": (6, 7, 8), "pooled_fine": BLOCKS}
POPULATIONS = {"E": 80000, "I": 20000, "all": 100000}
DURATION_TICKS = 400000
TICKS_PER_SECOND = 80000
REQUIRED_EPISODES = len(CIRCUITS) * len(BLOCKS) * len(CONDITIONS)
PAIRED_FIELDS = (
    "circuit_index", "block_index", "cue_condition_index", "omission_condition_index",
    "cue_condition_id", "omission_condition_id", "cue_episode_id", "omission_episode_id",
    "initial_state_seed", "background_seed", "phase_index", "start_tick", "stop_tick",
    "population", "population_units", "cue_count", "omission_count", "count_difference",
    "rate_difference_hz",
)
SUMMARY_FIELDS = (
    "circuit_index", "cohort", "block_indices", "n_histories", "cue_condition_index",
    "omission_condition_index", "cue_condition_id", "omission_condition_id", "phase_index",
    "start_tick", "stop_tick", "population", "population_units", "sum_count_differences",
    "count_variance_numerator", "count_variance_denominator", "mean_rate_difference_hz",
    "sample_variance_hz2", "estimated_sem_hz",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def episode_id(circuit, block, condition):
    return f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{condition:02d}"


def integer(value, field):
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None,
            "PHASE_NONNEGATIVE_INTEGER_REQUIRED:" + field)
    return int(value)


def milliseconds(value, tick, field):
    try:
        measured = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError("PHASE_TIME_INVALID:" + field) from None
    require(measured.is_finite() and measured == Decimal(tick) / 80,
            "PHASE_NATIVE_TIME_MISMATCH:" + field)


def load_phases(binding, identity, condition_id, base):
    require(isinstance(binding, dict) and isinstance(binding.get("path"), str)
            and type(binding.get("bytes")) is int and binding["bytes"] >= 0
            and valid_hash(binding.get("sha256")), "PHASE_FILE_BINDING_REQUIRED")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    data = path.read_bytes()
    require(len(data) == binding["bytes"] and sha256(data) == binding["sha256"],
            "PHASE_FILE_BINDING_MISMATCH:" + str(path))
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    required = {"circuit_index", "block_index", "episode_id", "condition_id", "phase_index",
                "start_tick", "stop_tick", "start_ms", "stop_ms",
                "excitatory_spikes", "inhibitory_spikes"}
    require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
            and required <= set(reader.fieldnames), "PHASE_HEADER_INVALID")
    phases = {}
    for row in reader:
        require(None not in row and all(row.get(field) is not None for field in required),
                "PHASE_ROW_INVALID")
        require(integer(row["circuit_index"], "circuit_index") == identity["circuit_index"]
                and integer(row["block_index"], "block_index") == identity["block_index"]
                and row["episode_id"] == identity["episode_id"]
                and row["condition_id"] == condition_id, "PHASE_EPISODE_IDENTITY_MISMATCH")
        index = integer(row["phase_index"], "phase_index")
        require(index in range(8) and index not in phases, "PHASE_INDEX_DUPLICATE_OR_INVALID")
        start = integer(row["start_tick"], "start_tick")
        stop = integer(row["stop_tick"], "stop_tick")
        require(start < stop <= DURATION_TICKS, "PHASE_NATIVE_BOUNDARY_INVALID")
        milliseconds(row["start_ms"], start, "start_ms")
        milliseconds(row["stop_ms"], stop, "stop_ms")
        excitatory = integer(row["excitatory_spikes"], "excitatory_spikes")
        inhibitory = integer(row["inhibitory_spikes"], "inhibitory_spikes")
        phases[index] = dict(start_tick=start, stop_tick=stop,
                             counts={"E": excitatory, "I": inhibitory, "all": excitatory + inhibitory})
    require(set(phases) == set(range(8)), "COMPLETE_EIGHT_PHASES_REQUIRED")
    ordered = [phases[index] for index in range(8)]
    require(ordered[0]["start_tick"] == 0 and ordered[-1]["stop_tick"] == DURATION_TICKS
            and all(left["stop_tick"] == right["start_tick"]
                    for left, right in zip(ordered, ordered[1:])), "PHASE_COVERAGE_NOT_CONTIGUOUS")
    return phases, dict(path=str(path), bytes=len(data), sha256=binding["sha256"])


def load_manifest(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    manifest = json.loads(data)
    require(manifest.get("schema") == SCHEMA, "MANIFEST_SCHEMA_INVALID")
    for field, expected in (("design_circuits", list(CIRCUITS)), ("block_indices", list(BLOCKS)),
                            ("condition_indices", list(CONDITIONS))):
        value = manifest.get(field)
        require(isinstance(value, list) and all(type(item) is int for item in value)
                and value == expected, "MANIFEST_SCOPE_CHANGED:" + field)
    require(type(manifest.get("dt_ms")) in (int, float) and manifest["dt_ms"] == .0125
            and type(manifest.get("duration_ticks")) is int and manifest["duration_ticks"] == DURATION_TICKS
            and type(manifest.get("core_units")) is int and manifest["core_units"] == 100000
            and manifest.get("population_sizes") == {"E": 80000, "I": 20000}
            and all(type(value) is int for value in manifest["population_sizes"].values()),
            "MANIFEST_FINE_SOURCE_SCALE_CHANGED")
    rows = manifest.get("episodes")
    require(isinstance(rows, list), "MANIFEST_EPISODE_LIST_REQUIRED")
    episodes, histories, streams, signatures, names = {}, {}, {}, {}, {}
    boundaries = None
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("identity"), dict), "EPISODE_IDENTITY_REQUIRED")
        identity = row["identity"]
        for field, allowed in (("circuit_index", CIRCUITS), ("block_index", BLOCKS),
                               ("condition_index", CONDITIONS)):
            require(type(identity.get(field)) is int and identity[field] in allowed,
                    "EPISODE_SCOPE_INVALID:" + field)
        circuit, block, condition = (identity[field] for field in
                                     ("circuit_index", "block_index", "condition_index"))
        key = circuit, block, condition
        expected_id = episode_id(*key)
        circuit_id = f"pfc_working_memory_circuit_{circuit:03d}"
        require(identity.get("episode_id") == expected_id and identity.get("circuit_id") == circuit_id
                and identity.get("block_id") == circuit_id + f"_block_{block:03d}",
                "CANONICAL_EPISODE_IDENTITY_CHANGED")
        require(key not in episodes, "DUPLICATE_EPISODE_IDENTITY:" + expected_id)
        condition_id = row.get("condition_id")
        require(isinstance(condition_id, str) and bool(condition_id)
                and identity.get("condition_id", condition_id) == condition_id,
                "SOURCE_CONDITION_METADATA_INVALID")
        require(condition not in names or names[condition] == condition_id, "SOURCE_CONDITION_ID_CHANGED")
        require(condition_id not in names.values() or names.get(condition) == condition_id,
                "SOURCE_CONDITION_ID_NOT_UNIQUE")
        names[condition] = condition_id
        require(valid_hash(row.get("config_sha256")), "SOURCE_CONFIG_HASH_REQUIRED")
        signature = row.get("source_signature")
        require(isinstance(signature, dict) and bool(signature), "VERIFIED_SOURCE_SIGNATURE_REQUIRED")
        encoded_signature = canonical_json(signature)
        signature_key = circuit, condition
        require(signature_key not in signatures or signatures[signature_key] == encoded_signature,
                "NON_STREAM_SOURCE_SIGNATURE_CHANGED")
        signatures[signature_key] = encoded_signature
        stream_identity = []
        for field in ("initial_state_seed", "background_seed"):
            value = identity.get(field)
            require(type(value) is int and value >= 0, "SOURCE_STREAM_IDENTITY_INVALID:" + field)
            owner = circuit, block
            require((field, value) not in streams or streams[field, value] == owner,
                    "SOURCE_STREAM_REUSED_ACROSS_HISTORIES:" + field)
            streams[field, value] = owner
            stream_identity.append(value)
        history_key = circuit, block
        require(history_key not in histories or histories[history_key] == tuple(stream_identity),
                "PAIRED_CONDITION_STREAM_IDENTITY_CHANGED")
        histories[history_key] = tuple(stream_identity)
        phases, binding = load_phases(row.get("phase_csv"), identity, condition_id, path.parent)
        current = tuple((phases[index]["start_tick"], phases[index]["stop_tick"]) for index in range(8))
        require(boundaries is None or current == boundaries, "SOURCE_PHASE_BOUNDARIES_CHANGED")
        boundaries = current
        episodes[key] = dict(identity=identity, condition_id=condition_id, phases=phases,
                             config_sha256=row["config_sha256"], phase_csv=binding,
                             source_signature_sha256=sha256(encoded_signature))
    return episodes, dict(path=str(path), bytes=len(data), sha256=sha256(data))


def contrast(episodes, circuit, block, condition, phase, population):
    cue = episodes[circuit, block, condition]
    omission = episodes[circuit, block, condition + 5]
    cue_phase, omission_phase = cue["phases"][phase], omission["phases"][phase]
    require(cue_phase["start_tick"] == omission_phase["start_tick"]
            and cue_phase["stop_tick"] == omission_phase["stop_tick"], "PAIRED_PHASE_BOUNDARIES_CHANGED")
    cue_count, omission_count = cue_phase["counts"][population], omission_phase["counts"][population]
    difference = cue_count - omission_count
    ticks = cue_phase["stop_tick"] - cue_phase["start_tick"]
    units = POPULATIONS[population]
    return dict(circuit_index=circuit, block_index=block, cue_condition_index=condition,
                omission_condition_index=condition + 5, cue_condition_id=cue["condition_id"],
                omission_condition_id=omission["condition_id"], cue_episode_id=cue["identity"]["episode_id"],
                omission_episode_id=omission["identity"]["episode_id"],
                initial_state_seed=cue["identity"]["initial_state_seed"],
                background_seed=cue["identity"]["background_seed"], phase_index=phase,
                start_tick=cue_phase["start_tick"], stop_tick=cue_phase["stop_tick"],
                population=population, population_units=units, cue_count=cue_count,
                omission_count=omission_count, count_difference=difference,
                rate_difference_hz=float(Fraction(difference * TICKS_PER_SECOND, units * ticks)))


def summary(rows, circuit, cohort, blocks):
    n = len(rows)
    values = [row["count_difference"] for row in rows]
    total = sum(values)
    numerator = n * sum(value * value for value in values) - total * total
    denominator = n * (n - 1)
    require(n > 1 and numerator >= 0, "EXACT_COUNT_VARIANCE_INVALID")
    first = rows[0]
    scale = Fraction(TICKS_PER_SECOND,
                     first["population_units"] * (first["stop_tick"] - first["start_tick"]))
    variance = Fraction(numerator, denominator) * scale * scale
    result = {field: first[field] for field in
              ("cue_condition_index", "omission_condition_index", "cue_condition_id", "omission_condition_id",
               "phase_index", "start_tick", "stop_tick", "population", "population_units")}
    result.update(circuit_index=circuit, cohort=cohort, block_indices=";".join(map(str, blocks)),
                  n_histories=n, sum_count_differences=total, count_variance_numerator=numerator,
                  count_variance_denominator=denominator,
                  mean_rate_difference_hz=float(Fraction(total, n) * scale),
                  sample_variance_hz2=float(variance), estimated_sem_hz=math.sqrt(float(variance / n)))
    return result


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(manifest_path, output):
    output = Path(output).resolve()
    require(not output.exists(), "IMMUTABLE_OUTPUT_EXISTS")
    episodes, manifest_binding = load_manifest(manifest_path)
    expected = [(circuit, block, condition) for circuit in CIRCUITS for block in BLOCKS for condition in CONDITIONS]
    missing = [episode_id(*key) for key in expected if key not in episodes]
    paired_rows, summaries = [], []
    paired = {}
    complete_cohorts = {cohort: [] for cohort in COHORTS}
    for circuit in CIRCUITS:
        for block in BLOCKS:
            for condition in range(5):
                if (circuit, block, condition) not in episodes or (circuit, block, condition + 5) not in episodes:
                    continue
                for phase in range(8):
                    for population in POPULATIONS:
                        row = contrast(episodes, circuit, block, condition, phase, population)
                        paired_rows.append(row)
                        paired[circuit, block, condition, phase, population] = row
        for cohort, blocks in COHORTS.items():
            if not all((circuit, block, condition) in episodes for block in blocks for condition in CONDITIONS):
                continue
            complete_cohorts[cohort].append(circuit)
            for condition in range(5):
                for phase in range(8):
                    for population in POPULATIONS:
                        rows = [paired[circuit, block, condition, phase, population] for block in blocks]
                        summaries.append(summary(rows, circuit, cohort, blocks))
    report = dict(schema=SCHEMA, status="PARTIAL" if missing else "COMPLETE",
                  actual_episodes=len(episodes), required_episodes=REQUIRED_EPISODES,
                  missing_episode_ids=missing, design_circuits=list(CIRCUITS), block_indices=list(BLOCKS),
                  condition_indices=list(CONDITIONS), paired_history_phase_rate_rows=len(paired_rows),
                  history_precision_summary_rows=len(summaries), complete_cohort_circuits=complete_cohorts,
                  manifest=manifest_binding, input_episode_bindings=[dict(episode_id=episodes[key]["identity"]["episode_id"],
                    config_sha256=episodes[key]["config_sha256"], source_signature_sha256=episodes[key]["source_signature_sha256"],
                    phase_csv=episodes[key]["phase_csv"]) for key in expected if key in episodes],
                  dt_ms=.0125, duration_ticks=DURATION_TICKS, core_units=100000,
                  population_sizes={"E": 80000, "I": 20000},
                  independent_unit="joint initial/background source history nested in circuit",
                  paired_condition_semantics="condition k minus omission k+5 under the same history, k=0..4",
                  measurement="native-phase integrated rates from exact integer counts; not matched-time covariance",
                  estimated_sem_interpretation="estimated sampling scale, not calibrated confidence coverage",
                  source_simulations=0, production_episodes=0, source_qualification_granted=False,
                  scaling_qualified=False, uncertainty_coverage_qualified=False)
    output.mkdir(parents=True)
    write_csv(output / "paired_history_phase_rates.csv", PAIRED_FIELDS, paired_rows)
    write_csv(output / "history_precision_summary.csv", SUMMARY_FIELDS, summaries)
    report["outputs"] = {name: dict(bytes=(output / name).stat().st_size, sha256=file_sha256(output / name))
                         for name in ("paired_history_phase_rates.csv", "history_precision_summary.csv")}
    with (output / "report.json").open("xb") as stream:
        stream.write(canonical_json(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.manifest, args.output)
    print(json.dumps({field: report[field] for field in ("status", "actual_episodes", "required_episodes")}))


if __name__ == "__main__":
    main()

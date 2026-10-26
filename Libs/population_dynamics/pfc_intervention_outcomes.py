"""Read existing native PFC outcomes after a frozen search and its replay.

The thirty source episodes already participated in checkpoint selection. Their
readback is descriptive development evidence, not independent intervention
qualification. Model seeds do not create additional native episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .circuit_data import CircuitDataset
from .pfc_intervention_experiment import (
    SEEDS, FACTUAL_EPISODES, compile_candidates, validate_config, _scores,
    _verify_replay_case, _reference_arrays, _save_npz, _write_csv,
)
from .pfc_intervention_references import load_reference
from .pfc_population_forecast_analysis import FrozenInputs
from .pfc_population_training_data import _declared_path, _execution_paths
from .pfc_renewal_data import PfcRenewalEpisode
from .population_intervention_replay import compare_arrays, verify_artifacts, _arrays
from .provenance import canonical_json, freeze_json, sha256_file
from .trajectory_scores import energy_score

READ_BIN_CHUNK = 100


def require(condition, message):
    if not condition:
        raise ValueError("PFC_SOURCE_OUTCOMES_" + message)


def read(path):
    return json.loads(Path(path).read_text())


def candidate_episode(factual, condition):
    return factual.rsplit("_condition_", 1)[0] + f"_condition_{condition:02d}"


def _case_index(summary):
    expected = {(seed, episode) for seed in SEEDS for episode in FACTUAL_EPISODES}
    rows = summary["cases"]
    index = {(row["training_seed"], row["factual_episode_id"]): row for row in rows}
    require(len(rows) == len(index) == 9 and set(index) == expected, "NINE_DISTINCT_SEED_BLOCK_CASES_REQUIRED")
    for (seed, episode), row in index.items():
        require(row["path"] == str(Path(f"seed_{seed}") / episode), "EXPLICIT_CASE_PATH_REQUIRED")
    return index


def admit_search(search, replay, config, target):
    """Check committed search/replay artifacts before source event admission."""
    search, replay = Path(search), Path(replay)
    summaries = [read(root / "analysis.json") for root in (search, replay)]
    statuses = ("COMPLETE_PFC_INTERVENTION_EXPERIMENT", "PASS_PFC_INTERVENTION_EXPERIMENT_REPLAY")
    for root, summary, status in zip((search, replay), summaries, statuses):
        require(summary.get("status") == status and summary.get("candidate_rows") == 90
                and summary.get("seed_block_cases") == 9
                and summary.get("generated_unit_count_values_across_completed_cases") == 141120000000
                and summary.get("scientific_claim_qualified") is False
                and summary.get("native_execution_performed") is False
                and summary.get("per_bin_full_unit_counts_exact_replay_claimed") is False,
                "COMPLETE_THREE_SEED_SEARCH_AND_REPLAY_REQUIRED")
        require(summary.get("artifacts", {}).keys() == {"candidates.csv"}
                and sha256_file(root / "candidates.csv") == summary["artifacts"]["candidates.csv"],
                "COMPLETE_SUMMARY_CSV_REQUIRED")
        with (root / "candidates.csv").open() as stream:
            require(len(list(csv.DictReader(stream))) == 90, "NINETY_SUMMARY_ROWS_REQUIRED")
    binding = summaries[0]["experiment_binding"]
    require(summaries[1]["experiment_binding"] == binding and binding["configuration"] == config
            and summaries[1]["saved_candidate_array_values_compared"] == 344982330,
            "SEARCH_REPLAY_CONFIGURATION_OR_BINDING_CHANGED")
    indexes = [_case_index(summary) for summary in summaries]
    cases = {}
    for key, entry in indexes[0].items():
        seed, episode = key
        path, replay_path = search / entry["path"], replay / indexes[1][key]["path"]
        require(sha256_file(path / "analysis.json") == entry["analysis_sha256"]
                and sha256_file(replay_path / "analysis.json") == indexes[1][key]["analysis_sha256"],
                "CASE_COMMIT_CHANGED")
        report = read(path / "analysis.json")
        case_binding = report["experiment_binding"]
        rows = report["candidates"]
        names = [row["candidate"] for row in rows]
        require(report.get("status") == "RANKED_DEVELOPMENT_COMPILED_PFC_CANDIDATES"
                and report.get("schema") == "pfc_population_intervention_case_v1"
                and report.get("source_unit_count") == 100000 and report.get("joint_draws") == 16
                and report.get("horizon_bins") == 980 and report.get("origin_tick") == 2000
                and report.get("sampling_stream_seeds") == list(SEEDS)
                and report.get("scientific_claim_qualified") is False
                and report.get("native_execution_performed") is False
                and len(rows) == len(set(names)) == 10
                and case_binding["experiment"] == binding and case_binding["training_seed"] == seed
                and case_binding["factual_episode_id"] == episode
                and case_binding["checkpoint_sha256"] == binding["seed_cells"][str(seed)]["consumed_files"]["training/best.pt"],
                "COMPLETE_BOUND_CANDIDATE_CASE_REQUIRED")
        conditions = [row["candidate_identity"]["condition_index"] for row in rows]
        require(conditions == list(range(10)), "ALL_TEN_SOURCE_CONDITIONS_REQUIRED")
        for condition, row in enumerate(rows):
            identity = row["source_config"]["benchmark_episode_candidate"]["identity"]
            require(identity == row["candidate_identity"] and identity["episode_id"] == candidate_episode(episode, condition),
                    "CANDIDATE_SOURCE_IDENTITY_CHANGED")
        verify_artifacts(path, report, names)
        compare_arrays(_arrays(path / "reference.npz"), _reference_arrays(target), "native_readback_reference")
        repeated = _verify_replay_case(replay_path, case_binding, entry["analysis_sha256"], names, 15680000000)
        selected = report["selected_candidate"]
        require(selected in names and repeated["selected_candidate"] == selected
                and entry["selected_candidate"] == indexes[1][key]["selected_candidate"] == selected
                and report["selected_action"]["source_config"] == rows[names.index(selected)]["source_config"]
                and report["selected_action"]["candidate_case_id"] == rows[names.index(selected)]["candidate_identity"]["episode_id"],
                "FROZEN_SELECTED_ACTION_CHANGED")
        cases[key] = dict(path=path, report=report)
    return binding, cases


def canonical_prefix(adapter, origin):
    keep = adapter.event_ticks < origin
    ticks = np.asarray(adapter.event_ticks[keep], dtype=np.int64)
    units = np.asarray(adapter.event_units[keep], dtype=np.int64)
    order = np.lexsort((units, ticks))
    return np.column_stack((ticks[order], units[order]))


def require_same_prefix(factual, candidate, origin):
    require(np.array_equal(factual.unit_ids, candidate.unit_ids)
            and np.array_equal(factual.unit_field_index, candidate.unit_field_index)
            and factual.field_names == candidate.field_names
            and factual.dt_ms == candidate.dt_ms and factual.bin_ticks == candidate.bin_ticks,
            "NATIVE_OBSERVATION_MAP_CHANGED")
    require(np.array_equal(canonical_prefix(factual, origin), canonical_prefix(candidate, origin)),
            "OBSERVED_NATIVE_SPIKE_PREFIX_DIFFERS")


def read_native_outcome(adapter, origin, horizon, *, chunk_bins=READ_BIN_CHUNK):
    """Aggregate every source unit using the existing aligned event reader."""
    require(type(origin) is int and origin >= 0 and origin % adapter.bin_ticks == 0
            and type(horizon) is int and horizon > 0 and type(chunk_bins) is int and chunk_bins > 0,
            "ALIGNED_POSITIVE_NATIVE_WINDOW_REQUIRED")
    stop = origin + horizon * adapter.bin_ticks
    require(stop <= adapter.duration_ticks, "NATIVE_WINDOW_EXCEEDS_SOURCE")
    units, fields = len(adapter.unit_ids), len(adapter.field_names)
    totals = np.zeros(units, dtype=np.int64)
    projected = []
    for start in range(origin, stop, chunk_bins * adapter.bin_ticks):
        end = min(stop, start + chunk_bins * adapter.bin_ticks)
        values = adapter.data._read(adapter.episode_id, start, end, adapter.bin_ticks, 0, units)
        bins = (end - start) // adapter.bin_ticks
        require(np.array_equal(values["unit_ids"], adapter.unit_ids)
                and np.array_equal(values["unit_field_index"], adapter.unit_field_index)
                and tuple(values["field_names"]) == tuple(adapter.field_names)
                and values["bin_seconds"] == adapter.bin_ticks * adapter.dt_ms / 1000,
                "SOURCE_READER_MAP_OR_UNITS_CHANGED")
        for name, shape in (("counts", (bins, units)), ("field_counts", (bins, fields))):
            array = values[name]
            require(array.shape == shape and array.dtype == np.int64 and np.all(array >= 0),
                    "COMPLETE_NONNEGATIVE_NATIVE_COUNTS_REQUIRED")
        totals += values["counts"].sum(axis=0)
        projected.append(values["field_counts"])
    counts = np.concatenate(projected, axis=0)
    field_units = np.bincount(adapter.unit_field_index, minlength=fields)
    require(field_units.shape == (fields,) and np.all(field_units > 0), "POSITIVE_NATIVE_FIELD_DENOMINATORS_REQUIRED")
    field_totals = np.zeros(fields, dtype=np.int64)
    np.add.at(field_totals, adapter.unit_field_index, totals)
    expected_events = np.count_nonzero((adapter.event_ticks >= origin) & (adapter.event_ticks < stop))
    require(np.array_equal(field_totals, counts.sum(axis=0)) and int(totals.sum()) == expected_events,
            "COMPLETE_NATIVE_EVENT_DENOMINATOR_REQUIRED")
    ready = np.asarray(adapter.ready_at(stop))
    require(ready.shape == (units,) and ready.dtype == np.int64 and np.all(ready >= 0),
            "COMPLETE_NATIVE_TERMINAL_CLOCKS_REQUIRED")
    return dict(field_counts=counts, field_rate_hz=counts / (field_units * adapter.bin_ticks * adapter.dt_ms / 1000),
                unit_total_counts=totals, terminal_ready_in_ticks=ready,
                unit_ids=np.asarray(adapter.unit_ids), field_ids=np.asarray(adapter.field_names))


def realized_reference_distance(rates, target):
    difference = (target.field_rate_hz - rates) / target.scale_hz
    return float(np.linalg.norm(difference.reshape(len(difference), -1), axis=1).mean() / np.sqrt(rates.size))


def _adapters(inputs, frozen):
    spec = read(inputs)
    require(spec == frozen.spec, "FROZEN_COHORT_CHANGED")
    mapping = _execution_paths(inputs, spec, frozen.config)
    def resolve(value, *, source_run=False):
        original = _declared_path(value, inputs, source_run=source_run)
        return mapping.get(original, original)
    rows = {row["episode_id"]: row for row in spec["episodes"] if row["partition"] == "development_validation"}
    expected = {candidate_episode(episode, condition) for episode in FACTUAL_EPISODES for condition in range(10)}
    require(len(rows) == 30 and set(rows) == expected, "THIRTY_NATIVE_VALIDATION_EPISODES_REQUIRED")
    def load(episode):
        row = rows[episode]
        dataset = CircuitDataset(resolve(row["dataset"]), splits=("development",))
        source = resolve(row["source_run"], source_run=True)
        adapter = PfcRenewalEpisode(dataset, episode, bin_ticks=100, source_run=source,
                                   source_entry=resolve(spec["source_entry"]))
        report = read(source / "report.json")
        require(sha256_file(source / "report.json") == adapter.binding["source_report_sha256"]
                and report["resolved_config"] == adapter.source_config
                and dataset.catalog["binding"]["resolved_source_config"] == adapter.source_config
                and len(adapter.unit_ids) == 100000 and len(adapter.field_names) == 17,
                "EXACT_NATIVE_PROGRAM_AND_SOURCE_SCALE_REQUIRED")
        return adapter
    return load


def run_outcome_analysis(inputs, preparation, config_path, reference_path, search, replay, output):
    inputs, preparation, config_path, reference_path, search, replay, output = map(
        Path, (inputs, preparation, config_path, reference_path, search, replay, output))
    config = validate_config(yaml.safe_load(config_path.read_text()))
    require(sha256_file(reference_path) == config["reference_sha256"], "REGISTERED_REFERENCE_CHANGED")
    target, reference_report = load_reference(reference_path)
    experiment, cases = admit_search(search, replay, config, target)
    require(experiment["inputs_sha256"] == reference_report["inputs_sha256"] == sha256_file(inputs)
            and experiment["preparation_sha256"] == sha256_file(preparation / "preparation.json")
            and experiment["config_sha256"] == sha256_file(config_path)
            and experiment["reference_sha256"] == config["reference_sha256"], "EXPERIMENT_INPUT_BINDING_CHANGED")
    # The wrapper authenticates this composed snapshot before execution. Binding
    # it also prevents reuse when an imported scoring or source-reader file changes.
    require((preparation / "source_snapshot.json").is_file(), "COMPOSED_RUNTIME_SNAPSHOT_REQUIRED")
    binding = dict(experiment=experiment, search_sha256=sha256_file(search / "analysis.json"),
        replay_sha256=sha256_file(replay / "analysis.json"), implementation_sha256=sha256_file(Path(__file__)),
        runtime_snapshot_sha256=sha256_file(preparation / "source_snapshot.json"))
    if output.exists():
        require((output / "analysis.json").is_file(), "INCOMPLETE_OUTPUT_PRESERVED_USE_NEW_DIRECTORY")
        previous = read(output / "analysis.json")
        require(previous.get("status") == "COMPLETE_PFC_INTERVENTION_SOURCE_OUTCOMES"
                and previous.get("binding") == binding, "EXISTING_NATIVE_READBACK_BINDING_CHANGED")
        for name, digest in previous["artifacts"].items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts
                    and sha256_file(output / relative) == digest, "COMMITTED_NATIVE_READBACK_CHANGED")
        return previous
    frozen = FrozenInputs(preparation)
    require(frozen.proof["input_sha256"] == experiment["inputs_sha256"], "FROZEN_INPUTS_CHANGED")
    load = _adapters(inputs, frozen)
    started = time.perf_counter()
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    native, native_rows, joined, artifact_names = {}, [], [], []
    programs = None
    for factual_id in FACTUAL_EPISODES:
        factual = load(factual_id)
        candidates = compile_candidates(factual, config["candidate_condition_indices"])
        names = [candidate.name for candidate in candidates]
        require(programs is None or programs == names, "COMMON_NATIVE_PROGRAM_NAMES_REQUIRED")
        programs = names
        for condition, candidate in enumerate(candidates):
            episode = candidate_episode(factual_id, condition)
            adapter = factual if condition == 0 else load(episode)
            require(canonical_json(candidate.source_config) == canonical_json(adapter.source_config),
                    "COMPILED_PROGRAM_DIFFERS_FROM_NATIVE_EXECUTION")
            require_same_prefix(factual, adapter, config["origin_tick"])
            require(tuple(adapter.field_names) == tuple(target.field_names)
                    and adapter.bin_ticks * adapter.dt_ms / 1000 == target.bin_seconds,
                    "NATIVE_REFERENCE_OBSERVATION_MAP_CHANGED")
            arrays = read_native_outcome(adapter, config["origin_tick"], config["horizon_bins"])
            filename = "sources/" + episode + ".npz"
            _save_npz(output / filename, arrays)
            artifact_names.append(filename)
            native[factual_id, candidate.name] = arrays["field_rate_hz"]
            distance = realized_reference_distance(arrays["field_rate_hz"], target)
            native_rows.append(dict(factual_episode_id=factual_id, candidate=candidate.name,
                source_episode_id=episode, condition_index=condition, source_binding=adapter.binding,
                source_arrays=filename, realized_reference_distance=distance))
            for seed in SEEDS:
                case = cases[seed, factual_id]
                report = case["report"]
                row = report["candidates"][condition]
                require(row["candidate"] == candidate.name
                        and canonical_json(row["source_config"]) == canonical_json(candidate.source_config)
                        and report["source_binding"] == factual.binding,
                        "PREDICTION_AND_NATIVE_PROGRAM_BINDING_CHANGED")
                with np.load(case["path"] / (candidate.name + ".npz"), allow_pickle=False) as saved:
                    rates = saved["field_rate_hz"]
                    require(rates.shape == (16, 980, 17) and np.isfinite(rates).all() and np.all(rates >= 0)
                            and np.array_equal(saved["unit_ids"], arrays["unit_ids"])
                            and np.array_equal(saved["field_ids"], arrays["field_ids"]), "PREDICTION_NATIVE_ARRAY_MAP_CHANGED")
                    predicted_score, _ = _scores(dict(field_rate_hz=rates), target)
                    require(np.isclose(predicted_score, row["objective"], rtol=1e-12, atol=1e-12),
                            "FROZEN_PREDICTED_OBJECTIVE_CHANGED")
                    source_score = energy_score(rates / target.scale_hz, arrays["field_rate_hz"] / target.scale_hz)
                joined.append(dict(training_seed=seed, factual_episode_id=factual_id, candidate=candidate.name,
                    source_episode_id=episode, condition_index=condition,
                    selected_action=candidate.name == report["selected_candidate"],
                    predicted_reference_energy_score=predicted_score,
                    source_forecast_energy_score=source_score, realized_reference_distance=distance))
    program_rows = []
    for name in programs:
        scores, _ = _scores(dict(field_rate_hz=np.stack([native[episode, name] for episode in FACTUAL_EPISODES])), target)
        program_rows.append(dict(candidate=name, native_source_blocks=3, reference_energy_score=scores))
    policy_rows = []
    for seed in SEEDS:
        selected = [cases[seed, episode]["report"]["selected_candidate"] for episode in FACTUAL_EPISODES]
        rates = np.stack([native[episode, name] for episode, name in zip(FACTUAL_EPISODES, selected)])
        score, _ = _scores(dict(field_rate_hz=rates), target)
        baseline = program_rows[0]["reference_energy_score"]
        policy_rows.append(dict(training_seed=seed, native_source_blocks=3, selected_programs=json.dumps(selected),
            reference_energy_score=score, factual_program=programs[0], factual_program_reference_energy_score=baseline,
            selected_minus_factual_reference_energy_score=score-baseline))
    require(len(native_rows) == 30 and len(joined) == 90 and sum(row["selected_action"] for row in joined) == 9
            and Counter(row["training_seed"] for row in joined) == {41: 30, 42: 30, 43: 30},
            "COMPLETE_NATIVE_AND_MODEL_DENOMINATORS_REQUIRED")
    native_binding = {row["source_binding"]["circuit_id"] for row in native_rows}
    native_blocks = {row["source_binding"]["block_id"] for row in native_rows}
    require(len(native_binding) == 1 and len(native_blocks) == 3, "ONE_CIRCUIT_THREE_NATIVE_BLOCKS_REQUIRED")
    for name, rows in (("candidates.csv", joined), ("programs.csv", program_rows), ("policies.csv", policy_rows)):
        _write_csv(output / name, rows)
        artifact_names.append(name)
    freeze_json(output / "sources.json", dict(episodes=native_rows))
    artifact_names.append("sources.json")
    result = dict(status="COMPLETE_PFC_INTERVENTION_SOURCE_OUTCOMES", binding=binding,
        candidate_rows=90, native_source_episodes=30, native_source_blocks=3, native_circuits=1,
        policy_rows=3, program_rows=10, observed_unit_bin_values_aggregated=2940000000,
        newly_generated_samples=0, existing_native_executions_reused=30,
        all_candidate_outcomes_retained=True, observed_prefixes_equal=True,
        observed_prefix_kind="native_spike_events_with_declared_source_current_commands",
        native_full_state_pairing_verified=False, native_future_noise_pairing_verified=False,
        independent_intervention_qualified=False, success_tolerance_qualified=False,
        memory_recovery_qualified=False, scientific_claim_qualified=False,
        native_execution_performed=False, training_performed=False,
        per_bin_full_unit_counts_saved=False, inferential_statistics_qualified=False,
        source_episodes_previously_used_for_checkpoint_selection=True,
        readback_and_analysis_seconds=time.perf_counter()-started,
        score_definitions=dict(realized_reference_distance="mean scaled trajectory distance to eight references; one native path",
            source_forecast_energy_score="energy_score(sixteen model paths, one native path), scaled by fitting reference RMS",
            program_and_policy_reference_energy_score="mean energy_score(three actual source-block paths, each reference)"),
        artifacts={name: sha256_file(output / name) for name in artifact_names})
    freeze_json(output / "analysis.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "preparation", "config", "reference", "search", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_outcome_analysis(args.inputs, args.preparation, args.config,
        args.reference, args.search, args.replay, args.output)))


if __name__ == "__main__":
    main()

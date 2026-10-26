"""Describe accepted PFC candidate fields without fitting or regenerating samples.

Native contrasts reuse development episodes. They are not paired causal effects.
The source acceptance supplies prior authentication; full scores check alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .trajectory_scores import energy_score

SEEDS = (41, 42, 43)
FACTUAL = tuple(f"pfc_working_memory_circuit_074_block_{b:03d}_condition_00" for b in range(3))
PROGRAMS = ("cue_none", "cue_aligned", "cue_off_target", "cue_antipodal",
            "cue_late_off_target", "omission_none", "omission_aligned",
            "omission_off_target", "omission_antipodal", "omission_late_off_target")
ORIGIN_MS = 100.0
BIN_MS = 5.0
BINS = 980


def require(condition, message):
    if not condition:
        raise ValueError("PFC_SAVED_DIAGNOSTIC_" + message)


def read(path):
    return json.loads(Path(path).read_text())


def index_rows(rows, keys, expected):
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(rows) == len(result) and set(result) == set(expected), "COMPLETE_UNIQUE_IDENTITIES_REQUIRED")
    return result


def components(samples, references):
    """Exact target-score distance and finite-ensemble dispersion subtraction."""
    x, r = np.asarray(samples, dtype=float), np.asarray(references, dtype=float)
    require(x.ndim >= 2 and r.ndim == x.ndim and x.shape[1:] == r.shape[1:]
            and len(x) >= 2 and len(r) > 0 and np.isfinite(x).all()
            and np.isfinite(r).all(), "FINITE_MATCHING_TRAJECTORIES_REQUIRED")
    x, r = x.reshape(len(x), -1), r.reshape(len(r), -1)
    norm = np.sqrt(x.shape[1])
    distance = float(np.mean([np.linalg.norm(x-y, axis=1).mean() for y in r]) / norm)
    dispersion = float(sum(np.linalg.norm(x[i+1:]-x[i], axis=1).sum()
                           for i in range(len(x)-1)) / (len(x)*(len(x)-1)*norm))
    mean_distance = float(np.linalg.norm(r-x.mean(0), axis=1).mean()/norm)
    return dict(target_distance=distance, dispersion_subtraction=dispersion,
                reference_energy_score=distance-dispersion,
                ensemble_mean_reference_distance=mean_distance)


def ranks(values):
    """Average ranks, with exact-value tie counts; lower scores rank first."""
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), "FINITE_RANK_VALUES_REQUIRED")
    return [(float(1 + np.count_nonzero(x < v) + (np.count_nonzero(x == v)-1)/2),
             int(np.count_nonzero(x == v))) for v in x]


def active_intervals(config):
    result = []
    for kind in ("stimulus", "distractor"):
        start, duration, strength = (float(config[kind+s]) for s in
                                     ("_start_ms", "_duration_ms", "_strength_nA"))
        require(np.isfinite([start, duration, strength]).all() and duration >= 0,
                "FINITE_CONTROL_SCHEDULE_REQUIRED")
        if duration > 0 and strength != 0:
            result.append((start, start+duration))
    return result


def phase_windows(configs):
    """Partition the complete task at actual active-control boundaries."""
    edges = {ORIGIN_MS, ORIGIN_MS+BINS*BIN_MS}
    for config in configs:
        for start, stop in active_intervals(config):
            edges.update(v for v in (start, stop) if ORIGIN_MS < v < ORIGIN_MS+BINS*BIN_MS)
    edges = sorted(edges)
    require(all((x-ORIGIN_MS)/BIN_MS == int((x-ORIGIN_MS)/BIN_MS) for x in edges),
            "CONTROL_BOUNDARIES_MUST_ALIGN_WITH_SAVED_BINS")
    return [(a, b, int((a-ORIGIN_MS)/BIN_MS), int((b-ORIGIN_MS)/BIN_MS))
            for a, b in zip(edges[:-1], edges[1:])]


def field_arrays(path, fields=None, units=None, *, reference=False, native=False):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in (("field_rate_hz", "field_ids", "scale_hz", "block_ids")
                                      if reference else ("field_rate_hz", "field_ids", "unit_ids"))}
    expected = (8, BINS, 17) if reference else ((BINS, 17) if native else (16, BINS, 17))
    require(result["field_rate_hz"].shape == expected
            and np.isfinite(result["field_rate_hz"]).all(), "COMPLETE_FINITE_FIELD_ARRAY_REQUIRED")
    require(result["field_ids"].shape == (17,) and len(set(result["field_ids"].tolist())) == 17,
            "DISTINCT_FIELD_IDENTITIES_REQUIRED")
    if fields is not None:
        require(np.array_equal(result["field_ids"], fields), "FIELD_ORDER_CHANGED")
    if reference:
        require(result["scale_hz"].shape == (17,) and np.isfinite(result["scale_hz"]).all()
                and np.all(result["scale_hz"] > 0), "POSITIVE_FROZEN_SCALES_REQUIRED")
    else:
        require(result["unit_ids"].shape == (100000,), "COMPLETE_UNIT_IDENTITIES_REQUIRED")
        if units is not None:
            require(np.array_equal(result["unit_ids"], units), "UNIT_ORDER_CHANGED")
    return result


def _rms(values, axis=None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def temporal_metrics(samples, native, factual_mean, factual_native):
    mean = samples.mean(0)
    predicted_contrast, native_contrast = mean-factual_mean, native-factual_native
    return dict(mean_error_rms=_rms(mean-native, axis=1),
                ensemble_spread_rms=np.sqrt(samples.var(0, ddof=1).mean(1)),
                predicted_contrast_rms=_rms(predicted_contrast, axis=1),
                native_contrast_rms=_rms(native_contrast, axis=1),
                contrast_error_rms=_rms(predicted_contrast-native_contrast, axis=1))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(execution, acceptance, output):
    started = time.perf_counter()
    execution, acceptance, output = map(Path, (execution, acceptance, output))
    prior = read(acceptance)
    require(prior["status"] == "ACCEPTED_COMPLETE_PFC_INTERVENTION_DEVELOPMENT_DELIVERY"
            and prior["verified_training_seeds"] == list(SEEDS)
            and prior["candidate_rows"] == 90 and prior["native_source_episodes_reused"] == 30
            and prior["scientific_claim_qualified"] is False, "ACCEPTED_DEVELOPMENT_DELIVERY_REQUIRED")
    require(hashlib.sha256((execution/"delivery.json").read_bytes()).hexdigest() == prior["delivery_sha256"],
            "ACCEPTED_DELIVERY_BINDING_CHANGED")
    search, source = execution/"results", execution/"source_outcomes"
    analysis, native_report = read(search/"analysis.json"), read(source/"analysis.json")
    require(analysis["experiment_binding"] == prior["verified_binding"]["experiment"]
            and native_report["candidate_rows"] == 90
            and native_report["scientific_claim_qualified"] is False, "ACCEPTED_EXPERIMENT_CHANGED")
    cases = index_rows(analysis["cases"], ("training_seed", "factual_episode_id"),
                       ((s, f) for s in SEEDS for f in FACTUAL))
    natives = index_rows(read(source/"sources.json")["episodes"], ("factual_episode_id", "candidate"),
                         ((f, p) for f in FACTUAL for p in PROGRAMS))
    with (source/"candidates.csv").open() as stream:
        joined = list(csv.DictReader(stream))
    for row in joined:
        row["training_seed"] = int(row["training_seed"])
    joins = index_rows(joined, ("training_seed", "factual_episode_id", "candidate"),
                       ((s, f, p) for s in SEEDS for f in FACTUAL for p in PROGRAMS))
    first = search/cases[SEEDS[0], FACTUAL[0]]["path"]
    ref = field_arrays(first/"reference.npz", reference=True)
    scale, fields = ref["scale_hz"], ref["field_ids"]
    references = ref["field_rate_hz"]/scale
    actual, unit_ids = {}, None
    for key, row in natives.items():
        binding = row["source_binding"]
        require(binding["source_dt_ms"] == .05 and binding["bin_ticks"] == 100
                and binding["episode_id"] == row["source_episode_id"], "NATIVE_TIME_OR_EPISODE_CHANGED")
        arr = field_arrays(source/row["source_arrays"], fields, unit_ids, native=True)
        if unit_ids is None:
            unit_ids = arr["unit_ids"]
        actual[key] = arr["field_rate_hz"]/scale
    full_rows, bins_rows, horizon_rows, phase_rows, pair_rows = [], [], [], [], []
    score_differences = []
    for (seed, factual), case in cases.items():
        path = search/case["path"]
        report = read(path/"analysis.json")
        local_ref = field_arrays(path/"reference.npz", fields, reference=True)
        require(all(np.array_equal(local_ref[k], ref[k]) for k in ref), "REFERENCE_CHANGED_BETWEEN_CASES")
        configurations = {row["candidate"]: row["source_config"] for row in report["candidates"]}
        require(set(configurations) == set(PROGRAMS), "ALL_PROGRAM_CONFIGURATIONS_REQUIRED")
        windows = phase_windows(configurations.values())
        generated = {p: field_arrays(path/(p+".npz"), fields, unit_ids)["field_rate_hz"]/scale for p in PROGRAMS}
        factual_mean = generated["cue_none"].mean(0)
        factual_native = actual[factual, "cue_none"]
        case_rows = []
        for program in PROGRAMS:
            samples, native = generated[program], actual[factual, program]
            ident = dict(training_seed=seed, factual_episode_id=factual, candidate=program,
                         source_episode_id=natives[factual, program]["source_episode_id"],
                         frozen_selected=program == case["selected_candidate"])
            parts = components(samples, references)
            forecast = energy_score(samples, native)
            native_distance = float(np.mean([_rms(native-r) for r in references]))
            expected = joins[seed, factual, program]
            require(expected["source_episode_id"] == ident["source_episode_id"], "JOINED_SOURCE_CHANGED")
            for value, key in ((parts["reference_energy_score"], "predicted_reference_energy_score"),
                               (forecast, "source_forecast_energy_score"),
                               (native_distance, "realized_reference_distance")):
                score_differences.append(abs(value-float(expected[key])))
            row = dict(ident, **parts, source_forecast_energy_score=forecast,
                       realized_reference_distance=native_distance)
            case_rows.append(row)
            temporal = temporal_metrics(samples, native, factual_mean, factual_native)
            bins_rows.extend(dict(ident, bin_index=i, start_ms=ORIGIN_MS+i*BIN_MS,
                                  **{k: float(v[i]) for k, v in temporal.items()}) for i in range(BINS))
            for horizon in (4, 16, 100, BINS):
                horizon_rows.append(dict(ident, horizon_ms=horizon*BIN_MS,
                    **components(samples[:, :horizon], references[:, :horizon]),
                    source_forecast_energy_score=energy_score(samples[:, :horizon], native[:horizon]),
                    mean_error_rms=float(_rms(samples[:, :horizon].mean(0)-native[:horizon]))))
            mean = samples.mean(0)
            cessation = max([ORIGIN_MS]+[end for cfg in (configurations[program], configurations["cue_none"])
                                        for _, end in active_intervals(cfg)])
            for start_ms, stop_ms, a, b in windows:
                for field, name in enumerate(fields):
                    phase_rows.append(dict(ident, start_ms=start_ms, stop_ms=stop_ms, field_id=str(name),
                        all_compared_controls_off=start_ms >= cessation,
                        predicted_rate_hz=float(mean[a:b, field].mean()*scale[field]),
                        native_rate_hz=float(native[a:b, field].mean()*scale[field]),
                        predicted_contrast_hz=float((mean-factual_mean)[a:b, field].mean()*scale[field]),
                        native_contrast_hz=float((native-factual_native)[a:b, field].mean()*scale[field])))
        for metric, label in (("reference_energy_score", "model"), ("target_distance", "distance"),
                              ("realized_reference_distance", "native_distance")):
            for row, (rank, ties) in zip(case_rows, ranks([r[metric] for r in case_rows])):
                row[label+"_rank"] = rank
                row[label+"_tie_count"] = ties
        full_rows.extend(case_rows)
        # The only difference in this equal-amplitude pair is target geometry.
        left, right = "cue_aligned", "cue_antipodal"
        delta_model = generated[left].mean(0)-generated[right].mean(0)
        delta_native = actual[factual, left]-actual[factual, right]
        cessation = max(end for p in (left, right) for _, end in active_intervals(configurations[p]))
        for start_ms, stop_ms, a, b in windows:
            pair_rows.append(dict(training_seed=seed, factual_episode_id=factual,
                left=left, right=right, start_ms=start_ms, stop_ms=stop_ms,
                all_compared_controls_off=start_ms >= cessation,
                model_contrast_rms=float(_rms(delta_model[a:b])),
                native_contrast_rms=float(_rms(delta_native[a:b])),
                contrast_error_rms=float(_rms(delta_model[a:b]-delta_native[a:b]))))
    require(max(score_differences) < 1e-10, "ORIGINAL_FULL_HORIZON_SCORES_NOT_REPRODUCED")
    require(not output.exists(), "OUTPUT_ALREADY_EXISTS")
    output.mkdir(parents=True)
    artifacts = {"full_horizon.csv": full_rows, "temporal.csv": bins_rows,
                 "horizons.csv": horizon_rows, "field_phases.csv": phase_rows,
                 "aligned_antipodal_phases.csv": pair_rows}
    for name, rows in artifacts.items():
        write_csv(output/name, rows)
    result = dict(status="COMPLETE_DESCRIPTIVE_SAVED_PFC_FIELD_DIAGNOSTIC",
        accepted_execution=prior["execution"], acceptance=str(acceptance),
        acceptance_sha256=hashlib.sha256(acceptance.read_bytes()).hexdigest(),
        delivery_sha256=prior["delivery_sha256"], training_seeds=list(SEEDS),
        candidate_rows=len(full_rows), temporal_rows=len(bins_rows), native_episodes=30,
        native_blocks=3, native_circuits=1, fields=17, origin_ms=ORIGIN_MS, bin_ms=BIN_MS,
        horizon_bins=BINS, horizon_ms=BINS*BIN_MS, full_horizon_scores_recomputed=len(score_differences),
        maximum_original_score_difference=max(score_differences),
        artifact_rows={name:len(rows) for name, rows in artifacts.items()},
        field_scales_hz=scale.tolist(), scale_refit=False, prior_payload_authentication_repeated=False,
        samples_regenerated=False, source_spikes_read=False, training_performed=False,
        frozen_selections_changed=False, native_full_state_pairing_verified=False,
        native_future_noise_pairing_verified=False, scientific_claim_qualified=False,
        inferential_statistics_performed=False, source_episodes_previously_used_for_checkpoint_selection=True,
        score_definitions=dict(target_distance="mean normalized draw-to-reference Euclidean distance",
            dispersion_subtraction="sum over unordered draw pairs divided by M*(M-1)*sqrt(T*F)",
            temporal_rms="RMS across fields after dividing by saved fitting-only field scales",
            ensemble_spread_rms="root mean across fields of unbiased draw variance",
            contrasts="candidate minus cue_none; descriptive native paths; no causal pairing claim",
            phases="complete nonoverlapping task partition at active-control boundaries"),
        elapsed_seconds=time.perf_counter()-started)
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("execution", "acceptance", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.acceptance, args.output), indent=2))


if __name__ == "__main__":
    main()

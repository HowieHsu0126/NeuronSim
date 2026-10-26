"""Read back shared PFC census blocks and compare frozen empirical primitives."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .pfc_history_profile_census import BOUNDARY, frame_requests, write_json
from .pfc_history_timing_census import require


INTEGER_FIELDS = (
    "source_rng_seed", "field_index", "row_index", "count_threshold", "physical_window_denominator",
    "coarse_cdf_count", "fine_cdf_count", "coarse_tail_count", "fine_tail_count",
    "paired_discordant_windows", "fine_minus_coarse_cdf_count",
)


def aggregate_counts(rows: Sequence[Mapping[str, Any]], required_seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Sum archived counts only after complete, equally sized block checks."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["auxiliary_repeat"], row["row_index"])].append(row)
    summaries = []
    for (repeat, row_index), values in sorted(groups.items()):
        require(sorted(row["source_rng_seed"] for row in values) == list(required_seeds),
                "PROFILE_SUMMARY_BLOCK_DENOMINATOR")
        sizes = {row["physical_window_denominator"] for row in values}
        require(len(sizes) == 1 and next(iter(sizes)) > 0, "PROFILE_SUMMARY_FRAME_SIZE_CHANGED")
        for row in values:
            n = row["physical_window_denominator"]
            for clock in ("coarse", "fine"):
                cdf, tail = row[clock + "_cdf_count"], row[clock + "_tail_count"]
                require(0 <= cdf <= n and 0 <= tail <= n and cdf + tail == n,
                        "PROFILE_SUMMARY_COUNT_DENOMINATOR")
            delta = row["fine_cdf_count"] - row["coarse_cdf_count"]
            require(delta == row["fine_minus_coarse_cdf_count"]
                    and abs(delta) <= row["paired_discordant_windows"] <= n
                    and math.isclose(row["fine_minus_coarse_empirical_cdf"], delta / n, abs_tol=1e-15),
                    "PROFILE_SUMMARY_PAIRED_COUNTS")
        sums = {key: sum(row[key] for row in values) for key in INTEGER_FIELDS[4:]}
        # Physical-window denominator is also summed, but windows remain dependent within blocks.
        denominator = sums["physical_window_denominator"]
        summaries.append(dict(auxiliary_repeat=repeat, row_index=row_index, original_blocks=len(values),
                              count_threshold=values[0]["count_threshold"], **sums,
                              fine_minus_coarse_empirical_cdf=sums["fine_minus_coarse_cdf_count"] / denominator,
                              source_law_probability_or_interval_established=False))
    return summaries


def _bound_bytes(root: Path, entry: Mapping[str, Any], destination: Path) -> bytes:
    data = (root / entry["path"]).read_bytes()
    require(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
            "PROFILE_SUMMARY_INPUT_CHANGED:" + entry["path"])
    target = destination / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def summarize(settings: Mapping[str, Any], run_index: Mapping[str, Any], repo: Path,
              output: Path, reference_csv: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    require(settings["required_source_seeds"] == list(range(95001, 95033)), "PROFILE_SUMMARY_ORIGINAL_BLOCKS")
    frame_requests(settings)
    required = settings["required_source_seeds"]
    seeds = sorted(row["source_rng_seed"] for row in run_index["runs"])
    require(len(seeds) == len(set(seeds)) and set(seeds) <= set(required) and seeds,
            "PROFILE_SUMMARY_DUPLICATE_OR_FOREIGN_BLOCK")
    complete = seeds == required
    require(complete or allow_partial, "PROFILE_SUMMARY_ALL32_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    reference = {}
    with reference_csv.open() as stream:
        for row in csv.DictReader(stream):
            reference[(row["realization"], int(row["domain_row_index"]))] = row
    reference_bytes = reference_csv.read_bytes()
    (output / "original_gap_moment_envelopes.csv").write_bytes(reference_bytes)
    expected_profiles = {(repeat, p["left_cohort"], p["right_cohort"], p["field_index"], p["phase"]): p
                         for repeat in settings["auxiliary_repeats"] for p in settings["profiles"]}
    expected_endpoint_ids = {(repeat, endpoint["row_index"])
                             for repeat in settings["auxiliary_repeats"]
                             for profile in settings["profiles"] for endpoint in profile["endpoints"]}
    all_rows, census_groups, block_costs = [], defaultdict(list), []
    verified_files = 0
    for binding in sorted(run_index["runs"], key=lambda row: row["source_rng_seed"]):
        seed = binding["source_rng_seed"]
        origin = repo / binding["archive"]
        staged = output / "inputs" / str(seed)
        data = (origin / "delivery.json").read_bytes()
        require(hashlib.sha256(data).hexdigest() == binding["delivery_sha256"], "PROFILE_SUMMARY_DELIVERY_CHANGED")
        delivery = json.loads(data)
        require(delivery["verified"] and delivery["status"] == "COMPLETE_PFC_HISTORY_PROFILE_CENSUS_COMPLETE",
                "PROFILE_SUMMARY_SOURCE_JOB_INCOMPLETE")
        staged.mkdir(parents=True)
        (staged / "delivery.json").write_bytes(data)
        entries = {row["path"]: row for row in delivery["files"]}
        require(entries["Libs/configs/neural_population_pfc_history_profile_census.yaml"]["sha256"] == run_index["config_sha256"],
                "PROFILE_SUMMARY_PROTOCOL_CHANGED")
        def read(name: str):
            nonlocal verified_files
            verified_files += 1
            return _bound_bytes(origin, entries[name], staged)
        wrapper = json.loads(read("report.json"))
        require(wrapper["exit_code"] == 0 and wrapper["error"] is None, "PROFILE_SUMMARY_WRAPPER_FAILED")
        block = json.loads(read("analysis/report.json"))
        require(block["status"] == "COMPLETE_PFC_HISTORY_PROFILE_BLOCK" and block["source_rng_seed"] == seed
                and block["completed_profile_pairs"] == 72 and len(block["results"]) == 72,
                "PROFILE_SUMMARY_BLOCK_INCOMPLETE")
        raw_rows = list(csv.DictReader(io.StringIO(read("analysis/endpoint_counts.csv").decode())))
        rows = [{key: int(value) if key in INTEGER_FIELDS else float(value) if key == "fine_minus_coarse_empirical_cdf" else value
                 for key, value in row.items()} for row in raw_rows]
        require(len(rows) == 128 and {(r["auxiliary_repeat"], r["row_index"]) for r in rows} == expected_endpoint_ids
                and all(r["source_rng_seed"] == seed for r in rows), "PROFILE_SUMMARY_ENDPOINT_COVERAGE")
        endpoint_lookup = {(r["auxiliary_repeat"], r["row_index"]): r for r in rows}
        seen = set()
        for item in block["results"]:
            result = json.loads(read("analysis/" + item["report"]))
            p = result["profile"]
            key = (result["auxiliary_repeat"], p["left_cohort"], p["right_cohort"], p["field_index"], p["phase"])
            require(key in expected_profiles and key not in seen and p == expected_profiles[key]
                    and result["source_rng_seed"] == seed, "PROFILE_SUMMARY_FRAME_IDENTITY")
            seen.add(key)
            size = result["frame_size"]
            require(sum(result["joint_time_bin_counts"].values()) == size
                    and result["discarded_physical_windows"] == 0, "PROFILE_SUMMARY_FULL_FRAME")
            for clock in ("coarse", "fine"):
                meta = result["clocks"][clock]
                require(meta["frame_size"] == size and meta["field_units"] == 5000
                        and meta["field_index"] == p["field_index"] and meta["phase"] == p["phase"]
                        and meta["window_ms"] == 5. and meta["time_edges_ms"] == [0., 1., 2., 3.],
                        "PROFILE_SUMMARY_CLOCK_FRAME_CHANGED")
            require(result["endpoints"] == item["endpoints"], "PROFILE_SUMMARY_BLOCK_FRAME_DISAGREEMENT")
            for endpoint in result["endpoints"]:
                row = endpoint_lookup[(key[0], endpoint["row_index"])]
                require(all(row[name] == value for name, value in endpoint.items())
                        and all(row[name] == p[name] for name in ("left_cohort", "right_cohort", "field_index", "phase")),
                        "PROFILE_SUMMARY_CSV_FRAME_DISAGREEMENT")
            require("analysis/" + str(Path(item["report"]).with_name("frame_assignments.npz")) in entries,
                    "PROFILE_SUMMARY_ASSIGNMENTS_NOT_DELIVERED")
            census_groups[key].append(result)
        require(seen == set(expected_profiles), "PROFILE_SUMMARY_ALL72_REQUIRED")
        all_rows.extend(rows)
        block_costs.append(dict(source_rng_seed=seed, wrapper_seconds=wrapper["wall_seconds"],
                               peak_rss_kib=block["peak_rss_kib"], source_cost=block["source_cost"],
                               delivered_files=len(entries), delivered_bytes=sum(r["bytes"] for r in entries.values())))
    summaries = aggregate_counts(all_rows, seeds)
    reference_checks = []
    timing = []
    for key, values in sorted(census_groups.items()):
        repeat, left, right, field, phase = key
        n = sum(row["frame_size"] for row in values)
        counts = {str(i): sum(row["joint_time_bin_counts"][str(i)] for row in values) for i in range(16)}
        metadata = dict(auxiliary_repeat=repeat, left_cohort=left, right_cohort=right, field_index=field,
                        phase=phase, original_blocks=len(values), frame_size=n,
                        joint_time_bin_counts=counts,
                        joint_time_bin_weights={k: v/n for k, v in counts.items()},
                        clocks={clock: {name: sum(row["clocks"][clock][name] for row in values)
                                        for name in ("observed_spikes", "no_first_spike_windows", "refractory_zero_windows", "eligible_windows", "multispike_windows")}
                                for clock in ("coarse", "fine")})
        timing.append(metadata)
        if complete:
            for endpoint in expected_profiles[key]["endpoints"]:
                prior = reference[(repeat, endpoint["row_index"])]
                identity = json.loads(prior["endpoint_identity"])
                require(all(str(expected_profiles[key][name]) == identity[name] for name in
                            ("condition", "field_index", "left_cohort", "phase", "right_cohort", "window_ms"))
                        and int(identity["count_threshold"]) == endpoint["count_threshold"], "PROFILE_SUMMARY_REFERENCE_IDENTITY")
                observed = next(row for row in summaries if row["auxiliary_repeat"] == repeat and row["row_index"] == endpoint["row_index"])
                cdf_ok = math.isclose(observed["fine_minus_coarse_empirical_cdf"], float(prior["empirical_difference"]), rel_tol=0, abs_tol=1e-12)
                means_ok = all(math.isclose(metadata["clocks"][clock]["observed_spikes"] / n, float(prior[name]), rel_tol=0, abs_tol=1e-12)
                               for clock, name in (("coarse", "left_mean"), ("fine", "right_mean")))
                reference_checks.append(dict(auxiliary_repeat=repeat, row_index=endpoint["row_index"], cdf_matches=cdf_ok, first_moments_match=means_ok))
    failures = [row for row in reference_checks if not row["cdf_matches"] or not row["first_moments_match"]]
    report = dict(status=("COMPLETE_SHARED_PROFILE_CENSUS_READBACK" if complete else "PARTIAL_SHARED_PROFILE_CENSUS_READBACK") if not failures else "FAILED_SHARED_PROFILE_REFERENCE_COMPARISON",
                  required_original_blocks=32, completed_original_blocks=len(seeds), completed_source_seeds=seeds,
                  profile_pairs_per_block=72, completed_profile_pairs=len(seeds)*72,
                  endpoint_realizations_per_block=128, completed_endpoint_records=len(all_rows),
                  original_endpoints_retained=84330, confirmation_block_cap=128,
                  distinct_physical_unit_windows_per_block=11600000,
                  paired_frame_visits=sum(r["frame_size"] for r in timing),
                  denominators_scope="Original source blocks are retained; A/B, clocks, windows and units are not extra independent blocks",
                  reference_comparison_full32=complete, reference_checks=reference_checks, failures=failures,
                  files_verified=verified_files, block_costs=block_costs, endpoint_summaries=summaries,
                  timing_profiles=timing, **BOUNDARY)
    write_json(output / "report.json", report)
    with (output / "endpoint_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    require(not failures, "PROFILE_SUMMARY_REFERENCE_MISMATCH")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "run-index", "project-root", "reference-csv", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = summarize(yaml.safe_load(args.config.read_text()), json.loads(args.run_index.read_text()),
                       args.project_root, args.output, args.reference_csv, allow_partial=args.allow_partial)
    print(json.dumps({key: report[key] for key in ("status", "completed_original_blocks", "completed_profile_pairs", "completed_endpoint_records", "files_verified")}))


if __name__ == "__main__":
    main()

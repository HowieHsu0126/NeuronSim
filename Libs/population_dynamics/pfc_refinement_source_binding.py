"""Bind exact PFC refinement parameter bounds to indexed frozen source metadata.

This verifies parameter and arithmetic applicability under ideal conditional
sampling. It does not certify the finite sampler, PRNG, neural clock comparison,
source-history confidence or production. Source payloads are never read.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import yaml

from .pfc_input_pairing import _relative
from .pfc_native_drive import _BINOMIAL_N, _RATE_HZ
from .pfc_refinement_input_law import _rational, routing_law
from .provenance import freeze_json

SUPPORTED_CODE_HASHES = {
    "pfc_clock_coupling": "45640118f4e6ea20db9a897fdb700ca254177cfc0a5dd06493dd9b7a3ad7230b",
    "pfc_native_drive": "8b32fb865c71c89ce6c0b75f99ab3acf1da628ed3ee976035c3e00a5865ab30e",
    "pfc_coupled_drive": "87b549468ce613d8fcb24a6518a0507efd0a703b1ca47f86a069ad0bc29962f5",
}
SOURCE_CLOCKS = {"native": 0.05, "half_A": 0.025, "quarter_A": 0.0125,
                 "half_B": 0.025, "quarter_B": 0.0125}
_MAX_READ_PATHS = 1000
_MAX_READ_BYTES = 256 * 1024 * 1024


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("PFC_REFINEMENT_SOURCE_BINDING_" + reason)


def _exact(value: dict) -> Fraction:
    _require(isinstance(value, dict) and type(value.get("numerator")) is str
             and type(value.get("denominator")) is str, "EXACT_RATIONAL_REQUIRED")
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    _require(int(value["denominator"]) > 0, "POSITIVE_RATIONAL_DENOMINATOR_REQUIRED")
    return result


def _number(value) -> Fraction:
    _require(type(value) in (int, float) and math.isfinite(value),
             "FINITE_BINARY64_METADATA_NUMBER_REQUIRED")
    return Fraction.from_float(float(value))


def _integer(value, expected: int, reason: str) -> None:
    _require(type(value) is int and value == expected, reason)


def _entries(delivery: dict) -> dict:
    _require(isinstance(delivery, dict) and isinstance(delivery.get("files"), list),
             "DELIVERY_FILE_TABLE_REQUIRED")
    result = {}
    for row in delivery["files"]:
        _require(isinstance(row, dict) and type(row.get("path")) is str,
                 "DELIVERY_FILE_ENTRY_REQUIRED")
        name = _relative(row["path"])
        _require(name not in result, "DUPLICATE_DELIVERY_FILE_ENTRY")
        _require(type(row.get("bytes")) is int and row["bytes"] >= 0
                 and type(row.get("sha256")) is str and len(row["sha256"]) == 64,
                 "DELIVERY_FILE_DIGEST_REQUIRED")
        result[name] = row
    return result


class _Reader:
    """Read each explicit small artifact once and bind parsing to those bytes."""

    def __init__(self):
        self.snapshots = {}
        self.total_bytes = 0

    def read(self, path: Path, entry: dict | None = None) -> bytes:
        path = Path(path).absolute()
        if path not in self.snapshots:
            _require(len(self.snapshots) < _MAX_READ_PATHS, "EXPLICIT_READ_PATH_LIMIT")
            remaining = _MAX_READ_BYTES - self.total_bytes
            with path.open("rb") as stream:
                data = stream.read(remaining + 1)
            _require(len(data) <= remaining, "EXPLICIT_READ_BYTE_LIMIT")
            digest = hashlib.sha256(data).hexdigest()
            self.snapshots[path] = (data, digest)
            self.total_bytes += len(data)
        data, digest = self.snapshots[path]
        if entry is not None:
            _require(len(data) == entry["bytes"] and digest == entry["sha256"],
                     "METADATA_OR_CODE_DIGEST_MISMATCH:" + str(path))
        return data

    def digest(self, path: Path) -> str:
        self.read(path)
        return self.snapshots[Path(path).absolute()][1]

    def records(self) -> list[dict]:
        return [dict(path=str(path), bytes=len(data), sha256=digest)
                for path, (data, digest) in self.snapshots.items()]


def _validate_analysis(analysis: dict) -> tuple[Fraction, Fraction]:
    _require(analysis.get("status") == "COMPLETE_EXACT_PFC_REFINEMENT_PARAMETER_INPUT_LAW_ANALYSIS"
             and analysis.get("scope") == "ideal_conditional_input_law_parameter_perturbation_only"
             and analysis.get("comparison_changes_only") == "binary64_r_to_exact_real_r_star_at_unchanged_binary64_q",
             "PARENT_ANALYSIS_SCOPE_CHANGED")
    for key, expected in (("binomial_n", _BINOMIAL_N), ("source_core_units", 100000),
                          ("excitatory_neurons", 80000), ("inhibitory_neurons", 20000),
                          ("coarse_ticks", 100000), ("final_ticks", 400000),
                          ("refinement_levels", 2)):
        _integer(analysis.get(key), expected, "PARENT_ANALYSIS_DIMENSIONS_CHANGED")
    _require(_number(analysis.get("coarse_dt_ms")) == Fraction.from_float(0.05)
             and _number(analysis.get("duration_ms")) == 5000,
             "PARENT_ANALYSIS_CLOCK_CHANGED")
    for key in ("finite_prng_law_certified", "finite_sampler_law_certified",
                "coarse_fine_neuronal_equality_certified", "source_history_confidence_certified",
                "confirmation_0p01_128_certified", "confirmation_execution_released",
                "confirmation_sampling_design_changed", "numerical_margins_changed",
                "error_budget_allocated", "protocol_amendment_adopted",
                "production_qualified", "scientific_fidelity_qualified"):
        _require(analysis.get(key) is False, "PARENT_ANALYSIS_QUALIFICATION_CHANGED:" + key)
    for key in ("new_episodes", "new_trials", "new_sampling", "source_simulations", "production_episodes"):
        _integer(analysis.get(key), 0, "PARENT_ANALYSIS_EXECUTION_COUNT_CHANGED")
    rows = analysis.get("levels")
    _require(isinstance(rows, list) and len(rows) == 2, "PARENT_TWO_LEVEL_ANALYSIS_REQUIRED")
    p0_float = _RATE_HZ * (0.05 * 1e-3)
    p0 = Fraction.from_float(p0_float)
    union = Fraction(0)
    for level, row in enumerate(rows):
        law = routing_law(p0_float / 2 ** level)
        _integer(row.get("level"), level, "PARENT_REFINEMENT_LEVEL_CHANGED")
        _integer(row.get("parent_ticks"), 100000 * 2 ** level, "PARENT_REFINEMENT_TICKS_CHANGED")
        _integer(row.get("child_ticks"), 200000 * 2 ** level, "PARENT_REFINEMENT_TICKS_CHANGED")
        _require(_number(row.get("parent_dt_ms")) == Fraction.from_float(0.05 / 2 ** level)
                 and _number(row.get("child_dt_ms")) == Fraction.from_float(0.025 / 2 ** level),
                 "PARENT_REFINEMENT_CLOCK_CHANGED")
        for key in ("parent_probability", "q", "q_exact_parent_quarter", "q_rounding_error",
                    "r", "r_star", "rho"):
            _require(_exact(row.get(key)) == law[key], "PARENT_EXACT_PARAMETER_CHANGED:" + key)
        _require(_exact(row.get("per_active_parent_tv")) == law["category_total_variation"],
                 "PARENT_CATEGORY_TV_CHANGED")
        for reported, expected in (("actual_category_probabilities", "actual_categories"),
                                   ("ideal_category_probabilities", "ideal_categories")):
            _require(set(row.get(reported, {})) == set(law[expected]), "PARENT_CATEGORY_FAMILY_CHANGED")
            for name, probability in law[expected].items():
                _require(_exact(row[reported][name]) == probability, "PARENT_CATEGORY_PROBABILITY_CHANGED")
        contribution = _BINOMIAL_N * 100000 * row["parent_ticks"] * law["category_total_variation"]
        _require(_exact(row.get("history_union_contribution")) == contribution,
                 "PARENT_HISTORY_CONTRIBUTION_CHANGED")
        union += contribution
    bound = min(Fraction(1), union)
    envelope = analysis.get("deterministic_parent_envelope", {})
    _require(_exact(envelope.get("unclamped_union")) == union
             and _exact(envelope.get("full_history_tv_bound")) == bound
             and envelope.get("parent_marginal_or_parent_independence_required") is False,
             "PARENT_FULL_HISTORY_ENVELOPE_CHANGED")
    processing = analysis.get("data_processing", {})
    _require(_exact(processing.get("observable_law_perturbation_tv_bound")) == bound
             and processing.get("coarse_fine_simulator_equality_implied") is False,
             "PARENT_DATA_PROCESSING_SCOPE_CHANGED")
    return p0, bound


def _validate_dimensions(config: dict, label: str) -> None:
    for key, expected in (("excitatory_neurons", 80000), ("inhibitory_neurons", 20000),
                          ("excitatory_preference_bins", 16)):
        _integer(config.get(key), expected, "SOURCE_GEOMETRY_CHANGED")
    _require(_number(config.get("dt_ms")) == Fraction.from_float(SOURCE_CLOCKS[label])
             and _number(config.get("duration_ms")) == 5000, "SOURCE_CLOCK_OR_DURATION_CHANGED")


def _validate_driver(driver: dict, seed: int, p0: Fraction) -> None:
    _require(isinstance(driver, dict), "COARSE_DRIVER_REQUIRED")
    for key, expected in (("binomial_n", _BINOMIAL_N), ("n_e", 80000), ("n_i", 20000),
                          ("total_ticks", 100000), ("ticks_generated", 100000)):
        _integer(driver.get(key), expected, "COARSE_DRIVER_DIMENSIONS_CHANGED")
    _require(_number(driver.get("binomial_probability")) == p0,
             "COARSE_BINARY64_PROBABILITY_CHANGED")
    _require(_number(driver.get("external_rate_Hz")) == Fraction.from_float(_RATE_HZ)
             and _number(driver.get("dt_seconds")) == Fraction.from_float(0.05 * 1e-3),
             "COARSE_RATE_OR_CLOCK_CHANGED")
    _require(driver.get("random_stream_policy") == "source_global"
             and driver.get("seeds") == {"source_rng_seed": seed}, "COARSE_SOURCE_IDENTITY_CHANGED")


def bind_sources(analysis_path: Path, source_index_path: Path, project_root: Path) -> dict:
    """Verify the complete indexed parameter scope; never load source arrays."""
    reader = _Reader()
    analysis_path, source_index_path, project_root = map(Path, (analysis_path, source_index_path, project_root))
    analysis = json.loads(reader.read(analysis_path))
    p0, one_tree = _validate_analysis(analysis)
    master_path = source_index_path.parent / "delivery.json"
    master = json.loads(reader.read(master_path))
    _require(master.get("verified") is True
             and master.get("status") == "COMPLETE_PFC_HISTORY_PROFILE_CENSUS_PREPARE",
             "ACCEPTED_MASTER_DELIVERY_REQUIRED")
    master_entries = _entries(master)
    _require(source_index_path.name in master_entries, "MASTER_SOURCE_INDEX_BINDING_REQUIRED")
    index = json.loads(reader.read(source_index_path, master_entries[source_index_path.name]))
    blocks = index.get("source_blocks")
    _integer(index.get("required_original_blocks"), 32, "REGISTERED_32_SOURCE_BLOCKS_REQUIRED")
    _require(isinstance(blocks, dict) and set(blocks) == {str(seed) for seed in range(95001, 95033)},
             "REGISTERED_SOURCE_BLOCK_IDENTITIES_CHANGED")
    expected_metadata = set()
    for seed, block in blocks.items():
        sources = block.get("sources", {})
        _require(set(sources) == set(SOURCE_CLOCKS), "ALL_FIVE_SOURCE_LABELS_REQUIRED")
        for label, source in sources.items():
            metadata = _relative(source["metadata"])
            _require(metadata == f"source_metadata/{seed}/{label}", "SOURCE_METADATA_IDENTITY_CHANGED")
            expected_metadata.update(metadata + "/" + name for name in ("config.yaml", "delivery.json", "report.json"))
    declared_metadata = {name for name in master_entries if name.startswith("source_metadata/")}
    _require(declared_metadata == expected_metadata, "COMPLETE_DECLARED_METADATA_FAMILY_REQUIRED")

    source_records, trees, representatives = [], [], {}
    coupled_archives = set()
    for seed_text in sorted(blocks, key=int):
        seed = int(seed_text)
        sources = blocks[seed_text]["sources"]
        checked = {}
        for label in SOURCE_CLOCKS:
            source = sources[label]
            archive = _relative(source["archive"])
            metadata = _relative(source["metadata"])
            snapshots = {}
            for name in ("config.yaml", "delivery.json", "report.json"):
                relative = metadata + "/" + name
                path = source_index_path.parent / relative
                data = reader.read(path, master_entries[relative])
                snapshots[name] = yaml.safe_load(data) if name == "config.yaml" else json.loads(data)
            original = snapshots["delivery.json"]
            original_digest = reader.digest(source_index_path.parent / metadata / "delivery.json")
            _require(original_digest == source["delivery_sha256"], "ORIGINAL_DELIVERY_DIGEST_CHANGED")
            original_entries = _entries(original)
            raw = source["raw"]
            raw_name = _relative(raw["path"])
            _require(raw_name in original_entries and original_entries[raw_name] == raw,
                     "DECLARED_RAW_IDENTITY_CHANGED")
            case_directory = Path(raw_name).parent
            original_report_name = (case_directory / "report.json").as_posix()
            copied_report = master_entries[metadata + "/report.json"]
            _require(original_report_name in original_entries
                     and (original_entries[original_report_name]["sha256"], original_entries[original_report_name]["bytes"])
                     == (copied_report["sha256"], copied_report["bytes"]),
                     "ORIGINAL_REPORT_PATH_BINDING_CHANGED")
            # Prepared configs may contain effective settings derived from the
            # original report. Their master digest and resolved_config equality
            # are required; an original direct config binding is additional.
            original_config_name = (case_directory / "config.yaml").as_posix()
            original_config_entry = original_entries.get(original_config_name)
            copied_config = master_entries[metadata + "/config.yaml"]
            direct_config_binding = (original_config_entry is not None
                and (original_config_entry["sha256"], original_config_entry["bytes"])
                == (copied_config["sha256"], copied_config["bytes"]))
            config, report = snapshots["config.yaml"], snapshots["report.json"]
            _require(isinstance(config, dict) and report.get("resolved_config") == config,
                     "SOURCE_CONFIG_REPORT_MISMATCH")
            _validate_dimensions(config, label)
            _require(_number(source.get("dt_ms")) == Fraction.from_float(SOURCE_CLOCKS[label]),
                     "INDEXED_SOURCE_CLOCK_CHANGED")
            _require(report.get("source_execution_complete") is True, "SOURCE_EXECUTION_INCOMPLETE")
            selected = report.get("selected_case", {})
            _integer(selected.get("source_rng_seed"), seed, "SOURCE_SEED_CHANGED")
            _require(selected.get("case_id") == f"pfc_{seed}_strong_cue", "SOURCE_CASE_CHANGED")
            ticks = round(5000 / SOURCE_CLOCKS[label])
            _integer(report.get("completed_duration_ticks"), ticks, "SOURCE_CONSUMED_TICKS_CHANGED")
            _integer(report.get("inputs_used", {}).get("ticks_consumed"), ticks, "SOURCE_CONSUMED_TICKS_CHANGED")
            for population, units in (("E", 80000), ("I", 20000)):
                _integer(report["inputs_used"].get(population, {}).get("values"), units * ticks,
                         "SOURCE_INPUT_VALUES_CHANGED")

            if label == "native":
                _require(original.get("status") == "ARCHIVED_PFC_SOURCE_EXECUTION"
                         and original.get("source_execution_complete") is True
                         and original.get("exit_code") == 0
                         and report.get("status") == "PFC_SOURCE_EXECUTION_COMPLETE",
                         "NATIVE_EXECUTION_RECEIPT_CHANGED")
                _require(source.get("refinement_seeds") is None and source.get("source_parent_dt_ms") is None,
                         "NATIVE_REFINEMENT_LABEL_CHANGED")
                _validate_driver(report.get("driver"), seed, p0)
                _require(report.get("code", {}).get("drive_sha256") == SUPPORTED_CODE_HASHES["pfc_native_drive"],
                         "UNSUPPORTED_NATIVE_DRIVER_CODE")
            else:
                allowed = {"COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT", "COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK"}
                if label.endswith("B"):
                    allowed = {"COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK"}
                _require(original.get("verified") is True and original.get("status") in allowed
                         and report.get("status") == "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE",
                         "COUPLED_EXECUTION_RECEIPT_CHANGED")
                _require(_number(source.get("source_parent_dt_ms")) == Fraction.from_float(0.05)
                         and _number(report.get("coarse_dt_ms")) == Fraction.from_float(0.05),
                         "SOURCE_PARENT_CLOCK_CHANGED")
                _integer(report.get("coupled_core_units"), 100000, "COUPLED_CORE_UNITS_CHANGED")
                seeds = source.get("refinement_seeds")
                _require(isinstance(seeds, list) and len(seeds) == 2
                         and all(type(value) is int for value in seeds) and len(set(seeds)) == 2
                         and seed not in seeds, "REFINEMENT_SOURCE_LABELS_REQUIRED")
                drive = report.get("drive", {})
                coupling = drive.get("coupling", {})
                _require(report.get("refinement_seeds") == seeds and coupling.get("refinement_seeds") == seeds,
                         "REFINEMENT_SOURCE_LABELS_CHANGED")
                factor = 2 if label.startswith("half") else 4
                for key, expected in (("refinement_factor", factor), ("total_ticks", ticks),
                                      ("ticks_generated", ticks), ("n_e", 80000), ("n_i", 20000)):
                    _integer(drive.get(key), expected, "COUPLED_DRIVER_CLOCK_OR_GEOMETRY_CHANGED")
                _require(drive.get("all_refinement_levels_generated") is True,
                         "COMPLETE_TWO_LEVEL_TREE_REQUIRED")
                _require(_number(drive.get("dt_seconds")) == Fraction.from_float(0.05 * 1e-3 / factor),
                         "COUPLED_DRIVER_CLOCK_CHANGED")
                coarse = coupling.get("coarse_source")
                _validate_driver(coarse, seed, p0)
                native = checked["native"]
                _require(coarse == native["report"]["driver"], "ORIGINAL_COARSE_DRIVER_BINDING_CHANGED")
                reference = report.get("coarse_reference", {})
                _require(reference.get("archive") == sources["native"]["archive"]
                         and reference.get("delivery_sha256") == sources["native"]["delivery_sha256"]
                         and reference.get("report_sha256") == native["report_sha256"],
                         "ORIGINAL_COARSE_REFERENCE_CHANGED")
                group = []
                for name, digest in SUPPORTED_CODE_HASHES.items():
                    original_name = f"Libs/population_dynamics/{name}.py"
                    _require(report.get("code", {}).get(name) == digest
                             and original_name in original_entries
                             and original_entries[original_name]["sha256"] == digest,
                             "UNSUPPORTED_SOURCE_CODE:" + name)
                    group.append(digest)
                representatives.setdefault(tuple(group), (archive, original_entries))
                coupled_archives.add(archive)
            report_sha = master_entries[metadata + "/report.json"]["sha256"]
            checked[label] = dict(source=source, report=report, report_sha256=report_sha)
            source_records.append(dict(source_block=seed_text, source_label=label, archive=archive,
                                       metadata=metadata, dt_ms=SOURCE_CLOCKS[label],
                                       delivery_sha256=original_digest, report_sha256=report_sha,
                                       prepared_config_sha256=copied_config["sha256"],
                                       prepared_config_binding="master_digest_and_authenticated_original_report_resolved_config",
                                       original_direct_config_binding_verified=direct_config_binding,
                                       original_config_manifest_entry=original_config_entry))
        for repeat in ("A", "B"):
            half, quarter = checked["half_" + repeat], checked["quarter_" + repeat]
            _require(half["source"]["archive"] == quarter["source"]["archive"]
                     and half["source"]["delivery_sha256"] == quarter["source"]["delivery_sha256"]
                     and half["source"]["refinement_seeds"] == quarter["source"]["refinement_seeds"]
                     and half["report"]["drive"]["coupling"]["coarse_source"]
                         == quarter["report"]["drive"]["coupling"]["coarse_source"],
                     "INCOMPATIBLE_HALF_QUARTER_TREE")
            trees.append(dict(source_block=seed_text, auxiliary_repeat=repeat,
                              members=["half_" + repeat, "quarter_" + repeat],
                              archive=half["source"]["archive"],
                              refinement_seeds=half["source"]["refinement_seeds"]))
        _require(checked["half_A"]["source"]["refinement_seeds"] != checked["half_B"]["source"]["refinement_seeds"],
                 "A_B_REFINEMENT_LABELS_MUST_REMAIN_DISTINCT")
    _require(len(trees) == 64 and len(coupled_archives) == 64,
             "REGISTERED_64_COUPLED_ARCHIVES_REQUIRED")
    code_bindings = []
    for group, (archive, entries) in representatives.items():
        files = []
        for name in SUPPORTED_CODE_HASHES:
            relative = f"Libs/population_dynamics/{name}.py"
            path = project_root / archive / relative
            reader.read(path, entries[relative])
            files.append(dict(path=str(path), sha256=reader.digest(path), bytes=entries[relative]["bytes"]))
        code_bindings.append(dict(representative_archive=archive, code_sha256=list(group), files=files))
    reader.read(Path(__file__))
    family = min(Fraction(1), len(trees) * one_tree)
    return {
        "status": "COMPLETE_INDEXED_PFC_REFINEMENT_PARAMETER_ARITHMETIC_BINDING",
        "parameter_arithmetic_binding_verified": True,
        "scope": "indexed_frozen_input_r_parameter_perturbation_under_ideal_conditional_sampling",
        "comparison_changes_only": analysis["comparison_changes_only"],
        "proof_uses_display_floats": False,
        "indexed_source_blocks": 32,
        "indexed_metadata_sources": len(source_records),
        "authenticated_source_metadata_files": len(expected_metadata),
        "coupled_source_labels_verified": len(source_records) - 32,
        "unique_coupled_source_archives": len(coupled_archives),
        "shared_two_level_trees": len(trees),
        "source_bindings": source_records,
        "trees": trees,
        "representative_frozen_code_bindings": code_bindings,
        "one_tree_tv_upper": _rational(one_tree),
        "per_block_two_A_B_tree_tv_upper": _rational(min(Fraction(1), 2 * one_tree)),
        "whole_indexed_family_tv_upper": _rational(family),
        "union_rule": "min(1, tree_count * one_tree_tv_upper); no independence between trees or source blocks is assumed",
        "ideal_conditional_sampler_assumptions": analysis["deterministic_parent_envelope"]["assumptions"],
        "data_processing": dict(analysis["data_processing"], indexed_family_observable_tv_upper=_rational(family)),
        "count_scope": "160 source metadata entries; no count of actual history reconstructions or benchmark development episodes",
        "statistical_independence_established": False,
        "seeds_used_to_infer_independence": False,
        "history_reconstructions_counted": False,
        "benchmark_development_episodes_counted": False,
        "full_source_runtime_qualified": False,
        "finite_sampler_law_certified": False,
        "finite_prng_law_certified": False,
        "source_history_confidence_certified": False,
        "confirmation_0p01_128_certified": False,
        "coarse_fine_neuronal_equality_certified": False,
        "scientific_fidelity_qualified": False,
        "production_qualified": False,
        "confirmation_execution_released": False,
        "confirmation_sampling_design_changed": False,
        "numerical_margins_changed": False,
        "error_budget_allocated": False,
        "protocol_amendment_adopted": False,
        "new_simulations": 0,
        "source_simulations": 0,
        "new_trials": 0,
        "new_episodes": 0,
        "production_episodes": 0,
        "provenance": {
            "parent_analysis_path": str(analysis_path),
            "parent_analysis_sha256": reader.digest(analysis_path),
            "source_index_path": str(source_index_path),
            "source_index_sha256": reader.digest(source_index_path),
            "accepted_master_delivery_path": str(master_path),
            "accepted_master_delivery_sha256": reader.digest(master_path),
            "implementation_sha256": reader.digest(Path(__file__)),
            "explicit_paths_read": len(reader.snapshots),
            "bytes_read": reader.total_bytes,
            "consumed_artifacts": reader.records(),
        },
    }


def run(analysis: Path, source_index: Path, project_root: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"PFC_REFINEMENT_SOURCE_BINDING_FRESH_OUTPUT_REQUIRED:{output}")
    report = bind_sources(analysis, source_index, project_root)
    freeze_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.analysis, args.source_index, args.project_root, args.output)
    print(json.dumps({key: report[key] for key in
                      ("status", "parameter_arithmetic_binding_verified", "shared_two_level_trees",
                       "whole_indexed_family_tv_upper")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic integrity tests; execute on Merlin, not the management node."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from . import population_readout_result_summary as reader


NATIVE_METRICS = ["native_" + name for name in reader.NATIVE_NAMES]


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _values(model: str, seed: int, block: int, template: int, origin: int) -> dict:
    offset = reader.MODELS.index(model) / 8
    a, b = 0.25 + offset, 0.125
    values = dict.fromkeys(reader.MOMENT_NAMES, 0.0)
    values.update(
        standardized_latent_mean_mse=1.0 + offset,
        standardized_latent_ensemble_variance_mean=2.0 + offset,
        standardized_latent_lag_one_covariance_mean=0.5 + offset,
        predicted_mean_log_hazard_unit_mean=2.0 + a,
        posterior_mean_log_hazard_unit_mean=2.0,
        empirical_log_expected_hazard_unit_mean=2.0 + a + 0.25,
        posterior_log_expected_hazard_unit_mean=2.125,
        empirical_log_hazard_variance_unit_mean=0.5,
        empirical_jensen_gap_unit_mean=0.25,
        posterior_lognormal_adjustment_unit_mean=0.125,
        mean_displacement_unit_mean=a,
        mean_displacement_unit_mean_square=a * a,
        stochastic_gap_displacement_unit_mean=b,
        stochastic_gap_displacement_unit_mean_square=b * b,
        total_log_expected_hazard_gap_unit_mean=a + b,
        total_log_expected_hazard_gap_unit_mean_square=(a + b) ** 2,
        decomposition_cross_term_unit_mean=2 * a * b,
    )
    return values


def _aggregates(origins: list[dict], metrics: list[str], *, by_block: bool) -> list[dict]:
    groups = {}
    for row in origins:
        key = (row["model"], row["seed"], row["horizon_ms"])
        if by_block:
            key += (row["block"],)
        groups.setdefault(key, []).append(row)
    rows = []
    for key, selected in groups.items():
        identity = dict(model=key[0], seed=key[1], horizon_ms=key[2])
        if by_block:
            identity["block"] = key[3]
        for metric in metrics:
            values = [row[metric] for row in selected if row[metric] is not None]
            maximum = metric.endswith("decomposition_closure_max_absolute")
            value = (max(values) if maximum else sum(values) / len(values)) if values else None
            rows.append(dict(identity, metric=metric, statistic="maximum" if maximum else "mean", value=value,
                             available_origins=len(values), structural_na_origins=len(selected) - len(values),
                             total_origins=len(selected), metric_scope=(
                                 "recorded_native_horizon_metric" if metric.startswith("native_") else
                                 "instantaneous_endpoint" if metric.startswith("endpoint_") else "prefix_moment_summary")))
    return rows


def _fixture(directory: Path) -> None:
    digest = "1" * 64
    consumed, deliveries = {}, []
    times, origins = [], []
    metrics = reader._moment_columns() + NATIVE_METRICS
    for model, seed in itertools.product(reader.MODELS, reader.SEEDS):
        prefix = "forecasts/" if model == "original" else f"results/{model}/forecasts/"
        files = {prefix + f"train_block_{block:03d}_familiar_{template}_origin_{origin}.npz": digest
                 for block, template, origin in itertools.product(reader.BLOCKS, reader.TEMPLATES, reader.ORIGINS_MS)}
        if model == "original":
            files["training/best.pt"] = digest
            files.update({f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt": digest
                          for block, template in itertools.product(reader.BLOCKS, reader.TEMPLATES)})
        else:
            files["provenance/fit.json"] = digest
        consumed[f"{seed}/{model}"] = files
        deliveries.append(dict(seed=seed, model=model, delivery_sha256=digest, parent_checkpoint_sha256=digest))
        for block, template, origin in itertools.product(reader.BLOCKS, reader.TEMPLATES, reader.ORIGINS_MS):
            identity = dict(model=model, seed=seed, block=block, case_id=f"train_block_{block:03d}_familiar_{template}",
                            template_id=f"familiar_{template}", origin_ms=origin)
            values = _values(model, seed, block, template, origin)
            for step in range(1, reader.FUTURE_BINS + 1):
                times.append(dict(identity, elapsed_ms=step * reader.BIN_MS,
                                  absolute_bin_end_ms=origin + step * reader.BIN_MS, **values))
            for horizon in reader.HORIZONS_MS:
                row = dict(identity, horizon_ms=horizon, prefix_bins=horizon // reader.BIN_MS,
                           native_method=model + "_method", native_hardware_device="fixture_cpu",
                           native_ensemble_artifact=f"train_block_{block:03d}_familiar_{template}_origin_{origin}.npz")
                for name, value in values.items():
                    prefix_name = "prefix_max_" if name == "decomposition_closure_max_absolute" else "prefix_mean_"
                    row[prefix_name + name] = value
                    row["endpoint_" + name] = value
                for index, metric in enumerate(NATIVE_METRICS):
                    row[metric] = (None if metric in {"native_peak_gpu_allocated_bytes", "native_unit_total_interval_width_90"}
                                   else 1.0 + reader.MODELS.index(model) / 8 + block / 16)
                # A real missing-availability pattern tests matched-pair denominators.
                if template == 0:
                    row["native_energy_score"] = None
                origins.append(row)
    _write_rows(directory / "time_moments.csv", times)
    _write_rows(directory / "origin_horizon_moments.csv", origins)
    _write_rows(directory / "seed_horizon_summary.csv", _aggregates(origins, metrics, by_block=False))
    _write_rows(directory / "source_block_summary.csv", _aggregates(origins, metrics, by_block=True))
    analysis = dict(status="COMPLETE_POSTHOC_NATIVE_READOUT_DIAGNOSTIC", training_seeds=list(reader.SEEDS),
                    models=list(reader.MODELS), source_blocks=list(reader.BLOCKS), horizon_ms=list(reader.HORIZONS_MS),
                    origins_per_model_seed=48, retained_draws=16, future_bins=100, source_units=100000,
                    coordinate_scale_fitting_blocks=list(range(12)), native_comparison_reused=True,
                    consumed_files=consumed, consumed_deliveries=deliveries, input_manifest_sha256=digest,
                    runtime=dict(execution_backend="merlin", hostname="fixture_merlin",
                                 environment={"RH2_JOB_RUN_ID": "fixture_run", "ARNOLD_TRIAL_ID": "fixture_trial",
                                              "RH2_CHECKPOINT_HDFS": "fixture_checkpoint"}),
                    q_reference="Cached q is descriptive, not physical latent truth.",
                    hazard_interpretation="Hazards do not decompose refractory count errors.",
                    native_metric_windows="Native metrics retain their recorded horizon definitions.")
    for name in ("trajectory_sampling_performed", "parameter_fitting_performed", "native_generation_performed",
                 "new_encoding_performed", "qualification_or_native_scores_recomputed", "significance_tests_computed",
                 "scientific_claim_qualified", "sota_qualified"):
        analysis[name] = False
    analysis["artifacts"] = {name: dict(rows=count, sha256=reader._sha256(directory / name))
                             for name, count in reader.ARTIFACT_ROWS.items()}
    (directory / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")


def _mutate(directory: Path, name: str, transform) -> None:
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    transform(rows)
    _write_rows(directory / name, rows)
    path = directory / "analysis.json"
    analysis = json.loads(path.read_text())
    analysis["artifacts"][name]["sha256"] = reader._sha256(directory / name)
    path.write_text(json.dumps(analysis), encoding="utf-8")


class ReadoutResultSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_path = Path(cls.fixture.name)
        _fixture(cls.fixture_path)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        for name in (*reader.ARTIFACT_ROWS, "analysis.json"):
            (self.path / name).write_bytes((self.fixture_path / name).read_bytes())

    def tearDown(self):
        self.directory.cleanup()

    def test_complete_ingestion_preserves_na_and_paired_denominators(self):
        result = reader.summarize(self.path)
        self.assertEqual(len(result["seed_horizon"]["rows"]), 27)
        self.assertEqual(len(result["block_500ms"]["rows"]), 36)
        self.assertEqual(len(result["paired_seed_horizon_deltas"]["rows"]), 27)
        self.assertEqual(len(result["paired_block_500ms_deltas"]["rows"]), 36)
        self.assertEqual(result["integrity"]["aggregate_values_crosschecked"], 7560)
        width_index = result["metrics"].index("native_unit_total_interval_width_90")
        self.assertIsNone(result["seed_horizon"]["rows"][0][3 + width_index])
        energy_index = result["metrics"].index("native_energy_score")
        self.assertIn([0, energy_index, 36], result["seed_horizon"]["partial_availability"])
        self.assertIn([0, energy_index, 36], result["paired_seed_horizon_deltas"]["partial_availability"])
        self.assertEqual(result["paired_seed_horizon_deltas"]["rows"][0][3], 0.125)
        self.assertLess(len(json.dumps(result, separators=(",", ":"))), 60000)

    def test_corrupted_artifact_is_rejected_before_ingestion(self):
        with (self.path / "time_moments.csv").open("a") as stream:
            stream.write("corruption\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            reader.summarize(self.path)

    def test_duplicate_time_identity_is_rejected_even_with_matching_hash(self):
        _mutate(self.path, "time_moments.csv", lambda rows: rows.__setitem__(-1, rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate retained time bin"):
            reader.summarize(self.path)
        valid = dict(model="original", seed="41", block="14",
                     case_id="train_block_014_familiar_0", template_id="familiar_0", origin_ms="1000")
        self.assertEqual(reader._identity(valid), ("original", 41, 14, 0, 1000))
        for template in ("0", "familiar_00", "familiar_4", "familiar_-1", "Familiar_0"):
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "canonical familiar_N"):
                reader._identity(dict(valid, template_id=template))

    def test_false_available_denominator_is_rejected(self):
        _mutate(self.path, "seed_horizon_summary.csv",
                lambda rows: rows[0].__setitem__("available_origins", "47"))
        with self.assertRaisesRegex(ValueError, "denominator differs"):
            reader.summarize(self.path)

    def test_mean_square_nonclosure_is_rejected(self):
        _mutate(self.path, "time_moments.csv", lambda rows: rows[0].__setitem__(
            "total_log_expected_hazard_gap_unit_mean_square", "0.5"))
        with self.assertRaisesRegex(ValueError, "mean-square closure"):
            reader.summarize(self.path)

    def test_changed_aggregate_is_rejected_against_origin_values(self):
        _mutate(self.path, "seed_horizon_summary.csv", lambda rows: rows[0].__setitem__("value", "3.0"))
        with self.assertRaisesRegex(ValueError, "numerical mismatch: seed_horizon_summary.csv"):
            reader.summarize(self.path)


if __name__ == "__main__":
    unittest.main()

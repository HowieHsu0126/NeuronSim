"""Checks for complete causal boundaries and archived witness identities."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from . import population_boundary_closure as closure
from .population_state import ControlledPopulationState


def tiny_model(representation="recurrent"):
    return ControlledPopulationState(
        unit_ids=[13, 2, 41, 7, 19, 3, 29],
        unit_field_index=[0, 1, 0, 1, 0, 1, 0],
        field_names=["source_E", "source_I", "empty"],
        minimum_interval_ticks=[3, 4, 5, 3, 4, 5, 3],
        tick_seconds=.001, bin_seconds=.005, input_size=1,
        latent_channels=2, encoder_size=5, representation=representation,
        initialization_seed=41, unit_chunk_size=3,
    ).double().eval()


class TinyNative:
    """Observed prefixes with the same interface as the native episode loader."""

    def __init__(self, model):
        self.unit_ids = np.asarray(model.unit_ids)
        self.field_ids = np.asarray(model.field_names)
        self.unit_field_index = model.unit_field_index.cpu().numpy().copy()
        self.minimum_interval_ticks = model.minimum_interval_ticks.cpu().numpy().copy()
        self.dt_ms, self.bin_ms, self.bin_ticks = 1., 5., 5
        self.duration_ticks = 60
        self.counts = np.zeros((12, len(self.unit_ids)), dtype=np.int64)
        self.counts[1::3, ::2] = 1
        self.stimulus = np.linspace(0., 1., 12)[:, None]
        self.window_calls, self.ready_calls = [], []

    def window(self, start, stop):
        self.window_calls.append((start, stop))
        if start != 0 or stop % self.bin_ticks:
            raise ValueError("aligned complete prefix required")
        return self.counts[:stop // self.bin_ticks].copy(), self.stimulus[:stop // self.bin_ticks].copy()

    def ready_at(self, tick):
        self.ready_calls.append(tick)
        ready = np.zeros(len(self.unit_ids), dtype=np.int64)
        prefix = self.counts[:tick // self.bin_ticks]
        for unit in range(len(self.unit_ids)):
            emitted = np.flatnonzero(prefix[:, unit])
            if len(emitted):
                last_tick = (int(emitted[-1]) + 1) * self.bin_ticks - 1
                ready[unit] = max(0, last_tick + self.minimum_interval_ticks[unit] - tick)
        return ready


class CompleteBoundaryTests(unittest.TestCase):
    representation = "recurrent"

    def setUp(self):
        self.model = tiny_model(self.representation)
        self.native = TinyNative(self.model)
        self.state = self.model.infer_state(dict(
            counts=self.native.counts[None, :4], stimulus=self.native.stimulus[None, :4],
            ready_in_ticks=np.array([[2, 1, 4, 0, 3, 2, 1]], dtype=np.int64),
        ))

    def test_vector_keeps_all_source_ordered_clocks_and_both_moments(self):
        vector = closure.boundary_vector(self.state, self.model)
        latent_width = len(self.model.field_names) * self.model.latent_channels
        self.assertEqual(vector.shape, (2 * latent_width + 7,))
        self.assertEqual(vector.dtype, np.float64)
        np.testing.assert_array_equal(vector[:latent_width], self.state.mean.detach().numpy().ravel())
        np.testing.assert_array_equal(vector[latent_width:2 * latent_width], self.state.variance.detach().numpy().ravel())
        np.testing.assert_array_equal(vector[2 * latent_width:], np.array([2 / 3, 1 / 4, 4 / 5, 0., 3 / 4, 2 / 5, 1 / 3]))
        altered = replace(self.state, ready_in_ticks=self.state.ready_in_ticks.clone())
        altered.ready_in_ticks[0, 6] = 2
        changed = closure.boundary_vector(altered, self.model)
        self.assertEqual(np.flatnonzero(vector != changed).tolist(), [len(vector) - 1])

    def test_malformed_intervals_and_source_identities_are_rejected(self):
        for invalid in (
            torch.tensor([0, 4, 5, 3, 4, 5, 3]),
            torch.tensor([-1, 4, 5, 3, 4, 5, 3]),
            torch.tensor([3., 4., 5., 3., 4., 5., 3.]),
            torch.tensor([float("nan"), 4., 5., 3., 4., 5., 3.]),
            torch.tensor([[3, 4, 5, 3, 4, 5, 3]]),
        ):
            with self.subTest(interval=invalid):
                model = tiny_model(self.representation)
                model.minimum_interval_ticks = invalid
                with self.assertRaises(ValueError):
                    closure.boundary_vector(self.state, model)
        for changed in (
            replace(self.state, unit_ids=tuple(reversed(self.state.unit_ids))),
            replace(self.state, field_names=("wrong", "source_I", "empty")),
            replace(self.state, bin_seconds=.01),
            replace(self.state, variance=torch.zeros_like(self.state.variance)),
            replace(self.state, ready_in_ticks=torch.zeros((1, 7), dtype=torch.float64)),
        ):
            with self.subTest(boundary=changed):
                with self.assertRaises(ValueError):
                    closure.boundary_vector(changed, self.model)

    def test_extraction_reads_one_window_preserves_order_and_does_not_use_rng(self):
        origins = [50., 20., 35.]
        counts_before, stimulus_before = self.native.counts.copy(), self.native.stimulus.copy()
        units_before, fields_before = self.native.unit_ids.copy(), self.native.field_ids.copy()
        mapping_before = self.native.unit_field_index.copy()
        rng_before = torch.get_rng_state().clone()
        vectors, timings = closure.extract_boundaries(self.model, self.native, origins)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(self.native.window_calls, [(0, 50)])
        self.assertEqual(self.native.ready_calls, [50, 20, 35])
        self.assertEqual(len(vectors), len(timings))
        self.assertTrue(all(np.isfinite(t) and t >= 0 for t in timings))
        np.testing.assert_array_equal(self.native.counts, counts_before)
        np.testing.assert_array_equal(self.native.stimulus, stimulus_before)
        np.testing.assert_array_equal(self.native.unit_ids, units_before)
        np.testing.assert_array_equal(self.native.field_ids, fields_before)
        np.testing.assert_array_equal(self.native.unit_field_index, mapping_before)
        latent_width = len(self.model.field_names) * self.model.latent_channels
        for origin, vector in zip(origins, vectors):
            bins = int(origin / self.native.bin_ms)
            readiness = self.native.ready_at(int(origin))
            state = self.model.infer_state(dict(
                counts=self.native.counts[None, :bins], stimulus=self.native.stimulus[None, :bins],
                ready_in_ticks=readiness[None],
            ))
            self.assertEqual(state.unit_ids, tuple(map(int, units_before)))
            self.assertEqual(state.field_names, tuple(map(str, fields_before)))
            self.assertEqual(vector.shape, (2 * latent_width + len(units_before),))
            np.testing.assert_array_equal(vector[:latent_width], state.mean.detach().numpy().ravel())
            np.testing.assert_array_equal(vector[latent_width:2 * latent_width], state.variance.detach().numpy().ravel())
            np.testing.assert_array_equal(vector[2 * latent_width:], readiness / self.native.minimum_interval_ticks)
            np.testing.assert_array_equal(vector, closure.boundary_vector(state, self.model))

    def test_changes_after_boundary_cannot_change_its_vector(self):
        before, _ = closure.extract_boundaries(self.model, self.native, [20., 50.])
        changed = TinyNative(self.model)
        changed.counts[4:] += 3
        changed.stimulus[4:] += 10
        after, _ = closure.extract_boundaries(self.model, changed, [20., 50.])
        np.testing.assert_array_equal(before[0], after[0])
        self.assertFalse(np.array_equal(before[1], after[1]))

    def test_extraction_rejects_clock_and_source_map_mismatch(self):
        for origins in ([20., 20.], [], [20.5], [0.], [65.]):
            with self.subTest(origins=origins):
                with self.assertRaises(ValueError):
                    closure.extract_boundaries(self.model, self.native, origins)
        for field, value in (
            ("unit_ids", self.native.unit_ids[::-1]),
            ("unit_field_index", 1 - self.native.unit_field_index),
            ("minimum_interval_ticks", self.native.minimum_interval_ticks + 1),
            ("dt_ms", 2.),
        ):
            with self.subTest(native_field=field):
                native = TinyNative(self.model)
                setattr(native, field, value)
                with self.assertRaises(ValueError):
                    closure.extract_boundaries(self.model, native, [20.])


class FixedCompleteBoundaryTests(CompleteBoundaryTests):
    """Apply the same causal/source/vector contracts to the fixed encoder."""

    representation = "fixed"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class HistoryFixture:
    """Small source-bound metadata with complete required witness dimensions."""

    def __init__(self, root):
        self.root, self.acquisition, self.fields = root / "history", root / "acquisition", root / "fields"
        (self.root / "features").mkdir(parents=True)
        self.acquisition.mkdir()
        self.fields.mkdir()
        self.config = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_boundary_closure.yaml").read_text())
        history_config = dict(
            schema="neural_population_field_history_probe_v1", evidence_role="exploratory_development",
            source_split="train", formal_training_seeds=[41, 42, 43], training_rng=None,
            fit_blocks=list(range(14)), selection_blocks=list(range(14, 18)),
            representations=["recent", "recent_and_history"], predictors=["linear", "rbf"],
            future_statistics=["mean_rate", "spatial_variogram", "cross_population_variogram", "temporal_variogram"],
            variogram_power=1., regularization=[.01, .1, 1., 10., 100.],
            recent_ms=20, history_ms=1000, history_bin_ms=50, temporal_lag_ms=5,
            templates=[f"familiar_{i}" for i in range(4)], origins_ms=[1000, 3000, 5000], horizons_ms=[50, 500],
        )
        rows, cases, bindings = [], [], []
        for block in range(18):
            for template in range(4):
                identity = f"train_block_{block:03d}_familiar_{template}"
                cases.append(dict(case_id=identity, block=block, template_id=f"familiar_{template}", split="train",
                                  settings=dict(pulse_control={"template": template})))
                bindings.append(dict(case_id=identity, field_summary_sha256="source-summary", spatial_fields_sha256="source-fields"))
                for origin_index, origin in enumerate(history_config["origins_ms"]):
                    rows.append(dict(case_id=identity, block=block, template_id=f"familiar_{template}",
                                     origin_ms=origin, cell=3 * template + origin_index))
        write_json(self.acquisition / "design.json", dict(cases=cases))
        write_json(self.fields / "dataset.json", dict(status="BUILT", split="train"))
        definitions = [[population, y, x] for population in range(8) for y in range(4) for x in range(4)]
        pairs = closure.field_pairs([v[0] for v in definitions], [v[2] for v in definitions], [v[1] for v in definitions])
        recent = np.zeros((216, 128))
        self.arrays = dict(
            recent=recent, recent_and_history=np.concatenate((recent, np.zeros((216, 2560))), axis=1),
            targets=np.zeros((216, 1920)), blocks=np.asarray([r["block"] for r in rows]),
            cells=np.asarray([r["cell"] for r in rows]), field_ids=np.asarray([json.dumps(v) for v in definitions]),
            spatial_pairs=pairs["spatial"], cross_population_pairs=pairs["cross_population"],
        )
        slices, offset = [], 0
        for horizon in (50, 500):
            for family, width in (("mean_rate", 128), ("spatial_variogram", 256),
                                  ("cross_population_variogram", 448), ("temporal_variogram", 128)):
                slices.append(dict(family=family, horizon_ms=horizon, start=offset, stop=offset + width,
                                   statistic_units="Hz", mse_units="Hz^2"))
                offset += width
        self.report = dict(
            status="BUILT_DEVELOPMENT", config=history_config, rows=rows, source_bindings=bindings,
            source_design_sha256=digest(self.acquisition / "design.json"),
            source_dataset_sha256=digest(self.fields / "dataset.json"),
            unit_count=100000, field_count=128, source_bin_ms=5, target_slices=slices,
            calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        )
        self.seal()

    def seal(self):
        arrays_path = self.root / "features/features.npz"
        report_path = self.root / "features/features.json"
        np.savez_compressed(arrays_path, **self.arrays)
        self.report["features_sha256"] = digest(arrays_path)
        write_json(report_path, self.report)
        files = [dict(path=str(path.relative_to(self.root)), bytes=path.stat().st_size, sha256=digest(path))
                 for path in (report_path, arrays_path)]
        manifest = self.root / "payload_manifest.json"
        write_json(manifest, dict(files=files))
        write_json(self.root / "delivery.json", dict(
            status="ARCHIVED_INDEPENDENTLY_REPLAYED_DEVELOPMENT_PROBE",
            calibration_outcomes_accessed=False, test_outcomes_accessed=False,
            payload_manifest_sha256=digest(manifest),
        ))

    def load(self):
        return closure.load_history(self.root, self.acquisition, self.fields, self.config)


class ArchivedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = HistoryFixture(Path(self.temporary.name))

    def test_accepts_complete_source_bound_rows_and_all_targets(self):
        report, arrays, names = self.fixture.load()
        self.assertEqual(len(report["rows"]), 216)
        self.assertEqual(int(np.isin(arrays["blocks"], range(14)).sum()), 168)
        self.assertEqual(int(np.isin(arrays["blocks"], range(14, 18)).sum()), 48)
        self.assertEqual(len(names), 1920)
        self.assertEqual(len(set(names)), 1920)
        self.assertEqual(names[0], "mean_rate_50ms_0")
        self.assertEqual(names[-1], "temporal_variogram_500ms_127")
        self.assertEqual(arrays["spatial_pairs"].shape, (256, 2))
        self.assertEqual(arrays["cross_population_pairs"].shape, (448, 2))

    def test_rejects_row_case_and_cell_identity_mismatch(self):
        self.fixture.report["rows"][0]["case_id"] = "train_block_001_familiar_0"
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "case/cell"):
            self.fixture.load()

    def test_rejects_missing_selection_coverage_and_source_binding(self):
        self.fixture.report["rows"][-1] = self.fixture.report["rows"][-2].copy()
        self.fixture.arrays["blocks"][-1] = self.fixture.arrays["blocks"][-2]
        self.fixture.arrays["cells"][-1] = self.fixture.arrays["cells"][-2]
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "216"):
            self.fixture.load()

    def test_rejects_duplicate_episode_binding(self):
        self.fixture.report["source_bindings"][-1] = self.fixture.report["source_bindings"][-2].copy()
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "72 unique"):
            self.fixture.load()

    def test_rejects_array_identity_and_recent_prefix_mismatch(self):
        self.fixture.arrays["cells"][0] = 1
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "216"):
            self.fixture.load()

    def test_rejects_wrong_control_cell_even_when_row_and_array_agree(self):
        self.fixture.report["rows"][0]["cell"] = 1
        self.fixture.arrays["cells"][0] = 1
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "case/cell"):
            self.fixture.load()

    def test_witness_reuses_every_archived_history_column_and_source_block(self):
        report, arrays, names = self.fixture.load()
        states = np.arange(216 * 19, dtype=np.float64).reshape(216, 19)
        with patch.object(closure, "assess_state", return_value={"status": "captured"}) as assess:
            result = closure.witness_from_boundaries(states, report, arrays, names, self.fixture.config)
        self.assertEqual(result["status"], "captured")
        arguments = assess.call_args.kwargs
        self.assertEqual(arguments["fit_state"].shape, (168, 19))
        self.assertEqual(arguments["selection_state"].shape, (48, 19))
        self.assertEqual(arguments["fit_history"].shape, (168, 2560))
        self.assertEqual(arguments["selection_history"].shape, (48, 2560))
        self.assertEqual(arguments["selection_targets"].shape, (48, 1920))
        np.testing.assert_array_equal(arguments["selection_blocks"], np.repeat([14, 15, 16, 17], 12))
        np.testing.assert_array_equal(arguments["selection_controls"], np.eye(12)[arguments["selection_cells"]])
        self.assertEqual(arguments["target_names"], names)
        self.assertEqual(arguments["penalty"], .01)
        self.assertEqual(self.fixture.config["training_seeds"], [41, 42, 43])
        self.fixture.arrays["cells"][0] = 0
        self.fixture.arrays["recent_and_history"][0, 0] = 1
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "216"):
            self.fixture.load()

    def test_rejects_changed_pair_and_target_units(self):
        self.fixture.arrays["spatial_pairs"][0] = [0, 0]
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "target pairs"):
            self.fixture.load()

    def test_rejects_target_semantics_even_when_dimensions_match(self):
        self.fixture.report["target_slices"][0]["horizon_ms"] = 500
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "target slices"):
            self.fixture.load()
        self.fixture.report["target_slices"][0]["horizon_ms"] = 50
        self.fixture.report["target_slices"][0]["statistic_units"] = "counts"
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "target slices"):
            self.fixture.load()

    def test_rejects_nonfinite_targets_and_changed_payload(self):
        self.fixture.arrays["targets"][0, 0] = np.nan
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "216"):
            self.fixture.load()
        self.fixture.arrays["targets"][0, 0] = 0
        self.fixture.seal()
        path = self.fixture.root / "features/features.npz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "features changed"):
            self.fixture.load()

    def test_rejects_outcome_access_and_formal_selection(self):
        self.fixture.report["test_outcomes_accessed"] = True
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "historical feature protocol"):
            self.fixture.load()
        config = self.fixture.config.copy()
        config["formal_tolerances"] = {"mean_rate": 0.}
        with self.assertRaisesRegex(ValueError, "development witness"):
            closure.validate_config(config)


class CachedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.fixture = HistoryFixture(self.base)
        # Scaled engineering fixture; production admission stays unchanged.
        self.fixture.config["expected_units"] = 7
        self.fixture.report["unit_count"] = 7
        self.fixture.arrays["targets"] = np.random.default_rng(41).normal(size=(216, 1920))
        self.fixture.seal()
        self.report, self.arrays, self.names = self.fixture.load()
        self.root = self.base / "boundary_run"
        self.root.mkdir()
        self.config_path = self.base / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(self.fixture.config))
        for seed in (41, 42, 43):
            cell = self.root / f"analysis/seed_{seed}"
            cell.mkdir(parents=True)
            states = np.random.default_rng(seed).normal(size=(216, 263))
            np.savez_compressed(cell / "boundaries.npz", boundaries=states,
                unit_ids=np.arange(7), field_ids=self.arrays["field_ids"],
                blocks=self.arrays["blocks"], cells=self.arrays["cells"],
                case_ids=np.asarray([row["case_id"] for row in self.report["rows"]]),
                origins_ms=np.asarray([row["origin_ms"] for row in self.report["rows"]]))
            witness = closure.witness_from_boundaries(states, self.report, self.arrays, self.names, self.fixture.config)
            witness.update(configuration=self.fixture.config, training_seed=seed,
                target_slices=self.report["target_slices"], boundary_dimensions=263,
                boundary_artifact_sha256=digest(cell / "boundaries.npz"),
                source_binding=dict(seed=seed, config=dict(latent_channels=1),
                    acquisition_design_sha256=self.report["source_design_sha256"],
                    dataset_sha256=self.report["source_dataset_sha256"]),
                selection_enabled=False, scientific_claim_qualified=False, checkpoint_sha256=f"checkpoint_{seed}")
            write_json(cell / "witness.json", witness)
        write_json(self.root / "analysis/analysis.json", dict(config=self.fixture.config,
            history_delivery_sha256=digest(self.fixture.root / "delivery.json"),
            history_feature_report_sha256=digest(self.fixture.root / "features/features.json")))
        self.seal()

    def seal(self):
        paths = [self.root / "analysis/analysis.json"]
        for seed in (41, 42, 43):
            paths.extend(self.root / f"analysis/seed_{seed}/{name}" for name in ("boundaries.npz", "witness.json"))
        self.index = {str(p.relative_to(self.root)):dict(path=str(p.relative_to(self.root)), bytes=p.stat().st_size,
                                                      sha256=digest(p)) for p in paths}
        write_json(self.root / "delivery.json", dict(status="PASS", exit_code=0, mode="inference_only",
            training_seeds=[41, 42, 43], calibration_or_test_accessed=False, files=list(self.index.values())))

    def load(self):
        with patch.object(closure, "validate_config"):
            return closure.load_cached_boundary(self.root, self.index, 41, self.report,
                                                self.arrays, self.names, self.fixture.config)

    def test_cached_run_reuses_linear_and_never_opens_neural_or_native_inputs(self):
        from . import closure_selection
        output = self.base / "output"
        with patch.object(closure, "validate_config"), patch.object(closure, "_load_model", side_effect=AssertionError("checkpoint opened")), patch.object(closure, "NativeRenewalEpisode", side_effect=AssertionError("native opened")), patch.object(closure, "extract_boundaries", side_effect=AssertionError("extraction repeated")), patch.object(closure_selection, "conditional_prediction", wraps=closure_selection.conditional_prediction) as prediction:
            result = closure.run_cached(self.fixture.acquisition, self.fixture.fields, self.fixture.root,
                                        self.root, self.config_path, output)
        self.assertEqual(prediction.call_count, 6)
        self.assertTrue(all(call.args[-2] == "rbf" for call in prediction.call_args_list))
        self.assertEqual(result["family_block_rows"], 96)
        self.assertFalse(result["linear_regression_repeated"])
        self.assertFalse(result["state_selection_enabled"])
        for seed in (41, 42, 43):
            report = json.loads((output / f"seed_{seed}/witness.json").read_text())
            self.assertEqual(len(report["target_names"]), 1920)
            self.assertEqual([row["block"] for row in report["source_block_rows"]], ["14", "15", "16", "17"])
            self.assertTrue(all(row["rows"] == 12 for row in report["source_block_rows"]))

    def test_changed_bytes_are_rejected_before_readout(self):
        with (self.root / "analysis/seed_41/boundaries.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.load()

    def test_resealed_row_mismatch_is_rejected(self):
        path = self.root / "analysis/seed_41/boundaries.npz"
        with np.load(path, allow_pickle=False) as data:
            values = {key:data[key] for key in data.files}
        values["case_ids"] = values["case_ids"][::-1]
        np.savez_compressed(path, **values)
        wp = path.with_name("witness.json")
        witness = json.loads(wp.read_text())
        witness["boundary_artifact_sha256"] = digest(path)
        write_json(wp, witness)
        self.seal()
        with self.assertRaisesRegex(ValueError, "row identities"):
            self.load()

    def test_comparison_rejects_target_scale_or_block_mismatch(self):
        _, _, old = self.load()
        changed = dict(old, training_target_scale=[2.] * 1920)
        with self.assertRaisesRegex(ValueError, "normalization"):
            closure.compare_readout_families(41, old, changed, self.report["target_slices"])
        changed = dict(old, source_block_rows=old["source_block_rows"][:-1])
        with self.assertRaisesRegex(ValueError, "selection blocks"):
            closure.compare_readout_families(41, old, changed, self.report["target_slices"])


if __name__ == "__main__":
    unittest.main()

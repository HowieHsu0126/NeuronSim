"""Native ancestral sampling, fixed-network assignment and fitting isolation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from .lfads import LFADSConditionalSimulator, LFADSRandomStreams
from .lfads_prior_refit import (
    PRIOR_VARIABLES, apply_prior_parameters, fitting_cases,
    sample_complete_posterior, sampling_seeds,
)
from . import lfads_prior_refit as refit


def cohort_fixture():
    cases = [dict(case_id=f"train_block_{block:03d}_familiar_{template}",
                  block=block, template_id=f"familiar_{template}")
             for block in range(18) for template in range(4)]
    return SimpleNamespace(fit=cases[:56], validation=cases[56:])


class FittingIsolationTests(unittest.TestCase):
    def test_complete_fitting_cohort_excludes_every_selection_episode(self):
        cohort = cohort_fixture()
        cohort.fit.reverse()
        cases = fitting_cases(cohort)
        self.assertEqual(len(cases), 56)
        self.assertEqual(cases[0]["case_id"], "train_block_000_familiar_0")
        self.assertEqual(cases[-1]["case_id"], "train_block_013_familiar_3")
        self.assertFalse({c["case_id"] for c in cases} & {c["case_id"] for c in cohort.validation})

    def test_selection_substitution_duplicate_and_false_block_are_rejected(self):
        for change in ("selection", "duplicate", "false_block"):
            with self.subTest(change=change):
                cohort = cohort_fixture()
                if change == "selection":
                    cohort.fit[0] = cohort.validation[0]
                elif change == "duplicate":
                    cohort.fit[0] = cohort.fit[1]
                else:
                    cohort.fit[0]["block"] = 14
                with self.assertRaises(ValueError):
                    fitting_cases(cohort)

    def test_missing_selection_partition_is_rejected(self):
        cohort = cohort_fixture()
        cohort.validation.pop()
        with self.assertRaisesRegex(ValueError, "16_SELECTION"):
            fitting_cases(cohort)

    def test_sampling_keys_repeat_and_separate_cases_and_training_seeds(self):
        keys = [sampling_seeds(seed, f"train_block_{block:03d}_familiar_{template}")
                for seed in (41, 42, 43) for block in range(14) for template in range(4)]
        self.assertEqual(len({value for pair in keys for value in pair}), 336)
        self.assertEqual(keys[0], sampling_seeds(41, "train_block_000_familiar_0"))
        with self.assertRaisesRegex(ValueError, "REGISTERED_TRAINING_SEED"):
            sampling_seeds(44, "train_block_000_familiar_0")


class NativeCompletePosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_dynamics.yaml").read_text())
        model_config = dict(spec["lfads_baseline_check"]["model"], co_dim=4,
                            controller_input_lag=2, do_feed_factors_to_controller=True)
        cls.model = LFADSConditionalSimulator([12, 4], 1, 3, 3, .05,
            model_config, initialization_seed=41,
            device=os.environ.get("NRNSIM_LFADS_TEST_DEVICE", "cpu"))
        cls.y = np.array([[[0, 2], [1, 0], [2, 1], [1, 3], [0, 1], [1, 1]]], dtype=np.int64)
        cls.u = np.linspace(0, .8, 6, dtype=np.float32).reshape(1, 6, 1)
        cls.seeds = sampling_seeds(41, "train_block_000_familiar_0")

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def native_feed(self):
        source = self.model._full.model
        feed = source.build_feed_dict("recording", np.repeat(self.y, 4, axis=0),
                                      np.repeat(self.u, 4, axis=0), keep_prob=1.)
        initial, controller = (np.random.default_rng(seed) for seed in self.seeds)
        feed[source.posterior_zs_g0.noise_bxn] = initial.standard_normal(
            (4, self.model.model_config["ic_dim"])).astype(np.float32)
        for posterior in source.posterior_zs_co:
            feed[posterior.noise_bxn] = controller.standard_normal((4, 4)).astype(np.float32)
        return feed

    def test_complete_paths_match_direct_native_ancestral_graph_bitwise(self):
        before = self.model._full.values()
        updates = self.model.training_updates
        sites = list(self.model._full.randomness.sites)
        actual = sample_complete_posterior(self.model, self.y, self.u, draws=4, seeds=self.seeds)
        source = self.model._full.model
        for key, attribute in (("paths", "sample"), ("means", "mean"), ("log_variances", "logvar")):
            native = self.model._full.session.run(
                [getattr(p, attribute) for p in source.posterior_zs_co], feed_dict=self.native_feed())
            np.testing.assert_array_equal(actual[key], np.stack(native, axis=1))
            self.assertEqual(actual[key].shape, (4, 6, 4))
        after = self.model._full.values()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
        self.assertEqual(self.model.training_updates, updates)
        self.assertEqual(self.model._full.randomness.sites, sites)

    def test_paths_replay_and_each_draw_keeps_feedback_ancestry(self):
        first = sample_complete_posterior(self.model, self.y, self.u, draws=4, seeds=self.seeds)
        again = sample_complete_posterior(self.model, self.y, self.u, draws=4, seeds=self.seeds)
        for name in first:
            np.testing.assert_array_equal(first[name], again[name])
        source = self.model._full.model
        feed = self.native_feed()
        feed[source.posterior_zs_co[0].noise_bxn] = np.full((4, 4), 8., dtype=np.float32)
        altered = np.stack(self.model._full.session.run(
            [p.mean for p in source.posterior_zs_co], feed_dict=feed), axis=1)
        np.testing.assert_array_equal(altered[:, 0], first["means"][:, 0])
        self.assertFalse(np.array_equal(altered[:, 1:], first["means"][:, 1:]))

    def test_single_draw_batches_match_native_paths_with_same_noise_and_batch(self):
        actual = sample_complete_posterior(self.model, self.y, self.u,
            draws=4, seeds=self.seeds, draw_batch_size=1)
        source = self.model._full.model
        full_feed = self.native_feed()
        for draw in range(4):
            feed = source.build_feed_dict("recording", self.y, self.u, keep_prob=1.)
            for noise in [source.posterior_zs_g0.noise_bxn,
                          *[p.noise_bxn for p in source.posterior_zs_co]]:
                feed[noise] = full_feed[noise][draw:draw + 1]
            direct = self.model._full.session.run(
                [p.sample for p in source.posterior_zs_co], feed_dict=feed)
            np.testing.assert_array_equal(actual["paths"][draw:draw + 1], np.stack(direct, axis=1))

    def test_partial_episodes_and_duplicated_rng_streams_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "COUNT_SEQUENCE"):
            sample_complete_posterior(self.model, self.y[:, :-1], self.u, draws=4, seeds=self.seeds)
        with self.assertRaisesRegex(ValueError, "TWO_DISTINCT"):
            sample_complete_posterior(self.model, self.y, self.u, draws=4, seeds=(1, 1))

    def test_prior_assignment_changes_exactly_eight_scalars_and_syncs_prefix(self):
        before = self.model._full.values()
        source = self.model._full.model
        fetches = dict(paths=[p.sample for p in source.posterior_zs_co], reconstruction=source.recon_cost)
        response_before = self.model._full.session.run(fetches, feed_dict=self.native_feed())
        updates = self.model.training_updates
        parameters = {"logatau": np.log(np.full((1, 4), 3., dtype=np.float32)),
                      "logevars": np.log(np.full((1, 4), .2, dtype=np.float32))}
        try:
            apply_prior_parameters(self.model, parameters)
            after = self.model._full.values()
            for name, value in before.items():
                if name not in PRIOR_VARIABLES.values():
                    np.testing.assert_array_equal(value, after[name], err_msg=name)
            for key, name in PRIOR_VARIABLES.items():
                np.testing.assert_array_equal(after[name], parameters[key])
            response_after = self.model._full.session.run(fetches, feed_dict=self.native_feed())
            for key in fetches:
                np.testing.assert_array_equal(response_before[key], response_after[key], err_msg=key)
            self.assertEqual(self.model.training_updates, updates)
            self.model._sync_prefix()
            prefix = self.model._prefix.values()
            for key, name in PRIOR_VARIABLES.items():
                np.testing.assert_array_equal(prefix[name], parameters[key])
        finally:
            self.model._full.restore(before)
            self.model._prefix_dirty = True

    def test_invalid_second_prior_parameter_causes_no_partial_assignment(self):
        before = self.model._full.values()
        with self.assertRaisesRegex(ValueError, "FOUR_CHANNEL_PRIOR"):
            apply_prior_parameters(self.model, {"logatau": np.ones((1, 4), np.float32),
                                               "logevars": np.zeros((1, 3), np.float32)})
        after = self.model._full.values()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_fitted_patch_reaches_native_causal_forecast_consumer(self):
        from .lfads_trajectory_evaluation import _loaded_model
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'original'
            self.model.save(directory)
            cohort = SimpleNamespace(unit_ids=np.asarray(self.model.unit_ids), verify_model=lambda model: None)
            binding = dict(seed=41, config=dict(history_bins=3, horizon_bins=3, bin_ms=50.,
                                                model=self.model.model_config))
            parameters = dict(logatau=np.ones((1, 4), np.float32), logevars=np.zeros((1, 4), np.float32))
            with _loaded_model(directory, cohort, binding, 2, 3, self.model.device,
                               prior_parameters=parameters) as loaded:
                state = loaded.infer_state(dict(counts=self.y[:, :2], stimulus=self.u[:, :2]),
                    metadata=dict(unit_ids=list(self.model.unit_ids), bin_seconds=.05))
                samples = loaded.simulate(state, self.u[:, 2:5], LFADSRandomStreams.from_seeds(1, 2, 3), 2)
                self.assertEqual(samples.counts.shape, (2, 1, 3, 2))
                for key, name in PRIOR_VARIABLES.items():
                    np.testing.assert_array_equal(loaded._full.session.run(loaded._full.variables[name]), parameters[key])
                    np.testing.assert_array_equal(loaded._prefix.session.run(loaded._prefix.variables[name]), parameters[key])


class CacheExecutionTests(unittest.TestCase):
    """Exercise real cache I/O/fit/recovery with explicitly synthetic paths.

    Source sampling is tested separately above. These fixtures are not source
    episodes, and never enter the repository's scientific evidence directories.
    """
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training = self.root / "training"
        self.training.mkdir()
        (self.training / "run.json").write_text('{"fixture": true}\n')
        self.output = self.root / "cache"
        self.cohort = cohort_fixture()
        self.cohort.binding = dict(scope="synthetic cache contract fixture")
        self.cohort.window = mock.Mock(return_value=(np.zeros((1200, 2), dtype=np.int32),
                                                     np.zeros((1200, 1), dtype=np.float32)))
        self.cohort.verify_model = mock.Mock()
        self.run = dict(binding=dict(seed=41))
        self.selected = dict(next_epoch=1, training_updates=56,
                             model_artifacts={"weights.npz": "fixture"})
        model = self.model = mock.MagicMock(history_bins=200, horizon_bins=1000)
        model.training_updates = 56
        model.model_config = dict(co_dim=4)
        model.execution_device = dict(scope="synthetic fixture")
        model._full.variables = {name: key for key, name in PRIOR_VARIABLES.items()}
        model._full.session.run.return_value = dict(
            logatau=np.log(np.full((1, 4), 10., dtype=np.float32)),
            logevars=np.log(np.full((1, 4), .1, dtype=np.float32)))
        model._full.values.return_value = {
            **{PRIOR_VARIABLES[key]: value for key, value in model._full.session.run.return_value.items()},
            'synthetic_unchanged_weight': np.ones((2, 2), np.float32)}
        self.loader = self.enterContext(mock.patch.object(refit.LFADSConditionalSimulator, "load"))
        self.loader.return_value.__enter__.return_value = model
        self.enterContext(mock.patch.object(refit, "_verified_training", return_value=(
            self.cohort, self.run, self.training, self.selected, None)))
        rng = np.random.default_rng(991)
        paths = np.empty((16, 1200, 4), dtype=np.float32)
        paths[:, 0] = rng.normal(size=(16, 4))
        for tick in range(1, 1200):
            paths[:, tick] = .65 * paths[:, tick - 1] + rng.normal(size=(16, 4))
        self.sampler = self.enterContext(mock.patch.object(refit, "sample_complete_posterior",
                                                          return_value=dict(paths=paths)))

    def run_cache(self, **kwargs):
        return refit.extract_and_fit(self.root / "acquisition", self.root / "fields",
            self.training, self.output, device="cpu", **kwargs)

    def test_interrupted_ledger_recovers_committed_paths_without_duplicate_sampling(self):
        original_write = refit._write_json

        def interrupt_after_second_file(path, value):
            if len(value["episodes"]) == 2:
                raise RuntimeError("intentional interruption after episode commit")
            original_write(path, value)

        with mock.patch.object(refit, "_write_json", side_effect=interrupt_after_second_file):
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                self.run_cache()
        saved = json.loads((self.output / "posterior_cache.json").read_text())
        self.assertEqual(len(saved["episodes"]), 1)
        self.assertTrue((self.output / "train_block_000_familiar_1.npz").exists())
        def interrupt_after_fit_artifact(path, value):
            if value["status"] == "FITTED_FROZEN_NETWORK_AR_PRIOR":
                raise RuntimeError("intentional interruption after prior commit")
            original_write(path, value)

        with mock.patch.object(refit, "_write_json", side_effect=interrupt_after_fit_artifact):
            with self.assertRaisesRegex(RuntimeError, "after prior commit"):
                self.run_cache(continue_cache=True)
        self.assertTrue((self.output / "prior_parameters.npz").exists())
        self.loader.side_effect = AssertionError("complete cache must not build another model")
        report = self.run_cache(continue_cache=True)
        self.assertEqual(report["status"], "FITTED_FROZEN_NETWORK_AR_PRIOR")
        self.assertEqual(report["complete_episodes"], 56)
        self.assertEqual(report["posterior_paths"], 896)
        self.assertEqual(report["fit_diagnostics"]["moments"]["sequence_length"], 1200)
        self.assertEqual(self.sampler.call_count, 56)
        self.assertEqual(self.cohort.window.call_count, 56)
        self.assertFalse(report["formal_test_accessed"])
        self.assertFalse(report["model_parameters_updated"])
        self.loader.side_effect = AssertionError("completed cache must not build another model")
        repeated = self.run_cache(continue_cache=True)
        self.assertEqual(repeated, report)
        parameters, evidence = refit.load_prior_parameters(self.output, run=self.run,
            selected=self.selected, cohort_binding=self.cohort.binding,
            training_run_sha256=refit.sha256_file(self.training / 'run.json'))
        self.assertEqual(set(parameters), set(PRIOR_VARIABLES))
        self.assertEqual(evidence['fitted_parameters'], 8)
        with self.assertRaisesRegex(ValueError, 'NETWORK_OR_FITTING_COHORT'):
            refit.load_prior_parameters(self.output, run=dict(binding=dict(seed=42)),
                selected=self.selected, cohort_binding=self.cohort.binding,
                training_run_sha256=refit.sha256_file(self.training / 'run.json'))
        changed = dict(self.selected, next_epoch=2)
        with self.assertRaisesRegex(ValueError, 'NETWORK_OR_FITTING_COHORT'):
            refit.load_prior_parameters(self.output, run=self.run, selected=changed,
                cohort_binding=self.cohort.binding,
                training_run_sha256=refit.sha256_file(self.training / 'run.json'))

    def test_changed_binding_and_cached_values_are_rejected_before_new_sampling(self):
        original_write = refit._write_json

        def stop_after_one(path, value):
            original_write(path, value)
            if len(value["episodes"]) == 1:
                raise RuntimeError("intentional interruption")

        with mock.patch.object(refit, "_write_json", side_effect=stop_after_one):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                self.run_cache()
        self.run["binding"]["seed"] = 42
        with self.assertRaisesRegex(ValueError, "FROZEN_POSTERIOR_CACHE_BINDING"):
            self.run_cache(continue_cache=True)
        self.run["binding"]["seed"] = 41
        path = self.output / "train_block_000_familiar_0.npz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "CACHE_IDENTITY_CHANGED"):
            self.run_cache(continue_cache=True)
        self.assertEqual(self.sampler.call_count, 1)

    def test_sampling_mutations_reject_before_atomic_episode_commit(self):
        baseline = {name: value.copy() for name, value in self.model._full.values.return_value.items()}
        for mutation in ("value", "dtype", "name", "counter", "counter_type"):
            with self.subTest(mutation=mutation):
                self.output = self.root / ("cache_" + mutation)
                self.model.training_updates = 56
                self.model._full.values.return_value = {name: value.copy() for name, value in baseline.items()}

                def mutate_network(*args, **kwargs):
                    values = self.model._full.values.return_value
                    if mutation == "value":
                        values["synthetic_unchanged_weight"][0, 0] += 1
                    elif mutation == "dtype":
                        values["synthetic_unchanged_weight"] = values["synthetic_unchanged_weight"].astype(np.float64)
                    elif mutation == "name":
                        values["renamed_weight"] = values.pop("synthetic_unchanged_weight")
                    elif mutation == "counter":
                        self.model.training_updates += 1
                    else:
                        self.model.training_updates = float(self.model.training_updates)
                    return self.sampler.return_value

                self.sampler.side_effect = mutate_network
                with mock.patch.object(refit, "_commit_paths", wraps=refit._commit_paths) as commit:
                    with self.assertRaisesRegex(ValueError, "CHANGED_FROZEN_NETWORK_OR_OPTIMIZER"):
                        self.run_cache()
                    commit.assert_not_called()
                self.assertFalse((self.output / "train_block_000_familiar_0.npz").exists())
                saved = json.loads((self.output / "posterior_cache.json").read_text())
                self.assertEqual(saved["episodes"], [])

    def test_update_count_mismatch_and_noninteger_values_fail_before_sampling(self):
        for index, updates in enumerate((55, 56., True)):
            with self.subTest(model_updates=updates):
                self.output = self.root / f"cache_model_update_{index}"
                self.model.training_updates = updates
                with self.assertRaisesRegex(ValueError, "CHECKPOINT_UPDATE_COUNT_CHANGED"):
                    self.run_cache()
                self.sampler.assert_not_called()
        self.model.training_updates = 56
        for index, updates in enumerate((56., True, -1, None)):
            with self.subTest(selected_updates=updates):
                self.output = self.root / f"cache_selected_update_{index}"
                self.selected["training_updates"] = updates
                with self.assertRaisesRegex(ValueError, "CHECKPOINT_TRAINING_UPDATE_COUNT_REQUIRED"):
                    self.run_cache()
                self.assertFalse(self.output.exists())
                self.sampler.assert_not_called()

    def test_last_atomic_commit_or_ledger_write_recovers_complete_proof_without_model_load(self):
        original_write = refit._write_json
        for ledger_written in (False, True):
            with self.subTest(ledger_written=ledger_written):
                self.output = self.root / ("cache_after_ledger" if ledger_written else "cache_after_commit")
                self.loader.side_effect = None
                self.loader.reset_mock()
                self.sampler.reset_mock()

                def interrupt_last_episode(path, value):
                    if len(value["episodes"]) == 56:
                        if ledger_written:
                            original_write(path, value)
                        raise RuntimeError("intentional interruption before complete-cache summary")
                    original_write(path, value)

                with mock.patch.object(refit, "_write_json", side_effect=interrupt_last_episode):
                    with self.assertRaisesRegex(RuntimeError, "before complete-cache summary"):
                        self.run_cache()
                saved = json.loads((self.output / "posterior_cache.json").read_text())
                self.assertEqual(len(saved["episodes"]), 56 if ledger_written else 55)
                self.assertNotIn("frozen_global_variables_verified", saved)
                last_path = self.output / "train_block_013_familiar_3.npz"
                with np.load(last_path, allow_pickle=False) as archive:
                    self.assertEqual(archive["frozen_global_variables_verified"].item(), 3)
                    self.assertEqual(archive["training_updates_before_and_after"].item(), 56)
                self.loader.side_effect = AssertionError("all episode commits must recover without model loading")
                report = self.run_cache(continue_cache=True)
                self.assertEqual(report["status"], "FITTED_FROZEN_NETWORK_AR_PRIOR")
                self.assertEqual(report["frozen_global_variables_verified"], 3)
                self.assertEqual(report["training_updates_before_and_after"], 56)
                self.assertTrue(all(row["frozen_global_variables_verified"] == 3
                                    and row["training_updates_before_and_after"] == 56
                                    for row in report["episodes"]))
                self.assertEqual(self.sampler.call_count, 56)
                self.assertEqual(self.loader.call_count, 1)

    def test_legacy_and_tampered_episode_proofs_reject_completed_fit_and_prior_loader(self):
        self.run_cache()
        record_path = self.output / "posterior_cache.json"
        original_record = record_path.read_text()
        episode_path = self.output / "train_block_000_familiar_0.npz"
        with np.load(episode_path, allow_pickle=False) as archive:
            original_payload = {name: archive[name] for name in archive.files}
        self.loader.side_effect = AssertionError("invalid cache must fail before loading a model")
        for change in ("legacy_protocol", "legacy_row", "legacy_payload", "row_count",
                       "payload_count", "payload_updates", "row_bool", "row_float",
                       "payload_bool", "payload_float", "payload_vector", "summary_count"):
            with self.subTest(change=change):
                record = json.loads(original_record)
                row = record["episodes"][0]
                payload = {name: value.copy() for name, value in original_payload.items()}
                if change == "legacy_protocol":
                    del record["binding"]["frozen_network_proof_protocol"]
                elif change == "legacy_row":
                    del row["frozen_global_variables_verified"]
                elif change == "legacy_payload":
                    del payload["frozen_global_variables_verified"]
                elif change == "row_count":
                    row["frozen_global_variables_verified"] = 4
                elif change == "payload_count":
                    payload["frozen_global_variables_verified"] = np.asarray(4)
                elif change == "payload_updates":
                    payload["training_updates_before_and_after"] = np.asarray(57)
                elif change == "row_bool":
                    row["frozen_global_variables_verified"] = True
                elif change == "row_float":
                    row["training_updates_before_and_after"] = 56.
                elif change == "payload_bool":
                    payload["frozen_global_variables_verified"] = np.asarray(True)
                elif change == "payload_float":
                    payload["training_updates_before_and_after"] = np.asarray(56.)
                elif change == "payload_vector":
                    payload["frozen_global_variables_verified"] = np.asarray([3])
                else:
                    record["frozen_global_variables_verified"] = 4
                np.savez_compressed(episode_path, **payload)
                # Keep the byte-integrity receipt valid so these checks exercise
                # proof consistency itself, including the actual NPZ payload.
                row["sha256"] = refit.sha256_file(episode_path)
                record_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "FROZEN.*PROOF"):
                    self.run_cache(continue_cache=True)
                with self.assertRaisesRegex(ValueError, "FROZEN.*PROOF"):
                    refit.load_prior_parameters(self.output, run=self.run,
                        selected=self.selected, cohort_binding=self.cohort.binding,
                        training_run_sha256=refit.sha256_file(self.training / "run.json"))
                if change in ("legacy_row", "legacy_payload"):
                    record["status"] = "EXTRACTING_FITTING_POSTERIOR"
                    record_path.write_text(json.dumps(record))
                    with mock.patch.object(refit, "fit_stationary_ar_prior") as fit:
                        with self.assertRaisesRegex(ValueError, "FROZEN.*PROOF"):
                            self.run_cache(continue_cache=True)
                        fit.assert_not_called()
        self.assertEqual(self.sampler.call_count, 56)

    def test_selection_substitution_fails_before_output_or_source_access(self):
        self.cohort.fit[0] = self.cohort.validation[0]
        with self.assertRaisesRegex(ValueError, "56_FITTING_EPISODES"):
            self.run_cache()
        self.assertFalse(self.output.exists())
        self.cohort.window.assert_not_called()
        self.sampler.assert_not_called()
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

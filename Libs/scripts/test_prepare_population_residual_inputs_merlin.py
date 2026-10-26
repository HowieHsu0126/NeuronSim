"""Keep generation-only inputs outside the fitting admission contract."""
import copy
import unittest

from prepare_population_residual_inputs_merlin import validate_input_contract


def generation_manifest():
    files = [dict(path=f"cohort/{area}/train_block_{block:03d}_familiar_{condition}/{name}",
                  bytes=1, sha256="source")
        for block in range(14, 18) for condition in range(4)
        for area, names in (("train", ("report.json", "spikes.npz", "initial_parameters.npz")),
                            ("train_fields", ("summary.json", "spatial_fields.npz")))
        for name in names]
    files.append(dict(path="generation_config/neural_population_matched_forecast.yaml",
                      bytes=1, sha256="parent_forecast"))
    return dict(schema="native_residual_observation_inputs_v1", training_seeds=[41, 42, 43],
                source_units=100000, scope="native_generation_additional_inputs",
                blocks=list(range(14, 18)), files=files, forecast_config_sha256="parent_forecast")


class InputPurposeTests(unittest.TestCase):
    def test_generation_requires_complete_native_episode_inputs(self):
        manifest = generation_manifest()
        validate_input_contract(manifest, "generation")
        for index in (0, 1, 2, 3, 4, 79, 80):
            with self.subTest(index=index):
                missing = copy.deepcopy(manifest)
                del missing["files"][index]
                with self.assertRaises(ValueError):
                    validate_input_contract(missing, "generation")

    def test_purposes_cannot_be_interchanged(self):
        with self.assertRaises(ValueError):
            validate_input_contract(generation_manifest(), "fitting")
        fitting = dict(schema="native_residual_observation_inputs_v1", training_seeds=[41, 42, 43],
                       source_units=100000, scope="complete_fitting_and_internal_selection",
                       blocks=list(range(14)))
        validate_input_contract(fitting, "fitting")
        with self.assertRaises(ValueError):
            validate_input_contract(fitting, "generation")

    def test_generation_rejects_duplicate_or_additional_files(self):
        for duplicate in (True, False):
            manifest = generation_manifest()
            manifest["files"].append(dict(manifest["files"][0]) if duplicate else
                                     dict(path="cohort/train/test_episode/spikes.npz", bytes=1, sha256="source"))
            with self.assertRaises(ValueError):
                validate_input_contract(manifest, "generation")

    def test_generation_requires_parent_frozen_forecast_identity(self):
        for identity in (None, "fitting_config"):
            manifest = generation_manifest()
            manifest["forecast_config_sha256"] = identity
            with self.assertRaises(ValueError):
                validate_input_contract(manifest, "generation")


if __name__ == "__main__":
    unittest.main()

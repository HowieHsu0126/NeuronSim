"""Exercise inference-to-report delivery using a tiny real population model."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import yaml

from . import population_boundary_closure as closure
from .test_population_boundary_closure import HistoryFixture, TinyNative, tiny_model


def test_three_seed_inference_to_complete_boundary_and_block_artifacts(tmp_path):
    fixture = HistoryFixture(tmp_path)
    for source in fixture.report["source_bindings"]:
        directory = fixture.fields / source["case_id"]
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(dict(test_episode=source["case_id"])))
        (directory / "spatial_fields.npz").write_bytes(b"tiny-observed-source-fixture")
        source["field_summary_sha256"] = closure.sha256_file(directory / "summary.json")
        source["spatial_fields_sha256"] = closure.sha256_file(directory / "spatial_fields.npz")
    fixture.seal()
    report, arrays, names = fixture.load()
    # Keep the registered row/target/block family. Only unit/clock dimensions
    # and source loading are substituted; inference, regression and delivery run.
    config = dict(fixture.config, expected_units=7, origins_ms=[10., 20., 30.])
    for row in report["rows"]:
        row["origin_ms"] = config["origins_ms"][row["cell"] % 3]
    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    models = {}

    def load_model(training, *args):
        seed = int(training.parent.name.removeprefix("seed_"))
        model = tiny_model()
        models[seed] = model
        return model, dict(seed=seed, native_source_identity={"scope": "tiny_test"}), str(seed) * 32

    def native_fixture(*args, **kwargs):
        return TinyNative(tiny_model())

    before = torch.get_rng_state().clone()
    output = tmp_path / "analysis"
    with patch.object(closure, "validate_config"), \
         patch.object(closure, "load_history", return_value=(report, arrays, names)), \
         patch.object(closure, "NativeRenewalEpisode", side_effect=native_fixture), \
         patch.object(closure, "_load_model", side_effect=load_model):
        result = closure.run(fixture.acquisition, fixture.fields, fixture.root,
                             tmp_path / "campaign", config_path, output)
    assert torch.equal(before, torch.get_rng_state())
    assert sorted(models) == [41, 42, 43]
    assert result["status"] == "COMPLETE_THREE_SEED_BOUNDARY_HISTORY_DIAGNOSTIC"
    assert result["independent_selection_blocks"] == 4
    assert not result["scientific_claim_qualified"] and not result["state_selection_enabled"]
    assert not result["calibration_outcomes_accessed"] and not result["test_outcomes_accessed"]
    for seed in [41, 42, 43]:
        cell = output / f"seed_{seed}"
        with np.load(cell / "boundaries.npz", allow_pickle=False) as saved:
            assert saved["boundaries"].shape == (216, 19)
            np.testing.assert_array_equal(saved["unit_ids"], models[seed].unit_ids)
            np.testing.assert_array_equal(saved["origins_ms"], [r["origin_ms"] for r in report["rows"]])
            np.testing.assert_array_equal(saved["case_ids"], [r["case_id"] for r in report["rows"]])
            assert set(saved["blocks"].tolist()) == set(range(18))
        witness = json.loads((cell / "witness.json").read_text())
        assert len(witness["target_names"]) == 1920
        assert [int(b["block"]) for b in witness["source_block_rows"]] == [14, 15, 16, 17]
        assert witness["ridge_diagonal_penalty"] == 1.68
        assert witness["selection_blocks_used_for_checkpoint_selection"]
    with (output / "errors_by_block.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 4 * 8
    assert {int(r["training_seed"]) for r in rows} == {41, 42, 43}
    assert {int(r["block"]) for r in rows} == {14, 15, 16, 17}
    assert all(float(r["history_gain"]) == 0 for r in rows)


def test_real_cli_rejects_formal_scope_before_loading_inputs(tmp_path):
    path = tmp_path / "formal.yaml"
    path.write_text(yaml.safe_dump(dict(scope="formal")))
    command = [sys.executable, "-m", "Libs.population_dynamics.population_boundary_closure",
               "--config", str(path), "--acquisition", str(tmp_path), "--fields", str(tmp_path),
               "--history", str(tmp_path), "--validate-only"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "registered complete inference-only development witness required" in result.stderr

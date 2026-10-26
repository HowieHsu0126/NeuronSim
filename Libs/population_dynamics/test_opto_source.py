"""Native staged Opto source contracts, using explicit format-2 Zarr chunks."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

from . import __main__ as population_cli
from .config import load_config
from .opto_source import check_opto_source


CANONICAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "neural_population_dynamics.yaml"
TRIAL_COLUMNS = (
    "id", "start_time", "stop_time", "duration", "power", "site", "num_pulses",
    "wavelength", "emission_location", "stimulation_type", "stim_id",
    "inter_pulse_interval", "inter_trial_interval", "rise_time",
)
UNIT_COLUMNS = ("id", "unit_name", "default_qc", "spike_times_index", "device_name", "brain_region")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _refresh_manifest(source: Path) -> None:
    records = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.name != "copy_out.json":
            payload = path.read_bytes()
            records.append({"path": path.relative_to(source).as_posix(), "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
    _write_json(source / "copy_out.json", {"files": records})


def _make_source(source: Path, config: dict) -> None:
    """Store every chunk, including zero-valued chunks, without implicit fill data."""
    opto = config["opto_source"]
    source.mkdir()
    _write_json(source / "source_context.json", {
        "source": {"origin": opto["source_origin"], "bucket": opto["source_bucket"]},
        "nwb_prefix": opto["nwb_prefix"],
        "engineering_scope": {"subject_id": opto["engineering_anchor_subject_id"]},
    })
    _write_json(source / "subject.json", {"subject_id": opto["engineering_anchor_subject_id"]})
    arrays = {
        "intervals/trials/id": np.array([20, 21], dtype="i8"),
        "intervals/trials/start_time": np.array([1.0, 2.0], dtype="f8"),
        "intervals/trials/stop_time": np.array([1.01, 2.01], dtype="f8"),
        "intervals/trials/duration": np.array([0.01, 0.01], dtype="f8"),
        "intervals/trials/power": np.array([0.1, 0.2], dtype="f8"),
        "intervals/trials/site": np.array([0, 1], dtype="i8"),
        "intervals/trials/num_pulses": np.array([1, 1], dtype="i8"),
        "intervals/trials/wavelength": np.array([470.0, 470.0], dtype="f8"),
        "intervals/trials/emission_location": np.array(["tip", "tip"], dtype=np.dtypes.StringDType()),
        "intervals/trials/stimulation_type": np.array(["pulse", "pulse"], dtype="U5"),
        "intervals/trials/stim_id": np.array([0, 1], dtype="i8"),
        "intervals/trials/inter_pulse_interval": np.array([0.0, 0.0], dtype="f8"),
        "intervals/trials/inter_trial_interval": np.array([1.0, 1.0], dtype="f8"),
        "intervals/trials/rise_time": np.array([0.0, 0.0], dtype="f8"),
        "units/id": np.array([4, 7, 11], dtype="i8"),
        "units/unit_name": np.array(["unit4", "unit7", "unit11"], dtype=np.dtypes.StringDType()),
        "units/default_qc": np.array([True, True, False], dtype="bool"),
        "units/spike_times_index": np.array([1, 3, 5], dtype="i8"),
        "units/device_name": np.array(["probeA", "probeA", "probeB"], dtype="U6"),
        "units/brain_region": np.array(["VISp", "VISp", "VISp"], dtype="U4"),
        "units/spike_times": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="f8"),
    }
    attributes = {
        "intervals/trials/start_time": {"unit": "seconds"},
        "intervals/trials/stop_time": {"unit": "seconds"},
        "intervals/trials/duration": {"unit": "ms", "description": "ms"},
        "intervals/trials/power": {"unit": "uW", "description": "uW"},
        "units/spike_times": {"unit": "seconds"},
    }
    metadata = {
        ".zattrs": {"neurodata_type": "NWBFile", "nwb_version": "2.8.0"},
        "intervals/trials/.zattrs": {"colnames": list(TRIAL_COLUMNS[1:])},
        "units/.zattrs": {"colnames": list(UNIT_COLUMNS[1:])},
    }
    for name, values in arrays.items():
        array_path = source / "selected_source_arrays" / name
        array = zarr.create_array(store=str(array_path), data=values, chunks=(2,),
                                  zarr_format=2, config={"write_empty_chunks": True})
        array.attrs.update(attributes.get(name, {}))
        for filename in (".zarray", ".zattrs"):
            metadata[f"{name}/{filename}"] = json.loads((array_path / filename).read_text(encoding="utf-8"))
    _write_json(source / "nwb_zmetadata.json", {"zarr_consolidated_format": 1, "metadata": metadata})
    _refresh_manifest(source)


class OptoSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="neuronsim-opto-source-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "read"
        self.config = load_config(CANONICAL_CONFIG)
        _make_source(self.source, self.config)

    def test_variable_utf8_exports_load_without_pickle(self):
        self.read_source()
        with np.load(self.output / "source_tables.npz", allow_pickle=False) as saved:
            for name in saved.files:
                self.assertNotEqual(saved[name].dtype.kind, "O")
            np.testing.assert_array_equal(saved["units/unit_name"], ["unit4", "unit7", "unit11"])
            np.testing.assert_array_equal(saved["intervals/trials/emission_location"], ["tip", "tip"])

    def read_source(self, unit_ids: list[int] | None = None, output: Path | None = None):
        return check_opto_source(self.source, self.output if output is None else output, self.config,
                                 ["python", "-m", "population_dynamics", "opto-source-read"],
                                 [4, 7] if unit_ids is None else unit_ids)

    def test_subset_preserves_source_denominators_and_exact_spikes(self) -> None:
        summary, code = self.read_source()
        self.assertEqual(code, 2)
        expected = {"status": "PARTIAL", "source_unit_count": 3, "source_default_qc_units": 2,
                    "source_trial_count": 2, "selected_unit_count": 2, "selected_spike_count": 3,
                    "total_source_spike_count": 5, "dataset_ready": False,
                    "physical_power_qualified": False}
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(summary[name], value)
        with np.load(self.output / "selected_spikes.npz", allow_pickle=False) as selected:
            np.testing.assert_array_equal(selected["unit_ids"], np.array([4, 7], dtype="i8"))
            np.testing.assert_array_equal(selected["spike_offsets"], np.array([0, 1, 3], dtype="i8"))
            np.testing.assert_array_equal(selected["spike_times_s"], np.array([0.1, 0.2, 0.3], dtype="f8"))
            self.assertEqual(selected["spike_times_s"].dtype, np.dtype("f8"))

    def test_source_power_and_duration_are_not_rescaled(self) -> None:
        self.read_source()
        with np.load(self.output / "source_tables.npz", allow_pickle=False) as tables:
            np.testing.assert_array_equal(tables["intervals/trials/power"], np.array([0.1, 0.2], dtype="f8"))
            np.testing.assert_array_equal(tables["intervals/trials/duration"], np.array([0.01, 0.01], dtype="f8"))

    def test_missing_selected_chunk_is_not_replaced_by_fill_values(self) -> None:
        (self.source / "selected_source_arrays/units/spike_times/1").unlink()
        # Removing the inventory row must not make the missing selected data admissible.
        _refresh_manifest(self.source)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read_source()

    def test_changed_selected_chunk_digest_is_rejected(self) -> None:
        path = self.source / "selected_source_arrays/units/spike_times/0"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read_source()

    def test_source_subject_identity_must_match_configuration(self) -> None:
        _write_json(self.source / "subject.json", {"subject_id": "different-subject"})
        _refresh_manifest(self.source)
        with self.assertRaises(ValueError):
            self.read_source()

    def test_source_context_identity_must_match_configuration(self) -> None:
        path = self.source / "source_context.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for index, (section, key, value) in enumerate((
            ("engineering_scope", "subject_id", "different-subject"),
            (None, "nwb_prefix", "another-session/recording.nwb/"),
            ("source", "bucket", "another-bucket"),
            ("source", "origin", "another-origin"),
        )):
            with self.subTest(field=key):
                context = copy.deepcopy(original)
                target = context if section is None else context[section]
                target[key] = value
                _write_json(path, context)
                _refresh_manifest(self.source)
                with self.assertRaises(ValueError):
                    self.read_source(output=self.root / f"context_{index}")

    def test_ragged_offsets_preserve_total_source_spike_denominator(self) -> None:
        array = zarr.open_array(str(self.source / "selected_source_arrays/units/spike_times_index"), mode="r+")
        for index, offsets in enumerate(([1, 3, 4], [1, 0, 5], [1, 3, 6])):
            with self.subTest(offsets=offsets):
                array[:] = np.array(offsets, dtype="i8")
                _refresh_manifest(self.source)
                with self.assertRaises(ValueError):
                    self.read_source(output=self.root / f"offsets_{index}")

    def test_requested_ids_must_be_unique_known_and_source_qc_eligible(self) -> None:
        for index, unit_ids in enumerate(([4, 4], [999], [11], [])):
            with self.subTest(unit_ids=unit_ids):
                with self.assertRaises(ValueError):
                    self.read_source(unit_ids=unit_ids, output=self.root / f"selection_{index}")

    def test_missing_required_table_metadata_is_rejected(self) -> None:
        path = self.source / "nwb_zmetadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        del metadata["metadata"]["intervals/trials/.zattrs"]
        _write_json(path, metadata)
        _refresh_manifest(self.source)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read_source()

    def test_missing_required_array_is_rejected(self) -> None:
        (self.source / "selected_source_arrays/intervals/trials/power/.zarray").unlink()
        _refresh_manifest(self.source)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read_source()

    def test_local_metadata_must_match_consolidated_source_metadata(self) -> None:
        path = self.source / "selected_source_arrays/units/spike_times/.zattrs"
        attributes = json.loads(path.read_text(encoding="utf-8"))
        attributes["unit"] = "milliseconds"
        _write_json(path, attributes)
        _refresh_manifest(self.source)
        with self.assertRaises(ValueError):
            self.read_source()

    def test_existing_output_cannot_be_overwritten(self) -> None:
        self.read_source()
        with self.assertRaises((ValueError, FileExistsError)):
            self.read_source()

    def cli_arguments(self, config_path: Path = CANONICAL_CONFIG) -> list[str]:
        return ["--config", str(config_path), "--project-root", str(self.root), "opto-source-read",
                "--source-dir", str(self.source), "--diagnostic-unit-ids", "4", "7",
                "--output-dir", str(self.output)]

    def test_cli_returns_partial_without_a_visual_coding_cohort(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = population_cli.main(self.cli_arguments())
        self.assertEqual(code, 2, stderr.getvalue())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertIs(summary["dataset_ready"], False)
        self.assertIs(summary["physical_power_qualified"], False)

    def test_cli_source_failure_returns_nonzero(self) -> None:
        (self.source / "selected_source_arrays/units/spike_times/1").unlink()
        _refresh_manifest(self.source)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = population_cli.main(self.cli_arguments())
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        failure = json.loads(stderr.getvalue())
        self.assertEqual(failure["status"], "FAILED")
        self.assertEqual(failure["operation"], "opto-source-read")
        self.assertIs(failure["dataset_ready"], False)

    def test_visual_only_config_stays_valid_but_opto_requires_settings(self) -> None:
        config = copy.deepcopy(self.config)
        del config["opto_source"]
        config_path = self.root / "visual_only.json"
        _write_json(config_path, config)
        self.assertNotIn("opto_source", load_config(config_path))
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = population_cli.main(self.cli_arguments(config_path))
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"], "OPTO_SOURCE_CONFIG_REQUIRED")


if __name__ == "__main__":
    unittest.main()

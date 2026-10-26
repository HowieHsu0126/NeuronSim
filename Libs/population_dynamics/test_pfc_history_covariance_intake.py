"""Test immutable covariance intake without running the covariance kernel."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from . import pfc_history_covariance_intake as intake


NEGATIVE_COVARIANCE = "-8.541666666666666e-05"
EDGES = (0, 8000, 24080, 32000, 36000, 192000, 196000, 320000, 400000)
VARIANCE_FIELDS = (
    "phase_index", "start_tick", "stop_tick", "population", "units", "independent_histories",
    "time_bins", "bin_ticks", "bin_ms", "complete_history_events",
    "unordered_history_pair_unit_bin_coincidences", "population_mean_variance_rate_Hz2",
    "diagonal_contribution_rate_Hz2", "ordered_offdiagonal_contribution_rate_Hz2", "sample_axis",
)
PAIR_FIELDS = ("left_history_index", "right_history_index", "phase_index",
               "excitatory_unit_bin_coincidences", "inhibitory_unit_bin_coincidences")


def file_binding(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    return file_binding(path)


@pytest.fixture
def original_tables(tmp_path):
    """Fixed string-valued input tables; this fixture performs no computation."""
    variance = []
    pairs = []
    for phase in range(8):
        for population, units in (("excitatory", 80000), ("inhibitory", 20000), ("all_units", 100000)):
            variance.append(dict(phase_index=str(phase), start_tick=str(EDGES[phase]),
                stop_tick=str(EDGES[phase + 1]), population=population, units=str(units),
                independent_histories="3", time_bins=str((EDGES[phase + 1] - EDGES[phase]) // 80),
                bin_ticks="80", bin_ms="1.0", complete_history_events="4125",
                unordered_history_pair_unit_bin_coincidences="0",
                population_mean_variance_rate_Hz2="0.0020630208333333334",
                diagonal_contribution_rate_Hz2="0.0021484375",
                ordered_offdiagonal_contribution_rate_Hz2=NEGATIVE_COVARIANCE,
                sample_axis="matched_time_across_independent_histories_within_circuit_and_condition"))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            pairs.append(dict(left_history_index=str(left), right_history_index=str(right),
                phase_index=str(phase), excitatory_unit_bin_coincidences="0",
                inhibitory_unit_bin_coincidences="0"))
    return {
        intake.VARIANCE_NAME: write_csv(tmp_path / intake.VARIANCE_NAME, VARIANCE_FIELDS, variance),
        intake.PAIR_NAME: write_csv(tmp_path / intake.PAIR_NAME, PAIR_FIELDS, pairs),
    }


def group_report(key, tables):
    circuit, condition = divmod(key, 30)
    circuit_id = f"pfc_working_memory_circuit_{circuit:03d}"
    split = "train" if circuit < 70 else "validation" if circuit < 80 else "calibration" if circuit < 90 else "test"
    bindings = []
    for position in range(3):
        block = position + 6
        block_id = circuit_id + f"_block_{block:03d}"
        identity = dict(circuit_index=circuit, condition_index=condition, block_index=block,
            circuit_id=circuit_id, block_id=block_id,
            episode_id=block_id + f"_condition_{condition:02d}", split=split,
            scenario_id="pfc_working_memory", sample_unit="complete_mechanistic_episode",
            initial_state_seed=210000000 + circuit * 100 + block,
            background_seed=220000000 + circuit * 100 + block)
        binding = dict(task_index=key + position * 10, identity=identity,
            raw_sha256="a" * 64, source_report_sha256="b" * 64,
            retained_binary_unit_bins=4125, terminal_events_excluded=0,
            source_execution_backend="merlin", source_run="fixture-source-run",
            source_trial="fixture-source-trial", source_bundle_sha256="c" * 64,
            manifest_sha256="d" * 64)
        if key == 0 and position == 0:
            binding.update(source_execution_backend="slurm", source_run="643446", source_trial=None,
                source_bundle_sha256=None, manifest_sha256=None,
                source_delivery_sha256="e" * 64, transfer_receipt_sha256="f" * 64,
                transfer_run="fixture-transfer-run", transfer_trial="fixture-transfer-trial")
        bindings.append(binding)
    return dict(schema="pfc_matched1ms_history_coincidences_v1",
        status="COMPLETE_REAL3_HISTORY_MATCHED1MS_UNIT_COVARIANCE",
        circuit_index=circuit, condition_index=condition, core_units=100000,
        duration_ticks=400000, observation_bin_ticks=80, phase_count=8,
        complete_source_episodes_reused=3, independent_histories=3, history_pairs=3,
        source_simulations=0, production_episodes=0, history_pairs_are_independent_replicates=False,
        source_qualification_granted=False, scaling_qualified=False, uncertainty_coverage_qualified=False,
        execution=dict(backend="merlin", run_id="fixture-covariance-run", trial_id="fixture-covariance-trial"),
        implementation_sha256="1" * 64, request_sha256="2" * 64,
        bindings=bindings, outputs={name: {field: binding[field] for field in ("bytes", "sha256")}
                                    for name, binding in tables.items()})


def write_group(tmp_path, key, tables, report=None):
    path = tmp_path / f"group_{key:04d}.json"
    path.write_text(json.dumps(group_report(key, tables) if report is None else report), encoding="utf-8")
    return dict(group_key=key, report=file_binding(path), tables=copy.deepcopy(tables))


def write_manifest(tmp_path, groups):
    path = tmp_path / "intake_manifest.json"
    path.write_text(json.dumps(dict(schema="pfc_history_covariance_intake_v1", groups=groups)), encoding="utf-8")
    return path


def test_missing_group_fails_before_opening_originals_or_creating_output(tmp_path):
    keys = [30 * circuit + condition for circuit in range(100) for condition in range(10)]
    manifest = write_manifest(tmp_path, [dict(group_key=key) for key in keys[:-1]])
    output = tmp_path / "authoritative"
    with pytest.raises(ValueError, match="MISSING_COVARIANCE_GROUPS:required1000:actual999"):
        intake.main(["--manifest", str(manifest), "--output", str(output)])
    assert not output.exists()


def test_duplicate_group_source_fails_before_opening_originals(tmp_path):
    manifest = write_manifest(tmp_path, [dict(group_key=0), dict(group_key=0)])
    output = tmp_path / "authoritative"
    with pytest.raises(ValueError, match="DUPLICATE_COVARIANCE_GROUP_SOURCE"):
        intake.run(manifest, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["repeated_task", "wrong_episode", "repeated_stream", "condition", "clock", "bin"])
def test_changed_history_identity_or_clock_is_rejected(tmp_path, original_tables, change):
    report = group_report(0, original_tables)
    if change == "repeated_task":
        report["bindings"][1]["task_index"] = 0
    elif change == "wrong_episode":
        report["bindings"][1]["identity"]["episode_id"] = report["bindings"][0]["identity"]["episode_id"]
    elif change == "repeated_stream":
        report["bindings"][1]["identity"]["initial_state_seed"] = report["bindings"][0]["identity"]["initial_state_seed"]
    elif change == "condition":
        report["condition_index"] = 1
    elif change == "clock":
        report["duration_ticks"] = 200000
    else:
        report["observation_bin_ticks"] = 40
    entry = write_group(tmp_path, 0, original_tables, report)
    with pytest.raises(ValueError, match="IDENTITY_CHANGED|HISTORIES_NOT_DISTINCT|CONTRACT_CHANGED"):
        intake.validate_single(entry, tmp_path)


@pytest.mark.parametrize("name", [intake.VARIANCE_NAME, intake.PAIR_NAME, "report"])
def test_changed_file_is_rejected(tmp_path, original_tables, name):
    entry = write_group(tmp_path, 0, original_tables)
    binding = entry["report"] if name == "report" else entry["tables"][name]
    with Path(binding["path"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="ORIGINAL_FILE_BINDING_CHANGED"):
        intake.validate_single(entry, tmp_path)


def test_report_table_hash_mismatch_is_rejected(tmp_path, original_tables):
    entry = write_group(tmp_path, 0, original_tables)
    entry["tables"][intake.PAIR_NAME]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="REPORT_TABLE_BINDING_CHANGED"):
        intake.validate_single(entry, tmp_path)


@pytest.mark.parametrize("change", ["coarse_bin", "phase_edge", "duplicate_pair", "nonfinite"])
def test_changed_native_table_cells_are_rejected(tmp_path, original_tables, change):
    name = intake.PAIR_NAME if change == "duplicate_pair" else intake.VARIANCE_NAME
    path = Path(original_tables[name]["path"])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if change == "coarse_bin":
        rows[0]["bin_ms"] = "2"
    elif change == "phase_edge":
        rows[0]["stop_tick"] = "8080"
    elif change == "duplicate_pair":
        rows[1] = dict(rows[0])
    else:
        rows[0]["ordered_offdiagonal_contribution_rate_Hz2"] = "NaN"
    original_tables[name] = write_csv(path, fields, rows)
    entry = write_group(tmp_path, 0, original_tables)
    with pytest.raises(ValueError, match="CLOCK_CHANGED|CELL_DUPLICATE_OR_INVALID|NONFINITE"):
        intake.validate_single(entry, tmp_path)


def test_single_group_retains_signed_numeric_strings_and_original_provenance(tmp_path, original_tables):
    entry = write_group(tmp_path, 0, original_tables)
    result = intake.validate_single(entry, tmp_path)
    assert len(result["variance_rows"]) == len(result["pair_rows"]) == 24
    assert result["variance_rows"][0]["ordered_offdiagonal_contribution_rate_Hz2"] == NEGATIVE_COVARIANCE
    assert result["variance_rows"][0]["population_mean_variance_rate_Hz2"] == "0.0020630208333333334"
    assert result["variance_rows"][0]["circuit_index"] == "0"
    assert result["source"]["bindings"][0]["source_execution_backend"] == "slurm"
    assert result["source"]["bindings"][0]["source_run"] == "643446"
    assert result["source"]["bindings"][0]["transfer_run"] == "fixture-transfer-run"
    assert result["report"]["source_qualification_granted"] is False


def test_exact1000_cli_intake_writes24000_rows_and_keeps_scientific_gates_closed(tmp_path, original_tables, capsys):
    keys = [30 * circuit + condition for circuit in range(100) for condition in range(10)]
    entries = [write_group(tmp_path, key, original_tables) for key in keys]
    manifest = write_manifest(tmp_path, list(reversed(entries)))
    output = tmp_path / "authoritative"
    intake.main(["--manifest", str(manifest), "--output", str(output)])
    status = json.loads(capsys.readouterr().out)
    assert status == dict(status="COMPLETE1000_DESCRIPTIVE_MATCHED1MS_COVARIANCE_INTAKE",
                          completed_groups=1000, required_groups=1000)
    report = json.loads((output / "report.json").read_text())
    assert report["completed_group_keys"] == keys
    assert report["total_history_units"] == 300
    assert report["complete_source_episodes_reused"] == 3000
    assert report["source_simulations"] == report["production_episodes"] == 0
    for flag in ("source_qualification_granted", "scientific_parameter_domain_qualified",
                 "scaling_qualified", "uncertainty_coverage_qualified", "history_pairs_are_independent_replicates"):
        assert report[flag] is False
    for name in (intake.VARIANCE_NAME, intake.PAIR_NAME):
        with (output / name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert reader.fieldnames[:2] == ["circuit_index", "condition_index"]
        assert len(rows) == report["outputs"][name]["rows"] == 24000
        assert {(int(row["circuit_index"]), int(row["condition_index"])) for row in rows} == {
            (circuit, condition) for circuit in range(100) for condition in range(10)}
        assert file_binding(output / name)["sha256"] == report["outputs"][name]["sha256"]
    with (output / intake.VARIANCE_NAME).open(newline="") as stream:
        assert next(csv.DictReader(stream))["ordered_offdiagonal_contribution_rate_Hz2"] == NEGATIVE_COVARIANCE
    with pytest.raises(ValueError, match="IMMUTABLE_INTAKE_OUTPUT_EXISTS"):
        intake.run(manifest, output)


def test_cross_condition_stream_change_fails_before_output(tmp_path, original_tables):
    entries = []
    for circuit in range(100):
        for condition in range(10):
            key = 30 * circuit + condition
            report = group_report(key, original_tables)
            if key == 1:
                report["bindings"][0]["identity"]["background_seed"] += 10000
            entries.append(write_group(tmp_path, key, original_tables, report))
    output = tmp_path / "authoritative"
    with pytest.raises(ValueError, match="PAIRED_CONDITION_SOURCE_STREAMS_CHANGED"):
        intake.run(write_manifest(tmp_path, entries), output)
    assert not output.exists()


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_malformed_single_entry_has_meaningful_failure(value, tmp_path):
    with pytest.raises(ValueError, match="INTAKE_GROUP_SOURCE_INVALID"):
        intake.validate_single(value, tmp_path)


def test_malformed_table_binding_has_meaningful_failure(tmp_path, original_tables):
    entry = write_group(tmp_path, 0, original_tables)
    entry["tables"][intake.PAIR_NAME] = None
    with pytest.raises(ValueError, match="REPORT_TABLE_BINDING_CHANGED"):
        intake.validate_single(entry, tmp_path)


def legacy_merlin_report(key, tables, omit_owner=False):
    """Original C23 and main report field shapes before explicit backend fields."""
    report = group_report(key, tables)
    if key == 690:
        source_run, source_trial = "6de397acc866aadd", "435906314"
        report["implementation_sha256"] = "6b0a4934412e4a7f6290df4dfbc3a106999825baacd7ba3c72ea4b47f004eefa"
    else:
        source_run, source_trial = "21ba1b0a0338a1c9", "435964747"
        report["implementation_sha256"] = "caba95a18dc517d996ec130a4a376fa178fb447b2b35f3a967c10fd56878769a"
    if omit_owner:
        del report["circuit_index"]
        del report["condition_index"]
    for item in report["bindings"]:
        del item["source_execution_backend"]
        item["source_run"], item["source_trial"] = source_run, source_trial
        item["checkpoint_path"] = ("hdfs://haruna/home/byte_aml_platform/public/rh2_prod/"
            "fixture/checkpoints/20261021/" + source_run + f"/pfc/task_{item['task_index']:04d}")
    return report


@pytest.mark.parametrize("key,omit_owner", [(690, True), (35, False)])
def test_original_legacy_merlin_fields_validate_without_modifying_evidence(tmp_path, original_tables, key, omit_owner):
    report = legacy_merlin_report(key, original_tables, omit_owner)
    result = intake.validate_single(write_group(tmp_path, key, original_tables, report), tmp_path)
    assert result["report"] == report
    assert result["source"]["bindings"] == report["bindings"]
    assert all("source_execution_backend" not in item for item in result["source"]["bindings"])
    assert ("circuit_index" not in result["report"]) is omit_owner
    assert result["variance_rows"][0]["circuit_index"] == str(key // 30)
    assert result["variance_rows"][0]["condition_index"] == str(key % 30)
    assert result["variance_rows"][0]["ordered_offdiagonal_contribution_rate_Hz2"] == NEGATIVE_COVARIANCE
    assert result["report"]["source_qualification_granted"] is False


@pytest.mark.parametrize("field,change", [
    ("circuit_index", "missing"), ("condition_index", "missing"),
    ("circuit_index", "wrong"), ("condition_index", "wrong"),
    ("circuit_index", "null"), ("condition_index", "boolean"),
])
def test_partial_or_inconsistent_top_owner_is_rejected_for_legacy_reports(tmp_path, original_tables, field, change):
    report = legacy_merlin_report(35, original_tables)
    if change == "missing":
        del report[field]
    elif change == "wrong":
        report[field] += 1
    elif change == "null":
        report[field] = None
    else:
        report[field] = True
    with pytest.raises(ValueError, match="GROUP_OWNER_IDENTITY_CHANGED"):
        intake.validate_single(write_group(tmp_path, 35, original_tables, report), tmp_path)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_owner_omission_still_checks_every_original_history(tmp_path, original_tables, position):
    report = legacy_merlin_report(690, original_tables, omit_owner=True)
    report["bindings"][position]["identity"]["condition_index"] = 1
    with pytest.raises(ValueError, match="GROUP_HISTORY_IDENTITY_CHANGED"):
        intake.validate_single(write_group(tmp_path, 690, original_tables, report), tmp_path)


@pytest.mark.parametrize("backend", [None, "", "unknown", True])
def test_explicit_invalid_backend_never_uses_legacy_provenance(tmp_path, original_tables, backend):
    report = legacy_merlin_report(35, original_tables)
    report["bindings"][0]["source_execution_backend"] = backend
    with pytest.raises(ValueError, match="ORIGINAL_SOURCE_BACKEND_INVALID"):
        intake.validate_single(write_group(tmp_path, 35, original_tables, report), tmp_path)


@pytest.mark.parametrize("field", ["source_run", "source_trial", "source_bundle_sha256", "manifest_sha256", "checkpoint_path"])
def test_legacy_backend_requires_all_original_merlin_provenance(tmp_path, original_tables, field):
    report = legacy_merlin_report(35, original_tables)
    del report["bindings"][1][field]
    with pytest.raises(ValueError, match="ORIGINAL_LEGACY_MERLIN_SOURCE_BINDING_REQUIRED"):
        intake.validate_single(write_group(tmp_path, 35, original_tables, report), tmp_path)


@pytest.mark.parametrize("field,value", [
    ("source_run", None), ("source_run", "643446"),
    ("source_trial", None), ("source_trial", "fixture-trial"),
    ("source_bundle_sha256", None), ("manifest_sha256", None),
    ("checkpoint_path", None),
    ("checkpoint_path", "hdfs://haruna/checkpoints/20261021/21ba1b0a0338a1c9/pfc/task_0046"),
    ("checkpoint_path", "hdfs://haruna/checkpoints/20261021/6de397acc866aadd/pfc/task_0045"),
    ("checkpoint_path", "https://haruna/checkpoints/20261021/21ba1b0a0338a1c9/pfc/task_0045"),
])
def test_legacy_backend_rejects_null_or_mismatched_merlin_bindings(tmp_path, original_tables, field, value):
    report = legacy_merlin_report(35, original_tables)
    report["bindings"][1][field] = value
    with pytest.raises(ValueError, match="ORIGINAL_LEGACY_MERLIN_SOURCE_BINDING_REQUIRED"):
        intake.validate_single(write_group(tmp_path, 35, original_tables, report), tmp_path)

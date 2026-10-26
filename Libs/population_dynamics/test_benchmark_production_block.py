"""Complete-block routing and resume tests on existing engineering archives."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from . import benchmark_production as production
from . import benchmark_production_block as blocks
from .test_benchmark_production import cohort, source_fixture


def block_inputs(root, cohort):
    root.mkdir(parents=True)
    fields = blocks.V1_SOURCE_FIELDS if cohort.scenario == "v1_visual" else blocks.CA3_SOURCE_FIELDS
    records = []
    for index in range(10):
        sources = {}
        for field in fields:
            name = f"{field}_{index}" if cohort.scenario == "v1_visual" and field != "authority_source" else field
            path = root / name
            if not path.exists():
                path.mkdir()
            sources[field] = name
        records.append(dict(condition_index=index, sources=sources))
    document = dict(schema=blocks.BLOCK_SCHEMA, scenario_id=cohort.scenario,
                    circuit_index=0, block_index=1, conditions=records)
    path = root / "block.json"
    path.write_text(json.dumps(document))
    return path, document


def test_block_rejects_missing_or_reordered_conditions_before_output(cohort, tmp_path, monkeypatch):
    path, document = block_inputs(tmp_path / "inputs", cohort)
    execute = Mock(side_effect=AssertionError("invalid block must not execute"))
    monkeypatch.setattr(production, "export_v1", execute)
    monkeypatch.setattr(production, "run_ca3", execute)
    for records in (document["conditions"][:-1],
                    [document["conditions"][1], document["conditions"][0], *document["conditions"][2:]]):
        changed = dict(document, conditions=records)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="TEN_CONDITIONS|CONDITION_ORDER"):
            blocks.process_block(cohort, path, 0, 1, tmp_path / "output")
    execute.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_block_rejects_reused_or_mixed_sources(cohort, tmp_path):
    path, document = block_inputs(tmp_path / "inputs", cohort)
    if cohort.scenario == "v1_visual":
        document["conditions"][9]["sources"]["native_source"] = document["conditions"][0]["sources"]["native_source"]
        message = "SOURCE_REUSED"
    else:
        document["conditions"][9]["sources"]["formation"] = "another_circuit"
        message = "SHARED_CIRCUIT_DIFFERS"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        blocks.load_block(cohort, path, 0, 1)


def test_block_rejects_last_missing_resource_before_first_execution(cohort, tmp_path, monkeypatch):
    path, document = block_inputs(tmp_path / "inputs", cohort)
    field = "native_source" if cohort.scenario == "v1_visual" else "current_replay_engine"
    (path.parent / document["conditions"][9]["sources"][field]).rmdir()
    execute = Mock(side_effect=AssertionError("missing input must be found before processing"))
    monkeypatch.setattr(production, "export_v1", execute)
    monkeypatch.setattr(production, "run_ca3", execute)
    with pytest.raises(ValueError, match="SOURCE_MISSING"):
        blocks.process_block(cohort, path, 0, 1, tmp_path / "output")
    execute.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_block_retains_commits_and_failure_then_resumes_without_rerunning(cohort, tmp_path, monkeypatch):
    path, document = block_inputs(tmp_path / "inputs", cohort)
    output = tmp_path / "output"
    calls = []
    fail = True

    def execute(actual_cohort, identity, *args, resume=False):
        nonlocal fail
        index = identity["condition_index"]
        calls.append(index)
        if fail and index == 3:
            raise RuntimeError("synthetic interrupted condition")
        destination = args[4]  # Both existing APIs place output after four source arguments.
        source, _, admission = source_fixture(tmp_path / "fixtures" / f"condition_{index}", actual_cohort, identity)
        return production.write_episode(actual_cohort, identity, source, destination,
                                        admission=admission, resume=resume)

    monkeypatch.setattr(production, "export_v1", execute)
    monkeypatch.setattr(production, "run_ca3", execute)
    with pytest.raises(RuntimeError, match="synthetic interrupted condition"):
        blocks.process_block(cohort, path, 0, 1, output)
    assert calls == [0, 1, 2, 3]
    reports = list((output / "attempts").glob("*/report.json"))
    assert len(reports) == 1
    failed_bytes = reports[0].read_bytes()
    failed = json.loads(failed_bytes)
    assert failed["status"] == "FAILED_QUALIFIED_PRODUCTION_BLOCK"
    assert failed["complete_episodes"] == 3
    assert failed["active_condition_index"] == 3
    assert not (output / "report.json").exists()
    dataset = production.CircuitDataset(output / "dataset", splits=("train",))
    assert set(dataset.episodes) == {cohort.identity(0, 1, index)["episode_id"] for index in range(3)}
    assert len(dataset.catalog["circuits"]) == 1
    if cohort.scenario == "v1_visual":
        for index in range(3):
            for name in ("rate_source", "spike_source", "native_source"):
                (path.parent / document["conditions"][index]["sources"][name]).rmdir()
    fail = False
    result = blocks.process_block(cohort, path, 0, 1, output, resume=True)
    assert calls == [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9]
    assert result["status"] == "COMPLETE_QUALIFIED_PRODUCTION_BLOCK"
    assert result["complete_episodes"] == 10
    assert result["reused_episodes"] == 3
    dataset = production.CircuitDataset(output / "dataset", splits=("train",))
    assert len(dataset.episodes) == 10
    assert len(dataset.catalog["circuits"]) == 1
    assert reports[0].read_bytes() == failed_bytes
    assert [r["identity"]["condition_index"] for r in result["results"]] == list(range(10))
    execute_again = Mock(side_effect=AssertionError("complete block must not execute again"))
    monkeypatch.setattr(production, "export_v1", execute_again)
    monkeypatch.setattr(production, "run_ca3", execute_again)
    final_bytes = (output / "report.json").read_bytes()
    reused = blocks.process_block(cohort, path, 0, 1, output, resume=True)
    assert reused["status"] == "REUSED_COMMITTED_PRODUCTION_BLOCK"
    assert reused["reused_episodes"] == 10
    execute_again.assert_not_called()
    assert (output / "report.json").read_bytes() == final_bytes
    with pytest.raises(ValueError, match="EXPLICIT_RESUME"):
        blocks.process_block(cohort, path, 0, 1, output)
    altered = json.loads(final_bytes)
    altered["complete_episodes"] = 9
    (output / "report.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="COMMITTED_REPORT_DIFFERS"):
        blocks.process_block(cohort, path, 0, 1, output, resume=True)
    (output / "report.json").unlink()
    recovered = blocks.process_block(cohort, path, 0, 1, output, resume=True)
    assert recovered["status"] == "COMPLETE_QUALIFIED_PRODUCTION_BLOCK"
    assert recovered["complete_episodes"] == recovered["reused_episodes"] == 10
    execute_again.assert_not_called()


def test_block_real_cli_keeps_unfrozen_campaign_closed(cohort, tmp_path):
    document = yaml.safe_load((cohort.qualification_path.parent / "campaign.yaml").read_text())
    document["simulation_campaign"]["scenarios"][cohort.scenario]["task_protocol"]["frozen"] = False
    config = tmp_path / "unfrozen_campaign.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False))
    output = tmp_path / "output"
    command = [sys.executable, "-B", "-m", "population_dynamics.benchmark_production_block",
               "--config", str(config), "--scenario", cohort.scenario,
               "--qualification", str(cohort.qualification_path),
               "--identity-table", str(tmp_path / "missing_table.csv"),
               "--circuit-index", "0", "--block-index", "1",
               "--inputs", str(tmp_path / "missing_inputs.json"), "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert "BENCHMARK_PRODUCTION_NOT_QUALIFIED" in result.stderr
    assert "task_protocol" in result.stderr
    assert not output.exists()


def test_block_rejects_changed_circuit_map_in_later_condition(cohort, tmp_path, monkeypatch):
    path, _ = block_inputs(tmp_path / "inputs", cohort)
    output = tmp_path / "output"
    calls = []

    def execute(actual_cohort, identity, *args, resume=False):
        index = identity["condition_index"]
        calls.append(index)
        source, episode, admission = source_fixture(tmp_path / "fixtures" / str(index), actual_cohort, identity)
        if index == 1:
            units = production.read_circuit(source, episode.circuit_id)
            changed = tmp_path / "changed_source"
            with production.CircuitArchiveWriter(changed, source.catalog["binding"]) as writer:
                writer.register_circuit(replace(units, coordinate_semantics="different source map"))
                writer.add(episode)
            source = production.CircuitDataset(changed, splits=("development",))
        return production.write_episode(actual_cohort, identity, source, args[4],
                                        admission=admission, resume=resume)

    monkeypatch.setattr(production, "export_v1", execute)
    monkeypatch.setattr(production, "run_ca3", execute)
    with pytest.raises(ValueError, match="CIRCUIT_OBSERVATION_MAP_REBOUND"):
        blocks.process_block(cohort, path, 0, 1, output)
    assert calls == [0, 1]
    dataset = production.CircuitDataset(output / "dataset", splits=("train",))
    assert set(dataset.episodes) == {cohort.identity(0, 1, 0)["episode_id"]}
    assert not (output / "report.json").exists()


def test_block_resume_rejects_catalog_from_another_block(cohort, tmp_path, monkeypatch):
    path, _ = block_inputs(tmp_path / "inputs", cohort)
    identity = cohort.identity(0, 2, 0)
    source, _, admission = source_fixture(tmp_path / "other_source", cohort, identity)
    output = tmp_path / "output"
    production.write_episode(cohort, identity, source, output / "dataset", admission=admission)
    execute = Mock(side_effect=AssertionError("mixed block must not execute"))
    monkeypatch.setattr(production, "export_v1", execute)
    monkeypatch.setattr(production, "run_ca3", execute)
    with pytest.raises(ValueError, match="CONTAINS_OTHER_IDENTITIES"):
        blocks.process_block(cohort, path, 0, 1, output, resume=True)
    execute.assert_not_called()


def test_failed_resume_reports_verified_later_condition(cohort, tmp_path, monkeypatch):
    path, _ = block_inputs(tmp_path / "inputs", cohort)
    identity = cohort.identity(0, 1, 9)
    source, _, admission = source_fixture(tmp_path / "existing_source", cohort, identity)
    output = tmp_path / "output"
    production.write_episode(cohort, identity, source, output / "dataset", admission=admission)
    execute = Mock(side_effect=RuntimeError("first missing condition failed"))
    monkeypatch.setattr(production, "export_v1", execute)
    monkeypatch.setattr(production, "run_ca3", execute)
    with pytest.raises(RuntimeError, match="first missing condition failed"):
        blocks.process_block(cohort, path, 0, 1, output, resume=True)
    report = json.loads(next((output / "attempts").glob("*/report.json")).read_text())
    assert report["complete_episodes"] == report["reused_episodes"] == 1
    assert report["results"][0]["identity"] == identity
    assert report["active_condition_index"] == 0

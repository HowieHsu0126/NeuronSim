"""Tests for the complete descriptive size-calibration boundary."""
import json
from pathlib import Path

import numpy as np
import pytest

from . import pfc_size_calibration_summary as summary


def test_size_difference_keeps_missing_and_negative_values():
    assert summary.descriptive_difference({"mean": -2}, {"mean": -4}, "mean")["expanded_minus_original"] == -2
    result = summary.descriptive_difference({"mean": None}, {"mean": "3"}, "mean")
    assert result["difference_defined"] is False and result["expanded_minus_original"] is None


def test_size_comparison_requires_identical_estimand_denominator():
    original = [dict(circuit_index=23, mean=1)]
    with pytest.raises(ValueError, match="MATCHED_ESTIMANDS"):
        summary.compare_rows(original, [dict(circuit_index=63, mean=1)], ("circuit_index",), ("mean",))
    with pytest.raises(ValueError, match="MATCHED_ESTIMANDS"):
        summary.compare_rows(original, original+original, ("circuit_index",), ("mean",))


def test_complete_calibration_plan_rejects_changed_split_and_stream_namespace(tmp_path):
    plan = dict(status="PREPARED120_ORIGINAL_SIZE_CALIBRATION_EPISODES",
        scope="descriptive_training_calibration_size_comparison", circuits=list(summary.CIRCUITS),
        blocks=list(summary.BLOCKS), conditions_per_block=10, source_qualification_granted=False,
        scientific_parameter_domain_qualified=False, production_episodes=0,
        comparison_sampling=dict(within_size_conditions_paired=True, cross_size_noise_pairing=False,
            comparator_namespace_offset=100000000), rows=[])
    for c in summary.CIRCUITS:
        for b in summary.BLOCKS:
            for k, condition in enumerate(summary.quality.CONDITIONS):
                identity = dict(circuit_index=c, block_index=b, condition_index=k,
                    split="calibration" if c == 86 else "train",
                    connectivity_seed=300000000+c, initial_state_seed=310000000+c*1000+b,
                    background_seed=320000000+c*1000+b, stimulus_seed=330000000+c*1000+b)
                canonical = dict(identity)
                for name in ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed"):
                    canonical[name] -= 100000000
                config = tmp_path/f"config_{len(plan['rows'])}.yaml";config.write_text("fixture: true\n")
                plan["rows"].append(dict(task_index=len(plan["rows"]), identity=identity,
                    canonical_campaign_identity=canonical, split=identity["split"], condition=dict(condition_id=condition),
                    config_path=config.name, config_sha256=summary.sha256_file(config)))
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan))
    assert len(summary.calibration_plan(path)["rows"]) == 120
    plan["rows"][0]["identity"]["background_seed"] += 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="STREAM_NAMESPACE"):
        summary.calibration_plan(path)
    plan["rows"][0]["identity"]["background_seed"] -= 1
    plan["rows"][0]["split"]="validation";path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        summary.calibration_plan(path)


def test_incomplete_calibration_withholds_output_before_array_loading(tmp_path, monkeypatch):
    plan=dict(rows=[dict(task_index=0, identity=dict(episode_id="missing"))])
    monkeypatch.setattr(summary,"calibration_plan",lambda path:plan)
    monkeypatch.setattr(summary,"reference_tables",lambda path:(dict(bindings=[]),{}))
    monkeypatch.setattr(summary.quality,"load_cell",lambda *args,**kwargs:pytest.fail("loaded incomplete denominator"))
    output=tmp_path/"output"
    with pytest.raises(ValueError,match="INCOMPLETE_CELL:0"):
        summary.aggregate(tmp_path/"plan.json",tmp_path/"reference.json",output)
    assert not output.exists()


def test_complete_calibration_preserves_undefined_angles_and_negative_covariance(tmp_path, monkeypatch):
    rows=[];bindings=[];cells={};reference_data={name:[] for name in
        ("phase_block_summary.csv","paired_contrast_summary.csv","population_variance_summary.csv")}
    edges=np.array([0,8000,24080,32000,36000,192000,196000,320000,400000])
    for c in summary.CIRCUITS:
        for b in summary.BLOCKS:
            for k,condition in enumerate(summary.quality.CONDITIONS):
                episode=f"pfc_{c}_{b}_{k}";identity=dict(circuit_index=c,block_index=b,condition_index=k,episode_id=episode)
                row=dict(task_index=len(rows),identity=identity,canonical_campaign_identity=identity)
                rows.append(row)
                directory=tmp_path/"readout"/f"task_{row['task_index']:03d}";directory.mkdir(parents=True)
                target=tmp_path/"reference"/episode;target.mkdir(parents=True)
                (target/"report.json").write_text(json.dumps(dict(identity=identity,source_cost=None)))
                bindings.append(dict(episode_id=episode,readout=str(target),report_sha256=summary.sha256_file(target/"report.json")))
                report=dict(source_size_comparison=dict(comparator_role="original_size2560",coupled_core_units=2560,
                    target_coupled_core_units=100000,cross_size_noise_pairing=False,canonical_campaign_identity=identity,
                    campaign_episodes=0,production_episodes=0),source_delivery_sha256="fixture")
                (directory/"report.json").write_text("{}")
                phases=[dict({name:float(k-b) for name in summary.quality.RATE_FIELDS},
                    circular_cue_error_deg="" if (b==4 and k==5) else "2") for _ in range(8)]
                arrays=dict(population_field_counts=np.zeros((1000,17),dtype=np.int32),
                    phase_unit_counts=np.zeros((8,2560),dtype=np.int32),
                    temporal_unit_variance_counts=np.ones(2560),phase_edges_ticks=edges)
                cells[str(directory)]=(report,phases,arrays)
        for k,condition in enumerate(summary.quality.CONDITIONS):
            for phase in range(8):
                for metric in (*summary.quality.RATE_FIELDS,"absolute_circular_cue_error_deg"):
                    reference_data["phase_block_summary.csv"].append(dict(circuit_index=c,condition_id=condition,
                        phase_index=phase,metric=metric,mean=2,sample_sd=1))
            for population in ("excitatory","inhibitory","all_units"):
                reference_data["population_variance_summary.csv"].append(dict(circuit_index=c,condition_id=condition,population=population,
                    **{name:0 for name in ("temporal_population_mean_variance_counts2","temporal_diagonal_contribution_counts2",
                    "temporal_ordered_offdiagonal_contribution_counts2","between_block_matched_time_population_mean_variance_counts2",
                    "between_block_matched_time_mean_single_unit_variance_counts2","between_block_matched_time_diagonal_contribution_counts2",
                    "between_block_matched_time_ordered_offdiagonal_contribution_counts2")}))
        for label,conditions in (("cue_minus_omission",summary.quality.CONDITIONS[:5]),("distractor_difference_of_differences",summary.quality.CONDITIONS[1:5])):
            for condition in conditions:
                for phase in range(8):
                    for metric in summary.quality.RATE_FIELDS:
                        reference_data["paired_contrast_summary.csv"].append(dict(circuit_index=c,condition_id=condition,
                            phase_index=phase,metric=metric,contrast=label,mean=0,sample_sd=0))
    monkeypatch.setattr(summary,"calibration_plan",lambda path:dict(rows=rows))
    monkeypatch.setattr(summary,"reference_tables",lambda path:(dict(bindings=bindings),reference_data))
    monkeypatch.setattr(summary.quality,"load_cell",lambda path,*args,**kwargs:cells[str(path)])
    monkeypatch.setattr(summary.quality,"paired_block_binding",lambda rows,reports:dict(initial_voltage_sha256=str(rows[0]["identity"]["block_index"]),
        final_rng_sha256=str(rows[0]["identity"]["block_index"])))
    monkeypatch.setattr(summary.quality,"archived_events",lambda *args:None)
    monkeypatch.setattr(summary.quality,"matched_time_unit_statistics",lambda *args,**kwargs:dict(population={name:dict(
        between_block_matched_time_mean_single_unit_variance_counts2=1,
        between_block_matched_time_diagonal_contribution_counts2=1,
        between_block_matched_time_ordered_offdiagonal_contribution_counts2=-1) for name in ("excitatory","inhibitory","all_units")}))
    monkeypatch.setattr(summary,"csv_rows",lambda path:[] if path.name=="temporal_responses.csv" else [])
    monkeypatch.setattr(summary.quality,"lagged_angular_displacement",lambda *args,**kwargs:[dict(direction_defined=False)])
    original_hash=summary.sha256_file
    monkeypatch.setattr(summary,"sha256_file",lambda path:original_hash(path) if path.exists() else "fixture")
    output=tmp_path/"output";result=summary.aggregate(tmp_path/"plan.json",tmp_path/"reference.json",output)
    assert result["original_size_episodes"]==120 and result["expanded_size_episodes_reused"]==120
    assert result["source_qualification_granted"] is False and result["production_episodes"]==0
    monkeypatch.undo()
    phase=summary.csv_rows(output/"phase_summary.csv")
    undefined=[row for row in phase if row["condition_id"]==summary.quality.CONDITIONS[5] and row["metric"]=="absolute_circular_cue_error_deg"]
    assert all(row["mean"]==row["sample_sd"]=="" and row["undefined_blocks"]=="1" for row in undefined)
    population=summary.csv_rows(output/"population_variance_summary.csv")
    assert all(float(row["between_block_matched_time_ordered_offdiagonal_contribution_counts2"])==-1 for row in population)

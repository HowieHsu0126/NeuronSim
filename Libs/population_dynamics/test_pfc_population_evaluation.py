"""PFC evaluation entry point and independent full-output replay."""
from __future__ import annotations

import numpy as np
import csv
import json
import pytest
import yaml

from . import pfc_population_evaluation as evaluator
from . import pfc_population_training_data as inputs
from . import population_trajectory_training as training
from . import population_trajectory_evaluation as public
from . import population_forecast_replay as replay_public
from .test_pfc_population_training_data import configuration, DifferentMapCohort
from .test_population_metadata import model as metadata_model


class EvaluationCohort(DifferentMapCohort):
    def __init__(self,path,config):
        super().__init__(path,config)
        self.validation.append(dict(case_id='validate_thirteen', circuit_id='fixture_13_5', n=13, f=5))
        self.bin_ticks=config['bin_ticks']
        self.binding['circuit_inputs_sha256']='fixture_inputs'

    def _episode(self, case):
        parent = self
        parent.active_case = case
        canonical = metadata_model()
        view = parent.model_for_case(canonical, case)
        class Episode:
            duration_ticks = 40
            binding = dict(fixture_case=case["case_id"], circuit_id=view.metadata["source_identity"]["circuit_id"],
                           block_id="fixture_source_" + case["case_id"])
            unit_ids = np.asarray(view.unit_ids)
            field_names = view.field_names
            unit_field_index = view.unit_field_index.numpy()
            minimum_interval_ticks = view.minimum_interval_ticks.numpy()
            dt_ms = view.tick_seconds * 1000
            bin_ticks = view.bin_ticks
            control_features = view.metadata["control_features"]
            def window(self, start, stop):
                whole = parent.window(case)
                return whole.counts[start//5:stop//5], whole.stimulus[start//5:stop//5]
            def ready_at(self, tick):
                return np.zeros(case["n"], dtype=np.int64)
        return Episode()


@pytest.fixture
def fitted(tmp_path,monkeypatch):
    config=configuration(); config['validation_circuit_ids'].append('fixture_13_5')
    path=tmp_path/'training.yaml';path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(inputs,'PfcPopulationTrainingCohort',EvaluationCohort)
    monkeypatch.setattr(evaluator,'PfcPopulationTrainingCohort',EvaluationCohort)
    def controls(episode, start, stop):
        return dict(known_controls=np.zeros(((stop-start)//episode.bin_ticks, len(episode.control_features))),
                    control_features=episode.control_features)
    monkeypatch.setattr(evaluator.shared, 'source_controls', controls)
    output=tmp_path/'training';training.train(tmp_path,None,path,output,41,stop_after_epochs=1)
    spec=tmp_path/'inputs.json';spec.write_text('{}')
    cfg=dict(schema='pfc_population_checkpoint_forecast_v1',evaluation_scope='declared_development_validation_checkpoint',
        formal_evaluation=False,field_units='source_field_mean_unit_rate_hz',causal_history='all_observed_bins_before_origin',
        finite_ensemble_corrections=True,origins_ticks=[10,20],horizons_bins=[1,2],support_windows=[1,2],joint_samples=3,
        sampling_seed_base=79001)
    evaluation=tmp_path/'evaluation.yaml';evaluation.write_text(yaml.safe_dump(cfg))
    return spec,output,evaluation


def test_public_evaluator_replays_every_unit_and_independent_metric(fitted,tmp_path):
    spec,training,config=fitted; output=tmp_path/'forecasts'
    result=public.evaluate(spec,None,training,config,output,'population','cpu')
    assert result['score_rows']==8 and len(result['origins'])==4
    receipt=replay_public.replay(spec,None,training,output,tmp_path/'replay.json','cpu',all_origins=True)
    assert receipt['unit_values_consumed']==3*2*(9+13)*2 and not receipt['scientific_claim_qualified']
    name=result['origins'][0]['artifact']
    with np.load(output/name) as stored:
        assert stored['generated_unit_counts'].shape==(3,2,9)
        assert stored['generated_field_counts'].shape==(3,2,2)


def test_replay_rejects_changed_output_without_receipt(fitted,tmp_path):
    spec,training,config=fitted; output=tmp_path/'forecasts'
    report=public.evaluate(spec,None,training,config,output,'population','cpu')
    with (output/report['origins'][0]['artifact']).open('ab') as stream:stream.write(b'changed')
    receipt=tmp_path/'replay.json'
    with pytest.raises(ValueError,match='ARTIFACT_CHANGED'):
        replay_public.replay(spec,None,training,output,receipt,'cpu')
    assert not receipt.exists()


def test_replay_rejects_within_field_unit_swap_with_updated_artifact_hash(fitted,tmp_path):
    spec,training,config=fitted; output=tmp_path/'forecasts'
    report=public.evaluate(spec,None,training,config,output,'population','cpu')
    name=report['origins'][0]['artifact']
    with np.load(output/name,allow_pickle=False) as stored:
        payload={k:stored[k].copy() for k in stored.files}
    payload['generated_unit_counts'][0,0,0] += 1
    np.savez_compressed(output/name,**payload)
    from .provenance import sha256_file
    report['artifacts'][name]=sha256_file(output/name)
    (output/'evaluation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='ALL_UNIT_OR_SOURCE_MAP'):
        replay_public.replay(spec,None,training,output,tmp_path/'replay.json','cpu')


def test_forecast_rejects_incomplete_source_clock(fitted,tmp_path):
    spec,training,config=fitted
    values=yaml.safe_load(config.read_text());values['origins_ticks']=[11];config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='ALIGNED_PREFIX'):
        public.evaluate(spec,None,training,config,tmp_path/'invalid','population','cpu')


@pytest.mark.parametrize('changed,reason', [
    ('scores', 'COMPLETE_SCORE_DENOMINATOR'),
    ('units', 'SOURCE_MAP_SIZE'),
])
def test_replay_rejects_changed_complete_denominator(fitted,tmp_path,changed,reason):
    from .provenance import sha256_file
    spec,training,config=fitted; output=tmp_path/'forecasts'
    report=public.evaluate(spec,None,training,config,output,'population','cpu')
    if changed == 'scores':
        scores=output/'scores.csv'
        lines=scores.read_text().splitlines(keepends=True)
        scores.write_text(''.join(lines+[lines[-1]]))
        report['artifacts']['scores.csv']=sha256_file(scores)
    else:
        report['origins'][0]['units'] += 1
    (output/'evaluation.json').write_text(json.dumps(report))
    receipt=tmp_path/'replay.json'
    with pytest.raises(ValueError,match=reason):
        replay_public.replay(spec,None,training,output,receipt,'cpu')
    assert not receipt.exists()


def _cost_inputs(fitted, output):
    spec, training_path, _ = fitted
    model, cohort, _ = evaluator._load(spec, training_path, evaluator.torch.device("cpu"))
    contracts = {}
    for case in cohort.validation:
        data = cohort._episode(case)
        cohort.active_case, cohort.active_episode = case, data
        contracts[case["case_id"]] = evaluator._case_cost_contract(case, cohort.model_for_case(model, case), data)
    with (output / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return cohort.validation, contracts, rows


def test_costs_load_each_map_once_and_infer_each_observed_prefix(fitted, tmp_path, monkeypatch):
    spec, training_path, config = fitted
    loads, prefixes = [], []
    original_load = evaluator._load
    original_prepare = evaluator.shared.prepare_factual_response
    def load(*args):
        model, cohort, run = original_load(*args)
        episode = cohort._episode
        def measured(case):
            loads.append(case["case_id"])
            return episode(case)
        cohort._episode = measured
        return model, cohort, run
    def prepare(view, data, origin, horizon):
        prefixes.append((data.binding["fixture_case"], origin))
        return original_prepare(view, data, origin, horizon)
    monkeypatch.setattr(evaluator, "_load", load)
    monkeypatch.setattr(evaluator.shared, "prepare_factual_response", prepare)
    output = tmp_path / "measured"
    report = evaluator.evaluate(spec, training_path, config, output)
    # Model reconstruction reads its fitted map before the measured episode loop.
    assert loads == ["validate_nine", "validate_thirteen"]
    assert prefixes == [(case, origin) for case in loads for origin in (10, 20)]
    assert len(report["episode_costs"]) == 2 and len(report["origin_costs"]) == 4
    assert [v["unit_count"] for v in report["episode_costs"]] == [9, 13]
    assert [v["field_count"] for v in report["origin_costs"]] == [2, 2, 5, 5]
    assert all(v["stored_full_unit_counts"] for v in report["origin_costs"])
    assert report["generated_unit_count_values"] == 3 * 2 * (9 + 13) * 2
    assert not report["speedup_qualified"]
    validation, contracts, rows = _cost_inputs(fitted, output)
    costs = evaluator.validate_forecast_costs(report, rows, validation=validation, case_cost_contracts=contracts)
    from .population_forecast_analysis import _measured_origin_columns
    for episode in costs["episode_costs"]:
        charged = [_measured_origin_columns({"measured_evaluation_costs": costs}, episode["case_id"], origin)
                   for origin in episode["origins_ms"]]
        assert charged[0]["episode_preparation_seconds"] == episode["episode_preparation_seconds"]
        assert charged[1]["episode_preparation_seconds"] is None
        assert episode["window_seconds"] == episode["transfer_seconds"] == 0
        assert episode["episode_inputs_materialized"] is False


@pytest.mark.parametrize("change", ["negative_timer", "missing_episode", "wrong_block", "wrong_units", "omit_full_units", "short_total"])
def test_replay_rejects_invalid_measured_cost_contract(fitted, tmp_path, change):
    spec, training_path, config = fitted
    output = tmp_path / "measured"
    report = evaluator.evaluate(spec, training_path, config, output)
    if change == "negative_timer": report["origin_costs"][0]["scoring_seconds"] = -1
    elif change == "missing_episode": report["episode_costs"].pop()
    elif change == "wrong_block": report["origin_costs"][0]["block"] = "unbound_source"
    elif change == "wrong_units": report["origin_costs"][0]["unit_count"] += 1
    elif change == "omit_full_units": report["origin_costs"][1]["stored_full_unit_counts"] = False
    else: report["origin_costs"][0]["origin_total_seconds"] = 0
    (output / "evaluation.json").write_text(json.dumps(report))
    receipt = tmp_path / "replay.json"
    with pytest.raises(ValueError):
        evaluator.replay(spec, training_path, output, receipt)
    assert not receipt.exists()


def test_legacy_reports_retain_missing_costs_and_replay(fitted, tmp_path):
    spec, training_path, config = fitted
    output = tmp_path / "legacy"
    report = evaluator.evaluate(spec, training_path, config, output)
    for name in ("origin_costs", "episode_costs"):
        report.pop(name)
    (output / "evaluation.json").write_text(json.dumps(report))
    assert evaluator.validate_forecast_costs(report, [], validation=[], case_cost_contracts={}) is None
    receipt = evaluator.replay(spec, training_path, output, tmp_path / "replay.json")
    assert receipt["unit_values_consumed"] == 3 * 2 * (9 + 13) * 2

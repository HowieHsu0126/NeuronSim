"""Bounded caller fixtures for future-assisted point diagnostics, not science."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import population_posterior_diagnostic as diagnostic
from .provenance import sha256_file


CONFIG = Path(__file__).parents[1] / "configs/neural_population_point_transition_diagnostic.yaml"


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    ids = np.arange(100000, dtype=np.int64)
    mapping = ids % 128
    field_ids = np.asarray([f"fixture_{field}" for field in range(128)])
    unit_count = np.bincount(mapping, minlength=128)
    interval = np.ones(100000, dtype=np.int64)
    cases = [dict(block=block, template_id=f"familiar_{template}", case_id=f"train_block_{block:03d}_familiar_{template}")
             for block in range(14,18) for template in range(4)]
    binding = dict(dataset_sha256="fixture_dataset", field_bindings={case["case_id"]: "fixture_source" for case in cases})
    truth = np.ones((1200,128), dtype=np.int64)
    source = SimpleNamespace(binding=binding, validation=cases, unit_ids=ids, field_ids=field_ids,
        unit_field_index=mapping, unit_count=unit_count, sd=np.ones(128), bin_ms=5., source_identity=dict(native="fixture"),
        truth={case["case_id"]:truth for case in cases})
    native = SimpleNamespace(dt_ms=.1, bin_ms=5., bin_ticks=50, duration_ticks=60000, field_ids=field_ids,
        unit_ids=ids, unit_field_index=mapping, minimum_interval_ticks=interval)
    items = []
    for role, ceiling in (("parent",20), ("continuation",80)):
        report = dict(common_training_budget=dict(maximum_epochs=ceiling), source_binding=binding,
            inputs=dict(forecasts=str(tmp_path / role)), model_seed_bindings={f"population_{seed}":dict(
                training_config=dict(representation="recurrent"), checkpoint_sha256=f"{role}_{seed}", training_report_sha256="fixture")
                for seed in (41,42,43)})
        items.append((report, [], [], [], dict(root=str(tmp_path / role), implementation_fixture=True)))
    monkeypatch.setattr(diagnostic, "_delivered_population_analysis", lambda path: items[0] if Path(path).name == "parent" else items[1])
    monkeypatch.setattr(diagnostic, "_population_budget_pair", lambda loaded: (items[0],items[1]))
    monkeypatch.setattr(diagnostic, "CanonicalFields", lambda acquisition, fields: source)
    native_calls, model_calls, calls = [], [], []

    def load_native(episode, field_view, **expected):
        native_calls.append((Path(episode).name, expected))
        np.testing.assert_array_equal(expected["expected_unit_ids"], ids)
        np.testing.assert_array_equal(expected["expected_unit_field_index"], mapping)
        assert expected["expected_source"] == source.source_identity
        return native

    def load_model(training, acquisition, fields, kind, device, view):
        role = Path(training).parents[1].name
        seed = int(Path(training).parent.name.removeprefix("seed_"))
        assert kind == "population" and device.type == "cpu" and view is native
        model_calls.append((role,seed))
        model = SimpleNamespace(representation="recurrent", role=role, seed=seed)
        return model, dict(seed=seed,config=dict(representation="recurrent")), f"{role}_{seed}"

    def plan(model, view, origin, stop):
        calls.append((model.role,model.seed,origin,stop))
        bins = (stop-origin)//50
        values = dict(bin_start_tick=np.arange(origin,stop,50,dtype=np.int64), controls=np.zeros((bins,1)),
            observed_total_spikes=np.full(bins,128.), eligible_tick_total=np.full(bins,5000000,dtype=np.int64),
            boundary_mean=np.zeros((1,128,4)), boundary_variance=np.ones((1,128,4)),
            boundary_ready_in_ticks=np.zeros((1,100000),dtype=np.int64))
        for name in ("latent_transition_residual_rms", "process_standardized_residual_rms", "delta_standardized_residual_rms",
                     "posterior_variance_mean", "propagated_previous_variance_mean", "process_variance_mean"):
            values[name]=np.ones(bins)
        for decoder, rate in (("posterior",2.),("one_step",3.)):
            values[decoder+"_event_nll"]=np.arange(1,bins+1,dtype=float)
            values[decoder+"_event_nll_per_unit_bin"]=values[decoder+"_event_nll"]/100000
            values[decoder+"_field_point_rate_hz"]=np.full((bins,128),rate)
            values[decoder+"_field_exposure_probability_sum"]=np.full((bins,128),.25)
            values[decoder+"_observed_exposure_probability_sum"]=np.full(bins,32.)
        return values

    monkeypatch.setattr(diagnostic, "NativeRenewalEpisode", load_native)
    monkeypatch.setattr(diagnostic, "_load_model", load_model)
    monkeypatch.setattr(diagnostic, "diagnose_native_plan", plan)
    return dict(analyses=[tmp_path/"parent",tmp_path/"continuation"], acquisition=tmp_path/"acquisition", fields=tmp_path/"fields",
                source=source, native=native, items=items, calls=calls, native_calls=native_calls, model_calls=model_calls)


def execute(campaign, output, config=CONFIG):
    return diagnostic.diagnose_population_campaign(campaign["analyses"],campaign["acquisition"],campaign["fields"],config,output)


def test_all_role_seed_origins_and_horizons_save_point_arrays_and_explicit_scope(campaign, tmp_path):
    output=tmp_path/"points"
    report=execute(campaign,output)
    assert report["diagnostic_windows"]==288 and report["score_rows"]==864
    assert len(campaign["native_calls"])==16 and set(campaign["model_calls"])=={(role,seed) for role in ("parent","continuation") for seed in (41,42,43)}
    for role in ("parent","continuation"):
        for seed in (41,42,43):
            windows=[call for call in campaign["calls"] if call[:2]==(role,seed)]
            assert len(windows)==48 and {call[2] for call in windows}=={10000,30000,50000}
            assert all(stop-origin==5000 for _,_,origin,stop in windows)
    assert report["future_observation_assisted"] is report["diagnostic_only"] is True
    assert report["samples_generated"]==0 and report["source_simulation_performed"] is report["training_performed"] is False
    assert report["formal_test_accessed"] is report["confidence_interval_or_significance"] is report["state_selection_enabled"] is False
    assert "not a generative expected count" in report["exposure_probability_sum_interpretation"]
    assert report["latent_evaluation"]=="means_not_integrated"
    with (output/"point_transition_by_condition.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==864 and {float(row["horizon_ms"]) for row in rows}=={20.,80.,500.}
    selected=next(row for row in rows if float(row["horizon_ms"])==20.)
    assert float(selected["posterior_event_nll"])==10.
    assert float(selected["posterior_observed_exposure_probability_sum"])==128.
    assert int(selected["eligible_tick_total"])==20000000
    assert float(selected["one_step_field_point_rate_mse_hz2"]) > float(selected["posterior_field_point_rate_mse_hz2"])
    assert not any("p_value" in row or "expected_count" in row for row in rows)
    with np.load(output/"observation_map.npz",allow_pickle=False) as mapping:
        np.testing.assert_array_equal(mapping["source_unit_ids"],campaign["source"].unit_ids)
        np.testing.assert_array_equal(mapping["unit_field_index"],campaign["source"].unit_field_index)
        assert mapping["source_dt_ms"]==.1 and mapping["bin_ms"]==5.
    with np.load(output/selected["diagnostic_array"],allow_pickle=False) as arrays:
        assert arrays["boundary_ready_in_ticks"].shape==(1,100000)
        assert {"posterior_field_point_rate_hz","one_step_field_exposure_probability_sum","boundary_variance",
                "observed_field_counts","observed_field_rate_hz"}.issubset(arrays.files)
        assert arrays["source_observation_map_sha256"].item()==report["source_observation_map_sha256"]
    for name,digest in report["artifacts"].items(): assert sha256_file(output/name)==digest


@pytest.mark.parametrize("failure",["analysis_count","config_seed","config_sampling","pfc_metadata","source","checkpoint","pair"])
def test_invalid_mode_source_and_checkpoint_reject_before_output(campaign,tmp_path,monkeypatch,failure):
    output=tmp_path/"rejected"
    config=CONFIG
    if failure=="analysis_count": campaign["analyses"]=campaign["analyses"][:1]
    elif failure in ("config_seed","config_sampling"):
        values=yaml.safe_load(CONFIG.read_text())
        values["training_seeds" if failure=="config_seed" else "samples_generated"]=[41,42] if failure=="config_seed" else 1
        config=tmp_path/"changed.yaml";config.write_text(yaml.safe_dump(values))
    elif failure=="pfc_metadata":
        campaign["acquisition"].write_text("{}")
    elif failure=="source":
        campaign["source"].binding=dict(dataset_sha256="foreign")
    elif failure=="checkpoint":
        monkeypatch.setattr(diagnostic,"_load_model",lambda *args:(_ for _ in ()).throw(ValueError("fitted model implementation changed")))
    else:
        monkeypatch.setattr(diagnostic,"_population_budget_pair",lambda *args:(_ for _ in ()).throw(ValueError("parent checkpoint/report/source lineage differs")))
    with pytest.raises(ValueError): execute(campaign,output,config)
    assert not output.exists() and not campaign["calls"]


def test_cli_rejects_one_analysis_before_output(tmp_path):
    output=tmp_path/"cli_rejected"
    result=subprocess.run([sys.executable,"-m","population_dynamics.population_posterior_diagnostic",
        "--analysis",str(tmp_path/"only"),"--acquisition",str(tmp_path/"acquisition"),"--fields",str(tmp_path/"fields"),
        "--config",str(CONFIG),"--output",str(output)],capture_output=True,text=True,check=False)
    assert result.returncode!=0 and "exactly two" in result.stderr
    assert not output.exists()

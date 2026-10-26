"""Source-ordered EAG evaluation, complete output replay and count-only semantics."""
import json

import numpy as np
import pytest
import torch
import yaml

from . import latent_energy_training as training
from . import v1_latent_energy_data as source_data
from . import v1_latent_energy_training as source_training
from . import v1_latent_energy_evaluation as evaluation
from .provenance import sha256_file
from .test_latent_energy_training import TinyCohort, setup


class SourceCohort(TinyCohort):
    def __init__(self,inputs,config,*,config_validator):
        super().__init__(inputs,None,config)
        self.config=config
        self.bin_ticks=5
        self.validation=[dict(case_id="validation_a",circuit_id="circuit_a",block_id="block_a",offset=9),
                         dict(case_id="validation_b",circuit_id="circuit_b",block_id="block_b",offset=10)]
        self.binding.update(circuit_inputs_sha256=sha256_file(inputs),
                            count_adapter=dict(control_features=["u"+str(i) for i in range(6)]))
        self.generated_draws=0

    def window(self,case):
        counts,controls=super().window(case)
        return counts,np.tile(controls,(1,6))

    def native_episode(self,case):
        parent=self
        counts,controls=self.window(case)
        second=case["case_id"]=="validation_b"
        class Episode:
            unit_ids=np.arange(900,916,2) if second else np.arange(100,108)
            field_ids=np.asarray(["source_a","source_b","empty"])
            unit_field_index=np.asarray([1]*4+[0]*4 if second else [0]*4+[1]*4)
            minimum_interval_ticks=np.full(8,2,dtype=np.int64)
            dt_ms=1.
            bin_ticks=5
            duration_ticks=160
            binding=dict(circuit_id=case["circuit_id"],block_id=case["block_id"],episode_id=case["case_id"])
            planned_controls=controls
            def window(self,start,stop):
                if start==0:
                    parent.generated_draws=0
                else:
                    assert parent.generated_draws==4,"future labels read before the full ensemble"
                return counts[start//5:stop//5],controls[start//5:stop//5]
        return Episode()


@pytest.fixture
def fitted(setup,tmp_path,monkeypatch):
    config,path=setup
    config["schema"]=training.V1_SCHEMA
    config["model"]["input_size"]=6
    config["energy"]["maximum_epochs"]=3
    config["sampling"].update(steps=2,guidance=1.,temperature=.7)
    path.write_text(yaml.safe_dump(config))
    spec=tmp_path/"inputs.json";spec.write_text("{}")
    monkeypatch.setattr(source_data,"V1LatentEnergyCohort",SourceCohort)
    monkeypatch.setattr(evaluation,"V1LatentEnergyCohort",SourceCohort)
    monkeypatch.setattr(evaluation,"validate_config",lambda *args:None)
    directory=tmp_path/"training"
    source_training.train(spec,path,directory,43)
    original=evaluation._load
    def load(*args):
        model,cohort,run=original(*args)
        sample=model.simulate
        def simulate(*values,**options):
            result=sample(*values,**options)
            cohort.generated_draws+=1
            return result
        model.simulate=simulate
        return model,cohort,run
    monkeypatch.setattr(evaluation,"_load",load)
    def controls(data,start,stop):
        return dict(known_controls=data.planned_controls[start//5:stop//5],
                    control_features=["u"+str(i) for i in range(6)])
    monkeypatch.setattr(evaluation.source,"source_controls",controls)
    forecast=dict(schema="v1_population_checkpoint_forecast_v1",evaluation_scope="original_validation_checkpoint",
        formal_evaluation=False,field_units="source_field_mean_unit_rate_hz",causal_history="all_observed_bins_before_origin",
        finite_ensemble_corrections=True,origins_ticks=[40,80,120],horizons_bins=[1,4,8],support_windows=[1,4],
        joint_samples=4,sampling_seed_base=41,random_seed_policy="exact_registered_stream_seeds")
    forecast_path=tmp_path/"forecast.yaml";forecast_path.write_text(yaml.safe_dump(forecast))
    return spec,directory,forecast_path


def test_source_eag_replays_native_ids_different_maps_and_all_unit_outputs(fitted,tmp_path):
    spec,directory,config=fitted
    output=tmp_path/"forecasts"
    # Exercise the real report schema, which keeps the binding in the checkpoint.
    assert "binding" not in json.loads((directory/"run.json").read_text())
    report=evaluation.evaluate(spec,directory,config,output)
    assert report["status"]==evaluation.EVALUATION_STATUS
    assert report["method"]=="eag" and report["observation_law"]=="softplus_poisson_counts"
    assert report["energy_presentations"]==18 and report["energy_optimizer_steps"]==5
    assert report["autoencoder_completed_epochs"]==2 and report["energy_completed_passes"]==3
    assert "committed_training_epochs" not in report
    assert report["score_rows"]==18 and len(report["origins"])==6
    assert report["generated_unit_count_values"]==1536
    assert not report["scientific_claim_qualified"] and not report["native_refractory_support_qualified"]
    for row in report["origins"]:
        with np.load(output/row["artifact"],allow_pickle=False) as arrays:
            expected=np.arange(900,916,2) if row["case_id"]=="validation_b" else np.arange(100,108)
            np.testing.assert_array_equal(arrays["unit_ids"],expected)
            assert arrays["generated_unit_counts"].shape==(4,8,8)
            assert arrays["generated_field_counts"].shape==(4,8,3)
            assert not arrays["generated_field_counts"][...,-1].any()
            assert "terminal_ready_in_ticks" not in arrays.files
            assert "initial_latent_states" not in arrays.files
    receipt=evaluation.replay(spec,directory,output,tmp_path/"replay.json")
    assert receipt["status"]=="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY"
    assert receipt["unit_values_consumed"]==1536 and len(receipt["origins"])==6


@pytest.mark.parametrize("name",["unit_ids","generated_unit_counts"])
def test_source_eag_replay_rejects_last_origin_tampering_after_rehash(fitted,tmp_path,name):
    spec,directory,config=fitted
    output=tmp_path/"forecasts"
    report=evaluation.evaluate(spec,directory,config,output)
    path=output/report["origins"][-1]["artifact"]
    with np.load(path,allow_pickle=False) as saved:
        arrays={key:saved[key].copy() for key in saved.files}
    arrays[name].flat[-1]+=1
    np.savez_compressed(path,**arrays)
    report["artifacts"][path.name]=sha256_file(path)
    (output/"evaluation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="ALL_UNIT_OR_SOURCE_MAP_MISMATCH"):
        evaluation.replay(spec,directory,output,tmp_path/"replay.json")
    assert not (tmp_path/"replay.json").exists()


def test_source_eag_load_rejects_report_seed_mismatch_before_model_creation(fitted,monkeypatch):
    spec,directory,_=fitted
    path=directory/"run.json"
    run=json.loads(path.read_text())
    assert run["training_seed"]==43
    run["training_seed"]=41
    path.write_text(json.dumps(run))
    monkeypatch.setattr(training,"_model",lambda *args:pytest.fail("model created before identity check"))
    with pytest.raises(ValueError,match="V1_EAG_COMPLETE_TRAINING_SOURCE_OR_BUDGET_CHANGED"):
        evaluation._load(spec,directory,torch.device("cpu"))


def test_source_eag_load_checks_original_admission_before_model_creation(fitted,monkeypatch):
    spec,directory,_=fitted
    def reject(inputs,config,*,config_validator):
        raise ValueError("original validation missing")
    monkeypatch.setattr(evaluation,"V1LatentEnergyCohort",reject)
    monkeypatch.setattr(training,"_model",lambda *args:pytest.fail("model created before admission"))
    with pytest.raises(ValueError,match="original validation missing"):
        evaluation._load(spec,directory,torch.device("cpu"))

from types import SimpleNamespace
import json
import numpy as np
import pytest
from .v1_glif_dataset import deliver, episode_from_source, BLOCK_PASS
from .test_v1_glif_network import cache_fixture


def test_delivery_rejects_incomplete_block(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","test")
    path=tmp_path/"coupled";path.mkdir()
    (path/"report.json").write_text(json.dumps(dict(status=BLOCK_PASS,passed_conditions=9,failed_conditions=1,cases=[])))
    with pytest.raises(ValueError,match="COMPLETE_BLOCK_REQUIRED"):
        deliver(tmp_path)
    assert not (tmp_path/"delivery.json").exists()


def test_export_rejects_changed_source_units_before_loading_controls(tmp_path):
    identity,original,data,_=cache_fixture()
    units=SimpleNamespace(unit_ids=np.array([0,1]))
    execution=dict(status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",units=2,
                   execution_clock=identity.clock,connection_rows=identity.connection_rows)
    arrays={"cortical_state.npz":data["cortical_state"],"cortical_spikes.npz":{},"input_events.npz":{}}
    arrays["cortical_state.npz"]["source_id"][0]=2
    with pytest.raises(ValueError,match="CELL_IDENTITY_DIFFERS:source_id"):
        episode_from_source(execution,arrays,original,identity,units,tmp_path,{},{})


def test_export_rejects_noninteger_counts_before_loading_controls(tmp_path):
    identity,original,data,_=cache_fixture()
    units=SimpleNamespace(unit_ids=np.array([0,1]))
    execution=dict(status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",units=2,
                   execution_clock=identity.clock,connection_rows=identity.connection_rows)
    arrays={"cortical_state.npz":data["cortical_state"],"cortical_spikes.npz":{},"input_events.npz":{}}
    arrays["cortical_state.npz"]["spike_count"]=np.array([1.,2.])
    with pytest.raises(ValueError,match="INVALID_TERMINAL_STATE"):
        episode_from_source(execution,arrays,original,identity,units,tmp_path,{},{})


def test_delivery_rejects_swapped_condition_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","test")
    path=tmp_path/"coupled";path.mkdir()
    cases=[dict(condition_index=i,status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",
                report=f"condition_{i:02d}/report.json") for i in range(10)]
    cases[0]["report"]=cases[1]["report"]
    (path/"report.json").write_text(json.dumps(dict(status=BLOCK_PASS,passed_conditions=10,failed_conditions=0,cases=cases)))
    with pytest.raises(ValueError,match="CONDITION_PATH_DIFFERS"):
        deliver(tmp_path)

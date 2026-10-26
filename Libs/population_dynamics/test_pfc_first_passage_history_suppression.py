"""History denominator, real replay dispatch, provenance and failure behavior."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from . import pfc_first_passage_history as history
from . import pfc_first_passage_history_summary as summary
from . import pfc_first_passage_reference as reference
from . import pfc_deterministic_replay as source
from . import pfc_native_drive as native
from .pfc_first_passage_suppression import replay_history_window
from .test_pfc_first_passage_suppression import fixture


def test_mixed_histories_keep_all_four_rows_and_use_complete_replay():
    config,case,initial,e,i=fixture()
    tape=reference.validate_tape(config,source.simulate(config,case,initial,e,i)["raw"])
    rows=[dict(sample_index=u,unit_index=u,window_start_tick=0,window_stop_tick=100) for u in range(3)]
    rows.append(dict(sample_index=3,unit_index=0,window_start_tick=0,window_stop_tick=20))
    rows=history.classify(config,tape,rows);before=deepcopy(rows)
    assert [r["category"] for r in rows]==["suppressed_network_reference_required",
        "conditional_integration_required","no_first_spike","refractory_covers_remaining_window"]
    called=[]
    def replay(row):
        called.append(row["sample_index"])
        actual,checks=replay_history_window(config,case,initial,tape,row,counts_E=e,counts_I=i)
        value=actual["suppressed_reference"]
        return (value["conditional_initial"],value["forcing"]),checks
    refs,diagnostic=history.extract(config,case,tape,rows,counts_E=e,counts_I=i,suppressed_replay=replay)
    assert len(rows)==4 and set(refs)=={0,1} and called==[0]
    assert diagnostic["source_window_denominator"]==4 and diagnostic["suppressed_network_replays"]==1
    assert all(all(row[k]==v for k,v in old.items()) for old,row in zip(before,rows))
    assert rows[0]["suppressed_reference_resolved"]
    assert not any("suppressed_reference_resolved" in row for row in rows[1:])


def test_failed_or_misidentified_replay_never_becomes_a_zero_or_replacement():
    config,case,initial,e,i=fixture();tape=reference.validate_tape(config,source.simulate(config,case,initial,e,i)["raw"])
    rows=history.classify(config,tape,[dict(sample_index=0,unit_index=0,window_start_tick=0,window_stop_tick=100)])
    before=deepcopy(rows)
    def failed(row):raise RuntimeError("intentional full-network replay failure")
    with pytest.raises(RuntimeError,match="intentional full-network replay failure"):
        history.extract(config,case,tape,rows,counts_E=e,counts_I=i,suppressed_replay=failed)
    assert rows==before
    def wrong(row):
        actual,checks=replay_history_window(config,case,initial,tape,row,counts_E=e,counts_I=i)
        result=actual["suppressed_reference"];result["conditional_initial"]["unit_index"]=1
        return (result["conditional_initial"],result["forcing"]),checks
    with pytest.raises(ValueError,match="SUPPRESSED_REFERENCE_IDENTITY_OR_FEEDBACK"):
        history.extract(config,case,tape,rows,counts_E=e,counts_I=i,suppressed_replay=wrong)
    assert rows==before


def build_source_fixture(root):
    # High current is an explicitly diagnostic fixture that makes the rare
    # multispike branch common; it is not a registered scientific condition.
    config,case,*_=fixture()
    config.update(duration_ms=100.,stimulus_duration_ms=100.,stimulus_strength_nA=50.)
    case.update(case_id="diagnostic_multispike_history",stimulus_strength_nA=50.)
    initial,drive=native.create_drive(config,case)
    raw=source.simulate(config,case,initial,drive=drive)["raw"]
    root.mkdir();(root/"config.yaml").write_text(yaml.safe_dump(config))
    (root/"report.json").write_text(json.dumps(dict(resolved_config=config,selected_case=case)))
    np.save(root/"initial_voltage.npy",initial);np.savez_compressed(root/"raw.npz",**raw)
    settings=dict(schema="pfc_first_passage_history_v1",source_rng_seed=95001,
        source=dict(archive="diagnostic_native_source",delivery_sha256="diagnostic_fixture"),
        case_id=case["case_id"],expected_units=40,dt_ms=.05,field_index=5,phase="baseline",window_ms=5.,
        sample_size=4,sampling_seed_components=[20261009,95001,5,1],
        importance_sampling=dict(samples=200,batch_size=100,proposal_multipliers=[1.,2.,4.,8.],
            mixture_probabilities=[.25]*4,maximum_witnesses=1,confidence_level=.95))
    return settings


def seal_fixture(root):
    (root/"report.json").write_text(json.dumps(dict(exit_code=0)))
    files=[]
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name!="delivery.json":
            data=path.read_bytes();files.append(dict(path=str(path.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    source_path=Path(history.__file__);sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
    files.append(dict(path="Libs/population_dynamics/pfc_first_passage_history.py",bytes=source_path.stat().st_size,sha256=sha))
    (root/"delivery.json").write_text(json.dumps(dict(verified=True,status="COMPLETE_PFC_FIRST_PASSAGE_HISTORY_EXECUTE",files=files)))
    return sha


def test_native_history_pipeline_integration_and_bound_reader(tmp_path):
    source_dir=tmp_path/"source";settings=build_source_fixture(source_dir)
    root=tmp_path/"delivery";root.mkdir()
    result=history.run(settings,source_dir,root/"validation",suppressed_replay=replay_history_window)
    rows=json.loads((root/"validation/integrated_frame.json").read_text())
    assert result["sample_denominator"]==len(rows)==4
    assert result["reconstruction"]["suppressed_network_replays"]>0
    assert all(row.get("suppressed_reference_resolved") for row in rows if row["observed_count"]>=2)
    assert result["scientific_fidelity_qualified"] is False and result["production_episodes"]==0
    sha=seal_fixture(root)
    options=dict(settings,frame_size=result["frame"]["frame_size"],importance_samples=200,
                 source_history_implementation_sha256=sha)
    accepted=summary.read_block(root,95001,options,settings["source"])
    assert accepted["sample_denominator"]==4 and accepted["sample_mean"]==result["sample_mean"]
    selected=next(row for row in rows if row["observed_count"]>=2)
    report_path=root/"validation"/selected["suppressed_reference_report"]
    receipt=json.loads(report_path.read_text());receipt["unit_index"]+=1;report_path.write_text(json.dumps(receipt))
    seal_fixture(root)
    with pytest.raises(ValueError,match="SUPPRESSED_REFERENCE_BINDING"):
        summary.read_block(root,95001,options,settings["source"])


def test_history_run_saves_replay_failure_and_retains_classified_frame(tmp_path):
    source_dir=tmp_path/"source";settings=build_source_fixture(source_dir);output=tmp_path/"failed"
    def fail(*args,**kwargs):raise RuntimeError("intentional native replay failure")
    with pytest.raises(RuntimeError,match="intentional native replay failure"):
        history.run(settings,source_dir,output,suppressed_replay=fail)
    report=json.loads((output/"report.json").read_text())
    rows=json.loads((output/"classified_frame.json").read_text())
    assert report["status"]=="FAILED_PFC_FIRST_PASSAGE_HISTORY" and len(rows)==4
    assert not (output/"integrated_frame.json").exists()
    failures=list(output.glob("suppressed_sample_*/report.json"));assert len(failures)==1
    assert json.loads(failures[0].read_text())["status"]=="FAILED_PFC_HISTORY_SUPPRESSED_REFERENCE"

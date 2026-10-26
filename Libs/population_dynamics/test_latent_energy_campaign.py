"""Storage admission uses normalized lengths from accepted native manifests."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from Libs.scripts import run_latent_energy_campaign as campaign

def fixture(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[2]
    helper=campaign.load_script(root/"Libs/scripts/run_population_trajectory_campaign.py","energy_storage_helper")
    monkeypatch.setattr(helper,"COHORT","native")
    native=tmp_path/"native"; native.mkdir()
    (native/"first.bin").write_bytes(b"first")
    (native/"second.bin").write_bytes(b"second_payload")
    (native/"delivery.json").write_text(json.dumps(dict(files=[
        dict(path=name,destination_bytes=(native/name).stat().st_size,sha256=helper.digest(native/name)) for name in ("first.bin","second.bin")])))
    work=tmp_path/"work"; work.mkdir()
    config=dict(autoencoder=dict(maximum_epochs=260,validation_every=10),
                energy=dict(maximum_epochs=4000,validation_every=100))
    return helper,work,config

def test_native_manifest_destination_bytes_are_normalized(tmp_path,monkeypatch):
    helper,work,config=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign.shutil,"disk_usage",lambda _:SimpleNamespace(free=100*1024**3))
    campaign.storage_reserve(helper,tmp_path,work,dict(serialized_full_model_bytes=1000),config)
    report=json.loads((work/"storage_reserve.json").read_text())
    assert report["input_upper_bound_bytes"]==19
    assert report["output_reserve_bytes"]==72*1000+3*1024**3

def test_insufficient_storage_rejects_before_input_staging(tmp_path,monkeypatch):
    helper,work,config=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign.shutil,"disk_usage",lambda _:SimpleNamespace(free=1024))
    with pytest.raises(helper.ResourceBlocked):
        campaign.storage_reserve(helper,tmp_path,work,dict(serialized_full_model_bytes=1000),config)
    assert json.loads((work/"storage_reserve.json").read_text())["status"]=="BLOCKED"

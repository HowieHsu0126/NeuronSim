"""Original V1 split admission and the real shared complete-episode trainer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import pytest
import torch
import yaml

from . import population_trajectory_training as training
from . import v1_population_training_data as consumer
from .test_v1_renewal_data import fixture as source_fixture


def configuration():
    path=Path(__file__).resolve().parents[1]/"configs/neural_population_v1_source_training.yaml"
    config=yaml.safe_load(path.read_text())
    config.update(fit_circuit_ids=["v1_fit"],validation_circuit_ids=["v1_validation"],
                  latent_channels=2,encoder_size=3,unit_chunk_size=2,bin_ticks=5,
                  maximum_epochs=2,early_stopping_patience=2,validation_draws=2,cpu_threads=1,device="cpu")
    return config


def inputs(root, *, validation_split="validation"):
    catalog=None; parameters={}
    for folder,split,circuit,episode in (("fit_one","train","v1_fit","fit_1"),
                                         ("fit_two","train","v1_fit","fit_2"),
                                         ("held_out",validation_split,"v1_validation","validation_1")):
        local=root/folder;local.mkdir()
        data,native=source_fixture(local,circuit_id=circuit,episode_id=episode,split=split)
        if catalog is None:
            catalog=deepcopy(data.catalog);catalog.update(circuits={},shards=[],complete_episodes=0)
        for cid,row in data.catalog["circuits"].items():
            item=deepcopy(row);item["path"]=str((data.root/item["path"]).relative_to(root))
            catalog["circuits"].setdefault(cid,item)
        for row in data.catalog["shards"]:
            item=deepcopy(row);item["path"]=str((data.root/item["path"]).relative_to(root))
            catalog["shards"].append(item)
        catalog["complete_episodes"]+=data.catalog["complete_episodes"]
        parameters.setdefault(circuit,str(native.relative_to(root)))
    # Disposable admitted-catalog fixture; no source qualification is claimed.
    (root/"dataset.json").write_text(json.dumps(catalog))
    path=root/"inputs.json"
    path.write_text(json.dumps(dict(schema=consumer.INPUT_SCHEMA,dataset=".",native_parameters=parameters)))
    return path


@pytest.mark.parametrize("split",["train","calibration","test"])
def test_absent_original_validation_rejects_before_source_or_parameter_access(tmp_path,split):
    path=inputs(tmp_path,validation_split=split)
    with patch.object(consumer,"V1RenewalEpisode",side_effect=AssertionError("array read")), \
         patch.object(consumer,"sha256_file",side_effect=AssertionError("parameter read")):
        with pytest.raises(ValueError,match="ORIGINAL_VALIDATION_CIRCUITS_MISSING_OR_MISASSIGNED"):
            consumer.V1PopulationTrainingCohort(path,configuration())


def test_overlapping_circuits_reject_before_input_access(tmp_path):
    config=configuration();config["validation_circuit_ids"]=config["fit_circuit_ids"]
    with pytest.raises(ValueError,match="SPLIT_LEAKAGE"):
        consumer.V1PopulationTrainingCohort(tmp_path/"must_not_be_read.json",config)


def test_source_admission_at_construction_occurs_once_per_circuit(tmp_path,monkeypatch):
    path=inputs(tmp_path);real=consumer.V1RenewalEpisode;consumed=[]
    def adapter(data,episode_id,**kwargs):
        consumed.append(episode_id);return real(data,episode_id,**kwargs)
    monkeypatch.setattr(consumer,"V1RenewalEpisode",adapter)
    cohort=consumer.V1PopulationTrainingCohort(path,configuration())
    assert consumed==["fit_1","validation_1"]
    assert len(cohort.fit)==2 and len(cohort.validation)==1
    cohort.window(cohort.fit[1])
    assert consumed[-1]=="fit_2"
    assert cohort.active_episode.episode_id=="fit_2"


def test_parameter_map_does_not_admit_unselected_circuits(tmp_path):
    path=inputs(tmp_path);spec=json.loads(path.read_text());spec["native_parameters"]["test_circuit"]="never_read"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="EXACT_SELECTED_PARAMETER_MAP"):
        consumer.V1PopulationTrainingCohort(path,configuration())


@pytest.mark.parametrize("seed",[41,42,43])
def test_real_v1_trainer_uses_complete_episodes_and_registered_rng_streams(tmp_path,seed):
    path=inputs(tmp_path);config=configuration();config_path=tmp_path/"training.yaml"
    config_path.write_text(yaml.safe_dump(config));out=tmp_path/"training"
    report=training.train(path,None,config_path,out,seed,stop_after_epochs=1)
    assert report["fitting_episodes"]==2 and report["validation_episodes"]==1
    assert report["complete_episode_updates"] and report["completed_epochs"]==1
    assert report["fitting_circuits"]==1 and report["validation_circuits"]==1
    assert not report["formal_training_completed"] and not report["test_outcomes_accessed"]
    checkpoint=torch.load(out/"latest.pt",weights_only=True)
    assert all(int(value["step"])==2 for value in checkpoint["optimizer"]["state"].values())
    assert report["input_family"]=="v1_qualified_source_episodes"
    assert "source_population_training.py" in report["trainer_sources"]
    assert "v1_population_training_data.py" in report["trainer_sources"]
    from .provenance import sha256_file
    assert report["binding"]["loader_implementation_sha256"] == sha256_file(
        Path(__file__).with_name("v1_renewal_data.py"))
    for role in ("training","validation"):
        assert training.shared.registered_variational_seed(config,seed,role) in (41,42,43)


def assert_same(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_same(a,b)
    else:assert left==right


def test_v1_resume_preserves_committed_optimizer_rng_and_validation(tmp_path):
    path=inputs(tmp_path);config_path=tmp_path/"training.yaml";config_path.write_text(yaml.safe_dump(configuration()))
    continuous,split=tmp_path/"continuous",tmp_path/"split"
    training.train(path,None,config_path,continuous,42,stop_after_epochs=2)
    training.train(path,None,config_path,split,42,stop_after_epochs=1)
    training.train(path,None,config_path,split,42,resume=True,stop_after_epochs=1)
    a,b=(torch.load(p/"latest.pt",weights_only=True) for p in (continuous,split))
    for key in ("model","optimizer","variational_rng","history","next_epoch","best","best_epoch","initialization"):
        assert_same(a[key],b[key])


def test_v1_resume_rejects_changed_source_buffers(tmp_path):
    path=inputs(tmp_path);config_path=tmp_path/"training.yaml";config_path.write_text(yaml.safe_dump(configuration()))
    out=tmp_path/"training";training.train(path,None,config_path,out,43,stop_after_epochs=1)
    checkpoint=torch.load(out/"latest.pt",weights_only=True);checkpoint["model"]["unit_features"][0,0]+=1
    torch.save(checkpoint,out/"latest.pt")
    from .provenance import sha256_file
    report=json.loads((out/"run.json").read_text());report["latest_checkpoint_sha256"]=sha256_file(out/"latest.pt")
    (out/"run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="source buffers changed"):
        training.train(path,None,config_path,out,43,resume=True)


def test_inspect_inputs_uses_only_catalog_before_source_admission(tmp_path):
    path = inputs(tmp_path)
    with patch.object(consumer, "V1RenewalEpisode", side_effect=AssertionError("array read")), \
         patch.object(consumer, "sha256_file", side_effect=AssertionError("parameter read")):
        inspected = consumer.inspect_inputs(path, configuration())
    assert inspected.raw == path.read_bytes()
    assert inspected.spec == json.loads(inspected.raw)
    assert inspected.data.root == tmp_path.resolve()
    assert set(inspected.parameters) == {"v1_fit", "v1_validation"}
    assert inspected.expected_parameter_sha256 == {}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def staged_inputs(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    original_input = inputs(original)
    local_parent = Path("/tmp/neuronsim")
    local_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neuronsim-v1-relocation-", dir=local_parent) as temporary:
        local = Path(temporary)
        dataset = local / "dataset"
        shutil.copytree(original, dataset)
        copied_input = local / "inputs.json"
        copied_input.write_bytes(original_input.read_bytes())
        spec = json.loads(copied_input.read_bytes())
        mapping = dict(schema="v1_population_execution_path_map_v1",
            inputs_sha256=_digest(copied_input),
            dataset=dict(path=str(dataset), sha256=_digest(dataset / "dataset.json")),
            native_parameters={cid: dict(path=str(dataset / value), sha256=_digest(dataset / value))
                               for cid, value in spec["native_parameters"].items()})
        map_path = local / "execution_path_map.json"
        map_path.write_text(json.dumps(mapping))
        monkeypatch.setenv("NEURONSIM_EXECUTION_PATH_MAP", str(map_path))
        yield dict(original=original, original_input=original_input, copied_input=copied_input,
                   dataset=dataset, mapping=mapping, map_path=map_path)


def test_relocation_preserves_bindings_when_original_relative_paths_disappear(staged_inputs, monkeypatch):
    staged = staged_inputs
    with monkeypatch.context() as original_environment:
        original_environment.delenv("NEURONSIM_EXECUTION_PATH_MAP")
        before = consumer.V1PopulationTrainingCohort(staged["original_input"], configuration())
    original_bytes = staged["original_input"].read_bytes()
    staged["original"].rename(staged["original"].with_name("unavailable_original"))
    inspected = consumer.inspect_inputs(staged["copied_input"], configuration())
    after = consumer.V1PopulationTrainingCohort(staged["copied_input"], configuration())
    assert inspected.raw == original_bytes
    assert inspected.data.catalog == before.data.catalog
    assert inspected.data.root == staged["dataset"]
    assert inspected.expected_parameter_sha256 == before.parameter_digests
    assert after.binding == before.binding
    assert after.source_dimensions == before.source_dimensions
    assert all(value["units"] == after.data.catalog["circuits"][cid]["units"]
               and value["fields"] > 0 and value["source_dt_ms"] > 0
               for cid, value in after.source_dimensions.items())
    assert {case["split"] for case in after.fit} == {"train"}
    assert {case["split"] for case in after.validation} == {"validation"}


def test_mapped_missing_validation_rejects_before_parameter_filesystem_or_arrays(staged_inputs):
    staged = staged_inputs
    catalog_path = staged["dataset"] / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    for shard in catalog["shards"]:
        for row in shard["episodes"]:
            if row["split"] == "validation":
                row["split"] = "train"
    catalog_path.write_text(json.dumps(catalog))
    staged["mapping"]["dataset"]["sha256"] = _digest(catalog_path)
    for row in staged["mapping"]["native_parameters"].values():
        row["path"] = "/tmp/neuronsim/unavailable-v1-parameter"
    staged["map_path"].write_text(json.dumps(staged["mapping"]))
    real_target, real_digest = consumer._execution_target, consumer.sha256_file

    def dataset_target_only(row, *, directory):
        assert directory, "parameter filesystem access before validation admission"
        return real_target(row, directory=directory)

    def dataset_digest_only(path):
        assert Path(path) == catalog_path, "parameter bytes read before validation admission"
        return real_digest(path)

    with patch.object(consumer, "_execution_target", side_effect=dataset_target_only), \
         patch.object(consumer, "sha256_file", side_effect=dataset_digest_only), \
         patch.object(consumer, "V1RenewalEpisode", side_effect=AssertionError("array read")), \
         patch("h5py.File", side_effect=AssertionError("HDF5 read")):
        with pytest.raises(ValueError, match="ORIGINAL_VALIDATION_CIRCUITS_MISSING_OR_MISASSIGNED"):
            consumer.V1PopulationTrainingCohort(staged["copied_input"], configuration())


@pytest.mark.parametrize("kind,match", [
    ("input", "PATH_MAP_INPUT_CHANGED"),
    ("catalog", "PATH_MAP_DATASET_METADATA_CHANGED"),
    ("parameter", "PATH_MAP_PARAMETER_CHANGED"),
    ("unknown", "PATH_MAP_SCHEMA_REQUIRED"),
    ("missing", "PATH_MAP_SCHEMA_REQUIRED"),
    ("dataset_field", "PATH_MAP_SCHEMA_REQUIRED"),
    ("parameter_field", "PATH_MAP_SCHEMA_REQUIRED"),
    ("extra_circuit", "PATH_MAP_EXACT_SELECTED_PARAMETERS_REQUIRED"),
    ("missing_circuit", "PATH_MAP_EXACT_SELECTED_PARAMETERS_REQUIRED"),
    ("outside_local", "PATH_MAP_LOCAL_TARGET_REQUIRED"),
])
def test_execution_map_changes_reject_before_source_admission(staged_inputs, kind, match):
    staged = staged_inputs
    mapping = staged["mapping"]
    if kind == "input": mapping["inputs_sha256"] = "0" * 64
    elif kind == "catalog": mapping["dataset"]["sha256"] = "0" * 64
    elif kind == "parameter": mapping["native_parameters"]["v1_fit"]["sha256"] = "0" * 64
    elif kind == "unknown": mapping["extra"] = True
    elif kind == "missing": mapping.pop("native_parameters")
    elif kind == "dataset_field": mapping["dataset"]["extra"] = True
    elif kind == "parameter_field": mapping["native_parameters"]["v1_fit"]["extra"] = True
    elif kind == "extra_circuit": mapping["native_parameters"]["test_circuit"] = mapping["native_parameters"]["v1_fit"]
    elif kind == "missing_circuit": mapping["native_parameters"].pop("v1_validation")
    elif kind == "outside_local": mapping["dataset"]["path"] = "/tmp"
    staged["map_path"].write_text(json.dumps(mapping))
    with patch.object(consumer, "V1RenewalEpisode", side_effect=AssertionError("array read")):
        with pytest.raises(ValueError, match=match):
            consumer.V1PopulationTrainingCohort(staged["copied_input"], configuration())


@pytest.mark.parametrize("key", ["inputs_sha256", "path"])
def test_execution_map_duplicate_top_or_nested_key_rejects(staged_inputs, key):
    staged = staged_inputs
    encoded = json.dumps(staged["mapping"])
    token = json.dumps(key) + ": "
    encoded = encoded.replace(token, token + '"duplicate", ' + token, 1)
    staged["map_path"].write_text(encoded)
    with pytest.raises(ValueError, match="PATH_MAP_DUPLICATE_KEY"):
        consumer.inspect_inputs(staged["copied_input"], configuration())

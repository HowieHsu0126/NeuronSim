"""Reference pilot identity, source-clock interval and full-unit count contracts."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from .benchmark_tasks import task_protocol
from .circuit_campaign import campaign_config
from .circuit_campaign import episode_identities
from .benchmark_sampling import prepare_pfc_block
from .benchmark_dataset import read_episode
from .circuit_data import CircuitDataset
from .pfc_kernel_parameters import circuit_provenance
from .pfc_native_drive import SOURCE_ENTRY_SHA256
from .provenance import sha256_file
from .pfc_reference_quality import quality_config, prepare, load_task, event_measurements, phase_edges, source_task
from . import pfc_reference_quality as quality

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROJECT = Path(__file__).resolve().parents[2]
SETTINGS = CONFIGS / "neural_population_pfc_reference_source_qualification.yaml"


def test_reference_pilot_retains_all_five_circuits_blocks_and_conditions(tmp_path):
    qualifications = {str(circuit): str(tmp_path / ("qualification_"+str(circuit))) for circuit in (23,63,74,86,52)}
    plan = prepare(PROJECT, SETTINGS, CONFIGS / "neural_population_pfc_100k_source_dt_0p0125ms.yaml",
                   qualifications, tmp_path / "prepared")
    assert len(plan["rows"]) == 150 and len(plan["block_reports"]) == 15
    assert plan["production_episodes"] == plan["complete_episodes"] == 0
    campaign = campaign_config(CONFIGS / "neural_population_dynamics.yaml")
    assert campaign["circuit_split_counts"] == dict(train=70,validation=10,calibration=10,test=10)
    for index, row in enumerate(plan["rows"]):
        _, loaded, path = load_task(tmp_path / "prepared", index)
        assert loaded == row
        config = yaml.safe_load(path.read_text())
        identity = row["identity"]
        assert config["dt_ms"] == 0.0125 and config["duration_ms"] == 5000.
        assert identity["block_index"] in (3,4,5)
        assert config["initial_rng_seed"] == identity["initial_state_seed"]
        assert config["external_noise_seed"] == identity["background_seed"]
        assert config["benchmark_episode_candidate"]["production_eligible"] is False
        assert config["benchmark_circuit_candidate"]["intended_split"] == identity["split"]
    assert len({row["identity"]["episode_id"] for row in plan["rows"]}) == 150
    with pytest.raises(ValueError, match="RECIPE_CHANGED"):
        first = tmp_path / "prepared" / plan["rows"][0]["config_path"]
        first.write_text(first.read_text()+"\n")
        load_task(tmp_path / "prepared", 0)


def test_source_float_schedule_and_full_phase_support_are_preserved():
    protocol = task_protocol("pfc_working_memory", pfc_dt_ms=.0125)
    edges = phase_edges(protocol)
    assert edges[0] == 0 and edges[-1] == 400000
    assert 8000 in edges and 24080 in edges  # Actual cue active through operation300.
    assert 32000 in edges and 192000 in edges
    assert len(edges) == len(set(edges))


def test_event_readout_conserves_all_units_and_preserves_undefined_angles():
    config = dict(excitatory_neurons=16,inhibitory_neurons=4,dt_ms=.0125,duration_ms=5000.)
    ticks = np.asarray([0,0,7999,8000,24079,24080,32000,192000,399999,400000],dtype=np.int64)
    ids = np.asarray([0,16,2,5,3,16,7,8,19,4],dtype=np.int64)
    episode = SimpleNamespace(spike_ticks=ticks,spike_unit_indices=ids,duration_ticks=400000)
    phases, arrays, temporal = event_measurements(episode,config)
    assert arrays["phase_unit_counts"].shape[1] == 20
    assert arrays["population_field_counts"].shape == (1000,17)
    assert int(arrays["phase_unit_counts"].sum()) == int(arrays["population_field_counts"].sum()) == 9
    assert sum(row["excitatory_spikes"]+row["inhibitory_spikes"] for row in temporal) == 9
    dense = np.zeros((1000,20),dtype=np.int64)
    for tick,unit in zip(ticks[:-1],ids[:-1]):dense[tick//400,unit]+=1
    np.testing.assert_array_equal(arrays["temporal_unit_mean_counts"],dense.mean(axis=0))
    np.testing.assert_allclose(arrays["temporal_unit_variance_counts"],dense.var(axis=0,ddof=1),rtol=0,atol=1e-18)
    silent = [row for row in phases if row["excitatory_spikes"]==0]
    assert silent and all(row["decoded_angle_deg"] is None and not row["direction_defined"] for row in silent)
    assert all(row["direction_missing_reason"] for row in silent)
    # The source terminal event remains in the episode but never enters half-open statistics.
    assert episode.spike_ticks[-1] == 400000


def test_source_quality_plan_cannot_change_scale_or_release_production(tmp_path):
    config = yaml.safe_load(SETTINGS.read_text())
    for key,value in [("production_eligible",True),("coupled_core_neurons",1000),("initial_noise_block_indices",[3])]:
        changed = dict(config,**{key:value})
        path=tmp_path/(key+".yaml");path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError,match="FIXED_PILOT_REQUIRED"):quality_config(path)


@pytest.mark.parametrize("node", ["cpu03", "cpu04", "cpu05", "cpu06"])
def test_source_dispatch_uses_allocated_cpu_node_and_preserves_identity(tmp_path, monkeypatch, node):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("SLURMD_NODENAME", node)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    row=dict(qualification=str(tmp_path/"qualification"), identity=dict(episode_id="fixture_episode"))
    monkeypatch.setattr(quality,"load_task",lambda *args: ({},row,tmp_path/"recipe.yaml"))
    calls=[]
    def run(command, *, cwd, env):
        calls.append((command,cwd,env))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(quality.subprocess,"run",run)
    assert source_task(tmp_path,tmp_path/"preparation",0,tmp_path/"output")==0
    assert len(calls)==1
    assert calls[0][2]["NEURONSIM_PFC_SOURCE_CASE_ID"]=="fixture_episode"
    assert calls[0][2]["CUDA_VISIBLE_DEVICES"]==""
    recovered = tmp_path / "recovered_exact_tuple"
    assert source_task(tmp_path,tmp_path/"preparation",0,tmp_path/"output",qualification=recovered)==0
    assert calls[-1][2]["NEURONSIM_PFC_SOURCE_QUALIFICATION"] == str(recovered.resolve())


def test_source_dispatch_rejects_control_node_or_gpu_before_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","fixture")
    monkeypatch.setenv("SLURMD_NODENAME","mgmt2")
    monkeypatch.setattr(quality,"load_task",lambda *args: pytest.fail("invalid allocation read source recipe"))
    with pytest.raises(ValueError,match="ALLOCATED_CPU_REQUIRED"):
        source_task(tmp_path,tmp_path,0,tmp_path)


def test_readout_passes_durable_origins_to_export_and_top_report(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    durable = tmp_path / "durable"
    for archive in (staged, durable):
        (archive / "source_run").mkdir(parents=True)
        (archive / "delivery.json").write_text('{}\n')
    native = dict(outputs={"initial_voltage.npy": {"sha256": "initial"},
                           "final_rng.npz": {"sha256": "rng"}}, driver={}, validations={})
    (staged / "source_run/report.json").write_text(json.dumps(native))
    origins = tmp_path / "origins.json"
    origins.write_text(json.dumps([dict(staged=str(staged), archive=str(durable),
                                       case_id="episode", delivery_sha256="unused_mock")]))
    identity = dict(episode_id="episode", circuit_index=86, block_index=3)
    row = dict(identity=identity, condition=dict(condition_id="cue_none"), config_sha256="recipe")
    config = tmp_path / "recipe.yaml"; config.write_text('{}\n')
    monkeypatch.setattr(quality, "load_task", lambda *args: ({}, row, config))
    seen = []
    def exporter(sources, recipe, output, *, input_origins):
        seen.append((sources, input_origins))
        return dict(status="EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET", complete_episodes=1,
                    dataset_catalog_sha256="catalog")
    monkeypatch.setattr(quality, "export", exporter)
    monkeypatch.setattr(quality, "CircuitDataset", lambda *args, **kwargs: SimpleNamespace(episodes={"episode": {}}))
    episode = SimpleNamespace(privileged={"initial_voltage_mV": np.array([-55.])},
                              duration_ticks=400000, dt_ms=.0125,
                              spike_ticks=np.array([0], dtype=np.int64))
    monkeypatch.setattr(quality, "read_episode", lambda *args: episode)
    monkeypatch.setattr(quality, "event_measurements", lambda *args: ([dict(value=1)], {}, [dict(value=1)]))
    result = quality.readout(tmp_path, 0, staged, tmp_path / "readout", input_origins=origins)
    assert seen == [([staged], origins)]
    assert result["source_archive"] == str(durable.resolve())
    assert result["source_qualification_granted"] is False and result["production_episodes"] == 0


def development_block_recipe(root):
    """Use the real compiler for a nonpilot circuit and independent block."""
    campaign_path = CONFIGS / "neural_population_dynamics.yaml"
    campaign = campaign_config(campaign_path)
    identity = next(row for row in episode_identities(campaign, "pfc_working_memory")
                    if (row["circuit_index"], row["block_index"], row["condition_index"]) == (0, 6, 0))
    base = yaml.safe_load((CONFIGS / "neural_population_pfc_100k_source_dt_0p0125ms.yaml").read_text())
    base["benchmark_circuit_candidate"] = dict(
        status="PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE", scenario_id="pfc_working_memory",
        circuit_id=identity["circuit_id"], circuit_index=0, intended_split="train",
        parameters=dict(sigma_weight_profile=20., Jpos_excit2excit=1.6),
        construction_scope="development", production_eligible=False,
        **circuit_provenance(base, SOURCE_ENTRY_SHA256))
    root.mkdir(parents=True)
    base_path = root / "circuit.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    plan = prepare_pfc_block(campaign_path, base_path, 0, 6, root / "block")
    row = plan["conditions"][0]
    path = root / "block" / row["config_path"]
    return path, row


@pytest.fixture(scope="module")
def development_readout_archive(tmp_path_factory):
    # Full-unit arrays are synthetic archival fixtures. No source is executed.
    from .test_pfc_benchmark_production import archive
    root = tmp_path_factory.mktemp("pfc-development-readout")
    path, row = development_block_recipe(root / "recipes")
    source = archive(root / "native", path, None, row["identity"])
    config = yaml.safe_load(path.read_text())
    dt = config["dt_ms"] * .001
    duration = round(config["duration_ms"] / config["dt_ms"])
    # The existing reusable archive fixture uses its historical .05ms clock.
    # Bind this fixture's observed event/current metadata to the reference clock.
    with np.load(source / "source_run/raw.npz") as stored:
        raw = {name: stored[name] for name in stored.files}
    raw["spike_seconds"] = raw["spike_ticks"] * dt
    np.savez(source / "source_run/raw.npz", **raw)
    with np.load(source / "source_run/applied_current_observation.npz") as stored:
        currents = {name: stored[name] for name in stored.files}
    currents["segment_start_ticks"] = np.asarray([0, 8000, 24080], dtype=np.int64)
    currents["segment_stop_ticks"] = np.asarray([8000, 24080, duration], dtype=np.int64)
    np.savez(source / "source_run/applied_current_observation.npz", **currents)
    from . import pfc_source_execution as executor
    metadata = executor.executor._applied_current_metadata(80000, dt, duration, 80)
    (source / "source_run/applied_current_metadata.json").write_text(json.dumps(metadata))
    report_path = source / "source_run/report.json"
    report = json.loads(report_path.read_text())
    report["driver"]["ticks_generated"] = duration
    report["completed_duration_ticks"] = duration
    report["inputs_used"]["ticks_consumed"] = duration
    for population, units in (("E", 80000), ("I", 20000)):
        report["inputs_used"][population]["values"] = units * duration
    for name, binding in report["outputs"].items():
        payload = source / "source_run" / name
        binding.update(bytes=payload.stat().st_size, sha256=sha256_file(payload))
    report_path.write_text(json.dumps(report))
    delivery_path = source / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    for binding in delivery["files"]:
        payload = source / binding["path"]
        binding.update(bytes=payload.stat().st_size, sha256=sha256_file(payload))
    delivery_path.write_text(json.dumps(delivery))
    return path, row, source


def test_development_episode_real_cli_preserves_complete_units_and_undefined_conditions(
    development_readout_archive, tmp_path,
):
    config, row, source = development_readout_archive
    output = tmp_path / "readout"
    command = [sys.executable, "-B", "-m", "population_dynamics.pfc_reference_quality",
               "readout-episode", "--config", str(config), "--source", str(source), "--output", str(output)]
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "Libs"), PYTHONDONTWRITEBYTECODE="1")

    child = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)

    assert child.returncode == 0, child.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["identity"] == row["identity"] and report["condition"] == row["condition"]
    assert report["core_units"] == 100000 and report["duration_ticks"] == 400000
    assert report["dt_ms"] == .0125 and report["temporal_bins"] == 1000
    assert report["source_delivery_sha256"] == sha256_file(source / "delivery.json")
    assert report["source_report_sha256"] == sha256_file(source / "source_run/report.json")
    assert report["source_qualification_granted"] is report["scientific_claim_qualified"] is False
    assert report["production_episodes"] == 0
    assert "one_episode_supplies_no_between_block_variance" in report["block_variance_sample_axis"]
    with np.load(output / "unit_population_statistics.npz") as arrays:
        assert arrays["phase_unit_counts"].shape[1] == 100000
        assert arrays["population_field_counts"].shape == (1000, 17)
        assert int(arrays["phase_unit_counts"].sum()) == report["events"] == 3
    dataset = CircuitDataset(output / "export/dataset", splits=("development",))
    episode = read_episode(dataset, dataset.episodes[row["identity"]["episode_id"]])
    assert episode.privileged["delivered_current__current_A"].shape == (3, 80000)
    assert episode.privileged["initial_voltage_mV"].shape == (100000,)
    with (output / "phase_responses.csv").open(newline="") as stream:
        phases = list(csv.DictReader(stream))
    undefined = [item for item in phases if item["direction_defined"] == "False"]
    assert undefined and all(item["direction_missing_reason"] for item in undefined)
    assert all(item["decoded_angle_deg"] == "" for item in undefined)


def test_fixed150_readout_uses_shared_episode_measurements(development_readout_archive, tmp_path, monkeypatch):
    config, row, source = development_readout_archive
    bound = dict(row, config_sha256=sha256_file(config))
    monkeypatch.setattr(quality, "load_task", lambda *args: ({}, bound, config))

    result = quality.readout(tmp_path, 0, source, tmp_path / "readout")

    assert result["identity"] == row["identity"] and result["core_units"] == 100000
    assert result["block_variance_sample_axis"] == "future_aggregate_of_blocks3_4_5_at_matched_times_within_circuit"
    assert result["source_report_sha256"] == sha256_file(source / "source_run/report.json")


@pytest.mark.parametrize("invalid", ["clock", "scale", "production", "identity", "compiler", "promoted"])
def test_development_episode_rejects_invalid_recipe_before_export(tmp_path, monkeypatch, invalid):
    path, _ = development_block_recipe(tmp_path / "recipes")
    config = yaml.safe_load(path.read_text())
    if invalid == "clock":
        config["dt_ms"] = .05
    elif invalid == "scale":
        config["excitatory_neurons"] = 16
    elif invalid == "production":
        config["benchmark_episode_candidate"]["production_eligible"] = True
    elif invalid == "identity":
        config["benchmark_episode_candidate"]["identity"]["episode_id"] = "wrong-episode"
    elif invalid == "compiler":
        config["cases"][0]["stimulus_strength_nA"] = .1
    else:
        config["scientific_claim_qualified"] = True
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(quality, "export", lambda *args, **kwargs: pytest.fail("invalid recipe reached exporter"))
    output = tmp_path / "readout"

    with pytest.raises(ValueError):
        quality.readout_episode(path, tmp_path / "missing-source", output)

    assert not output.exists()


def test_development_episode_rejects_missing_source_and_source_clock_mismatch(
    development_readout_archive, tmp_path,
):
    config_path, _, source = development_readout_archive
    with pytest.raises(ValueError, match="COMPLETE_SOURCE_REQUIRED"):
        quality.readout_episode(config_path, tmp_path / "missing-source", tmp_path / "missing-readout")
    assert not (tmp_path / "missing-readout").exists()
    config = yaml.safe_load(config_path.read_text())
    config["sigma_weight_profile"] = 21.
    circuit = config["benchmark_circuit_candidate"]
    circuit["parameters"]["sigma_weight_profile"] = 21.
    circuit.update(circuit_provenance(config, SOURCE_ENTRY_SHA256))
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="canonical configuration"):
        quality.readout_episode(changed, source, tmp_path / "changed-readout")
    assert not (tmp_path / "changed-readout").exists()


@pytest.mark.parametrize("stream", ["connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed"])
def test_development_episode_requires_canonical_dataset_stream_identity(tmp_path, monkeypatch, stream):
    path, _ = development_block_recipe(tmp_path / "recipes")
    config = yaml.safe_load(path.read_text())
    config["benchmark_episode_candidate"]["identity"][stream] += 1
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(quality, "export", lambda *args, **kwargs: pytest.fail("wrong dataset stream reached exporter"))
    output = tmp_path / "readout"

    with pytest.raises(ValueError, match="DATASET_STREAM_IDENTITY_INVALID"):
        quality.readout_episode(path, tmp_path / "missing-source", output)

    assert not output.exists()


@pytest.fixture(scope="module")
def reference_size_readout_archive(tmp_path_factory):
    """Complete2560-unit synthetic native archive; no mechanistic execution."""
    from .test_pfc_source_size_preparation import source_circuit
    from . import benchmark_sampling as sampling
    from .test_pfc_benchmark_production import archive
    from . import pfc_source_execution as executor
    from .pfc_native_drive import create_drive
    from . import pfc_scale_analysis as scale
    root=tmp_path_factory.mktemp("pfc-reference-size-reader")
    campaign,base=source_circuit(root / "source")
    plan=sampling.prepare_pfc_reference_size_block(campaign,base,0,6,root / "recipes")
    path=root / "recipes" / plan["conditions"][0]["config_path"]
    identity=plan["conditions"][0]["identity"]
    config=yaml.safe_load(path.read_text())
    source=archive(root / "native",path,None,identity)
    total,n_e,n_i,duration=2560,2048,512,400000
    ticks=np.asarray([0,8000,399999],dtype=np.int64)
    ids=np.asarray([0,1,2559],dtype=np.int64)
    counts=np.bincount(ids,minlength=total)
    terminal={}
    for pop,size,offset in (("E",n_e,0),("I",n_i,n_e)):
        for name in executor._STATE_FIELDS[pop]:terminal[pop+"__"+name]=np.zeros(size)
        terminal[pop+"__v"][:]=-.06
        terminal[pop+"__not_refractory"]=np.ones(size,dtype=np.bool_)
        terminal[pop+"__source_indices"]=np.arange(size,dtype=np.int64)
        terminal[pop+"__archive_indices"]=np.arange(size,dtype=np.int64)+offset
        terminal[pop+"__spike_counts"]=counts[offset:offset+size]
    native=source / "source_run"
    np.savez(native / "terminal_state.npz",**terminal)
    np.savez(native / "units.npz",**executor._units(n_e,n_i))
    np.savez(native / "raw.npz",spike_ticks=ticks,spike_unit_indices=ids,spike_seconds=ticks*.0000125,
             native_spike_counts=counts,terminal_voltage_mV=np.r_[terminal["E__v"],terminal["I__v"]]/.001)
    initial,drive=create_drive(config,config["cases"][0]);np.save(native / "initial_voltage.npy",initial)
    center=round(n_e/360.*config["stimulus_center_deg"])
    half=round(n_e/360.*config["stimulus_width_deg"]/2)
    currents=np.zeros((3,n_e));currents[1,np.asarray([i%n_e for i in range(center-half,center+half+1)])]=.5e-9
    np.savez(native / "applied_current_observation.npz",source_unit_indices=np.arange(n_e,dtype=np.int64),
        archive_unit_indices=np.arange(n_e,dtype=np.int64),segment_start_ticks=np.asarray([0,8000,24080],dtype=np.int64),
        segment_stop_ticks=np.asarray([8000,24080,duration],dtype=np.int64),cue_active=np.asarray([False,True,False]),
        distractor_active=np.zeros(3,dtype=np.bool_),current_A=currents)
    metadata=executor.executor._applied_current_metadata(n_e,.0000125,duration,80)
    (native / "applied_current_metadata.json").write_text(json.dumps(metadata))
    units=json.loads((native / "unit_metadata.json").read_text());units["unit_count"]=total
    (native / "unit_metadata.json").write_text(json.dumps(units))
    report=json.loads((native / "report.json").read_text())
    driver=drive.metadata();driver.update(ticks_generated=duration,external_event_totals=dict(E=0,I=0))
    report.update(driver=driver,completed_duration_ticks=duration,neural_event_count=3,
                  inputs_used=dict(input_mode="native_numpy_stream",ticks_consumed=duration,
                                   E=dict(values=n_e*duration,events=0),I=dict(values=n_i*duration,events=0)))
    for name,binding in report["outputs"].items():
        payload=native/name;binding.update(bytes=payload.stat().st_size,sha256=sha256_file(payload))
    (native / "report.json").write_text(json.dumps(report))
    delivery=json.loads((source / "delivery.json").read_text())
    for binding in delivery["files"]:
        payload=source/binding["path"];binding.update(bytes=payload.stat().st_size,sha256=sha256_file(payload))
    (source / "delivery.json").write_text(json.dumps(delivery))
    return path,source


def test_reference_size_reader_real_cli_retains_units_scope_and_source_binding(reference_size_readout_archive,tmp_path):
    config,source=reference_size_readout_archive
    output=tmp_path / "readout"
    env=dict(os.environ,PYTHONPATH=str(PROJECT / "Libs"),PYTHONDONTWRITEBYTECODE="1")
    child=subprocess.run([sys.executable,"-B","-m","population_dynamics.pfc_reference_quality",
        "readout-reference-size","--config",str(config),"--source",str(source),"--output",str(output)],
        env=env,cwd=tmp_path,capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    report=json.loads((output / "report.json").read_text())
    assert report["core_units"]==2560 and report["duration_ticks"]==400000 and report["dt_ms"]==.0125
    assert report["source_size_comparison"]["comparator_role"]=="original_size2560"
    assert report["campaign_episodes"]==report["production_episodes"]==0
    assert report["source_qualification_granted"] is report["scientific_claim_qualified"] is False
    assert report["source_delivery_sha256"]==sha256_file(source / "delivery.json")
    with np.load(output / "unit_population_statistics.npz") as arrays:
        assert arrays["phase_unit_counts"].shape[1]==2560 and arrays["population_field_counts"].shape==(1000,17)
        assert int(arrays["phase_unit_counts"].sum())==3
    dataset=CircuitDataset(output / "export/dataset",splits=("development",))
    episode=read_episode(dataset,next(dataset.iter_episodes()))
    assert episode.privileged["initial_voltage_mV"].shape==(2560,)
    assert episode.privileged["delivered_current__current_A"].shape==(3,2048)
    with pytest.raises(ValueError,match="FULL_DEVELOPMENT_SOURCE_REQUIRED"):
        quality.readout_episode(config,source,tmp_path / "wrong100k")
    assert not (tmp_path / "wrong100k").exists()


@pytest.mark.parametrize("damage",["marker","scale","namespace","original_provenance","production","task"])
def test_reference_size_reader_rejects_wrong_declaration_before_export(reference_size_readout_archive,tmp_path,monkeypatch,damage):
    original,_=reference_size_readout_archive;config=yaml.safe_load(original.read_text())
    if damage=="marker":config["pfc_source_size_comparison"]["scope"]="production"
    elif damage=="scale":config["excitatory_neurons"]=80000
    elif damage=="namespace":config["benchmark_episode_candidate"]["identity"]["initial_state_seed"]+=1
    elif damage=="original_provenance":config["pfc_source_size_comparison"]["original_circuit_candidate"]["parameter_identity"]="wrong"
    elif damage=="production":config["pfc_source_size_comparison"]["production_eligible"]=True
    else:config["cases"][0]["stimulus_strength_nA"]+=.1
    path=tmp_path / "changed.yaml";path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(quality,"export",lambda *args,**kwargs:pytest.fail("invalid size declaration reached exporter"))
    output=tmp_path / "readout"
    with pytest.raises(ValueError):quality.readout_reference_size(path,tmp_path / "source",output)
    assert not output.exists()

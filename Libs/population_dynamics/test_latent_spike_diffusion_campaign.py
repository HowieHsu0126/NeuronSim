"""LDNS prepared-source rejection, phase errors and verified fixture delivery."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import yaml


def launcher():
    path=Path(__file__).parents[1]/"scripts/run_latent_spike_diffusion_campaign.py"
    spec=importlib.util.spec_from_file_location("ldns_campaign",path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_phase_runs_real_process_and_records_nonzero_exit(tmp_path):
    runner=launcher()
    probe=lambda *args,**kwargs:dict(status="PASS")
    runner.run_phase(tmp_path,"success",[sys.executable,"-c","print('fixture')"],{},probe)
    assert json.loads((tmp_path/"phases/success/receipt.json").read_text())["status"]=="PASS"
    with pytest.raises(RuntimeError,match="phase failed"):
        runner.run_phase(tmp_path,"failure",[sys.executable,"-c","raise SystemExit(7)"],{},probe)
    failed=json.loads((tmp_path/"phases/failure/receipt.json").read_text())
    assert failed["status"]=="FAILED" and failed["exit_code"]==7


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    runner=launcher();prep=tmp_path/"preparation";prep.mkdir()
    source=prep/"source.py";source.write_text("bound source\n")
    paths=["Libs/population_dynamics/latent_spike_diffusion_training.py",
           "Libs/population_dynamics/latent_spike_diffusion.py",
           "Libs/population_dynamics/latent_spike_diffusion_device_check.py",
           "Libs/configs/neural_population_latent_spike_diffusion.yaml"]
    for name in paths:
        p=prep/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("bound fixture\n")
    (prep/paths[-1]).write_text(yaml.safe_dump(dict(maximum_autoencoder_epochs=260,
        maximum_diffusion_epochs=2000, autoencoder_validation_every=10,
        diffusion_validation_every=100, forecast_bins=100,
        fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
        forecast_origins_bins=[200, 600, 1000], validation_draws=4,
        model=dict(latent_channels=16))))
    np.savez_compressed(prep/"fitting_field_scale.npz", field_ids=np.asarray([str(i) for i in range(128)]))
    record=dict(status="VALIDATED_LDNS_CAUSAL_TRAINING_EVALUATION",training_seeds=[41,42,43],
        training_tests_passed=5,evaluation_tests_passed=3,adapter_tests_passed=4,
        real_episode_update_passed=True,files={"source.py":runner.digest(source),
            "fitting_field_scale.npz":runner.digest(prep/"fitting_field_scale.npz"),
            **{name:runner.digest(prep/name) for name in paths}},output_root="result")
    (prep/"preparation.json").write_text(json.dumps(record))
    monkeypatch.setenv("SLURMD_NODENAME","gpu03");monkeypatch.setenv("SLURM_JOB_PARTITION","gpu1")
    monkeypatch.setenv("SLURM_JOB_GPUS","0");monkeypatch.setenv("SLURM_JOB_ID","fixture")
    gate=tmp_path/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    gate.mkdir(parents=True)
    check=gate/"check.json"
    check.write_text(json.dumps(dict(diffusion_updates=dict(optimizer_steps=2,scheduler_steps=2,
        initial_zero_rate_parameters_unchanged=True,changed_parameter_tensors=1,maximum_parameter_change=1e-7))))
    checkpoint=gate/"checkpoint.pt";checkpoint.write_bytes(b"fixture checkpoint")
    (gate/"delivery.json").write_text(json.dumps(dict(status="PASS",files=[dict(path="check.json",sha256=runner.digest(check)),
        dict(path="checkpoint.pt",sha256=runner.digest(checkpoint),bytes=checkpoint.stat().st_size)],
        trainer_sha256=runner.digest(prep/paths[0]),adapter_sha256=runner.digest(prep/paths[1]),
        device_check_sha256=runner.digest(prep/paths[2]),configuration_sha256=runner.digest(prep/paths[3]))))
    return runner,prep


def test_storage_reserve_rejects_above_floor_but_below_known_write_demand(tmp_path,prepared,monkeypatch):
    from collections import namedtuple
    runner,prep=prepared
    gate=tmp_path/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    receipt=json.loads((gate/"delivery.json").read_text())
    usage=namedtuple("Usage","total used free")
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda path:usage(100*1024**3,79*1024**3,21*1024**3))
    def probe(output,**kwargs):
        value=dict(status="PASS");runner.write(output,value);return value
    def module(path,name):
        assert name=="ldns_resource_probe", "storage rejection must precede full-input module/read"
        return type("Probe",(),dict(probe=staticmethod(probe)))()
    monkeypatch.setattr(runner,"module",module)
    with pytest.raises(RuntimeError,match="insufficient storage reserve"):
        runner.run(tmp_path,prep,tmp_path/"work",41)
    report=json.loads((tmp_path/"work/storage_reserve.json").read_text())
    assert report["maximum_selected_snapshots"]==dict(autoencoder=26,diffusion=20)
    assert report["status"]=="BLOCKED" and not (tmp_path/"work/inputs.json").exists()


def test_storage_reserve_preserves_floor_and_declares_estimate(tmp_path,prepared,monkeypatch):
    from collections import namedtuple
    runner,prep=prepared
    gate=tmp_path/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    receipt=json.loads((gate/"delivery.json").read_text())
    usage=namedtuple("Usage","total used free")
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda path:usage(100*1024**3,50*1024**3,50*1024**3))
    result=runner.check_storage_reserve(tmp_path,prep/"Libs/configs/neural_population_latent_spike_diffusion.yaml",receipt,gate,reuse_inputs_from=tmp_path/"retained")
    assert result["unchanged_root_floor_bytes"]==20*1024**3
    assert result["required_free_bytes"]==result["unchanged_root_floor_bytes"]+result["known_write_reserve_bytes"]
    assert result["complete_training_peak_measured"] is False and result["input_reuse_requested"] is True
    assert result["original_node_watchdog_retained"] is True
    assert result["additional_input_copy_bytes"]==0 and "input copy" not in result["excluded_from_projection"]
    assert result["complete_forecast_origins"]==48
    assert result["all_origin_field_latent_identity_archive_projection_bytes"]>48*100000*8
    assert result["normalized_latent_cache_storage_projection_bytes"]==2369536
    assert result["normalized_latent_cache_dtype"]=="float32"
    assert result["simultaneous_loss_history_storage_projection_bytes"]>318105012
    assert result["independent_replay_adds_full_output_copy"] is False
    assert result["delivery_additional_node_local_full_copy"] is False


def test_storage_reserve_rejects_old_known_reserve_that_omitted_archives_and_history(tmp_path,prepared,monkeypatch):
    from collections import namedtuple
    runner,prep=prepared
    gate=tmp_path/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    receipt=json.loads((gate/"delivery.json").read_text())
    size=receipt["files"][-1]["bytes"]
    old_required=20*1024**3+(46+8)*size+17*100*100000*8
    usage=namedtuple("Usage","total used free")
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda path:usage(100*1024**3,0,old_required+1))
    with pytest.raises(RuntimeError,match="insufficient storage reserve"):
        runner.check_storage_reserve(tmp_path,prep/"Libs/configs/neural_population_latent_spike_diffusion.yaml",receipt,gate,reuse_inputs_from=tmp_path/"retained")
    report=json.loads((tmp_path/"storage_reserve.json").read_text())
    assert report["required_free_bytes"]>old_required and report["status"]=="BLOCKED"


def test_archive_projection_covers_actual_npz_write_without_compression_assumption(tmp_path):
    runner=launcher();rng=np.random.default_rng(41)
    arrays=dict(counts=rng.integers(0,2**40,size=(16,17,23),dtype=np.int64),
        latent=rng.normal(size=(16,16,17)).astype(np.float32),
        unit_ids=np.arange(23,dtype=np.int64),
        hashes=np.asarray(["f"*64]*16))
    target=tmp_path/"archive.npz";np.savez_compressed(target,**arrays)
    projected=runner.archive_storage_projection([(a.shape,a.dtype) for a in arrays.values()])
    assert projected>=target.stat().st_size


def test_storage_reserve_rejects_changed_checkpoint_size(tmp_path,prepared):
    runner,prep=prepared
    gate=tmp_path/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    receipt=json.loads((gate/"delivery.json").read_text());receipt["files"][-1]["bytes"]+=1
    with pytest.raises(ValueError,match="declared size differs"):
        runner.check_storage_reserve(tmp_path,prep/"Libs/configs/neural_population_latent_spike_diffusion.yaml",receipt,gate,reuse_inputs_from=None)


def test_changed_prepared_source_rejects_before_input_staging(tmp_path,prepared,monkeypatch):
    runner,prep=prepared;(prep/"source.py").write_text("changed\n")
    monkeypatch.setattr(runner,"module",lambda *args:pytest.fail("source rejection must precede input staging"))
    with pytest.raises(ValueError,match="prepared source changed"):
        runner.run(tmp_path,prep,tmp_path/"work",41)


def test_complete_fixture_seed_flow_delivers_verified_artifacts(tmp_path,prepared,monkeypatch):
    runner,prep=prepared
    def stage_inputs(repo,work):
        (work/"inputs.json").write_text("{}")
        (work/"prerequisites/cohort_delivery.json").write_text("{}")
    def module(path,name):
        def probe(output,**kwargs):
            value=dict(status="PASS");runner.write(output,value);return value
        return type("Module",(),dict(stage_inputs=staticmethod(stage_inputs),probe=staticmethod(probe)))()
    monkeypatch.setattr(runner,"module",module)
    called=[]
    def phase(work,name,command,env,probe):
        called.append(name)
        (work/"phases"/name).mkdir(parents=True)
        (work/"phases"/name/"receipt.json").write_text(json.dumps(dict(status="PASS")))
        if name=="train":
            (work/"training").mkdir();(work/"training/checkpoint.pt").write_bytes(b"fixture checkpoint")
        else:
            (work/"forecasts").mkdir(exist_ok=True)
            (work/"forecasts"/("replay.json" if name=="replay" else "evaluation.json")).write_text(json.dumps(dict(status="PASS")))
    monkeypatch.setattr(runner,"run_phase",phase)
    result=runner.run(tmp_path,prep,tmp_path/"work",42)
    assert called==["train","evaluate","replay"]
    receipt=json.loads((result/"delivery.json").read_text())
    assert receipt["status"]=="PASS" and receipt["training_seed"]==42
    for row in receipt["files"]:assert runner.digest(result/row["path"])==row["sha256"]


def test_gate_rejects_ineffective_update_before_training(tmp_path,prepared,monkeypatch):
    runner,prep=prepared
    gate=tmp_path/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    check=gate/"check.json";record=json.loads(check.read_text())
    record["diffusion_updates"]["maximum_parameter_change"]=0.
    check.write_text(json.dumps(record))
    receipt=json.loads((gate/"delivery.json").read_text())
    receipt["files"][0]["sha256"]=runner.digest(check)
    (gate/"delivery.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(runner,"module",lambda *args:pytest.fail("reject before input staging or training"))
    with pytest.raises(ValueError,match="must demonstrate a denoiser parameter update"):
        runner.run(tmp_path,prep,tmp_path/"work",41)


def test_resource_admission_precedes_cohort_copy_and_training(tmp_path,prepared,monkeypatch):
    runner,prep=prepared
    def module(path,name):
        assert name == "ldns_resource_probe", "BLOCKED_NODE_STARTED_COHORT_STAGING"
        def probe(output,**kwargs):
            value=dict(status="BLOCKED",reasons=["ROOT_STORAGE_CRITICAL"])
            runner.write(output,value)
            return value
        return type("Module",(),dict(probe=staticmethod(probe)))()
    monkeypatch.setattr(runner,"module",module)
    monkeypatch.setattr(runner,"run_phase",lambda *args:pytest.fail("BLOCKED_NODE_STARTED_TRAINING"))
    work=tmp_path/"work"
    with pytest.raises(RuntimeError,match="before input staging"):
        runner.run(tmp_path,prep,work,43)
    assert json.loads((work/"input_staging_preflight.json").read_text())["reasons"] == ["ROOT_STORAGE_CRITICAL"]
    assert not (work/"inputs").exists() and not (work/"training").exists()


@pytest.fixture
def reusable_inputs(tmp_path,prepared):
    runner,prep=prepared
    original=tmp_path/"original";original.mkdir()
    (original/"preparation.json").write_bytes((prep/"preparation.json").read_bytes())
    failure=original/"phases/train/preflight.json";failure.parent.mkdir(parents=True)
    failure.write_text(json.dumps(dict(status="BLOCKED",reasons=["ROOT_STORAGE_CRITICAL"],
        hostname="gpu03",slurm_job_id="611529")))
    cases=[dict(block=b,template_id=f"familiar_{c}",case_id=f"train_block_{b:03d}_familiar_{c}")
           for b in range(18) for c in range(4)]
    payloads={"train/design.json":json.dumps(dict(acquired_split="train",cases=cases)).encode(),
        "train/acquisition.json":b"fixture ledger", "train_fields/dataset.json":b"fixture dataset",
        "train_fields/episodes.csv":b"fixture episode table"}
    for case in cases:
        for prefix,names in (("train",("report.json","spikes.npz","initial_parameters.npz")),
                             ("train_fields",("summary.json","spatial_fields.npz","unit_rates.npz"))):
            for name in names:payloads[f"{prefix}/{case['case_id']}/{name}"]=b"bound fixture"
    rows=[]
    for name,payload in payloads.items():
        path=original/"inputs"/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        rows.append(dict(path=name,bytes=len(payload),sha256=runner.digest(path)))
    cohort=tmp_path/"cohort";cohort.mkdir()
    receipt=dict(status="VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE",completed_episodes=72,
        cortical_units=100000,episode_duration_ms=6000.,source_dt_ms=.1,field_bin_ms=5.,
        native_rng_family="per_edge_philox4_32_10",independent_field_replay=dict(status="PASS"),files=rows)
    receipt_path=cohort/"delivery.json";receipt_path.write_text(json.dumps(receipt))
    (original/"prerequisites").mkdir()
    (original/"prerequisites/cohort_delivery.json").write_bytes(receipt_path.read_bytes())
    (original/"inputs.json").write_text(json.dumps(dict(delivery_sha256=runner.digest(receipt_path),
        source_root=str(cohort),source_identity=receipt["independent_field_replay"],files=rows)))
    stager_path=Path(__file__).parents[1]/"scripts/run_population_trajectory_campaign.py"
    stager=runner.module(stager_path,"original_stager");stager.COHORT="cohort"
    # Keep the production node-local path boundary in the disposable fixture.
    with tempfile.TemporaryDirectory(prefix="ldns-input-reuse-test-",dir="/tmp/neuronsim") as temporary:
        retained=Path(temporary)/"original";shutil.copytree(original,retained)
        yield runner,prep,retained,stager


def test_reuse_inputs_verifies_all_files_without_copying_or_changing_original(tmp_path,reusable_inputs):
    runner,prep,source,stager=reusable_inputs
    before={str(p.relative_to(source)):runner.digest(p) for p in source.rglob("*") if p.is_file()}
    work=tmp_path/"work";work.mkdir()
    root=runner.reuse_inputs(tmp_path,prep,source,work,stager)
    assert root==source/"inputs" and not (work/"inputs").exists()
    receipt=json.loads((work/"prerequisites/input_reuse.json").read_text())
    assert receipt["verified_files"]==436 and receipt["original_cell_modified"] is False
    assert before=={str(p.relative_to(source)):runner.digest(p) for p in source.rglob("*") if p.is_file()}


def test_reuse_inputs_accepts_verified_relocated_input_link(tmp_path,reusable_inputs):
    runner,prep,source,stager=reusable_inputs
    target=tmp_path/"local_data_inputs"
    (source/"inputs").rename(target)
    (source/"inputs").symlink_to(target,target_is_directory=True)
    before={str(p.relative_to(target)):runner.digest(p) for p in target.rglob("*") if p.is_file()}
    work=tmp_path/"work";work.mkdir()
    assert runner.reuse_inputs(tmp_path,prep,source,work,stager)==source/"inputs"
    assert (source/"inputs").is_symlink() and not (work/"inputs").exists()
    assert before=={str(p.relative_to(target)):runner.digest(p) for p in target.rglob("*") if p.is_file()}
    receipt=json.loads((work/"prerequisites/input_reuse.json").read_text())
    assert receipt["verified_files"]==436 and receipt["original_cell_modified"] is False


@pytest.mark.parametrize("work_root",["default","data","relative","missing"])
def test_actual_wrapper_uses_declared_work_root_and_preserves_failed_scratch(tmp_path,work_root):
    import os
    script=Path(__file__).parents[1]/"scripts/run_latent_spike_diffusion_campaign.sbatch"
    binaries=tmp_path/"bin";binaries.mkdir()
    capture=tmp_path/"arguments.json"
    fake=binaries/"srun"
    fake.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                    'from pathlib import Path\n'
                    'Path(os.environ["LDNS_TEST_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
                    'work=Path(sys.argv[sys.argv.index("--work")+1])\n'
                    'work.mkdir(parents=True)\n(work/"retained").write_text("failed fixture")\n'
                    'raise SystemExit(7)\n')
    fake.chmod(0o755)
    arguments=["bash",str(script),str(tmp_path/"preparation"),str(tmp_path/"launcher.py"),"/tmp/neuronsim/retained-611529"]
    array_id="ldns-wrapper-test-"+tmp_path.name
    expected=Path('/tmp/neuronsim')/f'neuronsim-ldns-{array_id}-42'
    if work_root=="data":
        data=tmp_path/"data";data.mkdir();arguments.append(str(data)+"/")
        expected=data/f'neuronsim-ldns-{array_id}-42'
    elif work_root=="relative":arguments.append("relative")
    elif work_root=="missing":arguments.append(str(tmp_path/"absent"))
    env=dict(os.environ,PATH=str(binaries)+os.pathsep+os.environ['PATH'],
             SLURM_ARRAY_JOB_ID=array_id,SLURM_ARRAY_TASK_ID="1",LDNS_TEST_CAPTURE=str(capture))
    try:
        result=subprocess.run(arguments,env=env,capture_output=True,text=True)
        if work_root in ("relative","missing"):
            assert result.returncode==2 and not capture.exists()
            assert 'existing absolute node-local directory' in result.stderr
        else:
            assert result.returncode==7
            called=json.loads(capture.read_text())
            assert called[called.index('--work')+1]==str(expected)
            assert called[called.index('--reuse-inputs-from')+1]=='/tmp/neuronsim/retained-611529-42'
            assert called[called.index('--seed')+1]=='42'
            assert (expected/"retained").read_text()=="failed fixture"
    finally:
        if expected.exists():shutil.rmtree(expected)


@pytest.mark.parametrize("mutation",["changed_file","missing_file","missing_row","training_started","changed_preparation","changed_failure"])
def test_reuse_inputs_rejects_invalid_or_started_cells(tmp_path,reusable_inputs,mutation):
    runner,prep,source,stager=reusable_inputs
    bound=source/"inputs/train/train_block_000_familiar_0/spikes.npz"
    if mutation=="changed_file":bound.write_bytes(b"corrupted")
    elif mutation=="missing_file":bound.unlink()
    elif mutation=="missing_row":
        path=source/"inputs.json";value=json.loads(path.read_text());value["files"].pop();path.write_text(json.dumps(value))
    elif mutation=="training_started":(source/"training").mkdir()
    elif mutation=="changed_preparation":(source/"preparation.json").write_text("{}")
    else:
        path=source/"phases/train/preflight.json";value=json.loads(path.read_text());value["reasons"]=["PERSISTENT_SWAPPING"];path.write_text(json.dumps(value))
    work=tmp_path/"work";work.mkdir()
    with pytest.raises((ValueError,FileNotFoundError)):
        runner.reuse_inputs(tmp_path,prep,source,work,stager)
    assert not (work/"prerequisites/input_reuse.json").exists()


def test_seed_flow_reads_reused_inputs_for_every_phase(tmp_path,prepared,monkeypatch):
    runner,prep=prepared
    source=tmp_path/"retained";input_root=source/"inputs"
    def probe(output,**kwargs):
        value=dict(status="PASS");runner.write(output,value);return value
    def module(path,name):
        def stage_inputs(*args):pytest.fail("reused inputs must not be copied")
        return type("Module",(),dict(probe=staticmethod(probe),stage_inputs=staticmethod(stage_inputs)))()
    monkeypatch.setattr(runner,"module",module)
    reused=[]
    def reuse(repo,preparation,original,work,stager):
        reused.append(original)
        runner.write(work/"inputs.json",{})
        runner.write(work/"prerequisites/cohort_delivery.json",{})
        runner.write(work/"prerequisites/input_reuse.json",dict(status="fixture"))
        return input_root
    monkeypatch.setattr(runner,"reuse_inputs",reuse)
    called=[]
    def phase(work,name,command,env,probe):
        called.append(name)
        assert command[command.index("--acquisition")+1]==str(input_root/"train")
        assert command[command.index("--fields")+1]==str(input_root/"train_fields")
        runner.write(work/"phases"/name/"receipt.json",dict(status="PASS"))
        if name=="train":
            (work/"training").mkdir();(work/"training/checkpoint.pt").write_bytes(b"fixture")
        else:runner.write(work/"forecasts"/("replay.json" if name=="replay" else "evaluation.json"),dict(status="PASS"))
    monkeypatch.setattr(runner,"run_phase",phase)
    result=runner.run(tmp_path,prep,tmp_path/"work",43,source)
    assert reused==[source] and called==["train","evaluate","replay"]
    assert not (tmp_path/"work/inputs").exists()
    receipt=json.loads((result/"delivery.json").read_text())
    names={r["path"] for r in receipt["files"]}
    assert {"prerequisites/input_reuse.json","prerequisites/campaign_launcher.py",
            "input_staging_preflight.json"}<=names


def test_failed_admission_rejects_before_reused_input_inspection(tmp_path,prepared,monkeypatch):
    runner,prep=prepared
    def probe(output,**kwargs):
        value=dict(status="BLOCKED",reasons=["ROOT_STORAGE_CRITICAL"])
        runner.write(output,value);return value
    monkeypatch.setattr(runner,"module",lambda path,name:type("Module",(),dict(probe=staticmethod(probe)))())
    monkeypatch.setattr(runner,"reuse_inputs",lambda *args:pytest.fail("blocked node must not read complete inputs"))
    with pytest.raises(RuntimeError,match="before input staging"):
        runner.run(tmp_path,prep,tmp_path/"work",41,tmp_path/"retained")


def test_reuse_rejects_nested_work_before_writing(tmp_path,prepared):
    runner,prep=prepared
    retained=tmp_path/"retained";retained.mkdir()
    work=retained/"recovery"
    with pytest.raises(ValueError,match="must be separate"):
        runner.run(tmp_path,prep,work,41,retained)
    assert not work.exists()


def test_existing_seed_delivery_rejects_before_inputs_or_training(tmp_path,prepared,monkeypatch):
    runner,prep=prepared
    (tmp_path/"result/seed_42").mkdir(parents=True)
    monkeypatch.setattr(runner,"module",lambda *args:pytest.fail("existing delivery must not start input or training work"))
    with pytest.raises(FileExistsError,match="seed evidence must be reused"):
        runner.run(tmp_path,prep,tmp_path/"work",42)


@pytest.mark.parametrize("free_gib,used_fraction", [(19., .5), (25., .96)])
def test_work_disk_admission_rejects_before_child_even_when_root_probe_passes(tmp_path, monkeypatch, free_gib, used_fraction):
    from collections import namedtuple
    runner = launcher()
    usage = namedtuple("Usage", "total used free")
    total = 100 * 1024**3
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: usage(total, int(total * used_fraction), int(free_gib * 1024**3)))
    marker = tmp_path / "child_started"
    with pytest.raises(RuntimeError, match="work storage admission failed before phase child"):
        runner.run_phase(tmp_path, "rejected", [sys.executable, "-c", "from pathlib import Path; Path('child_started').touch()"],
                         {}, lambda *args, **kwargs: dict(status="PASS"))
    assert not marker.exists()
    report = json.loads((tmp_path / "phases/rejected/work_storage_preflight.json").read_text())
    assert report["reasons"] == ["WORK_STORAGE_CRITICAL"]
    assert report["unchanged_storage_floor_bytes"] == 20 * 1024**3


def test_work_disk_watchdog_stops_only_phase_child_and_preserves_checkpoint(tmp_path, monkeypatch):
    from collections import namedtuple
    import os
    import signal
    import time
    runner = launcher()
    usage = namedtuple("Usage", "total used free")
    samples = iter([50, 19])
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: usage(100 * 1024**3, 0, next(samples) * 1024**3))
    original_popen = runner.subprocess.Popen
    children = []
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"retained original checkpoint")
    started = tmp_path / "child_started"
    class FastObservation:
        def __init__(self, *args, **kwargs):
            self.child = original_popen(*args, **kwargs)
            children.append(self.child)
            self.timeout_sent = False
        @property
        def pid(self): return self.child.pid
        @property
        def returncode(self): return self.child.returncode
        def poll(self): return self.child.poll()
        def wait(self, timeout):
            if timeout == 60 and not self.timeout_sent:
                self.timeout_sent = True
                deadline = time.monotonic() + 5
                while not started.exists() and self.child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                assert started.exists()
                raise subprocess.TimeoutExpired(self.child.args, timeout)
            return self.child.wait(timeout=timeout)
    monkeypatch.setattr(runner.subprocess, "Popen", FastObservation)
    commands = [sys.executable, "-c", "from pathlib import Path; import time; Path('child_started').touch(); time.sleep(30)"]
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(RuntimeError, match="resource watchdog stopped child"):
            runner.run_phase(tmp_path, "guarded", commands, {}, lambda *args, **kwargs: dict(status="PASS"))
        assert len(children) == 1 and children[0].returncode == -signal.SIGTERM
        assert checkpoint.read_bytes() == b"retained original checkpoint"
        receipt = json.loads((tmp_path / "phases/guarded/receipt.json").read_text())
        assert receipt["status"] == "RESOURCE_BLOCKED" and receipt["reasons"] == ["WORK_STORAGE_CRITICAL"]
        assert (tmp_path / "phases/guarded/work_storage_watchdog.json").is_file()
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=5)
        for sig, handler in previous.items(): signal.signal(sig, handler)


def _copied_recovery_cell(tmp_path, prepared, *, legacy=False, executable=False):
    runner, prep = prepared
    if executable:
        modules = {
            "Libs/__init__.py": "",
            "Libs/population_dynamics/__init__.py": "",
            "Libs/population_dynamics/latent_spike_diffusion_training.py":
                "import argparse, json\nfrom pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "for name in ('acquisition', 'fields', 'config', 'output', 'seed'):\n"
                "    parser.add_argument('--' + name, required=True)\n"
                "parser.add_argument('--resume', action='store_true')\n"
                "args = parser.parse_args()\n"
                "assert args.resume, 'fixture requires explicit resume'\n"
                "Path(args.output, 'resume_arguments.json').write_text(json.dumps(vars(args)))\n",
            "Libs/population_dynamics/latent_spike_diffusion_evaluation.py":
                "import argparse, json\nfrom pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "for name in ('acquisition', 'fields', 'config', 'training', 'fitting-scale', 'output', 'seed'):\n"
                "    parser.add_argument('--' + name, required=True)\n"
                "parser.add_argument('--replay', action='store_true')\n"
                "args = parser.parse_args()\n"
                "output = Path(args.output); output.mkdir(exist_ok=True)\n"
                "(output / ('replay.json' if args.replay else 'evaluation.json')).write_text("
                "json.dumps(dict(status='PASS')))\n",
        }
        record = json.loads((prep / "preparation.json").read_text())
        for name, source in modules.items():
            (prep / name).write_text(source)
            record["files"][name] = runner.digest(prep / name)
        runner.write(prep / "preparation.json", record)
        gate = tmp_path / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012/delivery.json"
        delivery = json.loads(gate.read_text())
        delivery["trainer_sha256"] = record["files"]["Libs/population_dynamics/latent_spike_diffusion_training.py"]
        runner.write(gate, delivery)
    work = tmp_path / "recovery"
    shutil.copytree(prep, work)
    source_work = tmp_path / "failed"
    input_root = work / "inputs" if legacy else tmp_path / "external_inputs"
    (input_root / "train").mkdir(parents=True)
    (input_root / "train_fields").mkdir()
    seed, epoch = (41, 198) if legacy else (42, 199)
    node, job = ("gpu03", "617154") if legacy else ("cpu03", "618593")
    checkpoint = work / "training/checkpoints" / f"autoencoder_{epoch:04d}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture committed model optimizer scheduler and random states")
    pointer = dict(checkpoint=str(checkpoint.relative_to(work / "training")),
                   sha256=runner.digest(checkpoint), stage="autoencoder", next_epoch=epoch)
    runner.write(work / "training/committed.json", pointer)
    runner.write(work / "inputs.json", dict(fixture=True))
    runner.write(work / "input_staging_preflight.json", dict(status="PASS"))
    runner.write(work / "storage_reserve.json", dict(status="PASS"))
    runner.write(work / "phases/train/preflight.json",
                 dict(status="PASS", hostname=node, slurm_job_id=job))
    command = [sys.executable, "-B", "-m", "Libs.population_dynamics.latent_spike_diffusion_training",
               "--acquisition", str(input_root / "train"), "--fields", str(input_root / "train_fields"),
               "--config", str(source_work / "Libs/configs/neural_population_latent_spike_diffusion.yaml"),
               "--seed", str(seed), "--output", str(source_work / "training")]
    runner.write(work / "phases/train/receipt.json",
                 dict(status="INTERRUPTED" if legacy else "RESOURCE_BLOCKED", exit_code=-15,
                      reasons=["PERSISTENT_SWAPPING"], command=command))
    runner.write(work / "prerequisites/cohort_delivery.json", dict(fixture=True))
    gate = tmp_path / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012/delivery.json"
    shutil.copy2(gate, work / "prerequisites/cuda_delivery.json")
    shutil.copy2(Path(__file__).parents[1] / "scripts/run_latent_spike_diffusion_campaign.py",
                 work / "prerequisites/campaign_launcher.py")
    shutil.copytree(work, source_work)
    files = [dict(path=str(path.relative_to(work)), bytes=path.stat().st_size, sha256=runner.digest(path))
             for path in sorted(work.rglob("*")) if path.is_file()]
    migration = dict(status="PASS", source_node=node, source_job=job, committed=pointer, files=files)
    if not legacy:
        migration.update(schema="neural_population_latent_spike_diffusion_resource_recovery_v1",
                         training_seed=seed, source_work=str(source_work), input_root=str(input_root))
    runner.write(work / "migration.json", migration)
    return runner, prep, work, source_work, input_root


def _refresh_recovery_manifest(runner, work, migration, name):
    for row in migration["files"]:
        if row["path"] == name:
            path = work / name
            row.update(bytes=path.stat().st_size, sha256=runner.digest(path))
            return
    raise AssertionError("fixture file is absent from migration")


def test_resource_recovery_accepts_seed42_committed199_without_changing_original(tmp_path, prepared):
    runner, prep, work, source, inputs = _copied_recovery_cell(tmp_path, prepared)
    before = {str(path.relative_to(source)): runner.digest(path)
              for path in source.rglob("*") if path.is_file()}
    _, migration = runner.validate_recovery(work, prep, 42)
    assert migration["committed"]["next_epoch"] == 199
    assert Path(migration["input_root"]) == inputs
    assert not (work / "inputs").exists()
    assert before == {str(path.relative_to(source)): runner.digest(path)
                      for path in source.rglob("*") if path.is_file()}


def test_recovery_preserves_legacy_seed41_migration198(tmp_path, prepared):
    runner, prep, work, _, _ = _copied_recovery_cell(tmp_path, prepared, legacy=True)
    _, migration = runner.validate_recovery(work, prep, 41)
    assert migration["committed"]["next_epoch"] == 198
    with pytest.raises(ValueError, match="registered cancelled seed41"):
        runner.validate_recovery(work, prep, 42)
    migration["committed"]["next_epoch"] = 199
    runner.write(work / "migration.json", migration)
    with pytest.raises(ValueError, match="committed epoch identity"):
        runner.validate_recovery(work, prep, 41)


@pytest.mark.parametrize("mutation", [
    "source_job", "source_node", "training_seed", "input_root", "train_argv",
    "train_status", "train_exit", "train_reasons", "preflight_job", "checkpoint",
    "committed_epoch", "epoch_above_ceiling", "missing_phase_row", "escaping_row",
    "original_work", "cuda_configuration",
])
def test_resource_recovery_rejects_changed_lineage_inputs_and_checkpoint(tmp_path, prepared, mutation):
    runner, prep, work, _, _ = _copied_recovery_cell(tmp_path, prepared)
    migration = json.loads((work / "migration.json").read_text())
    if mutation == "source_job":
        migration["source_job"] = "618594"
    elif mutation == "source_node":
        migration["source_node"] = "cpu04"
    elif mutation == "training_seed":
        migration["training_seed"] = 43
    elif mutation == "input_root":
        migration["input_root"] = str(tmp_path / "other_inputs")
    elif mutation == "checkpoint":
        (work / "training" / migration["committed"]["checkpoint"]).write_bytes(b"changed state")
    elif mutation == "committed_epoch":
        migration["committed"]["next_epoch"] = 198
    elif mutation == "epoch_above_ceiling":
        pointer = dict(migration["committed"], next_epoch=261)
        runner.write(work / "training/committed.json", pointer)
        migration["committed"] = pointer
        _refresh_recovery_manifest(runner, work, migration, "training/committed.json")
    elif mutation == "missing_phase_row":
        migration["files"] = [row for row in migration["files"] if row["path"] != "phases/train/receipt.json"]
    elif mutation == "escaping_row":
        migration["files"][0]["path"] = "../failed/preparation.json"
    elif mutation == "original_work":
        migration["source_work"] = str(work)
    elif mutation == "preflight_job":
        name = "phases/train/preflight.json"
        value = json.loads((work / name).read_text()); value["slurm_job_id"] = "618594"
        runner.write(work / name, value); _refresh_recovery_manifest(runner, work, migration, name)
    elif mutation == "cuda_configuration":
        name = "prerequisites/cuda_delivery.json"
        value = json.loads((work / name).read_text()); value["configuration_sha256"] = "0" * 64
        runner.write(work / name, value); _refresh_recovery_manifest(runner, work, migration, name)
    else:
        name = "phases/train/receipt.json"
        value = json.loads((work / name).read_text())
        if mutation == "train_argv":
            value["command"][value["command"].index("--acquisition") + 1] = str(tmp_path / "other/train")
        elif mutation == "train_status":
            value["status"] = "PASS"
        elif mutation == "train_exit":
            value["exit_code"] = 0
        else:
            value["reasons"] = []
        runner.write(work / name, value); _refresh_recovery_manifest(runner, work, migration, name)
    runner.write(work / "migration.json", migration)
    with pytest.raises(ValueError, match="LDNS"):
        runner.validate_recovery(work, prep, 42)


def test_resume_runs_original_external_input_argv_and_preserves_failed_phase(tmp_path, prepared, monkeypatch):
    import signal
    runner, prep, work, source, inputs = _copied_recovery_cell(tmp_path, prepared, executable=True)
    before = {str(path.relative_to(source)): runner.digest(path)
              for path in source.rglob("*") if path.is_file()}
    def probe(output, **kwargs):
        value = dict(status="PASS"); runner.write(output, value); return value
    monkeypatch.setattr(runner, "module", lambda *args: type("Probe", (), dict(probe=staticmethod(probe)))())
    monkeypatch.setattr(runner, "work_storage_health", lambda *args: dict(status="PASS", reasons=[]))
    reserves = []
    def reserve(*args, reuse_inputs_from):
        reserves.append(reuse_inputs_from)
    monkeypatch.setattr(runner, "check_storage_reserve", reserve)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        result = runner.run(tmp_path, prep, work, 42, resume=True)
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
    args = json.loads((result / "training/resume_arguments.json").read_text())
    assert args["resume"] is True and args["seed"] == "42"
    assert args["acquisition"] == str(inputs / "train") and args["fields"] == str(inputs / "train_fields")
    assert args["config"] == str(work / "Libs/configs/neural_population_latent_spike_diffusion.yaml")
    assert args["output"] == str(work / "training") and reserves == [inputs]
    assert not (work / "inputs").exists()
    delivery = json.loads((result / "delivery.json").read_text())
    assert delivery["resumed_from_job"] == "618593"
    assert delivery["resumed_from_training_stage"] == "autoencoder"
    assert delivery["resumed_from_stage_epochs"] == delivery["resumed_from_autoencoder_epochs"] == 199
    assert json.loads((result / "phases/train/receipt.json").read_text())["status"] == "RESOURCE_BLOCKED"
    assert json.loads((result / "forecasts/replay.json").read_text())["status"] == "PASS"
    assert before == {str(path.relative_to(source)): runner.digest(path)
                      for path in source.rglob("*") if path.is_file()}


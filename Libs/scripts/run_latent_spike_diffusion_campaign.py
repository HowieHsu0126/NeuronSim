"""Deploy one prepared LDNS seed with native inputs and verified delivery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def archive_storage_projection(arrays):
    """Project an NPZ write without assuming favorable count compression."""
    total = 1024  # ZIP directory/end metadata allowance.
    for shape, dtype in arrays:
        dtype = np.dtype(dtype)
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(
            descr=np.lib.format.dtype_to_descr(dtype), fortran_order=False, shape=shape))
        size = int(np.prod(shape)) * dtype.itemsize + len(header.getvalue())
        # zlib compressBound expansion plus ZIP entry metadata allowance.
        total += size + (size >> 12) + (size >> 14) + (size >> 25) + 13 + 1024
    return total


def loss_history_projection(config):
    """Bound numeric width in the frozen complete loss-history structure.

    The four copies are the previous/new optimizer histories and the
    previous/new epochs.json. These values are width sentinels, not losses.
    """
    sentinel = -sys.float_info.max
    terms = dict(loss=sentinel, reconstruction_nll=sentinel,
                 latent_l2=sentinel, latent_temporal_difference=sentinel)
    fit = [f"train_block_{b:03d}_familiar_{c}" for b in config["fit_blocks"] for c in range(4)]
    validation = [f"train_block_{b:03d}_familiar_{c}" for b in config["validation_blocks"] for c in range(4)]
    origins = config["forecast_origins_bins"]
    chosen = max(origins)

    def ae_rows(cases, names, repetitions):
        return dict(mean=terms, episodes=[dict(case_id=case, windows=[
            dict(window=name, draw=draw, terms=terms)
            for name in names for draw in range(repetitions)]) for case in cases])

    def diffusion_rows(cases, repetitions):
        return dict(mean=dict(loss=sentinel), windows=[dict(case_id=case, origin=origin,
            losses=[sentinel] * repetitions) for case in cases for origin in origins])

    total = 3  # Opening/closing brackets and newline.
    for stage in ("autoencoder", "diffusion"):
        maximum = config["maximum_" + stage + "_epochs"]
        period = config[stage + "_validation_every"]
        validations = (maximum + period - 1) // period
        if stage == "autoencoder":
            training = ae_rows(fit, ["complete", f"prefix_{chosen}", f"future_{chosen}"], 1)
            checked = ae_rows(validation, ["complete"] + [name for origin in origins
                for name in (f"prefix_{origin}", f"future_{origin}")], config["validation_draws"])
        else:
            training = diffusion_rows(fit, 1)
            checked = diffusion_rows(validation, config["validation_draws"])
        for number, valid in ((maximum - validations, None), (validations, checked)):
            text = json.dumps(dict(stage=stage, epoch=maximum - 1,
                training=training, validation=valid), indent=2, allow_nan=False)
            # Two spaces indent each line within the outer JSON list.
            total += number * (len(text.encode()) + 2 * len(text.splitlines()) + 2)
    return 4 * total


def check_storage_reserve(work, config_path, cuda_delivery, gate, *, reuse_inputs_from):
    """Reserve the known model/checkpoint/output writes above the root floor.

    The successful full-unit CUDA checkpoint supplies serialized model size.
    Optimizer commits use a state-size projection, not a measured training peak.
    The original resource watchdog remains active for other node demand.
    """
    config = yaml.safe_load(config_path.read_text())
    checkpoints = [row for row in cuda_delivery["files"] if row["path"] == "checkpoint.pt"]
    if len(checkpoints) != 1:
        raise ValueError("LDNS storage reserve requires the verified full-unit CUDA checkpoint")
    checkpoint = checkpoints[0]
    size = (gate / "checkpoint.pt").stat().st_size
    if type(checkpoint.get("bytes")) is not int or checkpoint["bytes"] != size or size <= 0:
        raise ValueError("LDNS CUDA checkpoint declared size differs")
    selections = {}
    for stage in ("autoencoder", "diffusion"):
        maximum = config["maximum_" + stage + "_epochs"]
        period = config[stage + "_validation_every"]
        if type(maximum) is not int or type(period) is not int or min(maximum, period) < 1:
            raise ValueError("positive complete LDNS epoch/validation budgets required")
        selections[stage] = (maximum + period - 1) // period
    # A commit contains the full model, up to two model-sized Adam moments and
    # denoiser EMA. Two commits coexist until the atomic pointer has advanced.
    commit_projection = 2 * 4 * size
    selected_projection = sum(selections.values()) * size
    bins, draws, units = config["forecast_bins"], 16, 100000
    channels = config["model"]["latent_channels"]
    origins = len(config["forecast_origins_bins"])
    # This small fitting-scale artifact is prepared before full-input admission.
    with np.load(config_path.parents[2] / "fitting_field_scale.npz", allow_pickle=False) as scale:
        ids = scale["field_ids"]
    if ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids):
        raise ValueError("LDNS storage projection requires source field identities")
    field_count = len(ids)
    fitting_cases, validation_cases = 4 * len(config["fit_blocks"]), 4 * len(config["validation_blocks"])
    retained_counts = (draws + 1) * bins * units * 8
    full_unit_archive = archive_storage_projection([
        ((draws, bins, units), "int64"), ((bins, units), "int64"), ((units,), "int64")])
    per_origin_archive = archive_storage_projection([
        ((draws, bins, field_count), "int64"), ((bins, field_count), "int64"),
        ((draws, channels, bins), "float32"), ((1, channels, 1), "float32"),
        ((draws,), "U64"), ((units,), "int64"), ((field_count,), ids.dtype)])
    # set_latent_scale casts moments to model float32 before cache normalization.
    # Three tensors per window plus row metadata retain a serialization allowance.
    cache = ((fitting_cases + validation_cases) * origins
             * (channels * (1 + bins) * 4 + bins * 4 + 4 * 1024)
             + 2 * channels * 4 + 2 * 1024)
    history = loss_history_projection(config)
    field_archives = validation_cases * origins * per_origin_archive
    known_write_reserve = (selected_projection + commit_projection + full_unit_archive
                          + field_archives + cache + history)
    free = shutil.disk_usage(work).free
    floor = 20 * 1024 ** 3
    report = dict(status="PASS" if free >= floor + known_write_reserve else "BLOCKED",
        free_bytes=free, storage_path=str(work.resolve()),
        storage_device_id=work.stat().st_dev,
        storage_shares_root_device=work.stat().st_dev == Path('/').stat().st_dev,
        unchanged_root_floor_bytes=floor,
        verified_cuda_model_checkpoint_bytes=size, maximum_selected_snapshots=selections,
        selected_checkpoint_storage_projection_bytes=selected_projection,
        simultaneous_optimizer_commit_storage_projection_bytes=commit_projection,
        retained_full_unit_count_uncompressed_bytes=retained_counts,
        full_unit_archive_storage_projection_bytes=full_unit_archive,
        all_origin_field_latent_identity_archive_projection_bytes=field_archives,
        normalized_latent_cache_storage_projection_bytes=cache,
        normalized_latent_cache_dtype="float32",
        simultaneous_loss_history_storage_projection_bytes=history,
        complete_forecast_origins=validation_cases * origins,
        independent_replay_adds_full_output_copy=False,
        delivery_additional_node_local_full_copy=False,
        known_write_reserve_bytes=known_write_reserve,
        required_free_bytes=floor + known_write_reserve,
        complete_training_peak_measured=False,
        excluded_from_projection=["checkpoint binding/serialization overhead beyond reference-size projection",
            "phase logs and CSV/JSON receipts", "block-allocation overhead", "other node workloads"]
            + (["input copy"] if reuse_inputs_from is None else []),
        additional_input_copy_bytes=0 if reuse_inputs_from is not None else None,
        input_reuse_requested=reuse_inputs_from is not None,
        original_node_watchdog_retained=True)
    write(work / "storage_reserve.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("LDNS insufficient storage reserve before input reading or training")
    return report


def work_storage_health(work):
    """Keep the admitted work filesystem above its existing 20 GiB reserve.

    The shared node probe separately checks root storage, memory, I/O and GPU
    telemetry. A non-root work disk needs its own live check during each phase.
    This receipt is current free space, not a full training peak prediction.
    """
    usage = shutil.disk_usage(work)
    floor = 20 * 1024**3
    reasons = []
    if usage.free < floor or usage.used / usage.total >= .95:
        reasons.append("WORK_STORAGE_CRITICAL")
    return dict(status="PASS" if not reasons else "BLOCKED", reasons=reasons,
                storage_path=str(Path(work).resolve()), storage_device_id=Path(work).stat().st_dev,
                storage_shares_root_device=Path(work).stat().st_dev == Path("/").stat().st_dev,
                total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free,
                unchanged_storage_floor_bytes=floor, recorded_utc=datetime.now(timezone.utc).isoformat())


def run_phase(work, name, command, env, probe):
    folder = work / "phases" / name; folder.mkdir(parents=True, exist_ok=False)
    health = probe(folder / "preflight.json", required_memory_gib=32)
    if health["status"] != "PASS": raise RuntimeError("LDNS resource admission failed")
    storage = work_storage_health(work)
    write(folder / "work_storage_preflight.json", storage)
    if storage["status"] != "PASS":
        raise RuntimeError("LDNS work storage admission failed before phase child")
    started = time.monotonic()
    with (folder / "command.log").open("w") as log:
        child = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        def stop(signum, frame):
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
            write(folder / "receipt.json",dict(status="INTERRUPTED",signal=signum,exit_code=child.returncode,
                                              command=command,seconds=time.monotonic()-started))
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        write(folder / "receipt.json",dict(status="RUNNING",pid=child.pid,command=command))
        while child.poll() is None:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                health = probe(folder / "watchdog.json", required_memory_gib=8)
                storage = work_storage_health(work)
                write(folder / "work_storage_watchdog.json", storage)
                if health["status"] != "PASS" or storage["status"] != "PASS":
                    reasons = list(health.get("reasons", [])) + storage["reasons"]
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
                    write(folder/"receipt.json",dict(status="RESOURCE_BLOCKED",exit_code=child.returncode,
                          command=command,seconds=time.monotonic()-started,reasons=reasons))
                    raise RuntimeError("LDNS resource watchdog stopped child")
    write(folder/"receipt.json",dict(status="PASS" if child.returncode==0 else "FAILED",
          command=command,exit_code=child.returncode,seconds=time.monotonic()-started))
    if child.returncode:raise RuntimeError("LDNS phase failed: "+name)


def reuse_inputs(repo, preparation, source, work, stager):
    """Verify complete inputs from a root-storage rejection before training.

    The new cell reads the retained inputs directly. It owns all new phase and
    result files; the failed cell remains unchanged.
    """
    source = Path(source).resolve()
    if (not any(root in source.parents for root in (Path('/tmp/neuronsim'), Path('/data/neuronsim')))
            or source == work.resolve()):
        raise ValueError('LDNS reused inputs require separate node-local scratch')
    forbidden = ('training', 'forecasts', 'archive_complete', 'phases/train/receipt.json')
    if any((source / name).exists() for name in forbidden):
        raise ValueError('LDNS input reuse accepts only a failed cell before training')
    failure = source / 'phases/train/preflight.json'
    health = json.loads(failure.read_text())
    if (health['status'] != 'BLOCKED' or health['reasons'] != ['ROOT_STORAGE_CRITICAL']
            or health['hostname'] != 'gpu03'):
        raise ValueError('LDNS input reuse requires the original root-storage rejection')
    if digest(source / 'preparation.json') != digest(preparation / 'preparation.json'):
        raise ValueError('LDNS reused preparation identity changed')
    cohort = repo / stager.COHORT
    receipt_path = cohort / 'delivery.json'
    delivery_digest = digest(receipt_path)
    inputs = json.loads((source / 'inputs.json').read_text())
    receipt = json.loads(receipt_path.read_text())
    required = dict(status='VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE',
        completed_episodes=72, cortical_units=100000, episode_duration_ms=6000.0,
        source_dt_ms=.1, field_bin_ms=5., native_rng_family='per_edge_philox4_32_10')
    if (any(receipt[key] != value for key, value in required.items())
            or inputs['delivery_sha256'] != delivery_digest
            or digest(source / 'prerequisites/cohort_delivery.json') != delivery_digest
            or Path(inputs['source_root']).resolve() != cohort.resolve()
            or inputs['source_identity'] != receipt['independent_field_replay']):
        raise ValueError('LDNS reused cohort delivery identity changed')
    index = stager.manifest_rows(receipt)
    declared = stager.manifest_rows(inputs)
    design_path = source / 'inputs/train/design.json'
    if digest(design_path) != index['train/design.json']['sha256']:
        raise ValueError('LDNS reused design identity changed')
    design = json.loads(design_path.read_text())
    cases = design['cases']
    if (design['acquired_split'] != 'train' or len(cases) != 72
            or {c['case_id'] for c in cases} != {
                f'train_block_{block:03d}_familiar_{condition}' for block in range(18) for condition in range(4)}
            or any(c['case_id'] != f"train_block_{c['block']:03d}_{c['template_id']}" for c in cases)):
        raise ValueError('LDNS reused complete training identities required')
    expected = {'train/design.json', 'train/acquisition.json',
                'train_fields/dataset.json', 'train_fields/episodes.csv'}
    for case in cases:
        expected.update(f"train/{case['case_id']}/{name}" for name in
                        ('report.json', 'spikes.npz', 'initial_parameters.npz'))
        expected.update(f"train_fields/{case['case_id']}/{name}" for name in
                        ('summary.json', 'spatial_fields.npz', 'unit_rates.npz'))
    if set(declared) != expected:
        raise ValueError('LDNS reused complete input manifest required')
    for name, row in declared.items():
        if row != index[name]:
            raise ValueError('LDNS reused input manifest identity changed: ' + name)
        path = source / 'inputs' / name
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('LDNS reused input file changed: ' + name)
    for name in ('inputs.json', 'prerequisites/cohort_delivery.json'):
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    write(work / 'prerequisites/input_reuse.json', dict(
        status='VERIFIED_COMPLETE_PRETRAINING_INPUT_REUSE', source=str(source),
        original_job_id=health['slurm_job_id'], original_failure_sha256=digest(failure),
        preparation_sha256=digest(preparation / 'preparation.json'),
        cohort_delivery_sha256=delivery_digest, verified_files=len(declared),
        verified_bytes=sum(row['bytes'] for row in declared.values()),
        original_cell_modified=False, training_state_resumed=False))
    return source / 'inputs'


RESOURCE_RECOVERY_SCHEMA = "neural_population_latent_spike_diffusion_resource_recovery_v1"


def _validate_resource_recovery(work, migration, seed, pointer, files):
    """Bind a copied resource interruption to its original phase and inputs."""
    source_job = migration.get("source_job")
    if (migration.get("status") != "PASS"
            or migration.get("source_node") not in {"gpu03", "cpu03", "cpu04"}
            or type(migration.get("training_seed")) is not int
            or migration["training_seed"] != seed
            or not isinstance(source_job, str) or not source_job.isdecimal()
            or int(source_job) < 1):
        raise ValueError("LDNS resource recovery lineage differs")
    source_work = Path(migration.get("source_work", ""))
    input_root = Path(migration.get("input_root", ""))
    destination = work.resolve()
    if (not source_work.is_absolute() or not input_root.is_absolute()
            or source_work.resolve() == destination
            or source_work.resolve() in destination.parents
            or destination in source_work.resolve().parents
            or input_root.resolve() == destination
            or destination in input_root.resolve().parents):
        raise ValueError("LDNS resource recovery requires separate original work and inputs")
    required = {"preparation.json", "inputs.json", "input_staging_preflight.json",
                "storage_reserve.json", "prerequisites/cohort_delivery.json",
                "prerequisites/cuda_delivery.json", "prerequisites/campaign_launcher.py",
                "phases/train/preflight.json", "phases/train/receipt.json",
                "training/committed.json", "training/" + pointer["checkpoint"]}
    if not required <= files:
        raise ValueError("LDNS resource recovery requires authenticated original phase and checkpoint")
    preflight = json.loads((work / "phases/train/preflight.json").read_text())
    if (preflight.get("status") != "PASS"
            or preflight.get("hostname") != migration["source_node"]
            or str(preflight.get("slurm_job_id")) != source_job):
        raise ValueError("LDNS resource recovery original job identity differs")
    receipt = json.loads((work / "phases/train/receipt.json").read_text())
    reasons = receipt.get("reasons")
    if (receipt.get("status") != "RESOURCE_BLOCKED"
            or receipt.get("exit_code") != -signal.SIGTERM
            or not isinstance(reasons, list) or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)):
        raise ValueError("LDNS resource recovery requires the original watchdog interruption")
    expected = ["-B", "-m", "Libs.population_dynamics.latent_spike_diffusion_training",
                "--acquisition", str(input_root / "train"),
                "--fields", str(input_root / "train_fields"),
                "--config", str(source_work / "Libs/configs/neural_population_latent_spike_diffusion.yaml"),
                "--seed", str(seed), "--output", str(source_work / "training")]
    command = receipt.get("command")
    if (not isinstance(command, list) or any(not isinstance(arg, str) for arg in command)
            or not command or not command[0] or command[1:] != expected):
        raise ValueError("LDNS resource recovery original training arguments differ")
    config = yaml.safe_load((work / "Libs/configs/neural_population_latent_spike_diffusion.yaml").read_text())
    if (config.get("maximum_autoencoder_epochs") != 260
            or config.get("maximum_diffusion_epochs") != 2000
            or pointer.get("stage") not in {"autoencoder", "diffusion"}
            or type(pointer.get("next_epoch")) is not int
            or not 0 <= pointer["next_epoch"] <= config["maximum_" + pointer["stage"] + "_epochs"]):
        raise ValueError("LDNS resource recovery committed stage or frozen budget differs")


def validate_recovery(work, preparation, seed):
    """Authenticate a transferred cell without changing its trainer binding."""
    record = json.loads((preparation / "preparation.json").read_text())
    if digest(work / "preparation.json") != digest(preparation / "preparation.json"):
        raise ValueError("LDNS recovery preparation changed")
    for name, expected in record["files"].items():
        if digest(work / name) != expected:
            raise ValueError("LDNS recovery software changed: " + name)
    migration = json.loads((work / "migration.json").read_text())
    resource_recovery = migration.get("schema") == RESOURCE_RECOVERY_SCHEMA
    if (not resource_recovery and
            (migration["status"] != "PASS" or migration["source_node"] != "gpu03"
             or migration["source_job"] != "617154" or seed != 41)):
        raise ValueError("LDNS recovery requires the registered cancelled seed41 cell")
    files = set()
    for row in migration["files"]:
        name = row["path"]
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or ".." in Path(name).parts or name in files):
            raise ValueError("LDNS migrated file path differs")
        files.add(name)
        path = work / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("LDNS migrated file changed: " + name)
    pointer = json.loads((work / "training/committed.json").read_text())
    checkpoint = pointer.get("checkpoint")
    if (not isinstance(checkpoint, str) or not checkpoint
            or Path(checkpoint).is_absolute() or ".." in Path(checkpoint).parts
            or pointer != migration["committed"]
            or digest(work / "training" / checkpoint) != pointer["sha256"]):
        raise ValueError("LDNS recovery committed epoch identity changed")
    if resource_recovery:
        _validate_resource_recovery(work, migration, seed, pointer, files)
    elif pointer["stage"] != "autoencoder" or pointer["next_epoch"] != 198:
        raise ValueError("LDNS recovery committed epoch identity changed")
    # The original CUDA validation admitted this exact prepared trainer. The
    # trainer itself rechecks config, source, cohort and torch bindings on load.
    gate = json.loads((work / "prerequisites/cuda_delivery.json").read_text())
    bindings = {"trainer_sha256": "Libs/population_dynamics/latent_spike_diffusion_training.py"}
    if resource_recovery:
        bindings.update(adapter_sha256="Libs/population_dynamics/latent_spike_diffusion.py",
                        device_check_sha256="Libs/population_dynamics/latent_spike_diffusion_device_check.py",
                        configuration_sha256="Libs/configs/neural_population_latent_spike_diffusion.yaml")
    if gate["status"] != "PASS" or any(gate.get(key) != digest(work / name) for key, name in bindings.items()):
        raise ValueError("LDNS recovery CUDA qualification changed")
    return record, migration


def resume_campaign(repo, preparation, work, seed):
    record, migration = validate_recovery(work, preparation, seed)
    input_root = Path(migration["input_root"]) if migration.get("schema") == RESOURCE_RECOVERY_SCHEMA else work / "inputs"
    output = repo / record["output_root"] / f"seed_{seed}"
    if output.exists():
        raise FileExistsError("existing LDNS seed evidence must be reused: " + str(output))
    probe = module(work / "Libs/population_dynamics/resource_probe.py", "ldns_recovery_probe").probe
    gate = repo / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    receipt = json.loads((work / "prerequisites/cuda_delivery.json").read_text())
    check_storage_reserve(work, work / "Libs/configs/neural_population_latent_spike_diffusion.yaml",
                          receipt, gate, reuse_inputs_from=input_root)
    shutil.copy2(Path(__file__), work / "prerequisites/recovery_launcher.py")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(work), str(work / "Libs/external/LDNS")]),
        TMPDIR=str(work / "scratch"), PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
        NVIDIA_TF32_OVERRIDE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    common = ["--acquisition", str(input_root / "train"), "--fields", str(input_root / "train_fields"),
              "--config", str(work / "Libs/configs/neural_population_latent_spike_diffusion.yaml"),
              "--seed", str(seed)]
    train = [sys.executable, "-B", "-m", "Libs.population_dynamics.latent_spike_diffusion_training",
             *common, "--output", str(work / "training"), "--resume"]
    evaluate = [sys.executable, "-B", "-m", "Libs.population_dynamics.latent_spike_diffusion_evaluation",
        *common, "--training", str(work / "training"), "--fitting-scale", str(work / "fitting_field_scale.npz"),
        "--output", str(work / "forecasts")]
    for name, command in (("train_resume_" + os.environ["SLURM_JOB_ID"], train),
                          ("evaluate", evaluate), ("replay", [*evaluate, "--replay"])):
        run_phase(work, name, command, env, probe)
    if json.loads((work / "forecasts/replay.json").read_text())["status"] != "PASS":
        raise ValueError("LDNS independent recovery replay required")
    output.mkdir(parents=True)
    files = []
    roots = [work / name for name in ("training", "forecasts", "phases", "prerequisites")]
    sources = [path for root in roots for path in sorted(root.rglob("*")) if path.is_file()]
    sources += [work / name for name in ("preparation.json", "inputs.json", "migration.json",
                                        "input_staging_preflight.json", "storage_reserve.json")]
    for source in sources:
        name = str(source.relative_to(work))
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected = digest(source)
        if digest(target) != expected:
            raise ValueError("LDNS recovery delivery changed: " + name)
        files.append(dict(path=name, bytes=source.stat().st_size, sha256=expected))
    write(output / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
        job_id=os.environ["SLURM_JOB_ID"], files=files, resumed_from_job=migration["source_job"],
        resumed_from_training_stage=migration["committed"]["stage"],
        resumed_from_stage_epochs=migration["committed"]["next_epoch"],
        resumed_from_autoencoder_epochs=(migration["committed"]["next_epoch"]
            if migration["committed"]["stage"] == "autoencoder" else None), scientific_claim_qualified=False,
        recorded_utc=datetime.now(timezone.utc).isoformat()))
    (work / "archive_complete").write_text(str(output) + "\n")
    return output


def run(repo, preparation, work, seed, reuse_inputs_from=None, resume=False):
    repo, preparation, work = map(Path,(repo,preparation,work))
    if seed not in (41,42,43):raise ValueError("registered training seed41/42/43 required")
    if ((os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in {("gpu03", "gpu1"), ("cpu03", "gpu"), ("cpu04", "cpu")}
            or not os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("LDNS campaign requires an allocated GPU on an authorized node")
    if resume:
        return resume_campaign(repo, preparation, work, seed)
    if reuse_inputs_from is not None:
        source, destination=Path(reuse_inputs_from).resolve(),work.resolve()
        if source==destination or source in destination.parents or destination in source.parents:
            raise ValueError('LDNS new work must be separate from retained inputs')
    if work.exists():raise FileExistsError("new LDNS campaign cell required")
    work.mkdir(parents=True);(work/"scratch").mkdir();(work/"prerequisites").mkdir()
    record=json.loads((preparation/"preparation.json").read_text())
    if (record["training_seeds"]!=[41,42,43] or record["status"]!="VALIDATED_LDNS_CAUSAL_TRAINING_EVALUATION"
            or record["training_tests_passed"]!=5 or record["evaluation_tests_passed"]!=3
            or record["adapter_tests_passed"]!=4 or record["real_episode_update_passed"] is not True):
        raise ValueError("complete LDNS preparation gates required")
    output=repo/record["output_root"]/f"seed_{seed}"
    if output.exists():raise FileExistsError('existing LDNS seed evidence must be reused: '+str(output))
    for name,expected in record["files"].items():
        source=preparation/name
        if digest(source)!=expected:raise ValueError("LDNS prepared source changed: "+name)
        destination=work/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        if digest(destination)!=expected:raise ValueError("LDNS source staging differs: "+name)
    gate=repo/"Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012"
    receipt=json.loads((gate/"delivery.json").read_text())
    if (receipt["status"]!="PASS"
            or receipt["trainer_sha256"]!=digest(work/"Libs/population_dynamics/latent_spike_diffusion_training.py")
            or receipt["adapter_sha256"]!=digest(work/"Libs/population_dynamics/latent_spike_diffusion.py")
            or receipt["device_check_sha256"]!=digest(work/"Libs/population_dynamics/latent_spike_diffusion_device_check.py")
            or receipt["configuration_sha256"]!=digest(work/"Libs/configs/neural_population_latent_spike_diffusion.yaml")):
        raise ValueError("exact full-unit CUDA qualification required before campaign")
    for row in receipt["files"]:
        if digest(gate/row["path"])!=row["sha256"]:raise ValueError("CUDA validation evidence changed")
    if "check.json" not in {row["path"] for row in receipt["files"]}:
        raise ValueError("CUDA validation must bind its optimizer update report")
    updates=json.loads((gate/"check.json").read_text())["diffusion_updates"]
    if (updates["optimizer_steps"]!=2 or updates["scheduler_steps"]!=2
            or updates["initial_zero_rate_parameters_unchanged"] is not True
            or updates["changed_parameter_tensors"]<1
            or not 0<float(updates["maximum_parameter_change"])<float("inf")):
        raise ValueError("CUDA validation must demonstrate a denoiser parameter update")
    shutil.copy2(gate/"delivery.json",work/"prerequisites/cuda_delivery.json")
    shutil.copy2(preparation/"preparation.json",work/"preparation.json")
    shutil.copy2(Path(__file__),work/"prerequisites/campaign_launcher.py")
    probe=module(work/"Libs/population_dynamics/resource_probe.py","ldns_resource_probe").probe
    # Admit the node before copying the complete cohort. A failed admission
    # must not fill root storage further or start a training child.
    health=probe(work/"input_staging_preflight.json",required_memory_gib=32)
    if health["status"]!="PASS":
        raise RuntimeError("LDNS resource admission failed before input staging")
    check_storage_reserve(work, work/"Libs/configs/neural_population_latent_spike_diffusion.yaml",
                          receipt, gate, reuse_inputs_from=reuse_inputs_from)
    inputs=module(work/"Libs/scripts/run_population_trajectory_campaign.py","ldns_input_staging")
    if reuse_inputs_from is None:
        inputs.stage_inputs(repo,work)
        input_root=work/"inputs"
    else:
        input_root=reuse_inputs(repo,preparation,reuse_inputs_from,work,inputs)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(work),str(work/"Libs/external/LDNS")]),
        TMPDIR=str(work/"scratch"),PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",
        NVIDIA_TF32_OVERRIDE="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    common=["--acquisition",str(input_root/"train"),"--fields",str(input_root/"train_fields"),
            "--config",str(work/"Libs/configs/neural_population_latent_spike_diffusion.yaml"),"--seed",str(seed)]
    train=[sys.executable,"-B","-m","Libs.population_dynamics.latent_spike_diffusion_training",*common,
           "--output",str(work/"training")]
    evaluate=[sys.executable,"-B","-m","Libs.population_dynamics.latent_spike_diffusion_evaluation",*common,
        "--training",str(work/"training"),"--fitting-scale",str(work/"fitting_field_scale.npz"),
        "--output",str(work/"forecasts")]
    for name,command in (("train",train),("evaluate",evaluate),("replay",[*evaluate,"--replay"])):
        run_phase(work,name,command,env,probe)
    report=json.loads((work/"forecasts/replay.json").read_text())
    if report["status"]!="PASS":raise ValueError("LDNS independent forecast replay required")
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);files=[]
    # Source and native inputs are already bound by preparation and cohort
    # receipts. Archive result/checkpoint/phase evidence without duplicating data.
    for root in ("training","forecasts","phases"):
        for source in sorted((work/root).rglob("*")):
            if not source.is_file():continue
            name=source.relative_to(work);destination=output/name
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
            expected=digest(source)
            if digest(destination)!=expected:raise ValueError("LDNS delivery differs: "+str(name))
            files.append(dict(path=str(name),bytes=source.stat().st_size,sha256=expected))
    receipts=["preparation.json","inputs.json","input_staging_preflight.json","storage_reserve.json",
              "prerequisites/cohort_delivery.json","prerequisites/cuda_delivery.json",
              "prerequisites/campaign_launcher.py"]
    if reuse_inputs_from is not None:receipts.append("prerequisites/input_reuse.json")
    for name in receipts:
        source=work/name;destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        expected=digest(source)
        if digest(destination)!=expected:raise ValueError("LDNS receipt delivery differs")
        files.append(dict(path=name,bytes=source.stat().st_size,sha256=expected))
    write(output/"delivery.json",dict(status="PASS",exit_code=0,training_seed=seed,
        job_id=os.environ["SLURM_JOB_ID"],files=files,scientific_claim_qualified=False,
        recorded_utc=datetime.now(timezone.utc).isoformat()))
    (work/"archive_complete").write_text(str(output)+"\n")
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","preparation","work"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--reuse-inputs-from",type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed",type=int,required=True);args=parser.parse_args()
    print(run(args.repo,args.preparation,args.work,args.seed,args.reuse_inputs_from,args.resume))


if __name__=="__main__":main()

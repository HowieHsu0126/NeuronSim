"""Evaluate fitted native residual corrections without retraining the parent.

The legacy ridge route runs inside its existing Slurm allocation. The native
refit route runs on Merlin. Both execute the matched evaluator and independent
all-origin replay in fresh processes for each correction mode.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

SEEDS = (41, 42, 43)
MODES = ("current", "history")
FORECAST = "neural_population_matched_forecast.yaml"
GPU_NODES = {"cpu03", "cpu04", "cpu05", "cpu06", "gpu03"}
CPU_PLACEMENTS = {("cpu03", "gpu"), ("cpu04", "cpu")}


def campaign_helper(repo):
    path = repo / "Libs/scripts/run_population_trajectory_campaign.py"
    spec = importlib.util.spec_from_file_location("native_trajectory_campaign_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_receipt(helper, source, target, expected_sha256=None):
    row = dict(bytes=source.stat().st_size, sha256=expected_sha256 or helper.digest(source))
    helper.copy_bound(source, target, row)
    return row["sha256"]


def stage(helper, repo, preparation, fit_job, seed, work):
    execution_path = preparation / f"execution_{fit_job}.json"
    results = preparation / f"results_{fit_job}"
    fit_path = results / "fit.json"
    execution = json.loads(execution_path.read_text())
    fit = json.loads(fit_path.read_text())
    if (execution.get("exit_code") != 0 or str(execution.get("job_id")) != fit_job
            or fit.get("status") != "FITTED_ALL_NATIVE_RESIDUAL_CORRECTIONS"
            or str(fit.get("job_id")) != fit_job or fit.get("parent_training") is not False
            or fit.get("selection_arrays_accessed") is not False):
        raise ValueError("completed fitting-only correction job required")
    config = fit["config"]
    alpha_by_mode = fit.get("alpha_by_mode", {mode: config.get("alpha") for mode in MODES})
    if (config.get("training_seeds") != list(SEEDS) or config.get("modes") != list(MODES)
            or config.get("source_units") != 100000 or config.get("cpu_threads") != 2
            or config.get("parent_training") is not False
            or set(alpha_by_mode) != set(MODES)
            or any(isinstance(value, bool) or value not in (.0001, .01, 1., 100.) for value in alpha_by_mode.values())
            or config.get("fitting_blocks") != list(range(14))
            or config.get("excluded_selection_blocks") != list(range(14, 18))):
        raise ValueError("registered three-seed, two-mode fitting contract required")
    records = fit["corrections"]
    expected = {(registered_seed, mode) for registered_seed in SEEDS for mode in MODES}
    if len(records) != 6 or {(row["seed"], row["mode"]) for row in records} != expected:
        raise ValueError("exactly six distinct registered seed/mode exports required")
    manifests = {}
    for row in records:
        relative = f"seed_{row['seed']}/{row['mode']}"
        if row["path"] != relative or row["alpha"] != alpha_by_mode[row["mode"]]:
            raise ValueError("correction export path or alpha differs")
        path = results / relative / "manifest.json"
        if helper.digest(path) != row["manifest_sha256"]:
            raise ValueError("fitted correction manifest changed: " + relative)
        manifest = json.loads(path.read_text())
        identity = manifest["identity"]
        if (manifest["status"] != "FITTED_NATIVE_RESIDUAL_CORRECTION"
                or identity["seed"] != row["seed"] or identity["mode"] != row["mode"]
                or identity["alpha"] != row["alpha"]
                or identity["software_sources"] != fit["software_sources"]):
            raise ValueError("fitted correction export identity differs")
        manifests[(row["seed"], row["mode"])] = (row, manifest)

    package = work / "Libs/population_dynamics"
    snapshot = preparation / f"software_{fit_job}"
    shutil.copytree(snapshot, package, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name, digest in fit["software_sources"].items():
        if Path(name).name != name or helper.digest(package / name) != digest:
            raise ValueError("tested correction source snapshot changed: " + name)
    for name in ("population_trajectory_evaluation.py", "population_forecast_replay.py"):
        if not (package / name).is_file():
            raise ValueError("tested native evaluation source missing: " + name)

    parent = preparation.parent / "campaign" / f"seed_{seed}"
    delivery_path = parent / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    required = dict(status="PASS", exit_code=0, training_seed=seed, source_units=100000,
                    fit_episodes=56, validation_episodes=16, complete_origins=48,
                    verified_origins=48, score_rows=144)
    if any(delivery.get(key) != value for key, value in required.items()):
        raise ValueError("complete same-seed native parent delivery required")
    parent_files = helper.manifest_rows(delivery)
    # Resource monitoring is operational software retained in the accepted parent.
    probe_relative = "Libs/population_dynamics/resource_probe.py"
    helper.copy_bound(parent / probe_relative, package / "resource_probe.py", parent_files[probe_relative])
    parent_training = work / "inputs/parent_training"
    for name in ("run.json", "best.pt"):
        helper.copy_bound(parent / "training" / name, parent_training / name, parent_files["training/" + name])
    forecast_relative = "Libs/configs/" + FORECAST
    forecast_path = work / forecast_relative
    helper.copy_bound(parent / forecast_relative, forecast_path, parent_files[forecast_relative])
    forecast = yaml.safe_load(forecast_path.read_text())
    if (forecast.get("origins_ms") != [1000., 3000., 5000.]
            or forecast.get("horizons_ms") != [20., 80., 500.]
            or forecast.get("joint_samples") != 16 or forecast.get("formal_evaluation") is not False):
        raise ValueError("frozen 48-origin, 16-draw, 20/80/500ms protocol required")
    run = json.loads((parent_training / "run.json").read_text())
    parent_digest = parent_files["training/best.pt"]["sha256"]
    if (run.get("status") != "FITTED_DEVELOPMENT" or run.get("training_seed") != seed
            or run["binding"]["seed"] != seed or run["best_checkpoint_sha256"] != parent_digest
            or run["binding"]["config"]["cpu_threads"] != 2):
        raise ValueError("same-seed complete parent checkpoint and two-thread execution required")
    for key, name in (
        ("model_implementation_sha256", "population_state.py"),
        ("observation_implementation_sha256", "refractory_observation.py"),
        ("loader_implementation_sha256", "native_renewal_data.py"),
        ("latent_model_implementation_sha256", "poisson_state_space.py"),
        ("native_counts_implementation_sha256", "native_count_windows.py"),
    ):
        if helper.digest(package / name) != run["binding"][key]:
            raise ValueError("frozen parent implementation differs in tested snapshot: " + name)

    identities = {}
    for mode in MODES:
        row, manifest = manifests[(seed, mode)]
        identity = manifest["identity"]
        if identity["parent_checkpoint_sha256"] != parent_digest:
            raise ValueError("correction was fitted against a different parent checkpoint")
        source = results / row["path"]
        target = work / "corrections" / mode
        copy_receipt(helper, source / "manifest.json", target / "manifest.json", row["manifest_sha256"])
        copy_receipt(helper, source / "predictor.npz", target / "predictor.npz", identity["predictor_sha256"])
        identities[mode] = identity

    provenance = work / "provenance"
    receipts = dict(
        execution_sha256=copy_receipt(helper, execution_path, provenance / "fit_execution.json"),
        fit_sha256=copy_receipt(helper, fit_path, provenance / "fit.json"),
        parent_delivery_sha256=copy_receipt(helper, delivery_path, provenance / "parent_delivery.json"),
        parent_checkpoint_sha256=parent_digest, forecast_sha256=parent_files[forecast_relative]["sha256"],
        fit_job_id=fit_job, source_snapshot=str(snapshot), parent_campaign=str(parent),
        correction_software_sources=fit["software_sources"],
        evaluator_sha256=helper.digest(package / "population_trajectory_evaluation.py"),
        replay_sha256=helper.digest(package / "population_forecast_replay.py"),
        resource_probe_sha256=parent_files[probe_relative]["sha256"],
    )
    for name in ("run_population_trajectory_campaign.py", "run_population_residual_correction.py",
                 "run_population_residual_correction.sbatch"):
        copy_receipt(helper, repo / "Libs/scripts" / name, provenance / name)
    helper.write_json(provenance / "staging.json", receipts)
    return forecast, identities, receipts


def verify_mode(helper, directory, mode, seed, forecast, identity, parent_digest):
    evaluation = json.loads((directory / "forecasts/evaluation.json").read_text())
    replay = json.loads((directory / "replay.json").read_text())
    method = "population_residual_" + mode
    values = 48 * 16 * 100 * 100000
    if (evaluation.get("status") != "EVALUATED_DEVELOPMENT" or evaluation.get("model_type") != "population"
            or evaluation.get("method") != method or evaluation.get("training_seed") != seed
            or evaluation.get("checkpoint_sha256") != parent_digest
            or evaluation.get("correction_identity") != identity or evaluation.get("configuration") != forecast
            or evaluation.get("episodes") != 16 or evaluation.get("origins") != 48
            or evaluation.get("score_rows") != 144 or evaluation.get("generated_unit_count_values") != values):
        raise ValueError("complete matched corrected evaluation required")
    costs = evaluation["origin_costs"]
    if len(costs) != 48 or any(
        row["unit_count"] != 100000 or row["field_count"] != 128 or row["draws"] != 16
        or row["generated_bins"] != 100 or row["horizons_ms"] != [20., 80., 500.] for row in costs
    ):
        raise ValueError("every origin must include all 100000 units and 128 source fields")
    if (replay.get("status") != "PASS" or replay.get("method") != method
            or replay.get("training_seed") != seed or replay.get("correction_identity") != identity
            or replay.get("source_checkpoint_sha256") != parent_digest
            or replay.get("verified_origins") != 48 or replay.get("expected_origins") != 48
            or replay.get("verified_score_rows") != 144
            or replay.get("all_forecast_unit_metrics_replayed") is not True
            or replay.get("all_origin_fields_and_latents_exact") is not True
            or replay.get("sampled_unit_count_values_consumed") != values):
        raise ValueError("PASS independent all-origin unit replay required")
    return dict(status="PASS", exit_code=0, method=method, verified_origins=48, score_rows=144,
                generated_unit_count_values=values, scientific_claim_qualified=False,
                evaluation_sha256=helper.digest(directory / "forecasts/evaluation.json"),
                replay_sha256=helper.digest(directory / "replay.json"))


def merlin_identity():
    identity = dict(backend="merlin", run_id=os.environ.get("RH2_JOB_RUN_ID"),
                    trial_id=os.environ.get("ARNOLD_TRIAL_ID"), host=platform.node())
    if (not identity["run_id"] or not identity["trial_id"]
            or identity["host"].split(".")[0] == "mgmt2"):
        raise ValueError("native residual generation requires actual Merlin run/trial/host")
    return identity


def native_probe(output, required_memory_gib=32, *, device):
    """Record container resources without substituting a Slurm allocation ID."""
    report = dict(**merlin_identity(), recorded_utc=datetime.now(timezone.utc).isoformat(),
                  device=device, python=platform.python_version(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), required_memory_gib=required_memory_gib)
    memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    available = memory["MemAvailable"]
    for maximum, current in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(maximum).is_file() and Path(current).is_file():
            limit = Path(maximum).read_text().strip()
            if limit != "max":
                used = int(Path(current).read_text())
                memory.update(cgroup_limit=int(limit), cgroup_current=used)
                available = min(available, max(0, int(limit) - used))
            break
    memory["effective_available"] = available
    disks = {name: dict(zip(("total", "used", "free"), shutil.disk_usage(path)))
             for name, path in (("root", Path("/")), ("work", output.parent))}
    reasons = []
    if available < required_memory_gib * 1024**3:
        reasons.append("INSUFFICIENT_AVAILABLE_MEMORY")
    if any(row["free"] < 20 * 1024**3 or row["used"] / row["total"] >= .95 for row in disks.values()):
        reasons.append("STORAGE_CRITICAL")
    if device == "cuda":
        command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                   "--format=csv,noheader,nounits"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            command += ["-i", visible]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            report["gpu_query"] = dict(command=command, exit_code=result.returncode,
                                       stderr=result.stderr.strip())
            rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
            if result.returncode or not rows or any(len(row) != 5 for row in rows):
                reasons.append("GPU_TELEMETRY_FAILED")
            else:
                report["gpus"] = [dict(index=row[0], name=row[1], total_memory_mib=float(row[2]),
                    free_memory_mib=float(row[3]), utilization_percent=float(row[4])) for row in rows]
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            reasons.append("GPU_TELEMETRY_FAILED")
            report["gpu_query_error"] = str(error)
    report.update(status="PASS" if not reasons else "BLOCKED", reasons=reasons,
                  memory_bytes=memory, disk_bytes=disks)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Match the campaign helper's atomic JSON writes for its phase receipts.
    temporary = output.with_suffix(output.suffix + ".writing")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def input_bundle(helper, root, scope, blocks):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "native_residual_observation_inputs_v1"
            or manifest.get("scope") != scope or manifest.get("blocks") != list(blocks)
            or manifest.get("training_seeds") != list(SEEDS) or manifest.get("source_units") != 100000):
        raise ValueError("registered native input bundle required: " + scope)
    rows = helper.manifest_rows(manifest)
    for name, row in rows.items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or not path.resolve().is_relative_to(root)):
            raise ValueError("input path escapes staging root: " + name)
        if path.stat().st_size != row["bytes"] or helper.digest(path) != row["sha256"]:
            raise ValueError("staged input identity differs: " + name)
    # transfer_complete/local_payload_checks_performed describe manifest creation.
    # The byte checks above establish admission without rewriting the fit binding.
    return manifest, rows


def stage_native(helper, repo, fitting_inputs, generation_inputs, refits, fitting_config, seed, work):
    # The legacy route can still run from a snapshot without this native-only module.
    sys.path.insert(0, str(repo / "Libs"))
    from population_dynamics.population_residual_refit_provenance import build_refit_receipt, refit_identity

    _, fitting_rows = input_bundle(helper, fitting_inputs, "complete_fitting_and_internal_selection", range(14))
    generation, generation_rows = input_bundle(helper, generation_inputs, "native_generation_additional_inputs", range(14, 18))
    family = build_refit_receipt(refits)
    shared = family["shared_binding"]
    configuration = yaml.safe_load(fitting_config.read_text())
    if (shared["input_manifest_sha256"] != helper.digest(fitting_inputs / "manifest.json")
            or shared["config_sha256"] != helper.digest(fitting_config) or family["config"] != configuration):
        raise ValueError("native refits differ from the fitting input/configuration identities")

    sources = shared["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("native refit source identities required")
    for name, digest in sources.items():
        if Path(name).name != name or not name.endswith(".py") or helper.digest(repo / "Libs/population_dynamics" / name) != digest:
            raise ValueError("native refit runtime source differs: " + name)
    modules = ["population_trajectory_evaluation", "population_forecast_replay", "population_residual_refit_provenance"]
    names = set(helper.software_closure(repo, modules)) | {"Libs/population_dynamics/" + name for name in sources}
    for name in sorted(names):
        copy_receipt(helper, repo / name, work / name)

    cohort_path = fitting_inputs / "cohort/delivery.json"
    cohort = json.loads(cohort_path.read_text())
    required = dict(status="VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE", completed_episodes=72,
                    cortical_units=100000, episode_duration_ms=6000., source_dt_ms=.1,
                    field_bin_ms=5., native_rng_family="per_edge_philox4_32_10")
    if (any(cohort.get(key) != value for key, value in required.items())
            or generation.get("cohort_delivery_sha256") != helper.digest(cohort_path)):
        raise ValueError("complete common native Philox development cohort required")
    cohort_rows = helper.manifest_rows(cohort)
    cases = [f"train_block_{block:03d}_familiar_{template}" for block in range(18) for template in range(4)]
    names = ["train/design.json", "train/acquisition.json", "train_fields/dataset.json", "train_fields/episodes.csv"]
    by_case = lambda case: [f"train/{case}/{name}" for name in ("report.json", "spikes.npz", "initial_parameters.npz")] + [
        f"train_fields/{case}/{name}" for name in ("summary.json", "spatial_fields.npz")]
    expected_generation = {"cohort/" + name for case in cases[56:] for name in by_case(case)}
    forecast_relative = "generation_config/" + FORECAST
    if set(generation_rows) != expected_generation | {forecast_relative}:
        raise ValueError("exactly the sixteen additional source episodes and frozen forecast are required")
    names += [name for case in cases for name in by_case(case)]
    for name in names:
        relative = "cohort/" + name
        root, rows = (generation_inputs, generation_rows) if relative in expected_generation else (fitting_inputs, fitting_rows)
        row, source_row = rows[relative], cohort_rows[name]
        if row["sha256"] != source_row["sha256"] or row["bytes"] != source_row["bytes"]:
            raise ValueError("generation source differs from the accepted cohort: " + name)
        helper.copy_bound(root / relative, work / "inputs" / name, row)
    design = json.loads((work / "inputs/train/design.json").read_text())
    declared_cases = design.get("cases", [])
    if (design.get("acquired_split") != "train" or len(declared_cases) != 72
            or {case["case_id"] for case in declared_cases} != set(cases)
            or any(case["case_id"] != f"train_block_{case['block']:03d}_{case['template_id']}" for case in declared_cases)):
        raise ValueError("complete unique 72-episode development design required")

    forecast_path = generation_inputs / forecast_relative
    forecast_digest = generation_rows[forecast_relative]["sha256"]
    forecast = yaml.safe_load(forecast_path.read_text())
    if (generation.get("forecast_config_sha256") != forecast_digest
            or forecast.get("origins_ms") != [1000., 3000., 5000.]
            or forecast.get("horizons_ms") != [20., 80., 500.]
            or forecast.get("joint_samples") != 16 or forecast.get("formal_evaluation") is not False):
        raise ValueError("frozen 48-origin, 16-draw, 20/80/500ms forecast required")
    copy_receipt(helper, forecast_path, work / "Libs/configs" / FORECAST, forecast_digest)
    provenance = work / "provenance"
    identities, parents = {}, {}
    for registered_seed in SEEDS:
        parent = fitting_inputs / f"parent/seed_{registered_seed}"
        delivery = json.loads((parent / "delivery.json").read_text())
        required = dict(status="PASS", exit_code=0, training_seed=registered_seed, source_units=100000,
                        fit_episodes=56, validation_episodes=16, complete_origins=48, verified_origins=48, score_rows=144)
        if any(delivery.get(key) != value for key, value in required.items()):
            raise ValueError("complete same-seed native parent delivery required")
        parent_rows = helper.manifest_rows(delivery)
        if parent_rows["Libs/configs/" + FORECAST]["sha256"] != forecast_digest:
            raise ValueError("generation must use every parent's frozen matched forecast YAML")
        for name in ("training/run.json", "training/best.pt"):
            row = fitting_rows[f"parent/seed_{registered_seed}/{name}"]
            if any(row[key] != parent_rows[name][key] for key in ("bytes", "sha256")):
                raise ValueError("fitting input parent differs from its accepted delivery")
        run = json.loads((parent / "training/run.json").read_text())
        parent_digest = parent_rows["training/best.pt"]["sha256"]
        if (run.get("status") != "FITTED_DEVELOPMENT" or run.get("training_seed") != registered_seed
                or run["binding"]["seed"] != registered_seed or run["best_checkpoint_sha256"] != parent_digest
                or run["binding"]["config"]["cpu_threads"] != 2
                or run["binding"]["acquisition_design_sha256"] != cohort_rows["train/design.json"]["sha256"]
                or run["binding"]["dataset_sha256"] != cohort_rows["train_fields/dataset.json"]["sha256"]):
            raise ValueError("same-seed complete parent and cohort binding required")
        for key, name in (("model_implementation_sha256", "population_state.py"),
                ("observation_implementation_sha256", "refractory_observation.py"),
                ("loader_implementation_sha256", "native_renewal_data.py"),
                ("latent_model_implementation_sha256", "poisson_state_space.py"),
                ("native_counts_implementation_sha256", "native_count_windows.py")):
            if helper.digest(work / "Libs/population_dynamics" / name) != run["binding"][key]:
                raise ValueError("frozen parent runtime implementation differs: " + name)
        parents[registered_seed] = (parent_digest, hashlib.sha256(json.dumps(run["binding"],
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        if registered_seed == seed:
            for name in ("run.json", "best.pt"):
                helper.copy_bound(parent / "training" / name, work / "inputs/parent_training" / name,
                                  parent_rows["training/" + name])
            copy_receipt(helper, parent / "delivery.json", provenance / "parent_delivery.json")
    for row in family["fits"]:
        cell = refits / f"seed_{row['seed']}" / row["mode"]
        report = json.loads((cell / "fit.json").read_text())
        identity = refit_identity(report, helper.digest(cell / "fit.json"))
        binding = identity["binding"]
        if (identity["fit_sha256"] != row["sha256"]
                or (binding["parent_checkpoint_sha256"], binding.get("parent_binding_sha256")) != parents[row["seed"]]
                or binding.get("fit_cases") != cases[:56]
                or helper.digest(cell / "predictor.npz") != identity["predictor_sha256"]):
            raise ValueError("complete native refit predictor and parent binding required")
        copy_receipt(helper, cell / "fit.json", work / row["path"], row["sha256"])
        if row["seed"] == seed:
            target = work / "corrections" / row["mode"]
            copy_receipt(helper, cell / "fit.json", target / "fit.json", row["sha256"])
            copy_receipt(helper, cell / "predictor.npz", target / "predictor.npz", identity["predictor_sha256"])
            identities[row["mode"]] = identity
    helper.write_json(provenance / "fit.json", family)
    receipts = dict(fit_schema=family["schema"], fit_sha256=helper.digest(provenance / "fit.json"),
        parent_checkpoint_sha256=parents[seed][0], forecast_sha256=forecast_digest,
        parent_delivery_sha256=helper.digest(provenance / "parent_delivery.json"),
        fitting_input_manifest_sha256=copy_receipt(helper, fitting_inputs / "manifest.json", provenance / "fitting_inputs.json"),
        generation_input_manifest_sha256=copy_receipt(helper, generation_inputs / "manifest.json", provenance / "generation_inputs.json"),
        fitting_config_sha256=copy_receipt(helper, fitting_config, provenance / "fitting_config.yaml"),
        cohort_delivery_sha256=copy_receipt(helper, cohort_path, provenance / "cohort_delivery.json"),
        correction_software_sources=sources,
        evaluator_sha256=helper.digest(work / "Libs/population_dynamics/population_trajectory_evaluation.py"),
        replay_sha256=helper.digest(work / "Libs/population_dynamics/population_forecast_replay.py"))
    for name in ("run_population_trajectory_campaign.py", "run_population_residual_correction.py"):
        copy_receipt(helper, repo / "Libs/scripts" / name, provenance / name)
    helper.write_json(provenance / "staging.json", receipts)
    return forecast, identities, receipts


def run_native(repo, fitting_inputs, generation_inputs, refits, fitting_config, seed, work, output, device):
    execution = merlin_identity()
    if seed not in SEEDS:
        raise ValueError("registered training seed required")
    repo, fitting_inputs, generation_inputs, refits, fitting_config, work, output = (
        Path(value).resolve() for value in (repo, fitting_inputs, generation_inputs, refits, fitting_config, work, output))
    if (not work.is_relative_to(Path("/tmp")) or work == Path("/tmp") or work.exists()
            or any(path.is_relative_to(work) or work.is_relative_to(path)
                   for path in (repo, fitting_inputs, generation_inputs, refits))):
        raise ValueError("fresh task-specific local scratch separate from staged inputs required")
    if output.exists() or output.is_relative_to(work) or work.is_relative_to(output):
        raise FileExistsError("a fresh output outside scratch is required")
    helper = campaign_helper(repo)
    work.mkdir(parents=True)
    (work / "scratch").mkdir()
    (work / "phases").mkdir()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    probe = partial(native_probe, device=device)
    started = time.perf_counter()
    record = dict(status="RUNNING", exit_code=None, **execution, training_seed=seed, device=device,
        parent_training=False, source_units=100000, core_neurons=100000, complete_episodes=72,
        fit_episodes=56, validation_episodes=16, scientific_claim_qualified=False,
        formal_evaluation=False, calibration_or_test_accessed=False, matched_hardware_speedup_qualified=False,
        started_utc=datetime.now(timezone.utc).isoformat(),
        modes={mode: dict(status="NOT_RUN", exit_code=None) for mode in MODES})

    def interrupted(signum, frame):
        record["received_signal"] = signum
        raise InterruptedError(f"received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 2
    try:
        health = probe(work / "phases/staging_preflight.json")
        record["resources"] = health
        if health["status"] != "PASS":
            raise helper.ResourceBlocked(health["reasons"])
        forecast, identities, provenance = stage_native(helper, repo, fitting_inputs, generation_inputs,
                                                        refits, fitting_config, seed, work)
        record["provenance"] = provenance
        common = ["--acquisition", "inputs/train", "--fields", "inputs/train_fields",
                  "--training", "inputs/parent_training", "--device", device]
        helper.write_json(work / "run.json", record)
        code = 0
        for mode in MODES:
            directory = work / "results" / mode
            directory.mkdir(parents=True)
            result = dict(status="RUNNING", exit_code=None, scientific_claim_qualified=False, **execution)
            record["modes"][mode] = result
            commands = (
                ("evaluate", [sys.executable, "-m", "population_dynamics.population_trajectory_evaluation",
                    *common, "--config", "Libs/configs/" + FORECAST, "--output", str(directory / "forecasts"),
                    "--model-type", "population", "--correction", str(work / "corrections" / mode)]),
                ("replay", [sys.executable, "-m", "population_dynamics.population_forecast_replay",
                    *common, "--forecasts", str(directory / "forecasts"), "--output", str(directory / "replay.json"),
                    "--all-origins", "--correction", str(work / "corrections" / mode)]),
            )
            try:
                for name, command in commands:
                    phase_name = mode + "_" + name
                    phase_code = helper.phase(work, phase_name, command, probe)
                    receipt = json.loads((work / "phases" / phase_name / "receipt.json").read_text())
                    result[name + "_exit_code"] = receipt.get("exit_code")
                    if phase_code:
                        result.update(status="RESOURCE_BLOCKED" if phase_code == 3 else "FAILED", exit_code=phase_code)
                        break
                    if receipt.get("status") != "PASS" or receipt.get("exit_code") != 0:
                        raise ValueError("successful child and phase receipt required")
                else:
                    result.update(verify_mode(helper, directory, mode, seed, forecast, identities[mode],
                                              provenance["parent_checkpoint_sha256"]))
            except Exception as error:
                if record.get("received_signal"):
                    raise
                result.update(status="FAILED", exit_code=2, error_type=type(error).__name__,
                              error=str(error), traceback=traceback.format_exc())
            helper.write_json(directory / "mode.json", result)
            # The outer Merlin publisher can persist a complete mode at this boundary.
            helper.write_json(work / "run.json", record)
            if result["status"] != "PASS":
                code = result["exit_code"]
                if code == 3:
                    break
        if any(row["status"] != "PASS" or row["exit_code"] != 0 for row in record["modes"].values()):
            code = code or 2
        record.update(status="PASS" if code == 0 else "RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code)
    except BaseException as error:
        code = 128 + record["received_signal"] if record.get("received_signal") else 3 if isinstance(error, helper.ResourceBlocked) else 2
        record.update(status="RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        record["execution_seconds_before_archive"] = time.perf_counter() - started
        helper.write_json(work / "run.json", record)
        archive_started = time.perf_counter()
        helper.archive(work, output, record)
        record["archive_seconds"] = time.perf_counter() - archive_started
        record["elapsed_seconds"] = time.perf_counter() - started
        record["elapsed_scope"] = "input admission, staging, evaluation, replay and archival; final delivery write excluded"
        helper.write_json(output / "delivery.json", record)
        (work / "archive_complete").write_text(str(output))
    return code


def run(repo, preparation, fit_job, seed, work, output, device):
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    if (not os.environ.get("SLURM_JOB_ID") or platform.node().split(".")[0] == "mgmt2"
            or (device == "cpu" and (node, partition) not in CPU_PLACEMENTS)
            or (device == "cuda" and (node not in GPU_NODES or not os.environ.get("SLURM_JOB_GPUS")))):
        raise ValueError("an authorized CPU or GPU Slurm allocation is required")
    if seed not in SEEDS or not fit_job.isdecimal() or int(fit_job) < 1:
        raise ValueError("registered seed and explicit positive fit job required")
    if os.environ.get("SLURM_ARRAY_TASK_ID") not in (None, str(seed)):
        raise ValueError("Slurm array task must equal the registered seed")
    repo, preparation, work, output = (Path(value).resolve() for value in (repo, preparation, work, output))
    if not work.is_relative_to(Path("/tmp/neuronsim")) or work == Path("/tmp/neuronsim"):
        raise ValueError("task-specific node-local /tmp/neuronsim scratch required")
    if output.exists() or output.is_relative_to(work):
        raise FileExistsError("a fresh output outside scratch is required")
    helper = campaign_helper(repo)
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch").mkdir(exist_ok=True)
    (work / "prerequisites").mkdir(exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    record = dict(status="RUNNING", exit_code=None, job_id=os.environ["SLURM_JOB_ID"],
                  array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"), training_seed=seed,
                  node=node, partition=partition, device=device, fit_job_id=fit_job,
                  parent_training=False, core_neurons=100000, complete_episodes=72,
                  fit_episodes=56, validation_episodes=16, scientific_claim_qualified=False,
                  formal_evaluation=False, calibration_or_test_accessed=False,
                  matched_hardware_speedup_qualified=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  modes={mode: dict(status="NOT_RUN", exit_code=None) for mode in MODES})

    def interrupted(signum, frame):
        record["received_signal"] = signum
        raise InterruptedError(f"received signal {signum}")

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 2
    try:
        forecast, identities, provenance = stage(helper, repo, preparation, fit_job, seed, work)
        record["provenance"] = provenance
        probe = helper.staging_probe(work)
        health = probe(work / "phases/staging_preflight.json", required_memory_gib=32)
        if health["status"] != "PASS":
            raise helper.ResourceBlocked(health["reasons"])
        helper.stage_inputs(repo, work)
        common = ["--acquisition", "inputs/train", "--fields", "inputs/train_fields",
                  "--training", "inputs/parent_training", "--device", device]
        code = 0
        for mode in MODES:
            directory = work / "results" / mode
            directory.mkdir(parents=True)
            result = dict(status="RUNNING", exit_code=None, scientific_claim_qualified=False)
            record["modes"][mode] = result
            commands = (
                ("evaluate", [sys.executable, "-m", "population_dynamics.population_trajectory_evaluation",
                    *common, "--config", "Libs/configs/" + FORECAST,
                    "--output", str(directory / "forecasts"), "--model-type", "population",
                    "--correction", str(work / "corrections" / mode)]),
                ("replay", [sys.executable, "-m", "population_dynamics.population_forecast_replay",
                    *common, "--forecasts", str(directory / "forecasts"), "--output", str(directory / "replay.json"),
                    "--all-origins", "--correction", str(work / "corrections" / mode)]),
            )
            try:
                for name, command in commands:
                    phase_name = mode + "_" + name
                    phase_code = helper.phase(work, phase_name, command, probe)
                    phase_receipt = json.loads((work / "phases" / phase_name / "receipt.json").read_text())
                    result[name + "_exit_code"] = phase_receipt.get("exit_code")
                    if phase_code:
                        result.update(status="RESOURCE_BLOCKED" if phase_code == 3 else "FAILED", exit_code=phase_code)
                        break
                else:
                    result.update(verify_mode(helper, directory, mode, seed, forecast, identities[mode],
                                              provenance["parent_checkpoint_sha256"]))
            except Exception as error:
                if record.get("received_signal"):
                    raise
                result.update(status="FAILED", exit_code=2, error_type=type(error).__name__,
                              error=str(error), traceback=traceback.format_exc())
            helper.write_json(directory / "mode.json", result)
            if result["status"] != "PASS":
                code = result["exit_code"]
                if code == 3:
                    break
        record.update(status="PASS" if code == 0 else "RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code)
    except BaseException as error:
        code = 128 + record["received_signal"] if record.get("received_signal") else 3 if isinstance(error, helper.ResourceBlocked) else 2
        record.update(status="RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        record["execution_seconds_before_archive"] = time.perf_counter() - started
        helper.write_json(work / "run.json", record)
        archive_started = time.perf_counter()
        # Existing archive excludes inputs/ (cohort and parent training) and scratch/.
        helper.archive(work, output, record)
        record["archive_seconds"] = time.perf_counter() - archive_started
        record["elapsed_seconds"] = time.perf_counter() - started
        record["elapsed_scope"] = "staging, evaluation, replay and archival; final delivery write excluded"
        helper.write_json(output / "delivery.json", record)
        (work / "archive_complete").write_text(str(output))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preparation", type=Path, help="Legacy ridge fitting preparation")
    parser.add_argument("--fit-job", help="Legacy positive Slurm fitting job ID")
    parser.add_argument("--native-refits", type=Path, help="Merlin route: six complete seed_N/current|history refit directories")
    parser.add_argument("--fitting-inputs", type=Path, help="Admitted fitting bundle containing its original manifest.json")
    parser.add_argument("--generation-inputs", type=Path, help="Admitted additional source bundle with its own manifest.json")
    parser.add_argument("--fitting-config", type=Path, help="Byte-identical YAML bound by all native refits")
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.native_refits is not None:
        if args.preparation is not None or args.fit_job is not None or any(value is None for value in (
                args.fitting_inputs, args.generation_inputs, args.fitting_config)):
            parser.error("--native-refits requires fitting inputs, generation inputs and fitting config; omit legacy arguments")
        raise SystemExit(run_native(args.repo, args.fitting_inputs, args.generation_inputs, args.native_refits,
                                    args.fitting_config, args.seed, args.work, args.output, args.device))
    if args.preparation is None or args.fit_job is None or any(value is not None for value in (
            args.fitting_inputs, args.generation_inputs, args.fitting_config)):
        parser.error("legacy execution requires --preparation and --fit-job; native inputs require --native-refits")
    raise SystemExit(run(args.repo, args.preparation, args.fit_job, args.seed, args.work, args.output, args.device))


if __name__ == "__main__":
    main()
